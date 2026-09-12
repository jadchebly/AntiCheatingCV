"""End-to-end proctoring pipeline.

For each sampled frame:
  1. Pose-track all persons (YOLO-pose + ByteTrack).
  2. Detect devices in tiled mode (better small-object recall).
  3. Filter device detections through the spatial fixture filter — kills
     persistent FPs on desk grommets, outlets, etc.
  4. Associate each surviving device to a tracked person via wrist proximity
     (with body-centre fallback). Emit a ``cell_phone`` or ``laptop`` signal.
  5. Run the chatting heuristic over the pose tracks. Emit a ``chatting``
     signal for every person currently flagged.
  6. Feed all per-frame signals into the ``EventAggregator`` which maintains
     intervals across frames, bridging short gaps and dropping too-short bursts.

After the loop, write the events CSV, finalise the annotated MP4, and emit
``metrics.json`` summarising the run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from proctoring.alerts.csv_logger import write_events_csv
from proctoring.behaviour.aggregation import Event, EventAggregator
from proctoring.behaviour.chatting import ChattingDetector
from proctoring.detection.detector import Detection, DeviceDetector
from proctoring.detection.fixture_filter import SpatialFixtureFilter
from proctoring.evaluation.postprocess import filter_run
from proctoring.tracking.pose_tracker import PoseTracker, TrackedPerson
from proctoring.utils import visualization as viz
from proctoring.utils.io import VideoReader, VideoWriter, select_device


@dataclass
class RunSummary:
    video: str
    duration_sec: float
    n_frames_processed: int
    effective_fps: float
    events: list[dict]
    fixture_clusters: list[tuple[str, float, float, int]]
    # Events surviving the confidence/ghost pass, plus what it removed. Empty
    # and zeroed when postprocessing is disabled.
    filtered_events: list[dict] = field(default_factory=list)
    postprocess_stats: dict[str, Any] = field(default_factory=dict)


def _associate_device_to_person(
    device: Detection,
    persons: list[TrackedPerson],
    wrist_max: float,
    wrist_conf_min: float,
    body_max: float,
) -> tuple[int | None, str]:
    """Return ``(track_id, association_kind)`` for the device.

    Tries wrist proximity first (more semantic), then falls back to body-centre
    distance for the case of a phone clearly on a desk with the student's hand
    momentarily out of frame.
    """
    if not persons:
        return None, "none"
    dcx = 0.5 * (device.xyxy[0] + device.xyxy[2])
    dcy = 0.5 * (device.xyxy[1] + device.xyxy[3])

    best_id, best_d = None, math.inf
    for p in persons:
        for wx, wy in p.wrist_points(conf_min=wrist_conf_min):
            d = math.hypot(wx - dcx, wy - dcy)
            if d < best_d:
                best_d = d
                best_id = p.track_id
    if best_id is not None and best_d <= wrist_max:
        return best_id, "wrist"

    best_id, best_d = None, math.inf
    for p in persons:
        d = math.hypot(p.cx - dcx, p.cy - dcy)
        if d < best_d:
            best_d = d
            best_id = p.track_id
    if best_id is not None and best_d <= body_max:
        return best_id, "body"
    return None, "none"


def run_video(
    video_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    progress_cb: Callable[[int, int], None] | None = None,
) -> RunSummary:
    """Process ``video_path`` and write outputs into ``output_dir``.

    ``progress_cb`` is an optional ``(frames_done, frames_total)`` callback
    invoked once per sampled frame. The web UI uses it to drive a progress bar;
    the CLI leaves it unset and relies on the tqdm bar.
    """
    video_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(config["device"]["preference"])
    print(f"[proctoring] device={device}  video={video_path.name}")

    reader = VideoReader(video_path, frame_stride=config["processing"]["frame_stride"])
    effective_fps = reader.meta.fps / max(1, reader.frame_stride)

    detector = DeviceDetector(
        weights=config["models"]["detector_weights"],
        device=device,
        conf=config["detection"]["conf_threshold"],
        iou=config["detection"]["iou_threshold"],
        imgsz=config["models"]["imgsz"],
        device_classes=config["device_classes"],
        device_min_area_px=config["detection"]["device_min_area_px"],
        tiles_x=config["detection"].get("tiles_x", 1),
        tiles_y=config["detection"].get("tiles_y", 1),
        tile_overlap_frac=config["detection"].get("tile_overlap_frac", 0.0),
    )

    tracker = PoseTracker(
        weights=config["models"]["pose_weights"],
        device=device,
        imgsz=config["models"]["imgsz"],
        tracker_cfg=config["tracking"]["tracker"],
    )
    tracker.reset()

    fixture_cfg = config.get("fixture_filter", {})
    fixture: SpatialFixtureFilter | None = None
    if fixture_cfg.get("enabled", True):
        # The radius default was tuned at 1080p. Left absolute, a 4K frame gets a
        # cluster four times too tight, so a static false positive whose centre
        # jitters by a few dozen pixels splits across clusters and never locks.
        radius = float(fixture_cfg.get("cluster_radius_px", 18.0))
        radius *= max(1.0, reader.meta.width / 1920.0)
        # A fixture has to persist to count, but a fixed 6 s floor exceeds a
        # short clip outright and makes the filter inert on it. Clamp against
        # the footage we actually have.
        lifetime = float(fixture_cfg.get("min_lifetime_sec", 6.0))
        if reader.meta.duration_sec > 0:
            lifetime = min(lifetime, 0.4 * reader.meta.duration_sec)
        fixture = SpatialFixtureFilter(
            cluster_radius_px=radius,
            min_count_to_lock=fixture_cfg.get("min_count_to_lock", 6),
            min_lifetime_sec=lifetime,
        )
        print(f"[proctoring] fixture filter: radius={radius:.0f}px "
              f"lifetime={lifetime:.1f}s")

    chat_cfg = config.get("chatting", {})
    chat: ChattingDetector | None = None
    if chat_cfg.get("enabled", True):
        chat = ChattingDetector(
            proximity_max_shoulder_widths=chat_cfg.get("proximity_max_shoulder_widths", 2.5),
            min_head_turn_ratio=chat_cfg.get("min_head_turn_ratio", 0.15),
            min_pose_conf=chat_cfg.get("min_pose_conf", 0.3),
            min_consecutive_frames=chat_cfg.get("min_consecutive_frames", 2),
        )

    aggregator = EventAggregator(
        gap_tolerance_sec=config["aggregation"]["gap_tolerance_sec"],
        min_duration_sec=config["aggregation"]["min_duration_sec"],
    )

    assoc = config["association"]
    wrist_max = float(assoc["wrist_max_dist_px"])
    wrist_conf_min = float(assoc["wrist_conf_min"])
    body_max = float(assoc["body_max_dist_px"])

    events_csv = out_dir / config["output"]["events_filename"]
    annotated_video = out_dir / config["output"]["video_filename"]
    metrics_path = out_dir / config["output"]["metrics_filename"]
    draw_overlay = config["output"].get("draw_overlay", True)
    writer = VideoWriter(annotated_video, fps=effective_fps) if draw_overlay else None

    n_processed = 0
    pbar = tqdm(total=reader.meta.n_frames, unit="frame", desc=video_path.name)
    try:
        for frame_idx, ts, frame in reader:
            n_processed += 1

            persons = tracker.track(frame)
            devices = detector.detect(frame)

            # Suppress static fixtures.
            kept_devices: list[Detection] = []
            for d in devices:
                if fixture is not None and fixture.is_fixture(d.class_name, d.cx, d.cy, ts):
                    continue
                kept_devices.append(d)

            # Build per-frame signals: (subtype, track_id, conf, bbox)
            signals: list[tuple[str, int | None, float, tuple[float, float, float, float] | None]] = []

            for d in kept_devices:
                tid, _kind = _associate_device_to_person(
                    d, persons, wrist_max=wrist_max,
                    wrist_conf_min=wrist_conf_min, body_max=body_max,
                )
                signals.append((d.class_name, tid, d.confidence, d.xyxy))

            chat_pairs: dict[int, set[int]] = {}
            if chat is not None:
                chat_pairs = chat.detect(persons)
                for tid in chat_pairs:
                    signals.append(("chatting", tid, 1.0, None))

            aggregator.update(ts, signals)

            if writer is not None:
                viz.draw_persons(frame, [{
                    "xyxy": p.xyxy, "track_id": p.track_id,
                } for p in persons])
                viz.draw_devices(frame, [{
                    "xyxy": d.xyxy, "class_name": d.class_name, "confidence": d.confidence,
                } for d in kept_devices])
                # Chat link lines
                drawn = set()
                for tid, partners in chat_pairs.items():
                    for other in partners:
                        key = tuple(sorted((tid, other)))
                        if key in drawn:
                            continue
                        drawn.add(key)
                        a = next((p for p in persons if p.track_id == tid), None)
                        b = next((p for p in persons if p.track_id == other), None)
                        if a is not None and b is not None:
                            viz.draw_chat_link(frame, (a.cx, a.cy), (b.cx, b.cy))
                # Active-event banner
                actives = aggregator.active_subtypes_at(ts)
                if actives:
                    viz.draw_active_events(frame, [f"ACTIVE: {a}" for a in sorted(actives)])
                viz.draw_timestamp(frame, ts)
                writer.write(frame)

            pbar.update(reader.frame_stride)
            if progress_cb is not None:
                progress_cb(min(frame_idx + 1, reader.meta.n_frames), reader.meta.n_frames)
    finally:
        pbar.close()
        reader.close()
        if writer is not None:
            writer.close()

    events = aggregator.flush()
    write_events_csv(events_csv, events)

    # Confidence floor + ghost-track suppression. This used to exist only as a
    # separate CLI step nothing invoked, so every consumer saw the raw list with
    # its low-confidence false positives still in it.
    pp_cfg = config.get("postprocess", {})
    filtered_rows: list[dict] = []
    pp_stats: dict[str, Any] = {}
    if pp_cfg.get("enabled", True):
        pp_stats = filter_run(
            out_dir,
            min_mean_conf=pp_cfg.get("min_mean_conf"),
            ghost_max_span_px=pp_cfg.get("ghost_max_span_px", 60.0),
            ghost_max_events=pp_cfg.get("ghost_max_events", 4),
            merge_bridge_sec=pp_cfg.get("merge_bridge_sec", 1.0),
        )
        filtered_rows = _read_filtered(out_dir / "events_filtered.csv")
        print(f"[proctoring] postprocess: {pp_stats['n_input']} -> "
              f"{pp_stats['n_output']} events "
              f"(dropped {pp_stats['n_dropped_low_confidence']} low-confidence, "
              f"{pp_stats['n_dropped_ghost_tracks']} ghost)")

    summary = RunSummary(
        video=str(video_path),
        duration_sec=reader.meta.duration_sec,
        n_frames_processed=n_processed,
        effective_fps=effective_fps,
        events=[
            {
                "subtype": e.subtype,
                "start_sec": round(e.start_sec, 2),
                "end_sec": round(e.end_sec, 2),
                "duration_sec": round(e.duration_sec, 2),
                "track_id": e.track_id,
                "confidence": round(e.confidence, 3),
            }
            for e in sorted(events, key=lambda e: e.start_sec)
        ],
        fixture_clusters=fixture.locked_clusters() if fixture is not None else [],
        filtered_events=filtered_rows,
        postprocess_stats=pp_stats,
    )
    with open(metrics_path, "w") as f:
        json.dump({
            "video": summary.video,
            "duration_sec": summary.duration_sec,
            "n_frames_processed": summary.n_frames_processed,
            "effective_fps": summary.effective_fps,
            "n_events": len(events),
            "n_events_filtered": len(filtered_rows),
            "postprocess": pp_stats,
            "events_by_subtype": _count_by_subtype(events),
            "events": summary.events,
            "filtered_events": summary.filtered_events,
            "fixture_clusters": [
                {"class_name": c, "cx": cx, "cy": cy, "count": n}
                for (c, cx, cy, n) in summary.fixture_clusters
            ],
        }, f, indent=2)
    return summary


def _read_filtered(path: Path) -> list[dict]:
    """Read events_filtered.csv back into the same shape as RunSummary.events."""
    if not path.exists():
        return []
    import csv
    rows: list[dict] = []
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            tid = r.get("track_id") or ""
            rows.append({
                "subtype": r["subtype"],
                "start_sec": float(r["start_sec"]),
                "end_sec": float(r["end_sec"]),
                "duration_sec": float(r["duration_sec"]),
                "track_id": int(tid) if tid else None,
                "confidence": float(r["mean_confidence"] or 0.0),
            })
    return rows


def _count_by_subtype(events: list[Event]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        out[e.subtype] = out.get(e.subtype, 0) + 1
    return out
