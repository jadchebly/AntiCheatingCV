"""Post-process the raw events CSV to suppress recurring low-confidence ghosts.

The frame-level pipeline emits events whenever per-frame signals exceed the
aggregator's gap/duration thresholds. In practice some YOLO tracks become
"ghosts" — a static or slowly-drifting region that the detector keeps tagging
at low confidence. The spatial fixture filter catches these only when they
sit at the same pixel; tracks that drift slowly evade it.

This pass runs after the main pipeline and applies two simple heuristics:

1. **Mean-confidence floor** per subtype — drop events whose averaged per-frame
   confidence is below a class-specific threshold.
2. **Per-track ghost suppression** — for each ``(subtype, track_id)`` pair,
   if the bounding-box centres of all the events fit inside a tight spatial
   cluster (max pairwise distance < ``ghost_max_span_px``) AND the track has
   produced more events than ``ghost_max_events``, classify the entire track
   as a ghost and drop every event from it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

POSTPROCESS_HEADER = [
    "event_id", "subtype", "start_sec", "end_sec", "duration_sec",
    "track_id", "mean_confidence", "x1", "y1", "x2", "y2", "notes",
]


def _read(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _bbox_centre(row: dict) -> tuple[float, float] | None:
    if not (row.get("x1") and row.get("y1") and row.get("x2") and row.get("y2")):
        return None
    try:
        return (
            0.5 * (float(row["x1"]) + float(row["x2"])),
            0.5 * (float(row["y1"]) + float(row["y2"])),
        )
    except ValueError:
        return None


def filter_events(
    rows: list[dict],
    min_mean_conf: dict[str, float] | None = None,
    ghost_max_span_px: float = 60.0,
    ghost_max_events: int = 4,
) -> tuple[list[dict], dict]:
    """Return ``(kept_rows, stats)``."""
    min_mean_conf = min_mean_conf or {
        "cell_phone": 0.30, "laptop": 0.30, "chatting": 0.0,
    }
    # Pass 1: confidence floor.
    after_conf: list[dict] = []
    n_dropped_conf = 0
    for r in rows:
        try:
            mc = float(r.get("mean_confidence") or 0.0)
        except ValueError:
            mc = 0.0
        floor = min_mean_conf.get(r["subtype"], 0.0)
        if mc < floor:
            n_dropped_conf += 1
            continue
        after_conf.append(r)

    # Pass 2: per-track ghost suppression. Group by (subtype, track_id).
    by_track: dict[tuple[str, str], list[dict]] = {}
    for r in after_conf:
        tid = r.get("track_id") or ""
        if tid == "":
            continue  # cannot classify ghost without an ID
        by_track.setdefault((r["subtype"], tid), []).append(r)

    ghost_keys: set[tuple[str, str]] = set()
    for key, group in by_track.items():
        if len(group) <= ghost_max_events:
            continue
        centres = [c for c in (_bbox_centre(r) for r in group) if c is not None]
        if len(centres) < 2:
            continue
        # Compute max pairwise distance.
        max_d = 0.0
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                d = math.hypot(centres[i][0] - centres[j][0],
                               centres[i][1] - centres[j][1])
                if d > max_d:
                    max_d = d
        if max_d < ghost_max_span_px:
            ghost_keys.add(key)

    after_ghost: list[dict] = []
    n_dropped_ghost = 0
    for r in after_conf:
        tid = r.get("track_id") or ""
        if (r["subtype"], tid) in ghost_keys:
            n_dropped_ghost += 1
            continue
        after_ghost.append(r)

    stats = {
        "n_input": len(rows),
        "n_output": len(after_ghost),
        "n_dropped_low_confidence": n_dropped_conf,
        "n_dropped_ghost_tracks": n_dropped_ghost,
        "ghost_track_keys": [{"subtype": k[0], "track_id": k[1]} for k in sorted(ghost_keys)],
    }
    return after_ghost, stats


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(POSTPROCESS_HEADER)
        for i, r in enumerate(rows):
            w.writerow([
                i,
                r["subtype"],
                r["start_sec"], r["end_sec"], r["duration_sec"],
                r.get("track_id", ""),
                r.get("mean_confidence", ""),
                r.get("x1", ""), r.get("y1", ""), r.get("x2", ""), r.get("y2", ""),
                r.get("notes", ""),
            ])


def filter_run(run_dir: Path,
               min_mean_conf: dict[str, float] | None = None,
               ghost_max_span_px: float = 60.0,
               ghost_max_events: int = 4) -> dict:
    src = run_dir / "events.csv"
    rows = _read(src)
    kept, stats = filter_events(
        rows,
        min_mean_conf=min_mean_conf,
        ghost_max_span_px=ghost_max_span_px,
        ghost_max_events=ghost_max_events,
    )
    out = run_dir / "events_filtered.csv"
    write(out, kept)
    stats["output_path"] = str(out)
    return stats
