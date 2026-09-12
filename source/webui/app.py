"""Flask backend for the proctoring web UI.

One page, three steps: pick a video, run the pipeline, read the results.

The heavy work happens in a background thread so the browser can poll for
progress. Jobs live in memory only — this is a local demo tool, not a service,
and restarting the server forgets them. Finished artefacts (``events.csv``,
``annotated.mp4``, ``metrics.json``) persist on disk under ``outputs/``.

    python3 source/scripts/run_ui.py
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import threading
import traceback
import uuid
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

DEFAULT_CONFIG = SOURCE_DIR / "configs" / "default.yaml"
VIDEO_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = REPO_ROOT / "outputs"

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# --------------------------------------------------------------------------- #
# Job registry
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    id: str
    video: str
    out_dir: str
    status: str = "queued"          # queued | running | done | error
    frames_done: int = 0
    frames_total: int = 0
    started_at: str = ""
    finished_at: str = ""
    error: str = ""
    summary: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        pct = 0.0
        if self.frames_total > 0:
            pct = min(100.0, 100.0 * self.frames_done / self.frames_total)
        elif self.status == "done":
            pct = 100.0
        return {
            "id": self.id,
            "video": self.video,
            "out_dir": self.out_dir,
            "status": self.status,
            "percent": round(pct, 1),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "summary": self.summary,
            "evaluation": self.evaluation,
            "log": self.log[-40:],
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
# The pipeline grabs the GPU/MPS device and a lot of RAM, so only ever run one.
RUN_LOCK = threading.Lock()


class _LogStream(io.TextIOBase):
    """Collect pipeline stdout/stderr into the job log, one line per write."""

    def __init__(self, job: Job):
        self.job = job
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        # tqdm redraws with \r; treat it as a line break so the bar updates.
        while True:
            idx = min(
                (i for i in (self._buf.find("\n"), self._buf.find("\r")) if i >= 0),
                default=-1,
            )
            if idx < 0:
                break
            line, self._buf = self._buf[:idx].strip(), self._buf[idx + 1:]
            if line:
                with JOBS_LOCK:
                    self.job.log.append(line)
                    if len(self.job.log) > 400:
                        del self.job.log[:200]
        return len(text)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _run_job(job: Job, video_path: Path, out_dir: Path, overrides: dict[str, Any]) -> None:
    """Background worker: load config, run the pipeline, record the summary."""
    # Imported here, not at module scope: pulling in torch/ultralytics takes
    # several seconds and would stall server startup.
    from proctoring.pipeline import run_video
    from proctoring.utils.io import load_config

    stream = _LogStream(job)
    with RUN_LOCK:
        try:
            with JOBS_LOCK:
                job.status = "running"
                job.started_at = _now()
                job.log.append(f"[{_now()}] loading config")

            config = load_config(DEFAULT_CONFIG)
            config["processing"]["frame_stride"] = int(overrides["frame_stride"])
            config["output"]["draw_overlay"] = bool(overrides["draw_overlay"])
            config["chatting"]["enabled"] = bool(overrides["chatting"])
            config["fixture_filter"]["enabled"] = bool(overrides["fixture_filter"])
            if overrides.get("device"):
                config["device"]["preference"] = overrides["device"]

            def progress(done: int, total: int) -> None:
                with JOBS_LOCK:
                    job.frames_done = done
                    job.frames_total = total

            with redirect_stdout(stream), redirect_stderr(stream):
                summary = run_video(video_path, out_dir, config, progress_cb=progress)

            with JOBS_LOCK:
                job.summary = {
                    "video": Path(summary.video).name,
                    "duration_sec": round(summary.duration_sec, 1),
                    "n_frames_processed": summary.n_frames_processed,
                    "effective_fps": round(summary.effective_fps, 2),
                    "events": summary.events,
                    "counts": _count_subtypes(summary.events),
                    "n_fixture_clusters": len(summary.fixture_clusters),
                    "files": _artifact_links(out_dir),
                }
                job.status = "done"
                job.finished_at = _now()
                job.frames_done = job.frames_total or job.frames_done
                job.log.append(f"[{_now()}] done — {len(summary.events)} events")
        except Exception as exc:  # surfaced in the UI, not swallowed
            with JOBS_LOCK:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = _now()
                job.log.append(f"[{_now()}] FAILED: {job.error}")
            traceback.print_exc()


def _count_subtypes(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e["subtype"]] = counts.get(e["subtype"], 0) + 1
    return counts


def _artifact_links(out_dir: Path) -> dict[str, str | None]:
    """Map artefact name to a URL under /outputs, or None when absent."""
    rel = out_dir.relative_to(OUTPUT_DIR).as_posix()
    links: dict[str, str | None] = {}
    for key, name in (
        ("events_csv", "events.csv"),
        ("annotated_video", "annotated.mp4"),
        ("metrics_json", "metrics.json"),
    ):
        links[key] = f"/outputs/{rel}/{name}" if (out_dir / name).exists() else None
    return links


def _ground_truth_for(video_path: Path) -> Path | None:
    candidate = video_path.with_suffix(".json")
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent / "templates", "index.html")


@app.get("/api/videos")
def list_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for p in sorted(VIDEO_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            videos.append({
                "name": p.name,
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "has_ground_truth": _ground_truth_for(p) is not None,
            })
    return jsonify({"videos": videos, "dir": str(VIDEO_DIR)})


@app.post("/api/upload")
def upload_video():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "no file in request"}), 400
    name = secure_filename(file.filename)
    if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
        return jsonify({"error": f"unsupported file type: {Path(name).suffix}"}), 400
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dest = VIDEO_DIR / name
    file.save(dest)
    return jsonify({"name": dest.name})


@app.post("/api/run")
def start_run():
    payload = request.get_json(silent=True) or {}
    name = payload.get("video", "")
    # Reject anything that tries to escape data/raw.
    video_path = (VIDEO_DIR / Path(name).name).resolve()
    if not video_path.is_file() or VIDEO_DIR.resolve() not in video_path.parents:
        return jsonify({"error": f"video not found: {name}"}), 404

    with JOBS_LOCK:
        busy = [j for j in JOBS.values() if j.status in ("queued", "running")]
    if busy:
        return jsonify({"error": "a run is already in progress"}), 409

    out_dir = OUTPUT_DIR / video_path.stem
    overrides = {
        "frame_stride": max(1, int(payload.get("frame_stride", 4))),
        "draw_overlay": bool(payload.get("draw_overlay", True)),
        "chatting": bool(payload.get("chatting", True)),
        "fixture_filter": bool(payload.get("fixture_filter", True)),
        "device": payload.get("device") or None,
    }

    job = Job(id=uuid.uuid4().hex[:12], video=video_path.name, out_dir=str(out_dir))
    with JOBS_LOCK:
        JOBS[job.id] = job

    threading.Thread(
        target=_run_job, args=(job, video_path, out_dir, overrides), daemon=True
    ).start()
    return jsonify(job.as_dict())


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job.as_dict())


@app.post("/api/jobs/<job_id>/evaluate")
def evaluate_job(job_id: str):
    from proctoring.evaluation.metrics import evaluate

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None or job.status != "done":
        return jsonify({"error": "job not finished"}), 400

    video_path = VIDEO_DIR / job.video
    gt_path = _ground_truth_for(video_path)
    if gt_path is None:
        return jsonify({"error": f"no ground truth at {video_path.with_suffix('.json').name}"}), 404

    out_dir = Path(job.out_dir)
    try:
        metrics = evaluate(
            events_csv=out_dir / "events.csv",
            ground_truth_path=gt_path,
            iou_thresholds=[0.1, 0.3, 0.5],
        )
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with JOBS_LOCK:
        job.evaluation = metrics
    return jsonify(metrics)


@app.get("/outputs/<path:rel>")
def serve_output(rel: str):
    return send_from_directory(OUTPUT_DIR, rel, conditional=True)


def _port_is_taken(host: str, port: int) -> bool:
    """True if something already answers on ``port``.

    Deliberately a connect() test rather than a bind() test. On macOS, Control
    Center (AirPlay Receiver) listens on ``*:5000``, yet binding the narrower
    ``127.0.0.1:5000`` still succeeds — so a bind test looks clear while the
    browser, resolving localhost to ::1, reaches AirPlay and renders a blank
    page. Connecting catches that; binding does not. It is also why the default
    port below is 8000 and not Flask's usual 5000.
    """
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.25)
                if s.connect_ex(addr) == 0:
                    return True
        except OSError:
            continue
    return False


def _pick_port(host: str, preferred: int) -> int | None:
    """Return ``preferred`` if free, else the next free port above it."""
    for port in range(preferred, preferred + 20):
        if not _port_is_taken(host, port):
            return port
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the proctoring web UI.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to serve on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window on startup")
    args = parser.parse_args(argv)

    port = _pick_port(args.host, args.port)
    if port is None:
        print(f"[webui] ports {args.port}-{args.port + 19} are all in use; "
              f"pass --port with a free one", file=sys.stderr)
        return 1
    if port != args.port:
        print(f"[webui] port {args.port} is in use, using {port} instead")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{port}"
    print(f"[webui] videos   {VIDEO_DIR}")
    print(f"[webui] outputs  {OUTPUT_DIR}")
    print(f"[webui] open {url}")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # threaded=True so status polling stays responsive during a run.
    app.run(host=args.host, port=port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
