# Intelligent Exam Proctoring System

Computer-vision pipeline that monitors classroom exam footage for unauthorized
electronic devices and suspicious peer-to-peer behaviour. Built for the
Computer Vision group project (Option 1).

The system reads a video file, detects three subtypes of policy violations —
`cell_phone`, `laptop`, and `chatting` — and produces a time-stamped CSV log
of events, an annotated MP4 visualization, and a quantitative evaluation
against ground-truth interval annotations.

## Pipeline at a glance

```
                     video file
                          │
                          ▼
              ┌───────────────────────┐
              │ Frame loop (stride N) │
              └──────┬────────────────┘
                     │
       ┌─────────────┴──────────────┐
       ▼                            ▼
  YOLO11l + tiled inference     YOLO11m-pose + ByteTrack
  (cell_phone, laptop)          (persistent person identities)
       │                            │
       ▼                            │
  SpatialFixtureFilter              │
  (kills static FPs on              │
   desk grommets / outlets)         │
       │                            ▼
       ▼               ┌────────────────────────┐
  Wrist-aware          │ Chatting heuristic     │
  device→person        │ (proximity + mutual    │
  association          │  facing in shoulder    │
       │               │  widths)               │
       │               └──────┬─────────────────┘
       ▼                      ▼
       └───────────┬──────────┘
                   ▼
        EventAggregator
        (per-subtype min_duration
         + gap_tolerance)
                   │
                   ▼
        events.csv  +  annotated.mp4  +  metrics.json
```

## Project layout

```
.
├── source/
│   ├── proctoring/
│   │   ├── detection/        # YOLO + tiled inference + fixture filter
│   │   ├── tracking/         # YOLO-pose + ByteTrack wrapper
│   │   ├── behaviour/        # chatting heuristic + event aggregation
│   │   ├── alerts/           # CSV writer
│   │   ├── evaluation/       # ground-truth loader + temporal-IoU metrics
│   │   ├── utils/            # video I/O, visualization
│   │   ├── pipeline.py       # end-to-end orchestrator
│   │   └── cli.py            # argparse entry point
│   ├── webui/                # Flask app + single-page front-end
│   ├── scripts/              # convenience wrappers + UI launcher
│   ├── configs/default.yaml  # all thresholds, model paths, etc.
│   ├── requirements.txt
│   └── environment.yml
├── data/raw/                 # video files + ground-truth JSONs
├── outputs/                  # per-video run output
├── models/                   # YOLO weight files (auto-downloaded on first run)
├── docs/
│   ├── assignment-instructions.txt
│   └── template-report.pdf
└── report/                   # technical report + demo script
```

## Installation

Python ≥ 3.10. Tested on macOS Apple Silicon (MPS) and Linux. `ffmpeg` is
needed only if you want to re-encode source clips outside the pipeline.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r source/requirements.txt
```

Or with conda:

```bash
conda env create -f source/environment.yml
conda activate proctoring
```

The first pipeline run downloads the two YOLO weight files (~90 MB total)
from the official Ultralytics release into `models/`:

- `models/yolo11l.pt` — joint person + device detector
- `models/yolo11m-pose.pt` — body-keypoint model for behaviour analysis

## Running the pipeline

```bash
PYTHONPATH=source python3 -m proctoring.cli run \
    --video data/raw/video1.MOV \
    --out   outputs/video1
```

Or via the wrapper:

```bash
python3 source/scripts/run_proctoring.py \
    --video data/raw/video1.MOV \
    --out   outputs/video1
```

Output files in `outputs/video1/`:

- `events.csv` — assignment-mandated alert log (one row per merged event interval)
- `annotated.mp4` — annotated demonstration video
- `metrics.json` — run summary + raw event list

All thresholds live in [`source/configs/default.yaml`](source/configs/default.yaml)
— model paths, tile grid, confidence thresholds, fixture-filter parameters,
chatting heuristic, per-subtype aggregation timing, etc.

## Web UI

A small Flask front-end wraps the same pipeline for demos: pick a clip, watch
the progress bar, then read the events table and scrub the annotated video.

```bash
python3 source/scripts/run_ui.py
```

It serves on <http://127.0.0.1:8000> and opens a browser window. Override with
`--port 8080`, and pass `--no-browser` to skip the auto-open. If the port is
busy the launcher steps up to the next free one and prints where it landed.

> Port 5000 is avoided deliberately. On macOS, Control Center holds that port
> for AirPlay Receiver, and a browser reaching it instead of Flask shows a
> blank page.

The page lets you:

- pick any clip in `data/raw/`, or upload one from the browser
- override frame stride, compute device, and the chatting / fixture-filter toggles
- follow live progress and the pipeline log while the run executes
- browse detected events and click a row to jump the annotated video to that moment
- download `events.csv`, `metrics.json`, and `annotated.mp4`
- score the run against `data/raw/<video>.json` when ground truth exists

Runs are serialised — one video at a time, since the pipeline takes the whole
GPU. Job state is in memory, so restarting the server clears the history while
the files under `outputs/` remain.

## Evaluation

Ground-truth annotations are stored as JSON files alongside each video
(`data/raw/video1.json`, etc.) using the schema:

```json
{
  "video": "video1.MOV",
  "duration_s": 540.0,
  "events": [
    {"start_s": 33.0, "end_s": 43.0, "subtype": "cell_phone"},
    {"start_s": 178.0, "end_s": 191.0, "subtype": "chatting"}
  ]
}
```

Evaluate one run:

```bash
PYTHONPATH=source python3 -m proctoring.cli evaluate \
    --run-dir outputs/video1 \
    --gt      data/raw/video1.json \
    --iou     0.1,0.3,0.5
```

Evaluate every run-dir in `outputs/` against its corresponding GT JSON in
`data/raw/`, and aggregate:

```bash
PYTHONPATH=source python3 -m proctoring.cli evaluate-many \
    --runs-glob 'outputs/video*' \
    --gt-dir    data/raw \
    --iou       0.1,0.3,0.5
```

A predicted event matches a ground-truth event when their **temporal IoU**
(intersection-over-union of time intervals) exceeds the threshold. We report
at three thresholds (`0.1`, `0.3`, `0.5`) per subtype, plus aggregate.
Precision, recall and F1 are reported per subtype and overall.

## What we did differently from the baseline approach

A previous group built a similar system; the report is at
[`docs/template-report.pdf`](docs/template-report.pdf). Concrete deltas:

| Aspect | Baseline | This system |
|---|---|---|
| Detector backbone | YOLOv8 | **YOLO11l** (larger, newer) |
| Tiled inference | 2×2 | 2×2 with 20% overlap and per-class NMS |
| Static-fixture suppression | — | **`SpatialFixtureFilter`** (kills sustained FPs on desk grommets / outlets) |
| Person-device association | (unspecified) | **Wrist-aware** primary path with body-centre fallback |
| Chatting heuristic | proximity + facing | proximity (in shoulder-widths) + mutual head-yaw, measured by projecting the nose onto the shoulder axis so the vertical nose offset cannot swamp the turn signal |
| Evaluation | temporal IoU @ 0.3 | temporal IoU at **{0.1, 0.3, 0.5}** for sensitivity reporting |

Quantitative comparison numbers are in [`report/report.md`](report/report.md).
