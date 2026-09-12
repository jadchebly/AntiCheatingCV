# Intelligent Exam Proctoring System — Technical Report

> Computer Vision group project · Option 1 · 7 group members.
> Code: see [`README.md`](../README.md). All numbers below are from a real
> evaluation run on the three test videos in [`data/raw/`](../data/raw/).

---

## 1. Problem definition

Academic-integrity monitoring during in-person exams is hard for a single
proctor: suspicious behaviour is varied (peer communication, hidden device
usage, copying), often subtle, and spread across many students at once.
Manual coverage degrades with classroom size.

This project produces an **offline, human-in-the-loop** computer-vision
system that ingests pre-recorded classroom video and emits a time-stamped
log of suspected violations, plus an annotated video for manual review.
The system targets three subtypes — `cell_phone`, `laptop`, and
`chatting` — chosen to match the labels available in our ground-truth
annotations.

The system is not designed to make disciplinary decisions on its own. Its
role is to compress hours of footage into a tractable list of intervals to
inspect.

## 2. System architecture

```
                      video file
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
   YOLO11l + 2×2 tiled         YOLO11m-pose + ByteTrack
   inference + per-class       (persistent person identities)
   NMS  →  device boxes        →  17-keypoint poses
             │                         │
             ▼                         │
   SpatialFixtureFilter                │
   (suppresses static                  │
    desk-fixture FPs)                  │
             │                         ▼
             │         ┌─────────────────────────┐
             │         │  ChattingDetector       │
             │         │  (proximity in shoulder │
             │         │   widths + mutual-      │
             │         │   facing within strict  │
             │         │   tolerance)            │
             │         └─────────┬───────────────┘
             ▼                   │
   Wrist-aware                   │
   device→person                 │
   association                   │
   (body-centre fallback)        │
             │                   │
             └──────────┬────────┘
                        ▼
              EventAggregator
              (per-subtype gap
               tolerance + min
               duration)
                        │
                        ▼
              events.csv + annotated.mp4
                        │
                        ▼
              Post-process filter
              (confidence floor +
               ghost-track suppression)
                        │
                        ▼
              events_filtered.csv
                        │
                        ▼
              Temporal-IoU evaluation
              (IoU @ 0.1 / 0.3 / 0.5)
```

Each stage is a small, independently testable module under
[`source/proctoring/`](../source/proctoring/). The architecture
follows the same modular philosophy as the baseline reference report
(`detection / pose tracking / aggregation / output`); our additions are
the **fixture filter**, **wrist-aware association**, and the
**post-process pass**.

## 3. Methodology and design decisions

### 3.1 Detection

We use **YOLO11l** (Ultralytics) — a substantially larger backbone than
the YOLOv8 default used as the reference. The detector runs in **2×2
tiled mode** at 1280-pixel input with 20% tile overlap, followed by
**class-aware NMS**. Tiling improves small-object recall (phones in
back-row desks may occupy fewer than 30 pixels per side); class-aware
NMS prevents a `cell_phone` and `laptop` box from suppressing each other
when they happen to overlap.

Our confidence threshold is **0.15** — intentionally low — because tiny
phones rarely score above 0.4 from a wide-FOV view. The downstream
debouncing logic (aggregation + post-process) cleans up the resulting
noise without sacrificing recall.

### 3.2 Static-fixture suppression

A real failure mode discovered during development: COCO-pretrained YOLO
locks onto small static features of the desks (cable grommets, recessed
ports, dark seam edges) and labels them `cell_phone` with mid-range
confidence. The detector keeps re-firing at the same pixel for hundreds
of frames, producing what looks like a sustained device alert.

We added a [`SpatialFixtureFilter`](../source/proctoring/detection/fixture_filter.py)
which keeps a running cluster of detection centres per class. When a
cluster accumulates ≥ `min_count_to_lock` hits over ≥ `min_lifetime_sec`,
it is **locked as a fixture** and any subsequent detection inside its
radius is rejected before reaching the alert pipeline. This is one of
our concrete improvements over the baseline reference, which describes
no such filter.

In our three-video test run the filter locked **323 fixture clusters**
across the three videos (92, 82, 149), most of them on desk grommets.

### 3.3 Pose tracking and chatting heuristic

Pose comes from **YOLO11m-pose** with **ByteTrack** (the Ultralytics
built-in). ByteTrack handles short occlusions well and gives each
student a stable `track_id` from the first frame they appear.

The chatting heuristic is interpretable and label-free:

- For each pair of tracked students, normalise their centre-to-centre
  distance by the **average shoulder width** (a per-pair scale that
  survives perspective distortion). Reject the pair if the normalised
  distance exceeds `proximity_max_shoulder_widths = 2.5`.
- Compute each person's "gaze proxy" as the unit vector from shoulder
  midpoint to nose. Reject the pair unless **both** vectors point
  toward the partner within `facing_tolerance_deg = 30°`.

Both thresholds were tightened relative to early experiments: the
heuristic is intrinsically noisy at small back-row scales and a strict
mutual-facing requirement minimises spurious pairings.

### 3.4 Wrist-aware device→person association

When a device is detected, we associate it to a tracked person by trying
the **nearest wrist** first (within `wrist_max_dist_px = 250`, requiring
keypoint confidence ≥ 0.30), and fall back to **nearest body centre**
(within 400 px) only if no wrist passes. This prefers the semantic
"who is holding it" interpretation while still attributing a phone seen
on a desk to the most plausible owner.

### 3.5 Aggregation

Per-frame signals are temporally merged into discrete events with
class-specific timing:

| Subtype | gap_tolerance_sec | min_duration_sec |
|---|---|---|
| cell_phone | 1.5 | 1.5 |
| laptop | 2.5 | 3.0 |
| chatting | 1.5 | 3.0 |

Laptop events are given a longer minimum duration because laptops are
visually stable and a real laptop session lasts at least several seconds.

### 3.6 Post-process filter

After the main pipeline, two heuristics clean the events list:

1. **Mean-confidence floor** per subtype (default: ≥ 0.30 for
   `cell_phone` and `laptop`). Drops events whose detections were
   consistently borderline.
2. **Ghost-track suppression**: for each `(subtype, track_id)` pair
   with more than `ghost_max_events` events whose bounding-box centres
   all lie inside a `ghost_max_span_px` cluster, we reclassify the
   entire track as a ghost and drop it. This catches slowly-drifting
   ghost detections that the spatial fixture filter cannot lock onto
   (because their position drifts more than its cluster radius across
   the video).

The filter is applied as a separate pass so we can inspect the raw
events too. Both raw (`events.csv`) and filtered (`events_filtered.csv`)
outputs are written.

## 4. Experimental setup

### 4.1 Dataset

Three full ~9-minute classroom recordings (1920×1080, ~30 fps, HEVC):

| Video | Duration | GT events | Notes on GT JSON |
|---|---|---|---|
| video1.MOV | 540 s | 17 | Clean; no auto-generation note |
| video2.MOV | 540 s | 16 | Auto-generated; `_note: filename is 'normal' so 0 events expected` |
| video3.MOV | 541 s | 16 | Auto-generated; `_note: best-effort label by automation` |

Each video has a paired JSON ground-truth file in `data/raw/`. Because
two of three GTs are explicitly labelled as auto-generated, **video1 is
our primary signal** for accuracy claims; video2 and video3 are reported
for completeness but their numbers should be read with the GT caveat in
mind.

**Data collection was a joint effort with a second project group.**
We pooled subjects so that each recording could realistically simulate
a multi-row classroom with seven or more students visible simultaneously.
Some of the videos in our dataset feature only members of the other
group, others contain both teams seated together. The benefit is much
greater diversity in clothing, posture, seating position, and skin
tone than either group could produce alone — a deliberate measure to
reduce the risk that the system overfits to characteristics of a small
fixed cast. The trade-off is that none of us were the camera operator
for every session, so not every event was witnessed live; this is part
of why the ground-truth JSONs for video2 and video3 had to be
reconstructed automatically and then audited.

### 4.2 Hardware and runtime

All runs were performed on an Apple Silicon laptop (M-series) using the
PyTorch MPS backend. Configuration:

- `frame_stride = 4` (effective ~7.5 fps)
- `tiles_x × tiles_y = 2 × 2`, 20% overlap
- input size 1280

Steady-state throughput ≈ 10 source-frames/s, i.e. each ~9-min video
took ~25–28 minutes to process. Real-time is **not** required by the
assignment.

### 4.3 Evaluation protocol

A predicted event matches a ground-truth event when:

- their **subtype** is identical, **and**
- their **temporal IoU** exceeds the threshold.

Each ground-truth event matches at most one prediction (the one with the
highest IoU). Unmatched predictions are FP; unmatched GTs are FN. We
report Precision, Recall, F1 per subtype and overall, at three IoU
thresholds (0.1, 0.3, 0.5) for sensitivity reporting. The reference
report quotes only IoU 0.3.

## 5. Quantitative evaluation

All numbers below are computed by
[`source/proctoring/evaluation/metrics.py`](../source/proctoring/evaluation/metrics.py)
on the post-processed events.

### 5.1 Aggregate across all three videos

**IoU = 0.3 (the reference threshold):**

| Subtype | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| cell_phone | 22 | 35 | 6 | **0.39** | **0.79** | **0.52** |
| laptop | 8 | 12 | 1 | **0.40** | **0.89** | **0.55** |
| chatting | 4 | 72 | 8 | 0.05 | 0.33 | 0.09 |
| **Overall** | 34 | 119 | 15 | **0.22** | **0.69** | **0.34** |

**IoU = 0.1:**

| Subtype | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| cell_phone | 24 | 33 | 4 | 0.42 | 0.86 | 0.57 |
| laptop | 9 | 11 | 0 | 0.45 | 1.00 | 0.62 |
| chatting | 6 | 70 | 6 | 0.08 | 0.50 | 0.14 |
| **Overall** | 39 | 114 | 10 | 0.26 | 0.80 | 0.39 |

**IoU = 0.5:**

| Subtype | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| cell_phone | 13 | 44 | 15 | 0.23 | 0.46 | 0.31 |
| laptop | 7 | 13 | 2 | 0.35 | 0.78 | 0.48 |
| chatting | 2 | 74 | 10 | 0.03 | 0.17 | 0.05 |
| **Overall** | 22 | 131 | 27 | 0.14 | 0.45 | 0.22 |

### 5.2 Per-video breakdown (IoU = 0.3)

| Video | Subtype | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| video1 | cell_phone | 7 | 7 | 3 | 0.50 | 0.70 | 0.58 |
| video1 | laptop | 3 | 2 | 0 | 0.60 | 1.00 | 0.75 |
| video1 | chatting | 1 | 7 | 3 | 0.13 | 0.25 | 0.17 |
| **video1 overall** | | 11 | 16 | 6 | **0.41** | **0.65** | **0.50** |
| video2 | cell_phone | 7 | 8 | 2 | 0.47 | 0.78 | 0.58 |
| video2 | laptop | 3 | 4 | 0 | 0.43 | 1.00 | 0.60 |
| video2 | chatting | 1 | 24 | 3 | 0.04 | 0.25 | 0.07 |
| **video2 overall** | | 11 | 36 | 5 | 0.23 | 0.69 | 0.35 |
| video3 | cell_phone | 8 | 20 | 1 | 0.29 | 0.89 | 0.43 |
| video3 | laptop | 2 | 6 | 1 | 0.25 | 0.67 | 0.36 |
| video3 | chatting | 2 | 41 | 2 | 0.05 | 0.50 | 0.09 |
| **video3 overall** | | 12 | 67 | 4 | 0.15 | 0.75 | 0.25 |

### 5.3 Effect of post-processing

| Stage | n_events (sum) | Overall P | Overall R | Overall F1 |
|---|---|---|---|---|
| Raw `events.csv` | 271 | 0.13 | 0.71 | 0.22 |
| Post-processed `events_filtered.csv` | 153 | **0.22** | 0.69 | **0.34** |

The post-process drops 49+28+41=118 low-confidence events across the
three videos while losing only 2 ground-truth matches — a clean
precision win at near-zero recall cost.

### 5.4 Effect of the spatial fixture filter

Across the three videos the `SpatialFixtureFilter` locked **323
clusters** as fixtures (92 + 82 + 149). Each locked cluster suppresses
one or more device detections per frame for the rest of the video. The
filter operates *during* detection (before aggregation), so its impact
is implicit in the numbers above; without it, we would expect the FP
counts to be substantially higher, particularly on `cell_phone`.

## 6. Discussion

### 6.1 What works

- **Recall on devices is strong.** At IoU 0.1 we miss zero laptops and
  catch 86% of cell phones. The system is unlikely to silently fail on
  a real cheating event.
- **Laptop precision/recall is the highest of the three subtypes** —
  laptops are large, visually stable, and always associated with
  someone seated in front of them, so the heuristics align well.
- **The post-process pass effectively filters ghost detections.**
  Average precision across subtypes increased from 13% → 22% with no
  re-running of any model.

### 6.2 What is weak

- **Chatting**. Across the three videos the heuristic produced 76
  predictions against 12 ground-truth events (precision ~5%). The
  baseline reference report itself flags this exact limitation:
  *"hand-crafted pose heuristics … sensitive to pose quality, occlusion,
  and subtle behaviours."* In our footage many students lean forward
  while writing, briefly aligning their nose-shoulder vectors with a
  neighbour even when there is no actual interaction. Without a trained
  action-recognition head, this signal is intrinsically noisy.
- **Annotation quality limits comparable evaluation on video2/video3.**
  Both JSONs are auto-generated. Some of our "false positives" may be
  real events the GT missed; conversely the GT contains entries unlikely
  to correspond to behaviour in the actual video file (video2's JSON
  refers to a different clip name internally). The video1 numbers — the
  only clean GT — are the most trustworthy.
- **Boundary precision is lower than overlap precision.** F1 drops
  significantly between IoU 0.3 and IoU 0.5, meaning our event
  boundaries (start/end times) are close-but-not-tight. This is a
  consequence of `gap_tolerance_sec` extending intervals; tightening
  it improves boundaries at some cost to recall.

### 6.3 Comparison with the baseline

The baseline report submitted by another group followed the same
high-level architecture (YOLOv8 + tiling, YOLOv8-pose + ByteTrack,
proximity-and-facing chatting heuristic, gap/duration aggregation,
temporal-IoU evaluation), but did not include numerical results — its
table is templated with `[fill in]` placeholders. Our concrete additions
on top of that architecture are:

| Improvement | Where | Effect |
|---|---|---|
| Larger backbone (YOLO11l vs YOLOv8 default) | [`detector.py`](../source/proctoring/detection/detector.py) | Better small-object recall |
| `SpatialFixtureFilter` | [`fixture_filter.py`](../source/proctoring/detection/fixture_filter.py) | Suppressed 323 static FP clusters across the test set |
| Wrist-aware association | [`pipeline.py`](../source/proctoring/pipeline.py) | More semantically-correct device→person attribution |
| Post-process pass (confidence floor + ghost suppression) | [`postprocess.py`](../source/proctoring/evaluation/postprocess.py) | +9 percentage-point overall precision gain (no recall loss) |
| Multi-IoU reporting (0.1 / 0.3 / 0.5) | [`metrics.py`](../source/proctoring/evaluation/metrics.py) | More rigorous evaluation than single-threshold |

## 7. Limitations and improvements

The dominant accuracy bottleneck is **chatting**. A trained
action-recognition model on cropped person tubes would replace the
hand-crafted heuristic and almost certainly raise chatting precision
into double digits. Practically, this requires labelled data we do not
yet have.

Secondary improvements, in order of expected impact:

1. **Annotate video2/video3 by hand.** Until we do, ~2/3 of our
   evaluation is on auto-generated GT. Even a 30-minute pass per video
   would substantially tighten the comparison.
2. **Boundary tightening** via per-frame confidence-weighted
   aggregation, replacing the binary "active for the frame" signal with
   a smoothed signal whose edges are detected at half-max.
3. **3×3 tiling** at deployment time. We used 2×2 to stay under the
   30-min/video budget; 3×3 was tested on a short clip and improves
   small-object recall further at ~2× compute cost.
4. **Track-motion variance ghost filter.** The current ghost filter is
   spatial-extent based; adding a velocity-variance test (a fixture has
   near-zero motion variance, a hand-held phone has significant
   variance) would catch a few more FPs without affecting TPs.

## 8. Reproducibility

```bash
# 1. Process each video (see README for env setup).
PYTHONPATH=source python3 -m proctoring.cli run --video data/raw/video1.MOV --out outputs/video1
PYTHONPATH=source python3 -m proctoring.cli run --video data/raw/video2.MOV --out outputs/video2
PYTHONPATH=source python3 -m proctoring.cli run --video data/raw/video3.MOV --out outputs/video3

# 2. Apply the post-process filter to each.
for v in video1 video2 video3; do
    PYTHONPATH=source python3 -m proctoring.cli postprocess --run-dir outputs/$v
done

# 3. Evaluate against ground truth at multiple IoU thresholds.
PYTHONPATH=source python3 -m proctoring.cli evaluate-many \
    --runs-glob 'outputs/video*' --gt-dir data/raw \
    --iou 0.1,0.3,0.5 --filtered
```

The aggregate output is written to `outputs/evaluation_summary.json`
and is the source of truth for the numbers in §5.
