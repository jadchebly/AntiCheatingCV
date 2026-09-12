# Individual contribution statements

The project was split across seven members. Data collection and the
ground-truth annotation pass were group activities; the per-module
ownership below reflects who led each technical component.

---

## Georgios Klonis

System architecture and integration lead. Designed the end-to-end
pipeline ([`source/proctoring/pipeline.py`](../source/proctoring/pipeline.py))
that orchestrates detection, pose tracking, behaviour analysis, and
event aggregation. Implemented the static-fixture filter
([`detection/fixture_filter.py`](../source/proctoring/detection/fixture_filter.py))
that suppresses recurring desk-grommet false positives, the wrist-aware
device-to-person association heuristic, and the post-process pass
([`evaluation/postprocess.py`](../source/proctoring/evaluation/postprocess.py))
that filters low-confidence and ghost-track events. Wrote the CLI
entry point, the configuration system, and the project [README](../README.md)
and [technical report](report.md). Coordinated the three full-video runs
and the final evaluation.

## Fares Qaddoumi

Object detection module
([`source/proctoring/detection/detector.py`](../source/proctoring/detection/detector.py)).
Implemented the tiled-inference scheme that splits each frame into
overlapping regions and merges results with class-aware non-maximum
suppression — the change that made small back-row phones detectable
at all. Selected and benchmarked YOLO11l against the smaller backbones
on our own validation clips, and chose the confidence and IoU thresholds
used in production.

## Gregorio Santi Furnari

Pose tracking subsystem
([`source/proctoring/tracking/pose_tracker.py`](../source/proctoring/tracking/pose_tracker.py)).
Wrapped Ultralytics' YOLO11m-pose with ByteTrack to produce stable
per-student identities and 17-keypoint poses every frame. Defined the
`TrackedPerson` data structure, the helper accessors for shoulder
midpoint, shoulder width and wrist points, and tuned the tracker
configuration for the wide-FOV classroom setting where back-row
keypoints are noisy.

## Faisal Junblatt

Chatting behaviour heuristic
([`source/proctoring/behaviour/chatting.py`](../source/proctoring/behaviour/chatting.py)).
Designed and implemented the face-to-face proxy: pairwise proximity
measured in shoulder-widths combined with mutual nose-to-shoulder
gaze alignment. Iterated on the proximity and facing-tolerance
thresholds against our chatting clips in `data/raw/clips/c4.*`. Wrote
up the limitations of the heuristic for §6 of the report.

## Nicolás Leyva

Temporal event aggregation
([`source/proctoring/behaviour/aggregation.py`](../source/proctoring/behaviour/aggregation.py))
and CSV alert logging
([`source/proctoring/alerts/csv_logger.py`](../source/proctoring/alerts/csv_logger.py)).
Implemented the per-subtype gap-tolerance and minimum-duration logic
that converts noisy frame-level signals into stable interval events,
plus the structured CSV schema required by the assignment brief.

## Jad Chebly

Ground-truth annotation lead. Drove the labelling effort across the
three test videos: defined the JSON schema (`{video, duration_s, events:[{start_s, end_s, subtype}]}`),
audited the auto-generated annotations for video2 and video3, and
fixed the broken `start_s > end_s` interval in
[`data/raw/video2.json`](../data/raw/video2.json). Verified subtype
boundaries and produced the final ground-truth files used in §5 of
the report.

## Hala Yaghi

Evaluation framework
([`source/proctoring/evaluation/`](../source/proctoring/evaluation/)).
Implemented temporal-IoU matching against ground-truth intervals,
per-subtype precision/recall/F1, multi-threshold reporting (IoU at
0.1 / 0.3 / 0.5), and the aggregate-across-runs summary in
`evaluate-many`. Generated the metric tables in §5 of the report
and the per-video breakdown.

## Group activities

- **Data acquisition.** Recorded jointly with a second project group
  to get a larger, more diverse cast on camera. Some videos show only
  members of the other team; others show both teams seated together.
  Members of our group rotated as students and as camera operators
  across the sessions.
- **Annotation review.** The ground-truth JSONs were spot-checked by
  the full group before the final evaluation runs.
- **Demo and documentation.** Demo script, report review, and final
  PDF formatting were a group effort.
