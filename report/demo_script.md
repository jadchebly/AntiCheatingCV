# Demonstration video — recording script

Target length: **90–120 seconds**. Visuals come from
`outputs/video1/annotated.mp4` (the cleanest of the three runs).
Capture narration with screen-recording over real-time playback.

## Shot 1 — Title (0:00–0:08)
> "Intelligent exam-proctoring system. Computer vision pipeline that
> flags cell phones, laptops, and chatting in classroom exam footage."

## Shot 2 — Wide classroom view (0:08–0:25)
Play first ~15 seconds of `outputs/video1/annotated.mp4`.
> "The input is a wide classroom shot. Multiple students, several rows,
> a fixed camera. We process it offline at about 7.5 frames per second
> in steady state, on a laptop GPU."

## Shot 3 — Detection close-up (0:25–0:50)
Scrub to 1:01 in the annotated video where a `cell_phone 0.66` event
fires for track id 74.
> "YOLO11-large runs in 2-by-2 tiled mode for small-object recall —
> phones in back-row desks can be only 30 pixels across. ByteTrack on
> top of YOLO-pose gives us stable identities, so each alert is
> attributed to a specific student."

## Shot 4 — Fixture filter & post-process (0:50–1:15)
Cut to `events.csv` vs `events_filtered.csv` side by side.
> "We added two layers the baseline approach doesn't: a spatial fixture
> filter that locks onto desk grommets and outlets so the detector
> can't keep mistaking them for phones, and a post-process pass that
> drops low-confidence and ghost-track events. Together they took our
> overall precision from 13% to 22%, with no recall loss."

## Shot 5 — Chatting heuristic (1:15–1:35)
Show a frame with the cyan chat link drawn between two students.
> "Chatting is detected with a face-to-face proxy: two students close
> enough by shoulder-width and both heads turned toward each other.
> It's interpretable but noisy; it's the dominant source of false
> positives in our evaluation. The report discusses this honestly."

## Shot 6 — Output artefacts (1:35–1:50)
File-tree shot of `outputs/video1/`.
> "Each run produces a structured event log, an annotated MP4 for
> manual review, and a metrics file. A separate evaluation pass scores
> against time-stamped ground-truth JSON using temporal IoU at multiple
> thresholds."

## Production tips

- Record at 1920×1080 to match the source resolution.
- The annotated MP4 plays at ~7.5 fps because of the pipeline's frame
  stride. Slow it back down to real-time in the editor before
  narration.
- Caption every voiceover line — the assignment explicitly evaluates
  demo clarity.
