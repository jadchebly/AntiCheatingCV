# Submission checklist

The four mandatory deliverables (per [`docs/assignment-instructions.txt`](../docs/assignment-instructions.txt)) are:

1. **Source code** with install + execution instructions
2. **Technical report** (PDF)
3. **Recorded demonstration video**
4. **Individual contribution statement**

Below is what remains to do before we submit.

---

## 1. Source code — mostly done, verify before submitting

- [x] Modular package under [`source/proctoring/`](../source/proctoring/)
- [x] [`source/requirements.txt`](../source/requirements.txt) and [`source/environment.yml`](../source/environment.yml) populated
- [x] Configuration in [`source/configs/default.yaml`](../source/configs/default.yaml)
- [x] [`README.md`](../README.md) with install + run + evaluate instructions
- [ ] **Test the install on a clean machine** (fresh `venv` on a teammate's laptop). The brief says: *"If the professor is not able to execute your project, you will lose 50% of the credit."* This is the highest-risk item. Walk through the README from a fresh clone end-to-end.
- [ ] Decide whether to commit `models/yolo11l.pt` and `models/yolo11m-pose.pt` (~90 MB total) or rely on the auto-download on first run. Auto-download is what the README assumes; commit only if your submission target needs offline reproducibility.

## 2. Technical report — convert markdown to PDF

- [x] [`report/report.md`](report.md) populated with real numbers from the actual evaluation (§5)
- [ ] **Convert [`report/report.md`](report.md) to PDF** for submission. Options, easiest first:
  - VS Code: install *Markdown PDF* extension → right-click the file → *Markdown PDF: Export (pdf)*.
  - macOS Preview: open the markdown file in Typora / Marked 2 → File → Export → PDF.
  - Pandoc CLI: `pandoc report/report.md -o report/report.pdf --pdf-engine=xelatex`.
- [ ] **Proofread the PDF.** Check that:
  - All tables render correctly (the IoU 0.1/0.3/0.5 result tables are the highest-stakes content).
  - The ASCII pipeline diagram in §2 either renders as monospace or is replaced with a proper figure if your converter mangles it.
  - File paths in the report still point to real files in the repo.

## 3. Demonstration video — to record

- [x] Voiceover script in [`report/demo_script.md`](demo_script.md)
- [ ] **Record the demo (90–120 s).** Visuals: [`outputs/video1/annotated.mp4`](../outputs/video1/annotated.mp4) (cleanest of the three runs).
- [ ] Slow the annotated playback back to real-time before narrating — the pipeline writes at ~7.5 fps because of `frame_stride: 4`.
- [ ] Add captions / subtitles (the brief explicitly grades demo clarity).
- [ ] Export as MP4 at 1080p.

## 4. Individual contribution statement

- [x] [`report/contributions.md`](contributions.md) drafted with all seven members and per-module ownership
- [ ] **Verify the name spelling** for member listed as *Hala Yaghi* — guessed from the GitHub username `halayaghi-git`. Fix if wrong.
- [ ] Each member quickly skim their paragraph and adjust phrasing if they want.

## 5. Bundling and final submission

- [ ] Decide what to actually upload:
  - The full repo as a `.zip` is the safest option (code + report PDF + contributions). Exclude `outputs/*/annotated.mp4` (each ~300 MB) and `data/raw/*.MOV` (each ~500 MB) unless the LMS allows multi-GB uploads.
  - For the videos: either upload via the LMS if size allows, or a shared cloud link (Google Drive, Dropbox) referenced in the README.
- [ ] If excluding the source videos: add a short section to [`README.md`](../README.md) explaining how to obtain them.
- [ ] Submit through whatever portal the course uses, before the deadline.

## Optional, low-priority improvements (if there is time)

- [ ] Re-annotate `data/raw/video2.json` and `video3.json` by hand to remove the auto-generation caveat from the report's §4.1 footnote and §6.2.
- [ ] Re-run the pipeline with `tiles_x: 3, tiles_y: 3` in the config for slightly better small-object recall (~2× compute, ~1 hour total runtime).
- [ ] Render a fresh sample annotated frame for the report and embed it in §5 alongside the metric tables.
