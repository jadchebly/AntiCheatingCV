"""Command-line entry point.

    python -m proctoring.cli run --video data/raw/video1.MOV --out outputs/video1
    python -m proctoring.cli evaluate --run-dir outputs/video1 --gt data/raw/video1.json
    python -m proctoring.cli evaluate-many --runs-glob 'outputs/video*' --gt-dir data/raw
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from proctoring.evaluation.metrics import evaluate
from proctoring.evaluation.postprocess import filter_run
from proctoring.pipeline import run_video
from proctoring.utils.io import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "source" / "configs" / "default.yaml"


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    summary = run_video(args.video, args.out, config)
    print(json.dumps({
        "video": summary.video,
        "duration_sec": summary.duration_sec,
        "n_events": len(summary.events),
        "fixture_clusters": len(summary.fixture_clusters),
    }, indent=2))
    return 0


def cmd_postprocess(args: argparse.Namespace) -> int:
    stats = filter_run(Path(args.run_dir))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    events_filename = "events_filtered.csv" if args.filtered else "events.csv"
    metrics = evaluate(
        events_csv=Path(args.run_dir) / events_filename,
        ground_truth_path=Path(args.gt),
        iou_thresholds=[float(x) for x in args.iou.split(",")],
    )
    out = Path(args.run_dir) / "evaluation.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


def cmd_evaluate_many(args: argparse.Namespace) -> int:
    """Evaluate every run-dir matching the glob against its corresponding GT JSON."""
    iou_thresholds = [float(x) for x in args.iou.split(",")]
    events_filename = "events_filtered.csv" if args.filtered else "events.csv"
    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        print(f"no run dirs matched: {args.runs_glob}", file=sys.stderr)
        return 1
    gt_dir = Path(args.gt_dir)
    all_results: dict[str, dict] = {}
    for d in run_dirs:
        rd = Path(d)
        # Convention: run dir basename matches GT JSON basename
        candidates = list(gt_dir.glob(f"{rd.name}.json"))
        if not candidates:
            print(f"  skip {rd.name} (no matching GT json)", file=sys.stderr)
            continue
        gt_path = candidates[0]
        ev = rd / events_filename
        if not ev.exists():
            print(f"  skip {rd.name} (no {events_filename})", file=sys.stderr)
            continue
        metrics = evaluate(events_csv=ev, ground_truth_path=gt_path,
                           iou_thresholds=iou_thresholds)
        with open(rd / "evaluation.json", "w") as f:
            json.dump(metrics, f, indent=2)
        all_results[rd.name] = metrics
        # Print short summary line
        ov = metrics["by_iou"].get(f"iou_{iou_thresholds[1]:.1f}",
                                   metrics["by_iou"][f"iou_{iou_thresholds[0]:.1f}"])["overall"]
        print(f"  {rd.name}: P={ov['precision']:.2f} R={ov['recall']:.2f} F1={ov['f1']:.2f}")

    # Aggregate across runs
    agg = _aggregate(all_results, iou_thresholds)
    out = Path(args.runs_glob.split('*')[0]).parent / "evaluation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_run": all_results, "aggregate": agg}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


def _aggregate(results: dict[str, dict], iou_thresholds: list[float]) -> dict:
    agg: dict[str, dict] = {}
    for thr in iou_thresholds:
        key = f"iou_{thr:.1f}"
        per_class: dict[str, dict[str, int]] = {}
        for run_name, m in results.items():
            cls_data = m["by_iou"][key]["per_class"]
            for cls, v in cls_data.items():
                pc = per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
                for k in ("tp", "fp", "fn"):
                    pc[k] += v[k]
        # Compute per-class precision/recall/F1 from summed counts
        class_summary = {}
        agg_tp = agg_fp = agg_fn = 0
        for cls, c in per_class.items():
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            agg_tp += tp; agg_fp += fp; agg_fn += fn
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            class_summary[cls] = {**c, "precision": round(p, 4),
                                  "recall": round(r, 4), "f1": round(f1, 4)}
        p = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
        r = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        agg[key] = {
            "per_class": class_summary,
            "overall": {
                "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            },
        }
    return agg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proctoring")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Process a video.")
    pr.add_argument("--video", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--config", default=str(DEFAULT_CONFIG))
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("evaluate", help="Compare a run to ground truth.")
    pe.add_argument("--run-dir", required=True)
    pe.add_argument("--gt", required=True)
    pe.add_argument("--iou", default="0.1,0.3,0.5",
                    help="Comma-separated IoU thresholds")
    pe.add_argument("--filtered", action="store_true",
                    help="Use events_filtered.csv instead of events.csv")
    pe.set_defaults(func=cmd_evaluate)

    pp = sub.add_parser("postprocess",
                        help="Drop low-confidence and ghost-track events.")
    pp.add_argument("--run-dir", required=True)
    pp.set_defaults(func=cmd_postprocess)

    pem = sub.add_parser("evaluate-many",
                         help="Evaluate every run dir whose name matches a GT json.")
    pem.add_argument("--runs-glob", required=True)
    pem.add_argument("--gt-dir", required=True)
    pem.add_argument("--iou", default="0.1,0.3,0.5")
    pem.add_argument("--filtered", action="store_true",
                     help="Use events_filtered.csv instead of events.csv")
    pem.set_defaults(func=cmd_evaluate_many)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
