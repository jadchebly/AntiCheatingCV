"""Temporal-IoU evaluation: match predicted intervals to ground-truth intervals.

A predicted event of subtype ``s`` matches a ground-truth event of subtype ``s``
if their temporal IoU (intersection over union of time intervals) exceeds the
threshold ``iou_thr``. Each ground-truth event matches at most one prediction
(the one with the highest IoU). Unmatched predictions are FP; unmatched ground
truths are FN.

We report precision, recall, and F1 per subtype, plus an aggregated overall.
For a more rigorous picture we also report at multiple IoU thresholds
(0.1, 0.3, 0.5).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from proctoring.evaluation.ground_truth import GroundTruth


def _temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / max(union, 1e-9)


def _read_predictions(events_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(events_csv, "r") as f:
        for r in csv.DictReader(f):
            rows.append({
                "subtype": r["subtype"],
                "start_sec": float(r["start_sec"]),
                "end_sec": float(r["end_sec"]),
                "confidence": float(r["mean_confidence"]) if r["mean_confidence"] else 0.0,
            })
    return rows


def evaluate(
    events_csv: Path,
    ground_truth_path: Path,
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    iou_thresholds = iou_thresholds or [0.1, 0.3, 0.5]
    gt = GroundTruth.load(ground_truth_path)
    preds = _read_predictions(Path(events_csv))

    by_iou: dict[str, dict[str, Any]] = {}
    classes = sorted(set([e.subtype for e in gt.events]) | {p["subtype"] for p in preds})

    for thr in iou_thresholds:
        per_class: dict[str, dict[str, Any]] = {}
        for cls in classes:
            gt_items = [e for e in gt.events if e.subtype == cls]
            pred_items = sorted(
                [p for p in preds if p["subtype"] == cls],
                key=lambda r: r["confidence"], reverse=True,
            )
            gt_matched = [False] * len(gt_items)
            pred_tp = [False] * len(pred_items)
            for i, p in enumerate(pred_items):
                best_j, best_iou = -1, 0.0
                for j, e in enumerate(gt_items):
                    if gt_matched[j]:
                        continue
                    iou = _temporal_iou(p["start_sec"], p["end_sec"], e.start_s, e.end_s)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= thr and best_j >= 0:
                    gt_matched[best_j] = True
                    pred_tp[i] = True
            tp = sum(pred_tp)
            fp = len(pred_items) - tp
            fn = len(gt_items) - sum(gt_matched)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) else 0.0
            )
            per_class[cls] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n_pred": len(pred_items),
                "n_gt": len(gt_items),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        # Overall (micro-averaged)
        agg = {"tp": 0, "fp": 0, "fn": 0}
        for v in per_class.values():
            for k in agg:
                agg[k] += v[k]
        p_total = agg["tp"] + agg["fp"]
        r_total = agg["tp"] + agg["fn"]
        prec = agg["tp"] / p_total if p_total else 0.0
        rec = agg["tp"] / r_total if r_total else 0.0
        overall = {
            **agg,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        }
        by_iou[f"iou_{thr:.1f}"] = {"per_class": per_class, "overall": overall}

    return {
        "video": gt.video,
        "duration_s": gt.duration_s,
        "n_predicted_events": len(preds),
        "n_ground_truth_events": len(gt.events),
        "by_iou": by_iou,
    }
