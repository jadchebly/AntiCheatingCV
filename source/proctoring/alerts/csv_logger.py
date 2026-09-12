"""Write aggregated events to a CSV file.

Mandatory output of the assignment: a CSV log of every detected violation with
timestamps. We keep it human-readable: one row per event, columns
``event_id, subtype, start_sec, end_sec, duration_sec, track_id,
mean_confidence, x1, y1, x2, y2, notes``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from proctoring.behaviour.aggregation import Event

CSV_HEADER = [
    "event_id",
    "subtype",
    "start_sec",
    "end_sec",
    "duration_sec",
    "track_id",
    "mean_confidence",
    "x1", "y1", "x2", "y2",
    "notes",
]


def write_events_csv(path: str | Path, events: list[Event]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for i, ev in enumerate(sorted(events, key=lambda e: (e.start_sec, e.subtype))):
            x1, y1, x2, y2 = ev.bbox if ev.bbox is not None else (None, None, None, None)
            w.writerow([
                i,
                ev.subtype,
                f"{ev.start_sec:.2f}",
                f"{ev.end_sec:.2f}",
                f"{ev.duration_sec:.2f}",
                "" if ev.track_id is None else ev.track_id,
                f"{ev.confidence:.3f}",
                "" if x1 is None else f"{x1:.1f}",
                "" if y1 is None else f"{y1:.1f}",
                "" if x2 is None else f"{x2:.1f}",
                "" if y2 is None else f"{y2:.1f}",
                ev.notes,
            ])
