"""Event aggregation: convert per-frame booleans into stable interval events.

Each frame the pipeline produces a set of "active" signals — one per
``(subtype, track_id)`` pair (or ``track_id=None`` for non-attributable signals
like a phone with no nearby person). The aggregator buffers these, bridges
small gaps (``gap_tolerance_sec``), and discards too-short intervals
(``min_duration_sec``). The output is a list of ``Event`` records ready to be
written to the events CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Event:
    subtype: str
    start_sec: float
    end_sec: float
    track_id: int | None
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # for device events
    notes: str = ""

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class _OpenInterval:
    subtype: str
    track_id: int | None
    start_sec: float
    last_sec: float
    confidences: list[float] = field(default_factory=list)
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class EventAggregator:
    gap_tolerance_sec: dict[str, float]
    min_duration_sec: dict[str, float]
    _open: dict[tuple[str, int | None], _OpenInterval] = field(default_factory=dict)
    _closed: list[Event] = field(default_factory=list)

    def update(
        self,
        ts_sec: float,
        active_signals: Iterable[tuple[str, int | None, float, tuple[float, float, float, float] | None]],
    ) -> None:
        """Feed the aggregator one frame's worth of active signals.

        Each signal is ``(subtype, track_id, confidence, bbox_or_None)``. The
        aggregator opens new intervals for unseen pairs and extends ongoing ones.
        Pairs that go unseen for longer than their gap-tolerance are closed.
        """
        seen: set[tuple[str, int | None]] = set()
        for subtype, track_id, conf, bbox in active_signals:
            key = (subtype, track_id)
            seen.add(key)
            if key not in self._open:
                self._open[key] = _OpenInterval(
                    subtype=subtype,
                    track_id=track_id,
                    start_sec=ts_sec,
                    last_sec=ts_sec,
                )
            self._open[key].last_sec = ts_sec
            self._open[key].confidences.append(float(conf))
            if bbox is not None:
                self._open[key].bboxes.append(bbox)

        # Close any open interval whose gap exceeds tolerance.
        to_close: list[tuple[str, int | None]] = []
        for key, oi in self._open.items():
            if key in seen:
                continue
            gap_tol = float(self.gap_tolerance_sec.get(oi.subtype, 1.0))
            if (ts_sec - oi.last_sec) > gap_tol:
                to_close.append(key)
        for key in to_close:
            self._finalise(self._open.pop(key))

    def flush(self) -> list[Event]:
        """Close all remaining open intervals and return all events."""
        for oi in list(self._open.values()):
            self._finalise(oi)
        self._open.clear()
        return list(self._closed)

    def _finalise(self, oi: _OpenInterval) -> None:
        min_dur = float(self.min_duration_sec.get(oi.subtype, 0.0))
        dur = oi.last_sec - oi.start_sec
        if dur < min_dur:
            return
        avg_conf = sum(oi.confidences) / max(1, len(oi.confidences))
        bbox = None
        if oi.bboxes:
            xs1 = [b[0] for b in oi.bboxes]
            ys1 = [b[1] for b in oi.bboxes]
            xs2 = [b[2] for b in oi.bboxes]
            ys2 = [b[3] for b in oi.bboxes]
            bbox = (
                sum(xs1) / len(xs1), sum(ys1) / len(ys1),
                sum(xs2) / len(xs2), sum(ys2) / len(ys2),
            )
        self._closed.append(Event(
            subtype=oi.subtype,
            start_sec=oi.start_sec,
            end_sec=oi.last_sec,
            track_id=oi.track_id,
            confidence=avg_conf,
            bbox=bbox,
        ))

    def active_subtypes_at(self, ts_sec: float) -> set[str]:
        """For overlay banners: subtypes currently in an open interval."""
        return {key[0] for key in self._open}
