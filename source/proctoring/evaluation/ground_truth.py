"""Ground-truth annotation loader.

JSON schema (one file per video):

    {
      "video": "video1.MOV",
      "duration_s": 540.0,
      "events": [
        {"start_s": 33.0, "end_s": 43.0, "subtype": "cell_phone"},
        ...
      ],
      "_note": "..."   // ignored
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GTEvent:
    subtype: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class GroundTruth:
    video: str
    duration_s: float
    events: list[GTEvent]

    @classmethod
    def load(cls, path: str | Path) -> "GroundTruth":
        with open(path, "r") as f:
            data = json.load(f)
        events: list[GTEvent] = []
        for e in data.get("events", []):
            try:
                start = float(e["start_s"])
                end = float(e["end_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            events.append(GTEvent(subtype=e["subtype"], start_s=start, end_s=end))
        return cls(
            video=data.get("video", ""),
            duration_s=float(data.get("duration_s", 0.0)),
            events=events,
        )

    def by_subtype(self) -> dict[str, list[GTEvent]]:
        out: dict[str, list[GTEvent]] = {}
        for e in self.events:
            out.setdefault(e.subtype, []).append(e)
        return out
