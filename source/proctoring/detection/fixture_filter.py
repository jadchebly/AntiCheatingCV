"""Static-fixture suppression.

The COCO-pretrained YOLO detector occasionally locks onto small static features of
the desks (cable grommets, recessed handles, dark edges) and labels them as
``cell phone`` with mid-range confidence. A held phone — even when motionless —
has enough hand tremor to drift by several pixels between sampled frames, while
a fixture sits at the same pixel every time.

For each device class we maintain a running list of detection-centre clusters.
When a cluster accumulates many hits over a sustained interval it is "locked" as
a fixture, and any further detection falling inside it is rejected before the
alert pipeline sees it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _Cluster:
    cx: float
    cy: float
    count: int
    first_seen: float
    last_seen: float


@dataclass
class SpatialFixtureFilter:
    cluster_radius_px: float = 18.0
    min_count_to_lock: int = 6
    min_lifetime_sec: float = 6.0
    _by_class: dict[str, list[_Cluster]] = field(default_factory=dict)

    def is_fixture(self, class_name: str, cx: float, cy: float, ts: float) -> bool:
        cl = self._touch(class_name, cx, cy, ts)
        return (
            cl.count >= self.min_count_to_lock
            and (cl.last_seen - cl.first_seen) >= self.min_lifetime_sec
        )

    def _touch(self, class_name: str, cx: float, cy: float, ts: float) -> _Cluster:
        clusters = self._by_class.setdefault(class_name, [])
        for cl in clusters:
            if math.hypot(cl.cx - cx, cl.cy - cy) < self.cluster_radius_px:
                cl.count += 1
                cl.last_seen = ts
                w = 1.0 / cl.count
                cl.cx = (1 - w) * cl.cx + w * cx
                cl.cy = (1 - w) * cl.cy + w * cy
                return cl
        new = _Cluster(cx=cx, cy=cy, count=1, first_seen=ts, last_seen=ts)
        clusters.append(new)
        return new

    def locked_clusters(self) -> list[tuple[str, float, float, int]]:
        out = []
        for cls, clusters in self._by_class.items():
            for cl in clusters:
                if (
                    cl.count >= self.min_count_to_lock
                    and (cl.last_seen - cl.first_seen) >= self.min_lifetime_sec
                ):
                    out.append((cls, cl.cx, cl.cy, cl.count))
        return out
