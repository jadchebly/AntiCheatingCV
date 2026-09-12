"""Face-to-face chatting heuristic.

For each pair of tracked persons in a frame:

1. Compute centre-to-centre distance between their shoulder midpoints.
   Normalise by the average shoulder width across the two — this is a robust
   per-pair scale that survives perspective distortion.
2. Compute each person's gaze direction proxy as the unit vector from their
   shoulder midpoint to their nose.
3. Check if both vectors point toward the other person within
   ``facing_tolerance_deg`` of "directly facing".

A pair-frame fires the chatting flag for *both* track IDs in the pair if:
- proximity (in shoulder-widths) < ``proximity_max_shoulder_widths`` AND
- both nose-to-shoulder vectors face the partner within tolerance.

The pipeline maintains the per-track binary signal across frames and the
aggregator merges sustained intervals into events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from proctoring.tracking.pose_tracker import NOSE, TrackedPerson


@dataclass
class ChattingDetector:
    proximity_max_shoulder_widths: float = 3.0
    facing_tolerance_deg: float = 50.0
    min_pose_conf: float = 0.30

    def detect(self, persons: list[TrackedPerson]) -> dict[int, set[int]]:
        """Return mapping ``track_id -> set of track_ids it is chatting with``."""
        flags: dict[int, set[int]] = {}
        # Pre-compute per-person geometry so we don't redo it per pair.
        info: list[dict] = []
        for p in persons:
            sh_mid = p.shoulder_mid()
            sh_w = p.shoulder_width()
            if sh_mid is None or sh_w is None or sh_w < 5.0:
                info.append({})  # placeholder, will be skipped
                continue
            if p.keypoints_conf[NOSE] < self.min_pose_conf:
                info.append({})
                continue
            nose = (float(p.keypoints_xy[NOSE, 0]), float(p.keypoints_xy[NOSE, 1]))
            # Gaze proxy: nose direction relative to shoulder midpoint.
            gx, gy = nose[0] - sh_mid[0], nose[1] - sh_mid[1]
            mag = math.hypot(gx, gy)
            if mag < 1e-3:
                info.append({})
                continue
            info.append({
                "track_id": p.track_id,
                "sh_mid": sh_mid,
                "sh_w": sh_w,
                "gaze": (gx / mag, gy / mag),
            })

        cos_tol = math.cos(math.radians(self.facing_tolerance_deg))
        for i in range(len(info)):
            if not info[i]:
                continue
            for j in range(i + 1, len(info)):
                if not info[j]:
                    continue
                a, b = info[i], info[j]
                ax, ay = a["sh_mid"]
                bx, by = b["sh_mid"]
                dx, dy = bx - ax, by - ay
                dist = math.hypot(dx, dy)
                avg_sw = 0.5 * (a["sh_w"] + b["sh_w"])
                if avg_sw <= 0:
                    continue
                if (dist / avg_sw) > self.proximity_max_shoulder_widths:
                    continue
                # Direction from a to b, and b to a.
                if dist < 1e-3:
                    continue
                a_to_b = (dx / dist, dy / dist)
                b_to_a = (-a_to_b[0], -a_to_b[1])
                # Each gaze vector should point reasonably toward the partner.
                dot_a = a["gaze"][0] * a_to_b[0] + a["gaze"][1] * a_to_b[1]
                dot_b = b["gaze"][0] * b_to_a[0] + b["gaze"][1] * b_to_a[1]
                if dot_a >= cos_tol and dot_b >= cos_tol:
                    flags.setdefault(a["track_id"], set()).add(b["track_id"])
                    flags.setdefault(b["track_id"], set()).add(a["track_id"])
        return flags
