"""Face-to-face chatting heuristic.

For each pair of tracked persons in a frame:

1. Compute centre-to-centre distance between their shoulder midpoints,
   normalised by the average shoulder width across the two. That ratio is a
   robust per-pair scale that survives perspective distortion.
2. Estimate each person's head yaw by projecting the nose offset onto their own
   shoulder axis — see ``_head_yaw`` for why this, and not a raw direction
   vector, is the right proxy.
3. Require both heads to be turned toward the other person.

A pair-frame fires the chatting flag for *both* track IDs if:
- proximity (in shoulder-widths) < ``proximity_max_shoulder_widths``, AND
- each person's head is yawed toward the other by at least
  ``min_head_turn_ratio``, AND
- the pair has held for ``min_consecutive_frames`` frames.

The pipeline maintains the per-track binary signal across frames and the
aggregator merges sustained intervals into events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from proctoring.tracking.pose_tracker import (
    LEFT_SHOULDER,
    NOSE,
    RIGHT_SHOULDER,
    TrackedPerson,
)

# Below this, the partner sits too close to straight ahead (or straight behind)
# for a sideways yaw test to mean anything, so the pair is skipped rather than
# guessed at.
MIN_LATERAL_COMPONENT = 0.30


@dataclass
class _PersonGeometry:
    track_id: int
    sh_mid: tuple[float, float]
    sh_w: float
    shoulder_axis: tuple[float, float]  # unit vector, right shoulder -> left
    yaw: float                          # signed, in half-shoulder-widths


@dataclass
class ChattingDetector:
    proximity_max_shoulder_widths: float = 2.5
    min_head_turn_ratio: float = 0.15
    min_pose_conf: float = 0.30
    min_consecutive_frames: int = 2

    # pair key -> consecutive frames the pair has satisfied the geometry
    _streaks: dict[frozenset[int], int] = field(default_factory=dict)

    def reset(self) -> None:
        self._streaks.clear()

    # ------------------------------------------------------------------
    def _geometry(self, p: TrackedPerson) -> _PersonGeometry | None:
        """Per-person geometry, or None when the pose is too weak to use."""
        sh_mid = p.shoulder_mid()
        sh_w = p.shoulder_width()
        if sh_mid is None or sh_w is None or sh_w < 5.0:
            return None
        if p.keypoints_conf[NOSE] < self.min_pose_conf:
            return None

        lx = float(p.keypoints_xy[LEFT_SHOULDER, 0])
        ly = float(p.keypoints_xy[LEFT_SHOULDER, 1])
        rx = float(p.keypoints_xy[RIGHT_SHOULDER, 0])
        ry = float(p.keypoints_xy[RIGHT_SHOULDER, 1])
        ax, ay = lx - rx, ly - ry
        amag = math.hypot(ax, ay)
        if amag < 1e-3:
            return None
        axis = (ax / amag, ay / amag)

        return _PersonGeometry(
            track_id=p.track_id,
            sh_mid=sh_mid,
            sh_w=sh_w,
            shoulder_axis=axis,
            yaw=self._head_yaw(p, sh_mid, sh_w, axis),
        )

    @staticmethod
    def _head_yaw(
        p: TrackedPerson,
        sh_mid: tuple[float, float],
        sh_w: float,
        axis: tuple[float, float],
    ) -> float:
        """Signed head yaw, in units of half a shoulder width.

        The nose always sits *above* the shoulder midpoint in image
        coordinates, so a raw (nose - shoulder_mid) direction vector points
        mostly upward and drags that large constant offset into any angle test
        against a partner sitting sideways. Projecting onto the shoulder axis
        discards the vertical component entirely and keeps only the part that
        encodes an actual head turn: how far the nose has slid toward one
        shoulder or the other.

        Result is ~0 looking straight ahead, approaching +/-1 when the nose
        sits over one shoulder. Positive means turned toward the left shoulder.
        """
        nx = float(p.keypoints_xy[NOSE, 0]) - sh_mid[0]
        ny = float(p.keypoints_xy[NOSE, 1]) - sh_mid[1]
        along = nx * axis[0] + ny * axis[1]
        return along / (0.5 * sh_w)

    # ------------------------------------------------------------------
    def detect(self, persons: list[TrackedPerson]) -> dict[int, set[int]]:
        """Return mapping ``track_id -> set of track_ids it is chatting with``."""
        geoms = [g for g in (self._geometry(p) for p in persons) if g is not None]

        fired_now: set[frozenset[int]] = set()
        pairs: dict[frozenset[int], tuple[int, int]] = {}
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                a, b = geoms[i], geoms[j]
                if not self._pair_is_chatting(a, b):
                    continue
                key = frozenset((a.track_id, b.track_id))
                fired_now.add(key)
                pairs[key] = (a.track_id, b.track_id)

        # Streak bookkeeping: a pair must hold for a few frames before it counts,
        # which suppresses single-frame pose noise. Pairs that drop out decay by
        # one rather than resetting, so a brief occlusion does not restart the
        # count from scratch.
        for key in list(self._streaks):
            if key not in fired_now:
                self._streaks[key] -= 1
                if self._streaks[key] <= 0:
                    del self._streaks[key]
        for key in fired_now:
            self._streaks[key] = min(self._streaks.get(key, 0) + 1,
                                     self.min_consecutive_frames)

        flags: dict[int, set[int]] = {}
        for key in fired_now:
            if self._streaks.get(key, 0) < self.min_consecutive_frames:
                continue
            ta, tb = pairs[key]
            flags.setdefault(ta, set()).add(tb)
            flags.setdefault(tb, set()).add(ta)
        return flags

    def _pair_is_chatting(self, a: _PersonGeometry, b: _PersonGeometry) -> bool:
        dx = b.sh_mid[0] - a.sh_mid[0]
        dy = b.sh_mid[1] - a.sh_mid[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return False

        avg_sw = 0.5 * (a.sh_w + b.sh_w)
        if avg_sw <= 0 or (dist / avg_sw) > self.proximity_max_shoulder_widths:
            return False

        d_hat = (dx / dist, dy / dist)
        return (self._faces(a, d_hat)
                and self._faces(b, (-d_hat[0], -d_hat[1])))

    def _faces(self, g: _PersonGeometry, to_partner: tuple[float, float]) -> bool:
        """True when ``g``'s head is turned toward ``to_partner``."""
        # Which side of this person the partner sits on, along their shoulder axis.
        lateral = to_partner[0] * g.shoulder_axis[0] + to_partner[1] * g.shoulder_axis[1]
        if abs(lateral) < MIN_LATERAL_COMPONENT:
            return False
        if abs(g.yaw) < self.min_head_turn_ratio:
            return False
        return (g.yaw > 0) == (lateral > 0)
