"""Face-to-face chatting heuristic.

For each pair of tracked persons in a frame:

1. Compute the distance between their shoulder midpoints, normalised by the
   average shoulder width across the two. That ratio is a per-pair scale that
   survives perspective distortion.
2. Estimate each person's head yaw from where the nose sits between the eyes —
   see ``_head_yaw`` for why the face, and not the shoulders, has to supply
   this.
3. Require both heads to be turned toward the other person.

A pair-frame fires the chatting flag for *both* track IDs if:
- proximity (in shoulder-widths) < ``proximity_max_shoulder_widths``, AND
- the partner sits far enough to one side for "toward" to be meaningful, AND
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
    LEFT_EAR,
    LEFT_EYE,
    NOSE,
    RIGHT_EAR,
    RIGHT_EYE,
    TrackedPerson,
)

# Below this share of the centre-to-centre distance, the partner sits too close
# to directly in front of or behind this person for a sideways yaw test to mean
# anything, so the pair is skipped rather than guessed at.
MIN_LATERAL_COMPONENT = 0.30

# A turned head foreshortens the eye pair; past a point the normalisation blows
# up and the sign gets noisy, so the ratio is capped.
MAX_YAW = 3.0

# Facial landmarks narrower than this (in pixels) are too small to trust.
MIN_FACE_HALF_WIDTH_PX = 2.0


@dataclass
class _PersonGeometry:
    track_id: int
    sh_mid: tuple[float, float]
    sh_w: float
    yaw: float  # signed; positive means the head is turned toward +x on screen


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
    def _head_yaw(self, p: TrackedPerson) -> float | None:
        """Signed head yaw from the face alone, or None if unmeasurable.

        Measured as how far the nose sits from the midpoint of the eyes,
        horizontally, in units of half the eye separation. ~0 looking straight
        at the camera, growing toward +/-1 and beyond as the head turns.

        The shoulders cannot supply this. The nose sits well above them, so any
        shoulder-referenced vector carries a large vertical offset, and once the
        shoulder line tilts — which it does for every seated student viewed from
        an angle — that offset leaks into the measurement and swamps the turn.
        Measured on real footage, shoulder-based yaw identified zero face-to-face
        pairs because it was reading camera angle rather than head rotation.

        Eyes are used in preference to ears: turning the head occludes the far
        ear, so ear-based yaw goes undefined in exactly the cases that matter.
        """
        for left, right in ((LEFT_EYE, RIGHT_EYE), (LEFT_EAR, RIGHT_EAR)):
            if min(p.keypoints_conf[left], p.keypoints_conf[right],
                   p.keypoints_conf[NOSE]) < self.min_pose_conf:
                continue
            lx = float(p.keypoints_xy[left, 0])
            rx = float(p.keypoints_xy[right, 0])
            half = abs(lx - rx) * 0.5
            if half < MIN_FACE_HALF_WIDTH_PX:
                continue
            nx = float(p.keypoints_xy[NOSE, 0])
            yaw = (nx - 0.5 * (lx + rx)) / half
            return max(-MAX_YAW, min(MAX_YAW, yaw))
        return None

    def _geometry(self, p: TrackedPerson) -> _PersonGeometry | None:
        """Per-person geometry, or None when the pose is too weak to use."""
        sh_mid = p.shoulder_mid()
        sh_w = p.shoulder_width()
        if sh_mid is None or sh_w is None or sh_w < 5.0:
            return None
        yaw = self._head_yaw(p)
        if yaw is None:
            return None
        return _PersonGeometry(track_id=p.track_id, sh_mid=sh_mid, sh_w=sh_w, yaw=yaw)

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

        # "Toward each other" only means something when they sit side by side.
        if abs(dx) / dist < MIN_LATERAL_COMPONENT:
            return False

        # B is at +x from A, so A must be turned +x and B turned -x.
        return self._faces(a, dx > 0) and self._faces(b, dx < 0)

    def _faces(self, g: _PersonGeometry, partner_is_right: bool) -> bool:
        """True when ``g``'s head is turned toward the partner's side."""
        if abs(g.yaw) < self.min_head_turn_ratio:
            return False
        return (g.yaw > 0) == partner_is_right
