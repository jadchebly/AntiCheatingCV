"""Pose tracker.

Wraps Ultralytics' YOLO-pose with built-in ByteTrack to assign stable identities
to people across frames. Each frame yields a list of ``TrackedPerson`` records
containing the bounding box, track ID, and 17 COCO keypoints.

Keypoint order (COCO):
    0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
    5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
    9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
    13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Indices we use elsewhere.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10


@dataclass
class TrackedPerson:
    track_id: int
    xyxy: tuple[float, float, float, float]
    keypoints_xy: np.ndarray   # (17, 2)
    keypoints_conf: np.ndarray # (17,)
    confidence: float

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    def shoulder_mid(self) -> tuple[float, float] | None:
        if (self.keypoints_conf[LEFT_SHOULDER] < 0.3 or
                self.keypoints_conf[RIGHT_SHOULDER] < 0.3):
            return None
        return (
            0.5 * (self.keypoints_xy[LEFT_SHOULDER, 0] + self.keypoints_xy[RIGHT_SHOULDER, 0]),
            0.5 * (self.keypoints_xy[LEFT_SHOULDER, 1] + self.keypoints_xy[RIGHT_SHOULDER, 1]),
        )

    def shoulder_width(self) -> float | None:
        if (self.keypoints_conf[LEFT_SHOULDER] < 0.3 or
                self.keypoints_conf[RIGHT_SHOULDER] < 0.3):
            return None
        return float(np.hypot(
            self.keypoints_xy[LEFT_SHOULDER, 0] - self.keypoints_xy[RIGHT_SHOULDER, 0],
            self.keypoints_xy[LEFT_SHOULDER, 1] - self.keypoints_xy[RIGHT_SHOULDER, 1],
        ))

    def wrist_points(self, conf_min: float = 0.3) -> list[tuple[float, float]]:
        out = []
        if self.keypoints_conf[LEFT_WRIST] >= conf_min:
            out.append((float(self.keypoints_xy[LEFT_WRIST, 0]),
                        float(self.keypoints_xy[LEFT_WRIST, 1])))
        if self.keypoints_conf[RIGHT_WRIST] >= conf_min:
            out.append((float(self.keypoints_xy[RIGHT_WRIST, 0]),
                        float(self.keypoints_xy[RIGHT_WRIST, 1])))
        return out


class PoseTracker:
    def __init__(
        self,
        weights: str | Path,
        device: str,
        imgsz: int,
        tracker_cfg: str = "bytetrack.yaml",
        conf: float = 0.30,
    ):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.tracker_cfg = tracker_cfg
        self.conf = conf

    def track(self, frame: np.ndarray) -> list[TrackedPerson]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(xyxy), -1)
        kp_xy = r.keypoints.xy.cpu().numpy()
        kp_cf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kp_xy.shape[:2])

        out: list[TrackedPerson] = []
        for i in range(len(xyxy)):
            tid = int(ids[i])
            if tid < 0:
                continue
            box = (float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]))
            out.append(TrackedPerson(
                track_id=tid,
                xyxy=box,
                keypoints_xy=kp_xy[i],
                keypoints_conf=kp_cf[i],
                confidence=float(confs[i]),
            ))
        return out

    def reset(self) -> None:
        try:
            self.model.predictor.trackers[0].reset()  # type: ignore[attr-defined]
        except (AttributeError, IndexError):
            pass
