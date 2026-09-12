"""Device detector.

Runs YOLO in optional tiled mode for better recall on small objects.

Tiled inference cuts the frame into an ``Nx * Ny`` grid with ``overlap_frac``
overlap, runs the detector on each tile at full ``imgsz`` resolution, transforms
detections back to global image coordinates, and merges overlapping boxes with
class-aware NMS. This is much more sensitive to tiny objects (phones in back
rows) than a single global pass at the same resolution because the model sees a
larger relative size for the same object.

Persons are still extracted from the *global* pass; tiling persons would
fragment them across tiles and confuse downstream tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def cx(self) -> float:
        return 0.5 * (self.xyxy[0] + self.xyxy[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.xyxy[1] + self.xyxy[3])

    @property
    def area(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0]) * max(0.0, self.xyxy[3] - self.xyxy[1])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = rest[iou < iou_thr]
    return keep


class DeviceDetector:
    def __init__(
        self,
        weights: str | Path,
        device: str,
        conf: float,
        iou: float,
        imgsz: int,
        device_classes: dict[str, int],
        device_min_area_px: float,
        tiles_x: int = 1,
        tiles_y: int = 1,
        tile_overlap_frac: float = 0.0,
    ):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tiles_x = max(1, int(tiles_x))
        self.tiles_y = max(1, int(tiles_y))
        self.overlap = max(0.0, min(0.5, tile_overlap_frac))
        self.device_class_id_to_name: dict[int, str] = {v: k for k, v in device_classes.items()}
        self.device_min_area_px = device_min_area_px

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.tiles_x == 1 and self.tiles_y == 1:
            return self._detect_global(frame)
        return self._detect_tiled(frame)

    def _detect_global(self, frame: np.ndarray) -> list[Detection]:
        r = self._predict(frame)
        return self._extract_devices(r, ox=0, oy=0)

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tw = w / self.tiles_x
        th = h / self.tiles_y
        ow = tw * self.overlap
        oh = th * self.overlap
        all_dets: list[Detection] = []
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                x1 = max(0, int(tx * tw - ow))
                y1 = max(0, int(ty * th - oh))
                x2 = min(w, int((tx + 1) * tw + ow))
                y2 = min(h, int((ty + 1) * th + oh))
                tile = frame[y1:y2, x1:x2]
                r = self._predict(tile)
                all_dets.extend(self._extract_devices(r, ox=x1, oy=y1))
        return self._nms_by_class(all_dets)

    def _predict(self, image: np.ndarray):
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0] if results else None

    def _extract_devices(self, r, ox: int, oy: int) -> list[Detection]:
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        out: list[Detection] = []
        for box, c, cls_id in zip(xyxy, confs, cls_ids):
            if cls_id not in self.device_class_id_to_name:
                continue
            x1 = float(box[0]) + ox
            y1 = float(box[1]) + oy
            x2 = float(box[2]) + ox
            y2 = float(box[3]) + oy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < self.device_min_area_px:
                continue
            out.append(Detection(
                xyxy=(x1, y1, x2, y2),
                confidence=float(c),
                class_id=int(cls_id),
                class_name=self.device_class_id_to_name[int(cls_id)],
            ))
        return out

    def _nms_by_class(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        kept: list[Detection] = []
        # Group by class then NMS
        by_cls: dict[str, list[Detection]] = {}
        for d in dets:
            by_cls.setdefault(d.class_name, []).append(d)
        for cls, group in by_cls.items():
            arr = np.array([d.xyxy for d in group], dtype=np.float32)
            scores = np.array([d.confidence for d in group], dtype=np.float32)
            keep = _nms(arr, scores, self.iou)
            for i in keep:
                kept.append(group[i])
        return kept
