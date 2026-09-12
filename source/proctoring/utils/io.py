"""Video I/O, config loading, and torch-device selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def select_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


class VideoReader:
    def __init__(self, path: str | Path, frame_stride: int = 1):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")
        self.frame_stride = max(1, int(frame_stride))
        self.meta = VideoMeta(
            fps=self.cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __iter__(self) -> Iterator[tuple[int, float, "cv2.typing.MatLike"]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if idx % self.frame_stride == 0:
                ts = idx / self.meta.fps
                yield idx, ts, frame
            idx += 1
        self.cap.release()

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    def __init__(self, path: str | Path, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"failed to open writer: {self.path}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
