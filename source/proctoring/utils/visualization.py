"""Drawing helpers for the annotated demo video."""

from __future__ import annotations

from typing import Iterable

import cv2

PERSON_COLOR = (200, 200, 200)
DEVICE_COLORS = {
    "cell_phone": (0, 80, 255),
    "laptop": (255, 120, 0),
}
ALERT_COLOR = (0, 0, 255)
CHAT_COLOR = (0, 255, 255)


def draw_box(frame, xyxy, color, label: str | None = None, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_persons(frame, persons: Iterable[dict]) -> None:
    for p in persons:
        label = f"id{p['track_id']}" if p.get("track_id") is not None else "person"
        draw_box(frame, p["xyxy"], PERSON_COLOR, label, thickness=1)


def draw_devices(frame, devices: Iterable[dict]) -> None:
    for d in devices:
        color = DEVICE_COLORS.get(d["class_name"], (0, 0, 255))
        label = f"{d['class_name']} {d['confidence']:.2f}"
        draw_box(frame, d["xyxy"], color, label, thickness=2)


def draw_active_events(frame, banners: list[str]) -> None:
    """Persistent top-of-frame banners for currently-active events."""
    if not banners:
        return
    h, w = frame.shape[:2]
    bar_h = 28 * len(banners) + 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), ALERT_COLOR, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, txt in enumerate(banners):
        cv2.putText(frame, txt, (10, 22 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_timestamp(frame, ts_sec: float) -> None:
    h, w = frame.shape[:2]
    txt = f"t={ts_sec:6.2f}s"
    cv2.putText(frame, txt, (w - 130, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, txt, (w - 130, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def draw_chat_link(frame, p1_xy, p2_xy) -> None:
    x1, y1 = int(p1_xy[0]), int(p1_xy[1])
    x2, y2 = int(p2_xy[0]), int(p2_xy[1])
    cv2.line(frame, (x1, y1), (x2, y2), CHAT_COLOR, 2)
    cv2.putText(frame, "chat", ((x1 + x2) // 2, (y1 + y2) // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CHAT_COLOR, 2, cv2.LINE_AA)
