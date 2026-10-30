"""Explainer overlays for the notebook and report.

These draw on a copy of the input to show what each model saw. They exist only in the
development prototype; the camera build has no path that renders unmasked pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import FrameResult
from .regions import Box

POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

COLORS = {
    "pose": (255, 170, 0),
    "chest": (255, 120, 200),
    "pelvis": (200, 120, 255),
    "hand": (0, 220, 120),
    "device": (0, 200, 255),
    "pin": (60, 60, 255),
    "body": (255, 60, 160),
}


def _box(img: np.ndarray, b: Box, color, text: str = "", thick: int = 2) -> None:
    x1, y1, x2, y2 = b.as_int()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def overlay(raw: np.ndarray, r: FrameResult, regions_from_pose: bool = True) -> np.ndarray:
    img = raw.copy()
    for p in r.people:
        kp = p.keypoints
        for a, b in POSE_EDGES:
            if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                cv2.line(img, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), COLORS["pose"], 2, cv2.LINE_AA)
        if regions_from_pose:
            for name, box in p.regions().items():
                _box(img, box, COLORS[name], name, 1)
    for h in r.hands:
        pts = h.landmarks.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), COLORS["hand"], 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
    for d in r.devices:
        _box(img, d, COLORS["device"], f"{d.label} {d.score:.2f}")
    for reg in r.regions:
        color = COLORS["pin"] if reg.source == "pin" else COLORS["body"]
        _box(img, reg, color, reg.label.replace("_EXPOSED", "").lower() if reg.source != "pin" else "PIN entry", 3)
    return img


def label_strip(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(*imgs: np.ndarray, height: int = 360) -> np.ndarray:
    rs = [cv2.resize(i, (int(i.shape[1] * height / i.shape[0]), height)) for i in imgs]
    gap = np.full((height, 8, 3), 255, np.uint8)
    out = rs[0]
    for r in rs[1:]:
        out = np.hstack([out, gap, r])
    return out
