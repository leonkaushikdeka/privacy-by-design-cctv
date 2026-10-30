"""Irreversible masking, written into the frame buffer itself.

Gaussian blur is not offered: a known blur kernel can be partly inverted by deconvolution.
Body regions are pixelated to a handful of cells (each cell is an average of hundreds of
pixels, so the detail is gone). PIN regions are filled solid, because even a coarse
pixelated silhouette of a finger over a keypad can reveal which key row is being pressed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .regions import Box

PIN_FILL = (32, 32, 32)


def pixelate(frame: np.ndarray, box: Box, cells: int = 6) -> None:
    x1, y1, x2, y2 = box.as_int()
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h < 2 or w < 2:
        return
    scale = cells / max(h, w)
    small = cv2.resize(roi, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def solid(frame: np.ndarray, box: Box, color: tuple[int, int, int] = PIN_FILL) -> None:
    x1, y1, x2, y2 = box.as_int()
    frame[y1:y2, x1:x2] = color


def apply_masks(frame: np.ndarray, boxes: list[Box], cells: int = 6, style: str = "auto") -> np.ndarray:
    h, w = frame.shape[:2]
    for b in boxes:
        b = b.clip(w, h)
        if b.area <= 0:
            continue
        if style == "solid" or (style == "auto" and b.source == "pin"):
            solid(frame, b)
        else:
            pixelate(frame, b, cells)
    return frame


def mask_bitmap(shape: tuple[int, ...], boxes: list[Box]) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x1, y1, x2, y2 = b.clip(w, h).as_int()
        m[y1:y2, x1:x2] = True
    return m
