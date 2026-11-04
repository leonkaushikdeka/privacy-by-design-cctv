"""Simulated camera boundary for the laptop prototype.

On a real AI camera the sensor writes into a buffer that only the NPU pipeline can read, and
the encoder only reads the pipeline's output. Here one reusable buffer plays that role: each
frame is captured into it, masked in place, and only then handed out. There is no method that
returns a frame before masking.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .pipeline import FrameResult, PrivacyPipeline


class EdgeCamera:
    def __init__(self, source: int | str, pipeline: PrivacyPipeline, width: int = 640, height: int = 480,
                 mirror: bool = False) -> None:
        self.source = source
        self.pipeline = pipeline
        self.size = (width, height)
        self.mirror = mirror
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self) -> EdgeCamera:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.source!r}")
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
        self.pipeline.close()

    def frames(self, limit: int | None = None) -> Iterator[FrameResult]:
        if self._cap is None:
            raise RuntimeError("use EdgeCamera as a context manager")
        start = time.monotonic()
        n = 0
        while limit is None or n < limit:
            ok, frame = self._cap.read()
            if not ok:
                return
            if frame.shape[1] != self.size[0] and isinstance(self.source, str):
                frame = cv2.resize(frame, self.size)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            ts = int((time.monotonic() - start) * 1000)
            yield self.pipeline.process(frame, ts)
            n += 1


class MaskedRecorder:
    """Writes only pipeline output; it accepts a FrameResult, never a raw array."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self._w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, result: FrameResult) -> None:
        self._w.write(result.frame)

    def close(self) -> None:
        self._w.release()


REGION_STYLE = {
    "pin": ("PIN ENTRY", (60, 60, 255)),
    "device": ("PHONE / POS", (0, 190, 255)),
    "card": ("CARD", (180, 200, 40)),
}
BODY_STYLE = ("PRIVATE", (200, 80, 255))


def state_of(result: FrameResult) -> str:
    if result.fault:
        return "FAULT: FRAME WITHHELD"
    sources = {r.source for r in result.regions}
    for source, word in (("pin", "PIN ENTRY MASKED"), ("card", "CARD MASKED"), ("device", "PHONE MASKED")):
        if source in sources:
            return word
    return "BODY MASKED" if sources else "monitoring"


def outline(result: FrameResult) -> np.ndarray:
    """Labels around masked regions, for presenting. Drawn on the masked frame, so they only
    outline what is already hidden."""
    f = result.frame
    for r in result.regions:
        word, color = REGION_STYLE.get(r.source, BODY_STYLE)
        x1, y1, x2, y2 = r.clip(f.shape[1], f.shape[0]).as_int()
        cv2.rectangle(f, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(f, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(f, word, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return f


def hud(result: FrameResult, fps: float) -> np.ndarray:
    """Status line drawn on the (already masked) output frame."""
    f = result.frame
    text = f"{fps:5.1f} fps | {result.timings_ms.get('total', 0):5.1f} ms | masks {len(result.regions)} | {state_of(result)}"
    cv2.rectangle(f, (0, f.shape[0] - 26), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
    cv2.putText(f, text, (8, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return f
