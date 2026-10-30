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
    def __init__(self, source: int | str, pipeline: PrivacyPipeline, width: int = 640, height: int = 480) -> None:
        self.source = source
        self.pipeline = pipeline
        self.size = (width, height)
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


def hud(result: FrameResult, fps: float) -> np.ndarray:
    """Status line drawn on the (already masked) output frame."""
    f = result.frame
    state = "FAULT" if result.fault else ("PIN ENTRY - MASKED" if result.pin_active else "monitoring")
    text = f"{fps:5.1f} fps | {result.timings_ms.get('total', 0):5.1f} ms | masks {len(result.regions)} | {state}"
    cv2.rectangle(f, (0, f.shape[0] - 26), (f.shape[1], f.shape[0]), (0, 0, 0), -1)
    cv2.putText(f, text, (8, f.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return f
