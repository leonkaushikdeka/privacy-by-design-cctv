"""Axis-aligned boxes and the small amount of geometry the pipeline needs."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    score: float = 1.0
    source: str = ""

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def expand(self, frac: float) -> Box:
        dx, dy = self.w * frac / 2, self.h * frac / 2
        return replace(self, x1=self.x1 - dx, y1=self.y1 - dy, x2=self.x2 + dx, y2=self.y2 + dy)

    def clip(self, width: int, height: int) -> Box:
        return replace(
            self,
            x1=min(max(self.x1, 0.0), width),
            y1=min(max(self.y1, 0.0), height),
            x2=min(max(self.x2, 0.0), width),
            y2=min(max(self.y2, 0.0), height),
        )

    def shift(self, dx: float, dy: float) -> Box:
        return replace(self, x1=self.x1 + dx, y1=self.y1 + dy, x2=self.x2 + dx, y2=self.y2 + dy)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersection(self, other: Box) -> float:
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, w) * max(0.0, h)

    def iou(self, other: Box) -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def union(self, other: Box) -> Box:
        return replace(
            self,
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        return int(np.floor(self.x1)), int(np.floor(self.y1)), int(np.ceil(self.x2)), int(np.ceil(self.y2))

    @staticmethod
    def from_points(points: np.ndarray, **kw) -> Box:
        return Box(float(points[:, 0].min()), float(points[:, 1].min()),
                   float(points[:, 0].max()), float(points[:, 1].max()), **kw)

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float, **kw) -> Box:
        return Box(float(x), float(y), float(x + w), float(y + h), **kw)
