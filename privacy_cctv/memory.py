"""Temporal hold: a region stays masked for `hold_frames` after its last detection.

Detectors flicker. One missed frame of a PIN pad is enough to leak a digit, so masks are
added the moment evidence appears and only released after a run of empty frames.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .regions import Box


@dataclass(frozen=True)
class Track:
    box: Box
    ttl: int
    hits: int


class RegionMemory:
    def __init__(self, hold_frames: int, match_iou: float = 0.2) -> None:
        self.hold_frames = hold_frames
        self.match_iou = match_iou
        self._tracks: list[Track] = []

    def update(self, boxes: list[Box]) -> list[Box]:
        unmatched = list(range(len(self._tracks)))
        fresh: list[Track] = []
        for b in boxes:
            best, best_iou = None, self.match_iou
            for i in unmatched:
                iou = self._tracks[i].box.iou(b)
                if iou >= best_iou:
                    best, best_iou = i, iou
            if best is not None:
                unmatched.remove(best)
                fresh.append(Track(b, self.hold_frames, self._tracks[best].hits + 1))
            else:
                fresh.append(Track(b, self.hold_frames, 1))
        held = [replace(self._tracks[i], ttl=self._tracks[i].ttl - 1) for i in unmatched]
        # ttl counts the empty frames still allowed: hold_frames of them, then the region is released
        self._tracks = fresh + [t for t in held if t.ttl >= 0]
        return [t.box for t in self._tracks]

    def reset(self) -> None:
        self._tracks = []

    def __len__(self) -> int:
        return len(self._tracks)
