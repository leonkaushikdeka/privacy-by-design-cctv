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


def _match(track: Box, box: Box, min_iou: float) -> float:
    """Overlap, or for a fast-moving object of the same kind, closeness of centres. Without the
    second test a phone moved across the frame leaves a trail of held masks behind it."""
    iou = track.iou(box)
    if iou >= min_iou:
        return 1.0 + iou
    if track.source != box.source:
        return 0.0
    (tx, ty), (bx, by) = track.center, box.center
    reach = 0.75 * max(track.w, track.h, box.w, box.h)
    dist = ((tx - bx) ** 2 + (ty - by) ** 2) ** 0.5
    return 1.0 - dist / reach if dist < reach else 0.0


class RegionMemory:
    def __init__(self, hold_frames: int, match_iou: float = 0.2) -> None:
        self.hold_frames = hold_frames
        self.match_iou = match_iou
        self._tracks: list[Track] = []

    def update(self, boxes: list[Box]) -> list[Box]:
        unmatched = list(range(len(self._tracks)))
        fresh: list[Track] = []
        for b in boxes:
            best, best_score = None, 0.0
            for i in unmatched:
                score = _match(self._tracks[i].box, b, self.match_iou)
                if score > best_score:
                    best, best_score = i, score
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
