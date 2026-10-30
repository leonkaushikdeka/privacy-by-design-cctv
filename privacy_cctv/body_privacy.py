"""Pose estimation + body-part detection -> masks over nipples, buttocks and genitalia.

Pose is used three ways:
  1. ROI crops: the part detector runs on each person's chest-to-thigh crop, so a small
     person far down a corridor is seen at the detector's full 320 px resolution.
  2. Anatomical gating: a part found inside the skeleton region its class belongs to
     (breast -> chest, genitalia/buttocks -> pelvis) is accepted at a lower score.
  3. Anchoring: a detected part is stored relative to that person's hips, so when the
     detector misses a frame the mask keeps following the body instead of staying put.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import PART_REGION, PrivacyConfig
from .detectors.mp_models import Person
from .regions import Box


@dataclass(frozen=True)
class _Anchor:
    hip: np.ndarray
    torso: float
    rel: tuple[float, float, float, float]
    label: str
    ttl: int

    def project(self, person: Person) -> Box:
        t, (hx, hy) = person.torso_len, person.mid_hip
        rx1, ry1, rx2, ry2 = self.rel
        return Box(hx + rx1 * t, hy + ry1 * t, hx + rx2 * t, hy + ry2 * t,
                   label=self.label, score=0.0, source="body_anchor")


def _rel_iou(a: _Anchor, b: _Anchor) -> float:
    return Box(*a.rel).iou(Box(*b.rel))


def _anatomically_impossible(d: Box, key: str | None, p: Person) -> bool:
    """On an upright person a pelvis part cannot sit above the shoulders, nor a chest part above the nose."""
    if key is None or not p.box.expand(0.2).contains(*d.center):
        return False
    y = d.center[1]
    if key == "pelvis":
        return y < p.mid_shoulder[1]
    return y < p.keypoints[0, 1]


class BodyPrivacy:
    def __init__(self, detector, cfg: PrivacyConfig) -> None:
        self.detector = detector
        self.cfg = cfg
        self._anchors: list[_Anchor] = []

    def reset(self) -> None:
        self._anchors = []

    def detect_parts(self, frame: np.ndarray, people: list[Person]) -> list[Box]:
        h, w = frame.shape[:2]
        crops = [(frame, (0.0, 0.0))]
        for p in people:
            roi = p.roi().clip(w, h)
            if roi.w < 24 or roi.h < 24 or roi.area > 0.6 * w * h:
                continue
            x1, y1, x2, y2 = roi.as_int()
            crops.append((frame[y1:y2, x1:x2], (x1, y1)))
        if hasattr(self.detector, "batch"):
            return self.detector.batch(crops)
        return [b for img, off in crops for b in self.detector(img, offset=off)]

    def select(self, parts: list[Box], people: list[Person], hands: list = ()) -> list[tuple[Box, Person | None]]:
        """Keep sensitive parts that pass the (pose-dependent) score threshold. A detection with no
        pose support that lies mostly on a tracked hand is rejected: the detector confuses fingers
        with genitalia. One inside a person's matching region is kept even if a hand is near it."""
        regions = [(p, p.regions()) for p in people if p.plausible]
        chosen = []
        for d in parts:
            if d.label not in self.cfg.sensitive_parts:
                continue
            key = PART_REGION.get(d.label)
            if any(_anatomically_impossible(d, key, p) for p, _ in regions):
                continue
            owner = next((p for p, r in regions if key and r[key].contains(*d.center)), None)
            if owner is None and any(h.box.intersection(d) > 0.5 * d.area for h in hands):
                continue
            threshold = self.cfg.part_score_low if owner else self.cfg.part_score_high
            if d.score >= threshold:
                chosen.append((d, owner))
        return chosen

    def __call__(self, frame: np.ndarray, people: list[Person], hands: list = ()) -> list[Box]:
        return self.fuse(self.detect_parts(frame, people), people, hands)

    def fuse(self, parts: list[Box], people: list[Person], hands: list = ()) -> list[Box]:
        chosen = self.select(parts, people, hands)
        boxes = [d.expand(self.cfg.part_box_expand) for d, _ in chosen]
        carried, kept = self._carry(people)
        fresh = self._refresh(chosen)
        # drop a carried anchor once a fresh detection of the same part replaces it
        kept = [k for k in kept if not any(f.label == k.label and _rel_iou(f, k) > 0.3 for f in fresh)]
        self._anchors = fresh + kept
        return boxes + carried

    def _refresh(self, chosen: list[tuple[Box, Person | None]]) -> list[_Anchor]:
        out = []
        for d, p in chosen:
            if p is None:
                continue
            t, (hx, hy) = p.torso_len, p.mid_hip
            e = d.expand(self.cfg.part_box_expand)
            rel = ((e.x1 - hx) / t, (e.y1 - hy) / t, (e.x2 - hx) / t, (e.y2 - hy) / t)
            out.append(_Anchor(p.mid_hip.copy(), t, rel, d.label, self.cfg.hold_frames))
        return out

    def _carry(self, people: list[Person]) -> tuple[list[Box], list[_Anchor]]:
        """Each anchor may be carried for `ttl` more frames. If its person is not found this frame
        (pose flicker), the anchor waits and decays; the region's temporal hold covers the gap."""
        boxes, kept = [], []
        for a in self._anchors:
            if a.ttl < 1:
                continue
            near = [p for p in people if np.linalg.norm(p.mid_hip - a.hip) < 0.6 * max(a.torso, p.torso_len)]
            if not near:
                if a.ttl - 1 >= 1:
                    kept.append(replace(a, ttl=a.ttl - 1))
                continue
            p = min(near, key=lambda q: float(np.linalg.norm(q.mid_hip - a.hip)))
            boxes.append(a.project(p))
            kept.append(replace(a, hip=p.mid_hip.copy(), torso=p.torso_len, ttl=a.ttl - 1))
        return boxes, kept
