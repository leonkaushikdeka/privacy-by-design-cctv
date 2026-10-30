"""The three things the prototype is tested on:

1. Protection - how much of each sensitive ground-truth region ended up masked.
2. Over-masking - how much masked area falls outside every sensitive region.
3. Speed - per-stage latency and frames per second.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .masking import mask_bitmap
from .regions import Box


@dataclass(frozen=True)
class ImageScore:
    name: str
    gt_regions: int
    protected: int  # ground-truth regions covered at least `protect_at`
    mean_coverage: float  # mean fraction of each ground-truth region that is masked (nan if none)
    masked_frac: float  # masked pixels / frame pixels
    over_mask_frac: float  # masked pixels outside ground truth / frame pixels
    false_alarm: bool  # anything masked in an image with no sensitive region


def score_image(name: str, shape: tuple[int, ...], predicted: list[Box], truth: list[Box],
                ignore: list[Box] = (), protect_at: float = 0.9) -> ImageScore:
    """`ignore` marks ambiguous areas: masking them is neither required nor penalised."""
    pred = mask_bitmap(shape, predicted) & ~mask_bitmap(shape, list(ignore))
    gt = mask_bitmap(shape, truth)
    total = pred.size
    covers = []
    for b in truth:
        x1, y1, x2, y2 = b.clip(shape[1], shape[0]).as_int()
        if x2 > x1 and y2 > y1:
            covers.append(float(pred[y1:y2, x1:x2].mean()))
    return ImageScore(
        name=name,
        gt_regions=len(covers),
        protected=sum(c >= protect_at for c in covers),
        mean_coverage=float(np.mean(covers)) if covers else float("nan"),
        masked_frac=float(pred.sum() / total),
        over_mask_frac=float((pred & ~gt).sum() / total),
        false_alarm=bool(not covers and pred.any()),
    )


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / a.mean()),
    }


def leak_rate(detected: np.ndarray, hold_frames: int) -> float:
    """Fraction of truly-sensitive frames left unmasked, given a per-frame detection sequence
    (all frames sensitive) and a hold of `hold_frames` frames after each detection."""
    masked = np.zeros_like(detected, dtype=bool)
    ttl = -1
    for i, d in enumerate(detected):
        ttl = hold_frames if d else ttl - 1
        masked[i] = bool(d) or ttl >= 0
    return float(1.0 - masked.mean())
