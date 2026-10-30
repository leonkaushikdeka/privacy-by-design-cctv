"""Hand tracking + device detection -> mask the screen/keypad and hand during PIN entry.

A hand is treated as entering a PIN when a fingertip (index, middle or thumb) is on the face
of a phone or POS device. Pointing with an extended index finger and fingertip tapping over
recent frames raise the confidence, but are not required: the guard is deliberately
high-recall, because a missed keystroke is a leaked digit while an extra mask over a phone
costs the footage nothing of evidential value.

While the hand covers the device the detector often loses the device. The guard keeps the
last device box relative to the wrist and carries it with the hand for `hold_frames`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .config import PrivacyConfig
from .detectors.mp_models import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, WRIST, Hand
from .regions import Box

TOUCH_TIPS = (INDEX_TIP, MIDDLE_TIP, THUMB_TIP)


@dataclass(frozen=True)
class Interaction:
    hand: Hand
    device: Box
    confidence: float
    reasons: tuple[str, ...]

    @property
    def mask_box(self) -> Box:
        b = self.hand.box.union(self.device)
        return Box(b.x1, b.y1, b.x2, b.y2, label="PIN_ENTRY", score=self.confidence, source="pin")


@dataclass(frozen=True)
class _DeviceMemory:
    wrist: np.ndarray
    hand_size: float
    rel: tuple[float, float, float, float]
    label: str
    ttl: int


def at_head(device: Box, heads: list[Box], min_overlap: float = 0.4) -> bool:
    """A phone held to the ear sits on the head; a phone typed on at chin height only grazes it.
    A wrong 'call' verdict leaks a PIN, so the test is strict: the device centre lies on the head,
    or a large share of the device overlaps it."""
    return any(h.expand(0.3).contains(*device.center) or device.intersection(h) >= min_overlap * device.area
               for h in heads)


def touching_tips(hand: Hand, device: Box, margin: float = 0.12) -> list[int]:
    face = device.expand(margin)
    return [i for i in TOUCH_TIPS if face.contains(*hand.tip(i))]


class PinGuard:
    def __init__(self, cfg: PrivacyConfig, tap_window: int = 10) -> None:
        self.cfg = cfg
        self._memory: list[_DeviceMemory] = []
        self._tips: deque[np.ndarray] = deque(maxlen=tap_window)

    def reset(self) -> None:
        self._memory = []
        self._tips.clear()

    def score(self, hand: Hand, device: Box) -> tuple[float, tuple[str, ...]]:
        tips = touching_tips(hand, device)
        if not tips:
            return 0.0, ()
        reasons = ["fingertip_on_device"]
        conf = 0.6
        if INDEX_TIP in tips and hand.index_extended:
            conf += 0.2
            reasons.append("index_pointing")
        if THUMB_TIP in tips and hand.box.intersection(device) > 0.3 * min(hand.box.area, device.area):
            conf += 0.2
            reasons.append("thumb_on_held_phone")
        if self._is_tapping(hand):
            conf += 0.2
            reasons.append("tapping")
        return min(conf, 1.0), tuple(reasons)

    def _is_tapping(self, hand: Hand) -> bool:
        if len(self._tips) < self._tips.maxlen:
            return False
        motion = np.std(np.stack(self._tips), axis=0).max()
        return bool(motion > 0.02 * hand.size)

    def __call__(self, hands: list[Hand], devices: list[Box], heads: list[Box] = ()) -> list[Interaction]:
        """`heads` are face/head boxes; a device held against a head is a phone call, not PIN entry."""
        if hands:
            # fingertip relative to the wrist, so moving the whole hand is not read as tapping
            self._tips.append(hands[0].tip(INDEX_TIP) - hands[0].tip(WRIST))
        devices = [d for d in devices if not at_head(d, heads)]
        found: list[Interaction] = []
        ttls: list[int] = []
        used: set[int] = set()
        for hand in hands:
            best = max(((self.score(hand, d), d) for d in devices), key=lambda s: s[0][0], default=None)
            if best and best[0][0] > 0:
                (conf, why), dev = best
                found.append(Interaction(hand, dev, conf, why))
                ttls.append(self.cfg.hold_frames)
                continue
            mem = self._carried_device(hand, used)
            if mem is not None:
                used.add(id(mem))
                dev = self._project(mem, hand)
                if not at_head(dev, heads):
                    found.append(Interaction(hand, dev, 0.5, ("device_occluded_by_hand",)))
                    ttls.append(mem.ttl - 1)
        fresh = [self._remember(it, ttl) for it, ttl in zip(found, ttls)]
        # A memory not used this frame (e.g. the hand tracker missed the hand) keeps decaying instead
        # of being dropped, unless a fresh memory for the same hand replaced it.
        kept = [replace(m, ttl=m.ttl - 1) for m in self._memory
                if id(m) not in used and m.ttl - 1 >= 1
                and not any(np.linalg.norm(f.wrist - m.wrist) < 1.5 * m.hand_size for f in fresh)]
        self._memory = fresh + kept
        return found

    def _carried_device(self, hand: Hand, used: set[int]) -> _DeviceMemory | None:
        """`ttl` is how many more frames a remembered device may be carried with its hand."""
        w = hand.tip(WRIST)
        near = [m for m in self._memory if id(m) not in used and m.ttl >= 1
                and np.linalg.norm(m.wrist - w) < 1.5 * max(m.hand_size, hand.size)]
        return min(near, key=lambda m: float(np.linalg.norm(m.wrist - w)), default=None)

    def _project(self, mem: _DeviceMemory, hand: Hand) -> Box:
        (wx, wy), s = hand.tip(WRIST), hand.size
        rx1, ry1, rx2, ry2 = mem.rel
        return Box(wx + rx1 * s, wy + ry1 * s, wx + rx2 * s, wy + ry2 * s, label=mem.label, score=0.0,
                   source="device_memory")

    @staticmethod
    def _remember(it: Interaction, ttl: int) -> _DeviceMemory:
        (wx, wy), s, d = it.hand.tip(WRIST), it.hand.size, it.device
        rel = ((d.x1 - wx) / s, (d.y1 - wy) / s, (d.x2 - wx) / s, (d.y2 - wy) / s)
        return _DeviceMemory(it.hand.tip(WRIST).copy(), s, rel, d.label, ttl)
