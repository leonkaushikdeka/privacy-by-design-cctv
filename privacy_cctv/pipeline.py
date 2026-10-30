"""The per-frame pipeline that would run on the camera's NPU.

The frame is masked in place, so the caller's buffer never holds unmasked pixels after
`process` returns. If any stage raises, the whole frame is blanked (fail closed): a frame
the system could not check is a frame it must not release.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .body_privacy import BodyPrivacy
from .config import PrivacyConfig
from .detectors import BodyPartDetector, DeviceDetector, HandTracker, PoseEstimator, to_mp_image
from .masking import apply_masks
from .memory import RegionMemory
from .pin_guard import Interaction, PinGuard, at_head
from .regions import Box

SMALL = 0.2  # a target is "small" (worth a zoom pass) below this fraction of frame height
FACE_SCORE = 0.4  # only confident faces may mark a device as "at the head" (a call)


class _Done:
    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


@dataclass(frozen=True)
class FrameResult:
    frame: np.ndarray
    regions: tuple[Box, ...]
    timings_ms: dict[str, float]
    fault: str | None = None
    people: tuple = ()
    hands: tuple = ()
    devices: tuple[Box, ...] = ()
    parts: tuple[Box, ...] = ()
    interactions: tuple[Interaction, ...] = ()

    @property
    def pin_active(self) -> bool:
        return any(r.source == "pin" for r in self.regions)


@dataclass
class _Clock:
    t: dict[str, float] = field(default_factory=dict)

    def __call__(self, name: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return out


class PrivacyPipeline:
    def __init__(self, cfg: PrivacyConfig | None = None, *, pose=None, hands=None, hands_roi=None, devices=None,
                 devices_roi=None, body_parts=None) -> None:
        self.cfg = cfg = cfg or PrivacyConfig()
        m = cfg.models_dir
        if cfg.body_privacy:
            self.pose = pose or PoseEstimator(m / "pose_landmarker_lite.task", cfg.max_people, cfg.video_mode)
            self.body = BodyPrivacy(body_parts or BodyPartDetector(), cfg)
        if cfg.pin_privacy:
            self.hands = hands or HandTracker(m / "hand_landmarker.task", cfg.max_hands, cfg.video_mode)
            self.hands_roi = hands_roi or HandTracker(m / "hand_landmarker.task", 2, video=False)
            self.devices = devices or DeviceDetector(m / cfg.device_model, cfg.device_labels, cfg.device_score,
                                                     cfg.video_mode)
            self.devices_roi = devices_roi or DeviceDetector(m / cfg.device_model, cfg.device_labels,
                                                             cfg.device_score, video=False)
            self.pin = PinGuard(cfg)
        self._body_mem = RegionMemory(cfg.hold_frames)
        self._pin_mem = RegionMemory(cfg.hold_frames)
        self._pool = ThreadPoolExecutor(max_workers=3) if cfg.parallel else None
        self._pending: list = []
        self.frames = 0
        self.faults = 0

    def process(self, frame: np.ndarray, timestamp_ms: int | None = None) -> FrameResult:
        clock = _Clock()
        t0 = time.perf_counter()
        self.frames += 1
        try:
            found = self._detect(frame, timestamp_ms, clock)
            regions = clock("fuse", self._fuse, found["body"], found["pin"])
            clock("mask", apply_masks, frame, regions, self.cfg.mask_cells)
        except Exception as e:  # noqa: BLE001 - any failure must blank the frame
            self.faults += 1
            blank(frame)  # if even this raises, the exception propagates and nothing is released
            clock.t["total"] = (time.perf_counter() - t0) * 1000
            return FrameResult(frame, (), clock.t, fault=f"{type(e).__name__}: {e}")
        clock.t["total"] = (time.perf_counter() - t0) * 1000
        return FrameResult(frame, tuple(regions), clock.t, None, tuple(found["people"]), tuple(found["hands"]),
                           tuple(found["devices"]), tuple(found["parts"]), tuple(found["interactions"]))

    def _run(self, fn, *args):
        if self._pool is None:
            return _Done(fn(*args))
        future = self._pool.submit(fn, *args)
        self._pending.append(future)
        return future

    def _body_branch(self, frame: np.ndarray, image, ts: int | None, clock: _Clock) -> tuple[list, list]:
        people = clock("pose", self.pose, image, ts)
        return people, clock("body_parts", self.body.detect_parts, frame, people)

    def _detect(self, frame: np.ndarray, ts: int | None, clock: _Clock) -> dict:
        # Every submitted branch is joined before returning, even on failure, so no worker is still
        # reading the frame while it is blanked or still inside a model when the next frame arrives.
        self._pending = []
        try:
            return self._detect_all(frame, ts, clock)
        finally:
            wait(self._pending)

    def _detect_all(self, frame: np.ndarray, ts: int | None, clock: _Clock) -> dict:
        image = clock("prep", to_mp_image, frame)
        out = {"people": [], "hands": [], "devices": [], "parts": [], "interactions": [], "body": [], "pin": []}
        # The body branch (pose -> parts), the device detector and the hand tracker are independent
        # on the same frame, so they run concurrently, as separate engines would on a camera SoC.
        body = self._run(self._body_branch, frame, image, ts, clock) if self.cfg.body_privacy else None
        if self.cfg.pin_privacy:
            devices = self._run(clock, "devices", self.devices, image, ts)
            hands = self._run(clock, "hands", self.hands, image, ts)
        if body is not None:
            out["people"], out["parts"] = body.result()
        if self.cfg.pin_privacy:
            people = out["people"]
            out["devices"], out["hands"] = devices.result(), hands.result()
            out["hands"] += clock("hands_roi", self._hands_near_devices, frame, out["devices"], out["hands"])
            out["hands"] += clock("hands_roi", self._hands_at_wrists, frame, people, out["hands"])
            out["devices"] += clock("devices_roi", self._devices_in_hands, frame, out["hands"], out["devices"])
            heads = [h for h in (p.head() for p in people) if h] + [
                b for b in out["parts"] if b.label.startswith("FACE") and b.score >= FACE_SCORE]
            out["interactions"] = clock("fuse", self.pin, out["hands"], out["devices"], heads)
            out["pin"] = [it.mask_box.expand(self.cfg.pin_box_expand) for it in out["interactions"]]
            if self.cfg.mask_all_devices:
                out["pin"] += [replace(d, label="DEVICE", source="pin").expand(self.cfg.pin_box_expand)
                               for d in out["devices"] if not at_head(d, heads)]
        if self.cfg.body_privacy:
            out["body"] = clock("fuse", self.body.fuse, out["parts"], out["people"], out["hands"])
        return out

    # Small people in a wide CCTV frame have hands and phones of a few dozen pixels. Three
    # zoom passes look again at full model resolution, each guided by what was already found:
    #   device -> crop for the hand,  pose wrist -> crop for the hand,  hand -> crop for the device.
    # Only small targets are zoomed: a large hand or phone is already resolved in the full frame.

    def _hands_near_devices(self, frame: np.ndarray, devices: list[Box], hands: list) -> list:
        found = []
        small = [d for d in devices if max(d.w, d.h) < SMALL * frame.shape[0]]
        for d in self._budget(sorted(small, key=lambda b: -b.score)):
            zone = d.expand(1.0)
            if any(h.box.intersection(zone) > 0 for h in hands + found):
                continue
            crop = _square(d.center, max(d.w, d.h) * 3.0)
            found += [h for h in self.hands_roi.in_crop(frame, crop)
                      if h.box.intersection(zone) > 0 and _is_new(h, hands + found)]
        return found

    def _hands_at_wrists(self, frame: np.ndarray, people: list, hands: list) -> list:
        found = []
        wrists = [(w, p.torso_len) for p in people
                  if p.plausible and p.torso_len < 2 * SMALL * frame.shape[0] for w in p.wrists()]
        for w, torso in self._budget(wrists):
            if any(h.box.expand(0.5).contains(*w) for h in hands + found):
                continue
            crop = _square(w, 0.9 * torso)
            found += [h for h in self.hands_roi.in_crop(frame, crop) if _is_new(h, hands + found)]
        return found

    def _devices_in_hands(self, frame: np.ndarray, hands: list, devices: list[Box]) -> list[Box]:
        found = []
        small = [h for h in hands if h.size < SMALL * frame.shape[0]]
        for h in self._budget(small):
            if any(d.expand(0.5).intersection(h.box) > 0 for d in devices + found):
                continue
            crop = _square(h.box.center, 2.5 * h.size)
            found += [d for d in self.devices_roi.in_crop(frame, crop) if d.intersection(h.box.expand(0.5)) > 0]
        return found

    def _budget(self, candidates: list) -> list:
        """At most `zoom_crops` crops per pass; on video the start rotates each frame so every
        candidate is revisited within a few frames (the temporal hold covers the gap)."""
        n = self.cfg.zoom_crops
        if n <= 0 or not candidates:
            return []
        k = self.frames % len(candidates) if self.cfg.video_mode else 0
        return (candidates[k:] + candidates[:k])[:n]

    def _fuse(self, body: list[Box], pin: list[Box]) -> list[Box]:
        return self._body_mem.update(body) + self._pin_mem.update(pin)

    def reset(self) -> None:
        """Forget all temporal state (between unrelated images or scenes)."""
        self._body_mem.reset()
        self._pin_mem.reset()
        if self.cfg.body_privacy:
            self.body.reset()
        if self.cfg.pin_privacy:
            self.pin.reset()

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)
        for name in ("pose", "hands", "hands_roi", "devices", "devices_roi"):
            obj = getattr(self, name, None)
            if obj is not None and hasattr(obj, "close"):
                obj.close()


def _is_new(hand, existing: list) -> bool:
    """A zoom crop often re-finds a hand the full-frame pass already has."""
    w = hand.tip(0)
    return not any(hand.box.iou(o.box) > 0.3 or np.linalg.norm(o.tip(0) - w) < 0.5 * o.size for o in existing)


def _square(center, side: float) -> Box:
    cx, cy = center
    return Box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def blank(frame: np.ndarray) -> None:
    frame[:] = 0
    cv2.putText(frame, "PRIVACY FAULT - FRAME WITHHELD", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA)
