import numpy as np
import pytest

from privacy_cctv.config import PrivacyConfig
from privacy_cctv.detectors.mp_models import Hand
from privacy_cctv.pin_guard import PinGuard
from privacy_cctv.pipeline import PrivacyPipeline
from privacy_cctv.regions import Box


def make_hand(tip_xy, wrist_xy=(100, 200)):
    """A synthetic 21-point hand: wrist at `wrist_xy`, index fingertip (8) at `tip_xy`, index extended."""
    pts = np.tile(np.array(wrist_xy, dtype=np.float32), (21, 1))
    tip = np.array(tip_xy, dtype=np.float32)
    w = pts[0]
    for i in range(1, 21):
        pts[i] = w + (tip - w) * 0.5 + np.array([i % 4, i % 3], dtype=np.float32)
    pts[6] = w + (tip - w) * 0.6
    pts[8] = tip
    return Hand(pts, "Right", 0.9)


PHONE = Box(80, 80, 140, 160, label="cell phone", score=0.7, source="devices")


def test_fingertip_on_device_is_flagged_as_pin_entry():
    found = PinGuard(PrivacyConfig())([make_hand((110, 120))], [PHONE])
    assert len(found) == 1
    assert "fingertip_on_device" in found[0].reasons
    assert found[0].mask_box.source == "pin"


def test_hand_away_from_device_is_not_flagged():
    assert PinGuard(PrivacyConfig())([make_hand((400, 400), wrist_xy=(420, 480))], [PHONE]) == []


def test_phone_held_against_the_head_is_a_call_not_pin_entry():
    head = Box(90, 60, 170, 150, label="FACE_MALE")
    assert PinGuard(PrivacyConfig())([make_hand((110, 120))], [PHONE], heads=[head]) == []


def test_device_hidden_by_hand_is_carried_with_the_hand():
    guard = PinGuard(PrivacyConfig(hold_frames=5))
    guard([make_hand((110, 120))], [PHONE])
    moved = make_hand((130, 120), wrist_xy=(120, 200))
    found = guard([moved], [])
    assert len(found) == 1
    assert found[0].device.source == "device_memory"
    assert found[0].device.x1 == pytest.approx(PHONE.x1 + 20, abs=1)


def test_reset_forgets_remembered_devices():
    guard = PinGuard(PrivacyConfig(hold_frames=5))
    guard([make_hand((110, 120))], [PHONE])
    guard.reset()
    assert guard([make_hand((110, 120))], []) == []


def test_carried_device_expires_after_hold():
    guard = PinGuard(PrivacyConfig(hold_frames=2))
    guard([make_hand((110, 120))], [PHONE])
    assert len(guard([make_hand((110, 120))], [])) == 1
    assert len(guard([make_hand((110, 120))], [])) == 1
    assert guard([make_hand((110, 120))], []) == []


def test_device_memory_survives_a_frame_where_the_hand_is_missed():
    guard = PinGuard(PrivacyConfig(hold_frames=5))
    guard([make_hand((110, 120))], [PHONE])
    assert guard([], []) == []
    found = guard([make_hand((110, 120))], [])
    assert len(found) == 1 and found[0].device.source == "device_memory"


def test_phone_typed_on_below_the_chin_is_not_treated_as_a_call():
    chin = Box(80, 20, 160, 85, label="FACE_FEMALE")  # grazes the top of the phone only
    assert len(PinGuard(PrivacyConfig())([make_hand((110, 120))], [PHONE], heads=[chin])) == 1


class _Boom:
    def __call__(self, *a, **k):
        raise RuntimeError("npu fault")


class _Empty:
    def __call__(self, *a, **k):
        return []

    def in_crop(self, *a, **k):
        return []


def _pipeline(**stubs):
    cfg = PrivacyConfig()
    parts = {"pose": _Empty(), "hands": _Empty(), "hands_roi": _Empty(), "devices": _Empty(),
             "devices_roi": _Empty(), "body_parts": _Empty()}
    parts.update(stubs)
    return PrivacyPipeline(cfg, **parts)


def test_pipeline_fails_closed_when_a_model_raises():
    frame = np.full((48, 64, 3), 255, dtype=np.uint8)
    r = _pipeline(hands=_Boom()).process(frame)
    assert r.fault and "npu fault" in r.fault
    assert frame[40:].max() == 0  # below the banner text the frame is black


def test_pipeline_masks_pin_entry_in_the_callers_buffer():
    frame = np.random.default_rng(2).integers(0, 255, (240, 320, 3), dtype=np.uint8)

    class _Hands(_Empty):
        def __call__(self, *a, **k):
            return [make_hand((110, 120))]

    class _Phone(_Empty):
        def __call__(self, *a, **k):
            return [PHONE]

    r = _pipeline(hands=_Hands(), devices=_Phone()).process(frame)
    assert r.fault is None and r.pin_active
    assert len(np.unique(frame[100:140, 90:130].reshape(-1, 3), axis=0)) == 1


def test_pipeline_keeps_masking_for_hold_frames_after_evidence_disappears():
    cfg = PrivacyConfig(hold_frames=3)
    calls = {"n": 0}

    class _Flicker(_Empty):
        def __call__(self, *a, **k):
            calls["n"] += 1
            return [make_hand((110, 120))] if calls["n"] == 1 else []

    class _Phone(_Empty):
        def __call__(self, *a, **k):
            return [PHONE]

    p = PrivacyPipeline(cfg, pose=_Empty(), hands=_Flicker(), hands_roi=_Empty(), devices=_Phone(),
                        devices_roi=_Empty(), body_parts=_Empty())
    states = [p.process(np.zeros((240, 320, 3), np.uint8)).pin_active for _ in range(6)]
    assert states == [True, True, True, True, False, False]


def test_pipeline_fails_closed_when_masking_itself_raises(monkeypatch):
    import privacy_cctv.pipeline as pl

    def broken(*a, **k):
        raise ValueError("bad box")

    class _Hands(_Empty):
        def __call__(self, *a, **k):
            return [make_hand((110, 120))]

    class _Phone(_Empty):
        def __call__(self, *a, **k):
            return [PHONE]

    monkeypatch.setattr(pl, "apply_masks", broken)
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    r = _pipeline(hands=_Hands(), devices=_Phone()).process(frame)
    assert r.fault and frame[60:].max() == 0
