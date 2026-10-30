import numpy as np
import pytest

from privacy_cctv.masking import apply_masks, mask_bitmap, pixelate
from privacy_cctv.memory import RegionMemory
from privacy_cctv.metrics import latency_summary, leak_rate, score_image
from privacy_cctv.regions import Box


def test_iou_of_identical_boxes_is_one_and_disjoint_is_zero():
    a = Box(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(Box(20, 20, 30, 30)) == 0.0


def test_expand_keeps_centre_and_scales_size():
    b = Box(10, 10, 20, 30).expand(0.5)
    assert b.center == pytest.approx((15, 20))
    assert (b.w, b.h) == pytest.approx((15, 30))


def test_clip_keeps_box_inside_frame():
    b = Box(-5, -5, 700, 500).clip(640, 480)
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, 640, 480)


def test_pin_regions_are_filled_solid_and_carry_no_original_pixels():
    frame = np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    apply_masks(frame, [Box(10, 10, 50, 50, source="pin")])
    region = frame[10:50, 10:50].reshape(-1, 3)
    assert len(np.unique(region, axis=0)) == 1


def test_pixelate_reduces_region_to_a_few_cells():
    frame = np.random.default_rng(1).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    pixelate(frame, Box(0, 0, 120, 120), cells=6)
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) <= 36


def test_masking_is_in_place_and_leaves_rest_of_frame_untouched():
    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    before = frame.copy()
    out = apply_masks(frame, [Box(0, 0, 10, 10, source="pin")])
    assert out is frame
    assert (frame[20:, 20:] == before[20:, 20:]).all()
    assert (frame[:10, :10] != 200).all()


def test_memory_holds_a_region_for_hold_frames_after_last_detection():
    mem = RegionMemory(hold_frames=3)
    box = Box(0, 0, 10, 10)
    assert len(mem.update([box])) == 1
    assert [len(mem.update([])) for _ in range(4)] == [1, 1, 1, 0]


def test_zero_hold_releases_on_the_first_empty_frame():
    mem = RegionMemory(hold_frames=0)
    mem.update([Box(0, 0, 10, 10)])
    assert mem.update([]) == []


def test_memory_refreshes_a_moving_region_instead_of_duplicating_it():
    mem = RegionMemory(hold_frames=5)
    mem.update([Box(0, 0, 10, 10)])
    active = mem.update([Box(2, 0, 12, 10)])
    assert len(active) == 1 and active[0].x1 == 2


def test_score_image_full_cover_and_no_over_mask():
    gt = [Box(10, 10, 30, 30)]
    s = score_image("x", (100, 100, 3), gt, gt)
    assert s.protected == 1 and s.mean_coverage == pytest.approx(1.0) and s.over_mask_frac == 0.0


def test_score_image_counts_masks_on_clean_images_as_false_alarms_except_in_ignore_areas():
    shape = (100, 100, 3)
    assert score_image("x", shape, [Box(0, 0, 10, 10)], []).false_alarm
    assert not score_image("x", shape, [Box(0, 0, 10, 10)], [], ignore=[Box(0, 0, 10, 10)]).false_alarm


def test_mask_bitmap_area_matches_box():
    assert mask_bitmap((50, 50), [Box(0, 0, 10, 5)]).sum() == 50


def test_hold_removes_leaks_from_isolated_detector_misses():
    detected = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    assert leak_rate(detected, hold_frames=0) == pytest.approx(0.4)
    assert leak_rate(detected, hold_frames=3) == 0.0


def test_latency_summary_reports_fps():
    assert latency_summary([10.0, 10.0])["fps"] == pytest.approx(100.0)
