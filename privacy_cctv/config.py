"""Pipeline settings and per-site profiles."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

EXPOSED_PARTS = (
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
)
MALE_CHEST = ("MALE_BREAST_EXPOSED",)
UNDERWEAR_PARTS = ("FEMALE_GENITALIA_COVERED", "BUTTOCKS_COVERED", "ANUS_COVERED")

# Which pose-derived region a body-part class is expected to fall in.
PART_REGION = {
    "FEMALE_BREAST_EXPOSED": "chest",
    "MALE_BREAST_EXPOSED": "chest",
    "BUTTOCKS_EXPOSED": "pelvis",
    "BUTTOCKS_COVERED": "pelvis",
    "FEMALE_GENITALIA_EXPOSED": "pelvis",
    "FEMALE_GENITALIA_COVERED": "pelvis",
    "MALE_GENITALIA_EXPOSED": "pelvis",
    "ANUS_EXPOSED": "pelvis",
    "ANUS_COVERED": "pelvis",
}


@dataclass(frozen=True)
class PrivacyConfig:
    body_privacy: bool = True
    pin_privacy: bool = True
    sensitive_parts: tuple[str, ...] = EXPOSED_PARTS + MALE_CHEST

    # Body-part detector: a detection anywhere needs `part_score_high`; one that lands in
    # the pose region its class belongs to only needs `part_score_low`.
    part_score_high: float = 0.45
    part_score_low: float = 0.25
    part_box_expand: float = 0.35

    max_people: int = 4
    max_hands: int = 4
    device_labels: tuple[str, ...] = ("cell phone", "remote")
    device_score: float = 0.25
    device_model: str = "efficientdet_lite0.tflite"

    # Mask immediately, release only after this many frames without evidence.
    hold_frames: int = 12
    pin_box_expand: float = 0.25
    # Crops per zoom pass per frame (0 disables the zoom passes).
    zoom_crops: int = 3
    # Strict policy for PIN-pad zones (banks, ATMs): mask every payment device, touched or not.
    mask_all_devices: bool = False

    mask_cells: int = 6
    video_mode: bool = False
    parallel: bool = True

    models_dir: Path = field(default=MODELS_DIR)

    def with_(self, **kw) -> PrivacyConfig:
        return replace(self, **kw)


PROFILES = {
    "hostel": PrivacyConfig(pin_privacy=False, sensitive_parts=EXPOSED_PARTS + MALE_CHEST + UNDERWEAR_PARTS),
    "retail": PrivacyConfig(sensitive_parts=EXPOSED_PARTS),
    "bank": PrivacyConfig(sensitive_parts=EXPOSED_PARTS, hold_frames=18, mask_all_devices=True),
    "all": PrivacyConfig(),
}
