"""Privacy-by-design CCTV: mask intimate body regions and PIN entry before a frame leaves the camera."""
from .config import PROFILES, PrivacyConfig
from .pipeline import FrameResult, PrivacyPipeline
from .regions import Box

__all__ = ["PROFILES", "Box", "FrameResult", "PrivacyConfig", "PrivacyPipeline"]
