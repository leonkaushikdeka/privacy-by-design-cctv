"""MediaPipe Tasks wrappers: pose landmarks, hand landmarks and a COCO object detector."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

from ..regions import Box

# MediaPipe pose landmark indices
NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 7, 8, 11, 12, 15, 16, 23, 24
# MediaPipe hand landmark indices
WRIST, THUMB_TIP, INDEX_PIP, INDEX_TIP, MIDDLE_TIP = 0, 4, 6, 8, 12


def to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _mode(video: bool) -> vision.RunningMode:
    return vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE


@dataclass(frozen=True)
class Person:
    keypoints: np.ndarray  # (33, 3): x px, y px, visibility
    box: Box

    def _pt(self, i: int) -> np.ndarray:
        return self.keypoints[i, :2]

    @property
    def mid_shoulder(self) -> np.ndarray:
        return (self._pt(L_SHOULDER) + self._pt(R_SHOULDER)) / 2

    @property
    def mid_hip(self) -> np.ndarray:
        return (self._pt(L_HIP) + self._pt(R_HIP)) / 2

    @property
    def torso_len(self) -> float:
        return float(max(np.linalg.norm(self.mid_shoulder - self.mid_hip), 1.0))

    def regions(self) -> dict[str, Box]:
        """Chest and pelvis areas derived from the skeleton, where intimate parts can be."""
        t = self.torso_len
        sw = float(np.linalg.norm(self._pt(L_SHOULDER) - self._pt(R_SHOULDER)))
        hw = float(np.linalg.norm(self._pt(L_HIP) - self._pt(R_HIP)))
        cx, cy = self.mid_shoulder
        chest_w = max(sw * 1.15, 0.55 * t)
        chest = Box(cx - chest_w / 2, cy - 0.05 * t, cx + chest_w / 2, cy + 0.5 * t, label="chest", source="pose")
        hx, hy = self.mid_hip
        pelvis_w = max(hw * 2.2, 0.6 * t)
        pelvis = Box(hx - pelvis_w / 2, hy - 0.3 * t, hx + pelvis_w / 2, hy + 0.5 * t, label="pelvis", source="pose")
        return {"chest": chest, "pelvis": pelvis}

    def roi(self) -> Box:
        """Crop that holds chest to upper thighs, used to run the part detector at higher resolution."""
        r = self.regions()
        return r["chest"].union(r["pelvis"]).expand(0.35)

    @property
    def plausible(self) -> bool:
        """Upright human proportions. The pose model sometimes builds a 'person' out of two hands;
        those skeletons have shoulders squeezed together or a nose below the shoulder line."""
        t = self.torso_len
        shoulder_w = float(np.linalg.norm(self._pt(L_SHOULDER) - self._pt(R_SHOULDER)))
        return shoulder_w >= 0.25 * t and float(self.mid_shoulder[1] - self._pt(NOSE)[1]) >= 0.1 * t

    def head(self) -> Box | None:
        if not self.plausible:
            return None
        pts = self.keypoints[[NOSE, L_EAR, R_EAR]]
        pts = pts[pts[:, 2] > 0.3]
        if not len(pts):
            return None
        cx, cy = pts[:, :2].mean(0)
        ear = float(np.linalg.norm(self._pt(L_EAR) - self._pt(R_EAR)))
        half = max(0.8 * ear, 0.18 * self.torso_len)
        return Box(cx - half, cy - half * 1.2, cx + half, cy + half, label="head", source="pose")

    def wrists(self, min_visibility: float = 0.5) -> list[np.ndarray]:
        return [self.keypoints[i, :2] for i in (L_WRIST, R_WRIST) if self.keypoints[i, 2] >= min_visibility]


@dataclass(frozen=True)
class Hand:
    landmarks: np.ndarray  # (21, 2) px
    handedness: str
    score: float

    @property
    def box(self) -> Box:
        return Box.from_points(self.landmarks, label="hand", score=self.score, source="hands")

    @property
    def size(self) -> float:
        return max(self.box.w, self.box.h, 1.0)

    def tip(self, i: int) -> np.ndarray:
        return self.landmarks[i]

    @property
    def index_extended(self) -> bool:
        w = self.landmarks[WRIST]
        return float(np.linalg.norm(self.landmarks[INDEX_TIP] - w)) > 1.05 * float(
            np.linalg.norm(self.landmarks[INDEX_PIP] - w)
        )


class _Timestamped:
    def __init__(self, video: bool) -> None:
        self.video = video
        self._last_ts = -1

    def _ts(self, timestamp_ms: int | None) -> int:
        ts = self._last_ts + 33 if timestamp_ms is None else int(timestamp_ms)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts
        return ts


class PoseEstimator(_Timestamped):
    def __init__(self, model: Path, max_people: int = 4, video: bool = False) -> None:
        super().__init__(video)
        opts = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)),
            running_mode=_mode(video),
            num_poses=max_people,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, image: mp.Image, timestamp_ms: int | None = None) -> list[Person]:
        h, w = image.height, image.width
        res = self._lm.detect_for_video(image, self._ts(timestamp_ms)) if self.video else self._lm.detect(image)
        people = []
        for lms in res.pose_landmarks:
            kp = np.array([[p.x * w, p.y * h, p.visibility or 0.0] for p in lms], dtype=np.float32)
            visible = kp[kp[:, 2] > 0.3]
            if len(visible) < 5:
                continue
            people.append(Person(kp, Box.from_points(visible, label="person", source="pose")))
        return people

    def close(self) -> None:
        self._lm.close()


class HandTracker(_Timestamped):
    def __init__(self, model: Path, max_hands: int = 4, video: bool = False) -> None:
        super().__init__(video)
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)),
            running_mode=_mode(video),
            num_hands=max_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, image: mp.Image, timestamp_ms: int | None = None) -> list[Hand]:
        h, w = image.height, image.width
        res = self._lm.detect_for_video(image, self._ts(timestamp_ms)) if self.video else self._lm.detect(image)
        hands = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
            hands.append(Hand(pts, hd[0].category_name, float(hd[0].score)))
        return hands

    def in_crop(self, frame_bgr: np.ndarray, box: Box) -> list[Hand]:
        """Run on a crop (IMAGE mode only) and return landmarks in full-frame pixels."""
        x1, y1, x2, y2 = box.clip(frame_bgr.shape[1], frame_bgr.shape[0]).as_int()
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        hands = self(to_mp_image(frame_bgr[y1:y2, x1:x2]))
        offset = np.array([x1, y1], dtype=np.float32)
        return [Hand(hd.landmarks + offset, hd.handedness, hd.score) for hd in hands]

    def close(self) -> None:
        self._lm.close()


class DeviceDetector(_Timestamped):
    """COCO EfficientDet-Lite restricted to handheld payment devices (phones, POS-like handsets)."""

    def __init__(self, model: Path, labels: tuple[str, ...], score: float, video: bool = False) -> None:
        super().__init__(video)
        opts = vision.ObjectDetectorOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model)),
            running_mode=_mode(video),
            score_threshold=score,
            max_results=10,
            category_allowlist=list(labels),
        )
        self._det = vision.ObjectDetector.create_from_options(opts)

    def __call__(self, image: mp.Image, timestamp_ms: int | None = None) -> list[Box]:
        res = self._det.detect_for_video(image, self._ts(timestamp_ms)) if self.video else self._det.detect(image)
        out = []
        for d in res.detections:
            b = d.bounding_box
            c = d.categories[0]
            out.append(Box.from_xywh(b.origin_x, b.origin_y, b.width, b.height,
                                     label=c.category_name, score=float(c.score), source="devices"))
        return out

    def in_crop(self, frame_bgr: np.ndarray, box: Box) -> list[Box]:
        """Run on a crop (IMAGE mode only) and return boxes in full-frame pixels."""
        x1, y1, x2, y2 = box.clip(frame_bgr.shape[1], frame_bgr.shape[0]).as_int()
        if x2 - x1 < 32 or y2 - y1 < 32:
            return []
        return [b.shift(x1, y1) for b in self(to_mp_image(frame_bgr[y1:y2, x1:x2]))]

    def close(self) -> None:
        self._det.close()
