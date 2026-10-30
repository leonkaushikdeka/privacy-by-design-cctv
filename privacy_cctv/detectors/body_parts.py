"""Pre-trained body-part detector (NudeNet v3, a YOLOv8n trained on 18 body-part classes).

The upstream package decodes the 2100 anchors in a Python loop; this wrapper runs the same
ONNX graph and decodes with numpy so it can sit inside a real-time loop.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..regions import Box

LABELS = (
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEET_EXPOSED",
    "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE",
    "BELLY_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)


def default_model_path() -> Path:
    import nudenet

    return Path(nudenet.__file__).with_name("320n.onnx")


class BodyPartDetector:
    def __init__(self, model: Path | None = None, input_size: int = 320, min_score: float = 0.2) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self._sess = ort.InferenceSession(str(model or default_model_path()), opts,
                                          providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self.size = input_size
        self.min_score = min_score

    def __call__(self, image_bgr: np.ndarray, offset: tuple[float, float] = (0.0, 0.0)) -> list[Box]:
        return self.batch([(image_bgr, offset)])

    def batch(self, crops: list[tuple[np.ndarray, tuple[float, float]]]) -> list[Box]:
        """Run several images (whole frame + person crops) through the network in one call."""
        blobs, metas = [], []
        for img, offset in crops:
            h, w = img.shape[:2]
            side = max(h, w)
            padded = cv2.copyMakeBorder(img, 0, side - h, 0, side - w, cv2.BORDER_CONSTANT)
            blobs.append(cv2.dnn.blobFromImage(padded, 1 / 255.0, (self.size, self.size), swapRB=True))
            metas.append((side / self.size, offset, w, h))
        out = self._sess.run(None, {self._input: np.concatenate(blobs)})[0]
        boxes: list[Box] = []
        for k, (scale, offset, w, h) in enumerate(metas):
            boxes += self._decode(out[k].T, scale, offset, w, h)
        return boxes

    def _decode(self, rows: np.ndarray, scale: float, offset: tuple[float, float], w: int, h: int) -> list[Box]:
        # Multi-label: every (anchor, class) pair above the floor is a candidate, and NMS runs per
        # class. With argmax + class-agnostic NMS (as upstream does) an exposed part overlapping a
        # higher-scoring covered/belly box would be thrown away.
        anchor, cls = np.nonzero(rows[:, 4:] >= self.min_score)
        if not len(anchor):
            return []
        score = rows[anchor, 4 + cls]
        cx, cy, bw, bh = (rows[anchor, i] * scale for i in range(4))
        x1 = np.clip(cx - bw / 2, 0, w)
        y1 = np.clip(cy - bh / 2, 0, h)
        x2 = np.clip(cx + bw / 2, 0, w)
        y2 = np.clip(cy + bh / 2, 0, h)
        xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist()
        idx = cv2.dnn.NMSBoxesBatched(xywh, score.tolist(), cls.tolist(), self.min_score, 0.45)
        ox, oy = offset
        return [
            Box(float(x1[i] + ox), float(y1[i] + oy), float(x2[i] + ox), float(y2[i] + oy),
                label=LABELS[int(cls[i])], score=float(score[i]), source="body_parts")
            for i in np.array(idx).reshape(-1)
        ]
