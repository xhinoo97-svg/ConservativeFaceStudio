from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.opencv_zoo_face import OpenCVZooFaceEngine


@dataclass(frozen=True)
class FaceCropResult:
    image: np.ndarray
    source_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    detector_score: float
    detector_backend: str


class _FaceObservation(Protocol):
    bbox: tuple[int, int, int, int]
    score: float


class _FaceEngine(Protocol):
    def analyze(self, image: np.ndarray) -> _FaceObservation: ...


def _context_square(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    context_scale: float,
) -> tuple[int, int, int, int]:
    if width < 2 or height < 2:
        raise ValueError("image dimensions are too small")
    x, y, w, h = (int(value) for value in bbox)
    if w < 2 or h < 2:
        raise ValueError("face bbox is too small")
    if not 1.0 <= float(context_scale) <= 2.0:
        raise ValueError("context_scale must be in [1.0, 2.0]")

    x1_face = max(0, min(width - 1, x))
    y1_face = max(0, min(height - 1, y))
    x2_face = max(x1_face + 1, min(width, x + w))
    y2_face = max(y1_face + 1, min(height, y + h))
    face_w = x2_face - x1_face
    face_h = y2_face - y1_face
    side = int(round(max(face_w, face_h) * float(context_scale)))
    side = max(max(face_w, face_h), side)
    side = min(side, width, height)

    center_x = (x1_face + x2_face) * 0.5
    center_y = (y1_face + y2_face) * 0.5
    left = int(round(center_x - side * 0.5))
    top = int(round(center_y - side * 0.5))
    left = max(0, min(width - side, left))
    top = max(0, min(height - side, top))
    return left, top, side, side


def crop_main_face(
    image_bgr: np.ndarray,
    engine: _FaceEngine,
    *,
    output_size: int = 256,
    context_scale: float = 1.35,
    minimum_detector_score: float = 0.75,
) -> FaceCropResult:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image is empty")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8 HxWx3")
    if int(output_size) < 64:
        raise ValueError("output_size must be >= 64")
    if not 0.0 <= float(minimum_detector_score) <= 1.0:
        raise ValueError("minimum_detector_score must be in [0,1]")

    observation = engine.analyze(image_bgr)
    score = float(observation.score)
    if not np.isfinite(score) or score < float(minimum_detector_score):
        raise RuntimeError(
            f"YuNet face detection below required confidence: {score:.6f} < {minimum_detector_score:.6f}"
        )
    height, width = image_bgr.shape[:2]
    crop_bbox = _context_square(
        observation.bbox,
        width=int(width),
        height=int(height),
        context_scale=float(context_scale),
    )
    x, y, w, h = crop_bbox
    crop = image_bgr[y : y + h, x : x + w]
    if crop.size == 0 or crop.shape[0] != h or crop.shape[1] != w:
        raise RuntimeError("face crop geometry is invalid")
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) >= int(output_size) else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (int(output_size), int(output_size)), interpolation=interpolation)
    if resized.dtype != np.uint8 or resized.shape != (int(output_size), int(output_size), 3):
        raise RuntimeError("face crop resize produced an invalid image")
    return FaceCropResult(
        image=np.ascontiguousarray(resized),
        source_bbox=tuple(int(value) for value in observation.bbox),
        crop_bbox=tuple(int(value) for value in crop_bbox),
        detector_score=score,
        detector_backend="opencv_zoo_yunet",
    )


def build_yunet_cropper(
    yunet_path: str | Path,
    *,
    score_threshold: float = 0.75,
) -> OpenCVZooFaceEngine:
    return OpenCVZooFaceEngine(
        yunet_path=yunet_path,
        sface_path=None,
        score_threshold=float(score_threshold),
        dnn_target="cpu",
    )
