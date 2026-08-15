from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from app.restoration import detect_occlusion_candidates, detail_reliability_map


class RestorationCase(str, Enum):
    MULTI_REFERENCE = "multi_reference"
    SINGLE_IMAGE = "single_image"
    TRANSLUCENT_OCCLUSION = "translucent_occlusion"
    OPAQUE_OCCLUSION = "opaque_occlusion"
    STRONG_BLUR = "strong_blur"
    MIXED = "mixed"


@dataclass(frozen=True)
class CaseAssessment:
    route: RestorationCase
    reference_count: int
    occlusion_fraction: float
    reliable_fraction: float
    translucent_fraction: float
    strong_blur: bool
    notes: tuple[str, ...]


def _texture_retention(image: np.ndarray, mask: np.ndarray) -> float:
    """Estimate whether an apparent overlay still retains underlying structure.

    This is intentionally a routing hint, not an occlusion remover. Semi-transparent
    overlays often preserve gradients/texture inside the marked region; opaque marker
    or stickers usually destroy most local structure. The metric is cheap enough for
    the target CPU and is evaluated on the observed image only.
    """
    if image is None or image.size == 0 or not np.any(mask):
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    inside = grad[mask > 0]
    ring = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    ring = (ring > 0) & (mask == 0)
    outside = grad[ring]
    if inside.size < 32 or outside.size < 32:
        return 0.0
    denom = max(1.0, float(np.median(outside)))
    return float(np.clip(np.median(inside) / denom, 0.0, 1.5) / 1.5)


def assess_restoration_case(primary: np.ndarray, references: list[np.ndarray]) -> CaseAssessment:
    mask = detect_occlusion_candidates(primary)
    reliability = detail_reliability_map(primary, mask)
    occlusion_fraction = float(np.mean(mask > 0))
    reliable_fraction = float(np.mean(reliability >= 40))
    texture = _texture_retention(primary, mask)
    translucent_fraction = float(occlusion_fraction * texture)
    strong_blur = reliable_fraction < 0.22

    notes: list[str] = []
    if references:
        notes.append("real references available: component-bank route has priority")
    else:
        notes.append("no external references: single-image evidence only")
    if strong_blur:
        notes.append("observed detail reliability is low")
    if occlusion_fraction > 0.03:
        notes.append("occlusion candidate present")
    if texture >= 0.38 and occlusion_fraction > 0.01:
        notes.append("overlay retains measurable underlying structure")

    if references:
        route = RestorationCase.MULTI_REFERENCE
        if strong_blur and occlusion_fraction > 0.03:
            route = RestorationCase.MIXED
    elif occlusion_fraction > 0.03 and texture >= 0.38:
        route = RestorationCase.TRANSLUCENT_OCCLUSION
    elif occlusion_fraction > 0.03:
        route = RestorationCase.OPAQUE_OCCLUSION
    elif strong_blur:
        route = RestorationCase.STRONG_BLUR
    else:
        route = RestorationCase.SINGLE_IMAGE

    return CaseAssessment(
        route=route,
        reference_count=len(references),
        occlusion_fraction=occlusion_fraction,
        reliable_fraction=reliable_fraction,
        translucent_fraction=translucent_fraction,
        strong_blur=strong_blur,
        notes=tuple(notes),
    )
