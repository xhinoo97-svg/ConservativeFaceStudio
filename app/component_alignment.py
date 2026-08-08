from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ComponentAlignmentResult:
    image: np.ndarray
    support_mask: np.ndarray
    dx: float
    dy: float
    response: float
    accepted: bool


def _roi_bounds(mask: np.ndarray, margin: int = 8) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(np.where(mask > 0, 255, 0).astype(np.uint8))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    height, width = mask.shape
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def refine_component_translation(
    aligned_reference: np.ndarray,
    primary: np.ndarray,
    support_mask: np.ndarray,
    component_mask: np.ndarray,
    *,
    maximum_shift: float = 5.0,
    minimum_response: float = 0.08,
) -> ComponentAlignmentResult:
    """Strict sub-pixel translation refinement for one facial component.

    The global affine alignment remains authoritative. This function can only correct
    a small residual translation inside an already aligned component. It cannot scale,
    shear, mirror or reshape facial anatomy. If local evidence is weak it abstains.
    """
    if aligned_reference.shape != primary.shape:
        raise ValueError("Reference e primary devono avere la stessa forma")
    shape = primary.shape[:2]
    if support_mask.shape != shape or component_mask.shape != shape:
        raise ValueError("Maschere non compatibili")

    active = ((support_mask > 0) & (component_mask > 0)).astype(np.uint8) * 255
    bounds = _roi_bounds(active)
    if bounds is None or int(np.count_nonzero(active)) < 48:
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, 0.0, False)

    x1, y1, x2, y2 = bounds
    ref_gray = cv2.cvtColor(aligned_reference[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    pri_gray = cv2.cvtColor(primary[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mask = active[y1:y2, x1:x2] > 0

    # Remove local brightness offset so exposure differences do not drive the shift.
    if np.count_nonzero(local_mask) < 48:
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, 0.0, False)
    ref_gray = ref_gray - float(np.median(ref_gray[local_mask]))
    pri_gray = pri_gray - float(np.median(pri_gray[local_mask]))
    ref_gray[~local_mask] = 0.0
    pri_gray[~local_mask] = 0.0

    window = cv2.createHanningWindow((ref_gray.shape[1], ref_gray.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_gray, pri_gray, window)
    dx, dy = float(shift[0]), float(shift[1])
    response = float(response)
    if (
        not np.isfinite([dx, dy, response]).all()
        or response < minimum_response
        or abs(dx) > maximum_shift
        or abs(dy) > maximum_shift
    ):
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response if np.isfinite(response) else 0.0, False)

    height, width = shape
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    refined = cv2.warpAffine(
        aligned_reference,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    refined_support = cv2.warpAffine(
        support_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return ComponentAlignmentResult(refined, refined_support, dx, dy, response, True)
