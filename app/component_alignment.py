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


def _gradient_similarity(a: np.ndarray, b: np.ndarray, active: np.ndarray) -> float:
    """Brightness-insensitive local structural similarity used only as an alignment gate."""
    if np.count_nonzero(active) < 48:
        return 0.0
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    ag = cv2.magnitude(ax, ay)[active].astype(np.float32)
    bg = cv2.magnitude(bx, by)[active].astype(np.float32)
    if ag.size < 48 or bg.size != ag.size:
        return 0.0
    ag -= float(np.mean(ag))
    bg -= float(np.mean(bg))
    denom = float(np.linalg.norm(ag) * np.linalg.norm(bg))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(ag, bg) / denom, -1.0, 1.0))


def refine_component_translation(
    aligned_reference: np.ndarray,
    primary: np.ndarray,
    support_mask: np.ndarray,
    component_mask: np.ndarray,
    *,
    maximum_shift: float = 5.0,
    minimum_response: float = 0.08,
    minimum_similarity_gain: float = 0.015,
) -> ComponentAlignmentResult:
    """Strict sub-pixel translation refinement for one facial component.

    The global affine alignment remains authoritative. This function can only correct
    a small residual translation inside an already aligned component. It cannot scale,
    shear, mirror or reshape facial anatomy. A phase-correlation proposal is accepted
    only if it produces a measurable improvement in local gradient similarity; this
    prevents blur or interpolation noise from moving an already aligned donor.
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

    if np.count_nonzero(local_mask) < 48:
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, 0.0, False)

    # Keep an uncentered copy for the structural before/after gate.
    raw_ref = ref_gray.copy()
    raw_pri = pri_gray.copy()

    # Remove local brightness offset so exposure differences do not drive phase correlation.
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
        return ComponentAlignmentResult(
            aligned_reference.copy(), support_mask.copy(), 0.0, 0.0,
            response if np.isfinite(response) else 0.0, False
        )

    # Do not resample for a numerically negligible shift.
    if abs(dx) < 0.15 and abs(dy) < 0.15:
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response, False)

    # Verify the proposal before touching the full-resolution donor. The test uses
    # gradient structure so exposure differences and moderate blur have little effect.
    roi_matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted_roi = cv2.warpAffine(
        raw_ref,
        roi_matrix,
        (raw_ref.shape[1], raw_ref.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shifted_mask = cv2.warpAffine(
        local_mask.astype(np.uint8) * 255,
        roi_matrix,
        (raw_ref.shape[1], raw_ref.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    comparison = local_mask & shifted_mask
    if np.count_nonzero(comparison) < 48:
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response, False)

    before_similarity = _gradient_similarity(raw_ref, raw_pri, comparison)
    after_similarity = _gradient_similarity(shifted_roi, raw_pri, comparison)
    if after_similarity - before_similarity < float(minimum_similarity_gain):
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response, False)

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
