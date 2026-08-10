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
    minimum_shift_magnitude: float = 0.75,
    preserve_observed_pixels: bool = True,
) -> ComponentAlignmentResult:
    """Strict local translation refinement for one facial component.

    Phase correlation estimates a residual displacement after global alignment.  In the
    conservative default, the estimate is quantized to the nearest integer translation
    and the donor is warped with nearest-neighbour sampling.  This keeps transferred
    colour values equal to pixels that were actually photographed; sub-pixel Lanczos or
    bilinear interpolation would create new values and make exact provenance false.

    The global affine alignment remains authoritative. This function cannot scale,
    shear, mirror or reshape facial anatomy, and a proposal is accepted only when the
    applied translation measurably improves local gradient similarity.  Callers may set
    ``preserve_observed_pixels=False`` only for a non-strict preview path; production
    reference transfer keeps the default enabled.
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

    raw_ref = ref_gray.copy()
    raw_pri = pri_gray.copy()

    ref_gray = ref_gray - float(np.median(ref_gray[local_mask]))
    pri_gray = pri_gray - float(np.median(pri_gray[local_mask]))
    ref_gray[~local_mask] = 0.0
    pri_gray[~local_mask] = 0.0

    window = cv2.createHanningWindow((ref_gray.shape[1], ref_gray.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_gray, pri_gray, window)
    estimated_dx, estimated_dy = float(shift[0]), float(shift[1])
    response = float(response)
    if (
        not np.isfinite([estimated_dx, estimated_dy, response]).all()
        or response < minimum_response
        or abs(estimated_dx) > maximum_shift
        or abs(estimated_dy) > maximum_shift
    ):
        return ComponentAlignmentResult(
            aligned_reference.copy(), support_mask.copy(), 0.0, 0.0,
            response if np.isfinite(response) else 0.0, False
        )

    estimated_magnitude = float(np.hypot(estimated_dx, estimated_dy))
    if estimated_magnitude < max(0.0, float(minimum_shift_magnitude)):
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response, False)

    if preserve_observed_pixels:
        dx = float(np.rint(estimated_dx))
        dy = float(np.rint(estimated_dy))
        # A non-zero sub-pixel estimate can round to the identity transform. In strict
        # mode that is an abstention, not a reason to resample the donor.
        if dx == 0.0 and dy == 0.0:
            return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response, False)
        interpolation = cv2.INTER_NEAREST
    else:
        dx, dy = estimated_dx, estimated_dy
        interpolation = cv2.INTER_LINEAR

    if abs(dx) > maximum_shift or abs(dy) > maximum_shift:
        return ComponentAlignmentResult(aligned_reference.copy(), support_mask.copy(), 0.0, 0.0, response, False)

    roi_matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    shifted_roi = cv2.warpAffine(
        raw_ref,
        roi_matrix,
        (raw_ref.shape[1], raw_ref.shape[0]),
        flags=interpolation,
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
        flags=interpolation,
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
