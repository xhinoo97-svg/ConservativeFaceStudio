from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.strict_repair import repair_from_observed_references


@dataclass(frozen=True)
class VerifiedReferenceRepairResult:
    image: np.ndarray
    target_mask: np.ndarray
    repaired_mask: np.ndarray
    unresolved_mask: np.ndarray
    provenance_map: np.ndarray
    requested_pixels: int
    repaired_pixels: int
    unresolved_pixels: int
    source_pixel_counts: tuple[int, ...]
    local_shifts: tuple[tuple[int, int], ...]
    context_scores: tuple[float, ...]
    agreement_rejected_pixels: int


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != shape:
        raise ValueError("Maschera non compatibile con la fotografia")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _context_ring(mask: np.ndarray, radius: int = 9) -> np.ndarray:
    radius = max(3, int(radius))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask, kernel)
    return ((dilated > 0) & (mask == 0)).astype(np.uint8) * 255


def _overlap_slices(height: int, width: int, dx: int, dy: int):
    dst_x1 = max(0, dx)
    dst_x2 = min(width, width + dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(height, height + dy)
    src_x1 = max(0, -dx)
    src_x2 = min(width, width - dx)
    src_y1 = max(0, -dy)
    src_y2 = min(height, height - dy)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return None
    return (
        slice(dst_y1, dst_y2),
        slice(dst_x1, dst_x2),
        slice(src_y1, src_y2),
        slice(src_x1, src_x2),
    )


def _best_context_translation(
    primary: np.ndarray,
    reference: np.ndarray,
    target_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    max_shift: int = 5,
    minimum_context_pixels: int = 96,
) -> tuple[int, int, float]:
    """Find a tiny translation from the visible ring around the damaged area.

    Registration must tolerate the normal exposure/white-balance differences between
    photographs of the same person. Raw LAB distance alone rejects otherwise useful
    references, so matching uses median-centred LAB structure plus Sobel magnitude.
    This changes registration only: transferred pixels still come from real references.
    """
    shape = primary.shape[:2]
    ring = _context_ring(target_mask) > 0
    base_lab = cv2.cvtColor(primary, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    base_gray = cv2.cvtColor(primary, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base_gx = cv2.Sobel(base_gray, cv2.CV_32F, 1, 0, ksize=3)
    base_gy = cv2.Sobel(base_gray, cv2.CV_32F, 0, 1, ksize=3)
    ref_gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    base_grad = cv2.magnitude(base_gx, base_gy)
    ref_grad = cv2.magnitude(ref_gx, ref_gy)
    ref_valid = reference_mask == 0
    h, w = shape

    best = (0, 0, float("inf"), 0)
    limit = max(0, int(max_shift))
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            slices = _overlap_slices(h, w, dx, dy)
            if slices is None:
                continue
            dy_s, dx_s, sy_s, sx_s = slices
            active = ring[dy_s, dx_s] & ref_valid[sy_s, sx_s]
            count = int(np.count_nonzero(active))
            if count < minimum_context_pixels:
                continue

            left = base_lab[dy_s, dx_s][active]
            right = ref_lab[sy_s, sx_s][active]
            left_centered = left - np.median(left, axis=0, keepdims=True)
            right_centered = right - np.median(right, axis=0, keepdims=True)
            delta = np.abs(left_centered - right_centered)
            colour_cost = float(np.median(delta[:, 0]) + 0.20 * np.median(delta[:, 1:]))

            left_grad = base_grad[dy_s, dx_s][active]
            right_grad = ref_grad[sy_s, sx_s][active]
            grad_scale = max(16.0, float(np.median(left_grad) + np.median(right_grad)))
            gradient_cost = float(np.median(np.abs(left_grad - right_grad)) / grad_scale * 32.0)

            cost = 0.72 * colour_cost + 0.28 * gradient_cost
            if cost < best[2]:
                best = (dx, dy, cost, count)

    if not np.isfinite(best[2]):
        return 0, 0, 0.0
    score = float(np.clip(1.0 - best[2] / 48.0, 0.0, 1.0))
    return int(best[0]), int(best[1]), score


def _shift_reference(
    image: np.ndarray,
    mask: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    if dx == 0 and dy == 0:
        return image.copy(), mask.copy()
    h, w = image.shape[:2]
    matrix = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    shifted = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    shifted_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return shifted, shifted_mask


def _agreement_mask(
    references: list[np.ndarray],
    masks: list[np.ndarray],
    target: np.ndarray,
    *,
    threshold: float = 24.0,
) -> np.ndarray:
    """Return target pixels supported by at least two structurally agreeing references.

    Different photographs of the same person routinely have different exposure and
    white balance. Comparing raw LAB values made valid references disagree. We remove
    one robust per-reference LAB offset over the requested region before comparing
    structure, while also checking gradient magnitude. A conflicting eye/mouth shape
    still produces a large structural/gradient residual and is rejected.
    """
    if len(references) < 2:
        return target.copy()

    valid = np.stack(
        [(mask == 0) & (np.max(item, axis=2) > 2) for item, mask in zip(references, masks)],
        axis=0,
    )
    target_bool = target > 0
    labs = np.stack(
        [cv2.cvtColor(item, cv2.COLOR_BGR2LAB).astype(np.float32) for item in references],
        axis=0,
    )
    gradients: list[np.ndarray] = []
    centred_labs: list[np.ndarray] = []

    for index, item in enumerate(references):
        active = valid[index] & target_bool
        lab = labs[index].copy()
        if int(np.count_nonzero(active)) >= 24:
            offset = np.median(lab[active], axis=0)
        else:
            broader = valid[index]
            offset = np.median(lab[broader], axis=0) if np.any(broader) else np.zeros(3, np.float32)
        centred_labs.append(lab - offset.reshape(1, 1, 3))

        gray = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(gx, gy))

    values = np.stack(centred_labs, axis=0)
    grad_values = np.stack(gradients, axis=0)
    values[~valid[..., None].repeat(3, axis=3)] = np.nan
    grad_values[~valid] = np.nan

    with np.errstate(invalid="ignore"):
        median_lab = np.nanmedian(values, axis=0)
        lab_delta = np.mean(np.abs(values - median_lab[None, ...]), axis=3)
        lab_disagreement = np.nanmedian(lab_delta, axis=0)

        median_grad = np.nanmedian(grad_values, axis=0)
        grad_delta = np.abs(grad_values - median_grad[None, ...])
        grad_disagreement = np.nanmedian(grad_delta, axis=0)
        grad_scale = np.maximum(16.0, np.nanmedian(grad_values, axis=0))
        normalized_grad = grad_disagreement / grad_scale * 18.0

    combined = 0.78 * lab_disagreement + 0.22 * normalized_grad
    valid_count = np.sum(valid, axis=0)
    accepted = (
        target_bool
        & (valid_count >= 2)
        & np.isfinite(combined)
        & (combined <= float(threshold))
    )
    return accepted.astype(np.uint8) * 255


def verified_reference_repair(
    primary: np.ndarray,
    references: list[np.ndarray],
    target_mask: np.ndarray,
    reference_masks: list[np.ndarray] | None = None,
    *,
    identity_scores: list[float | None] | None = None,
    identity_threshold: float = 0.363,
    identity_verification_available: bool = False,
    max_local_shift: int = 5,
    minimum_context_score: float = 0.42,
    agreement_threshold: float = 24.0,
    feather_sigma: float = 1.0,
) -> VerifiedReferenceRepairResult:
    """Reference-guided inpainting that never invents pixels.

    Principles deliberately mirror the face-inpainting literature:
    * identity-level filtering before transfer;
    * alignment immediately around the missing region;
    * multi-reference agreement before accepting a source;
    * exact source provenance for every transferred pixel.
    """
    if primary is None or primary.size == 0 or primary.ndim != 3 or primary.shape[2] != 3:
        raise ValueError("Immagine principale non valida")
    if not references:
        raise ValueError("Serve almeno una fotografia reale di riferimento")
    shape = primary.shape[:2]
    target = _binary(target_mask, shape)
    masks = reference_masks or [np.zeros(shape, dtype=np.uint8) for _ in references]
    if len(masks) != len(references):
        raise ValueError("Numero di maschere riferimento non valido")
    masks = [_binary(mask, shape) for mask in masks]

    accepted_refs: list[np.ndarray] = []
    accepted_masks: list[np.ndarray] = []
    original_indices: list[int] = []
    local_shifts: list[tuple[int, int]] = []
    context_scores: list[float] = []

    for index, (reference, mask) in enumerate(zip(references, masks)):
        if reference.shape != primary.shape:
            raise ValueError("Le fotografie devono essere allineate alla stessa dimensione")
        if identity_verification_available:
            score = None if identity_scores is None or index >= len(identity_scores) else identity_scores[index]
            if score is None or float(score) < float(identity_threshold):
                continue
        dx, dy, context = _best_context_translation(
            primary,
            reference,
            target,
            mask,
            max_shift=max_local_shift,
        )
        if context < minimum_context_score and np.count_nonzero(target) > 0:
            continue
        shifted, shifted_mask = _shift_reference(reference, mask, dx, dy)
        accepted_refs.append(shifted)
        accepted_masks.append(shifted_mask)
        original_indices.append(index)
        local_shifts.append((dx, dy))
        context_scores.append(context)

    requested = int(np.count_nonzero(target))
    if not accepted_refs or requested == 0:
        empty = np.zeros(shape, dtype=np.uint16)
        return VerifiedReferenceRepairResult(
            primary.copy(),
            target,
            np.zeros(shape, dtype=np.uint8),
            target.copy(),
            empty,
            requested,
            0,
            requested,
            tuple(0 for _ in references),
            tuple(local_shifts),
            tuple(context_scores),
            0,
        )

    agreed_target = _agreement_mask(
        accepted_refs,
        accepted_masks,
        target,
        threshold=agreement_threshold,
    )
    agreement_rejected = int(np.count_nonzero((target > 0) & (agreed_target == 0)))
    repaired = repair_from_observed_references(
        primary,
        accepted_refs,
        agreed_target,
        accepted_masks,
        feather_sigma=feather_sigma,
    )

    provenance = np.zeros(shape, dtype=np.uint16)
    source_counts = [0 for _ in references]
    for local_index, original_index in enumerate(original_indices, start=1):
        active = repaired.provenance_map == local_index
        provenance[active] = np.uint16(original_index + 1)
        source_counts[original_index] = int(np.count_nonzero(active))

    repaired_mask = (provenance > 0).astype(np.uint8) * 255
    unresolved = ((target > 0) & (repaired_mask == 0)).astype(np.uint8) * 255
    repaired_pixels = int(np.count_nonzero(repaired_mask))
    return VerifiedReferenceRepairResult(
        repaired.image,
        target,
        repaired_mask,
        unresolved,
        provenance,
        requested,
        repaired_pixels,
        requested - repaired_pixels,
        tuple(source_counts),
        tuple(local_shifts),
        tuple(context_scores),
        agreement_rejected,
    )
