from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.component_alignment import refine_component_translation
from app.regional_fusion import facial_region_masks


@dataclass(frozen=True)
class MemoryCandidate:
    source_index: int
    quality: float
    similarity: float
    agreement: float
    visibility: float
    score: float


@dataclass(frozen=True)
class MemoryRegionDecision:
    name: str
    confidence: float
    primary_quality: float
    selected_sources: tuple[int, ...]
    transferred_pixels: int
    candidate_count: int
    candidates: tuple[MemoryCandidate, ...]


@dataclass(frozen=True)
class SpecificReferenceMemoryResult:
    image: np.ndarray
    provenance_map: np.ndarray
    confidence_map: np.ndarray
    decisions: tuple[MemoryRegionDecision, ...]
    transferred_pixels: int


def _validate_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Sono supportate immagini BGR a 3 canali")
    return image


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != shape:
        raise ValueError("Maschera non compatibile con l'immagine")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _pixel_quality(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(_validate_image(image), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    sharpness = cv2.GaussianBlur(lap, (0, 0), 1.4)
    exposure = 1.0 - np.clip(np.abs(gray - 0.5) / 0.5, 0.0, 1.0)
    return sharpness + 0.025 * exposure


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(np.where(mask > 0, 255, 0).astype(np.uint8))
    if points is None:
        return None
    return cv2.boundingRect(points)


def _normalized_correlation(a: np.ndarray, b: np.ndarray, active: np.ndarray) -> float:
    values_a = a[active].astype(np.float32)
    values_b = b[active].astype(np.float32)
    if values_a.size < 24 or values_b.size != values_a.size:
        return 0.5
    values_a -= float(np.mean(values_a))
    values_b -= float(np.mean(values_b))
    denom = float(np.linalg.norm(values_a) * np.linalg.norm(values_b))
    if denom <= 1e-8:
        mean_gap = abs(float(np.mean(a[active])) - float(np.mean(b[active]))) / 255.0
        return float(np.clip(1.0 - mean_gap, 0.0, 1.0))
    corr = float(np.dot(values_a, values_b) / denom)
    return float(np.clip((corr + 1.0) * 0.5, 0.0, 1.0))


def _multiscale_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    bounds = _bounds(mask)
    if bounds is None:
        return 0.5
    x, y, w, h = bounds
    if w < 3 or h < 3:
        return 0.5
    gray_a = cv2.cvtColor(a[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    crop_mask = mask[y:y + h, x:x + w]
    total = 0.0
    weight_sum = 0.0
    for scale, weight in ((1.0, 0.20), (0.5, 0.30), (0.25, 0.50)):
        if scale != 1.0:
            sw = max(3, int(round(w * scale)))
            sh = max(3, int(round(h * scale)))
            aa = cv2.resize(gray_a, (sw, sh), interpolation=cv2.INTER_AREA)
            bb = cv2.resize(gray_b, (sw, sh), interpolation=cv2.INTER_AREA)
            mm = cv2.resize(crop_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            aa, bb, mm = gray_a, gray_b, crop_mask
        active = mm > 0
        if np.count_nonzero(active) < 24:
            continue
        total += weight * _normalized_correlation(aa, bb, active)
        weight_sum += weight
    return float(total / weight_sum) if weight_sum else 0.5


def _reference_agreement(
    reference: np.ndarray,
    median_reference: np.ndarray,
    region_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    active = (region_mask > 0) & valid_mask
    if np.count_nonzero(active) < 24:
        return 0.5
    structure = _multiscale_similarity(reference, median_reference, active.astype(np.uint8) * 255)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    med_lab = cv2.cvtColor(median_reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    colour_gap = float(np.mean(np.abs(ref_lab[active] - med_lab[active]))) / 255.0
    colour = float(np.clip(1.0 - colour_gap / 0.16, 0.0, 1.0))
    return float(0.75 * structure + 0.25 * colour)


def _candidate_score(
    quality: float,
    primary_quality: float,
    similarity: float,
    agreement: float,
    visibility: float,
    *,
    damaged_visibility: float = 0.0,
) -> float:
    gain = (quality - primary_quality) / max(0.02, abs(primary_quality))
    quality_advantage = float(np.clip(0.5 + 0.5 * np.tanh(gain), 0.0, 1.0))
    if damaged_visibility > 0.0:
        # A donor that uniquely covers damaged pixels must not be rejected merely
        # because its crop occupies a small fraction of the full facial region.
        return float(
            0.16 * similarity
            + 0.22 * agreement
            + 0.08 * visibility
            + 0.44 * damaged_visibility
            + 0.10 * quality_advantage
        )
    return float(0.30 * similarity + 0.30 * agreement + 0.20 * visibility + 0.20 * quality_advantage)


def _median_reference_with_support(refs: list[np.ndarray], support_masks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(refs, axis=0).astype(np.float32)
    support = np.stack([mask > 0 for mask in support_masks], axis=0)
    counts = np.sum(support, axis=0).astype(np.intp)
    observed_any = counts > 0
    result = np.zeros(stack.shape[1:], dtype=np.float32)
    if not np.any(observed_any):
        return result.astype(np.uint8)
    lower_index = np.maximum((counts - 1) // 2, 0)
    upper_index = np.maximum(counts // 2, 0)
    for channel in range(3):
        values = np.where(support, stack[..., channel], np.inf)
        values.sort(axis=0)
        lower = np.take_along_axis(values, lower_index[None, ...], axis=0)[0]
        upper = np.take_along_axis(values, upper_index[None, ...], axis=0)[0]
        median = 0.5 * (lower + upper)
        result[..., channel][observed_any] = median[observed_any]
    return np.clip(result, 0, 255).astype(np.uint8)


def _shift_mask(mask: np.ndarray, dx: float, dy: float, *, border_value: int) -> np.ndarray:
    h, w = mask.shape
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _agreement_mask(source_images: list[np.ndarray], valid_stack: np.ndarray, threshold: float) -> np.ndarray:
    accepted = np.sum(valid_stack, axis=0) > 0
    if len(source_images) < 2:
        return accepted
    labs = [cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) for image in source_images]
    for first in range(len(source_images) - 1):
        for second in range(first + 1, len(source_images)):
            overlap = valid_stack[first] & valid_stack[second]
            if not np.any(overlap):
                continue
            gap = np.mean(np.abs(labs[first] - labs[second]), axis=2)
            accepted[overlap] &= gap[overlap] <= threshold
    return accepted


def _limit_by_gain(mask: np.ndarray, gain: np.ndarray, cap: int) -> np.ndarray:
    count = int(np.count_nonzero(mask))
    if count == 0 or cap <= 0:
        return np.zeros_like(mask)
    if count <= cap:
        return mask
    coords = np.flatnonzero(mask)
    gains = gain.ravel()[coords]
    keep = coords[np.argpartition(gains, -cap)[-cap:]]
    limited = np.zeros_like(mask)
    limited.ravel()[keep] = True
    return limited


def specific_reference_memory_fusion(
    images: list[np.ndarray],
    occlusion_masks: list[np.ndarray],
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    reference_support_masks: list[np.ndarray] | None = None,
    top_k: int = 2,
    minimum_region_confidence: float = 0.64,
    minimum_quality_gain: float = 0.03,
    maximum_replace_fraction: float = 0.35,
    agreement_colour_threshold: float = 22.0,
    local_refinement_max_shift: float = 4.0,
    local_refinement_min_response: float = 0.10,
) -> SpecificReferenceMemoryResult:
    """Fuse observed reference evidence while protecting intact primary pixels.

    Repair and enhancement deliberately use different acceptance policies:
    damaged pixels are filled from any sufficiently credible *observed* donor,
    while intact primary pixels keep the stricter quality/confidence gate.
    """
    if len(images) < 2:
        raise ValueError("Serve almeno una fotografia di riferimento")
    base = _validate_image(images[0])
    shape = base.shape
    if any(_validate_image(item).shape != shape for item in images):
        raise ValueError("Le immagini devono essere allineate e avere la stessa forma")
    if len(occlusion_masks) != len(images):
        raise ValueError("Numero immagini/maschere non compatibile")
    masks = [_validate_mask(item, shape[:2]) for item in occlusion_masks]
    if top_k < 1:
        raise ValueError("top_k deve essere almeno 1")

    refs = images[1:]
    ref_masks = masks[1:]
    primary_mask = masks[0]
    if reference_support_masks is None:
        support_masks = [
            np.where(np.max(reference, axis=2) > 2, 255, 0).astype(np.uint8)
            for reference in refs
        ]
    else:
        if len(reference_support_masks) != len(refs):
            raise ValueError("Numero reference/support mask non compatibile")
        support_masks = [_validate_mask(item, shape[:2]) for item in reference_support_masks]

    regions = facial_region_masks(shape[:2], landmarks5, bbox)
    output = base.copy()
    provenance = np.zeros(shape[:2], dtype=np.uint16)
    confidence_map = np.zeros(shape[:2], dtype=np.uint8)
    decisions: list[MemoryRegionDecision] = []
    primary_quality_map = _pixel_quality(base)

    for name, region_mask in regions.items():
        active = region_mask > 0
        area = int(np.count_nonzero(active))
        if area == 0:
            decisions.append(MemoryRegionDecision(name, 0.0, 0.0, (), 0, 0, ()))
            continue

        repair_target = active & (primary_mask > 0)
        primary_visible = active & (primary_mask == 0)
        repair_area = int(np.count_nonzero(repair_target))
        primary_quality = float(np.mean(primary_quality_map[primary_visible])) if np.any(primary_visible) else 0.0

        regional_refs: list[np.ndarray] = []
        regional_masks: list[np.ndarray] = []
        regional_support: list[np.ndarray] = []
        regional_quality: list[np.ndarray] = []

        for reference, ref_mask, support_mask in zip(refs, ref_masks, support_masks):
            # Component refinement must use only pixels that are actually visible in
            # both the primary and this donor. It must never align against a sticker.
            visible_support = np.where(
                (support_mask > 0) & (ref_mask == 0) & (primary_mask == 0),
                255,
                0,
            ).astype(np.uint8)
            visible_coverage = float(np.count_nonzero(active & (visible_support > 0)) / max(1, area))
            refined = None
            if visible_coverage >= 0.18:
                refined = refine_component_translation(
                    reference,
                    base,
                    visible_support,
                    region_mask,
                    maximum_shift=float(local_refinement_max_shift),
                    minimum_response=float(local_refinement_min_response),
                )
            if refined is not None and refined.accepted:
                regional_refs.append(refined.image)
                regional_support.append(_shift_mask(support_mask, refined.dx, refined.dy, border_value=0))
                regional_masks.append(_shift_mask(ref_mask, refined.dx, refined.dy, border_value=255))
                regional_quality.append(_pixel_quality(refined.image))
            else:
                regional_refs.append(reference)
                regional_support.append(support_mask)
                regional_masks.append(ref_mask)
                regional_quality.append(_pixel_quality(reference))

        median_reference = _median_reference_with_support(regional_refs, regional_support)
        candidates: list[MemoryCandidate] = []
        damaged_visibility_by_source: dict[int, float] = {}
        damage_pixels_by_source: dict[int, int] = {}

        for ref_index, (reference, ref_mask, ref_quality_map, support_mask) in enumerate(
            zip(regional_refs, regional_masks, regional_quality, regional_support),
            start=1,
        ):
            valid = active & (ref_mask == 0) & (support_mask > 0)
            valid_count = int(np.count_nonzero(valid))
            if valid_count < 8:
                continue
            visibility = float(valid_count / max(1, area))
            damage_pixels = int(np.count_nonzero(valid & repair_target))
            damaged_visibility = float(damage_pixels / max(1, repair_area)) if repair_area else 0.0
            damaged_visibility_by_source[ref_index] = damaged_visibility
            damage_pixels_by_source[ref_index] = damage_pixels
            quality = float(np.mean(ref_quality_map[valid]))
            compare_mask = valid & primary_visible
            similarity = (
                _multiscale_similarity(base, reference, compare_mask.astype(np.uint8) * 255)
                if np.count_nonzero(compare_mask) >= 24
                else 0.5
            )
            agreement = (
                _reference_agreement(reference, median_reference, region_mask, valid)
                if len(regional_refs) > 1
                else similarity
            )
            score = _candidate_score(
                quality,
                primary_quality,
                similarity,
                agreement,
                visibility,
                damaged_visibility=damaged_visibility,
            )
            candidates.append(MemoryCandidate(ref_index, quality, similarity, agreement, visibility, score))

        candidates.sort(key=lambda item: item.score, reverse=True)

        # Repair donors are evaluated by evidence *inside the missing area*. A tiny
        # eye/nose/mouth crop can therefore be accepted even though its full-region
        # visibility is low. Enhancement donors remain intentionally conservative.
        repair_threshold = max(0.42, float(minimum_region_confidence) - 0.20)
        repair_selected: list[MemoryCandidate] = []
        if repair_area:
            for item in candidates:
                if damage_pixels_by_source.get(item.source_index, 0) < 8:
                    continue
                has_visible_identity_anchor = item.similarity >= 0.40
                has_reference_consistency = item.agreement >= 0.38 or len(regional_refs) == 1
                if item.score >= repair_threshold and has_visible_identity_anchor and has_reference_consistency:
                    repair_selected.append(item)
            repair_selected = repair_selected[: min(5, len(repair_selected))]

        enhancement_selected = [
            item for item in candidates
            if item.score >= minimum_region_confidence
            and item.similarity >= 0.86
            and item.agreement >= 0.82
        ][: max(1, top_k)]

        selected_by_source: dict[int, MemoryCandidate] = {}
        for item in repair_selected + enhancement_selected:
            selected_by_source[item.source_index] = item
        selected = sorted(selected_by_source.values(), key=lambda item: item.score, reverse=True)
        confidence = float(np.mean([item.score for item in selected])) if selected else 0.0
        transferred = 0

        if selected:
            selected_indices = [item.source_index for item in selected]
            source_images = [regional_refs[index - 1] for index in selected_indices]
            source_masks = [regional_masks[index - 1] for index in selected_indices]
            source_qualities = [regional_quality[index - 1] for index in selected_indices]
            source_support = [regional_support[index - 1] for index in selected_indices]
            valid_stack = np.stack(
                [(mask == 0) & (support > 0) for mask, support in zip(source_masks, source_support)],
                axis=0,
            )

            # Rank each valid donor per pixel using local quality plus regional
            # confidence. This lets complementary donors form a true union rather
            # than forcing one global winner for the whole component.
            quality_stack = np.stack(source_qualities, axis=0).astype(np.float32)
            score_bias = np.asarray([item.score for item in selected], dtype=np.float32)[:, None, None]
            ranked_stack = quality_stack + 0.035 * score_bias
            ranked_stack[~valid_stack] = -np.inf
            best_local_slot = np.argmax(ranked_stack, axis=0)
            best_local_quality = np.max(quality_stack * valid_stack, axis=0)
            finite = np.any(valid_stack, axis=0)
            local_gain = best_local_quality - primary_quality_map

            repair_source_slots = {
                selected_indices.index(item.source_index)
                for item in repair_selected
                if item.source_index in selected_indices
            }
            if repair_source_slots:
                repair_valid_stack = np.zeros_like(valid_stack)
                for slot in repair_source_slots:
                    repair_valid_stack[slot] = valid_stack[slot]
                repair_ranked = ranked_stack.copy()
                repair_ranked[~repair_valid_stack] = -np.inf
                repair_best_slot = np.argmax(repair_ranked, axis=0)
                repair_finite = np.any(repair_valid_stack, axis=0)
            else:
                repair_best_slot = best_local_slot
                repair_finite = np.zeros(shape[:2], dtype=bool)

            # Do not apply pairwise colour-agreement rejection to damaged pixels:
            # exposure differences between valid references are expected and the
            # donor is already identity/support gated. Agreement remains a guard for
            # changing intact primary pixels.
            repair_eligible = repair_target & repair_finite

            enhancement_slots = {
                selected_indices.index(item.source_index)
                for item in enhancement_selected
                if item.source_index in selected_indices
            }
            if enhancement_slots:
                enhancement_valid_stack = np.zeros_like(valid_stack)
                for slot in enhancement_slots:
                    enhancement_valid_stack[slot] = valid_stack[slot]
                agreement_ok = _agreement_mask(source_images, enhancement_valid_stack, agreement_colour_threshold)
                enhancement_eligible = (
                    primary_visible
                    & finite
                    & agreement_ok
                    & (local_gain >= minimum_quality_gain)
                    & (confidence >= max(minimum_region_confidence, 0.78))
                )
            else:
                enhancement_eligible = np.zeros(shape[:2], dtype=bool)

            visible_cap_fraction = min(maximum_replace_fraction, 0.06)
            if name == "face":
                visible_cap_fraction = min(visible_cap_fraction, 0.02)
            enhancement_eligible = _limit_by_gain(
                enhancement_eligible,
                local_gain,
                max(0, int(round(area * visible_cap_fraction))),
            )

            rows, cols = np.indices(shape[:2])
            stack = np.stack(source_images, axis=0)
            selected_source_array = np.asarray(selected_indices, dtype=np.uint16)
            selected_score_array = np.asarray([item.score for item in selected], dtype=np.float32)

            if np.any(repair_eligible):
                repair_chosen = stack[repair_best_slot, rows, cols]
                output[repair_eligible] = repair_chosen[repair_eligible]
                provenance[repair_eligible] = selected_source_array[repair_best_slot[repair_eligible]]
                repair_conf = selected_score_array[repair_best_slot]
                confidence_map[repair_eligible] = np.clip(
                    np.rint(repair_conf[repair_eligible] * 255.0),
                    0,
                    255,
                ).astype(np.uint8)

            if np.any(enhancement_eligible):
                enhancement_chosen = stack[best_local_slot, rows, cols]
                output[enhancement_eligible] = enhancement_chosen[enhancement_eligible]
                provenance[enhancement_eligible] = selected_source_array[best_local_slot[enhancement_eligible]]
                enhancement_conf = selected_score_array[best_local_slot]
                confidence_map[enhancement_eligible] = np.clip(
                    np.rint(enhancement_conf[enhancement_eligible] * 255.0),
                    0,
                    255,
                ).astype(np.uint8)

            transferred = int(np.count_nonzero(repair_eligible | enhancement_eligible))

        decisions.append(
            MemoryRegionDecision(
                name,
                confidence,
                primary_quality,
                tuple(item.source_index for item in selected),
                transferred,
                len(candidates),
                tuple(candidates),
            )
        )

    return SpecificReferenceMemoryResult(
        image=output,
        provenance_map=provenance,
        confidence_map=confidence_map,
        decisions=tuple(decisions),
        transferred_pixels=int(np.count_nonzero(provenance)),
    )
