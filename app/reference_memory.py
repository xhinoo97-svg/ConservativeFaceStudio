from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

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
        return 0.0
    x, y, w, h = bounds
    if w < 3 or h < 3:
        return 0.0
    gray_a = cv2.cvtColor(a[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    crop_mask = mask[y : y + h, x : x + w]
    scales = ((1.0, 0.20), (0.5, 0.30), (0.25, 0.50))
    total = 0.0
    weight_sum = 0.0
    for scale, weight in scales:
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
        return 0.0
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
) -> float:
    gain = (quality - primary_quality) / max(0.02, abs(primary_quality))
    quality_advantage = float(np.clip(0.5 + 0.5 * np.tanh(gain), 0.0, 1.0))
    return float(
        0.30 * similarity
        + 0.30 * agreement
        + 0.20 * visibility
        + 0.20 * quality_advantage
    )


def specific_reference_memory_fusion(
    images: list[np.ndarray],
    occlusion_masks: list[np.ndarray],
    landmarks5: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    top_k: int = 2,
    minimum_region_confidence: float = 0.64,
    minimum_quality_gain: float = 0.03,
    maximum_replace_fraction: float = 0.35,
    agreement_colour_threshold: float = 22.0,
) -> SpecificReferenceMemoryResult:
    """DMD-inspired specific memory using only observed pixels from same-identity references.

    The primary image is index 0 and references are indices 1..N in the returned
    provenance map. Multi-scale matching ranks all references for each facial
    component, while transfer is allowed only where selected references agree and
    are locally better. No generic face prior or synthesized texture is used.
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
    regions = facial_region_masks(shape[:2], landmarks5, bbox)
    output = base.copy()
    provenance = np.zeros(shape[:2], dtype=np.uint16)
    confidence_map = np.zeros(shape[:2], dtype=np.uint8)
    decisions: list[MemoryRegionDecision] = []

    quality_maps = [_pixel_quality(item) for item in images]
    median_reference = np.median(np.stack(refs, axis=0).astype(np.float32), axis=0).astype(np.uint8)

    for name, region_mask in regions.items():
        active = region_mask > 0
        area = int(np.count_nonzero(active))
        if area == 0:
            decisions.append(MemoryRegionDecision(name, 0.0, 0.0, (), 0, 0, ()))
            continue

        primary_visible = active & (primary_mask == 0)
        primary_quality = float(np.mean(quality_maps[0][primary_visible])) if np.any(primary_visible) else 0.0
        candidates: list[MemoryCandidate] = []

        for ref_index, (reference, ref_mask, ref_quality_map) in enumerate(
            zip(refs, ref_masks, quality_maps[1:]), start=1
        ):
            valid = active & (ref_mask == 0) & (np.max(reference, axis=2) > 2)
            visibility = float(np.count_nonzero(valid) / max(1, area))
            if np.count_nonzero(valid) < 24:
                continue
            quality = float(np.mean(ref_quality_map[valid]))
            compare_mask = valid & primary_visible
            similarity = (
                _multiscale_similarity(base, reference, compare_mask.astype(np.uint8) * 255)
                if np.count_nonzero(compare_mask) >= 24
                else 0.5
            )
            agreement = (
                _reference_agreement(reference, median_reference, region_mask, valid)
                if len(refs) > 1
                else similarity
            )
            score = _candidate_score(quality, primary_quality, similarity, agreement, visibility)
            candidates.append(MemoryCandidate(ref_index, quality, similarity, agreement, visibility, score))

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = [item for item in candidates if item.score >= minimum_region_confidence][:top_k]
        confidence = float(np.mean([item.score for item in selected])) if selected else 0.0
        transferred = 0

        if selected:
            selected_indices = [item.source_index for item in selected]
            source_images = [images[index] for index in selected_indices]
            source_masks = [masks[index] for index in selected_indices]
            source_qualities = [quality_maps[index] for index in selected_indices]

            eligible = active & (primary_mask == 0)
            valid_stack = np.stack(
                [(mask == 0) & (np.max(image, axis=2) > 2) for image, mask in zip(source_images, source_masks)],
                axis=0,
            )
            quality_stack = np.stack(source_qualities, axis=0).copy()
            quality_stack[~valid_stack] = -np.inf
            best_local_slot = np.argmax(quality_stack, axis=0)
            best_local_quality = np.max(quality_stack, axis=0)
            local_gain = best_local_quality - quality_maps[0]
            eligible &= np.isfinite(best_local_quality) & (local_gain >= minimum_quality_gain)

            if len(source_images) >= 2:
                first_lab = cv2.cvtColor(source_images[0], cv2.COLOR_BGR2LAB).astype(np.float32)
                second_lab = cv2.cvtColor(source_images[1], cv2.COLOR_BGR2LAB).astype(np.float32)
                pair_gap = np.mean(np.abs(first_lab - second_lab), axis=2)
                pair_valid = valid_stack[0] & valid_stack[1]
                eligible &= pair_valid & (pair_gap <= agreement_colour_threshold)
            else:
                only = selected[0]
                if only.similarity < 0.82 or only.agreement < 0.82:
                    eligible[:] = False

            region_cap = maximum_replace_fraction
            if name == "face":
                region_cap = min(region_cap, 0.08)
                if confidence < max(minimum_region_confidence, 0.76):
                    eligible[:] = False

            eligible_count = int(np.count_nonzero(eligible))
            cap = max(0, int(round(area * region_cap)))
            if eligible_count > cap > 0:
                coords = np.flatnonzero(eligible)
                gains = local_gain.ravel()[coords]
                keep = coords[np.argpartition(gains, -cap)[-cap:]]
                limited = np.zeros_like(eligible)
                limited.ravel()[keep] = True
                eligible = limited
            elif cap == 0:
                eligible[:] = False

            if np.any(eligible):
                rows, cols = np.indices(shape[:2])
                stack = np.stack(source_images, axis=0)
                chosen = stack[best_local_slot, rows, cols]
                output[eligible] = chosen[eligible]
                selected_source_array = np.asarray(selected_indices, dtype=np.uint16)
                provenance[eligible] = selected_source_array[best_local_slot[eligible]]
                confidence_map[eligible] = np.uint8(round(np.clip(confidence, 0.0, 1.0) * 255.0))
                transferred = int(np.count_nonzero(eligible))

        decisions.append(
            MemoryRegionDecision(
                name=name,
                confidence=confidence,
                primary_quality=primary_quality,
                selected_sources=tuple(item.source_index for item in selected),
                transferred_pixels=transferred,
                candidate_count=len(candidates),
                candidates=tuple(candidates),
            )
        )

    return SpecificReferenceMemoryResult(
        image=output,
        provenance_map=provenance,
        confidence_map=confidence_map,
        decisions=tuple(decisions),
        transferred_pixels=int(np.count_nonzero(provenance)),
    )
