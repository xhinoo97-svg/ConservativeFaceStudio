from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

import app.reference_memory_legacy as _legacy
from app.reference_limits import MAX_REFERENCE_IMAGES, validate_reference_count
from app.reference_memory_legacy import (
    MemoryCandidate,
    MemoryRegionDecision,
    SpecificReferenceMemoryResult,
)

# Keep the helper surface compatible with the original module. The implementation
# lives in reference_memory_legacy so the public fusion entry point can add the
# ten-image product contract without duplicating every low-level primitive.
_validate_image = _legacy._validate_image
_validate_mask = _legacy._validate_mask
_pixel_quality = _legacy._pixel_quality
_bounds = _legacy._bounds
_normalized_correlation = _legacy._normalized_correlation
_multiscale_similarity = _legacy._multiscale_similarity
_reference_agreement = _legacy._reference_agreement
_candidate_score = _legacy._candidate_score
_median_reference_with_support = _legacy._median_reference_with_support
_shift_mask = _legacy._shift_mask
_agreement_mask = _legacy._agreement_mask
_limit_by_gain = _legacy._limit_by_gain


def _default_support_masks(refs: list[np.ndarray]) -> list[np.ndarray]:
    return [
        np.where(np.max(reference, axis=2) > 2, 255, 0).astype(np.uint8)
        for reference in refs
    ]


def _global_intact_conflict_mask(
    refs: list[np.ndarray],
    ref_masks: list[np.ndarray],
    support_masks: list[np.ndarray],
    primary_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return intact primary pixels where credible observed donors disagree.

    Damage repair is deliberately excluded: disagreement must not suppress a donor
    that uniquely covers a missing region. On intact pixels, however, any strong
    pairwise colour conflict is a reason to abstain and preserve the primary.
    """
    shape = primary_mask.shape
    disputed = np.zeros(shape, dtype=bool)
    if len(refs) < 2:
        return disputed
    labs = [cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) for reference in refs]
    valid = [
        (mask == 0) & (support > 0) & (primary_mask == 0)
        for mask, support in zip(ref_masks, support_masks)
    ]
    for first in range(len(refs) - 1):
        for second in range(first + 1, len(refs)):
            overlap = valid[first] & valid[second]
            if np.count_nonzero(overlap) < 8:
                continue
            gap = np.mean(np.abs(labs[first] - labs[second]), axis=2)
            disputed |= overlap & (gap > float(threshold))
    return disputed


def _map_decision(decision: MemoryRegionDecision, global_indices: list[int]) -> MemoryRegionDecision:
    def mapped_source(local_index: int) -> int:
        if local_index <= 0 or local_index > len(global_indices):
            return 0
        return int(global_indices[local_index - 1])

    candidates = tuple(
        MemoryCandidate(
            mapped_source(candidate.source_index),
            candidate.quality,
            candidate.similarity,
            candidate.agreement,
            candidate.visibility,
            candidate.score,
        )
        for candidate in decision.candidates
    )
    selected = tuple(
        source
        for source in (mapped_source(index) for index in decision.selected_sources)
        if source > 0
    )
    return MemoryRegionDecision(
        decision.name,
        decision.confidence,
        decision.primary_quality,
        selected,
        decision.transferred_pixels,
        decision.candidate_count,
        candidates,
    )


def _aggregate_decisions(decisions: list[MemoryRegionDecision]) -> tuple[MemoryRegionDecision, ...]:
    grouped: dict[str, list[MemoryRegionDecision]] = defaultdict(list)
    order: list[str] = []
    for decision in decisions:
        if decision.name not in grouped:
            order.append(decision.name)
        grouped[decision.name].append(decision)

    merged: list[MemoryRegionDecision] = []
    for name in order:
        items = grouped[name]
        selected = tuple(dict.fromkeys(source for item in items for source in item.selected_sources))
        candidates_by_source: dict[int, MemoryCandidate] = {}
        for item in items:
            for candidate in item.candidates:
                current = candidates_by_source.get(candidate.source_index)
                if current is None or candidate.score > current.score:
                    candidates_by_source[candidate.source_index] = candidate
        candidates = tuple(sorted(candidates_by_source.values(), key=lambda item: item.score, reverse=True))
        merged.append(
            MemoryRegionDecision(
                name=name,
                confidence=max((item.confidence for item in items), default=0.0),
                primary_quality=max((item.primary_quality for item in items), default=0.0),
                selected_sources=selected,
                transferred_pixels=sum(item.transferred_pixels for item in items),
                candidate_count=len(candidates),
                candidates=candidates,
            )
        )
    return tuple(merged)


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
    """Fuse one primary with up to nine observed references.

    References are processed in CPU-friendly batches of at most five, while every
    valid donor remains eligible globally. Per-pixel results are merged by confidence
    and provenance is remapped to the original source index 1..9. Intact primary
    pixels receive a final all-reference conflict veto; damaged pixels keep the
    evidence-first repair policy of the baseline implementation.
    """
    if len(images) < 2:
        raise ValueError("Serve almeno una fotografia di riferimento")
    reference_count = len(images) - 1
    validate_reference_count(reference_count)
    if reference_count > MAX_REFERENCE_IMAGES:
        raise ValueError("Numero di riferimenti oltre il limite supportato")
    if len(occlusion_masks) != len(images):
        raise ValueError("Numero immagini/maschere non compatibile")
    if top_k < 1:
        raise ValueError("top_k deve essere almeno 1")

    base = _validate_image(images[0])
    shape = base.shape
    if any(_validate_image(item).shape != shape for item in images):
        raise ValueError("Le immagini devono essere allineate e avere la stessa forma")
    masks = [_validate_mask(item, shape[:2]) for item in occlusion_masks]
    refs = images[1:]
    primary_mask = masks[0]
    ref_masks = masks[1:]

    if reference_support_masks is None:
        support_masks = _default_support_masks(refs)
    else:
        if len(reference_support_masks) != reference_count:
            raise ValueError("Numero reference/support mask non compatibile")
        support_masks = [_validate_mask(item, shape[:2]) for item in reference_support_masks]

    merged_image = base.copy()
    merged_provenance = np.zeros(shape[:2], dtype=np.uint16)
    merged_confidence = np.zeros(shape[:2], dtype=np.uint8)
    all_decisions: list[MemoryRegionDecision] = []

    # The legacy kernel is intentionally kept at a small working set. Batching keeps
    # the EliteBook memory footprint bounded while making sources 6..9 first-class.
    for start in range(0, reference_count, 5):
        stop = min(reference_count, start + 5)
        global_indices = list(range(start + 1, stop + 1))
        batch_refs = refs[start:stop]
        batch_masks = ref_masks[start:stop]
        batch_support = support_masks[start:stop]
        batch = _legacy.specific_reference_memory_fusion(
            [base, *batch_refs],
            [primary_mask, *batch_masks],
            landmarks5,
            bbox,
            reference_support_masks=batch_support,
            top_k=min(max(1, int(top_k)), len(batch_refs)),
            minimum_region_confidence=minimum_region_confidence,
            minimum_quality_gain=minimum_quality_gain,
            maximum_replace_fraction=maximum_replace_fraction,
            agreement_colour_threshold=agreement_colour_threshold,
            local_refinement_max_shift=local_refinement_max_shift,
            local_refinement_min_response=local_refinement_min_response,
        )

        local_provenance = batch.provenance_map
        global_provenance = np.zeros_like(local_provenance, dtype=np.uint16)
        for local_index, global_index in enumerate(global_indices, start=1):
            global_provenance[local_provenance == local_index] = np.uint16(global_index)

        choose = (global_provenance > 0) & (batch.confidence_map > merged_confidence)
        merged_image[choose] = batch.image[choose]
        merged_provenance[choose] = global_provenance[choose]
        merged_confidence[choose] = batch.confidence_map[choose]
        all_decisions.extend(_map_decision(item, global_indices) for item in batch.decisions)

    # Preserve intact pixels whenever observed references conflict. This guard uses
    # all references, not merely the top-k donors selected by one regional pass.
    disputed = _global_intact_conflict_mask(
        refs,
        ref_masks,
        support_masks,
        primary_mask,
        agreement_colour_threshold,
    )
    if np.any(disputed):
        merged_image[disputed] = base[disputed]
        merged_provenance[disputed] = 0
        merged_confidence[disputed] = 0

    return SpecificReferenceMemoryResult(
        image=merged_image,
        provenance_map=merged_provenance,
        confidence_map=merged_confidence,
        decisions=_aggregate_decisions(all_decisions),
        transferred_pixels=int(np.count_nonzero(merged_provenance)),
    )
