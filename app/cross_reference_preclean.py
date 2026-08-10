from __future__ import annotations

"""Observed-only cleanup of aligned references before component selection.

The first alignment pass puts every usable reference in the primary canvas.  The
occlusion block then provides one frozen damage mask per source.  This module uses
only *observed* pixels from the other aligned sources to clean damaged pixels in each
reference working copy.  It never upgrades generated/symmetry pixels to evidence.

The cleaned copies are intended for landmark/component scoring and the component
bank.  Authoritative source support/provenance remain separate in metadata.
"""

from dataclasses import dataclass
from functools import wraps
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind, BlockSpec
from app.reference_limits import MAX_REFERENCE_IMAGES, validate_reference_count


@dataclass(frozen=True)
class ReferencePrecleanStats:
    reference_index: int
    damaged_pixels: int
    repaired_observed_pixels: int
    unresolved_pixels: int
    donor_sources: tuple[int, ...]


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Maschera reference non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _support_for_reference(workspace, slot: int, shape: tuple[int, int]) -> np.ndarray:
    supports = workspace.metadata.get("aligned_reference_support_masks")
    if isinstance(supports, list) and slot < len(supports):
        candidate = np.asarray(supports[slot])
        if candidate.shape == shape:
            return _binary(candidate, shape)
    reference = workspace.aligned_references[slot]
    # Fallback only when no geometric support exists.  Exact-zero padding connected
    # to the border is considered absent; interior black pixels remain evidence.
    exact_zero = np.all(reference == 0, axis=2).astype(np.uint8)
    if not np.any(exact_zero):
        return np.full(shape, 255, dtype=np.uint8)
    _, labels = cv2.connectedComponents(exact_zero, connectivity=8)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    missing = np.isin(labels, border[border != 0])
    return np.where(~missing, 255, 0).astype(np.uint8)


def _reference_damage_masks(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and len(masks) >= count + 1:
        return [_binary(np.asarray(masks[index + 1]), shape) for index in range(count)]
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and len(frozen) >= count + 1:
        return [_binary(np.asarray(frozen[index + 1]), shape) for index in range(count)]
    return [np.zeros(shape, dtype=np.uint8) for _ in range(count)]


def _source_indices(workspace, count: int) -> list[int]:
    raw = workspace.metadata.get("aligned_reference_original_source_indices")
    if isinstance(raw, list) and len(raw) == count:
        try:
            return [int(value) for value in raw]
        except (TypeError, ValueError):
            pass
    # source 0 is primary; references are 1..9 in imported order.
    return list(range(1, count + 1))


def _robust_observed_donor(
    donor_images: list[np.ndarray],
    donor_valid: list[np.ndarray],
    donor_sources: list[int],
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose/aggregate observed donor pixels without inventing texture.

    A single observed donor is copied exactly.  With multiple donors the channel-wise
    median is used only when donors agree photometrically; otherwise the donor closest
    to the local median is selected pixel-by-pixel.  Returned provenance always names
    an actual source, never the synthetic median.
    """
    shape = target.shape
    if not donor_images:
        return np.zeros((*shape, 3), dtype=np.uint8), np.zeros(shape, np.uint8), np.zeros(shape, np.uint16)

    stack = np.stack(donor_images, axis=0).astype(np.float32)
    valid = np.stack(donor_valid, axis=0)
    counts = np.sum(valid, axis=0)
    usable = (counts > 0) & target
    chosen = np.zeros((*shape, 3), dtype=np.uint8)
    provenance = np.zeros(shape, dtype=np.uint16)
    confidence = np.zeros(shape, dtype=np.uint8)
    if not np.any(usable):
        return chosen, confidence, provenance

    # Median only guides donor choice; output remains an observed donor pixel.
    masked = np.where(valid[..., None], stack, np.nan)
    median = np.nanmedian(masked, axis=0)
    distances = np.mean(np.abs(stack - median[None, ...]), axis=3)
    distances[~valid] = np.inf
    best = np.argmin(distances, axis=0)
    rows, cols = np.indices(shape)
    selected = stack[best, rows, cols]
    chosen[usable] = np.clip(selected[usable], 0, 255).astype(np.uint8)
    source_array = np.asarray(donor_sources, dtype=np.uint16)
    provenance[usable] = source_array[best[usable]]

    # Confidence expresses donor agreement, not evidence percentage.  One donor is
    # still real evidence but lower-confidence than several agreeing donors.
    confidence[usable & (counts == 1)] = 180
    multi = usable & (counts >= 2)
    if np.any(multi):
        finite = np.where(valid, distances, np.nan)
        spread = np.nanmedian(finite, axis=0)
        score = np.clip(1.0 - spread / 32.0, 0.0, 1.0)
        confidence[multi] = np.clip(np.rint(190.0 + 65.0 * score[multi]), 0, 255).astype(np.uint8)
    return chosen, confidence, provenance


def preclean_aligned_references(workspace) -> tuple[list[np.ndarray], list[np.ndarray], list[ReferencePrecleanStats]]:
    refs = [np.asarray(item).copy() for item in workspace.aligned_references]
    count = len(refs)
    validate_reference_count(count)
    if count == 0:
        return refs, [], []
    if count > MAX_REFERENCE_IMAGES:
        raise ValueError("Il preclean supporta al massimo nove reference")

    shape = workspace.primary.shape[:2]
    if any(item.shape[:2] != shape for item in refs):
        raise ValueError("Le reference devono essere allineate prima del preclean")

    supports = [_support_for_reference(workspace, index, shape) for index in range(count)]
    damage = _reference_damage_masks(workspace, count, shape)
    source_ids = _source_indices(workspace, count)

    cleaned: list[np.ndarray] = []
    evidence_maps: list[np.ndarray] = []
    stats: list[ReferencePrecleanStats] = []

    for target_index, reference in enumerate(refs):
        target_damage = damage[target_index] > 0
        working = reference.copy()
        # Original observed pixels of this reference retain their own source id.
        evidence = np.zeros(shape, dtype=np.uint16)
        original_observed = (supports[target_index] > 0) & ~target_damage
        evidence[original_observed] = np.uint16(source_ids[target_index])

        donor_images: list[np.ndarray] = []
        donor_valid: list[np.ndarray] = []
        donor_sources: list[int] = []
        for donor_index, donor in enumerate(refs):
            if donor_index == target_index:
                continue
            valid = (supports[donor_index] > 0) & (damage[donor_index] == 0) & target_damage
            if not np.any(valid):
                continue
            donor_images.append(donor)
            donor_valid.append(valid)
            donor_sources.append(source_ids[donor_index])

        chosen, confidence, provenance = _robust_observed_donor(
            donor_images,
            donor_valid,
            donor_sources,
            target_damage,
        )
        repaired = provenance > 0
        working[repaired] = chosen[repaired]
        evidence[repaired] = provenance[repaired]
        unresolved = target_damage & ~repaired

        cleaned.append(working)
        evidence_maps.append(evidence)
        stats.append(
            ReferencePrecleanStats(
                reference_index=target_index,
                damaged_pixels=int(np.count_nonzero(target_damage)),
                repaired_observed_pixels=int(np.count_nonzero(repaired)),
                unresolved_pixels=int(np.count_nonzero(unresolved)),
                donor_sources=tuple(sorted({int(value) for value in np.unique(provenance[repaired]) if int(value) > 0})),
            )
        )

    workspace.metadata["preclean_reference_evidence_maps"] = [item.copy() for item in evidence_maps]
    workspace.metadata["preclean_reference_stats"] = [item.__dict__.copy() for item in stats]
    workspace.metadata["preclean_reference_unresolved_masks"] = [
        np.where((damage[index] > 0) & (evidence_maps[index] == 0), 255, 0).astype(np.uint8)
        for index in range(count)
    ]
    return cleaned, evidence_maps, stats


def install_cross_reference_preclean(executor) -> None:
    """Run after block 6 so block 7 sees cleaned working references."""
    original = executor._handlers.get(BlockKind.OCCLUSION_MASK)
    if original is None:
        return

    @wraps(original)
    def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        result = original(block, parameters)
        if not executor.workspace.aligned_references:
            return result
        cleaned, evidence_maps, stats = preclean_aligned_references(executor.workspace)
        executor.workspace.aligned_references = [item.copy() for item in cleaned]
        details = dict(result.details)
        details["cross_reference_preclean"] = True
        details["preclean_reference_count"] = len(cleaned)
        details["preclean_observed_repaired_pixels"] = int(
            sum(item.repaired_observed_pixels for item in stats)
        )
        details["preclean_unresolved_pixels"] = int(sum(item.unresolved_pixels for item in stats))
        details["preclean_evidence_maps"] = len(evidence_maps)
        return ExecutionResult(result.block, result.image, details)

    executor._handlers[BlockKind.OCCLUSION_MASK] = handler
