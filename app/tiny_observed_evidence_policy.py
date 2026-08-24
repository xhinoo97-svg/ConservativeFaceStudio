from __future__ import annotations

from functools import wraps
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind, BlockSpec


def _binary(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        return np.zeros(shape, dtype=bool)
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=bool)
    return item > 0


def _trusted_flags(workspace, count: int) -> list[bool]:
    identity = workspace.metadata.get("aligned_reference_identity_verified")
    partial = workspace.metadata.get("aligned_reference_partial_geometry_verified")
    identity_flags = [False] * count
    partial_flags = [False] * count
    if isinstance(identity, list) and len(identity) == count:
        identity_flags = [bool(value) for value in identity]
    if isinstance(partial, list) and len(partial) == count:
        partial_flags = [bool(value) for value in partial]
    # Either a verified same-identity full reference or a geometry-verified partial
    # reference is allowed to donate exact observed pixels. Alignment itself must have
    # been established earlier; this policy never estimates geometry from a tiny crop.
    return [a or b for a, b in zip(identity_flags, partial_flags)]


def _source_codes(workspace, count: int) -> list[int]:
    values = workspace.metadata.get("aligned_reference_original_source_indices")
    if isinstance(values, list) and len(values) == count:
        try:
            # ``runtime_source_order`` and therefore original source indices already use
            # 0 for MAIN and 1..9 for imported references. Do not add one again.
            parsed = [int(value) for value in values]
            if all(1 <= value <= 65533 for value in parsed):
                return parsed
        except (TypeError, ValueError):
            pass
    runtime = workspace.metadata.get("aligned_reference_source_indices")
    if isinstance(runtime, list) and len(runtime) == count:
        try:
            # Runtime reference slots are zero based and do need conversion to 1..N.
            return [int(value) + 1 for value in runtime]
        except (TypeError, ValueError):
            pass
    return [index + 1 for index in range(count)]


def _reliability_maps(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    stored = workspace.metadata.get("aligned_reference_detail_reliability_maps")
    if isinstance(stored, list) and len(stored) == count:
        result: list[np.ndarray] = []
        for value in stored:
            item = np.asarray(value)
            if item.ndim == 3:
                item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            if item.shape == shape:
                result.append(item.astype(np.float32, copy=False))
            else:
                result.append(np.ones(shape, dtype=np.float32))
        return result
    return [np.ones(shape, dtype=np.float32) for _ in range(count)]


def _reference_masks(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and len(masks) == count + 1:
        return [_binary(np.asarray(value), shape) for value in masks[1:]]
    return [np.zeros(shape, dtype=bool) for _ in range(count)]


def _support_masks(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    stored = workspace.metadata.get("aligned_reference_support_masks")
    if isinstance(stored, list) and len(stored) == count:
        return [_binary(np.asarray(value), shape) for value in stored]
    return [np.ones(shape, dtype=bool) for _ in range(count)]


def _preclean_evidence_maps(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray] | None:
    """Return authoritative per-pixel observed-source ids when preclean produced them."""
    stored = workspace.metadata.get("preclean_reference_evidence_maps")
    if not isinstance(stored, list) or len(stored) != count:
        return None
    result: list[np.ndarray] = []
    for value in stored:
        item = np.asarray(value)
        if item.shape != shape:
            return None
        if item.ndim != 2:
            return None
        result.append(item.astype(np.uint16, copy=False))
    return result


def _complete_observed_pixels(workspace, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    refs = list(workspace.aligned_references)
    if not refs:
        return image, {"tiny_observed_pixels": 0, "tiny_observed_sources": []}

    shape = image.shape[:2]
    count = len(refs)
    trusted = _trusted_flags(workspace, count)
    support = _support_masks(workspace, count, shape)
    blocked = _reference_masks(workspace, count, shape)
    reliability = _reliability_maps(workspace, count, shape)
    source_codes = _source_codes(workspace, count)
    evidence_maps = _preclean_evidence_maps(workspace, count, shape)

    target = _binary(workspace.metadata.get("inpaint_target_mask"), shape)
    if not np.any(target) and isinstance(workspace.occlusion_masks, list) and workspace.occlusion_masks:
        target = _binary(np.asarray(workspace.occlusion_masks[0]), shape)
    if not np.any(target):
        return image, {"tiny_observed_pixels": 0, "tiny_observed_sources": []}

    provenance = workspace.provenance_map
    if not isinstance(provenance, np.ndarray) or provenance.shape != shape:
        provenance = np.zeros(shape, dtype=np.uint16)
    else:
        provenance = provenance.astype(np.uint16, copy=True)

    # Only complete pixels still attributed to the primary. Previously transferred,
    # symmetry, or generated pixels are left untouched here.
    unresolved = target & (provenance == 0)
    if not np.any(unresolved):
        return image, {"tiny_observed_pixels": 0, "tiny_observed_sources": []}

    output = image.copy()
    best_score = np.full(shape, -np.inf, dtype=np.float32)
    best_source = np.full(shape, -1, dtype=np.int16)
    best_provenance = np.zeros(shape, dtype=np.uint16)
    best_value = np.zeros_like(image)
    conflict = np.zeros(shape, dtype=bool)

    for index, reference in enumerate(refs):
        if not trusted[index] or reference.shape != image.shape:
            continue

        if evidence_maps is not None:
            evidence = evidence_maps[index]
            # The preclean evidence map is authoritative once available: zero means the
            # working pixel is not supported by any observed photograph. Non-zero means
            # it is observed, possibly copied from another reference; that true source id
            # must survive the transfer instead of being replaced by this working slot.
            evidence_valid = (evidence > 0) & (evidence < np.uint16(65534))
            valid = unresolved & support[index] & evidence_valid
            provenance_candidate = evidence
        else:
            # Backward-compatible fail-closed path when no preclean evidence map exists.
            valid = unresolved & support[index] & ~blocked[index]
            provenance_candidate = np.full(shape, np.uint16(source_codes[index]), dtype=np.uint16)

        if not np.any(valid):
            continue
        score = reliability[index]
        better = valid & (score > best_score + 1e-6)
        tie = valid & ~better & np.isfinite(best_score) & (np.abs(score - best_score) <= 1e-6)
        if np.any(tie):
            # Equal-confidence contradictory donors are not silently averaged. This
            # prevents Frankenstein seams when two references disagree geometrically.
            delta = np.max(np.abs(reference.astype(np.int16) - best_value.astype(np.int16)), axis=2)
            conflict |= tie & (delta > 18)
        best_score[better] = score[better]
        best_source[better] = index
        best_value[better] = reference[better]
        best_provenance[better] = provenance_candidate[better]

    accepted = unresolved & (best_source >= 0) & ~conflict & (best_provenance > 0) & (best_provenance < np.uint16(65534))
    if not np.any(accepted):
        return image, {
            "tiny_observed_pixels": 0,
            "tiny_observed_sources": [],
            "tiny_observed_conflicts": int(np.count_nonzero(conflict & unresolved)),
            "preclean_evidence_authoritative": evidence_maps is not None,
        }

    output[accepted] = best_value[accepted]
    provenance[accepted] = best_provenance[accepted]
    used_source_codes = sorted(int(value) for value in np.unique(best_provenance[accepted]) if 0 < int(value) < 65534)
    workspace.provenance_map = provenance

    observed_mask = _binary(workspace.metadata.get("inpaint_observed_mask"), shape)
    observed_mask |= accepted
    workspace.metadata["inpaint_observed_mask"] = observed_mask.astype(np.uint8) * 255

    return output, {
        "tiny_observed_pixels": int(np.count_nonzero(accepted)),
        "tiny_observed_sources": used_source_codes,
        "tiny_observed_conflicts": int(np.count_nonzero(conflict & unresolved)),
        "minimum_transfer_pixels": 1,
        "requires_preverified_geometry": True,
        "preclean_evidence_authoritative": evidence_maps is not None,
        "true_source_provenance_preserved": True,
    }


def install_tiny_observed_evidence_policy(executor) -> None:
    """Consume every trusted observed donor pixel after regional selection/fusion.

    The policy deliberately does *not* relax alignment requirements. A one-pixel crop
    cannot estimate a transform. Once the reference transform has already been trusted
    by block 5, however, one observed pixel is still valid evidence and must not be
    discarded by regional minimum-area thresholds.
    """
    for kind in (BlockKind.REGION_SELECT, BlockKind.FUSION):
        original = executor._handlers.get(kind)
        if original is None:
            continue

        @wraps(original)
        def handler(block: BlockSpec, parameters: dict[str, Any], _original=original) -> ExecutionResult:
            result = _original(block, parameters)
            completed, details_extra = _complete_observed_pixels(executor.workspace, result.image)
            details = dict(result.details)
            details.update(details_extra)
            return ExecutionResult(result.block, completed, details)

        executor._handlers[kind] = handler
