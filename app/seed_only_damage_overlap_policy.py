from __future__ import annotations

"""Narrow fallback for coordinate-preserving, damage-only partial references.

This policy runs after the normal/anatomical same-canvas verifier. It exists for the
specific case where a reference is an explicitly sparse full-canvas component sheet:
all photographed support lies inside a frozen damaged region, so no unaffected MAIN
pixels exist for photometric alignment validation. Geometry can still be verified by
the unchanged canvas coordinates, while identity remains explicitly unverified.
"""

from functools import wraps
from typing import Any

import cv2
import numpy as np

_INSTALLED = False


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return None
    return np.where(item > 0, 255, 0).astype(np.uint8)


def verify_seed_only_damage_overlap(
    workspace,
    reference: np.ndarray,
    runtime_reference_index: int,
    *,
    minimum_pixels: int = 64,
    maximum_support_fraction: float = 0.45,
    minimum_damage_overlap: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    from app.same_canvas_seed_precision_policy import (
        _original_reference_for_runtime_slot,
        _patch_structure,
        _sparse_canvas_support,
    )

    primary = np.asarray(workspace.primary)
    candidate = np.asarray(reference)
    if primary.ndim != 3 or candidate.shape != primary.shape or candidate.ndim != 3:
        return None
    shape = primary.shape[:2]

    source = _original_reference_for_runtime_slot(workspace, runtime_reference_index, candidate)
    support = _sparse_canvas_support(source)
    support_pixels = int(np.count_nonzero(support))
    support_fraction = float(support_pixels / max(1, support.size))
    if support_pixels < max(1, int(minimum_pixels)) or support_fraction > float(maximum_support_fraction):
        return None

    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if not isinstance(frozen, list) or not frozen:
        return None
    primary_damage = _binary(np.asarray(frozen[0]), shape)
    if primary_damage is None or not np.any(primary_damage):
        return None
    reference_damage = np.zeros(shape, dtype=np.uint8)
    if runtime_reference_index + 1 < len(frozen):
        parsed = _binary(np.asarray(frozen[runtime_reference_index + 1]), shape)
        if parsed is not None:
            reference_damage = parsed

    clean_support = (support > 0) & (reference_damage == 0)
    clean_pixels = int(np.count_nonzero(clean_support))
    if clean_pixels < max(1, int(minimum_pixels)):
        return None
    overlap = float(np.count_nonzero(clean_support & (primary_damage > 0)) / max(1, clean_pixels))
    if overlap < float(minimum_damage_overlap):
        return None

    texture_std, edge_fraction = _patch_structure(source, support)
    if texture_std < 4.0 and edge_fraction < 0.012:
        return None

    reliability = np.zeros(shape, dtype=np.uint8)
    maps = workspace.metadata.get("preflight_detail_reliability_maps")
    if isinstance(maps, list) and runtime_reference_index + 1 < len(maps):
        item = np.asarray(maps[runtime_reference_index + 1])
        if item.shape == shape:
            reliability = item.astype(np.uint8, copy=True)
    # A padded canvas can confuse a whole-image detail estimator. Keep a deliberately
    # moderate floor only inside photographed support; this is geometry eligibility,
    # not an identity-confidence upgrade.
    reliability[clean_support] = np.maximum(reliability[clean_support], np.uint8(96))
    reliability[~clean_support] = 0

    return support, reliability, {
        "runtime_reference_index": int(runtime_reference_index),
        "method": "verified-same-canvas-partial",
        "verification_basis": "seed-only-coordinate-preserving-damage-overlap",
        "global_transform_required": False,
        "local_identity_transform": True,
        "identity_status": "not_enough_evidence",
        "may_expand_damage_seed": False,
        "support_pixels": support_pixels,
        "support_fraction": support_fraction,
        "damage_overlap_fraction": overlap,
        "texture_std": texture_std,
        "edge_fraction": edge_fraction,
        "black_pixels_preserved_by_geometric_support": True,
    }


def install_seed_only_damage_overlap_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.case_aware_runtime as runtime

    original = runtime._same_canvas_partial_verification
    if getattr(original, "_seed_only_damage_overlap_policy", False):
        _INSTALLED = True
        return

    @wraps(original)
    def verified(workspace, reference: np.ndarray, runtime_reference_index: int):
        normal = original(workspace, reference, runtime_reference_index)
        if normal is not None:
            return normal
        return verify_seed_only_damage_overlap(workspace, reference, runtime_reference_index)

    verified._seed_only_damage_overlap_policy = True  # type: ignore[attr-defined]
    runtime._same_canvas_partial_verification = verified
    _INSTALLED = True
