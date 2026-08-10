from __future__ import annotations

"""Recover tiny detector-missed borders from verified same-canvas partial references.

A dark sticker/scribble proposal can lose one or two boundary pixels after morphology.
For a coordinate-preserving sparse reference this must not make the real donor unusable.
This policy is deliberately narrow: geometry may tolerate only support contained inside
a 2-pixel dilation of the frozen damage seed, and target expansion is restricted to the
same donor support connected to an already-confirmed seed. Distant pixels never expand
or create a new damage component.
"""

from functools import wraps
from typing import Any

import cv2
import numpy as np

_INSTALLED = False
_EDGE_RADIUS = 2


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(value, np.ndarray):
        return None
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return None
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _dilate(mask: np.ndarray, radius: int = _EDGE_RADIUS) -> np.ndarray:
    size = max(3, int(radius) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def verify_edge_connected_seed_overlap(
    workspace,
    reference: np.ndarray,
    runtime_reference_index: int,
    *,
    minimum_exact_pixels: int = 32,
    minimum_dilated_overlap: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Verify a sparse donor whose only mismatch is a <=2 px detector border."""
    from app.seed_only_damage_overlap_policy import verify_seed_only_damage_overlap

    candidate = verify_seed_only_damage_overlap(
        workspace,
        reference,
        runtime_reference_index,
        minimum_damage_overlap=0.0,
    )
    if candidate is None:
        return None
    support, reliability, details = candidate
    shape = workspace.primary.shape[:2]

    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if not isinstance(frozen, list) or not frozen:
        return None
    primary_damage = _binary(np.asarray(frozen[0]), shape)
    if primary_damage is None or not np.any(primary_damage):
        return None

    support_bool = (np.asarray(support) > 0) & (np.asarray(reliability) > 0)
    support_pixels = int(np.count_nonzero(support_bool))
    if support_pixels <= 0:
        return None

    exact_pixels = int(np.count_nonzero(support_bool & (primary_damage > 0)))
    if exact_pixels < max(1, int(minimum_exact_pixels)):
        return None

    dilated = _dilate(primary_damage) > 0
    dilated_overlap = float(np.count_nonzero(support_bool & dilated) / support_pixels)
    if dilated_overlap < float(minimum_dilated_overlap):
        return None

    exact_overlap = float(exact_pixels / support_pixels)
    # The normal seed-only verifier already accepts >=95%. This fallback exists only
    # for the narrow boundary-loss case; using it for an already-normal donor would
    # blur diagnostics and make regressions harder to reason about.
    if exact_overlap >= 0.95:
        return None

    updated = dict(details)
    updated.update(
        {
            "verification_basis": "seed-only-coordinate-preserving-edge-tolerant-damage-overlap",
            "edge_connected_seed_tolerance": True,
            "edge_tolerance_pixels": int(_EDGE_RADIUS),
            "damage_overlap_fraction": exact_overlap,
            "dilated_damage_overlap_fraction": dilated_overlap,
            "may_expand_damage_seed": True,
        }
    )
    return support, reliability, updated


def expand_edge_connected_seed(workspace) -> dict[str, Any]:
    """Add only verified donor pixels immediately connected to the frozen seed."""
    shape = workspace.primary.shape[:2]
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if not isinstance(frozen, list) or not frozen:
        return {"added_pixels": 0, "eligible_donors": 0, "reason": "no_frozen_seed"}
    primary_damage = _binary(np.asarray(frozen[0]), shape)
    if primary_damage is None or not np.any(primary_damage):
        return {"added_pixels": 0, "eligible_donors": 0, "reason": "no_frozen_seed"}

    diagnostics = workspace.metadata.get("same_canvas_partial_alignment_diagnostics")
    supports = workspace.metadata.get("aligned_reference_support_masks")
    runtime_indices = workspace.metadata.get("aligned_reference_source_indices")
    if not isinstance(diagnostics, list) or not isinstance(supports, list) or not isinstance(runtime_indices, list):
        return {"added_pixels": 0, "eligible_donors": 0, "reason": "no_verified_partial_metadata"}
    if len(supports) != len(runtime_indices):
        return {"added_pixels": 0, "eligible_donors": 0, "reason": "aligned_metadata_mismatch"}

    existing = workspace.metadata.get("reference_consensus_occlusion")
    merged = _binary(existing, shape)
    if merged is None:
        merged = primary_damage.copy()
    else:
        merged = cv2.bitwise_or(merged, primary_damage)

    frozen_bool = primary_damage > 0
    edge_band = (_dilate(primary_damage) > 0) & ~frozen_bool
    total_added = 0
    eligible = 0
    donor_reports: list[dict[str, Any]] = []

    for item in diagnostics:
        if not isinstance(item, dict) or not bool(item.get("edge_connected_seed_tolerance", False)):
            continue
        runtime_index = item.get("runtime_reference_index")
        try:
            slot = [int(v) for v in runtime_indices].index(int(runtime_index))
        except (TypeError, ValueError):
            continue
        support = _binary(np.asarray(supports[slot]), shape)
        if support is None:
            continue
        support_bool = support > 0
        supported_seed = support_bool & frozen_bool
        if not np.any(supported_seed):
            continue

        # Expansion must stay in this donor's own support and within two pixels of
        # seed pixels supported by the very same donor. This prevents a second/distant
        # support island from becoming a new damage proposal.
        donor_seed = np.where(supported_seed, 255, 0).astype(np.uint8)
        connected_near = _dilate(donor_seed) > 0
        expansion = support_bool & edge_band & connected_near
        local = np.where(expansion, 255, 0).astype(np.uint8)
        before = merged.copy()
        merged = cv2.bitwise_or(merged, local)
        added = int(np.count_nonzero((merged > 0) & (before == 0)))
        total_added += added
        eligible += 1
        donor_reports.append({
            "runtime_reference_index": int(runtime_index),
            "slot": int(slot),
            "added_pixels": added,
            "edge_tolerance_pixels": int(_EDGE_RADIUS),
        })

    workspace.metadata["reference_consensus_occlusion"] = merged
    result = {
        "added_pixels": int(total_added),
        "eligible_donors": int(eligible),
        "edge_tolerance_pixels": int(_EDGE_RADIUS),
        "distant_expansion_allowed": False,
        "diagnostics": donor_reports,
        "reason": "edge_connected_verified_partial_expansion" if total_added else "no_edge_pixels_added",
    }
    workspace.metadata["edge_connected_seed_expansion_diagnostics"] = result
    return result


def install_edge_connected_seed_expansion_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.case_aware_runtime as case_runtime
    import app.partial_reference_runtime as partial_runtime

    previous_verify = case_runtime._same_canvas_partial_verification

    @wraps(previous_verify)
    def verified(workspace, reference: np.ndarray, runtime_reference_index: int):
        normal = previous_verify(workspace, reference, runtime_reference_index)
        if normal is not None:
            return normal
        return verify_edge_connected_seed_overlap(workspace, reference, runtime_reference_index)

    case_runtime._same_canvas_partial_verification = verified

    previous_merge = partial_runtime._merge_frozen_primary_hint

    @wraps(previous_merge)
    def merged_hint(workspace) -> int:
        base_added = int(previous_merge(workspace))
        edge = expand_edge_connected_seed(workspace)
        return base_added + int(edge.get("added_pixels", 0))

    partial_runtime._merge_frozen_primary_hint = merged_hint
    _INSTALLED = True
