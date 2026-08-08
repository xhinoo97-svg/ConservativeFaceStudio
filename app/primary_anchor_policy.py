from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.restoration import detect_occlusion_candidates


@dataclass(frozen=True)
class PrimaryAnchorDecision:
    applied: bool
    reason: str
    matched_reference_count: int
    original_selected_source_index: int


def _same_canvas_match(primary: np.ndarray, reference: np.ndarray) -> bool:
    if primary.shape != reference.shape or primary.ndim != 3 or primary.shape[2] != 3:
        return False
    primary_occ = detect_occlusion_candidates(primary)
    reference_occ = detect_occlusion_candidates(reference)
    observed = (
        (np.max(primary, axis=2) > 2)
        & (np.max(reference, axis=2) > 2)
        & (primary_occ == 0)
        & (reference_occ == 0)
    )
    observed_u8 = observed.astype(np.uint8) * 255
    observed_u8 = cv2.erode(
        observed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    comparable = observed_u8 > 0
    if int(np.count_nonzero(comparable)) < max(256, int(round(primary.shape[0] * primary.shape[1] * 0.05))):
        return False

    base_lab = cv2.cvtColor(primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    delta = np.mean(np.abs(base_lab - ref_lab), axis=2)[comparable]
    if float(np.median(delta)) > 0.035 or float(np.percentile(delta, 90.0)) > 0.10:
        return False

    base_gray = cv2.cvtColor(primary, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base_gx = cv2.Sobel(base_gray, cv2.CV_32F, 1, 0, ksize=3)
    base_gy = cv2.Sobel(base_gray, cv2.CV_32F, 0, 1, ksize=3)
    ref_gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    base_grad = cv2.magnitude(base_gx, base_gy)
    ref_grad = cv2.magnitude(ref_gx, ref_gy)
    edges = comparable & ((base_grad >= 12.0) | (ref_grad >= 12.0))
    if int(np.count_nonzero(edges)) >= 64:
        edge_delta = np.abs(base_grad[edges] - ref_grad[edges])
        if float(np.median(edge_delta)) > 8.0 or float(np.percentile(edge_delta, 90.0)) > 42.0:
            return False
    return True


def _record_same_canvas_evidence(workspace, matches: list[int], selected: int, *, applied: bool) -> None:
    workspace.metadata["same_canvas_primary_anchor"] = {
        "applied": bool(applied),
        "matched_original_reference_indices": [int(value) for value in matches],
        "preflight_selected_source_index": int(selected),
        "restored_source_index": 0,
    }


def restore_imported_primary_for_same_canvas(workspace, originals: list[np.ndarray]) -> PrimaryAnchorDecision:
    """Keep imported target semantics and persist verified same-canvas donor evidence.

    Same-canvas verification is useful even when preflight already kept source 0 as the
    runtime primary. Persisting the matched original reference indices lets later strict
    repair expand an observed damage seed without re-running or weakening geometry.
    """
    if len(originals) < 2:
        return PrimaryAnchorDecision(False, "no_references", 0, int(workspace.metadata.get("selected_primary_original_source_index", 0)))

    primary_occ = detect_occlusion_candidates(originals[0])
    if int(np.count_nonzero(primary_occ)) == 0:
        return PrimaryAnchorDecision(False, "primary_has_no_damage_seed", 0, int(workspace.metadata.get("selected_primary_original_source_index", 0)))

    matches = [index for index, reference in enumerate(originals[1:], start=1) if _same_canvas_match(originals[0], reference)]
    selected = int(workspace.metadata.get("selected_primary_original_source_index", 0))
    if not matches:
        return PrimaryAnchorDecision(False, "already_primary_or_no_same_canvas_match", 0, selected)

    if selected == 0:
        _record_same_canvas_evidence(workspace, matches, selected, applied=False)
        return PrimaryAnchorDecision(False, "already_primary_same_canvas_verified", len(matches), selected)

    runtime = [workspace.primary, *workspace.references]
    order_raw = workspace.metadata.get("runtime_source_order")
    if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
        _record_same_canvas_evidence(workspace, matches, selected, applied=False)
        return PrimaryAnchorDecision(False, "missing_runtime_source_order", len(matches), selected)
    order = [int(value) for value in order_raw]
    try:
        primary_slot = order.index(0)
    except ValueError:
        _record_same_canvas_evidence(workspace, matches, selected, applied=False)
        return PrimaryAnchorDecision(False, "imported_primary_missing_from_runtime", len(matches), selected)

    reordered_slots = [primary_slot, *[slot for slot in range(len(runtime)) if slot != primary_slot]]
    reordered_runtime = [runtime[slot].copy() for slot in reordered_slots]
    reordered_order = [order[slot] for slot in reordered_slots]
    workspace.primary = reordered_runtime[0]
    workspace.references = reordered_runtime[1:]
    workspace.metadata["runtime_source_order"] = reordered_order
    workspace.metadata["selected_primary_original_source_index"] = 0

    for key in ("preflight_original_occlusion_masks", "preflight_detail_reliability_maps"):
        values = workspace.metadata.get(key)
        if isinstance(values, list) and len(values) == len(runtime):
            workspace.metadata[key] = [values[slot] for slot in reordered_slots]

    _record_same_canvas_evidence(workspace, matches, selected, applied=True)
    return PrimaryAnchorDecision(True, "verified_same_canvas_donor_semantics", len(matches), selected)
