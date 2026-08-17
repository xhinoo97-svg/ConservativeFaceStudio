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


def _face_local_same_canvas_identity_match(
    primary: np.ndarray,
    reference: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> bool:
    """Require observed same-canvas agreement inside the MAIN face box.

    Whole-canvas equality is useful for local repair, but a shared/static background
    must never become identity authority. This second gate reuses the same photometric
    and gradient limits as `_same_canvas_match`, restricted to observed face pixels.
    """
    if bbox is None or primary.shape != reference.shape or primary.ndim != 3:
        return False
    try:
        x, y, w, h = (int(value) for value in bbox)
    except (TypeError, ValueError):
        return False
    if w <= 0 or h <= 0:
        return False

    height, width = primary.shape[:2]
    margin_x = max(2, int(round(w * 0.12)))
    margin_y = max(2, int(round(h * 0.12)))
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(width, x + w + margin_x)
    y1 = min(height, y + h + margin_y)
    if x1 <= x0 or y1 <= y0:
        return False

    region = np.zeros((height, width), dtype=bool)
    region[y0:y1, x0:x1] = True
    region_pixels = int(np.count_nonzero(region))

    primary_occ = detect_occlusion_candidates(primary)
    reference_occ = detect_occlusion_candidates(reference)
    observed = (
        region
        & (np.max(primary, axis=2) > 2)
        & (np.max(reference, axis=2) > 2)
        & (primary_occ == 0)
        & (reference_occ == 0)
    )
    observed_u8 = observed.astype(np.uint8) * 255
    observed_u8 = cv2.erode(
        observed_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    comparable = observed_u8 > 0
    if int(np.count_nonzero(comparable)) < max(96, int(round(region_pixels * 0.15))):
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
    if int(np.count_nonzero(edges)) >= 32:
        edge_delta = np.abs(base_grad[edges] - ref_grad[edges])
        if float(np.median(edge_delta)) > 8.0 or float(np.percentile(edge_delta, 90.0)) > 42.0:
            return False
    return True


def _record_same_canvas_evidence(
    workspace,
    matches: list[int],
    selected: int,
    *,
    identity_bridge_matches: list[int],
    applied: bool,
    imported_primary: np.ndarray | None = None,
) -> None:
    workspace.metadata["same_canvas_primary_anchor"] = {
        "applied": bool(applied),
        "matched_original_reference_indices": [int(value) for value in matches],
        "identity_bridge_original_reference_indices": [int(value) for value in identity_bridge_matches],
        "identity_bridge_requires_face_local_observed_agreement": True,
        "preflight_selected_source_index": int(selected),
        "restored_source_index": 0,
    }
    # Keep exactly one frozen imported target for later same-canvas difference maps.
    # This avoids comparing clean donors with an already deblurred/enhanced runtime
    # primary, which otherwise creates false whole-face differences.
    if isinstance(imported_primary, np.ndarray) and imported_primary.size:
        workspace.metadata["same_canvas_imported_primary"] = imported_primary.copy()


def restore_imported_primary_for_same_canvas(workspace, originals: list[np.ndarray]) -> PrimaryAnchorDecision:
    """Keep imported target semantics and persist verified same-canvas donor evidence."""
    selected = int(workspace.metadata.get("selected_primary_original_source_index", 0))
    if len(originals) < 2:
        return PrimaryAnchorDecision(False, "no_references", 0, selected)

    primary_occ = detect_occlusion_candidates(originals[0])
    if int(np.count_nonzero(primary_occ)) == 0:
        return PrimaryAnchorDecision(False, "primary_has_no_damage_seed", 0, selected)

    matches = [
        index
        for index, reference in enumerate(originals[1:], start=1)
        if _same_canvas_match(originals[0], reference)
    ]
    if not matches:
        return PrimaryAnchorDecision(False, "already_primary_or_no_same_canvas_match", 0, selected)

    bboxes = workspace.metadata.get("preflight_face_bboxes")
    primary_bbox = None
    if isinstance(bboxes, list) and bboxes:
        raw_bbox = bboxes[0]
        if isinstance(raw_bbox, (tuple, list)) and len(raw_bbox) == 4:
            primary_bbox = tuple(int(value) for value in raw_bbox)
    identity_matches = [
        index
        for index in matches
        if _face_local_same_canvas_identity_match(originals[0], originals[index], primary_bbox)
    ]

    if selected == 0:
        _record_same_canvas_evidence(
            workspace,
            matches,
            selected,
            identity_bridge_matches=identity_matches,
            applied=False,
            imported_primary=originals[0],
        )
        return PrimaryAnchorDecision(False, "already_primary_same_canvas_verified", len(matches), selected)

    runtime = [workspace.primary, *workspace.references]
    order_raw = workspace.metadata.get("runtime_source_order")
    if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
        _record_same_canvas_evidence(
            workspace,
            matches,
            selected,
            identity_bridge_matches=identity_matches,
            applied=False,
            imported_primary=originals[0],
        )
        return PrimaryAnchorDecision(False, "missing_runtime_source_order", len(matches), selected)
    order = [int(value) for value in order_raw]
    try:
        primary_slot = order.index(0)
    except ValueError:
        _record_same_canvas_evidence(
            workspace,
            matches,
            selected,
            identity_bridge_matches=identity_matches,
            applied=False,
            imported_primary=originals[0],
        )
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

    _record_same_canvas_evidence(
        workspace,
        matches,
        selected,
        identity_bridge_matches=identity_matches,
        applied=True,
        imported_primary=originals[0],
    )
    return PrimaryAnchorDecision(True, "verified_same_canvas_donor_semantics", len(matches), selected)
