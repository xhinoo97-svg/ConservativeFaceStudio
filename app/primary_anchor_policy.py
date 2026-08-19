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
    comparable_count = int(np.count_nonzero(comparable))
    if comparable_count < max(256, int(round(primary.shape[0] * primary.shape[1] * 0.05))):
        return False

    base_lab = cv2.cvtColor(primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    delta_map = np.mean(np.abs(base_lab - ref_lab), axis=2)
    delta = delta_map[comparable]
    # The global photometric rule remains strict. Local-damage tolerance is allowed only
    # after this check, and only for the secondary gradient test below.
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
    raw_edges = comparable & ((base_grad >= 12.0) | (ref_grad >= 12.0))
    raw_edge_count = int(np.count_nonzero(raw_edges))

    # Mosaic/pixelation may alter only a few source pixels while creating many artificial
    # gradient boundaries around that local region. When the strict global Lab check has
    # already passed and the photometric mismatch is <=10% of comparable pixels, exclude
    # only a small dilated neighborhood of those local mismatches from the edge check.
    # The reference still needs substantial stable edge support, so a same-background but
    # structurally different image cannot pass by having all informative edges discarded.
    edge_comparable = raw_edges
    local_mismatch = comparable & (delta_map > 0.035)
    mismatch_count = int(np.count_nonzero(local_mismatch))
    mismatch_fraction = mismatch_count / max(1, comparable_count)
    if 0 < mismatch_count and mismatch_fraction <= 0.10 and raw_edge_count >= 64:
        mismatch_u8 = local_mismatch.astype(np.uint8) * 255
        exclusion = cv2.dilate(
            mismatch_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0
        stable_edges = raw_edges & ~exclusion
        stable_edge_count = int(np.count_nonzero(stable_edges))
        minimum_stable_edges = max(64, int(round(raw_edge_count * 0.35)))
        if stable_edge_count < minimum_stable_edges:
            return False
        edge_comparable = stable_edges

    if int(np.count_nonzero(edge_comparable)) >= 64:
        edge_delta = np.abs(base_grad[edge_comparable] - ref_grad[edge_comparable])
        if float(np.median(edge_delta)) > 8.0 or float(np.percentile(edge_delta, 90.0)) > 42.0:
            return False
    return True


def _face_peripheral_band(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    """Return an inner-face perimeter band, excluding the usually damaged centre."""
    if bbox is None:
        return None
    try:
        x, y, w, h = (int(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if w <= 8 or h <= 8:
        return None

    height, width = shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    if x1 - x0 <= 8 or y1 - y0 <= 8:
        return None

    outer = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(outer, (x0, y0), (x1 - 1, y1 - 1), 255, -1)

    inset_x = max(3, int(round((x1 - x0) * 0.24)))
    inset_y = max(3, int(round((y1 - y0) * 0.24)))
    inner_x0 = min(x1, x0 + inset_x)
    inner_y0 = min(y1, y0 + inset_y)
    inner_x1 = max(x0, x1 - inset_x)
    inner_y1 = max(y0, y1 - inset_y)
    if inner_x1 > inner_x0 and inner_y1 > inner_y0:
        cv2.rectangle(outer, (inner_x0, inner_y0), (inner_x1 - 1, inner_y1 - 1), 0, -1)
    return outer > 0


def _face_local_same_canvas_identity_match(
    primary: np.ndarray,
    reference: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> bool:
    """Require observed agreement on the peripheral band inside the MAIN face bbox.

    Whole-canvas agreement is useful for local repair, but a shared/static background
    must never become identity authority. The centre of the face is intentionally
    excluded because censor blur/mosaic/stickers usually occupy eyes-nose-mouth. The
    remaining forehead/temple/cheek/jaw perimeter is still person-specific evidence.
    """
    if primary.shape != reference.shape or primary.ndim != 3 or primary.shape[2] != 3:
        return False
    region = _face_peripheral_band(primary.shape[:2], bbox)
    if region is None:
        return False
    region_pixels = int(np.count_nonzero(region))
    if region_pixels < 64:
        return False

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
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    comparable = observed_u8 > 0
    if int(np.count_nonzero(comparable)) < max(64, int(round(region_pixels * 0.20))):
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
    if int(np.count_nonzero(edges)) >= 24:
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
    primary_occlusion_seed_present: bool,
) -> None:
    identity_bridge_values = [int(value) for value in identity_bridge_matches]
    workspace.metadata["same_canvas_primary_anchor"] = {
        "applied": bool(applied),
        "matched_original_reference_indices": [int(value) for value in matches],
        # Keep the original key for frozen/tracked compatibility and write the explicit
        # canonical alias expected by downstream release evidence. Both lists contain
        # only the stricter face-local identity bridge, never the global same-canvas list.
        "identity_bridge_original_reference_indices": identity_bridge_values,
        "identity_bridge_matched_original_reference_indices": identity_bridge_values,
        "identity_bridge_requires_face_local_observed_agreement": True,
        "identity_bridge_region": "inner_face_peripheral_band_v1",
        "identity_bridge_rule": "global_same_canvas_plus_face_peripheral_band_v1",
        "primary_occlusion_seed_present": bool(primary_occlusion_seed_present),
        "preflight_selected_source_index": int(selected),
        "restored_source_index": 0,
    }
    if isinstance(imported_primary, np.ndarray) and imported_primary.size:
        workspace.metadata["same_canvas_imported_primary"] = imported_primary.copy()


def restore_imported_primary_for_same_canvas(workspace, originals: list[np.ndarray]) -> PrimaryAnchorDecision:
    """Keep imported target semantics and persist same-canvas evidence independently of damage type."""
    selected = int(workspace.metadata.get("selected_primary_original_source_index", 0))
    if len(originals) < 2:
        return PrimaryAnchorDecision(False, "no_references", 0, selected)

    primary_occ = detect_occlusion_candidates(originals[0])
    has_occlusion_seed = bool(np.count_nonzero(primary_occ))

    matches = [
        index
        for index, reference in enumerate(originals[1:], start=1)
        if _same_canvas_match(originals[0], reference)
    ]
    if not matches:
        reason = "already_primary_or_no_same_canvas_match" if has_occlusion_seed else "no_same_canvas_match_without_occlusion_seed"
        return PrimaryAnchorDecision(False, reason, 0, selected)

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
            primary_occlusion_seed_present=has_occlusion_seed,
        )
        reason = "already_primary_same_canvas_verified" if has_occlusion_seed else "already_primary_same_canvas_verified_without_occlusion_seed"
        return PrimaryAnchorDecision(False, reason, len(matches), selected)

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
            primary_occlusion_seed_present=has_occlusion_seed,
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
            primary_occlusion_seed_present=has_occlusion_seed,
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
        primary_occlusion_seed_present=has_occlusion_seed,
    )
    return PrimaryAnchorDecision(True, "verified_same_canvas_donor_semantics", len(matches), selected)
