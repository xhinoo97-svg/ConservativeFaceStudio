from __future__ import annotations

import cv2
import numpy as np


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return None
    return np.where(item > 0, 255, 0).astype(np.uint8)


def precise_same_canvas_damage_seed(workspace, shape: tuple[int, int]) -> np.ndarray:
    """Return the narrowest authoritative damage seed available.

    The primary heuristic occlusion proposal is intentionally recall-oriented and can
    cover a large part of a face. Once the inpaint stage has produced a non-empty
    verified target, same-canvas transfer must not OR that target with the broad
    proposal: seed pixels are deliberately never discarded by the expansion limiter,
    so doing so can authorize a large false transfer. The verified inpaint target is
    therefore authoritative. Reference consensus is the next-best source, followed by
    the frozen preflight proposal and finally the live primary occlusion mask.
    """
    current = workspace.metadata.get("inpaint_target_mask")
    if isinstance(current, np.ndarray):
        target = _binary(current, shape)
        if target is not None and np.any(target):
            return target

    consensus = workspace.metadata.get("reference_consensus_occlusion")
    if isinstance(consensus, np.ndarray):
        target = _binary(consensus, shape)
        if target is not None and np.any(target):
            return target

    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        target = _binary(np.asarray(frozen[0]), shape)
        if target is not None and np.any(target):
            return target

    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        target = _binary(np.asarray(masks[0]), shape)
        if target is not None and np.any(target):
            return target

    return np.zeros(shape, dtype=np.uint8)


def _original_reference_for_runtime_slot(workspace, runtime_reference_index: int, fallback: np.ndarray) -> np.ndarray:
    """Read support geometry from the immutable imported photograph when available."""
    try:
        from app.immutable_input_store import ensure_immutable_input_store

        store = ensure_immutable_input_store(workspace)
        order_raw = workspace.metadata.get("runtime_source_order")
        if isinstance(order_raw, list) and runtime_reference_index + 1 < len(order_raw):
            original_source = int(order_raw[runtime_reference_index + 1])
            if original_source > 0 and original_source - 1 < len(store.references):
                candidate = store.copy_reference(original_source - 1)
                if candidate.shape == fallback.shape:
                    return candidate
        if runtime_reference_index < len(store.references):
            candidate = store.copy_reference(runtime_reference_index)
            if candidate.shape == fallback.shape:
                return candidate
    except Exception:
        pass
    return np.asarray(fallback)


def _sparse_canvas_support(image: np.ndarray) -> np.ndarray:
    """Infer geometric support without treating dark photographed pixels as invalid.

    Only exact-zero padding connected to the outer frame is considered absent. Exact
    black pixels enclosed by the photographed patch remain valid evidence whenever the
    surrounding geometry proves that they belong to the patch.
    """
    shape = image.shape[:2]
    exact_zero = np.all(np.asarray(image) == 0, axis=2).astype(np.uint8)
    if not np.any(exact_zero):
        return np.full(shape, 255, dtype=np.uint8)
    _, labels = cv2.connectedComponents(exact_zero, connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    padding_labels = border_labels[border_labels != 0]
    if padding_labels.size == 0:
        return np.full(shape, 255, dtype=np.uint8)
    padding = np.isin(labels, padding_labels)
    support = ~padding

    # Fill small enclosed zero holes so genuine black pupils/hair/shadows are not
    # discarded merely because their RGB value is exactly zero.
    binary = support.astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return binary


def _patch_structure(image: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    active = support > 0
    if int(np.count_nonzero(active)) < 32:
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    texture_std = float(np.std(gray[active]))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    edge_fraction = float(np.mean(magnitude[active] >= 10.0))
    return texture_std, edge_fraction


def _chroma_is_plausible(primary: np.ndarray, reference: np.ndarray, face: np.ndarray, support: np.ndarray, primary_occ: np.ndarray) -> bool:
    visible_face = (face > 0) & (primary_occ == 0)
    donor = support > 0
    if int(np.count_nonzero(visible_face)) < 128 or int(np.count_nonzero(donor)) < 32:
        return False
    main_lab = cv2.cvtColor(primary, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    main_ab = main_lab[visible_face, 1:3]
    donor_ab = ref_lab[donor, 1:3]
    low = np.percentile(main_ab, 1.0, axis=0) - 28.0
    high = np.percentile(main_ab, 99.0, axis=0) + 28.0
    median = np.median(donor_ab, axis=0)
    return bool(np.all(median >= low) and np.all(median <= high))


def _install_sparse_partial_same_canvas_recovery() -> None:
    """Recover a local identity transform when intact baseline overlap is impossible.

    The normal partial verifier remains authoritative whenever it can compare intact
    pixels. This fallback is intentionally narrower: it applies only to sparse images
    already encoded on exactly the MAIN canvas, whose useful support falls mostly on a
    damaged facial ROI. It therefore selects the correct *alignment class* instead of
    relaxing global affine/ORB thresholds.
    """
    from app import case_aware_runtime as case_runtime
    from app.strict_repair import face_support_mask

    original = case_runtime._same_canvas_partial_verification
    if getattr(original, "_sparse_damage_overlap_recovery", False):
        return

    def patched(workspace, reference: np.ndarray, runtime_reference_index: int):
        verified = original(workspace, reference, runtime_reference_index)
        if verified is not None:
            return verified

        primary_working = np.asarray(workspace.primary)
        if reference.shape != primary_working.shape or reference.ndim != 3 or reference.shape[2] != 3:
            return None
        shape = primary_working.shape[:2]
        source = _original_reference_for_runtime_slot(workspace, runtime_reference_index, reference)
        support = _sparse_canvas_support(source)
        support_pixels = int(np.count_nonzero(support))
        support_fraction = float(support_pixels / max(1, support.size))
        if support_fraction < 0.012 or support_fraction > 0.58:
            return None

        bbox_raw = workspace.metadata.get("primary_bbox")
        if not isinstance(bbox_raw, (tuple, list)) or len(bbox_raw) != 4:
            return None
        try:
            bbox = tuple(int(value) for value in bbox_raw)
        except (TypeError, ValueError):
            return None
        face = face_support_mask(shape, bbox)
        inside_face = int(np.count_nonzero((support > 0) & (face > 0)))
        face_fraction = float(inside_face / max(1, support_pixels))
        if face_fraction < 0.72:
            return None

        frozen = workspace.metadata.get("preflight_original_occlusion_masks")
        primary_occ = np.zeros(shape, dtype=np.uint8)
        if isinstance(frozen, list) and frozen:
            candidate = _binary(np.asarray(frozen[0]), shape)
            if candidate is not None:
                primary_occ = candidate
        damage_overlap_pixels = int(np.count_nonzero((support > 0) & (primary_occ > 0)))
        damage_overlap_fraction = float(damage_overlap_pixels / max(1, support_pixels))

        # Component-local sheets may include visible context, while complementary
        # severe sheets can be almost entirely inside the damaged ROI. Accept either
        # situation only when the support is anatomically localised on the MAIN canvas.
        component_overlap = 0.0
        points = workspace.metadata.get("primary_landmarks5")
        if points is not None:
            try:
                masks = case_runtime.canonical_component_masks(
                    shape,
                    np.asarray(points, dtype=np.float32),
                    bbox,
                )
                component_overlap = max(
                    (
                        float(np.count_nonzero((support > 0) & (mask > 0)) / max(1, support_pixels))
                        for mask in masks.values()
                    ),
                    default=0.0,
                )
            except Exception:
                component_overlap = 0.0
        if damage_overlap_fraction < 0.55 and not (
            damage_overlap_fraction >= 0.08 and component_overlap >= 0.40
        ):
            return None

        texture_std, edge_fraction = _patch_structure(source, support)
        if texture_std < 4.0 and edge_fraction < 0.012:
            return None

        try:
            from app.immutable_input_store import ensure_immutable_input_store

            primary_original = ensure_immutable_input_store(workspace).copy_main()
            if primary_original.shape != primary_working.shape:
                primary_original = primary_working
        except Exception:
            primary_original = primary_working
        if not _chroma_is_plausible(primary_original, source, face, support, primary_occ):
            return None

        reliability_maps = workspace.metadata.get("preflight_detail_reliability_maps")
        reliability = np.zeros(shape, dtype=np.uint8)
        if isinstance(reliability_maps, list) and runtime_reference_index + 1 < len(reliability_maps):
            candidate = np.asarray(reliability_maps[runtime_reference_index + 1])
            if candidate.shape == shape:
                reliability = candidate.astype(np.uint8, copy=True)
        # A verified photographed patch remains evidence even if a generic whole-image
        # quality estimator was confused by the surrounding zero canvas. Keep its real
        # ranking conservative rather than upgrading it to maximum confidence.
        reliability[support > 0] = np.maximum(reliability[support > 0], np.uint8(96))
        reliability[support == 0] = 0

        geometry_confidence = float(np.clip(
            0.55
            + 0.22 * min(1.0, damage_overlap_fraction)
            + 0.15 * min(1.0, component_overlap / 0.6)
            + 0.08 * min(1.0, edge_fraction / 0.08),
            0.0,
            0.92,
        ))
        return support, reliability, {
            "runtime_reference_index": int(runtime_reference_index),
            "observed_fraction": support_fraction,
            "reliable_pixels": support_pixels,
            "comparable_pixels": 0,
            "median_lab_delta": None,
            "p90_lab_delta": None,
            "method": "verified-same-canvas-partial",
            "verification_basis": "coordinate-preserving-sparse-damage-overlap",
            "global_transform_required": False,
            "local_identity_transform": True,
            "damage_overlap_fraction": damage_overlap_fraction,
            "face_support_fraction": face_fraction,
            "component_overlap_fraction": component_overlap,
            "texture_std": texture_std,
            "edge_fraction": edge_fraction,
            "geometry_confidence": geometry_confidence,
            "black_pixels_preserved_by_geometric_support": True,
        }

    patched._sparse_damage_overlap_recovery = True  # type: ignore[attr-defined]
    case_runtime._same_canvas_partial_verification = patched


def install_same_canvas_seed_precision_policy() -> None:
    """Install precise repair seeds and partial same-canvas geometry recovery."""
    from app import same_canvas_repair_runtime as runtime

    runtime._damage_seed = precise_same_canvas_damage_seed
    _install_sparse_partial_same_canvas_recovery()
