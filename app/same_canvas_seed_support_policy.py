from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.strict_repair import face_support_mask


def exact_same_canvas_observed_repair_seed_support(
    workspace,
    image: np.ndarray,
    *,
    difference_threshold: float = 0.075,
    maximum_face_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reference-driven same-canvas repair with authoritative observed support.

    Verified damage seeds are allowed wherever a verified aligned reference explicitly
    observes the pixel. Face geometry constrains only evidence expansion beyond those
    seeds; it never vetoes an observed damaged pixel. Pixel intensity is never used as
    a proxy for whether a donor was photographed.
    """
    from app import same_canvas_repair_runtime as base

    shape = workspace.primary.shape[:2]
    aligned = list(workspace.aligned_references)
    runtime_indices_raw = workspace.metadata.get("aligned_reference_source_indices")
    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    if not aligned:
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_aligned_references", "repaired_pixels": 0}
    if not isinstance(runtime_indices_raw, list) or len(runtime_indices_raw) != len(aligned):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "missing_runtime_source_mapping", "repaired_pixels": 0}
    if not isinstance(supports_raw, list) or len(supports_raw) != len(aligned):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "missing_observed_support", "repaired_pixels": 0}

    runtime_indices = [int(value) for value in runtime_indices_raw]
    originals = base._original_source_indices(workspace, runtime_indices, len(aligned))
    verified_runtime, verified_original = base._verified_donor_slots(workspace, runtime_indices, originals)
    verified_slots = [
        runtime_indices[index] in verified_runtime or originals[index] in verified_original
        for index in range(len(aligned))
    ]
    if not any(verified_slots):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False,
            "reason": "no_verified_same_canvas_reference",
            "repaired_pixels": 0,
            "verified_runtime_indices": sorted(verified_runtime),
            "verified_original_indices": sorted(verified_original),
        }

    seed = base._damage_seed(workspace, shape)
    if not np.any(seed):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_observed_damage_seed", "repaired_pixels": 0}

    bbox_raw = workspace.metadata.get("primary_bbox")
    bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
    face = face_support_mask(shape, bbox) > 0
    face_pixels = max(1, int(np.count_nonzero(face)))
    maximum_pixels = max(0, int(round(face_pixels * float(maximum_face_fraction))))

    # A verified damage seed plus a verified donor support mask is stronger evidence
    # than an approximate face-support template. This also covers jaw/hair-edge damage.
    seed_bool = seed > 0
    expansion_geometry = face | seed_bool
    seed_reach = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

    frozen_primary = base._frozen_primary(workspace)
    base_lab = cv2.cvtColor(frozen_primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

    result = image.copy()
    provenance = np.zeros(shape, dtype=np.uint16)
    repaired_union = np.zeros(shape, dtype=bool)
    source_counts: dict[int, int] = {}
    seed_pixel_count = 0
    expanded_pixel_count = 0
    hysteresis_pixel_count = 0
    flat_occluder_pixel_count = 0
    unseeded_strong_pixel_count = 0
    threshold_diagnostics: list[dict[str, Any]] = []

    for slot, (reference, support_raw, original_index) in enumerate(zip(aligned, supports_raw, originals)):
        if not verified_slots[slot] or reference.shape != workspace.primary.shape:
            continue
        support = base._binary(np.asarray(support_raw), shape) > 0
        # The support mask is the authoritative evidence of photographed pixels.
        # Never infer validity from RGB intensity: true hair/pupil pixels can be black.
        observed = support & expansion_geometry
        if not np.any(observed):
            continue

        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        difference = np.mean(np.abs(base_lab - ref_lab), axis=2)
        adaptive_threshold, baseline_stats = base._adaptive_difference_threshold(
            difference,
            observed,
            seed_bool,
            float(difference_threshold),
        )
        strong = observed & (difference >= adaptive_threshold)
        expansion, unseeded_pixels = base._strong_components(strong, seed_reach, difference, adaptive_threshold)
        expansion &= observed & ~repaired_union
        unseeded_strong_pixel_count += int(unseeded_pixels)

        seeded_observed = seed_bool & observed & ~repaired_union
        verified_envelope = base._filled_component(expansion | seeded_observed)
        verified_envelope &= observed & ~repaired_union
        weak_threshold = max(
            float(baseline_stats.get("baseline_p95", 0.0)) + 0.006,
            float(adaptive_threshold) * 0.35,
        )
        hysteresis_envelope = base._seed_connected_hysteresis(
            verified_envelope,
            difference,
            observed & ~repaired_union,
            weak_threshold=weak_threshold,
        )
        flat_envelope, flat_tolerance = base._seed_connected_flat_occluder(
            frozen_primary,
            seed_bool,
            observed & ~repaired_union,
        )
        combined_envelope = (hysteresis_envelope | flat_envelope) & observed & ~repaired_union
        hysteresis_pixel_count += int(np.count_nonzero(hysteresis_envelope & ~verified_envelope))
        flat_occluder_pixel_count += int(np.count_nonzero(flat_envelope & ~hysteresis_envelope))
        threshold_diagnostics.append({
            "slot": int(slot),
            "runtime_reference_index": int(runtime_indices[slot]),
            "original_source_index": int(original_index),
            "adaptive_difference_threshold": float(adaptive_threshold),
            "hysteresis_weak_threshold": float(weak_threshold),
            "flat_occluder_colour_tolerance": float(flat_tolerance),
            **baseline_stats,
        })

        selected = base._limit_expansion(
            seeded_observed,
            combined_envelope,
            difference,
            maximum_pixels - int(np.count_nonzero(repaired_union)),
        )
        selected &= ~repaired_union
        if not np.any(selected):
            continue

        result[selected] = reference[selected]
        code = np.uint16(max(1, int(original_index)))
        provenance[selected] = code
        repaired_union |= selected
        count = int(np.count_nonzero(selected))
        source_counts[int(code)] = source_counts.get(int(code), 0) + count
        seed_pixel_count += int(np.count_nonzero(selected & seeded_observed))
        expanded_pixel_count += int(np.count_nonzero(selected & ~seeded_observed))

    repaired_pixels = int(np.count_nonzero(repaired_union))
    return result, provenance, {
        "applied": repaired_pixels > 0,
        "reason": "exact_observed_transfer" if repaired_pixels else "no_seeded_observed_or_strong_difference",
        "verified_reference_count": int(sum(verified_slots)),
        "verified_runtime_indices": sorted(verified_runtime),
        "verified_original_indices": sorted(verified_original),
        "repaired_pixels": repaired_pixels,
        "seed_repaired_pixels": int(seed_pixel_count),
        "expanded_repaired_pixels": int(expanded_pixel_count),
        "hysteresis_recovered_pixels": int(hysteresis_pixel_count),
        "flat_occluder_recovered_pixels": int(flat_occluder_pixel_count),
        "unseeded_strong_component_pixels": int(unseeded_strong_pixel_count),
        "source_pixel_counts": source_counts,
        "difference_threshold_ceiling": float(difference_threshold),
        "threshold_diagnostics": threshold_diagnostics,
        "maximum_face_fraction": float(maximum_face_fraction),
        "difference_anchor": "frozen_imported_primary" if isinstance(workspace.metadata.get("same_canvas_imported_primary"), np.ndarray) else "runtime_primary_fallback",
        "seed_pixels_are_never_discarded_by_expansion_cap": True,
        "verified_seed_support_overrides_face_template": True,
        "support_mask_is_authoritative_for_donor_validity": True,
        "partial_same_canvas_supported": True,
        "interpolation": "none",
        "generated_pixels": 0,
    }


def install_same_canvas_seed_support_policy() -> None:
    """Install the policy before executors capture the same-canvas repair handler."""
    from app import same_canvas_repair_runtime as runtime

    runtime.exact_same_canvas_observed_repair = exact_same_canvas_observed_repair_seed_support
