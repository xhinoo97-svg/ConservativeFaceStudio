from __future__ import annotations

import cv2
import numpy as np


_INSTALLED = False


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Maschera non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _trusted_reference_disagreement(workspace, frozen: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Validate frozen damage seeds against trusted observed references.

    Full/same-canvas donors with enough unaffected overlap must pass a baseline
    consistency guard before they may confirm a seed.  A trusted component-only donor
    can legitimately have no such baseline.  In that case it may preserve only the
    already-detected seed pixels that lie inside its explicit observed support; it may
    never create or expand a damage region by itself.
    """
    from app.observed_target_repair_runtime import _trusted_slots

    shape = workspace.primary.shape[:2]
    aligned = list(getattr(workspace, "aligned_references", []) or [])
    supports = workspace.metadata.get("aligned_reference_support_masks")
    if not aligned or not isinstance(supports, list) or len(supports) != len(aligned):
        return np.zeros(shape, np.uint8), {"trusted_donors": 0, "reason": "no_aligned_support"}

    trusted_flags = _trusted_slots(workspace, len(aligned))
    if not any(trusted_flags):
        return np.zeros(shape, np.uint8), {
            "trusted_donors": 0,
            "rejected_donors": len(aligned),
            "refined_pixels": 0,
            "reason": "no_identity_or_geometry_trusted_reference",
        }

    frozen_primary = workspace.metadata.get("same_canvas_imported_primary")
    if not isinstance(frozen_primary, np.ndarray) or frozen_primary.shape != workspace.primary.shape:
        frozen_primary = workspace.primary
    primary_lab = cv2.cvtColor(frozen_primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    frozen_bool = frozen > 0
    exclusion = cv2.dilate(
        frozen,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    ) > 0

    refined = np.zeros(shape, dtype=np.uint8)
    trusted = 0
    rejected = 0
    seed_only_pixels = 0
    baseline_confirmed_pixels = 0
    diagnostics: list[dict[str, object]] = []

    for slot, (reference, support_raw, slot_trusted) in enumerate(zip(aligned, supports, trusted_flags)):
        if not slot_trusted:
            diagnostics.append({"slot": slot, "accepted": False, "reason": "untrusted_identity_or_geometry"})
            rejected += 1
            continue
        if reference.shape != workspace.primary.shape:
            diagnostics.append({"slot": slot, "accepted": False, "reason": "shape_mismatch"})
            rejected += 1
            continue

        support = _binary(np.asarray(support_raw), shape) > 0
        supported_seed = support & frozen_bool
        if not np.any(supported_seed):
            diagnostics.append({"slot": slot, "accepted": False, "reason": "no_seed_support_overlap"})
            rejected += 1
            continue

        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        delta = np.mean(np.abs(primary_lab - ref_lab), axis=2)
        baseline_region = support & ~exclusion
        baseline_count = int(np.count_nonzero(baseline_region))

        # Component-only references often observe only the missing eye/nose/mouth and
        # therefore cannot prove a same-canvas baseline.  They are still useful as
        # observed evidence, but only for an existing detector seed and never for seed
        # expansion.  This is the conservative analogue of reference-guided inpainting:
        # reference evidence controls the missing region rather than inventing a new one.
        if baseline_count < 96:
            local = np.where(supported_seed, 255, 0).astype(np.uint8)
            refined = cv2.bitwise_or(refined, local)
            count = int(np.count_nonzero(local))
            seed_only_pixels += count
            trusted += 1
            diagnostics.append({
                "slot": slot,
                "accepted": True,
                "reason": "trusted_partial_seed_only_no_baseline",
                "baseline_pixels": baseline_count,
                "seed_pixels": count,
                "may_expand_seed": False,
            })
            continue

        baseline = delta[baseline_region]
        median = float(np.median(baseline))
        p90 = float(np.percentile(baseline, 90.0))
        if median > 0.055 or p90 > 0.14:
            diagnostics.append({
                "slot": slot,
                "accepted": False,
                "reason": "baseline_mismatch",
                "baseline_pixels": baseline_count,
                "baseline_median": median,
                "baseline_p90": p90,
            })
            rejected += 1
            continue

        # A detector seed needs only exceed the measured unaffected noise floor; the
        # previous hard 0.035 floor discarded faint/semipermeable overlays.  The lower
        # bound stays above ordinary compression noise and is still donor-baseline gated.
        noise_floor = max(0.018, p90 + 0.010, median + 0.015)
        disagreement = supported_seed & (delta >= noise_floor)
        if np.any(disagreement):
            local = np.where(disagreement, 255, 0).astype(np.uint8)
            local = cv2.morphologyEx(
                local,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            local = cv2.bitwise_and(local, np.where(supported_seed, 255, 0).astype(np.uint8))
            refined = cv2.bitwise_or(refined, local)
            baseline_confirmed_pixels += int(np.count_nonzero(local))
        trusted += 1
        diagnostics.append({
            "slot": slot,
            "accepted": True,
            "reason": "baseline_verified_seed_confirmation",
            "baseline_pixels": baseline_count,
            "baseline_median": median,
            "baseline_p90": p90,
            "difference_threshold": float(noise_floor),
            "seed_pixels": int(np.count_nonzero(disagreement)),
            "may_expand_seed": False,
        })

    refined_pixels = int(np.count_nonzero(refined))
    return refined, {
        "trusted_donors": trusted,
        "rejected_donors": rejected,
        "refined_pixels": refined_pixels,
        "trusted_partial_seed_only_pixels": int(seed_only_pixels),
        "baseline_confirmed_seed_pixels": int(baseline_confirmed_pixels),
        "diagnostics": diagnostics,
        "reason": "reference_guided_frozen_seed" if refined_pixels else "no_trusted_supported_seed",
        "seed_expansion_from_partial_reference": False,
    }


def install_reference_guided_seed_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.partial_reference_runtime as module

    original_frozen = module._frozen_primary_occlusion
    original_merge = module._merge_frozen_primary_hint

    def precise_merge(workspace) -> int:
        frozen = original_frozen(workspace)
        if frozen is None or not np.any(frozen):
            workspace.metadata["reference_guided_seed_diagnostics"] = {
                "trusted_donors": 0,
                "refined_pixels": 0,
                "reason": "no_frozen_proposal",
            }
            return 0

        aligned = list(getattr(workspace, "aligned_references", []) or [])
        supports = workspace.metadata.get("aligned_reference_support_masks")
        if not aligned or not isinstance(supports, list) or len(supports) != len(aligned):
            added = int(original_merge(workspace))
            workspace.metadata["reference_guided_seed_diagnostics"] = {
                "trusted_donors": 0,
                "refined_pixels": 0,
                "reason": "no_aligned_support_legacy_frozen_seed_preserved",
                "fallback_added_pixels": added,
            }
            return added

        shape = workspace.primary.shape[:2]
        refined, diagnostics = _trusted_reference_disagreement(workspace, frozen)
        existing = workspace.metadata.get("reference_consensus_occlusion")
        if not isinstance(existing, np.ndarray) or existing.shape != shape:
            existing = np.zeros(shape, dtype=np.uint8)
        merged = cv2.bitwise_or(_binary(existing, shape), refined)
        added = int(np.count_nonzero((merged > 0) & (existing == 0)))
        workspace.metadata["reference_consensus_occlusion"] = merged
        workspace.metadata["reference_guided_seed_diagnostics"] = diagnostics
        return added

    module._merge_frozen_primary_hint = precise_merge
    _INSTALLED = True
