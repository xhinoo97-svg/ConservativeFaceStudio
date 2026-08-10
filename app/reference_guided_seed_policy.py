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
    """Return donor-supported disagreement only when same-canvas baseline is plausible."""
    shape = workspace.primary.shape[:2]
    aligned = list(getattr(workspace, "aligned_references", []) or [])
    supports = workspace.metadata.get("aligned_reference_support_masks")
    if not aligned or not isinstance(supports, list) or len(supports) != len(aligned):
        return np.zeros(shape, np.uint8), {"trusted_donors": 0, "reason": "no_aligned_support"}

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
    diagnostics: list[dict[str, object]] = []
    for slot, (reference, support_raw) in enumerate(zip(aligned, supports)):
        if reference.shape != workspace.primary.shape:
            rejected += 1
            continue
        support = _binary(np.asarray(support_raw), shape) > 0
        baseline_region = support & ~exclusion
        if np.count_nonzero(baseline_region) < 96:
            diagnostics.append({"slot": slot, "accepted": False, "reason": "insufficient_baseline"})
            rejected += 1
            continue

        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        delta = np.mean(np.abs(primary_lab - ref_lab), axis=2)
        baseline = delta[baseline_region]
        median = float(np.median(baseline))
        p90 = float(np.percentile(baseline, 90.0))
        if median > 0.055 or p90 > 0.14:
            diagnostics.append({
                "slot": slot,
                "accepted": False,
                "reason": "baseline_mismatch",
                "baseline_median": median,
                "baseline_p90": p90,
            })
            rejected += 1
            continue

        noise_floor = max(0.035, p90 + 0.018, median + 0.025)
        disagreement = support & frozen_bool & (delta >= noise_floor)
        if np.any(disagreement):
            local = np.where(disagreement, 255, 0).astype(np.uint8)
            local = cv2.morphologyEx(
                local,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            refined = cv2.bitwise_or(refined, local)
        trusted += 1
        diagnostics.append({
            "slot": slot,
            "accepted": True,
            "baseline_median": median,
            "baseline_p90": p90,
            "difference_threshold": float(noise_floor),
            "seed_pixels": int(np.count_nonzero(disagreement)),
        })

    return refined, {
        "trusted_donors": trusted,
        "rejected_donors": rejected,
        "refined_pixels": int(np.count_nonzero(refined)),
        "diagnostics": diagnostics,
        "reason": "reference_guided_frozen_seed" if trusted else "no_same_canvas_baseline",
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
