from __future__ import annotations

from dataclasses import replace

import numpy as np

_INSTALLED = False


def _candidate_by_source(result, source_index: int):
    for candidate in getattr(result, "candidates", ()):
        if int(candidate.source_index) == int(source_index):
            return candidate
    return None


def _strongly_better(primary, challenger) -> tuple[bool, dict[str, float | bool | str]]:
    """Require a material, explainable advantage before overriding user intent."""
    if primary is None or challenger is None:
        return False, {"reason": "missing_candidate_metrics"}
    if not bool(challenger.accepted_identity):
        return False, {"reason": "challenger_identity_not_verified"}

    q0 = float(primary.quality)
    q1 = float(challenger.quality)
    f0 = float(primary.frontalness)
    f1 = float(challenger.frontalness)

    primary_pose_valid = np.isfinite(f0) and f0 < 1e5
    challenger_pose_valid = np.isfinite(f1) and f1 < 1e5
    quality_gain = q1 - q0
    frontal_gain = (f0 - f1) if primary_pose_valid and challenger_pose_valid else 0.0
    frontal_ratio = (f1 / max(1e-6, f0)) if primary_pose_valid and challenger_pose_valid else 1.0

    # Normal case: do not replace the user-selected base for small metric noise.
    # Promotion requires a clearly more frontal pose AND no quality loss, or a very
    # large quality gain accompanied by a meaningful pose improvement.
    promote = False
    reason = "user_primary_preferred"
    if not bool(primary.accepted_identity) and bool(challenger.accepted_identity):
        promote = True
        reason = "primary_identity_analysis_failed"
    elif challenger_pose_valid and not primary_pose_valid and quality_gain >= -0.03:
        promote = True
        reason = "primary_pose_unavailable_challenger_verified"
    elif primary_pose_valid and challenger_pose_valid:
        if frontal_ratio <= 0.72 and quality_gain >= -0.01:
            promote = True
            reason = "material_frontalness_gain"
        elif frontal_ratio <= 0.82 and quality_gain >= 0.10:
            promote = True
            reason = "frontalness_and_quality_gain"
    elif quality_gain >= 0.18:
        promote = True
        reason = "very_large_quality_gain"

    return promote, {
        "reason": reason,
        "primary_quality": q0,
        "challenger_quality": q1,
        "quality_gain": quality_gain,
        "primary_frontalness": f0,
        "challenger_frontalness": f1,
        "frontalness_gain": frontal_gain,
        "frontalness_ratio": frontal_ratio,
        "promoted": promote,
    }


def _restore_user_primary(workspace, result):
    order_raw = workspace.metadata.get("runtime_source_order")
    runtime = [workspace.primary, *workspace.references]
    if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
        return result, False
    order = [int(value) for value in order_raw]
    if 0 not in order:
        return result, False
    slot = order.index(0)
    if slot == 0:
        return result, False

    new_slots = [slot, *[i for i in range(len(runtime)) if i != slot]]
    reordered = [runtime[i].copy() for i in new_slots]
    new_order = [order[i] for i in new_slots]
    workspace.primary = reordered[0]
    workspace.references = reordered[1:]
    workspace.metadata["runtime_source_order"] = new_order
    workspace.metadata["selected_primary_original_source_index"] = 0

    for key in ("preflight_original_occlusion_masks", "preflight_detail_reliability_maps"):
        values = workspace.metadata.get(key)
        if isinstance(values, list) and len(values) == len(runtime):
            workspace.metadata[key] = [values[i] for i in new_slots]

    return replace(
        result,
        selected_source_index=0,
        reason="mantenuta la foto principale scelta dall'utente; nessun vantaggio verificato sufficiente per sostituirla",
    ), True


def install_user_primary_preference_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.automatic as automatic

    original = automatic.preprocess_and_select_front_base

    def preferred_preflight(workspace, model_paths):
        result = original(workspace, model_paths)
        if not bool(workspace.metadata.get("user_selected_primary", False)):
            return result

        selected = int(result.selected_source_index)
        primary = _candidate_by_source(result, 0)
        challenger = _candidate_by_source(result, selected)
        promoted, diagnostics = _strongly_better(primary, challenger)
        diagnostics = dict(diagnostics)
        diagnostics["user_selected_primary"] = True
        diagnostics["proposed_source_index"] = selected

        if selected != 0 and not promoted:
            result, restored = _restore_user_primary(workspace, result)
            diagnostics["restored_user_primary"] = bool(restored)
            diagnostics["final_source_index"] = 0 if restored else selected
        else:
            diagnostics["restored_user_primary"] = False
            diagnostics["final_source_index"] = selected

        workspace.metadata["primary_preference_decision"] = diagnostics
        return result

    automatic.preprocess_and_select_front_base = preferred_preflight
    _INSTALLED = True
