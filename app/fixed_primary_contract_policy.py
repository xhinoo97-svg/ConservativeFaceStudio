from __future__ import annotations

"""Prefer the user-selected primary while permitting a strongly better verified base.

Source 0 is the user's target and therefore receives a strong prior. Preflight may
promote another same-identity photograph only when the measured gain is large enough
to justify changing the reconstruction canvas. Small ranking noise never overrides
user intent. The full decision is preserved in metadata.
"""

import math

_INSTALLED = False


def _candidate(result, source_index: int):
    for item in result.candidates:
        if int(item.source_index) == int(source_index):
            return item
    return None


def _promotion_is_strong(primary, challenger) -> tuple[bool, dict[str, object]]:
    if primary is None or challenger is None:
        return False, {"reason": "candidate_metrics_missing"}
    if not bool(challenger.accepted_identity):
        return False, {"reason": "challenger_identity_not_verified"}

    q0 = float(primary.quality)
    q1 = float(challenger.quality)
    f0 = float(primary.frontalness)
    f1 = float(challenger.frontalness)
    pose0 = math.isfinite(f0) and f0 < 1e5
    pose1 = math.isfinite(f1) and f1 < 1e5
    quality_gain = q1 - q0
    frontal_ratio = f1 / max(1e-6, f0) if pose0 and pose1 else 1.0

    promote = False
    reason = "user_primary_preferred"
    if not bool(primary.accepted_identity) and bool(challenger.accepted_identity):
        promote = True
        reason = "primary_identity_analysis_failed"
    elif pose1 and not pose0 and quality_gain >= -0.03:
        promote = True
        reason = "verified_pose_available_only_on_challenger"
    elif pose0 and pose1 and frontal_ratio <= 0.72 and quality_gain >= -0.01:
        promote = True
        reason = "material_frontalness_gain"
    elif pose0 and pose1 and frontal_ratio <= 0.82 and quality_gain >= 0.10:
        promote = True
        reason = "frontalness_and_quality_gain"
    elif not pose0 and not pose1 and quality_gain >= 0.18:
        promote = True
        reason = "very_large_quality_gain"

    return promote, {
        "reason": reason,
        "primary_quality": q0,
        "challenger_quality": q1,
        "quality_gain": quality_gain,
        "primary_frontalness": f0,
        "challenger_frontalness": f1,
        "frontalness_ratio": frontal_ratio,
        "promoted": promote,
    }


def _restore_source_zero(workspace) -> bool:
    runtime = [workspace.primary, *workspace.references]
    order_raw = workspace.metadata.get("runtime_source_order")
    if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
        return False
    order = [int(value) for value in order_raw]
    if 0 not in order:
        return False
    slot = order.index(0)
    if slot == 0:
        return True

    slots = [slot, *[i for i in range(len(runtime)) if i != slot]]
    restored = [runtime[i].copy() for i in slots]
    restored_order = [order[i] for i in slots]
    workspace.primary = restored[0]
    workspace.references = restored[1:]
    workspace.metadata["runtime_source_order"] = restored_order

    for key in ("preflight_original_occlusion_masks", "preflight_detail_reliability_maps"):
        values = workspace.metadata.get(key)
        if isinstance(values, list) and len(values) == len(runtime):
            workspace.metadata[key] = [values[i] for i in slots]
    return True


def install_fixed_primary_contract_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as preflight

    original = preflight.preprocess_and_select_front_base

    def wrapped(workspace, model_paths):
        result = original(workspace, model_paths)
        recommended = int(getattr(result, "selected_source_index", 0))
        user_primary = _candidate(result, 0)
        challenger = _candidate(result, recommended)
        promote, diagnostics = _promotion_is_strong(user_primary, challenger)

        user_selected = bool(workspace.metadata.get("user_selected_primary", True))
        final_source = recommended
        restored = False
        if user_selected and recommended != 0 and not promote:
            restored = _restore_source_zero(workspace)
            if restored:
                final_source = 0
                workspace.metadata["selected_primary_original_source_index"] = 0

        workspace.metadata["preflight_recommended_front_source_index"] = recommended
        workspace.metadata["primary_contract"] = {
            "user_selected_primary_original_source_index": 0,
            "maximum_reference_count": 9,
            "preflight_recommended_front_source_index": recommended,
            "final_primary_original_source_index": final_source,
            "user_primary_priority": True,
            "restored_user_primary": restored,
            "promotion_allowed": bool(promote),
            "promotion_diagnostics": diagnostics,
        }

        if final_source == recommended:
            return preflight.PreflightResult(
                selected_source_index=final_source,
                candidates=result.candidates,
                deblurred_count=result.deblurred_count,
                identity_cluster_size=result.identity_cluster_size,
                reason=(
                    result.reason
                    if final_source == 0
                    else "reference promossa a base: vantaggio frontale/qualitativo forte e verificato"
                ),
            )

        return preflight.PreflightResult(
            selected_source_index=0,
            candidates=result.candidates,
            deblurred_count=result.deblurred_count,
            identity_cluster_size=result.identity_cluster_size,
            reason="mantenuta la primary scelta dall'utente; vantaggio della reference insufficiente per sostituirla",
        )

    preflight.preprocess_and_select_front_base = wrapped
    _INSTALLED = True
