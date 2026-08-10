from __future__ import annotations

"""Keep imported source 0 as the permanent reconstruction target.

Preflight may rank another same-identity photograph as a better analysis/canonical
anchor, but that source is never allowed to replace the user's MAIN canvas, pose or
frame. This wrapper is a defensive contract around preflight so older/internal ranking
logic cannot reintroduce target promotion.
"""

_INSTALLED = False


def _restore_source_zero(workspace) -> bool:
    runtime = [workspace.primary, *workspace.references]
    order_raw = workspace.metadata.get("runtime_source_order")
    if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
        return False
    try:
        order = [int(value) for value in order_raw]
        slot = order.index(0)
    except (TypeError, ValueError):
        return False
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

        # Source 0 is not a preference: it is the product contract. Restore it
        # defensively if any underlying/legacy preflight implementation reordered the
        # runtime sources, while preserving the recommendation for analysis/donor use.
        restored = _restore_source_zero(workspace)
        workspace.metadata["selected_primary_original_source_index"] = 0
        workspace.metadata["preflight_recommended_front_source_index"] = recommended
        workspace.metadata["preflight_analysis_anchor_source_index"] = recommended
        workspace.metadata["preflight_target_canvas_source_index"] = 0
        workspace.metadata["primary_contract"] = {
            "user_selected_primary_original_source_index": 0,
            "maximum_reference_count": 9,
            "preflight_recommended_front_source_index": recommended,
            "analysis_anchor_source_index": recommended,
            "final_primary_original_source_index": 0,
            "user_primary_priority": True,
            "restored_user_primary": bool(restored),
            "promotion_allowed": False,
            "reference_can_be_analysis_anchor": True,
            "reference_can_replace_final_target": False,
        }

        return preflight.PreflightResult(
            selected_source_index=0,
            candidates=result.candidates,
            deblurred_count=result.deblurred_count,
            identity_cluster_size=result.identity_cluster_size,
            reason=(
                "MAIN #1 mantenuta come target; source consigliata conservata soltanto come analysis/donor anchor"
            ),
        )

    preflight.preprocess_and_select_front_base = wrapped
    _INSTALLED = True
