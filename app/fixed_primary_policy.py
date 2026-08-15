from __future__ import annotations

"""Keep image #1 as the immutable project primary during preflight.

Preflight may still score every imported photo for identity, pose and quality, but a
reference is never promoted to primary. The best ranked source remains available as an
analysis/donor anchor without changing the user's target canvas.
"""

from functools import wraps


_INSTALLED = False


def install_fixed_primary_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as module

    original = module.preprocess_and_select_front_base

    @wraps(original)
    def patched(workspace, model_paths):
        result = original(workspace, model_paths)
        runtime = [workspace.primary, *workspace.references]
        order_raw = workspace.metadata.get("runtime_source_order")
        if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
            workspace.metadata["fixed_primary_policy"] = {
                "applied": False,
                "reason": "runtime_source_order_missing",
            }
            return result

        try:
            order = [int(value) for value in order_raw]
            primary_slot = order.index(0)
        except (TypeError, ValueError):
            workspace.metadata["fixed_primary_policy"] = {
                "applied": False,
                "reason": "imported_primary_missing",
            }
            return result

        if primary_slot != 0:
            slots = [primary_slot, *[index for index in range(len(runtime)) if index != primary_slot]]
            runtime = [runtime[index].copy() for index in slots]
            order = [order[index] for index in slots]
            workspace.primary = runtime[0]
            workspace.references = runtime[1:]
            workspace.metadata["runtime_source_order"] = order
            for key in ("preflight_original_occlusion_masks", "preflight_detail_reliability_maps"):
                values = workspace.metadata.get(key)
                if isinstance(values, list) and len(values) == len(slots):
                    workspace.metadata[key] = [values[index] for index in slots]

        # The inner absolute contract returns selected_source_index=0 by design. Keep
        # the independently recorded recommendation rather than accidentally erasing it.
        recommended_raw = workspace.metadata.get(
            "preflight_recommended_front_source_index",
            workspace.metadata.get("preflight_analysis_anchor_source_index", result.selected_source_index),
        )
        try:
            recommended = int(recommended_raw)
        except (TypeError, ValueError):
            recommended = 0
        if recommended < 0 or recommended >= len(runtime):
            recommended = 0

        workspace.metadata["best_reference_source_index"] = recommended
        workspace.metadata["preflight_analysis_anchor_source_index"] = recommended
        workspace.metadata["selected_primary_original_source_index"] = 0
        workspace.metadata["fixed_primary_policy"] = {
            "applied": True,
            "primary_original_source_index": 0,
            "best_reference_or_primary_by_preflight": recommended,
            "reference_count": len(workspace.references),
            "target_replaced": False,
        }

        return module.PreflightResult(
            selected_source_index=0,
            candidates=result.candidates,
            deblurred_count=result.deblurred_count,
            identity_cluster_size=result.identity_cluster_size,
            reason="foto #1 mantenuta come primary; ranking preflight usato solo per analysis/donor",
        )

    module.preprocess_and_select_front_base = patched
    _INSTALLED = True
