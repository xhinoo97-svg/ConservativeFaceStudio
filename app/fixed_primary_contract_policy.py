from __future__ import annotations

"""Keep the first imported photograph as the target image.

Preflight may rank another source as a better frontal donor, but the product contract
is explicit: source 0 is always the primary to reconstruct and sources 1..9 are
references.  We preserve all preprocessing results and only restore source ordering.
"""

_INSTALLED = False


def install_fixed_primary_contract_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as preflight

    original = preflight.preprocess_and_select_front_base

    def wrapped(workspace, model_paths):
        result = original(workspace, model_paths)
        runtime = [workspace.primary, *workspace.references]
        order_raw = workspace.metadata.get("runtime_source_order")
        if not isinstance(order_raw, list) or len(order_raw) != len(runtime):
            order = list(range(len(runtime)))
        else:
            order = [int(value) for value in order_raw]

        # Preserve the preflight recommendation for diagnostics/donor ranking only.
        recommended = int(getattr(result, "selected_source_index", 0))
        workspace.metadata["preflight_recommended_front_source_index"] = recommended

        if 0 in order:
            slots = sorted(range(len(runtime)), key=lambda slot: order[slot])
            restored = [runtime[slot].copy() for slot in slots]
            restored_order = [order[slot] for slot in slots]
            workspace.primary = restored[0]
            workspace.references = restored[1:]
            workspace.metadata["runtime_source_order"] = restored_order

            for key in ("preflight_original_occlusion_masks", "preflight_detail_reliability_maps"):
                values = workspace.metadata.get(key)
                if isinstance(values, list) and len(values) == len(runtime):
                    workspace.metadata[key] = [values[slot] for slot in slots]

        workspace.metadata["selected_primary_original_source_index"] = 0
        workspace.metadata["primary_contract"] = {
            "primary_original_source_index": 0,
            "maximum_reference_count": 9,
            "preflight_recommended_front_source_index": recommended,
            "reference_order_preserved": True,
        }

        # Keep the public result consistent with the product contract while retaining
        # the original candidate ranking in metadata.
        return preflight.PreflightResult(
            selected_source_index=0,
            candidates=result.candidates,
            deblurred_count=result.deblurred_count,
            identity_cluster_size=result.identity_cluster_size,
            reason=(
                "prima foto mantenuta come primary; ranking frontale usato soltanto per donor selection"
            ),
        )

    preflight.preprocess_and_select_front_base = wrapped
    _INSTALLED = True
