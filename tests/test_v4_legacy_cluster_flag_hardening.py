from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.identity_anchor_v4_hardening import _harden_bridge_result


def _workspace(*, with_matrix: bool = True):
    metadata = {
        "runtime_source_order": [0, 1, 2],
        "reference_identity_verified": [True, True],
        "reference_identity_reasons": ["direct_sface", "main_bridged_cross_reference_cluster"],
        "reference_identity_scores": [0.50, 0.20],
    }
    if with_matrix:
        metadata["preflight_identity_similarity"] = {
            "source_indices": [0, 1, 2],
            "matrix": [
                [1.0, 0.50, 0.20],
                [0.50, 1.0, 0.50],
                [0.20, 0.50, 1.0],
            ],
            "minimum": 0.363,
            "source": "preflight_existing_sface_embeddings",
        }
    return SimpleNamespace(
        references=[np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8)],
        metadata=metadata,
    )


def _legacy_bridge(workspace):
    return (
        list(workspace.metadata["reference_identity_verified"]),
        list(workspace.metadata["reference_identity_reasons"]),
        {1, 2},
    )


def test_legacy_single_link_promotion_is_revoked_but_direct_sface_survives() -> None:
    workspace = _workspace()
    flags, reasons, trusted = _harden_bridge_result(workspace, _legacy_bridge)

    assert flags == [True, False]
    assert reasons == ["direct_sface", "rejected_transitive_component_only"]
    assert trusted == {1}
    assert workspace.metadata["identity_transitive_component_authority_disabled"] is True


def test_missing_preflight_matrix_fails_closed_for_cluster_promoted_flag() -> None:
    workspace = _workspace(with_matrix=False)
    flags, reasons, trusted = _harden_bridge_result(workspace, _legacy_bridge)

    assert flags == [True, False]
    assert reasons == ["direct_sface", "rejected_transitive_component_only"]
    assert trusted == {1}
    assert workspace.metadata["identity_direct_sface_matrix_valid"] is False
