from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.identity_anchor_v4_hardening import _direct_identity_authority, _preflight_direct_sface_edges


def _workspace(payload):
    return SimpleNamespace(
        references=[np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8)],
        metadata={"preflight_identity_similarity": payload},
    )


def _payload(matrix, *, sources=(0, 1, 2), minimum=0.363, source="preflight_existing_sface_embeddings"):
    return {
        "source_indices": list(sources),
        "matrix": matrix,
        "minimum": minimum,
        "source": source,
    }


def test_valid_direct_sface_matrix_is_accepted() -> None:
    workspace = _workspace(
        _payload([
            [1.0, 0.50, 0.20],
            [0.50, 1.0, 0.10],
            [0.20, 0.10, 1.0],
        ])
    )
    parsed = _preflight_direct_sface_edges(workspace)
    assert parsed is not None
    assert _direct_identity_authority(workspace) == {1: (0,)}


def test_nonfinite_or_asymmetric_matrix_fails_closed() -> None:
    nonfinite = _workspace(
        _payload([
            [1.0, float("inf"), 0.0],
            [float("inf"), 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    )
    assert _preflight_direct_sface_edges(nonfinite) is None
    assert _direct_identity_authority(nonfinite) == {}
    assert nonfinite.metadata["identity_direct_sface_matrix_valid"] is False

    asymmetric = _workspace(
        _payload([
            [1.0, 0.50, 0.0],
            [0.20, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    )
    assert _preflight_direct_sface_edges(asymmetric) is None
    assert _direct_identity_authority(asymmetric) == {}


def test_wrong_matrix_provenance_or_out_of_range_source_fails_closed() -> None:
    wrong_source = _workspace(
        _payload(
            [[1.0, 0.5], [0.5, 1.0]],
            sources=(0, 1),
            source="untrusted_recomputed_matrix",
        )
    )
    assert _preflight_direct_sface_edges(wrong_source) is None

    bad_index = _workspace(
        _payload(
            [[1.0, 0.5], [0.5, 1.0]],
            sources=(0, 9),
        )
    )
    assert _preflight_direct_sface_edges(bad_index) is None
