from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.face_domain_guard_v2_policy import IDENTITY_ACCEPTED, _candidate_status
from app.preflight import _direct_main_component_transfer_sources
from app.primary_anchor_policy import _record_same_canvas_evidence


def test_component_transfer_authority_uses_only_direct_main_sface_edge() -> None:
    # MAIN(0)-REF1 is above threshold, REF1-REF2 is above threshold, but MAIN-REF2 is
    # below threshold. REF2 must not inherit authority through the A-B-C chain.
    matrix = np.asarray(
        [
            [1.0, 0.50, 0.20],
            [0.50, 1.0, 0.50],
            [0.20, 0.50, 1.0],
        ],
        dtype=np.float32,
    )
    assert _direct_main_component_transfer_sources([0, 1, 2], matrix) == {1}


def test_direct_component_transfer_survives_ranking_component_rejection() -> None:
    candidate = {
        "source_index": 1,
        "identity_embedding_available": True,
        "accepted_identity": False,
        "accepted_for_component_transfer": True,
    }
    assert _candidate_status(candidate) == IDENTITY_ACCEPTED


def test_same_canvas_face_local_bridge_writes_compatible_equal_aliases() -> None:
    workspace = SimpleNamespace(metadata={})
    _record_same_canvas_evidence(
        workspace,
        matches=[1, 2],
        selected=0,
        identity_bridge_matches=[2],
        applied=False,
        primary_occlusion_seed_present=False,
    )
    evidence = workspace.metadata["same_canvas_primary_anchor"]
    assert evidence["matched_original_reference_indices"] == [1, 2]
    assert evidence["identity_bridge_original_reference_indices"] == [2]
    assert evidence["identity_bridge_matched_original_reference_indices"] == [2]
    assert evidence["identity_bridge_original_reference_indices"] == evidence[
        "identity_bridge_matched_original_reference_indices"
    ]
    assert evidence["identity_bridge_requires_face_local_observed_agreement"] is True
