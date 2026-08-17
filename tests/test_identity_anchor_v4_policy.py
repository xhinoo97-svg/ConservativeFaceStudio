from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.identity_anchor_v4_policy import (
    POLICY_NAME,
    _bridge_reference_identity,
    _effective_identity_eligibility,
    _identity_check_anchors,
    _run_identity_check_with_trusted_anchors,
    _same_canvas_original_sources,
    _trusted_identity_source_indices,
    _trusted_raw_reference_positions,
)
from app.immutable_input_store import ensure_immutable_input_store


def _candidate(source: int, accepted: bool, embedding: bool = True) -> dict:
    return {
        "source_index": source,
        "accepted_identity": accepted,
        "identity_embedding_available": embedding,
    }


def _workspace(
    *,
    accepted: set[int],
    same_canvas: list[int] | None,
    scores: list[float | None] | None = None,
    flags: list[bool] | None = None,
):
    count = 3
    primary = np.full((8, 8, 3), 77, dtype=np.uint8)
    refs = [np.full((8, 8, 3), index + 1, dtype=np.uint8) for index in range(count)]
    metadata = {
        "runtime_source_order": [0, 1, 2, 3],
        "preflight_candidates": [
            _candidate(0, 0 in accepted),
            _candidate(1, 1 in accepted),
            _candidate(2, 2 in accepted),
            _candidate(3, 3 in accepted),
        ],
    }
    if same_canvas is not None:
        metadata["same_canvas_primary_anchor"] = {
            # `False` is a valid verified state when MAIN was already source 0.
            "applied": False,
            "matched_original_reference_indices": same_canvas,
            "restored_source_index": 0,
        }
    if scores is not None:
        metadata["reference_identity_scores"] = list(scores)
    if flags is not None:
        metadata["reference_identity_verified"] = list(flags)
        metadata["reference_identity_reasons"] = ["rejected"] * count
    return SimpleNamespace(primary=primary, references=refs, metadata=metadata)


def test_same_canvas_evidence_remains_valid_when_main_was_already_source_zero() -> None:
    workspace = _workspace(accepted=set(), same_canvas=[2])
    assert _same_canvas_original_sources(workspace) == {2}


def test_same_canvas_source_inside_preflight_cluster_bridges_only_that_cluster() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=[2],
        scores=[0.10, 0.12, -0.20],
        flags=[False, False, False],
    )

    flags, reasons, trusted = _bridge_reference_identity(workspace)

    assert flags == [True, True, False]
    assert reasons[0] == "same_canvas_bridged_cross_reference_cluster"
    assert reasons[1] == "verified_same_canvas_main_bridge"
    assert reasons[2] == "rejected"
    assert trusted == {1, 2}
    assert workspace.metadata["identity_anchor_policy"] == POLICY_NAME


def test_same_canvas_source_outside_selected_cluster_does_not_promote_reference_only_cluster() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=[3],
        scores=[0.10, 0.12, -0.20],
        flags=[False, False, False],
    )

    flags, reasons, trusted = _bridge_reference_identity(workspace)

    assert flags == [False, False, True]
    assert reasons == ["rejected", "rejected", "verified_same_canvas_main_bridge"]
    assert trusted == {3}


def test_reference_only_cluster_without_main_bridge_is_never_promoted() -> None:
    workspace = _workspace(
        accepted={1, 2, 3},
        same_canvas=None,
        scores=[0.10, 0.12, 0.11],
        flags=[False, False, False],
    )

    flags, reasons, trusted = _bridge_reference_identity(workspace)

    assert flags == [False, False, False]
    assert reasons == ["rejected", "rejected", "rejected"]
    assert trusted == set()


def test_partial_same_canvas_sheet_never_becomes_global_identity_anchor() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=[2],
        scores=[0.15, None, -0.10],
        flags=[False, False, False],
    )

    flags, _, _ = _bridge_reference_identity(workspace)
    trusted = _trusted_identity_source_indices(workspace, 3)

    assert flags == [True, False, False]
    assert trusted == {1}
    assert 2 not in trusted


def test_v2_firewall_override_is_exact_source_only() -> None:
    workspace = _workspace(accepted={1}, same_canvas=[2])
    eligibility = {1: "IDENTITY_ACCEPTED", 2: "IDENTITY_REJECTED", 3: "IDENTITY_REJECTED"}

    effective = _effective_identity_eligibility(
        workspace,
        eligibility,
        accepted_value="IDENTITY_ACCEPTED",
    )

    assert effective == {
        1: "IDENTITY_ACCEPTED",
        2: "IDENTITY_ACCEPTED",
        3: "IDENTITY_REJECTED",
    }
    assert workspace.metadata["identity_firewall_same_canvas_override_original_source_indices"] == [2]


def test_pre_landmarks_global_anchors_require_main_or_same_canvas_bridge() -> None:
    reference_only = _workspace(accepted={1, 2}, same_canvas=None)
    assert _trusted_identity_source_indices(reference_only, 3) == set()

    bridged = _workspace(accepted={1, 2}, same_canvas=[2])
    assert _trusted_identity_source_indices(bridged, 3) == {1, 2}


def test_exact_same_canvas_rejected_from_cluster_is_global_only_with_embedding() -> None:
    workspace = _workspace(accepted={1, 2}, same_canvas=[3])
    assert _trusted_identity_source_indices(workspace, 3) == {3}

    workspace.metadata["preflight_candidates"][3]["identity_embedding_available"] = False
    assert _trusted_identity_source_indices(workspace, 3) == set()


def test_final_identity_check_positions_exclude_wrong_person_raw_reference() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=[2],
        scores=[0.20, 0.18, -0.25],
        flags=[False, False, False],
    )
    _bridge_reference_identity(workspace)

    positions, sources = _trusted_raw_reference_positions(workspace)

    assert positions == [0, 1]
    assert sources == [1, 2]
    assert 2 not in positions  # raw slot 2 is original source 3, the wrong-person source


def test_final_identity_firewall_always_keeps_immutable_main_anchor() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=None,
        scores=[0.10, 0.12, -0.20],
        flags=[False, False, False],
    )
    original_main = workspace.primary.copy()

    anchors, sources = _identity_check_anchors(workspace)

    assert sources == []
    assert len(anchors) == 1
    assert np.array_equal(anchors[0], original_main)
    anchors[0][:] = 0
    assert np.array_equal(workspace.metadata["_immutable_input_store"].main, original_main)


def test_final_identity_firewall_adds_only_trusted_refs_after_immutable_main() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=[2],
        scores=[0.20, 0.18, -0.25],
        flags=[False, False, False],
    )
    _bridge_reference_identity(workspace)

    anchors, sources = _identity_check_anchors(workspace)

    assert sources == [1, 2]
    assert len(anchors) == 3
    assert np.array_equal(anchors[0], workspace.primary)
    assert np.array_equal(anchors[1], workspace.references[0])
    assert np.array_equal(anchors[2], workspace.references[1])


def test_final_identity_anchors_resolve_original_sources_after_runtime_reordering() -> None:
    workspace = _workspace(
        accepted=set(),
        same_canvas=None,
        scores=[0.4, None, None],
        flags=[True, False, False],
    )
    original_refs = [item.copy() for item in workspace.references]
    ensure_immutable_input_store(workspace)

    # Runtime order is now: original source 3, then 1, then 2. The verified flag at
    # runtime position 0 therefore means original source 3, not immutable ref slot 0.
    workspace.references = [original_refs[2].copy(), original_refs[0].copy(), original_refs[1].copy()]
    workspace.metadata["runtime_source_order"] = [0, 3, 1, 2]

    anchors, sources = _identity_check_anchors(workspace)

    assert sources == [3]
    assert len(anchors) == 2
    assert np.array_equal(anchors[1], original_refs[2])
    assert not np.array_equal(anchors[1], workspace.references[1])


def test_identity_wrapper_sees_only_immutable_main_and_trusted_refs_then_restores_runtime_refs() -> None:
    workspace = _workspace(
        accepted={1, 2},
        same_canvas=[2],
        scores=[0.20, 0.18, -0.25],
        flags=[False, False, False],
    )
    ensure_immutable_input_store(workspace)
    _bridge_reference_identity(workspace)
    runtime_refs = list(workspace.references)
    seen: list[list[np.ndarray]] = []

    def fake_handler(block, parameters):
        seen.append([item.copy() for item in workspace.references])
        return {"block": block, "parameters": dict(parameters)}

    result, sources, raw_count = _run_identity_check_with_trusted_anchors(
        fake_handler,
        workspace,
        "identity",
        {"minimum": 0.363},
    )

    assert result["block"] == "identity"
    assert sources == [1, 2]
    assert raw_count == 3
    assert len(seen) == 1
    assert len(seen[0]) == 3
    assert np.array_equal(seen[0][0], workspace.primary)
    assert np.array_equal(seen[0][1], runtime_refs[0])
    assert np.array_equal(seen[0][2], runtime_refs[1])
    assert len(workspace.references) == 3
    assert all(current is original for current, original in zip(workspace.references, runtime_refs))


def test_identity_wrapper_restores_runtime_refs_when_identity_handler_raises() -> None:
    workspace = _workspace(
        accepted={1},
        same_canvas=None,
        scores=[0.50, -0.20, None],
        flags=[True, False, False],
    )
    ensure_immutable_input_store(workspace)
    runtime_refs = list(workspace.references)

    def failing_handler(block, parameters):
        assert len(workspace.references) == 2  # immutable MAIN + source 1
        raise RuntimeError("forced identity failure")

    with pytest.raises(RuntimeError, match="forced identity failure"):
        _run_identity_check_with_trusted_anchors(
            failing_handler,
            workspace,
            "identity",
            {},
        )

    assert len(workspace.references) == 3
    assert all(current is original for current, original in zip(workspace.references, runtime_refs))


def test_existing_direct_sface_flags_are_preserved_without_same_canvas() -> None:
    workspace = _workspace(
        accepted=set(),
        same_canvas=None,
        scores=[0.50, -0.20, None],
        flags=[True, False, False],
    )
    workspace.metadata["reference_identity_reasons"] = ["direct_sface", "rejected", "rejected"]

    flags, reasons, trusted = _bridge_reference_identity(workspace)

    assert flags == [True, False, False]
    assert reasons[0] == "direct_sface"
    assert trusted == {1}
