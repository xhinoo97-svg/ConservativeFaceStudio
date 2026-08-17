from __future__ import annotations

from types import SimpleNamespace

import app.automatic as automatic
import app.face_domain_guard_v2_policy as v2
import app.pretrained_face_handlers as handlers
from app.identity_anchor_v4_policy import POLICY_NAME, install_identity_anchor_v4_policy


def _workspace():
    return SimpleNamespace(
        metadata={
            "preflight_candidates": [
                {
                    "source_index": 0,
                    "accepted_identity": False,
                    "identity_embedding_available": False,
                    "identity_eligibility": v2.PARTIAL_IDENTITY_UNKNOWN,
                },
                {
                    "source_index": 1,
                    "accepted_identity": True,
                    "identity_embedding_available": True,
                    "identity_eligibility": v2.IDENTITY_ACCEPTED,
                },
                {
                    "source_index": 2,
                    "accepted_identity": False,
                    "identity_embedding_available": True,
                    "identity_eligibility": v2.IDENTITY_REJECTED,
                },
            ],
            "same_canvas_primary_anchor": {
                "applied": False,
                "restored_source_index": 0,
                "matched_original_reference_indices": [2],
            },
        }
    )


def test_v4_policy_is_bound_into_installer_and_global_guardrail() -> None:
    install_identity_anchor_v4_policy()
    assert getattr(handlers.install_pretrained_face_handlers, "_cfs_v4_identity_anchor", False) is True
    assert getattr(automatic.AutomaticPipelineRunner._global_identity_anchors, "_cfs_v4_identity_anchor", False) is True


def test_v2_firewall_reads_v4_same_canvas_override_at_runtime() -> None:
    install_identity_anchor_v4_policy()
    workspace = _workspace()

    eligibility = v2._identity_eligibility_by_source(workspace)

    assert eligibility[1] == v2.IDENTITY_ACCEPTED
    assert eligibility[2] == v2.IDENTITY_ACCEPTED
    assert workspace.metadata["identity_firewall_same_canvas_override_original_source_indices"] == [2]
    assert workspace.metadata["identity_anchor_policy"] == POLICY_NAME


def test_non_same_canvas_rejected_source_remains_rejected() -> None:
    install_identity_anchor_v4_policy()
    workspace = _workspace()
    workspace.metadata["same_canvas_primary_anchor"]["matched_original_reference_indices"] = []

    eligibility = v2._identity_eligibility_by_source(workspace)

    assert eligibility[2] == v2.IDENTITY_REJECTED
