from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.pipeline import BlockKind, default_pipeline
from app.pretrained_face_resilience_policy import (
    _has_full_verified_identity_reference,
    _reference_derived_landmarks,
)


def _block(kind: BlockKind):
    return next(item for item in default_pipeline() if item.kind is kind)


def test_landmarks_without_pretrained_backend_complete_as_explicit_abstention() -> None:
    primary = np.full((64, 64, 3), 120, dtype=np.uint8)
    workspace = SimpleNamespace(
        primary=primary,
        references=[primary.copy()],
        metadata={},
        copy_primary=lambda: primary.copy(),
    )
    executor = SimpleNamespace(workspace=workspace)

    result = _reference_derived_landmarks(
        executor,
        _block(BlockKind.LANDMARKS),
        RuntimeError("legacy detector unavailable"),
    )

    assert result.details["abstained"] is True
    assert result.details["generated_landmarks"] == 0
    assert result.details["backend"] == "landmark-unavailable-conservative-abstain"
    assert workspace.metadata["primary_landmarks5"] is None
    assert workspace.metadata["primary_bbox"] is None


def test_component_only_preflight_flag_is_not_full_sface_identity_evidence() -> None:
    workspace = SimpleNamespace(
        metadata={
            "reference_identity_verified": [True],
            "reference_identity_scores": [None],
        }
    )
    assert _has_full_verified_identity_reference(workspace) is False


def test_real_sface_score_above_threshold_counts_as_full_verified_reference() -> None:
    workspace = SimpleNamespace(
        metadata={
            "reference_identity_verified": [True],
            "reference_identity_scores": [0.60],
        }
    )
    assert _has_full_verified_identity_reference(workspace) is True
