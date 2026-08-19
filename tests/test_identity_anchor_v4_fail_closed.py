from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.execution import BlockExecutionError, ExecutionResult
from app.identity_anchor_v4_policy import (
    _require_identity_result_evidence,
    _run_identity_check_with_trusted_anchors,
)
from app.immutable_input_store import ensure_immutable_input_store


def _workspace():
    primary = np.full((16, 16, 3), 80, dtype=np.uint8)
    references = [
        np.full((16, 16, 3), 20, dtype=np.uint8),
        np.full((16, 16, 3), 40, dtype=np.uint8),
    ]
    workspace = SimpleNamespace(
        primary=primary,
        references=references,
        metadata={
            "runtime_source_order": [0, 1, 2],
            "reference_identity_verified": [False, False],
            "reference_identity_scores": [None, None],
            "preflight_candidates": [
                {"source_index": 0, "accepted_identity": False, "identity_embedding_available": False},
                {"source_index": 1, "accepted_identity": False, "identity_embedding_available": False},
                {"source_index": 2, "accepted_identity": False, "identity_embedding_available": False},
            ],
        },
    )
    ensure_immutable_input_store(workspace)
    return workspace


def test_empty_identity_scores_are_not_a_pass() -> None:
    result = ExecutionResult(
        "identity_check",
        np.zeros((4, 4, 3), dtype=np.uint8),
        {"scores": [], "best": 1.0, "minimum": 0.363},
    )
    with pytest.raises(BlockExecutionError, match="nessun confronto"):
        _require_identity_result_evidence(result)


def test_nonempty_proxy_or_sface_scores_remain_valid_evidence() -> None:
    """Legacy structured score results did not always label the engine."""
    for scores in ([0.41], [0.80, 0.39]):
        result = ExecutionResult(
            "identity_check",
            np.zeros((4, 4, 3), dtype=np.uint8),
            {"scores": list(scores), "best": max(scores), "minimum": 0.363},
        )
        _require_identity_result_evidence(result)


def test_explicit_non_sface_proxy_engine_is_not_identity_authority() -> None:
    result = ExecutionResult(
        "identity_check",
        np.zeros((4, 4, 3), dtype=np.uint8),
        {
            "engine": "histogram-proxy",
            "scores": [0.91],
            "best": 0.91,
            "minimum": 0.363,
        },
    )
    with pytest.raises(BlockExecutionError, match="fallback proxy"):
        _require_identity_result_evidence(result)


def test_explicit_sface_engine_with_nonempty_scores_is_valid() -> None:
    result = ExecutionResult(
        "identity_check",
        np.zeros((4, 4, 3), dtype=np.uint8),
        {
            "engine": "opencv-zoo-sface-cpu",
            "scores": [0.41],
            "best": 0.41,
            "minimum": 0.363,
        },
    )
    _require_identity_result_evidence(result)


def test_empty_score_failure_restores_runtime_reference_list() -> None:
    workspace = _workspace()
    runtime_references = list(workspace.references)

    def historically_vacuous_handler(block, parameters):
        # With no trusted donor, V4 still presents immutable MAIN as an anchor.
        assert len(workspace.references) == 1
        return ExecutionResult(
            block,
            workspace.primary.copy(),
            {"scores": [], "best": 1.0, "minimum": 0.363},
        )

    with pytest.raises(BlockExecutionError, match="nessun confronto"):
        _run_identity_check_with_trusted_anchors(
            historically_vacuous_handler,
            workspace,
            "identity_check",
            {"minimum": 0.363},
        )

    assert len(workspace.references) == len(runtime_references)
    assert all(current is original for current, original in zip(workspace.references, runtime_references))
