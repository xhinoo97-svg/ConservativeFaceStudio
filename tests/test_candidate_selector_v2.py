from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if "app" not in sys.modules:
    package = types.ModuleType("app")
    package.__path__ = [str(APP)]
    sys.modules["app"] = package


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# face_restorer_adapter imports resource_budget, so load the narrow dependency first.
_load("app.resource_budget", APP / "resource_budget.py")
adapter = _load("app.face_restorer_adapter", APP / "face_restorer_adapter.py")
selector = _load("app.candidate_selector_v2", APP / "candidate_selector_v2.py")

RestorationCandidate = adapter.RestorationCandidate
CalibratedRankingWeights = selector.CalibratedRankingWeights
CandidateQualityEvidence = selector.CandidateQualityEvidence
CandidateSelectionPolicy = selector.CandidateSelectionPolicy
select_candidate = selector.select_candidate


def _weights(split: str = "DEVELOPMENT") -> CalibratedRankingWeights:
    return CalibratedRankingWeights(
        calibration_id="dev-calibration-fixture",
        split=split,
        weights={name: 1.0 / len(selector.RANKING_METRICS) for name in selector.RANKING_METRICS},
    )


def _policy(**kwargs) -> CandidateSelectionPolicy:
    values = {
        "weights": _weights(),
        "max_landmark_geometry_drift_px": 3.0,
        "max_healthy_region_mae": 8.0,
    }
    values.update(kwargs)
    return CandidateSelectionPolicy(**values)


def _candidate(key: str, value: int = 100) -> RestorationCandidate:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    return RestorationCandidate(
        image=image,
        model_key=key,
        model_version="test",
        backend="cpu",
        generated_mask=mask,
    )


def _evidence(**kwargs) -> CandidateQualityEvidence:
    values = {
        "sface_similarity": 0.90,
        "component_reference_agreement": 0.80,
        "landmark_geometry_quality": 0.90,
        "landmark_geometry_drift_px": 1.0,
        "healthy_region_mae": 2.0,
        "perceptual_quality": 0.80,
        "artifact_quality": 0.80,
        "boundary_quality": 0.80,
        "colour_consistency": 0.80,
        "wrong_person_observed_pixels": 0,
        "provenance_violations": 0,
    }
    values.update(kwargs)
    return CandidateQualityEvidence(**values)


def test_final_holdout_cannot_be_declared_as_weight_calibration_source() -> None:
    with pytest.raises(ValueError, match="DEVELOPMENT/VALIDATION"):
        _weights("FINAL_HOLDOUT")


def test_frozen_sface_threshold_cannot_be_lowered_or_raised() -> None:
    with pytest.raises(ValueError, match="frozen"):
        _policy(identity_threshold=0.360)
    with pytest.raises(ValueError, match="frozen"):
        _policy(identity_threshold=0.400)


def test_inherited_healthy_region_gate_cannot_be_weakened_above_eight() -> None:
    with pytest.raises(ValueError, match="may not weaken"):
        _policy(max_healthy_region_mae=8.01)


def test_better_looking_candidate_cannot_bypass_identity_hard_gate() -> None:
    unsafe = _candidate("pretty-but-drifted")
    safe = _candidate("identity-safe")
    result = select_candidate(
        [unsafe, safe],
        [
            _evidence(
                sface_similarity=0.362,
                perceptual_quality=1.0,
                artifact_quality=1.0,
                boundary_quality=1.0,
                colour_consistency=1.0,
            ),
            _evidence(perceptual_quality=0.70),
        ],
        _policy(),
    )
    assert result.winner_model_key == "identity-safe"
    assert result.evaluations[0].hard_gate_pass is False
    assert any(reason.startswith("sface_below_threshold") for reason in result.evaluations[0].rejection_reasons)


def test_wrong_person_and_provenance_violations_are_absolute_rejections() -> None:
    result = select_candidate(
        [_candidate("wrong"), _candidate("provenance")],
        [
            _evidence(wrong_person_observed_pixels=1),
            _evidence(provenance_violations=1),
        ],
        _policy(),
    )
    assert result.winner_index is None
    assert "wrong_person_observed_pixels:1" in result.evaluations[0].rejection_reasons
    assert "provenance_violations:1" in result.evaluations[1].rejection_reasons


def test_geometry_and_healthy_main_are_hard_gates_before_ranking() -> None:
    result = select_candidate(
        [_candidate("geometry"), _candidate("healthy-drift")],
        [
            _evidence(landmark_geometry_drift_px=3.01),
            _evidence(healthy_region_mae=8.01),
        ],
        _policy(),
    )
    assert result.winner_index is None
    assert any(reason.startswith("landmark_geometry_drift_px") for reason in result.evaluations[0].rejection_reasons)
    assert any(reason.startswith("healthy_region_mae") for reason in result.evaluations[1].rejection_reasons)


def test_calibrated_score_selects_best_only_after_all_hard_gates_pass() -> None:
    first = _candidate("first")
    second = _candidate("second")
    result = select_candidate(
        [first, second],
        [
            _evidence(perceptual_quality=0.50, artifact_quality=0.50),
            _evidence(perceptual_quality=0.95, artifact_quality=0.95),
        ],
        _policy(),
    )
    assert result.winner_index == 1
    assert result.winner_model_key == "second"
    assert second.accepted is True
    assert first.accepted is False
    assert first.rejection_reason == "lower_calibrated_ranking_score"
    assert set(result.evaluations[1].score_breakdown) == set(selector.RANKING_METRICS)
    assert result.calibration_id == "dev-calibration-fixture"


def test_tie_break_is_stable_router_order_not_model_name() -> None:
    first = _candidate("z-model")
    second = _candidate("a-model")
    evidence = _evidence()
    result = select_candidate([first, second], [evidence, evidence], _policy())
    assert result.winner_index == 0
    assert result.winner_model_key == "z-model"


def test_invalid_generated_provenance_class_is_rejected() -> None:
    candidate = _candidate("bad-provenance-class")
    candidate.provenance_class = "OBSERVED_REFERENCE"
    result = select_candidate([candidate], [_evidence()], _policy())
    assert result.winner_index is None
    assert "invalid_generated_provenance_class" in result.evaluations[0].rejection_reasons


def test_weight_contract_is_exact_and_normalized() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        CalibratedRankingWeights("x", "DEVELOPMENT", {"identity": 1.0})
    bad = {name: 0.1 for name in selector.RANKING_METRICS}
    with pytest.raises(ValueError, match="sum to 1.0"):
        CalibratedRankingWeights("x", "VALIDATION", bad)
