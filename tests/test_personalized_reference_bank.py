from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "app" / "personalized_reference_bank.py"
SPEC = importlib.util.spec_from_file_location("cfs_personalized_reference_bank_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
# Supply the small reference_limits dependency without importing app/__init__.
limits_path = Path(__file__).resolve().parents[1] / "app" / "reference_limits.py"
limits_spec = importlib.util.spec_from_file_location("app.reference_limits", limits_path)
assert limits_spec is not None and limits_spec.loader is not None
limits_module = importlib.util.module_from_spec(limits_spec)
sys.modules["app.reference_limits"] = limits_module
limits_spec.loader.exec_module(limits_module)
# Minimal package namespace for the direct module load.
if "app" not in sys.modules:
    import types
    package = types.ModuleType("app")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "app")]
    sys.modules["app"] = package
SPEC.loader.exec_module(module)

ReferenceObservation = module.ReferenceObservation
build_personalized_reference_bank = module.build_personalized_reference_bank


def _full(source: int, embedding: list[float], *, eye=0.8, nose=0.8, accepted=True):
    return ReferenceObservation(
        source_index=source,
        reference_kind="full",
        identity_accepted=accepted,
        identity_similarity=0.8 if accepted else 0.1,
        embedding=np.asarray(embedding, dtype=np.float32),
        face_quality=0.8,
        exposure_quality=0.8,
        pose_quality=0.8,
        resolution_quality=0.8,
        occlusion_quality=0.8,
        component_visibility={"left_eye": eye, "nose": nose, "mouth": 0.8},
        component_sharpness={"left_eye": eye, "nose": nose, "mouth": 0.8},
        component_coverage={"left_eye": eye, "nose": nose, "mouth": 0.8},
    )


def test_wrong_person_never_becomes_global_or_local_authority() -> None:
    good = _full(1, [1.0, 0.0, 0.0], eye=0.7)
    wrong = _full(2, [0.0, 1.0, 0.0], eye=1.0, accepted=False)
    bank = build_personalized_reference_bank([good, wrong])
    assert bank.global_anchor_source_indices == (1,)
    assert [row.source_index for row in bank.ranked("left_eye")] == [1]


def test_partial_eye_reference_is_component_local_and_never_global_anchor() -> None:
    full = _full(1, [1.0, 0.0], eye=0.55)
    partial = ReferenceObservation(
        source_index=3,
        reference_kind="partial",
        identity_accepted=True,
        face_quality=0.9,
        exposure_quality=0.9,
        pose_quality=0.9,
        resolution_quality=0.9,
        occlusion_quality=0.9,
        component_visibility={"left_eye": 1.0, "mouth": 0.0},
        component_sharpness={"left_eye": 1.0, "mouth": 0.0},
        component_coverage={"left_eye": 1.0, "mouth": 0.0},
        component_same_person_verified={"left_eye": True, "mouth": False},
    )
    bank = build_personalized_reference_bank([full, partial])
    assert bank.global_anchor_source_indices == (1,)
    assert bank.ranked("left_eye")[0].source_index == 3
    assert [row.source_index for row in bank.ranked("mouth")] == [1]


def test_ranking_is_per_component_not_one_global_best_reference() -> None:
    eye_best = _full(1, [1.0, 0.0, 0.0], eye=1.0, nose=0.35)
    nose_best = _full(2, [0.95, 0.05, 0.0], eye=0.40, nose=1.0)
    bank = build_personalized_reference_bank([eye_best, nose_best])
    assert bank.ranked("left_eye")[0].source_index == 1
    assert bank.ranked("nose")[0].source_index == 2


def test_consensus_embedding_uses_only_accepted_full_references() -> None:
    a = _full(1, [1.0, 0.0])
    b = _full(2, [0.8, 0.2])
    wrong = _full(3, [-1.0, 0.0], accepted=False)
    partial = ReferenceObservation(
        source_index=4,
        reference_kind="partial",
        identity_accepted=True,
        embedding=np.asarray([-1.0, 0.0], dtype=np.float32),
        component_same_person_verified={"left_eye": True},
    )
    bank = build_personalized_reference_bank([a, b, wrong, partial])
    expected = np.median(
        np.stack([
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32) / np.linalg.norm([0.8, 0.2]),
        ]),
        axis=0,
    )
    expected = expected / np.linalg.norm(expected)
    assert bank.global_anchor_source_indices == (1, 2)
    np.testing.assert_allclose(bank.consensus_embedding, expected, atol=1e-6)


def test_duplicate_original_source_indices_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_personalized_reference_bank([
            _full(1, [1.0, 0.0]),
            _full(1, [0.9, 0.1]),
        ])


def test_reference_count_cannot_exceed_nine() -> None:
    refs = [
        _full(index + 1, [1.0, float(index + 1) / 100.0])
        for index in range(9)
    ]
    bank = build_personalized_reference_bank(refs)
    assert len(bank.references) == 9
