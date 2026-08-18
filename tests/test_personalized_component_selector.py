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


limits = _load("app.reference_limits", APP / "reference_limits.py")
component_bank_module = _load("app.component_bank", APP / "component_bank.py")
reference_bank_module = _load("app.personalized_reference_bank", APP / "personalized_reference_bank.py")
selector_module = _load("app.personalized_component_selector", APP / "personalized_component_selector.py")

ComponentCoverage = component_bank_module.ComponentCoverage
ReferenceObservation = reference_bank_module.ReferenceObservation
build_personalized_reference_bank = reference_bank_module.build_personalized_reference_bank
select_personalized_components = selector_module.select_personalized_components


def _full(source: int, *, eye: float, nose: float, accepted: bool = True):
    return ReferenceObservation(
        source_index=source,
        reference_kind="full",
        identity_accepted=accepted,
        embedding=np.asarray([1.0, source / 100.0], dtype=np.float32),
        face_quality=0.8,
        exposure_quality=0.8,
        pose_quality=0.8,
        resolution_quality=0.8,
        occlusion_quality=0.8,
        component_visibility={"left_eye": eye, "nose": nose},
        component_sharpness={"left_eye": eye, "nose": nose},
        component_coverage={"left_eye": eye, "nose": nose},
    )


def _coverage(component: str, source: int, coverage: float = 1.0, usable: bool = True):
    return ComponentCoverage(source, component, coverage, usable)


def test_different_observed_sources_can_win_eye_and_nose() -> None:
    bank = build_personalized_reference_bank([
        _full(1, eye=1.0, nose=0.35),
        _full(2, eye=0.40, nose=1.0),
    ])
    observed = {
        "left_eye": [_coverage("left_eye", 1), _coverage("left_eye", 2)],
        "nose": [_coverage("nose", 1), _coverage("nose", 2)],
    }
    selected = select_personalized_components(bank, observed)
    assert selected["left_eye"].best_source_index == 1
    assert selected["nose"].best_source_index == 2


def test_wrong_person_is_not_restored_by_geometric_coverage() -> None:
    good = _full(1, eye=0.7, nose=0.7)
    wrong = _full(2, eye=1.0, nose=1.0, accepted=False)
    bank = build_personalized_reference_bank([good, wrong])
    selected = select_personalized_components(
        bank,
        {"left_eye": [_coverage("left_eye", 1), _coverage("left_eye", 2)]},
    )
    assert selected["left_eye"].selected_source_indices == (1,)


def test_partial_reference_can_win_verified_eye_but_never_unverified_mouth() -> None:
    full = ReferenceObservation(
        source_index=1,
        reference_kind="full",
        identity_accepted=True,
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        face_quality=0.7,
        component_visibility={"left_eye": 0.5, "mouth": 0.8},
        component_sharpness={"left_eye": 0.5, "mouth": 0.8},
        component_coverage={"left_eye": 0.5, "mouth": 0.8},
    )
    partial = ReferenceObservation(
        source_index=3,
        reference_kind="partial",
        identity_accepted=False,
        face_quality=1.0,
        exposure_quality=1.0,
        pose_quality=1.0,
        resolution_quality=1.0,
        occlusion_quality=1.0,
        component_visibility={"left_eye": 1.0, "mouth": 1.0},
        component_sharpness={"left_eye": 1.0, "mouth": 1.0},
        component_coverage={"left_eye": 1.0, "mouth": 1.0},
        component_same_person_verified={"left_eye": True, "mouth": False},
    )
    bank = build_personalized_reference_bank([full, partial])
    selected = select_personalized_components(
        bank,
        {
            "left_eye": [_coverage("left_eye", 1), _coverage("left_eye", 3)],
            "mouth": [_coverage("mouth", 1), _coverage("mouth", 3)],
        },
    )
    assert selected["left_eye"].best_source_index == 3
    assert selected["mouth"].selected_source_indices == (1,)
    assert bank.global_anchor_source_indices == (1,)


def test_component_bank_is_a_hard_observed_support_veto() -> None:
    better = _full(1, eye=1.0, nose=0.8)
    weaker = _full(2, eye=0.5, nose=0.8)
    bank = build_personalized_reference_bank([better, weaker])
    selected = select_personalized_components(
        bank,
        {"left_eye": [_coverage("left_eye", 2)]},
    )
    assert selected["left_eye"].selected_source_indices == (2,)
    assert 1 not in selected["left_eye"].observed_coverage_by_source


def test_unusable_component_coverage_has_zero_authority() -> None:
    bank = build_personalized_reference_bank([_full(1, eye=1.0, nose=1.0)])
    selected = select_personalized_components(
        bank,
        {"left_eye": [_coverage("left_eye", 1, coverage=0.1, usable=False)]},
    )
    assert selected["left_eye"].selected_source_indices == ()


def test_unknown_component_bank_key_fails_closed() -> None:
    bank = build_personalized_reference_bank([_full(1, eye=1.0, nose=1.0)])
    with pytest.raises(ValueError, match="Unknown component-bank keys"):
        select_personalized_components(bank, {"ear": []})


def test_max_sources_per_component_is_bounded() -> None:
    bank = build_personalized_reference_bank([
        _full(index, eye=0.9, nose=0.9)
        for index in range(1, 5)
    ])
    observed = {"left_eye": [_coverage("left_eye", index) for index in range(1, 5)]}
    selected = select_personalized_components(bank, observed, max_sources_per_component=2)
    assert len(selected["left_eye"].selected_source_indices) == 2
    with pytest.raises(ValueError, match="1..9"):
        select_personalized_components(bank, observed, max_sources_per_component=10)
