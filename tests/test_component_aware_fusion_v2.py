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


_load("app.resource_budget", APP / "resource_budget.py")
adapter = _load("app.face_restorer_adapter", APP / "face_restorer_adapter.py")
_load("app.reference_limits", APP / "reference_limits.py")
reference_bank = _load("app.personalized_reference_bank", APP / "personalized_reference_bank.py")
component_bank = _load("app.component_bank", APP / "component_bank.py")
fusion = _load("app.component_aware_fusion_v2", APP / "component_aware_fusion_v2.py")

RestorationCandidate = adapter.RestorationCandidate
GeneratedPlacement = fusion.GeneratedPlacement
component_aware_fusion = fusion.component_aware_fusion


def _geometry():
    main = np.full((64, 64, 3), 50, dtype=np.uint8)
    landmarks = np.array(
        [[23.0, 24.0], [41.0, 24.0], [32.0, 33.0], [26.0, 44.0], [38.0, 44.0]],
        dtype=np.float32,
    )
    bbox = (14, 10, 36, 47)
    return main, landmarks, bbox


def _candidate(key: str, value: int, *, accepted: bool = True, mask: np.ndarray | None = None):
    image = np.full((64, 64, 3), value, dtype=np.uint8)
    generated_mask = np.full((64, 64), 255, dtype=np.uint8) if mask is None else mask.astype(np.uint8)
    candidate = RestorationCandidate(
        image=image,
        model_key=key,
        model_version="test",
        backend="cpu",
        generated_mask=generated_mask,
    )
    candidate.accepted = accepted
    return candidate


def test_observed_reference_has_absolute_priority_over_whole_face_generator() -> None:
    main, landmarks, bbox = _geometry()
    masks = component_bank.canonical_component_masks(main.shape[:2], landmarks, bbox)
    nose = masks["nose"] > 0
    mouth = masks["mouth"] > 0
    observed_image = main.copy()
    observed_image[nose] = 130
    reference_map = np.zeros(main.shape[:2], dtype=np.uint16)
    reference_map[nose] = 3
    authority = np.where(nose | mouth, 255, 0).astype(np.uint8)

    result = component_aware_fusion(
        main,
        observed_image,
        reference_map,
        authority,
        [GeneratedPlacement(fusion.WHOLE_FACE, _candidate("generator", 220), 1)],
        landmarks5=landmarks,
        bbox=bbox,
    )

    assert np.all(result.image[nose] == 130)
    assert np.all(result.reference_source_map[nose] == 3)
    assert np.all(result.provenance_class_map[nose] == fusion.OBSERVED_SAME_PERSON_REFERENCE)
    mouth_only = mouth & ~nose
    assert np.all(result.image[mouth_only] == 220)
    assert np.all(result.provenance_class_map[mouth_only] == fusion.GENERATED_MODEL)
    healthy = ~(nose | mouth)
    assert np.array_equal(result.image[healthy], main[healthy])
    assert np.all(result.provenance_class_map[healthy] == fusion.OBSERVED_MAIN)


def test_generated_output_is_clipped_to_authority_even_when_model_mask_is_full_face() -> None:
    main, landmarks, bbox = _geometry()
    mouth = component_bank.canonical_component_masks(main.shape[:2], landmarks, bbox)["mouth"] > 0
    authority = np.where(mouth, 255, 0).astype(np.uint8)
    result = component_aware_fusion(
        main,
        main.copy(),
        np.zeros(main.shape[:2], np.uint16),
        authority,
        [GeneratedPlacement(fusion.WHOLE_FACE, _candidate("full-face", 200), 7)],
        landmarks5=landmarks,
        bbox=bbox,
    )
    assert np.all(result.image[mouth] == 200)
    assert np.array_equal(result.image[~mouth], main[~mouth])
    assert np.count_nonzero(result.generated_mask) == np.count_nonzero(mouth)


def test_specific_component_candidate_precedes_whole_face_fallback() -> None:
    main, landmarks, bbox = _geometry()
    masks = component_bank.canonical_component_masks(main.shape[:2], landmarks, bbox)
    mouth = masks["mouth"] > 0
    nose = masks["nose"] > 0
    authority = np.where(mouth | nose, 255, 0).astype(np.uint8)
    specific = _candidate("mouth-specialist", 180)
    fallback = _candidate("whole-face-fallback", 230)
    result = component_aware_fusion(
        main,
        main.copy(),
        np.zeros(main.shape[:2], np.uint16),
        authority,
        [
            GeneratedPlacement("mouth", specific, 1),
            GeneratedPlacement(fusion.WHOLE_FACE, fallback, 2),
        ],
        landmarks5=landmarks,
        bbox=bbox,
    )
    assert np.all(result.generated_candidate_map[mouth] == 1)
    nose_only = nose & ~mouth
    assert np.all(result.generated_candidate_map[nose_only] == 2)
    assert [decision.model_key for decision in result.decisions] == ["mouth-specialist", "whole-face-fallback"]


def test_unaccepted_candidate_has_zero_fusion_authority() -> None:
    main, landmarks, bbox = _geometry()
    authority = np.full(main.shape[:2], 255, dtype=np.uint8)
    result = component_aware_fusion(
        main,
        main.copy(),
        np.zeros(main.shape[:2], np.uint16),
        authority,
        [GeneratedPlacement(fusion.WHOLE_FACE, _candidate("rejected", 200, accepted=False), 1)],
        landmarks5=landmarks,
        bbox=bbox,
    )
    assert np.array_equal(result.image, main)
    assert result.generated_pixels == 0


def test_generated_candidate_cannot_masquerade_as_observed_reference() -> None:
    main, landmarks, bbox = _geometry()
    candidate = _candidate("bad", 200)
    candidate.provenance_class = "OBSERVED_REFERENCE"
    with pytest.raises(RuntimeError, match="invalid provenance class"):
        component_aware_fusion(
            main,
            main.copy(),
            np.zeros(main.shape[:2], np.uint16),
            np.full(main.shape[:2], 255, np.uint8),
            [GeneratedPlacement(fusion.WHOLE_FACE, candidate, 1)],
            landmarks5=landmarks,
            bbox=bbox,
        )


def test_reference_source_map_is_exact_original_index_1_to_9() -> None:
    main, landmarks, bbox = _geometry()
    provenance = np.zeros(main.shape[:2], np.uint16)
    provenance[10:20, 10:20] = 9
    observed = main.copy()
    observed[10:20, 10:20] = 111
    result = component_aware_fusion(
        main,
        observed,
        provenance,
        np.zeros(main.shape[:2], np.uint8),
        [],
        landmarks5=landmarks,
        bbox=bbox,
    )
    assert np.all(result.reference_source_map[10:20, 10:20] == 9)
    bad = provenance.copy()
    bad[0, 0] = 10
    with pytest.raises(ValueError, match="0..9"):
        component_aware_fusion(
            main,
            observed,
            bad,
            np.zeros(main.shape[:2], np.uint8),
            [],
            landmarks5=landmarks,
            bbox=bbox,
        )


def test_generated_candidate_ids_must_be_unique() -> None:
    main, landmarks, bbox = _geometry()
    with pytest.raises(ValueError, match="unique"):
        component_aware_fusion(
            main,
            main.copy(),
            np.zeros(main.shape[:2], np.uint16),
            np.full(main.shape[:2], 255, np.uint8),
            [
                GeneratedPlacement("mouth", _candidate("a", 170), 1),
                GeneratedPlacement(fusion.WHOLE_FACE, _candidate("b", 190), 1),
            ],
            landmarks5=landmarks,
            bbox=bbox,
        )
