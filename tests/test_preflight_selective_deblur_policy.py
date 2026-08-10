from __future__ import annotations

import numpy as np

import app.preflight_selective_deblur_policy as policy
from app.preflight_selective_deblur_policy import _learned_deblur_indices, _selective_learned_blend


def test_sparse_component_reference_is_byte_identical_after_deblur_decision() -> None:
    reference = np.zeros((80, 80, 3), dtype=np.uint8)
    reference[28:44, 20:60] = (90, 130, 180)
    candidate = np.full_like(reference, 210)

    result, details = _selective_learned_blend(reference, candidate)

    assert details["observed_fraction"] < 0.30
    assert details["active_fraction"] == 0.0
    assert np.array_equal(result, reference)


def test_high_detail_observed_image_is_not_globally_replaced_by_learned_candidate() -> None:
    yy, xx = np.indices((96, 96))
    checker = (((xx // 3 + yy // 3) % 2) * 150 + 50).astype(np.uint8)
    original = np.dstack((checker, np.roll(checker, 1, axis=0), np.roll(checker, 1, axis=1)))
    candidate = np.full_like(original, 127)

    result, details = _selective_learned_blend(original, candidate)

    changed_fraction = float(np.mean(np.any(result != original, axis=2)))
    assert details["observed_fraction"] > 0.95
    assert changed_fraction < 0.10


def test_learned_router_excludes_clean_and_light_before_inference(monkeypatch) -> None:
    images = [
        np.full((64, 64, 3), 80, np.uint8),
        np.full((64, 64, 3), 120, np.uint8),
        np.full((64, 64, 3), 160, np.uint8),
        np.full((64, 64, 3), 200, np.uint8),
    ]
    levels = iter(("none", "mild", "medium", "strong"))

    def fake_classify(_image):
        return {"level": next(levels), "score": 0.5}

    monkeypatch.setattr(policy, "classify_blur", fake_classify)
    selected, diagnostics = _learned_deblur_indices(images)

    assert selected == [2, 3]
    assert [item["level"] for item in diagnostics] == ["none", "mild", "medium", "strong"]


def test_sparse_reference_is_never_routed_to_nafnet_even_when_labelled_strong(monkeypatch) -> None:
    sparse = np.zeros((80, 80, 3), np.uint8)
    sparse[30:38, 30:38] = 180
    monkeypatch.setattr(policy, "classify_blur", lambda _image: {"level": "strong", "score": 1.0})

    selected, _ = _learned_deblur_indices([sparse])

    assert selected == []
