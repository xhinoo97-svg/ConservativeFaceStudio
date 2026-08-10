from __future__ import annotations

import numpy as np

from scripts.run_face_anchored_practical_benchmark import (
    _target95_policy_before_score,
    make_face_anchored_scenarios,
)


def _portrait() -> np.ndarray:
    yy, xx = np.indices((160, 160))
    image = np.zeros((160, 160, 3), np.uint8)
    image[..., 0] = (40 + (xx % 170)).astype(np.uint8)
    image[..., 1] = (55 + (yy % 150)).astype(np.uint8)
    image[..., 2] = (75 + ((xx + yy) % 140)).astype(np.uint8)
    return image


def test_quick_release_matrix_contains_every_reference_count_zero_through_nine() -> None:
    scenarios = make_face_anchored_scenarios(_portrait(), profile="quick")
    counts = {len(item.references) for item in scenarios}
    assert counts == set(range(10))


def test_destructive_single_image_cases_are_not_predeclared_target95() -> None:
    scenarios = {item.name: item for item in make_face_anchored_scenarios(_portrait(), profile="full")}
    for name in ("face_blur_heavy_single", "face_mosaic_single"):
        applicable, reason, _coverage = _target95_policy_before_score(scenarios[name])
        assert applicable is False
        assert "single_image" in reason


def test_reference_complete_case_is_target95_before_any_result_exists() -> None:
    scenarios = {item.name: item for item in make_face_anchored_scenarios(_portrait(), profile="full")}
    applicable, reason, coverage = _target95_policy_before_score(scenarios["face_severe_ref9"])
    assert applicable is True
    assert coverage >= 0.95
    assert "reference_union" in reason
