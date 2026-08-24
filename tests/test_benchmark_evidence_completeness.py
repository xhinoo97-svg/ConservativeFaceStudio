from __future__ import annotations

import cv2
import numpy as np

from app.practical_benchmark import make_scenarios
from app.practical_benchmark_matrix import make_extended_scenarios


def _portrait() -> np.ndarray:
    image = np.full((192, 192, 3), 28, dtype=np.uint8)
    cv2.ellipse(image, (96, 98), (58, 72), 0, 0, 360, (145, 175, 205), -1)
    cv2.circle(image, (74, 80), 6, (25, 25, 25), -1)
    cv2.circle(image, (118, 80), 6, (25, 25, 25), -1)
    cv2.line(image, (96, 86), (96, 116), (70, 80, 95), 3)
    cv2.ellipse(image, (96, 132), (22, 8), 0, 0, 180, (45, 45, 90), 3)
    return image


def _assert_reference_damage_is_observed(scenarios) -> None:
    for scenario in scenarios:
        if not scenario.recoverable or not scenario.references:
            continue
        observed = np.zeros(scenario.damage_mask.shape, dtype=bool)
        for reference in scenario.references:
            observed |= np.max(reference, axis=2) > 2
        damaged = scenario.damage_mask > 0
        assert np.count_nonzero(damaged) > 0, scenario.name
        assert np.all(observed[damaged]), scenario.name


def test_quick_and_full_reference_cases_are_evidence_complete() -> None:
    clean = _portrait()
    _assert_reference_damage_is_observed(make_scenarios(clean, profile="full"))


def test_extended_reference_cases_are_evidence_complete() -> None:
    clean = _portrait()
    _assert_reference_damage_is_observed(make_extended_scenarios(clean))
