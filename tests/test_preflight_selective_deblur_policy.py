from __future__ import annotations

import numpy as np

from app.preflight_selective_deblur_policy import _selective_learned_blend


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

    # A high-detail image must remain overwhelmingly observed rather than being
    # globally rewritten just because NAFNet was available in preflight.
    changed_fraction = float(np.mean(np.any(result != original, axis=2)))
    assert details["observed_fraction"] > 0.95
    assert changed_fraction < 0.10
