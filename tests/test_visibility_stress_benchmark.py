from __future__ import annotations

import numpy as np

from app.visibility_stress_benchmark import _split_support, _visible_masks, make_visibility_cases


def test_visible_masks_keep_about_forty_percent() -> None:
    masks = _visible_masks((100, 100))
    assert set(masks) == {"left40", "right40", "upper40", "lower40", "center40"}
    for mask in masks.values():
        visible_fraction = np.count_nonzero(mask) / mask.size
        assert 0.395 <= visible_fraction <= 0.405


def test_complementary_supports_cover_exact_damage_union() -> None:
    damage = np.zeros((80, 120), dtype=np.uint8)
    damage[:, 48:] = 255
    for count in (2, 3, 5):
        supports = _split_support(damage, count)
        assert len(supports) == count
        union = np.zeros_like(damage)
        for support in supports:
            union = np.maximum(union, support)
            assert not np.any((support > 0) & (damage == 0))
        assert np.array_equal(union, damage)


def test_visibility_cases_separate_single_from_strict_multi() -> None:
    clean = np.full((100, 100, 3), 140, dtype=np.uint8)
    cases = make_visibility_cases(clean)
    single = [item for item in cases if item.mode == "single"]
    multi = [item for item in cases if item.mode == "multi"]
    assert len(single) == 5
    assert len(multi) == 16

    for item in single:
        assert item.scenario.recoverable is False
        assert item.scenario.opaque_without_evidence is True
        assert np.count_nonzero(item.expected_union_support) == 0

    for item in multi:
        assert item.scenario.recoverable is True
        assert item.scenario.opaque_without_evidence is False
        damage = item.scenario.damage_mask > 0
        expected = item.expected_union_support > 0
        assert np.array_equal(expected, damage)
        assert 0.395 <= item.visible_fraction <= 0.405
