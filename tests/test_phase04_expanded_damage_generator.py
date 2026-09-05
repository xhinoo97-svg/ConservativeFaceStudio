from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from phase04_damage_evaluation import build_matrix  # noqa: E402
from phase04_expanded_damage_generator import apply_expanded_damage  # noqa: E402


def _face(size: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    return np.dstack(
        (
            ((x * 1.5 + y * 0.2) % 256).astype(np.uint8),
            ((x * 0.5 + y * 1.2) % 256).astype(np.uint8),
            ((x * 0.8 + y * 0.7) % 256).astype(np.uint8),
        )
    )


@pytest.mark.parametrize("case", build_matrix(), ids=lambda case: case.case_id)
def test_expanded_generator_is_deterministic_and_exact(case) -> None:
    clean = _face()
    first = apply_expanded_damage(clean, case, seed=240905)
    second = apply_expanded_damage(clean, case, seed=240905)
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.binary_mask, second.binary_mask)
    assert set(np.unique(first.binary_mask)).issubset({0, 255})
    truth = first.binary_mask > 0
    if case.damage_type == "HEALTHY":
        assert not truth.any()
        assert np.array_equal(first.image, clean)
    else:
        assert truth.any(), case.case_id
        assert np.array_equal(first.image[~truth], clean[~truth]), case.case_id
        assert np.all(np.any(first.image[truth] != clean[truth], axis=1)), case.case_id


def test_translucent_sticker_has_three_predeclared_opacity_levels() -> None:
    rows = [row for row in build_matrix() if row.damage_type == "TRANSLUCENT_STICKER"]
    assert [row.opacity for row in rows] == ["LOW", "MEDIUM", "HIGH"]
    clean = _face()
    changed_counts = []
    for row in rows:
        sample = apply_expanded_damage(clean, row, seed=7)
        changed_counts.append(int(np.count_nonzero(sample.binary_mask)))
    assert all(value > 0 for value in changed_counts)


def test_different_seed_changes_sticker_content() -> None:
    case = next(row for row in build_matrix() if row.damage_type == "OPAQUE_STICKER")
    clean = _face()
    first = apply_expanded_damage(clean, case, seed=1)
    second = apply_expanded_damage(clean, case, seed=2)
    assert not np.array_equal(first.image, second.image)
