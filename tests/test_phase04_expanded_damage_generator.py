from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from phase04_damage_evaluation import (  # noqa: E402
    POSITIONS,
    SEVERITIES,
    SIZES,
    TRANSLUCENT_OPACITIES,
    build_matrix,
)
from phase04_expanded_damage_generator import apply_expanded_damage  # noqa: E402


def _face(size: int = 128) -> np.ndarray:
    """Deterministic structured QA image with edges/texture in every facial region.

    A purely affine gradient is an invalid blur oracle because normalized blur
    kernels can preserve affine fields exactly away from borders. This fixture
    keeps a smooth face-like base while adding bounded multi-scale texture so
    blur, motion-blur and defocus operations have measurable pixel authority at
    every declared position.
    """
    y, x = np.mgrid[0:size, 0:size]
    checker = (((x // 4) + (y // 4)) % 2).astype(np.float32)
    fine = (((x * 17 + y * 29 + (x * y) % 31) % 23) - 11).astype(np.float32)
    radial = (((x - size / 2.0) ** 2 + (y - size / 2.0) ** 2) ** 0.5) % 19
    channels = (
        82.0 + x * 0.72 + y * 0.16 + checker * 28.0 + fine * 1.7,
        74.0 + x * 0.28 + y * 0.66 + checker * 19.0 - fine * 1.3 + radial * 0.7,
        96.0 + x * 0.44 + y * 0.36 + checker * 24.0 + fine * 1.1 - radial * 0.5,
    )
    return np.dstack([np.clip(channel, 0, 255).astype(np.uint8) for channel in channels])


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


def test_translucent_sticker_has_full_predeclared_opacity_cross_product() -> None:
    rows = [row for row in build_matrix() if row.damage_type == "TRANSLUCENT_STICKER"]
    assert len(rows) == len(POSITIONS) * len(SIZES) * len(SEVERITIES) * len(TRANSLUCENT_OPACITIES)
    assert {row.opacity for row in rows} == set(TRANSLUCENT_OPACITIES)

    grouped: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (row.position, row.size, row.severity)
        grouped.setdefault(key, set()).add(row.opacity)
    assert len(grouped) == len(POSITIONS) * len(SIZES) * len(SEVERITIES)
    assert all(opacities == set(TRANSLUCENT_OPACITIES) for opacities in grouped.values())

    clean = _face()
    representative = [
        next(
            row
            for row in rows
            if row.position == POSITIONS[0]
            and row.size == SIZES[1]
            and row.severity == SEVERITIES[1]
            and row.opacity == opacity
        )
        for opacity in TRANSLUCENT_OPACITIES
    ]
    changed_counts = [
        int(np.count_nonzero(apply_expanded_damage(clean, row, seed=7).binary_mask))
        for row in representative
    ]
    assert all(value > 0 for value in changed_counts)


def test_different_seed_changes_sticker_content() -> None:
    case = next(row for row in build_matrix() if row.damage_type == "OPAQUE_STICKER")
    clean = _face()
    first = apply_expanded_damage(clean, case, seed=1)
    second = apply_expanded_damage(clean, case, seed=2)
    assert not np.array_equal(first.image, second.image)
