from __future__ import annotations

import numpy as np
import pytest

from app.strict_execution import _remap_aligned_provenance


def test_filtered_reference_slots_map_back_to_original_imports() -> None:
    local = np.array(
        [
            [0, 1, 1, 2],
            [2, 0, 1, 2],
        ],
        dtype=np.uint16,
    )

    # Aligned slot 1 came from imported reference #2 (zero-based index 1),
    # aligned slot 2 came from imported reference #4 (zero-based index 3).
    mapped = _remap_aligned_provenance(local, [1, 3])

    expected = np.array(
        [
            [0, 2, 2, 4],
            [4, 0, 2, 4],
        ],
        dtype=np.uint16,
    )
    assert np.array_equal(mapped, expected)


def test_unmappable_local_reference_is_rejected() -> None:
    local = np.array([[0, 1, 2]], dtype=np.uint16)
    with pytest.raises(ValueError, match="non mappabile"):
        _remap_aligned_provenance(local, [0])
