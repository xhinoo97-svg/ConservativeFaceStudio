from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.observed_target_repair_runtime import _restore_outside_target


def test_observed_repair_does_not_reset_current_context_to_runner_start_anchor() -> None:
    current = np.full((32, 32, 3), 120, dtype=np.uint8)
    current[4:8, 4:8] = (150, 151, 152)
    old_anchor = np.full_like(current, 90)

    result, restored = _restore_outside_target(SimpleNamespace(), current.copy(), old_anchor)

    assert restored == 0
    assert np.array_equal(result, current)
