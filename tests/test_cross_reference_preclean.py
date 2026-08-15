from __future__ import annotations

import warnings

import numpy as np

from app.cross_reference_preclean import _robust_observed_donor


def test_no_evidence_pixels_do_not_trigger_all_nan_reduction_warning() -> None:
    donor = np.full((16, 16, 3), 120, dtype=np.uint8)
    valid = np.zeros((16, 16), dtype=bool)
    valid[5:8, 5:8] = True
    target = np.zeros((16, 16), dtype=bool)
    target[5:8, 5:8] = True

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        chosen, confidence, provenance = _robust_observed_donor(
            [donor], [valid], [2], target
        )

    assert np.all(chosen[target] == 120)
    assert np.all(confidence[target] == 180)
    assert np.all(provenance[target] == 2)
    assert np.all(provenance[~target] == 0)
