from __future__ import annotations

"""Protect already accepted observed-reference pixels from Block-8 reprocessing.

Block 7 can reconstruct part of the MAIN directly from geometry-trusted reference
pixels before the adaptive LIGHT->MEDIUM->SEVERE cascade starts. Those pixels already
have authoritative photographic provenance and must not be touched again merely because
the original immutable MAIN still classifies the same location as damaged.
"""

from functools import wraps
from typing import Any

import cv2
import numpy as np

_INSTALLED = False
_SYMMETRY = np.uint16(65534)
_GENERATED = np.uint16(65535)


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        return np.zeros(shape, dtype=np.uint8)
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return np.where(item > 0, 255, 0).astype(np.uint8)


def install_preexisting_observed_protection_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.adaptive_restoration_cascade as module

    previous = module.build_severity_map
    if getattr(previous, "_preexisting_observed_protection_policy", False):
        _INSTALLED = True
        return

    @wraps(previous)
    def protected_severity(workspace):
        severity = previous(workspace)
        shape = workspace.primary.shape[:2]
        protected = _binary(workspace.metadata.get("protected_region_mask"), shape)

        provenance = workspace.provenance_map
        observed_reference = np.zeros(shape, dtype=bool)
        if isinstance(provenance, np.ndarray) and provenance.shape == shape:
            codes = provenance.astype(np.uint16, copy=False)
            # 0 = MAIN/unassigned. 1..65533 = observed reference provenance.
            # 65534 = symmetry, 65535 = generated. Only real reference pixels qualify.
            observed_reference = (codes > 0) & (codes < _SYMMETRY)

        already = int(np.count_nonzero(observed_reference))
        if already:
            protected[observed_reference] = 255
            workspace.metadata["protected_region_mask"] = protected
        workspace.metadata["adaptive_preexisting_observed_protected_pixels"] = already
        workspace.metadata["adaptive_preexisting_observed_protection"] = True
        return severity

    protected_severity._preexisting_observed_protection_policy = True  # type: ignore[attr-defined]
    module.build_severity_map = protected_severity
    _INSTALLED = True
