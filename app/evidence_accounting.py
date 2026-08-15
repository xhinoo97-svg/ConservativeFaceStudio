from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        return np.zeros(shape, dtype=bool)
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise AssertionError(f"Spatial accounting map shape {item.shape} does not match final MAIN {shape}")
    return item > 0


def reconcile_evidence_accounting(workspace) -> dict[str, int]:
    """Enforce that damaged-but-unrepaired pixels remain unresolved."""
    shape = workspace.primary.shape[:2]
    provenance = workspace.provenance_map
    if not isinstance(provenance, np.ndarray) or provenance.shape != shape:
        raise AssertionError("Final provenance map is not registered to final MAIN")
    provenance = provenance.astype(np.uint16, copy=False)

    target = _binary(workspace.metadata.get("inpaint_target_mask"), shape)
    observed = _binary(workspace.metadata.get("inpaint_observed_mask"), shape)
    generated = _binary(workspace.metadata.get("inpaint_generated_mask"), shape) | (provenance == 65535)
    symmetry = _binary(workspace.metadata.get("inpaint_symmetry_mask"), shape) | (provenance == 65534)
    reference = (provenance > 0) & (provenance < 65534)

    observed |= reference
    generated &= ~observed
    symmetry &= ~(observed | generated)
    unresolved = _binary(workspace.metadata.get("inpaint_unresolved_mask"), shape)
    unresolved |= target & ~(observed | generated | symmetry)
    unresolved &= ~(observed | generated | symmetry)

    categories = observed.astype(np.uint8) + generated.astype(np.uint8) + symmetry.astype(np.uint8) + unresolved.astype(np.uint8)
    if np.any(categories > 1):
        raise AssertionError("Evidence categories overlap")
    residual = target & ~(observed | generated | symmetry | unresolved)
    if np.any(residual):
        raise AssertionError("Damaged pixels are missing from final evidence accounting")

    workspace.metadata["inpaint_observed_mask"] = observed.astype(np.uint8) * 255
    workspace.metadata["inpaint_generated_mask"] = generated.astype(np.uint8) * 255
    workspace.metadata["inpaint_symmetry_mask"] = symmetry.astype(np.uint8) * 255
    workspace.metadata["inpaint_unresolved_mask"] = unresolved.astype(np.uint8) * 255
    report = {
        "target_pixels": int(np.count_nonzero(target)),
        "observed_repair_pixels": int(np.count_nonzero(observed & target)),
        "generated_pixels": int(np.count_nonzero(generated & target)),
        "symmetry_pixels": int(np.count_nonzero(symmetry & target)),
        "unresolved_pixels": int(np.count_nonzero(unresolved & target)),
        "unclassified_target_pixels": 0,
        "overlap_pixels": 0,
    }
    workspace.metadata["final_evidence_accounting"] = report
    return report
