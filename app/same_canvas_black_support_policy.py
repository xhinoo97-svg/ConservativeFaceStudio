from __future__ import annotations

from functools import wraps

import cv2
import numpy as np


_INSTALLED = False


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    if value.shape != shape:
        raise ValueError("Maschera non compatibile")
    return value > 0


def install_same_canvas_black_support_policy() -> None:
    """Make geometric support authoritative for genuinely black donor pixels.

    The legacy exact same-canvas path also required ``max(RGB)>2`` even after a
    verified geometric support mask existed. That discarded valid pupils, hair,
    eyeliner and deep-shadow pixels. We keep the existing repair algorithm unchanged,
    temporarily mark only support-confirmed dark pixels as nonzero so they remain
    eligible, then restore the exact original donor RGB using the returned provenance.
    Warp padding remains excluded because it has no geometric support.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import app.same_canvas_repair_runtime as module

    original = module.exact_same_canvas_observed_repair

    @wraps(original)
    def patched(workspace, image: np.ndarray, **kwargs):
        aligned = list(workspace.aligned_references)
        supports_raw = workspace.metadata.get("aligned_reference_support_masks")
        if not aligned or not isinstance(supports_raw, list) or len(supports_raw) != len(aligned):
            return original(workspace, image, **kwargs)

        shape = workspace.primary.shape[:2]
        originals = [item.copy() for item in aligned]
        surrogate: list[np.ndarray] = []
        changed_any = False
        for reference, support_raw in zip(aligned, supports_raw):
            if reference.shape[:2] != shape:
                surrogate.append(reference)
                continue
            try:
                support = _binary(np.asarray(support_raw), shape)
            except ValueError:
                surrogate.append(reference)
                continue
            dark_observed = support & (np.max(reference, axis=2) <= 2)
            if not np.any(dark_observed):
                surrogate.append(reference)
                continue
            candidate = reference.copy()
            candidate[dark_observed] = np.maximum(candidate[dark_observed], 3)
            surrogate.append(candidate)
            changed_any = True

        if not changed_any:
            return original(workspace, image, **kwargs)

        explicit_codes = workspace.metadata.get("aligned_reference_original_source_indices")
        if isinstance(explicit_codes, list) and len(explicit_codes) == len(aligned):
            codes = [max(1, int(value)) for value in explicit_codes]
        else:
            runtime = workspace.metadata.get("aligned_reference_source_indices")
            runtime = [int(v) for v in runtime] if isinstance(runtime, list) and len(runtime) == len(aligned) else list(range(len(aligned)))
            codes = module._original_source_indices(workspace, runtime, len(aligned))

        workspace.aligned_references = surrogate
        try:
            result, provenance, details = original(workspace, image, **kwargs)
        finally:
            workspace.aligned_references = aligned

        corrected = result.copy()
        restored_black_pixels = 0
        for donor, code in zip(originals, codes):
            selected = provenance == np.uint16(max(1, int(code)))
            if not np.any(selected):
                continue
            dark_selected = selected & (np.max(donor, axis=2) <= 2)
            restored_black_pixels += int(np.count_nonzero(dark_selected))
            corrected[selected] = donor[selected]

        enriched = dict(details)
        enriched["observed_support_source"] = "aligned_reference_support_masks"
        enriched["rgb_intensity_used_as_support_gate"] = False
        enriched["exact_dark_observed_pixels_restored"] = restored_black_pixels
        return corrected, provenance, enriched

    module.exact_same_canvas_observed_repair = patched
    _INSTALLED = True
