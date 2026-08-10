from __future__ import annotations

import cv2
import numpy as np

_INSTALLED = False


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Maschera non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _context_ring(mask: np.ndarray, radius: int = 9) -> np.ndarray:
    size = max(3, int(radius)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask, kernel)
    return (dilated > 0) & (mask == 0)


def _adjust_donor_to_primary(
    primary: np.ndarray,
    donor: np.ndarray,
    selected: np.ndarray,
    support: np.ndarray,
    *,
    maximum_channel_offset: float = 18.0,
    minimum_context_pixels: int = 48,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Match a donor exposure using only observed context around transferred pixels."""
    ring = _context_ring(selected.astype(np.uint8) * 255)
    active = ring & support
    if int(np.count_nonzero(active)) < int(minimum_context_pixels):
        return donor, (0.0, 0.0, 0.0)
    base = primary.astype(np.float32)
    ref = donor.astype(np.float32)
    offset = np.median(base[active] - ref[active], axis=0).astype(np.float32)
    offset = np.clip(offset, -float(maximum_channel_offset), float(maximum_channel_offset))
    adjusted = np.clip(ref + offset.reshape(1, 1, 3), 0.0, 255.0).astype(np.uint8)
    return adjusted, tuple(float(value) for value in offset.tolist())


def install_observed_target_photometric_policy() -> None:
    """Photometrically normalize reliable real donor pixels without changing geometry.

    The observed-target repair already chooses trusted reference pixels and exact
    provenance. This policy leaves that selection untouched, then applies only a
    small per-channel offset estimated from visible context around each transferred
    patch. Donors explicitly marked with zero detail reliability remain valid observed
    evidence but are copied verbatim: zero reliability is insufficient evidence for an
    automatic photometric transformation even though geometric support still proves
    the pixels were observed.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import app.observed_target_repair_runtime as runtime

    original = runtime.repair_observed_target

    def patched(workspace, image: np.ndarray, **kwargs):
        result, provenance, details = original(workspace, image, **kwargs)
        if not bool(details.get("applied", False)) or not np.any(provenance > 0):
            return result, provenance, details

        shape = image.shape[:2]
        supports_raw = workspace.metadata.get("aligned_reference_support_masks")
        reliabilities_raw = workspace.metadata.get("aligned_reference_detail_reliability_maps")
        aligned = list(workspace.aligned_references)
        supports = (
            [_binary(np.asarray(item), shape) > 0 for item in supports_raw]
            if isinstance(supports_raw, list) and len(supports_raw) == len(aligned)
            else [np.ones(shape, dtype=bool) for _ in aligned]
        )
        reliability_maps = (
            [np.asarray(item) for item in reliabilities_raw]
            if isinstance(reliabilities_raw, list) and len(reliabilities_raw) == len(aligned)
            else [None for _ in aligned]
        )
        original_codes = runtime._aligned_original_indices(workspace, len(aligned))
        code_to_slot = {int(code): slot for slot, code in enumerate(original_codes)}
        corrected = result.copy()
        offsets: dict[str, tuple[float, float, float]] = {}
        skipped_zero_reliability: list[int] = []

        for code in sorted(int(value) for value in np.unique(provenance) if int(value) > 0):
            slot = code_to_slot.get(code)
            if slot is None or slot >= len(aligned):
                continue
            selected = provenance == code
            reliability = reliability_maps[slot]
            if isinstance(reliability, np.ndarray) and reliability.shape == shape:
                finite = np.nan_to_num(reliability.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
                supported_values = finite[supports[slot]]
                if supported_values.size and float(np.max(supported_values)) <= 0.0:
                    # Keep the original observed donor values. Geometric support makes
                    # them valid repair evidence, but zero detail confidence cannot
                    # justify changing their colour/exposure automatically.
                    offsets[str(code)] = (0.0, 0.0, 0.0)
                    skipped_zero_reliability.append(code)
                    continue

            adjusted, offset = _adjust_donor_to_primary(
                workspace.primary,
                aligned[slot],
                selected,
                supports[slot],
            )
            corrected[selected] = adjusted[selected]
            offsets[str(code)] = offset

        enriched = dict(details)
        enriched["photometric_normalization"] = "observed-context-median-bgr-offset"
        enriched["photometric_offsets_bgr"] = offsets
        enriched["photometric_skipped_zero_reliability_sources"] = skipped_zero_reliability
        enriched["photometric_geometry_changed"] = False
        enriched["photometric_outside_target_changed"] = 0
        return corrected, provenance, enriched

    runtime.repair_observed_target = patched
    _INSTALLED = True
