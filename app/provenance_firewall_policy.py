from __future__ import annotations

"""Prevent working/restored reference pixels from being promoted to original evidence.

Cross-reference cleanup is useful for analysis and component selection, but the working
reference is not the authority for provenance.  The per-pixel evidence map is.  This
policy makes that distinction enforceable in both observed-target repair and the
one-pixel evidence-completion path.
"""

from functools import wraps
from typing import Any

import cv2
import numpy as np

_INSTALLED = False


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        return np.zeros(shape, dtype=bool)
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=bool)
    return item > 0


def _reference_damage(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and len(masks) >= count + 1:
        return [_binary(np.asarray(masks[index + 1]), shape) for index in range(count)]
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and len(frozen) >= count + 1:
        return [_binary(np.asarray(frozen[index + 1]), shape) for index in range(count)]
    return [np.zeros(shape, dtype=bool) for _ in range(count)]


def _support(workspace, count: int, shape: tuple[int, int]) -> list[np.ndarray]:
    raw = workspace.metadata.get("aligned_reference_support_masks")
    if isinstance(raw, list) and len(raw) == count:
        return [_binary(np.asarray(value), shape) for value in raw]
    return [np.ones(shape, dtype=bool) for _ in range(count)]


def _authoritative_evidence_maps(
    workspace,
    count: int,
    shape: tuple[int, int],
    source_codes: list[int],
) -> list[np.ndarray]:
    """Return per-pixel photographic source ids for each aligned working reference."""
    stored = workspace.metadata.get("preclean_reference_evidence_maps")
    if isinstance(stored, list) and len(stored) == count:
        valid: list[np.ndarray] = []
        for item in stored:
            value = np.asarray(item)
            if value.shape != shape:
                valid = []
                break
            value = value.astype(np.uint16, copy=True)
            # 65534/65535 are symmetry/generated and can never be donor evidence.
            value[(value >= np.uint16(65534))] = np.uint16(0)
            valid.append(value)
        if len(valid) == count:
            workspace.metadata["aligned_reference_source_eligibility_masks"] = [
                np.where(item > 0, 255, 0).astype(np.uint8) for item in valid
            ]
            return valid

    supports = _support(workspace, count, shape)
    damaged = _reference_damage(workspace, count, shape)
    maps: list[np.ndarray] = []
    for index in range(count):
        code = int(source_codes[index]) if index < len(source_codes) else index + 1
        code = max(1, min(65533, code))
        eligible = supports[index] & ~damaged[index]
        source = np.zeros(shape, dtype=np.uint16)
        source[eligible] = np.uint16(code)
        maps.append(source)
    workspace.metadata["aligned_reference_source_eligibility_masks"] = [
        np.where(item > 0, 255, 0).astype(np.uint8) for item in maps
    ]
    return maps


def _apply_source_map_firewall(
    before: np.ndarray,
    result: np.ndarray,
    provenance: np.ndarray,
    evidence_maps: list[np.ndarray],
    container_codes: list[int],
    *,
    selection_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    output = result.copy()
    corrected = provenance.astype(np.uint16, copy=True)
    rejected = 0
    reassigned = 0
    selected_domain = np.ones(provenance.shape, dtype=bool) if selection_mask is None else selection_mask.astype(bool)

    for slot, container_code in enumerate(container_codes):
        if slot >= len(evidence_maps):
            continue
        selected = selected_domain & (provenance == np.uint16(max(1, min(65533, int(container_code)))))
        if not np.any(selected):
            continue
        actual = evidence_maps[slot]
        invalid = selected & (actual == 0)
        if np.any(invalid):
            output[invalid] = before[invalid]
            corrected[invalid] = 0
            rejected += int(np.count_nonzero(invalid))
        valid = selected & (actual > 0) & (actual < np.uint16(65534))
        changed_source = valid & (actual != corrected)
        if np.any(valid):
            corrected[valid] = actual[valid]
        reassigned += int(np.count_nonzero(changed_source))
    return output, corrected, rejected, reassigned


def install_provenance_firewall_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.observed_target_repair_runtime as observed_runtime
    import app.tiny_observed_evidence_policy as tiny_runtime

    observed_original = observed_runtime.repair_observed_target
    tiny_original = tiny_runtime._complete_observed_pixels

    @wraps(observed_original)
    def observed_with_firewall(workspace, image: np.ndarray, **kwargs):
        result, provenance, details = observed_original(workspace, image, **kwargs)
        aligned = list(workspace.aligned_references)
        if not aligned or not isinstance(provenance, np.ndarray) or not np.any(provenance > 0):
            return result, provenance, details
        shape = image.shape[:2]
        container_codes = [int(value) for value in observed_runtime._aligned_original_indices(workspace, len(aligned))]
        maps = _authoritative_evidence_maps(workspace, len(aligned), shape, container_codes)
        output, corrected, rejected, reassigned = _apply_source_map_firewall(
            image, result, provenance, maps, container_codes
        )
        enriched = dict(details)
        enriched["provenance_firewall"] = True
        enriched["firewall_rejected_non_evidence_pixels"] = rejected
        enriched["firewall_reassigned_true_source_pixels"] = reassigned
        enriched["evidence_authority"] = "preclean_reference_evidence_maps_or_original_support"
        enriched["repaired_pixels"] = int(np.count_nonzero(corrected > 0))
        return output, corrected, enriched

    @wraps(tiny_original)
    def tiny_with_firewall(workspace, image: np.ndarray):
        before_provenance = workspace.provenance_map
        if isinstance(before_provenance, np.ndarray) and before_provenance.shape == image.shape[:2]:
            before_provenance = before_provenance.astype(np.uint16, copy=True)
        else:
            before_provenance = np.zeros(image.shape[:2], dtype=np.uint16)

        result, details = tiny_original(workspace, image)
        current = workspace.provenance_map
        aligned = list(workspace.aligned_references)
        shape = image.shape[:2]

        # The native tiny-evidence path now consumes preclean evidence maps directly
        # and writes the true photographic source code into provenance. Reinterpreting
        # that code as a working-reference slot would reject valid cross-cleaned pixels
        # (for example, slot 0 containing an observed pixel from REF2). When authoritative
        # maps were used, preserve the native result and report only unresolved candidate
        # pixels for which no trusted reference supplied photographic evidence.
        authoritative_maps = (
            tiny_runtime._preclean_evidence_maps(workspace, len(aligned), shape)
            if aligned
            else None
        )
        if (
            authoritative_maps is not None
            and bool(details.get("preclean_evidence_authoritative"))
        ):
            target = tiny_runtime._binary(workspace.metadata.get("inpaint_target_mask"), shape)
            if (
                not np.any(target)
                and isinstance(workspace.occlusion_masks, list)
                and workspace.occlusion_masks
            ):
                target = tiny_runtime._binary(np.asarray(workspace.occlusion_masks[0]), shape)

            unresolved_before = target & (before_provenance == 0)
            trusted = tiny_runtime._trusted_flags(workspace, len(aligned))
            supports = tiny_runtime._support_masks(workspace, len(aligned), shape)
            proposed = np.zeros(shape, dtype=bool)
            observed = np.zeros(shape, dtype=bool)
            for index, evidence in enumerate(authoritative_maps):
                if not trusted[index]:
                    continue
                candidate = unresolved_before & supports[index]
                proposed |= candidate
                observed |= candidate & (evidence > 0) & (evidence < np.uint16(65534))

            enriched = dict(details)
            enriched["provenance_firewall"] = True
            enriched["firewall_rejected_non_evidence_pixels"] = int(
                np.count_nonzero(proposed & ~observed)
            )
            enriched.setdefault("firewall_reassigned_true_source_pixels", 0)
            enriched["evidence_authority"] = (
                "preclean_reference_evidence_maps_or_original_support"
            )
            return result, enriched

        if not aligned or not isinstance(current, np.ndarray) or current.shape != shape:
            return result, details

        current = current.astype(np.uint16, copy=True)
        newly_attributed = (before_provenance == 0) & (current > 0) & (current < np.uint16(65534))
        if not np.any(newly_attributed):
            return result, details

        container_codes = [int(value) for value in tiny_runtime._source_codes(workspace, len(aligned))]
        maps = _authoritative_evidence_maps(workspace, len(aligned), shape, container_codes)
        output, corrected, rejected, reassigned = _apply_source_map_firewall(
            image,
            result,
            current,
            maps,
            container_codes,
            selection_mask=newly_attributed,
        )
        workspace.provenance_map = corrected

        if rejected:
            observed_mask = workspace.metadata.get("inpaint_observed_mask")
            if isinstance(observed_mask, np.ndarray) and observed_mask.shape == shape:
                mask = np.asarray(observed_mask).copy()
                invalid = newly_attributed & (corrected == 0)
                mask[invalid] = 0
                workspace.metadata["inpaint_observed_mask"] = mask

        enriched = dict(details)
        enriched["provenance_firewall"] = True
        enriched["firewall_rejected_non_evidence_pixels"] = rejected
        enriched["firewall_reassigned_true_source_pixels"] = reassigned
        enriched["evidence_authority"] = "preclean_reference_evidence_maps_or_original_support"
        enriched["tiny_observed_pixels"] = int(np.count_nonzero(newly_attributed & (corrected > 0)))
        return output, enriched

    observed_runtime.repair_observed_target = observed_with_firewall
    tiny_runtime._complete_observed_pixels = tiny_with_firewall
    _INSTALLED = True
