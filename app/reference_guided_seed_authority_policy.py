from __future__ import annotations

"""Preserve verified reference-guided damage authority through later repair stages.

The generic multi-reference inpaint detector is intentionally allowed to discover
strong reference-supported differences outside a heuristic hint. Once the
reference-guided seed policy has already validated a same-canvas/coordinate-preserving
authority, however, that immutable authority is stronger evidence and explicitly
promises not to expand the frozen seed. Later repair stages must therefore preserve
both seed and final transfer authority.
"""

from functools import wraps

import cv2
import numpy as np


_INSTALLED = False


def _binary(value, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(value, np.ndarray):
        return None
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return None
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _verified_reference_guided_authority(workspace, shape: tuple[int, int]) -> np.ndarray | None:
    diagnostics = workspace.metadata.get("reference_guided_seed_diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    if str(diagnostics.get("reason", "")) != "reference_guided_frozen_seed":
        return None
    try:
        trusted = int(diagnostics.get("trusted_donors", 0))
        refined = int(diagnostics.get("refined_pixels", 0))
    except (TypeError, ValueError):
        return None
    if trusted <= 0 or refined <= 0:
        return None
    if diagnostics.get("seed_expansion_from_partial_reference") is not False:
        return None

    authority = _binary(workspace.metadata.get("reference_guided_authority_mask"), shape)
    if authority is None or not np.any(authority):
        return None
    return authority


def constrain_verified_reference_guided_target(
    workspace,
    target: np.ndarray,
) -> np.ndarray:
    """Clamp a proposed observed-reference target to immutable verified authority."""
    shape = workspace.primary.shape[:2]
    proposed = _binary(target, shape)
    if proposed is None:
        return np.asarray(target).copy()

    authority = _verified_reference_guided_authority(workspace, shape)
    if authority is None:
        return proposed.copy()
    return cv2.bitwise_and(proposed, authority)


def prefer_verified_reference_guided_seed(
    workspace,
    shape: tuple[int, int],
    fallback_seed: np.ndarray,
) -> np.ndarray:
    """Return the narrowest mutually supported authoritative repair seed."""
    authority = _verified_reference_guided_authority(workspace, shape)
    if authority is None:
        return np.asarray(fallback_seed).copy()

    inpaint = _binary(workspace.metadata.get("inpaint_target_mask"), shape)
    if inpaint is not None and np.any(inpaint):
        return cv2.bitwise_and(authority, inpaint)
    return authority.copy()


def constrain_verified_reference_guided_transfer(
    workspace,
    input_image: np.ndarray,
    repaired: np.ndarray,
    provenance: np.ndarray,
    diagnostics: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Prevent a same-canvas transfer from modifying pixels outside verified authority."""
    shape = workspace.primary.shape[:2]
    authority = _verified_reference_guided_authority(workspace, shape)
    if authority is None:
        return repaired, provenance, diagnostics
    if input_image.shape != workspace.primary.shape or repaired.shape != input_image.shape:
        return repaired, provenance, diagnostics
    if provenance.shape != shape:
        return repaired, provenance, diagnostics

    changed = np.any(np.asarray(repaired) != np.asarray(input_image), axis=2)
    unauthorized = changed & ~(authority > 0)
    unauthorized_pixels = int(np.count_nonzero(unauthorized))

    constrained = repaired.copy()
    constrained_provenance = provenance.copy()
    if unauthorized_pixels:
        constrained[unauthorized] = input_image[unauthorized]
        constrained_provenance[unauthorized] = 0

    details = dict(diagnostics)
    details["reference_guided_authority_applied"] = True
    details["reference_guided_authority_pixels"] = int(np.count_nonzero(authority))
    details["reference_guided_clamped_transfer_pixels"] = unauthorized_pixels
    return constrained, constrained_provenance, details


def install_reference_guided_seed_authority_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.same_canvas_seed_precision_policy as precision

    original = precision.precise_same_canvas_damage_seed
    if getattr(original, "_cfs_reference_guided_seed_authority", False):
        _INSTALLED = True
        return

    @wraps(original)
    def authoritative_seed(workspace, shape: tuple[int, int]) -> np.ndarray:
        fallback = original(workspace, shape)
        return prefer_verified_reference_guided_seed(workspace, shape, fallback)

    authoritative_seed._cfs_reference_guided_seed_authority = True  # type: ignore[attr-defined]
    precision.precise_same_canvas_damage_seed = authoritative_seed

    import app.same_canvas_repair_runtime as same_canvas_runtime
    import app.observed_target_repair_runtime as observed_target_runtime

    same_canvas_runtime._damage_seed = authoritative_seed
    observed_target_runtime._target_mask = authoritative_seed

    original_exact = same_canvas_runtime.exact_same_canvas_observed_repair

    @wraps(original_exact)
    def authoritative_exact_transfer(workspace, image: np.ndarray, *args, **kwargs):
        repaired, provenance, diagnostics = original_exact(workspace, image, *args, **kwargs)
        return constrain_verified_reference_guided_transfer(
            workspace,
            image,
            repaired,
            provenance,
            diagnostics,
        )

    authoritative_exact_transfer._cfs_reference_guided_transfer_authority = True  # type: ignore[attr-defined]
    same_canvas_runtime.exact_same_canvas_observed_repair = authoritative_exact_transfer
    _INSTALLED = True
