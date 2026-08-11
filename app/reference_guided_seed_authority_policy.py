from __future__ import annotations

"""Preserve verified reference-guided damage authority through later repair stages.

The generic multi-reference inpaint detector is intentionally allowed to discover
strong reference-supported differences outside a heuristic hint.  Once the
reference-guided seed policy has already validated a same-canvas/coordinate-preserving
consensus, however, that consensus is stronger evidence and explicitly promises not to
expand the frozen seed.  Later INPAINT target discovery must therefore not broaden the
final repair authorization again.
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


def _verified_reference_guided_consensus(workspace, shape: tuple[int, int]) -> np.ndarray | None:
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

    consensus = _binary(workspace.metadata.get("reference_consensus_occlusion"), shape)
    if consensus is None or not np.any(consensus):
        return None
    return consensus


def prefer_verified_reference_guided_seed(
    workspace,
    shape: tuple[int, int],
    fallback_seed: np.ndarray,
) -> np.ndarray:
    """Return the narrowest mutually supported authoritative repair seed.

    With a verified reference-guided consensus, a later non-empty INPAINT target may
    validate fewer pixels but may not authorize new ones outside that consensus.  The
    intersection therefore preserves whichever of the two sources is narrower.  When
    no verified consensus exists the caller's historical fallback is returned exactly.
    """
    consensus = _verified_reference_guided_consensus(workspace, shape)
    if consensus is None:
        return np.asarray(fallback_seed).copy()

    inpaint = _binary(workspace.metadata.get("inpaint_target_mask"), shape)
    if inpaint is not None and np.any(inpaint):
        return cv2.bitwise_and(consensus, inpaint)
    return consensus.copy()


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

    # same_canvas_seed_precision_policy may already have rebound this runtime function
    # before the V2 release policy is installed, so refresh both consumers explicitly.
    import app.same_canvas_repair_runtime as same_canvas_runtime
    import app.observed_target_repair_runtime as observed_target_runtime

    same_canvas_runtime._damage_seed = authoritative_seed
    observed_target_runtime._target_mask = authoritative_seed
    _INSTALLED = True
