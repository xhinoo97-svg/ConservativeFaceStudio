from __future__ import annotations

import cv2
import numpy as np


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return None
    return np.where(item > 0, 255, 0).astype(np.uint8)


def precise_same_canvas_damage_seed(workspace, shape: tuple[int, int]) -> np.ndarray:
    """Return the narrowest authoritative damage seed available.

    The primary heuristic occlusion proposal is intentionally recall-oriented and can
    cover a large part of a face. Once the inpaint stage has produced a non-empty
    verified target, same-canvas transfer must not OR that target with the broad
    proposal: seed pixels are deliberately never discarded by the expansion limiter,
    so doing so can authorize a large false transfer. The verified inpaint target is
    therefore authoritative. Reference consensus is the next-best source, followed by
    the frozen preflight proposal and finally the live primary occlusion mask.
    """
    current = workspace.metadata.get("inpaint_target_mask")
    if isinstance(current, np.ndarray):
        target = _binary(current, shape)
        if target is not None and np.any(target):
            return target

    consensus = workspace.metadata.get("reference_consensus_occlusion")
    if isinstance(consensus, np.ndarray):
        target = _binary(consensus, shape)
        if target is not None and np.any(target):
            return target

    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        target = _binary(np.asarray(frozen[0]), shape)
        if target is not None and np.any(target):
            return target

    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        target = _binary(np.asarray(masks[0]), shape)
        if target is not None and np.any(target):
            return target

    return np.zeros(shape, dtype=np.uint8)


def install_same_canvas_seed_precision_policy() -> None:
    """Install before executors capture the same-canvas repair handler."""
    from app import same_canvas_repair_runtime as runtime

    runtime._damage_seed = precise_same_canvas_damage_seed
