from __future__ import annotations

from functools import wraps
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind

_INSTALLED = False
_CORE = {
    BlockKind.ALIGN,
    BlockKind.OCCLUSION_MASK,
    BlockKind.REGION_SELECT,
    BlockKind.INPAINT,
    BlockKind.FUSION,
}


def _binary(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        return np.zeros(shape, dtype=bool)
    item = np.asarray(value)
    if item.ndim == 3:
        item = cv2.cvtColor(item.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        return np.zeros(shape, dtype=bool)
    return item > 0


def _authorized_damage(workspace, shape: tuple[int, int]) -> np.ndarray:
    target = np.zeros(shape, dtype=bool)
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        target |= _binary(np.asarray(frozen[0]), shape)
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        target |= _binary(np.asarray(masks[0]), shape)
    for key in ("reference_consensus_occlusion", "inpaint_target_mask"):
        target |= _binary(workspace.metadata.get(key), shape)
    if np.any(target):
        # Feathering/registration can legitimately touch a narrow boundary only.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        target = cv2.dilate(target.astype(np.uint8) * 255, kernel, iterations=1) > 0
    return target


def _unresolved_count(workspace, shape: tuple[int, int]) -> int | None:
    value = workspace.metadata.get("inpaint_unresolved_mask")
    if not isinstance(value, np.ndarray):
        return None
    return int(np.count_nonzero(_binary(value, shape)))


def _generated_overwrites_observed(workspace, shape: tuple[int, int]) -> int:
    provenance = workspace.provenance_map
    if not isinstance(provenance, np.ndarray) or provenance.shape != shape:
        return 0
    generated = provenance.astype(np.uint16, copy=False) == np.uint16(65535)
    if not np.any(generated):
        return 0

    # An aligned support mask means a photographed donor exists at that coordinate.
    supports = workspace.metadata.get("aligned_reference_support_masks")
    observed = np.zeros(shape, dtype=bool)
    if isinstance(supports, list):
        for support in supports:
            observed |= _binary(np.asarray(support), shape)

    # Generated pixels are forbidden only where a trusted observed reference actually
    # supports the target. A model may still fill coordinates absent from all donors.
    trusted = workspace.metadata.get("aligned_reference_partial_geometry_verified")
    identity = workspace.metadata.get("aligned_reference_identity_verified")
    if isinstance(trusted, list) or isinstance(identity, list):
        # Support masks have already been filtered by the alignment/reference policies;
        # therefore the union is conservative enough for the final overwrite veto.
        return int(np.count_nonzero(generated & observed))
    return 0


def install_core_quality_gate_policy() -> None:
    """Reject destructive results from blocks 5-9 and keep the previous valid state.

    This is an invariant gate, not a perceptual beauty score. It deliberately checks
    conditions that must never regress: valid numeric output, no broad modification of
    intact pixels, unresolved damage must not increase after INPAINT/FUSION, and a
    generated pixel cannot replace a coordinate for which trusted observed evidence is
    already available. The existing identity guardrail remains a separate gate.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app.automatic import AutomaticPipelineRunner

    original = AutomaticPipelineRunner._apply_guardrail

    @wraps(original)
    def gated(self, block, before: np.ndarray, result: ExecutionResult, state_before=None):
        shape = before.shape[:2]
        unresolved_before = None
        if isinstance(state_before, dict):
            metadata = state_before.get("metadata", {})
            if isinstance(metadata, dict):
                item = metadata.get("inpaint_unresolved_mask")
                if isinstance(item, tuple) and len(item) == 2 and item[0] and isinstance(item[1], np.ndarray):
                    unresolved_before = int(np.count_nonzero(_binary(item[1], shape)))

        accepted = original(self, block, before, result, state_before)
        if block.kind not in _CORE or bool(accepted.details.get("rolled_back", False)):
            return accepted

        reasons: list[str] = []
        image = np.asarray(accepted.image)
        if image.shape != before.shape:
            reasons.append("shape_changed")
        elif not np.issubdtype(image.dtype, np.integer):
            if not np.isfinite(image).all():
                reasons.append("nan_or_inf")
        if image.size == 0:
            reasons.append("empty_output")

        if image.shape == before.shape and block.kind in {BlockKind.REGION_SELECT, BlockKind.INPAINT, BlockKind.FUSION}:
            changed = np.any(image != before, axis=2)
            authorized = _authorized_damage(self.executor.workspace, shape)
            outside = changed & ~authorized
            # Exact preservation is the target; allow only a tiny numerical/boundary
            # residue so photometric feathering cannot cause a false whole-run failure.
            outside_fraction = float(np.count_nonzero(outside) / max(1, outside.size))
            if outside_fraction > 0.0025:
                reasons.append(f"outside_damage_change={outside_fraction:.6f}")

        unresolved_after = _unresolved_count(self.executor.workspace, shape)
        if (
            block.kind in {BlockKind.INPAINT, BlockKind.FUSION}
            and unresolved_before is not None
            and unresolved_after is not None
            and unresolved_after > unresolved_before
        ):
            reasons.append(f"unresolved_increased={unresolved_before}->{unresolved_after}")

        generated_over_observed = _generated_overwrites_observed(self.executor.workspace, shape)
        if block.kind in {BlockKind.INPAINT, BlockKind.FUSION} and generated_over_observed > 0:
            reasons.append(f"generated_over_observed={generated_over_observed}")

        if not reasons:
            details = dict(accepted.details)
            details["core_quality_gate"] = {"accepted": True, "reasons": []}
            return ExecutionResult(accepted.block, accepted.image, details)

        # Roll back both image history and provenance/metadata to the exact pre-block
        # state. This mirrors the identity rollback and guarantees block N-1 survives.
        if not np.array_equal(before, self.executor.workspace.primary):
            try:
                self.executor.history.rollback_discard_current()
            except Exception:
                pass
        self.executor.workspace.primary = before.copy()
        if state_before is not None:
            self._restore_guardrail_state(state_before)

        details = dict(accepted.details)
        details["rolled_back"] = True
        details["rollback_reason"] = "core_quality_gate: " + "; ".join(reasons)
        details["core_quality_gate"] = {"accepted": False, "reasons": reasons}
        details.pop("snapshot_sha256", None)
        replacement = self.executor.block_artifacts.replace_last(before, details)
        details["snapshot_sha256"] = replacement.sha256
        if self.executor.project.operations:
            self.executor.project.operations[-1].parameters.update({
                "rolled_back": True,
                "rollback_reason": details["rollback_reason"],
                "core_quality_gate": details["core_quality_gate"],
                "snapshot_sha256": replacement.sha256,
            })
        return ExecutionResult(accepted.block, before.copy(), details)

    AutomaticPipelineRunner._apply_guardrail = gated
    _INSTALLED = True
