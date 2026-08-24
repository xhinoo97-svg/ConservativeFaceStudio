from __future__ import annotations

from functools import wraps

import numpy as np

from app.execution import ExecutionResult
from app.observed_target_repair_runtime import _binary, repair_observed_target
from app.pipeline import BlockKind, BlockSpec


def install_observed_target_fusion_runtime(executor) -> None:
    """Re-apply trusted observed target pixels after FUSION.

    FUSION follows INPAINT in the pipeline and may otherwise overwrite an exact observed
    repair. Re-applying the same conservative transfer here keeps the donor evidence in
    the final face while the normal identity guardrail still evaluates this block output.

    The post-fusion pass must not silently reintroduce stricter coverage or texture
    thresholds than the direct observed-target repair. Geometrically supported donor
    pixels are valid evidence even when smooth or dark; detail reliability is therefore
    a ranking signal by default, not a hard eligibility floor. Likewise, a verified
    damage target may legitimately span most of the face, so the default safety ceiling
    follows the 100% target-aware limit used by the repair runtime. Both values remain
    explicitly overridable through block parameters.
    """
    original = executor._handlers.get(BlockKind.FUSION)
    if original is None:
        return

    @wraps(original)
    def handler(block: BlockSpec, parameters: dict) -> ExecutionResult:
        base_result = original(block, parameters)
        repaired, local_provenance, diagnostics = repair_observed_target(
            executor.workspace,
            base_result.image,
            minimum_reliability=int(parameters.get("observed_target_minimum_reliability", 0)),
            agreement_colour_threshold=float(parameters.get("observed_target_agreement_colour_threshold", 24.0)),
            maximum_face_fraction=float(parameters.get("observed_target_maximum_face_fraction", 1.0)),
        )
        details = dict(base_result.details)
        details["post_fusion_observed_target_repair"] = diagnostics
        if diagnostics.get("applied"):
            current = executor.workspace.provenance_map
            if not isinstance(current, np.ndarray) or current.shape != local_provenance.shape:
                current = np.zeros(local_provenance.shape, dtype=np.uint16)
            else:
                current = current.copy()
            used = local_provenance > 0
            current[used] = local_provenance[used]
            executor.workspace.provenance_map = current

            observed = executor.workspace.metadata.get("inpaint_observed_mask")
            if isinstance(observed, np.ndarray) and observed.shape == local_provenance.shape:
                observed = _binary(observed, local_provenance.shape)
            else:
                observed = np.zeros(local_provenance.shape, dtype=np.uint8)
            observed[used] = 255
            executor.workspace.metadata["inpaint_observed_mask"] = observed
        return ExecutionResult(base_result.block, repaired, details)

    executor._handlers[BlockKind.FUSION] = handler
