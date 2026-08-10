from __future__ import annotations

from functools import wraps

from app.execution import ExecutionResult
from app.pipeline import BlockKind

_INSTALLED = False


def install_single_image_core_policy() -> None:
    """Keep blocks 5-9 executable even when the project has only the MAIN IMAGE.

    Alignment, component-bank selection and fusion cannot import evidence that does not
    exist, but they still complete deterministically and create checkpoints. INPAINT is
    left to the residual reconstruction policy, which may use a verified local model
    and must mark generated provenance. This avoids a misleading 'skipped core block'
    while preserving the observed-first contract.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app.strict_execution import StrictBlockExecutor
    from app.automatic import AutomaticPipelineRunner

    original_init = StrictBlockExecutor.__init__

    @wraps(original_init)
    def patched_init(self, workspace, *, history_limit: int = 12) -> None:
        original_init(self, workspace, history_limit=history_limit)

        align = self._handlers.get(BlockKind.ALIGN)
        region = self._handlers.get(BlockKind.REGION_SELECT)
        fusion = self._handlers.get(BlockKind.FUSION)

        if align is not None:
            @wraps(align)
            def single_safe_align(block, parameters):
                if not self.workspace.references:
                    self.workspace.aligned_references = []
                    self.workspace.metadata["aligned_reference_source_indices"] = []
                    return ExecutionResult(block.key, self.workspace.copy_primary(), {
                        "engine": "single-image-align-abstain",
                        "reference_count": 0,
                        "aligned": 0,
                        "abstained": True,
                        "reason": "no_reference_available",
                    })
                return align(block, parameters)
            self._handlers[BlockKind.ALIGN] = single_safe_align

        if region is not None:
            @wraps(region)
            def single_safe_region(block, parameters):
                if not self.workspace.aligned_references:
                    return ExecutionResult(block.key, self.workspace.copy_primary(), {
                        "engine": "single-image-component-bank-abstain",
                        "reference_count": 0,
                        "transferred_pixels": 0,
                        "abstained": True,
                        "reason": "no_observed_reference_evidence",
                    })
                return region(block, parameters)
            self._handlers[BlockKind.REGION_SELECT] = single_safe_region

        if fusion is not None:
            @wraps(fusion)
            def single_safe_fusion(block, parameters):
                if not self.workspace.aligned_references:
                    return ExecutionResult(block.key, self.workspace.copy_primary(), {
                        "engine": "single-image-fusion-abstain",
                        "reference_count": 0,
                        "transferred_pixels": 0,
                        "abstained": True,
                        "reason": "no_reference_evidence_to_fuse",
                    })
                return fusion(block, parameters)
            self._handlers[BlockKind.FUSION] = single_safe_fusion

    StrictBlockExecutor.__init__ = patched_init

    original_skip_reason = AutomaticPipelineRunner._skip_reason

    @wraps(original_skip_reason)
    def patched_skip_reason(self, kind):
        # Blocks 5, 7 and 9 now have explicit single-image abstention handlers.
        if kind in {BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.FUSION}:
            return None
        return original_skip_reason(self, kind)

    AutomaticPipelineRunner._skip_reason = patched_skip_reason
    _INSTALLED = True
