from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from app.pipeline import BlockKind
from app.reference_limits import MAX_REFERENCE_IMAGES
from app.validation import identity_anchor_score


_INSTALLED = False


def install_multi_reference_runtime_policy() -> None:
    """Close runtime gaps left by the legacy strict executor.

    Conservative rules installed here:

    1. REGION_SELECT must not silently fall back to the legacy ``top_k=2``.  When
       callers do not request an explicit K, every aligned donor (up to nine
       references) is made eligible.

    2. The final SFace identity check is anchored to the observed primary selected by
       preflight rather than to sparse component sheets.

    3. Once block 5 has already established trusted geometry for a reference, blocks
       7/9 may consume even a single genuinely observed donor pixel. Minimum-area
       thresholds remain valid for *estimating alignment* but are never used to throw
       away already aligned photographic evidence.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app.execution import BlockExecutionError, ExecutionResult
    from app.strict_execution import StrictBlockExecutor
    from app.tiny_observed_evidence_policy import install_tiny_observed_evidence_policy

    original_init = StrictBlockExecutor.__init__
    original_select = StrictBlockExecutor._specific_memory_select

    @wraps(original_init)
    def patched_init(self, workspace, *, history_limit: int = 12) -> None:
        trusted_anchor = workspace.copy_primary()
        original_init(self, workspace, history_limit=history_limit)
        self._trusted_identity_anchor = trusted_anchor
        self._handlers[BlockKind.IDENTITY_CHECK] = self._trusted_identity_check
        install_tiny_observed_evidence_policy(self)

    @wraps(original_select)
    def patched_select(self, block, parameters: dict[str, Any]) -> ExecutionResult:
        p = dict(parameters)
        requested = p.get("top_k")
        if requested is None:
            # The legacy strict wrapper otherwise injects top_k=2 even though the
            # public reference-memory API already supports dynamic selection.
            p["top_k"] = max(1, min(MAX_REFERENCE_IMAGES, len(self.workspace.aligned_references)))
        return original_select(self, block, p)

    def trusted_identity_check(self, block, parameters: dict[str, Any]) -> ExecutionResult:
        minimum = float(parameters.get("minimum", 0.35))
        anchor = getattr(self, "_trusted_identity_anchor", None)
        if isinstance(anchor, np.ndarray) and anchor.size:
            backend = self.workspace.metadata.get("_identity_backend")
            score, engine = identity_anchor_score(
                self.workspace.primary,
                [anchor],
                backend=backend,
            )
            if score < minimum:
                raise BlockExecutionError(
                    f"Controllo identità {engine} sotto soglia: {score:.3f} < {minimum:.3f}"
                )
            return ExecutionResult(
                block.key,
                self.workspace.copy_primary(),
                {
                    "engine": engine,
                    "scores": [float(score)],
                    "best": float(score),
                    "minimum": minimum,
                    "trusted_preflight_anchor": True,
                },
            )

        # Defensive fallback for direct StrictBlockExecutor use where no observed
        # anchor could be captured. Preserve the original multi-reference check.
        return super(StrictBlockExecutor, self)._identity(block, parameters)

    StrictBlockExecutor.__init__ = patched_init
    StrictBlockExecutor._specific_memory_select = patched_select
    StrictBlockExecutor._trusted_identity_check = trusted_identity_check
    _INSTALLED = True
