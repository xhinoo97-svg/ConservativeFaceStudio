from __future__ import annotations

from functools import wraps

from app.evidence_confidence import compute_evidence_confidence
from app.execution import ExecutionResult
from app.pipeline import BlockKind

_INSTALLED = False


def install_evidence_confidence_runtime() -> None:
    """Attach face-evidence accounting to the final export result.

    Evidence confidence is provenance based: observed primary/reference pixels count,
    while symmetry, generated and unresolved pixels are reported separately.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from app.strict_execution import StrictBlockExecutor

    original_init = StrictBlockExecutor.__init__

    @wraps(original_init)
    def patched_init(self, workspace, *, history_limit: int = 12) -> None:
        original_init(self, workspace, history_limit=history_limit)
        export_handler = self._handlers.get(BlockKind.EXPORT)
        if export_handler is None:
            return

        @wraps(export_handler)
        def export_with_confidence(block, parameters):
            report = compute_evidence_confidence(self.workspace)
            self.workspace.metadata["evidence_confidence"] = report.as_dict()
            result = export_handler(block, parameters)
            details = dict(result.details)
            details["evidence_confidence"] = report.as_dict()
            if self.project.operations:
                self.project.operations[-1].parameters["evidence_confidence"] = report.as_dict()
            return ExecutionResult(result.block, result.image, details)

        self._handlers[BlockKind.EXPORT] = export_with_confidence

    StrictBlockExecutor.__init__ = patched_init
    _INSTALLED = True
