from __future__ import annotations

from functools import wraps

from app.immutable_input_store import ensure_immutable_input_store

_INSTALLED = False


def install_immutable_input_policy() -> None:
    """Capture imported photographs before any preflight/restoration mutation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from app.automatic import AutomaticPipelineRunner
    from app.strict_execution import StrictBlockExecutor

    automatic_init = AutomaticPipelineRunner.__init__
    strict_init = StrictBlockExecutor.__init__

    @wraps(automatic_init)
    def automatic_with_immutable_sources(self, workspace) -> None:
        store = ensure_immutable_input_store(workspace)
        workspace.metadata["immutable_input_policy"] = {
            "captured_before_preflight": True,
            "source_count": 1 + len(store.references),
            "reference_count": len(store.references),
        }
        automatic_init(self, workspace)

    @wraps(strict_init)
    def strict_with_immutable_sources(self, workspace, *, history_limit: int = 12) -> None:
        ensure_immutable_input_store(workspace)
        strict_init(self, workspace, history_limit=history_limit)

    AutomaticPipelineRunner.__init__ = automatic_with_immutable_sources
    StrictBlockExecutor.__init__ = strict_with_immutable_sources
    _INSTALLED = True
