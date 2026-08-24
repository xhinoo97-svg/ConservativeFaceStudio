from __future__ import annotations

from functools import wraps

_INSTALLED = False


def install_adaptive_restoration_autoinstall() -> None:
    """Ensure the cascade wraps the final Block-8 handler, not an intermediate one."""
    global _INSTALLED
    if _INSTALLED:
        return

    import app.observed_target_repair_runtime as module
    from app.adaptive_restoration_cascade import install_adaptive_restoration_cascade

    previous = module.install_observed_target_repair_runtime

    @wraps(previous)
    def patched(executor) -> None:
        previous(executor)
        install_adaptive_restoration_cascade(executor)

    module.install_observed_target_repair_runtime = patched
    _INSTALLED = True
