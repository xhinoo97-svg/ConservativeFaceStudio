from __future__ import annotations

"""Install the one-pixel observed-evidence completion policy in autorun."""

from functools import wraps

_INSTALLED = False


def install_tiny_observed_evidence_autoinstall() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.case_aware_runtime as runtime
    from app.tiny_observed_evidence_policy import install_tiny_observed_evidence_policy

    original = runtime.install_case_aware_runtime

    @wraps(original)
    def patched(executor, model_paths):
        original(executor, model_paths)
        install_tiny_observed_evidence_policy(executor)

    runtime.install_case_aware_runtime = patched
    _INSTALLED = True
