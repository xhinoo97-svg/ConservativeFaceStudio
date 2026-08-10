from __future__ import annotations

"""Wire observed-only reference preclean into every automatic executor.

This patches the already-used case-aware installer before ``app.automatic`` imports
it, avoiding a second runner implementation.  Block 6 first freezes/detects damage;
then the preclean wrapper updates only aligned reference working copies before block 7
builds the component bank.
"""

from functools import wraps

_INSTALLED = False


def install_cross_reference_preclean_autoinstall() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.case_aware_runtime as runtime
    from app.cross_reference_preclean import install_cross_reference_preclean

    original = runtime.install_case_aware_runtime

    @wraps(original)
    def patched(executor, model_paths):
        original(executor, model_paths)
        install_cross_reference_preclean(executor)

    runtime.install_case_aware_runtime = patched
    _INSTALLED = True
