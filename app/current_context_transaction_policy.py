from __future__ import annotations

"""Keep Block-8/9 repair transactional against the current accepted context.

The observed-target wrapper used to restore every pixel outside the current damage
mask to a runner-start anchor. After blocks 2-7 had already been accepted, that reset
changed a large part of the image and the core quality gate correctly rolled Block 8
back. A regional repair must instead leave the incoming accepted context untouched
outside its ROI; the core quality gate remains responsible for rejecting any underlying
handler that itself changes unauthorized pixels.
"""

import numpy as np

_INSTALLED = False


def preserve_current_context(_executor, image: np.ndarray, _legacy_anchor: np.ndarray) -> tuple[np.ndarray, int]:
    """Do not rewrite pixels outside the ROI to an older pipeline checkpoint."""
    return np.asarray(image), 0


def install_current_context_transaction_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from app import observed_target_repair_runtime as runtime

    runtime._restore_outside_target = preserve_current_context
    _INSTALLED = True
