from __future__ import annotations

"""Keep AutomaticPipelineRunner bound to the final resilient face installer.

`automatic.py` imports `install_pretrained_face_handlers` by value.  Package-level
policies later wrap the function in `app.pretrained_face_handlers`; without rebinding,
the automatic runner can keep calling the stale pre-wrapper function and fall through
to the removed OpenCV Haar backend when YuNet cannot see an occluded MAIN image.
"""

_INSTALLED = False


def install_face_resilience_binding_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.automatic as automatic
    import app.pretrained_face_handlers as handlers

    automatic.install_pretrained_face_handlers = handlers.install_pretrained_face_handlers
    _INSTALLED = True
