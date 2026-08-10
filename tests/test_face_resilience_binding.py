from __future__ import annotations

import app.automatic as automatic
import app.pretrained_face_handlers as handlers


def test_automatic_runner_uses_final_resilient_face_installer() -> None:
    # Package initialization first wraps the installer with reference-derived
    # YuNet/RANSAC resilience, then explicitly rebinds automatic.py's imported symbol.
    # A stale binding previously re-entered cv2.CascadeClassifier on OpenCV 5.
    assert automatic.install_pretrained_face_handlers is handlers.install_pretrained_face_handlers
