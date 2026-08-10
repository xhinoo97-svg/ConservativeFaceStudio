"""Conservative Face Studio application package."""

from __future__ import annotations

import os

# OpenCV 5 introduced a new DNN graph engine. On the CPU-first models used by this
# application it can emit unsupported-target warnings, and upstream has documented
# graph-fusion regressions for restoration/inpainting models. Keep the stable classic
# engine as the conservative default; advanced users can still override this before
# importing ``app`` by setting OPENCV_FORCE_DNN_ENGINE explicitly.
os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "1")

# Install the reference-driven same-canvas policy before executors capture handlers.
# Explicit observed support, not RGB intensity or an approximate face template, is
# authoritative for verified damaged seed pixels.
from app.same_canvas_seed_support_policy import install_same_canvas_seed_support_policy
from app.same_canvas_seed_precision_policy import install_same_canvas_seed_precision_policy

install_same_canvas_seed_support_policy()
install_same_canvas_seed_precision_policy()
