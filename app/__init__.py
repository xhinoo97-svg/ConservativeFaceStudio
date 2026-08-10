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
from app.same_canvas_black_support_policy import install_same_canvas_black_support_policy
from app.multi_reference_runtime_policy import install_multi_reference_runtime_policy
from app.automatic_quality_policy import install_automatic_quality_policy
from app.observed_target_photometric_policy import install_observed_target_photometric_policy
from app.automatic_integrity_policy import install_automatic_integrity_policy
from app.pretrained_face_resilience_policy import install_pretrained_face_resilience_policy
from app.preflight_selective_deblur_policy import install_preflight_selective_deblur_policy
from app.reference_guided_seed_policy import install_reference_guided_seed_policy

install_same_canvas_seed_support_policy()
install_same_canvas_seed_precision_policy()
install_same_canvas_black_support_policy()
install_multi_reference_runtime_policy()
install_automatic_quality_policy()
install_observed_target_photometric_policy()
install_automatic_integrity_policy()
install_pretrained_face_resilience_policy()
install_preflight_selective_deblur_policy()
install_reference_guided_seed_policy()
