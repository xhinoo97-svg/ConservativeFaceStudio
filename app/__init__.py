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
from app.explicit_damage_domain_policy import install_explicit_damage_domain_policy
from app.multi_reference_runtime_policy import install_multi_reference_runtime_policy
from app.automatic_quality_policy import install_automatic_quality_policy
from app.observed_target_photometric_policy import install_observed_target_photometric_policy
from app.automatic_integrity_policy import install_automatic_integrity_policy
from app.pretrained_face_resilience_policy import install_pretrained_face_resilience_policy
from app.preflight_selective_deblur_policy import install_preflight_selective_deblur_policy
from app.reference_guided_seed_policy import install_reference_guided_seed_policy
from app.fixed_primary_contract_policy import install_fixed_primary_contract_policy
from app.fixed_primary_policy import install_fixed_primary_policy
from app.cross_reference_preclean_autoinstall import install_cross_reference_preclean_autoinstall
from app.tiny_observed_evidence_autoinstall import install_tiny_observed_evidence_autoinstall
from app.full_residual_reconstruction_policy import install_full_residual_reconstruction_policy
from app.evidence_confidence_runtime import install_evidence_confidence_runtime

install_same_canvas_seed_support_policy()
install_same_canvas_seed_precision_policy()
install_explicit_damage_domain_policy()
install_multi_reference_runtime_policy()
install_automatic_quality_policy()
install_observed_target_photometric_policy()
install_automatic_integrity_policy()
install_pretrained_face_resilience_policy()
install_preflight_selective_deblur_policy()
install_reference_guided_seed_policy()
install_fixed_primary_contract_policy()
# Latest product contract is strict: photo #1 is always the target canvas.
install_fixed_primary_policy()
# Every aligned reference is cleaned from observed donor pixels before component-bank use.
install_cross_reference_preclean_autoinstall()
# Once geometry is verified, even one observed donor pixel remains eligible for repair.
install_tiny_observed_evidence_autoinstall()
# Only residual pixels with no usable observed evidence may be generated.
install_full_residual_reconstruction_policy()
# Final export always reports the provenance-derived Original Information Confidence.
install_evidence_confidence_runtime()
