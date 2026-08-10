"""Conservative Face Studio application package."""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_FORCE_DNN_ENGINE", "1")

from app.same_canvas_seed_support_policy import install_same_canvas_seed_support_policy
from app.same_canvas_seed_precision_policy import install_same_canvas_seed_precision_policy
from app.seed_only_damage_overlap_policy import install_seed_only_damage_overlap_policy
from app.current_context_transaction_policy import install_current_context_transaction_policy
from app.explicit_damage_domain_policy import install_explicit_damage_domain_policy
from app.multi_reference_runtime_policy import install_multi_reference_runtime_policy
from app.automatic_quality_policy import install_automatic_quality_policy
from app.observed_target_photometric_policy import install_observed_target_photometric_policy
from app.automatic_integrity_policy import install_automatic_integrity_policy
from app.pretrained_face_resilience_policy import install_pretrained_face_resilience_policy
from app.face_resilience_binding_policy import install_face_resilience_binding_policy
from app.preflight_selective_deblur_policy import install_preflight_selective_deblur_policy
from app.severity_aware_deblur_policy import install_severity_aware_deblur_policy
from app.reference_guided_seed_policy import install_reference_guided_seed_policy
from app.edge_connected_seed_expansion_policy import install_edge_connected_seed_expansion_policy
from app.fixed_primary_contract_policy import install_fixed_primary_contract_policy
from app.fixed_primary_policy import install_fixed_primary_policy
from app.cross_reference_preclean_autoinstall import install_cross_reference_preclean_autoinstall
from app.coordinate_reference_evidence_policy import install_coordinate_reference_evidence_policy
from app.tiny_observed_evidence_autoinstall import install_tiny_observed_evidence_autoinstall
from app.full_residual_reconstruction_policy import install_full_residual_reconstruction_policy
from app.single_image_core_policy import install_single_image_core_policy
from app.evidence_confidence_runtime import install_evidence_confidence_runtime
from app.core_quality_gate_policy import install_core_quality_gate_policy
from app.adaptive_restoration_autoinstall import install_adaptive_restoration_autoinstall
from app.preexisting_observed_protection_policy import install_preexisting_observed_protection_policy
from app.immutable_input_autoinstall import install_immutable_input_policy
from app.provenance_firewall_policy import install_provenance_firewall_policy
from app.component_bank_evidence_policy import install_component_bank_evidence_policy
from app.adaptive_guardrail_state_policy import install_adaptive_guardrail_state_policy

install_same_canvas_seed_support_policy()
install_same_canvas_seed_precision_policy()
# If a deliberately sparse full-canvas donor exists only under the frozen damage seed,
# its exact coordinates remain usable geometry even when no unaffected baseline exists.
install_seed_only_damage_overlap_policy()
# Regional Block-8/9 repair must preserve the current accepted context, not reset intact
# pixels to the runner-start image and trigger a destructive quality-gate rollback.
install_current_context_transaction_policy()
install_explicit_damage_domain_policy()
install_multi_reference_runtime_policy()
install_automatic_quality_policy()
install_observed_target_photometric_policy()
install_automatic_integrity_policy()
install_pretrained_face_resilience_policy()
# automatic.py imports the installer by value; rebind it after resilience wrapping so
# an occluded MAIN never falls back into the removed OpenCV Haar path.
install_face_resilience_binding_policy()
# Classify immutable inputs first; NAFNet only runs on justified medium/strong cases.
install_preflight_selective_deblur_policy()
install_severity_aware_deblur_policy()
install_reference_guided_seed_policy()
# Allow only a two-pixel, seed-connected detector-border correction from a verified
# coordinate-preserving partial donor. This never creates distant damage components.
install_edge_connected_seed_expansion_policy()
install_fixed_primary_contract_policy()
install_fixed_primary_policy()
install_cross_reference_preclean_autoinstall()
# The generic dark/chroma detector is a proposal, not proof that a clean sparse donor
# pixel is unusable. Exact coordinate peers that overlap and agree can preserve those
# unresolved observed pixels as evidence without changing donor provenance.
install_coordinate_reference_evidence_policy()
install_tiny_observed_evidence_autoinstall()
install_full_residual_reconstruction_policy()
install_single_image_core_policy()
install_evidence_confidence_runtime()
install_core_quality_gate_policy()
# Must patch the final repair installer so LIGHT→MEDIUM→SEVERE wraps the real Block-8 handler.
install_adaptive_restoration_autoinstall()
# Pixels already reconstructed from authoritative Block-7 reference provenance are final
# observed evidence; the adaptive Block-8 cascade must not process them again.
install_preexisting_observed_protection_policy()
# Capture imported MAIN/reference pixels before AutomaticPipelineRunner preflight mutates working copies.
install_immutable_input_policy()
# The working reference may be cleaned, but only its per-pixel evidence map may authorize ORIGINAL_REFERENCE provenance.
install_provenance_firewall_policy()
# Block 7 may inspect cleaned working references, but transfer eligibility remains tied to original observed pixels.
install_component_bank_evidence_policy()

# IMPORTANT: app.automatic imports several installers by value. Some policies above wrap
# those installers only after app.automatic has already been imported by the face binding.
# Refresh the bound names last so autorun uses the exact final policy chain rather than
# stale pre-wrapper functions.
import app.automatic as _automatic
import app.case_aware_runtime as _case_runtime
import app.observed_target_repair_runtime as _target_runtime
import app.pretrained_face_handlers as _face_handlers
import app.pretrained_inpaint_handler as _inpaint_handlers

_automatic.install_pretrained_face_handlers = _face_handlers.install_pretrained_face_handlers
_automatic.install_case_aware_runtime = _case_runtime.install_case_aware_runtime
_automatic.install_verified_inpainting_handler = _inpaint_handlers.install_verified_inpainting_handler
_automatic.install_observed_target_repair_runtime = _target_runtime.install_observed_target_repair_runtime

# Outer guardrail rollback must restore adaptive masks/reports together with image and
# provenance, otherwise Block 9 can inherit stale state from a rejected Block 8.
install_adaptive_guardrail_state_policy()
