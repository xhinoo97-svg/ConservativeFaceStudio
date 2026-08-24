from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CoreFallback:
    order: int
    key: str
    kind: str
    status: str
    role: str
    generated: bool = False
    requires_identity_guardrail: bool = False


# These chains describe the permitted routing order, not a claim that every optional
# research backend is installed. Runtime may only execute entries whose registry state
# is ACTIVE/FALLBACK and whose smoke test is green. TESTING entries are never silently
# promoted. The deterministic SAFE mode is always last.
CORE_FALLBACK_CHAINS: dict[str, tuple[CoreFallback, ...]] = {
    "block_05_align": (
        CoreFallback(1, "yunet_sface_landmarks_ransac", "model+algorithm", "ACTIVE", "5-point identity-aware global alignment"),
        CoreFallback(2, "mediapipe_face_landmarker", "model", "TESTING", "dense landmark alignment"),
        CoreFallback(3, "3ddfa_mb1", "model", "TESTING", "3D pose/geometry alignment"),
        CoreFallback(4, "orb_ransac", "algorithm", "ACTIVE", "texture-feature geometric fallback"),
        CoreFallback(5, "verified_same_canvas_identity", "algorithm", "ACTIVE", "exact identity transform for same-canvas/crops"),
        CoreFallback(6, "component_integer_translation", "algorithm", "ACTIVE", "local strict component refinement"),
        CoreFallback(7, "alignment_abstain", "safe", "ACTIVE", "preserve source when geometry is not trustworthy"),
    ),
    "block_06_occlusion": (
        CoreFallback(1, "reference_consensus", "algorithm", "ACTIVE", "multi-reference difference/agreement mask"),
        CoreFallback(2, "face_parsing_resnet18_onnx", "model", "ACTIVE", "semantic face support"),
        CoreFallback(3, "bisenet_face_parsing", "model", "TESTING", "alternate semantic parsing"),
        CoreFallback(4, "frozen_original_occlusion", "algorithm", "ACTIVE", "pre-restoration scribble/sticker seed"),
        CoreFallback(5, "reference_guided_seed", "algorithm", "ACTIVE", "trusted partial donor seed confirmation"),
        CoreFallback(6, "occlusion_abstain", "safe", "ACTIVE", "do not classify unsupported difference as damage"),
    ),
    "block_07_component_bank": (
        CoreFallback(1, "specific_reference_memory", "algorithm", "ACTIVE", "per-component observed donor selection"),
        CoreFallback(2, "tiny_observed_evidence", "algorithm", "ACTIVE", "retain even one verified donor pixel"),
        CoreFallback(3, "dmdnet_specific_memory", "model", "TESTING", "same-identity specific memory prior"),
        CoreFallback(4, "refface_reference_encoder", "model", "TESTING", "reference-guided identity/texture prior"),
        CoreFallback(5, "median_reference_agreement", "algorithm", "ACTIVE", "conflict/consensus support"),
        CoreFallback(6, "primary_preserve", "safe", "ACTIVE", "keep intact MAIN IMAGE when donors conflict"),
    ),
    "block_08_inpaint": (
        CoreFallback(1, "verified_observed_reference_repair", "algorithm", "ACTIVE", "exact observed donor repair"),
        CoreFallback(2, "cross_reference_preclean", "algorithm", "ACTIVE", "repair references from other observed references"),
        CoreFallback(3, "symmetry_supported", "algorithm", "ACTIVE", "small controlled symmetry fallback", generated=True, requires_identity_guardrail=True),
        CoreFallback(4, "opencv_lama_inpaint", "model", "FALLBACK", "residual hole completion only", generated=True, requires_identity_guardrail=True),
        CoreFallback(5, "refface_inpainting", "model", "TESTING", "reference-guided large-hole inpainting", generated=True, requires_identity_guardrail=True),
        CoreFallback(6, "codeformer_inpainting", "model", "TESTING", "aligned-face residual fallback", generated=True, requires_identity_guardrail=True),
        CoreFallback(7, "dmdnet", "model", "TESTING", "identity-aware blind/specific restoration fallback", generated=True, requires_identity_guardrail=True),
        CoreFallback(8, "gfpgan_v13", "model", "TESTING", "blind residual face restoration", generated=True, requires_identity_guardrail=True),
        CoreFallback(9, "restoreformer_v13_asset", "model", "TESTING", "blind residual face restoration", generated=True, requires_identity_guardrail=True),
    ),
    "block_09_fusion": (
        CoreFallback(1, "observed_target_fusion", "algorithm", "ACTIVE", "provenance-preserving observed fusion"),
        CoreFallback(2, "photometric_context_match", "algorithm", "ACTIVE", "local donor colour/exposure normalization"),
        CoreFallback(3, "component_conflict_veto", "algorithm", "ACTIVE", "reject geometrically/photometrically conflicting donors"),
        CoreFallback(4, "tiny_observed_evidence", "algorithm", "ACTIVE", "complete residual observed pixels"),
        CoreFallback(5, "primary_preserve", "safe", "ACTIVE", "never overwrite a correct observed region without gain"),
    ),
}


def exportable_core_fallbacks() -> dict[str, list[dict[str, object]]]:
    return {key: [asdict(item) for item in chain] for key, chain in CORE_FALLBACK_CHAINS.items()}
