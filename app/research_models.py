from __future__ import annotations

from app.model_registry import ModelManifest


RESEARCH_MODELS: tuple[ModelManifest, ...] = (
    ModelManifest(
        key="refstar_research",
        title="RefSTAR multi-reference restoration research candidate",
        filename="refstar-unverified.checkpoint",
        destination="models/optional/refstar/refstar-unverified.checkpoint",
        source_url=None,
        code_license="UNVERIFIED — production installation prohibited",
        weights_license="UNVERIFIED — production installation prohibited",
        conservative_default=False,
        notes=(
            "Catalog-only candidate for component-wise multi-reference transfer. Official code, pretrained "
            "weights, license, CPU feasibility and provenance behaviour must all be verified after "
            "PRODUCT_COMPLETE_PRE_TUNING; there is no loader or automatic download in this release."
        ),
    ),
    ModelManifest(
        key="instantrestore_research",
        title="InstantRestore severe personalized restoration research candidate",
        filename="instantrestore-unverified.checkpoint",
        destination="models/optional/instantrestore/instantrestore-unverified.checkpoint",
        source_url=None,
        code_license="UNVERIFIED — production installation prohibited",
        weights_license="UNVERIFIED — production installation prohibited",
        conservative_default=False,
        notes=(
            "Catalog-only severe multi-reference candidate. CUDA, RAM/VRAM, license, weights and single-ROI "
            "runtime remain unverified, so it cannot be ACTIVE/FALLBACK and cannot block the base release."
        ),
    ),
    ModelManifest(
        key="osdface_research",
        title="OSDFace restoration research candidate",
        filename="osdface-unverified.checkpoint",
        destination="models/optional/osdface/osdface-unverified.checkpoint",
        source_url=None,
        code_license="UNVERIFIED — production installation prohibited",
        weights_license="UNVERIFIED — production installation prohibited",
        conservative_default=False,
        notes=(
            "Fourth-choice catalog-only candidate. Evaluate only after RefSTAR/InstantRestore leave a measured "
            "severe-restoration gap; no loader, weights or production route is present."
        ),
    ),
)
