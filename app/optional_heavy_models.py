from __future__ import annotations

from dataclasses import dataclass

from app.model_registry import ModelManifest


@dataclass(frozen=True)
class HeavyModelRuntimeProfile:
    key: str
    weight_bytes: int
    sha256: str
    allowed_blocks: tuple[str, ...]
    provenance_class: str
    load_policy: str
    identity_guardrail_required: bool
    rollback_required: bool
    measured_peak_ram_mb: int | None
    measured_seconds_512: float | None
    benchmark_status: str


# These checkpoints are intentionally not part of the base installer.  They are blind
# face restoration models and may synthesize identity-critical texture.  The official
# upstream release URL is registered for auditability, while runtime activation remains
# opt-in and is allowed only after the observed-reference route has abstained.
OPTIONAL_HEAVY_MODELS: tuple[ModelManifest, ...] = (
    ModelManifest(
        key="codeformer_v010",
        title="CodeFormer v0.1.0 optional blind-face restoration",
        filename="codeformer.pth",
        destination="models/optional/codeformer/codeformer.pth",
        source_url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        code_license="NTU S-Lab License 1.0",
        weights_license="NTU S-Lab License 1.0 / upstream terms",
        conservative_default=False,
        expected_sha256="1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        max_bytes=376_637_898,
        notes=(
            "Heavy optional fallback only. Never default strict. The upstream release asset is "
            "376,637,898 bytes. Generated/low-confidence provenance is mandatory; only INPAINT "
            "or ENHANCE residuals may use it after observed same-identity evidence is exhausted."
        ),
    ),
    ModelManifest(
        key="gfpgan_v13",
        title="GFPGAN v1.3 optional blind-face restoration",
        filename="GFPGANv1.3.pth",
        destination="models/optional/gfpgan/GFPGANv1.3.pth",
        source_url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        code_license="Apache-2.0",
        weights_license="Apache-2.0 / upstream model terms",
        conservative_default=False,
        expected_sha256="c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70",
        max_bytes=348_632_874,
        notes=(
            "Heavy optional fallback only. Never default strict. Official v1.3 release asset is "
            "348,632,874 bytes. Use one heavy restorer at a time and require identity rollback."
        ),
    ),
    ModelManifest(
        key="restoreformer_v13_asset",
        title="RestoreFormer optional blind-face restoration",
        filename="RestoreFormer.pth",
        destination="models/optional/restoreformer/RestoreFormer.pth",
        source_url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/RestoreFormer.pth",
        code_license="Apache-2.0",
        weights_license="Apache-2.0 / upstream model terms",
        conservative_default=False,
        expected_sha256="07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0",
        max_bytes=290_785_322,
        notes=(
            "Heavy optional fallback only. Never default strict. Official GFPGAN release asset is "
            "290,785,322 bytes; original RestoreFormer project is Apache-2.0."
        ),
    ),
)


OPTIONAL_HEAVY_RUNTIME_PROFILES: tuple[HeavyModelRuntimeProfile, ...] = (
    HeavyModelRuntimeProfile(
        key="codeformer_v010",
        weight_bytes=376_637_898,
        sha256="1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
        allowed_blocks=("enhance", "inpaint"),
        provenance_class="generated-low-confidence",
        load_policy="single-heavy-model-sequential",
        identity_guardrail_required=True,
        rollback_required=True,
        measured_peak_ram_mb=None,
        measured_seconds_512=None,
        benchmark_status="target-hardware measurement required before enablement",
    ),
    HeavyModelRuntimeProfile(
        key="gfpgan_v13",
        weight_bytes=348_632_874,
        sha256="c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70",
        allowed_blocks=("enhance", "inpaint"),
        provenance_class="generated-low-confidence",
        load_policy="single-heavy-model-sequential",
        identity_guardrail_required=True,
        rollback_required=True,
        measured_peak_ram_mb=None,
        measured_seconds_512=None,
        benchmark_status="target-hardware measurement required before enablement",
    ),
    HeavyModelRuntimeProfile(
        key="restoreformer_v13_asset",
        weight_bytes=290_785_322,
        sha256="07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0",
        allowed_blocks=("enhance", "inpaint"),
        provenance_class="generated-low-confidence",
        load_policy="single-heavy-model-sequential",
        identity_guardrail_required=True,
        rollback_required=True,
        measured_peak_ram_mb=None,
        measured_seconds_512=None,
        benchmark_status="target-hardware measurement required before enablement",
    ),
)


def heavy_profile_by_key() -> dict[str, HeavyModelRuntimeProfile]:
    return {item.key: item for item in OPTIONAL_HEAVY_RUNTIME_PROFILES}
