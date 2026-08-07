from __future__ import annotations

from dataclasses import dataclass

from app.pipeline import BlockKind


@dataclass(frozen=True)
class BlockModelChoice:
    block: BlockKind
    primary_models: tuple[str, ...]
    fallback: str
    execution_policy: str
    reason: str


PRETRAINED_BLOCK_PLAN: tuple[BlockModelChoice, ...] = (
    BlockModelChoice(
        BlockKind.IMPORT,
        (),
        "OpenCV/Pillow deterministic import",
        "deterministic",
        "Import/export should not depend on learned weights.",
    ),
    BlockModelChoice(
        BlockKind.DEBLUR,
        (
            "nafnet_gopro_width32",
            "nafnet_sidd_width32",
            "restormer_motion_deblur",
            "restormer_real_denoise",
        ),
        "conservative OpenCV denoise/unsharp",
        "guarded-pretrained",
        "NAFNet width32 is the CPU-first preference. Restormer remains a heavier quality alternative. Every neural result must pass the per-block identity guardrail.",
    ),
    BlockModelChoice(
        BlockKind.ENHANCE,
        ("mirnet_fivek_enhance", "zero_dce_plus"),
        "conservative CLAHE blend",
        "guarded-pretrained",
        "MIRNet FiveK is the general enhancement candidate; Zero-DCE++ is only appropriate when exposure analysis detects a genuinely low-light image.",
    ),
    BlockModelChoice(
        BlockKind.LANDMARKS,
        ("opencv_yunet", "mediapipe_face_landmarker", "insightface_identity"),
        "OpenCV Haar refined 5-point geometry",
        "pretrained-preferred",
        "YuNet is the tiny verified core detector with 5 landmarks and no extra runtime. MediaPipe adds dense landmarks; InsightFace remains a high-quality optional alternative.",
    ),
    BlockModelChoice(
        BlockKind.ALIGN,
        ("opencv_yunet", "mediapipe_face_landmarker", "insightface_identity"),
        "RANSAC partial-affine plus ORB fallback",
        "model-assisted-deterministic",
        "A pretrained detector supplies geometry, while the actual image transform remains deterministic and auditable.",
    ),
    BlockModelChoice(
        BlockKind.OCCLUSION_MASK,
        ("bisenet_face_parsing",),
        "heuristic mask plus multi-reference consensus",
        "model-assisted-strict",
        "BiSeNet face parsing supplies semantic facial support masks; reference consensus remains required before strict repair because parsing alone cannot prove a real occluder.",
    ),
    BlockModelChoice(
        BlockKind.REGION_SELECT,
        ("dmdnet",),
        "DMD-inspired observed-pixel specific reference memory",
        "strict-memory-plus-optional-model",
        "Strict mode uses same-identity observed pixels with provenance; DMDNet Full is an optional comparison/restoration backend rather than the source of strict pixels.",
    ),
    BlockModelChoice(
        BlockKind.INPAINT,
        ("lama_big",),
        "observed-reference repair",
        "generative-optional-only",
        "LaMa is useful when no reference observes a damaged area, but it can synthesize content and therefore cannot be the strict default.",
    ),
    BlockModelChoice(
        BlockKind.FUSION,
        ("dmdnet",),
        "specific reference memory plus exact provenance map",
        "strict-memory-plus-optional-model",
        "DMDNet's multi-reference specific-memory design is useful, while strict fusion keeps exact source provenance and agreement checks.",
    ),
    BlockModelChoice(
        BlockKind.FRONTALIZE,
        ("3ddfa_mb1",),
        "2D roll-only normalization",
        "model-assisted-geometry",
        "3DDFA_V2 supplies pretrained 3DMM pose/shape estimates and has an ONNX Runtime CPU path; unseen texture must not be synthesized in strict mode.",
    ),
    BlockModelChoice(
        BlockKind.IDENTITY_CHECK,
        ("opencv_sface", "insightface_identity"),
        "LAB-histogram proxy",
        "pretrained-required-for-high-confidence",
        "SFace gives the base installer a real pretrained identity embedding through OpenCV alone; InsightFace can be used as a heavier optional backend.",
    ),
    BlockModelChoice(
        BlockKind.UPSCALE,
        ("realesrgan_x2plus",),
        "Lanczos",
        "generative-optional-only",
        "Real-ESRGAN x2plus is useful as an optional learned upscaler; Lanczos remains the strict default because it does not synthesize texture.",
    ),
    BlockModelChoice(
        BlockKind.EXPORT,
        (),
        "lossless PNG/JPEG95 plus provenance ZIP",
        "deterministic",
        "Export should preserve results and metadata rather than invoke a learned model.",
    ),
)


def plan_by_block() -> dict[BlockKind, BlockModelChoice]:
    return {item.block: item for item in PRETRAINED_BLOCK_PLAN}


def validate_pretrained_plan() -> None:
    expected = set(BlockKind)
    actual = {item.block for item in PRETRAINED_BLOCK_PLAN}
    if actual != expected:
        missing = sorted(item.value for item in expected - actual)
        extra = sorted(item.value for item in actual - expected)
        raise ValueError(f"Pretrained plan incompleto: missing={missing}, extra={extra}")
    if len(PRETRAINED_BLOCK_PLAN) != len(expected):
        raise ValueError("Pretrained plan contiene blocchi duplicati")
