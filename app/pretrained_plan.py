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
        "deterministic-final",
        "Import is an I/O operation; learned weights would add risk without adding information.",
    ),
    BlockModelChoice(
        BlockKind.DEBLUR,
        ("opencv_nafnet_deblur", "restormer_motion_deblur"),
        "conservative OpenCV denoise/unsharp",
        "guarded-pretrained",
        "The normal production path uses the verified OpenCV Zoo NAFNet ONNX checkpoint with tiled inference and identity rollback. Restormer remains an optional heavier alternative.",
    ),
    BlockModelChoice(
        BlockKind.ENHANCE,
        ("zero_dce_plus",),
        "calibrated luminance/CLAHE blend",
        "deterministic-final-plus-conditional-pretrained",
        "Generic learned enhancement can change skin colour and contrast. The strict default is calibrated deterministic enhancement; Zero-DCE++ is only appropriate after a genuine low-light classification.",
    ),
    BlockModelChoice(
        BlockKind.LANDMARKS,
        ("opencv_yunet", "mediapipe_face_landmarker", "insightface_identity"),
        "OpenCV Haar refined 5-point geometry",
        "pretrained-default",
        "YuNet is the verified lightweight pretrained production detector and supplies five facial anchors. Dense/heavier models remain optional.",
    ),
    BlockModelChoice(
        BlockKind.ALIGN,
        ("opencv_yunet",),
        "RANSAC partial-affine plus ORB fallback",
        "pretrained-geometry-deterministic-warp",
        "Pretrained YuNet supplies facial geometry; the actual partial-affine/RANSAC image transform stays deterministic and auditable.",
    ),
    BlockModelChoice(
        BlockKind.OCCLUSION_MASK,
        ("face_parsing_resnet18_onnx",),
        "heuristic mask plus multi-reference consensus",
        "pretrained-semantic-plus-strict-consensus",
        "The verified CelebAMask-HQ ResNet18 ONNX model supplies semantic face/accessory masks. It never authorizes repair alone: same-identity reference consensus is still required.",
    ),
    BlockModelChoice(
        BlockKind.REGION_SELECT,
        ("opencv_yunet", "face_parsing_resnet18_onnx", "opencv_sface"),
        "DMD-inspired observed-pixel specific reference memory",
        "pretrained-assisted-strict-memory",
        "Region selection combines pretrained geometry/semantic/identity evidence with real observed pixels. A generative dictionary is deliberately not required for the strict path.",
    ),
    BlockModelChoice(
        BlockKind.INPAINT,
        ("lama_big",),
        "observed-reference repair",
        "strict-observed-plus-generative-optional",
        "Strict mode first repairs only from aligned real references. Pretrained LaMa is reserved for an explicit non-strict unresolved-area pass because it can synthesize content.",
    ),
    BlockModelChoice(
        BlockKind.FUSION,
        ("opencv_yunet", "face_parsing_resnet18_onnx", "opencv_sface"),
        "specific reference memory plus exact provenance map",
        "pretrained-assisted-strict-fusion",
        "Fusion itself is deterministic over observed source pixels; pretrained models decide geometry, semantic support and identity safety rather than inventing the final pixels.",
    ),
    BlockModelChoice(
        BlockKind.FRONTALIZE,
        ("head_pose_mobilenetv2_onnx", "3ddfa_mb1"),
        "2D roll-only normalization",
        "pretrained-pose-gated-geometry",
        "The production path uses a verified MobileNetV2 6D pose checkpoint. Yaw/pitch gate unsafe transforms; strict mode only performs supported 2D roll normalization and never hallucinates the hidden side.",
    ),
    BlockModelChoice(
        BlockKind.IDENTITY_CHECK,
        ("opencv_sface", "insightface_identity"),
        "LAB-histogram proxy",
        "pretrained-default",
        "Verified SFace embeddings are the normal production guardrail and are reused after transformation blocks; the histogram proxy exists only for offline/failure fallback.",
    ),
    BlockModelChoice(
        BlockKind.UPSCALE,
        ("realesrgan_x2plus",),
        "Lanczos4",
        "strict-deterministic-plus-generative-optional",
        "Strict mode keeps deterministic Lanczos because learned super-resolution can synthesize texture. Real-ESRGAN is an optional pretrained enhanced export and must pass identity validation.",
    ),
    BlockModelChoice(
        BlockKind.EXPORT,
        (),
        "lossless PNG/JPEG95 plus provenance ZIP",
        "deterministic-final",
        "Export must preserve pixels, per-block images, hashes and provenance; a learned model is neither useful nor appropriate.",
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
