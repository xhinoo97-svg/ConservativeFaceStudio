from __future__ import annotations

DAMAGE_CLASSES: tuple[str, ...] = (
    "HEALTHY",
    "BLUR",
    "MOTION_BLUR",
    "PIXELATION",
    "BLOCK_MOSAIC",
    "JPEG_ARTIFACT",
    "SCRIBBLE",
    "STICKER",
    "OPAQUE_BLOCK",
    "BLACK_BAR",
    "PARTIAL_OCCLUSION",
    "MISSING_COMPONENT",
)

CLASS_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(DAMAGE_CLASSES)}
INDEX_TO_CLASS: dict[int, str] = {index: name for name, index in CLASS_TO_INDEX.items()}
HEALTHY_INDEX = CLASS_TO_INDEX["HEALTHY"]


def validate_damage_class(name: str) -> str:
    value = str(name).upper()
    if value not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown damage class: {name}")
    return value
