from __future__ import annotations

from app.model_registry import OFFICIAL_MODELS, ModelManifest
from app.standard_pretrained import STANDARD_MODELS


def all_model_manifests() -> tuple[ModelManifest, ...]:
    """Return one de-duplicated catalog used by UI, selection and packaging."""
    merged: dict[str, ModelManifest] = {}
    for manifest in (*OFFICIAL_MODELS, *STANDARD_MODELS):
        merged[manifest.key] = manifest
    return tuple(merged.values())


def all_models_by_key() -> dict[str, ModelManifest]:
    return {item.key: item for item in all_model_manifests()}
