from __future__ import annotations

from app.model_catalog import all_models_by_key
from app.model_registry import validate_manifest
from app.optional_heavy_models import OPTIONAL_HEAVY_MODELS, heavy_profile_by_key


def test_optional_heavy_models_are_never_strict_defaults() -> None:
    profiles = heavy_profile_by_key()
    assert {item.key for item in OPTIONAL_HEAVY_MODELS} == set(profiles)
    for manifest in OPTIONAL_HEAVY_MODELS:
        validate_manifest(manifest)
        assert manifest.conservative_default is False
        assert manifest.expected_sha256 is not None
        assert manifest.source_url is not None and manifest.source_url.startswith("https://github.com/")
        profile = profiles[manifest.key]
        assert profile.weight_bytes <= manifest.max_bytes
        assert profile.sha256 == manifest.expected_sha256
        assert profile.load_policy == "single-heavy-model-sequential"
        assert profile.provenance_class == "generated-low-confidence"
        assert profile.identity_guardrail_required is True
        assert profile.rollback_required is True
        assert set(profile.allowed_blocks) <= {"enhance", "inpaint"}


def test_optional_heavy_models_are_audited_but_not_base_installer_models() -> None:
    catalog = all_models_by_key()
    for key in ("codeformer_v010", "gfpgan_v13", "restoreformer_v13_asset"):
        assert key in catalog
        assert catalog[key].conservative_default is False
