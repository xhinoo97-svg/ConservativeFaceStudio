from __future__ import annotations

from app.model_runtime_registry import ACTIVE, FALLBACK, _declared_status


def test_only_explicit_release_models_can_be_active_or_fallback() -> None:
    for key in ACTIVE:
        assert _declared_status(key) == "ACTIVE"
    for key in FALLBACK:
        assert _declared_status(key) == "FALLBACK"


def test_unlisted_conservative_manifest_does_not_auto_promote() -> None:
    # These manifests are useful research candidates, but neither has the complete
    # release contract (production routing + packaged smoke) required for activation.
    assert _declared_status("nafnet_gopro_width32") == "TESTING"
    assert _declared_status("nafnet_sidd_width32") == "TESTING"
    assert _declared_status("mirnet_fivek_enhance") == "TESTING"
