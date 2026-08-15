from __future__ import annotations

import cv2
import numpy as np

from app.practical_benchmark import PUBLIC_PORTRAITS, _score, make_scenarios


def _portrait() -> np.ndarray:
    image = np.full((160, 160, 3), 32, dtype=np.uint8)
    cv2.ellipse(image, (80, 82), (48, 60), 0, 0, 360, (145, 175, 205), -1)
    cv2.circle(image, (62, 68), 5, (25, 25, 25), -1)
    cv2.circle(image, (98, 68), 5, (25, 25, 25), -1)
    cv2.line(image, (80, 72), (80, 96), (70, 80, 95), 3)
    cv2.ellipse(image, (80, 110), (18, 7), 0, 0, 180, (45, 45, 90), 3)
    return image


def test_public_manifest_has_ten_unique_public_domain_portraits() -> None:
    assert len(PUBLIC_PORTRAITS) == 10
    assert len({item.key for item in PUBLIC_PORTRAITS}) == 10
    assert len({item.filename for item in PUBLIC_PORTRAITS}) == 10
    assert all(item.page_url.startswith("https://commons.wikimedia.org/") for item in PUBLIC_PORTRAITS)
    assert all("Public domain" in item.license for item in PUBLIC_PORTRAITS)


def test_full_practical_scenarios_cover_required_failure_modes() -> None:
    scenarios = make_scenarios(_portrait(), profile="full")
    by_name = {item.name: item for item in scenarios}
    assert len(scenarios) == 10
    assert {"gaussian_mild_single", "gaussian_heavy_single", "motion_blur_single", "noise_jpeg_single", "mosaic_single", "translucent_single", "opaque_sticker_single", "opaque_sticker_full_reference", "scribble_two_partial", "component_only_references"} == set(by_name)
    assert by_name["opaque_sticker_single"].opaque_without_evidence is True
    assert by_name["opaque_sticker_single"].recoverable is False
    assert len(by_name["scribble_two_partial"].references) == 2
    assert len(by_name["component_only_references"].references) == 3
    assert np.count_nonzero(by_name["opaque_sticker_full_reference"].damage_mask) > 0


def test_quick_profile_still_exercises_single_and_multi_reference_paths() -> None:
    scenarios = make_scenarios(_portrait(), profile="quick")
    assert len(scenarios) == 5
    assert any(len(item.references) == 0 for item in scenarios)
    assert any(len(item.references) == 1 for item in scenarios)
    assert any(len(item.references) >= 2 for item in scenarios)
    assert any(item.opaque_without_evidence for item in scenarios)


def test_conservative_score_rewards_identity_recovery_and_observed_provenance() -> None:
    strong, strong_components = _score(identity=0.95, ssim=0.96, damage_mae=2.0, outside_mae=1.0, generated_fraction=0.0)
    weak, _ = _score(identity=0.25, ssim=0.45, damage_mae=70.0, outside_mae=28.0, generated_fraction=0.05)
    generated, _ = _score(identity=0.95, ssim=0.96, damage_mae=2.0, outside_mae=1.0, generated_fraction=0.05)
    assert strong > 95.0
    assert weak < strong
    assert generated < strong
    assert set(strong_components) == {"identity", "ssim", "damaged_region_recovery", "outside_region_preservation", "provenance_discipline"}
