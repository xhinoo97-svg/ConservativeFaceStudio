from __future__ import annotations

import cv2
import numpy as np

from app.practical_benchmark_matrix import (
    REAL_POSE_REFERENCES,
    _disk_blur,
    _real_pose_scenario,
    make_extended_scenarios,
)


def _portrait() -> np.ndarray:
    image = np.full((160, 160, 3), 36, dtype=np.uint8)
    cv2.ellipse(image, (80, 82), (48, 60), 0, 0, 360, (145, 175, 205), -1)
    cv2.circle(image, (62, 68), 5, (25, 25, 25), -1)
    cv2.circle(image, (98, 68), 5, (25, 25, 25), -1)
    cv2.line(image, (80, 72), (80, 96), (70, 80, 95), 3)
    cv2.ellipse(image, (80, 110), (18, 7), 0, 0, 180, (45, 45, 90), 3)
    return image


def test_extended_matrix_covers_defocus_half_face_and_partial_reference_modes() -> None:
    scenarios = make_extended_scenarios(_portrait())
    by_name = {item.name: item for item in scenarios}
    assert set(by_name) == {
        "defocus_mild_single",
        "defocus_heavy_single",
        "half_face_opaque_single",
        "eye_only_reference",
        "nose_only_reference",
        "mouth_chin_only_reference",
        "two_partial_crops",
        "multi_reference_complementary",
    }
    assert by_name["half_face_opaque_single"].opaque_without_evidence is True
    assert by_name["half_face_opaque_single"].recoverable is False
    assert len(by_name["eye_only_reference"].references) == 1
    assert len(by_name["two_partial_crops"].references) == 2
    assert len(by_name["multi_reference_complementary"].references) == 3
    assert all(np.count_nonzero(item.damage_mask) > 0 for item in scenarios)


def test_disk_defocus_is_deterministic_and_heavier_radius_changes_more() -> None:
    clean = _portrait()
    mild_a = _disk_blur(clean, 3)
    mild_b = _disk_blur(clean, 3)
    heavy = _disk_blur(clean, 7)
    assert np.array_equal(mild_a, mild_b)
    mild_error = float(np.mean(np.abs(clean.astype(np.float32) - mild_a.astype(np.float32))))
    heavy_error = float(np.mean(np.abs(clean.astype(np.float32) - heavy.astype(np.float32))))
    assert mild_error > 0.0
    assert heavy_error > mild_error


def test_real_pose_sources_are_separate_public_domain_identity_references() -> None:
    assert set(REAL_POSE_REFERENCES) == {"mae_jemison", "sally_ride"}
    for identity_key, source in REAL_POSE_REFERENCES.items():
        assert source.key != identity_key
        assert "commons.wikimedia.org/wiki/File:" in source.page_url
        assert "Public domain" in source.license
        assert source.filename.lower().endswith(".jpg")


def test_real_pose_scenario_is_observed_alignment_stress_not_pixel_exact_target() -> None:
    clean = _portrait()
    reference = np.roll(clean, 3, axis=1)
    scenario = _real_pose_scenario(clean, reference)
    assert scenario.name == "real_same_identity_pose_reference"
    assert scenario.recoverable is False
    assert scenario.opaque_without_evidence is False
    assert len(scenario.references) == 1
    assert np.array_equal(scenario.references[0], reference)
    assert np.count_nonzero(scenario.damage_mask) > 0
