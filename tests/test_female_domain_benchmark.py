from __future__ import annotations

import numpy as np

from app.female_domain_benchmark import CURATED_FEMALE_DOMAIN, _component_masks_from_landmarks, _license_allowed


def test_curated_domain_has_at_least_thirty_unique_identities() -> None:
    assert len(CURATED_FEMALE_DOMAIN) >= 30
    assert len({item.key for item in CURATED_FEMALE_DOMAIN}) == len(CURATED_FEMALE_DOMAIN)
    assert len({item.person for item in CURATED_FEMALE_DOMAIN}) == len(CURATED_FEMALE_DOMAIN)


def test_license_filter_accepts_reusable_and_rejects_unknown() -> None:
    assert _license_allowed({"LicenseShortName": {"value": "Public domain"}})
    assert _license_allowed({"LicenseShortName": {"value": "CC BY-SA 4.0"}})
    assert not _license_allowed({"LicenseShortName": {"value": "All rights reserved"}})


def test_component_masks_are_geometry_driven_and_nonempty() -> None:
    landmarks = np.array(
        [[80.0, 90.0], [140.0, 90.0], [110.0, 125.0], [90.0, 155.0], [130.0, 155.0]],
        dtype=np.float32,
    )
    masks = _component_masks_from_landmarks((240, 220), landmarks)
    assert set(masks) == {"eyes_brows", "nose", "philtrum", "lips", "chin_jaw", "face_edge"}
    assert all(mask.shape == (240, 220) for mask in masks.values())
    assert all(np.count_nonzero(mask) > 0 for mask in masks.values())


def test_component_masks_abstain_without_landmarks() -> None:
    masks = _component_masks_from_landmarks((64, 64), None)
    assert all(np.count_nonzero(mask) == 0 for mask in masks.values())
