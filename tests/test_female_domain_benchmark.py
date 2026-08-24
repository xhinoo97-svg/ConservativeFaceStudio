from __future__ import annotations

import numpy as np

from app import female_domain_benchmark as benchmark
from app.female_domain_benchmark import CURATED_FEMALE_DOMAIN, CuratedPortrait, _component_masks_from_landmarks, _license_allowed


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


def test_commons_resolver_requests_and_selects_bounded_thumbnail(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_request(params: dict[str, str], timeout: int = 45) -> dict:
        captured.update(params)
        return {
            "query": {
                "pages": {
                    "1": {
                        "title": "File:Portrait.jpg",
                        "imageinfo": [{
                            "url": "https://upload.wikimedia.org/original.jpg",
                            "thumburl": "https://upload.wikimedia.org/thumb/640px-Portrait.jpg",
                            "width": 4000,
                            "height": 5000,
                            "thumbwidth": 640,
                            "thumbheight": 800,
                            "extmetadata": {"LicenseShortName": {"value": "CC BY 4.0"}},
                        }],
                    }
                }
            }
        }

    monkeypatch.setattr(benchmark, "_request_json", fake_request)
    item = CuratedPortrait("portrait", "Portrait", "Portrait query", "domain")
    result = benchmark.resolve_commons_portrait(item)

    assert captured["iiurlwidth"] == str(benchmark.COMMONS_THUMBNAIL_WIDTH)
    assert result["download_url"].endswith("640px-Portrait.jpg")
    assert result["download_kind"] == "thumbnail"
    assert result["download_width"] == 640
    assert result["download_height"] == 800
    assert result["width"] == 4000
    assert result["height"] == 5000


def test_commons_resolver_falls_back_to_original_when_thumbnail_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_request_json",
        lambda params, timeout=45: {
            "query": {
                "pages": {
                    "1": {
                        "title": "File:Portrait.png",
                        "imageinfo": [{
                            "url": "https://upload.wikimedia.org/original.png",
                            "width": 800,
                            "height": 900,
                            "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
                        }],
                    }
                }
            }
        },
    )
    item = CuratedPortrait("portrait", "Portrait", "Portrait query", "domain")
    result = benchmark.resolve_commons_portrait(item)

    assert result["download_url"].endswith("original.png")
    assert result["download_kind"] == "original_fallback"
    assert result["download_width"] == 800
    assert result["download_height"] == 900
