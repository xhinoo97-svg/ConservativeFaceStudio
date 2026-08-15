from __future__ import annotations

import numpy as np
import pytest
import cv2
import hashlib

from research.face_restoration_v2.dataset import build_development_dataset
from research.face_restoration_v2.degradations import apply_degradation
from research.face_restoration_v2.splits import validate_development_manifest, validate_identity_disjoint


def _sample() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:160, :160]
    image = np.stack(((x + y) % 256, (2 * x) % 256, (2 * y) % 256), axis=2).astype(np.uint8)
    mask = np.zeros((160, 160), dtype=np.uint8)
    yy, xx = np.ogrid[:160, :160]
    mask[((xx - 80) / 48) ** 2 + ((yy - 78) / 62) ** 2 <= 1] = 255
    return image, mask


@pytest.mark.parametrize("kind", [
    "gaussian_blur", "motion_blur", "defocus_blur", "anisotropic_blur", "resize_blur",
    "pixelation", "jpeg", "noise", "low_light", "marker_strokes",
    "scribble", "opaque_paint", "opaque_sticker", "blur_rectangle",
    "smartphone_mixed",
])
def test_v2_degradation_is_deterministic_face_only(kind: str) -> None:
    image, face = _sample()
    clean_target = image.copy()
    first, first_mask, record = apply_degradation(image, face, kind=kind, severity=4, seed=20260815)
    second, second_mask, _ = apply_degradation(image, face, kind=kind, severity=4, seed=20260815)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert record.damaged_pixels > 0
    assert record.face_target_fraction >= 0.95
    assert np.array_equal(first[first_mask == 0], image[first_mask == 0])
    assert np.array_equal(image, clean_target)


def test_different_seed_produces_valid_variation() -> None:
    image, face = _sample()
    first, first_mask, first_record = apply_degradation(image, face, kind="opaque_sticker", severity=4, seed=10)
    second, second_mask, second_record = apply_degradation(image, face, kind="opaque_sticker", severity=4, seed=11)
    assert not np.array_equal(first, second)
    assert first_record.face_target_fraction >= 0.95
    assert second_record.face_target_fraction >= 0.95
    assert np.array_equal(first[first_mask == 0], image[first_mask == 0])
    assert np.array_equal(second[second_mask == 0], image[second_mask == 0])


def test_extreme_opaque_without_reference_marks_abstention_target() -> None:
    image, face = _sample()
    _degraded, _mask, record = apply_degradation(image, face, kind="opaque_sticker", severity=5, seed=7)
    assert record.abstention_expected is True


def test_identity_split_leakage_is_rejected() -> None:
    base = {
        "clean_path": "clean.jpg", "face_mask_path": "mask.png", "clean_sha256": "a" * 64,
        "license": "CC0", "source_url": "https://example.test/image",
        "domain_label": "female", "seed": 1,
    }
    rows = [
        {**base, "sample_id": "a-1", "identity_id": "person-a", "split": "train"},
        {**base, "sample_id": "a-2", "identity_id": "person-a", "split": "validation"},
    ]
    with pytest.raises(ValueError, match="identity leakage"):
        validate_identity_disjoint(rows)


def test_development_manifest_rejects_holdout() -> None:
    row = {
        "sample_id": "h-1", "identity_id": "person-h", "split": "final_holdout",
        "clean_path": "clean.jpg", "face_mask_path": "mask.png", "clean_sha256": "a" * 64,
        "license": "CC0", "source_url": "https://example.test/image",
        "domain_label": "female", "seed": 1,
    }
    with pytest.raises(ValueError, match="final_holdout is prohibited"):
        validate_development_manifest([row])


def test_dataset_builder_separates_clean_and_degraded(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image, face = _sample()
    clean_path, mask_path = source / "clean.png", source / "mask.png"
    assert cv2.imwrite(str(clean_path), image)
    assert cv2.imwrite(str(mask_path), face)
    digest = hashlib.sha256(clean_path.read_bytes()).hexdigest()
    row = {
        "sample_id": "train-1", "identity_id": "person-train-1", "split": "train",
        "clean_path": "clean.png", "face_mask_path": "mask.png", "clean_sha256": digest,
        "license": "CC0", "source_url": "https://example.test/image",
        "domain_label": "female", "seed": 23,
    }
    output = tmp_path / "dataset"
    report = build_development_dataset(
        [row], source_root=source, output_root=output,
        kinds=("gaussian_blur", "opaque_sticker"), severities=(2, 5),
    )
    assert report["generated_pairs"] == 4
    assert report["final_holdout_present"] is False
    assert not (output / "final_holdout").exists()
    target = cv2.imread(str(output / "train/train-1/clean/target.png"), cv2.IMREAD_COLOR)
    assert np.array_equal(target, image)
    assert len(list((output / "train/train-1/degraded").glob("*.png"))) == 4
