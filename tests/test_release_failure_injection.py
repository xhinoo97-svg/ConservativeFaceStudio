from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.execution import BlockExecutionError, Workspace
from app.imaging import read_image
from app.restoration import DeblurSettings, conservative_deblur
import app.update_manager as updates


def test_wrong_model_checksum_keeps_previous_working_weight(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "models" / "deblur" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"previous-working-weight")
    candidate = b"corrupt-candidate"
    monkeypatch.setattr(
        updates,
        "_download_https",
        lambda url, destination, **kwargs: destination.write_bytes(candidate),
    )
    entry = updates.ModelUpdateEntry(
        key="deblur",
        version="2.0.0",
        url="https://updates.example.test/model.onnx",
        sha256=hashlib.sha256(b"expected-weight").hexdigest(),
        destination="models/deblur/model.onnx",
        max_bytes=1024,
    )

    updater = updates.ModelUpdater(tmp_path, smoke_tests={"deblur": lambda path: None})
    with pytest.raises(updates.UpdateError, match="Checksum failed"):
        updater.install(entry)

    assert target.read_bytes() == b"previous-working-weight"
    assert not target.with_name("model.onnx.previous").exists()


def test_unreadable_reference_is_rejected_without_pixels(tmp_path: Path) -> None:
    path = tmp_path / "unreadable-reference.jpg"
    path.write_bytes(b"not an image")

    assert read_image(str(path)) is None


def test_tiny_main_is_processed_or_rejected_explicitly() -> None:
    tiny = np.full((1, 1, 3), 127, dtype=np.uint8)
    restored = conservative_deblur(tiny, DeblurSettings())

    assert restored.shape == tiny.shape
    assert restored.dtype == np.uint8

    empty = Workspace(primary=np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(BlockExecutionError, match="Immagine principale non valida"):
        empty.copy_primary()


def test_unicode_windows_style_path_and_exif_orientation_are_respected(tmp_path: Path) -> None:
    # Orientation 6 means the stored 4x2 raster is displayed as a 2x4 portrait.
    # cv2.imread applies EXIF orientation by default; the non-ASCII path exercises
    # the exact GUI loader used for MAIN and references on Windows.
    path = tmp_path / "Ritratti è人" / "foto à orientata.jpg"
    path.parent.mkdir()
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb[:, :2] = (255, 0, 0)
    rgb[:, 2:] = (0, 255, 0)
    image = Image.fromarray(rgb)
    exif = image.getexif()
    exif[274] = 6
    image.save(path, format="JPEG", quality=100, subsampling=0, exif=exif)

    loaded = read_image(str(path))

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (4, 2, 3)
    # BGR output: the rotated top remains the red half of the stored raster.
    assert float(np.mean(loaded[:2, :, 2])) > 240.0
    assert float(np.mean(loaded[2:, :, 1])) > 240.0
