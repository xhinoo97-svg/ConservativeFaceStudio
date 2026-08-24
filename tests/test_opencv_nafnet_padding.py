from __future__ import annotations

from pathlib import Path

import numpy as np

from app.opencv_nafnet import NafNetDeblurEngine


def _engine_without_loading_model(tmp_path: Path) -> NafNetDeblurEngine:
    model = tmp_path / "nafnet.onnx"
    model.write_bytes(b"placeholder")
    return NafNetDeblurEngine(model, target="cpu", tile_size=128, overlap=16)


def test_small_input_is_padded_to_safe_nafnet_extent_and_cropped_back(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_without_loading_model(tmp_path)
    seen: dict[str, tuple[int, ...]] = {}

    def fake_forward(blob: np.ndarray) -> np.ndarray:
        seen["shape"] = tuple(int(value) for value in blob.shape)
        return blob.copy()

    monkeypatch.setattr(engine, "_forward", fake_forward)
    image = np.full((128, 160, 3), 96, dtype=np.uint8)
    result = engine.infer(image)

    assert engine.tile_size == NafNetDeblurEngine.MIN_INFERENCE_SIZE
    assert seen["shape"] == (1, 3, 384, 384)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_non_multiple_dimensions_are_padded_to_stride(tmp_path: Path, monkeypatch) -> None:
    engine = _engine_without_loading_model(tmp_path)
    seen: dict[str, tuple[int, ...]] = {}

    def fake_forward(blob: np.ndarray) -> np.ndarray:
        seen["shape"] = tuple(int(value) for value in blob.shape)
        return blob.copy()

    monkeypatch.setattr(engine, "_forward", fake_forward)
    image = np.full((401, 377, 3), 64, dtype=np.uint8)
    result = engine._infer_tile(image)

    assert seen["shape"] == (1, 3, 416, 384)
    assert result.shape == image.shape
