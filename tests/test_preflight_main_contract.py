from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app.preflight as preflight
from app.execution import Workspace


class _FakeFaceEngine:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def analyze(self, image: np.ndarray):
        return SimpleNamespace(
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            bbox=(4, 4, image.shape[1] - 8, image.shape[0] - 8),
            score=0.95,
        )


def test_better_reference_can_be_analysis_anchor_but_never_replace_main(monkeypatch, tmp_path) -> None:
    main = np.full((48, 48, 3), 40, np.uint8)
    better_reference = np.full((48, 48, 3), 220, np.uint8)
    workspace = Workspace(primary=main.copy(), references=[better_reference.copy()])

    yunet = tmp_path / "yunet.onnx"
    sface = tmp_path / "sface.onnx"
    yunet.write_bytes(b"test")
    sface.write_bytes(b"test")

    monkeypatch.setattr(preflight, "OpenCVZooFaceEngine", _FakeFaceEngine)
    monkeypatch.setattr(
        preflight,
        "_deblur_all",
        lambda images, _model_path, _hardware: ([item.copy() for item in images], len(images)),
    )
    monkeypatch.setattr(
        preflight,
        "_quality_score",
        lambda image, _bbox, _score: float(np.mean(image) / 255.0),
    )

    result = preflight.preprocess_and_select_front_base(
        workspace,
        {"opencv_yunet": yunet, "opencv_sface": sface},
    )

    # The outer fixed-primary policy may expose source 0 as the final selection, but
    # the better source must still be retained as a donor/analysis recommendation.
    assert result.selected_source_index == 0
    assert workspace.metadata["preflight_recommended_front_source_index"] == 1
    assert workspace.metadata["best_reference_source_index"] == 1
    assert workspace.metadata["selected_primary_original_source_index"] == 0
    assert workspace.metadata["preflight_target_canvas_source_index"] == 0
    assert workspace.metadata["runtime_source_order"] == [0, 1]
    assert np.array_equal(workspace.primary, main)
    assert np.array_equal(workspace.references[0], better_reference)
