from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _module():
    sys.path.insert(0, str(RESEARCH))
    try:
        return importlib.import_module("evaluate_phase04_deeplab_checkpoint")
    finally:
        if sys.path and sys.path[0] == str(RESEARCH):
            sys.path.pop(0)


def _row(source_id: str, identity: str, split: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "filename": f"{source_id}.jpg",
        "clean_source_sha256": "0" * 64,
        "face_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        "dataset_split": split,
        "identity_key": identity,
    }


def test_validation_manifest_rejects_identity_leakage(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "final_holdout_used": False,
                "v3_used": False,
                "v4_used": False,
                "sources": [
                    _row("train-a", "person-a", "train"),
                    _row("val-a", "person-a", "validation"),
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Identity leakage"):
        module._validation_records(manifest)


def test_validation_manifest_is_fail_closed_for_final_holdout(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "final_holdout_used": True,
                "sources": [_row("val-a", "person-a", "validation")],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Final holdout is forbidden"):
        module._validation_records(manifest)


def test_binary_truth_is_derived_from_expanded_class_target() -> None:
    module = _module()
    target = np.zeros((3, 4), dtype=np.uint8)
    target[1, 2] = 7
    truth = module._binary_truth_from_target(target)
    assert truth.dtype == np.bool_
    assert int(np.count_nonzero(truth)) == 1
    assert bool(truth[1, 2]) is True
    assert bool(truth[0, 0]) is False
    with pytest.raises(ValueError, match="2D class-index map"):
        module._binary_truth_from_target(np.zeros((1, 3, 4), dtype=np.uint8))


def test_metric_math_matches_binary_confusion() -> None:
    module = _module()
    metrics = module._metrics({"tp": 8, "fp": 2, "fn": 2, "tn": 88})
    assert metrics["precision"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(0.8)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["iou"] == pytest.approx(8 / 12)
    assert metrics["false_positive_rate"] == pytest.approx(2 / 90)
    assert metrics["false_negative_rate"] == pytest.approx(0.2)
