from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.execution import ExecutionResult, Workspace
from app.pipeline import BlockKind


def sample_image() -> np.ndarray:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(image, (48, 48), 30, (130, 170, 200), -1)
    cv2.line(image, (20, 20), (76, 76), (255, 255, 255), 2)
    cv2.line(image, (76, 20), (20, 76), (70, 70, 70), 2)
    return image


def _install_synthetic_landmark_handler(runner: AutomaticPipelineRunner) -> None:
    """Keep autorun plumbing tests independent from external face-model inference.

    Real YuNet/SFace/semantic model inference has dedicated CI smoke tests. These two
    tests use a deliberately synthetic drawing and only verify 13-block routing,
    automatic progression and export integrity, so deterministic geometry is the
    correct fixture instead of silently relying on a legacy Haar cascade.
    """
    primary_points = np.asarray(
        [[37.0, 41.0], [59.0, 41.0], [48.0, 53.0], [41.0, 65.0], [55.0, 65.0]],
        dtype=np.float32,
    )

    def landmarks(block, parameters):
        reference_points: list[np.ndarray] = []
        for _ in runner.executor.workspace.references:
            reference_points.append(primary_points.copy())
        runner.executor.workspace.metadata.update(
            {
                "primary_landmarks5": primary_points.copy(),
                "primary_bbox": (18, 18, 60, 60),
                "primary_landmark_confidence": 1.0,
                "reference_landmarks5": reference_points,
                "reference_landmark_confidence": [1.0] * len(reference_points),
                "face_backend": "synthetic-test-fixture",
            }
        )
        return ExecutionResult(
            block.key,
            runner.executor.workspace.copy_primary(),
            {
                "backend": "synthetic-test-fixture",
                "bbox": [18, 18, 60, 60],
                "landmark_count": 5,
                "landmark_confidence": 1.0,
                "reference_faces": len(reference_points),
            },
        )

    runner.executor._handlers[BlockKind.LANDMARKS] = landmarks


def test_automatic_pipeline_exports_every_block(tmp_path: Path) -> None:
    output = tmp_path / "final.png"
    runner = AutomaticPipelineRunner(Workspace(primary=sample_image()))
    _install_synthetic_landmark_handler(runner)
    result = runner.run(output, upscale=1)

    assert result.final_image.exists()
    assert result.blocks_zip.exists()
    assert result.provenance is not None and result.provenance.exists()
    assert len(result.results) == 13
    by_block = {item.block: item for item in result.results}
    enhance = by_block["enhance"]
    assert enhance.details.get("automatic_conservative") is True
    assert enhance.details.get("requested_blend") == 0.0
    assert float(enhance.details.get("effective_blend", 0.0)) > 0.0
    assert float(enhance.details.get("blend", 0.0)) > 0.0
    assert by_block["landmarks"].details.get("backend") == "synthetic-test-fixture"

    with zipfile.ZipFile(result.blocks_zip, "r") as archive:
        names = archive.namelist()
        block_images = [name for name in names if name.startswith("blocks/") and name.endswith(".png")]
        assert len(block_images) == 13
        assert f"results/{result.final_image.name}" in names
        assert f"results/{result.provenance.name}" in names

        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["version"] == 2
        assert manifest["snapshot_count"] == 13
        assert manifest["snapshots"][0]["block"] == "import"
        assert manifest["snapshots"][-1]["block"] == "export"
        attachments = {item["filename"]: item for item in manifest["attachments"]}
        assert result.final_image.name in attachments
        assert result.provenance.name in attachments

        final_bytes = archive.read(f"results/{result.final_image.name}")
        provenance_bytes = archive.read(f"results/{result.provenance.name}")
        assert hashlib.sha256(final_bytes).hexdigest() == attachments[result.final_image.name]["sha256"]
        assert hashlib.sha256(provenance_bytes).hexdigest() == attachments[result.provenance.name]["sha256"]


def test_automatic_pipeline_uses_references_without_confirmation(tmp_path: Path) -> None:
    primary = sample_image()
    matrix = np.float32([[1, 0, 2], [0, 1, -1]])
    reference = cv2.warpAffine(primary, matrix, (96, 96))
    runner = AutomaticPipelineRunner(Workspace(primary=primary, references=[reference]))
    _install_synthetic_landmark_handler(runner)
    result = runner.run(tmp_path / "with-reference.png", upscale=1)
    by_block = {item.block: item for item in result.results}
    assert by_block["align"].details.get("skipped") is not True
    assert result.blocks_zip.exists()
