from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.execution import BlockExecutionError, BlockExecutor, Workspace
from app.pipeline import BlockKind, default_pipeline


def textured(size: int = 96) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(8, size - 8, 12):
        for x in range(8, size - 8, 12):
            cv2.circle(image, (x, y), 3, ((x * 3) % 255, (y * 5) % 255, 180), -1)
    cv2.rectangle(image, (24, 24), (72, 72), (110, 150, 190), 2)
    return image


def block(kind: BlockKind):
    return next(item for item in default_pipeline() if item.kind is kind)


def test_executor_runs_basic_cpu_pipeline(tmp_path: Path) -> None:
    primary = textured()
    executor = BlockExecutor(Workspace(primary=primary))
    executor.execute(block(BlockKind.IMPORT))
    executor.execute(block(BlockKind.DEBLUR), denoise=3, sharpen=0.5)
    executor.execute(block(BlockKind.ENHANCE))
    executor.execute(block(BlockKind.IDENTITY_CHECK))
    result = executor.execute(block(BlockKind.UPSCALE), scale=2)
    assert result.image.shape == (192, 192, 3)
    output = tmp_path / "result.png"
    exported = executor.execute(block(BlockKind.EXPORT), path=output)
    assert output.exists()
    assert (tmp_path / "result.png.provenance.json").exists()
    archive = Path(exported.details["blocks_zip"])
    assert archive.exists()
    assert exported.details["block_images"] == 6
    assert len(executor.project.operations) == 6

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        names = bundle.namelist()
        assert "manifest.json" in names
        block_images = [name for name in names if name.startswith("blocks/") and name.endswith(".png")]
        assert len(block_images) == 6
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["snapshot_count"] == 6
        assert [item["block"] for item in manifest["snapshots"]] == [
            "import", "deblur", "enhance", "identity_check", "upscale", "export"
        ]
        assert all(len(item["sha256"]) == 64 for item in manifest["snapshots"])


def test_custom_block_archive_path(tmp_path: Path) -> None:
    executor = BlockExecutor(Workspace(primary=textured()))
    executor.execute(block(BlockKind.IMPORT))
    output = tmp_path / "result.jpg"
    requested = tmp_path / "all-blocks.zip"
    result = executor.execute(block(BlockKind.EXPORT), path=output, blocks_zip=requested)
    assert Path(result.details["blocks_zip"]) == requested
    assert requested.exists()


def test_executor_undo_redo_roundtrip() -> None:
    primary = textured()
    executor = BlockExecutor(Workspace(primary=primary))
    changed = executor.execute(block(BlockKind.ENHANCE)).image
    restored = executor.undo()
    assert np.array_equal(restored, primary)
    redone = executor.redo()
    assert np.array_equal(redone, changed)


def test_region_selection_requires_references() -> None:
    executor = BlockExecutor(Workspace(primary=textured()))
    with pytest.raises(BlockExecutionError):
        executor.execute(block(BlockKind.REGION_SELECT))


def test_unsupported_external_block_is_explicit() -> None:
    executor = BlockExecutor(Workspace(primary=textured()))
    with pytest.raises(BlockExecutionError, match="modello esterno"):
        executor.execute(block(BlockKind.FRONTALIZE))


def test_identity_guard_rejects_dissimilar_reference() -> None:
    primary = textured()
    reference = np.full_like(primary, 255)
    executor = BlockExecutor(Workspace(primary=primary, references=[reference]))
    with pytest.raises(BlockExecutionError, match="sotto soglia"):
        executor.execute(block(BlockKind.IDENTITY_CHECK), minimum=0.95)


def test_alignment_and_selection_preserve_dimensions() -> None:
    primary = textured()
    matrix = np.float32([[1, 0, 3], [0, 1, -2]])
    reference = cv2.warpAffine(primary, matrix, (96, 96))
    executor = BlockExecutor(Workspace(primary=primary, references=[reference]))
    executor.execute(block(BlockKind.ALIGN))
    executor.execute(block(BlockKind.OCCLUSION_MASK))
    result = executor.execute(block(BlockKind.REGION_SELECT))
    assert result.image.shape == primary.shape
    assert executor.workspace.provenance_map is not None
    assert executor.workspace.provenance_map.shape == primary.shape[:2]
