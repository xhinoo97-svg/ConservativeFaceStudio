from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace


def sample_image() -> np.ndarray:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(image, (48, 48), 30, (130, 170, 200), -1)
    cv2.line(image, (20, 20), (76, 76), (255, 255, 255), 2)
    cv2.line(image, (76, 20), (20, 76), (70, 70, 70), 2)
    return image


def test_automatic_pipeline_exports_every_block(tmp_path: Path) -> None:
    output = tmp_path / "final.png"
    result = AutomaticPipelineRunner(Workspace(primary=sample_image())).run(output, upscale=1)

    assert result.final_image.exists()
    assert result.blocks_zip.exists()
    assert result.provenance is not None and result.provenance.exists()
    assert len(result.results) == 13
    by_block = {item.block: item for item in result.results}
    assert by_block["enhance"].details.get("blend") == 0.0

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
    result = AutomaticPipelineRunner(Workspace(primary=primary, references=[reference])).run(
        tmp_path / "with-reference.png", upscale=1
    )
    by_block = {item.block: item for item in result.results}
    assert by_block["align"].details.get("skipped") is not True
    assert result.blocks_zip.exists()
