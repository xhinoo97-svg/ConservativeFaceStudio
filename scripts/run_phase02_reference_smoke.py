"""Real-weight computational smoke; never a restoration-quality certification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: F401
import cv2

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace
from app.production_models import _verified_bundled_models


def run_smoke(image_path: Path, root: Path, output: Path) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Input image cannot be decoded: {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, 320 / max(height, width))
    image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    paths = _verified_bundled_models(root.resolve())
    if len(paths) != 6:
        raise RuntimeError("Smoke requires all six locally installed checksum-verified models")
    cases = []
    for count in (0, 1, 3):
        workspace = Workspace(
            primary=image.copy(),
            references=[image.copy() for _ in range(count)],
            metadata={
                "core_model_paths": {key: str(path) for key, path in paths.items()},
                "user_selected_primary": True,
                "primary_priority_policy": "fixed-photo-1-main-image",
            },
        )
        started = time.perf_counter()
        result = AutomaticPipelineRunner(workspace).run(output / f"refs-{count}" / "final.png", upscale=1)
        final = cv2.imread(str(result.final_image), cv2.IMREAD_COLOR)
        if final is None or final.shape != image.shape or len(result.results) != 13:
            raise RuntimeError(f"Invalid output or block count for MAIN+{count}")
        if workspace.provenance_map is None or workspace.provenance_map.shape != image.shape[:2]:
            raise RuntimeError(f"Invalid provenance dimensions for MAIN+{count}")
        if result.provenance is None or not result.provenance.is_file() or not result.blocks_zip.is_file():
            raise RuntimeError(f"Missing export evidence for MAIN+{count}")
        cases.append({
            "reference_count": count,
            "seconds": time.perf_counter() - started,
            "block_count": len(result.results),
            "output_sha256": hashlib.sha256(result.final_image.read_bytes()).hexdigest(),
            "direct_identity_scores": workspace.metadata.get("reference_identity_scores"),
            "face_backend": workspace.metadata.get("face_backend"),
            "blocks": [{
                "key": item.block,
                "engine": item.details.get("engine", item.details.get("backend")),
                "skipped": item.details.get("skipped", False),
                "rollback": item.details.get("rolled_back", False),
                "abstained": item.details.get("abstained", False),
            } for item in result.results],
        })
        print(f"MAIN+{count}: computational smoke PASS", flush=True)
    return {
        "fixture": "supplied photograph; duplicate references; clean input",
        "input_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "cases": cases,
        "runtime_errors": 0,
        "quality_qualified": False,
        "final_holdout_used": False,
        "wrong_person_pixels_measured": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run_smoke(args.image, args.root, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
