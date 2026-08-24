from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app  # noqa: F401  # Exercise the production boot policy, not ambient defaults.
import cv2
import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace
from app.production_models import ensure_production_pretrained_models
from app.evidence_confidence import SYMMETRY_PROVENANCE_CODE


def _face(size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), 38, dtype=np.uint8)
    center = size // 2
    cv2.ellipse(image, (center, center + 3), (size // 4, size // 3), 0, 0, 360, (150, 174, 202), -1)
    cv2.circle(image, (center - 13, center - 10), 4, (25, 25, 25), -1)
    cv2.circle(image, (center + 13, center - 10), 4, (25, 25, 25), -1)
    cv2.line(image, (center, center - 2), (center, center + 15), (80, 90, 100), 2)
    cv2.line(image, (center - 12, center + 27), (center + 12, center + 27), (55, 55, 70), 2)
    return image


def run_reference_count_smoke(root: Path) -> dict[str, object]:
    bootstrap = ensure_production_pretrained_models(root, face_timeout_seconds=5, restoration_timeout_seconds=5)
    if not (bootstrap.face_ready and bootstrap.standard_ready and bootstrap.inpaint_ready):
        raise RuntimeError(f"Production model pack incomplete: {bootstrap.errors}")
    clean = _face()
    damaged = clean.copy()
    cv2.rectangle(damaged, (38, 42), (90, 72), (10, 10, 10), -1)
    toxic = np.full_like(clean, 240)
    cases: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="cfs-reference-counts-") as directory:
        output_root = Path(directory)
        for count in range(10):
            references = [clean.copy() for _ in range(max(0, count - 1))]
            if count:
                references.append(toxic.copy())
            workspace = Workspace(
                primary=damaged.copy(),
                references=references,
                metadata={
                    "core_model_paths": {key: str(path) for key, path in bootstrap.paths.items()},
                    "user_selected_primary": True,
                    "primary_priority_policy": "fixed-photo-1-main-image",
                },
            )
            result = AutomaticPipelineRunner(workspace).run(output_root / f"refs-{count}" / "final.png", upscale=1)
            final = cv2.imread(str(result.final_image), cv2.IMREAD_COLOR)
            if final is None or final.shape != damaged.shape:
                raise RuntimeError(f"Invalid output for reference count {count}")
            if workspace.provenance_map is None or workspace.provenance_map.shape != damaged.shape[:2]:
                raise RuntimeError(f"Invalid provenance for reference count {count}")
            if workspace.metadata.get("primary_priority_policy") != "fixed-photo-1-main-image":
                raise RuntimeError(f"MAIN contract changed for reference count {count}")
            observed_sources = workspace.provenance_map[
                (workspace.provenance_map > 0) & (workspace.provenance_map < SYMMETRY_PROVENANCE_CODE)
            ]
            if count and np.any(observed_sources == count):
                raise RuntimeError(f"Incompatible final reference contaminated provenance for count {count}")
            cases.append({
                "reference_count": count,
                "output_shape": list(final.shape),
                "result_blocks": len(result.results),
                "main_contract": True,
                "provenance_valid": True,
                "incompatible_reference_abstained": bool(count == 0 or not np.any(observed_sources == count)),
            })
    return {"status": "PASS", "cases": cases, "counts": list(range(10))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reference-count-smoke.json")
    args = parser.parse_args()
    report = run_reference_count_smoke(Path(args.root).resolve())
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
