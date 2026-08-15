#!/usr/bin/env python3
"""Audit Conservative Face Studio block archives for no-ops and contradictions."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _operations(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    project = manifest.get("project", {})
    values = project.get("operations", []) if isinstance(project, dict) else []
    return {
        str(item.get("block")): dict(item.get("parameters", {}))
        for item in values
        if isinstance(item, dict) and item.get("block")
    }


def _number(mapping: dict[str, Any], *names: str) -> int:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _comparison(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    same_shape = before.shape == after.shape
    result: dict[str, Any] = {
        "before_dimensions": [int(before.shape[1]), int(before.shape[0])],
        "after_dimensions": [int(after.shape[1]), int(after.shape[0])],
        "same_shape": same_shape,
    }
    if not same_shape:
        result.update({"mae": None, "changed_pixels": None, "changed_pixel_fraction": None})
        return result
    delta = np.abs(after.astype(np.int16) - before.astype(np.int16))
    changed = np.any(delta != 0, axis=2) if delta.ndim == 3 else delta != 0
    result.update(
        {
            "mae": float(np.mean(delta)),
            "changed_pixels": int(np.count_nonzero(changed)),
            "changed_pixel_fraction": float(np.mean(changed)),
        }
    )
    return result


def _final_provenance(root: Path) -> dict[str, Any]:
    candidates = sorted((root / "results").glob("*provenance*.json"))
    if not candidates:
        return {}
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def analyze(root: Path, label: str) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    snapshots = sorted(manifest.get("snapshots", []), key=lambda item: int(item["order"]))
    operations = _operations(manifest)
    images: list[tuple[dict[str, Any], np.ndarray]] = []
    for snapshot in snapshots:
        image = cv2.imread(str(root / "blocks" / snapshot["filename"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Unreadable block image: {snapshot['filename']}")
        images.append((snapshot, image))

    comparisons = []
    for index, (snapshot, image) in enumerate(images):
        if index == 0:
            metric = {
                "before_dimensions": None,
                "after_dimensions": [int(image.shape[1]), int(image.shape[0])],
                "same_shape": None,
                "mae": None,
                "changed_pixels": None,
                "changed_pixel_fraction": None,
            }
        else:
            metric = _comparison(images[index - 1][1], image)
        details = dict(snapshot.get("details", {}))
        comparisons.append(
            {
                "order": int(snapshot["order"]),
                "block": str(snapshot["block"]),
                "snapshot_sha256": str(snapshot["sha256"]),
                "metrics_from_previous": metric,
                "details": details,
            }
        )

    findings: list[dict[str, Any]] = []
    damage = max(
        (_number(value, "damage_mask_pixels", "damage_pixels", "target_pixels") for value in operations.values()),
        default=0,
    )
    inpaint = operations.get("inpaint", {})
    requested = _number(inpaint, "requested_pixels", "target_pixels")
    repaired = _number(inpaint, "repaired_pixels", "observed_pixels", "observed_repaired_pixels")
    generated = _number(inpaint, "generated_pixels")
    symmetry = _number(inpaint, "symmetry_pixels")
    unresolved_at_inpaint = _number(inpaint, "unresolved_pixels")
    provenance = _final_provenance(root)
    final_counts = provenance.get("counts", {}) if isinstance(provenance.get("counts"), dict) else {}
    final_unresolved = _number(final_counts, "unresolved")
    if (damage or requested) and unresolved_at_inpaint and final_unresolved == 0:
        findings.append(
            {
                "severity": "FAIL",
                "code": "UNRESOLVED_DISAPPEARED",
                "message": "Unrepaired damaged pixels disappeared from final unresolved accounting.",
                "damage_pixels": damage,
                "requested_pixels": requested,
                "unresolved_at_inpaint": unresolved_at_inpaint,
                "final_unresolved": final_unresolved,
            }
        )
    if damage and repaired == 0 and generated == 0 and symmetry == 0 and final_unresolved == 0:
        findings.append(
            {
                "severity": "FAIL",
                "code": "DAMAGE_RECLASSIFIED_AS_ORIGINAL",
                "message": "Damage had no repair source but final accounting reports no unresolved pixels.",
                "damage_pixels": damage,
            }
        )

    align = operations.get("align", {})
    region = operations.get("region_select", {})
    aligned = _number(align, "aligned", "aligned_references")
    transferred = _number(region, "transferred_pixels")
    if aligned == 0 and transferred > 0:
        findings.append({"severity": "HARD_FAIL", "code": "TRANSFER_WITHOUT_ALIGNED_REFERENCE"})

    upscale = operations.get("upscale", {})
    scale = _number(upscale, "scale", "requested_scale")
    if scale > 1 and len(images) >= 12:
        before_dims = images[10][1].shape[:2]
        after_dims = images[11][1].shape[:2]
        explicit = bool(upscale.get("rolled_back") or upscale.get("skipped"))
        if before_dims == after_dims and not explicit:
            findings.append({"severity": "FAIL", "code": "UPSCALE_NO_DIMENSION_CHANGE_WITHOUT_STATUS"})

    deblur = comparisons[1] if len(comparisons) > 1 else None
    if deblur is not None:
        changed = deblur["metrics_from_previous"].get("changed_pixels")
        engine = str(operations.get("deblur", {}).get("engine", ""))
        abstained = bool(operations.get("deblur", {}).get("abstained"))
        if "nafnet" in engine.lower() and changed == 0 and not abstained:
            findings.append(
                {
                    "severity": "FAIL",
                    "code": "NAFNET_REPORTED_BUT_DEBLUR_NOOP",
                    "message": "NAFNet was reported but Block 02 changed zero pixels without explicit abstention.",
                }
            )

    return {
        "case": label,
        "snapshot_count": len(images),
        "operations": operations,
        "final_provenance": provenance,
        "adjacent_block_comparisons": comparisons,
        "findings": findings,
        "status": "PASS" if not findings else "FAIL",
    }


def _extract(path: Path, temporary: Path) -> Path:
    if path.is_dir():
        return path
    destination = temporary / path.stem
    destination.mkdir(parents=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cfs-block-audit-") as name:
        temporary = Path(name)
        report = [analyze(_extract(path, temporary), path.name) for path in args.archives]
    payload = {"format": "ConservativeFaceStudio block archive audit", "version": 1, "cases": report}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if any(case["status"] != "PASS" for case in report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
