from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.core_models import ensure_core_pretrained_models
from app.evaluation import normalized_landmark_error, psnr, structural_similarity_global
from app.execution import Workspace
from app.practical_benchmark import GENERATED_PROVENANCE_CODE, SYMMETRY_PROVENANCE_CODE, Scenario, _fit_portrait, _masked_mae, _score, make_scenarios
from app.validation import identity_anchor_score


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_THUMBNAIL_WIDTH = 640
_ALLOWED_LICENSE_MARKERS = ("public domain", "cc0", "cc by", "cc-by", "cc by-sa", "cc-by-sa")


@dataclass(frozen=True)
class CuratedPortrait:
    key: str
    person: str
    search_query: str
    domain_note: str


# Curated by public identity/biographical metadata; no visual gender classification is used.
CURATED_FEMALE_DOMAIN: tuple[CuratedPortrait, ...] = (
    CuratedPortrait("eileen_collins", "Eileen Collins", "Eileen Collins NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("mae_jemison", "Mae Jemison", "Mae Jemison NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("sally_ride", "Sally Ride", "Sally Ride NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("katherine_johnson", "Katherine Johnson", "Katherine Johnson NASA portrait", "adult woman, mathematician"),
    CuratedPortrait("peggy_whitson", "Peggy Whitson", "Peggy Whitson NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("jessica_meir", "Jessica Meir", "Jessica Meir NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("christina_koch", "Christina Koch", "Christina Koch NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("nicole_mann", "Nicole Mann", "Nicole Mann NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("sunita_williams", "Sunita Williams", "Sunita Williams NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("anne_mcclain", "Anne McClain", "Anne McClain NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("shannon_lucid", "Shannon Lucid", "Shannon Lucid NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("kathryn_sullivan", "Kathryn Sullivan", "Kathryn Sullivan NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("ellen_ochoa", "Ellen Ochoa", "Ellen Ochoa NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("joan_higginbotham", "Joan Higginbotham", "Joan Higginbotham NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("stephanie_wilson", "Stephanie Wilson", "Stephanie Wilson NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("jeanette_epps", "Jeanette Epps", "Jeanette Epps NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("tracy_dyson", "Tracy Caldwell Dyson", "Tracy Caldwell Dyson NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("kathleen_rubins", "Kathleen Rubins", "Kathleen Rubins NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("karen_nyberg", "Karen Nyberg", "Karen Nyberg NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("megan_mcarthur", "Megan McArthur", "Megan McArthur NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("susan_helms", "Susan Helms", "Susan Helms NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("serena_aunon", "Serena Aunon-Chancellor", "Serena Aunon-Chancellor NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("jasmin_moghbeli", "Jasmin Moghbeli", "Jasmin Moghbeli NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("zena_cardman", "Zena Cardman", "Zena Cardman NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("jessica_watkins", "Jessica Watkins", "Jessica Watkins NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("kayla_barron", "Kayla Barron", "Kayla Barron NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("loral_ohara", "Loral O'Hara", "Loral O'Hara NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("nancy_currie", "Nancy Currie-Gregg", "Nancy Currie NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("mary_cleave", "Mary Cleave", "Mary Cleave NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("bonnie_dunbar", "Bonnie Dunbar", "Bonnie Dunbar NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("marsha_ivins", "Marsha Ivins", "Marsha Ivins NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("wendy_lawrence", "Wendy Lawrence", "Wendy Lawrence NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("sandra_magnus", "Sandra Magnus", "Sandra Magnus NASA portrait", "adult woman, astronaut"),
    CuratedPortrait("heide_piper", "Heidemarie Stefanyshyn-Piper", "Heidemarie Stefanyshyn-Piper NASA portrait", "adult woman, astronaut"),
)


def _request_json(params: dict[str, str], timeout: int = 45) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{COMMONS_API}?{query}", headers={"User-Agent": "ConservativeFaceStudio-domain-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _license_allowed(extmetadata: dict[str, Any]) -> bool:
    text = " ".join(
        str(extmetadata.get(key, {}).get("value", ""))
        for key in ("LicenseShortName", "UsageTerms", "Copyrighted")
    ).lower()
    return any(marker in text for marker in _ALLOWED_LICENSE_MARKERS) or "false" in str(extmetadata.get("Copyrighted", {}).get("value", "")).lower()


def resolve_commons_portrait(item: CuratedPortrait, *, timeout: int = 45) -> dict[str, Any]:
    search = _request_json({
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": item.search_query,
        "gsrnamespace": "6",
        "gsrlimit": "8",
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": str(COMMONS_THUMBNAIL_WIDTH),
    }, timeout=timeout)
    pages = list(search.get("query", {}).get("pages", {}).values())
    candidates: list[dict[str, Any]] = []
    for page in pages:
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        width, height = int(info.get("width", 0)), int(info.get("height", 0))
        extmetadata = info.get("extmetadata") or {}
        if min(width, height) < 256 or not _license_allowed(extmetadata):
            continue
        original_url = str(info.get("url") or "")
        thumbnail_url = str(info.get("thumburl") or "")
        download_url = thumbnail_url or original_url
        if not download_url:
            continue
        title = str(page.get("title", ""))
        page_url = "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe=":_()'-")
        candidates.append({
            "key": item.key,
            "person": item.person,
            "domain_note": item.domain_note,
            "query": item.search_query,
            "title": title,
            "page_url": page_url,
            "download_url": download_url,
            "download_kind": "thumbnail" if thumbnail_url else "original_fallback",
            "download_width": int(info.get("thumbwidth", width)),
            "download_height": int(info.get("thumbheight", height)),
            "original_url": original_url,
            "width": width,
            "height": height,
            "license": extmetadata.get("LicenseShortName", {}).get("value", "unknown"),
        })
    if not candidates:
        raise RuntimeError(f"Nessun ritratto Commons riutilizzabile risolto per {item.person}")
    candidates.sort(key=lambda row: (min(row["width"], row["height"]), row["width"] * row["height"]), reverse=True)
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_and_download(root: Path, *, limit: int = 30, timeout: int = 45) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root.mkdir(parents=True, exist_ok=True)
    resolved: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for item in CURATED_FEMALE_DOMAIN[: max(0, int(limit))]:
        try:
            row = resolve_commons_portrait(item, timeout=timeout)
            target = root / f"{item.key}.jpg"
            if not target.is_file():
                request = urllib.request.Request(str(row["download_url"]), headers={"User-Agent": "ConservativeFaceStudio-domain-benchmark/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    target.write_bytes(response.read())
            row["local_path"] = str(target)
            row["sha256"] = _sha256(target)
            resolved.append(row)
        except Exception as exc:
            errors.append({"key": item.key, "person": item.person, "error": str(exc)})
    (root / "female-domain-manifest.json").write_text(json.dumps({"resolved": resolved, "errors": errors}, indent=2, sort_keys=True), encoding="utf-8")
    return resolved, errors


def _component_masks_from_landmarks(shape: tuple[int, int], landmarks5: np.ndarray | None) -> dict[str, np.ndarray]:
    h, w = shape
    masks = {name: np.zeros((h, w), dtype=np.uint8) for name in ("eyes_brows", "nose", "philtrum", "lips", "chin_jaw", "face_edge")}
    if landmarks5 is None or np.asarray(landmarks5).shape != (5, 2):
        return masks
    pts = np.asarray(landmarks5, dtype=np.float32)
    left_eye, right_eye, nose, mouth_l, mouth_r = pts
    eye_dist = max(8.0, float(np.linalg.norm(right_eye - left_eye)))
    mouth_center = (mouth_l + mouth_r) * 0.5
    eye_center = (left_eye + right_eye) * 0.5
    face_center = (eye_center + mouth_center) * 0.5

    def ellipse(name: str, center: np.ndarray, ax: float, ay: float) -> None:
        cv2.ellipse(masks[name], (int(center[0]), int(center[1])), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 255, -1)

    ellipse("eyes_brows", eye_center - np.array([0.0, 0.12 * eye_dist], dtype=np.float32), 0.78 * eye_dist, 0.34 * eye_dist)
    ellipse("nose", nose, 0.30 * eye_dist, 0.42 * eye_dist)
    philtrum_center = nose * 0.40 + mouth_center * 0.60
    ellipse("philtrum", philtrum_center, 0.20 * eye_dist, 0.22 * eye_dist)
    ellipse("lips", mouth_center, 0.45 * eye_dist, 0.22 * eye_dist)
    chin_center = mouth_center + np.array([0.0, 0.48 * eye_dist], dtype=np.float32)
    ellipse("chin_jaw", chin_center, 0.72 * eye_dist, 0.40 * eye_dist)
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(outer, (int(face_center[0]), int(face_center[1] + 0.22 * eye_dist)), (int(1.18 * eye_dist), int(1.50 * eye_dist)), 0, 0, 360, 255, -1)
    cv2.ellipse(inner, (int(face_center[0]), int(face_center[1] + 0.22 * eye_dist)), (int(0.88 * eye_dist), int(1.18 * eye_dist)), 0, 0, 360, 255, -1)
    masks["face_edge"] = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
    return masks


def evaluate_domain_scenario(clean: np.ndarray, scenario: Scenario, output_dir: Path, *, core_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    case_dir = output_dir / scenario.name
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}
    if core_paths:
        metadata["core_model_paths"] = {key: str(value) for key, value in core_paths.items()}
    workspace = Workspace(primary=scenario.primary.copy(), references=[item.copy() for item in scenario.references], metadata=metadata)
    runner = AutomaticPipelineRunner(workspace)
    result = runner.run(case_dir / "final.png", upscale=1)
    final = cv2.imread(str(result.final_image), cv2.IMREAD_COLOR)
    if final is None:
        raise RuntimeError("Output benchmark non leggibile")
    if final.shape != clean.shape:
        raise RuntimeError(f"Output benchmark corrotto: atteso {clean.shape}, ottenuto {final.shape}")

    damage = scenario.damage_mask > 0
    outside = ~damage
    provenance = workspace.provenance_map
    if provenance is None or provenance.shape != clean.shape[:2]:
        raise RuntimeError("Mappa provenance mancante o corrotta")
    reference_pixels = (provenance > 0) & (provenance < SYMMETRY_PROVENANCE_CODE)
    generated_pixels = provenance == GENERATED_PROVENANCE_CODE
    damage_count = max(1, int(np.count_nonzero(damage)))
    damage_reference_coverage = float(np.count_nonzero(reference_pixels & damage) / damage_count) if scenario.references else None
    uncovered_damage_fraction = float(np.count_nonzero(damage & ~reference_pixels) / damage_count) if scenario.references else None
    delta = np.max(cv2.absdiff(scenario.primary, final), axis=2)
    outside_count = max(1, int(np.count_nonzero(outside)))
    outside_damage_change_fraction = float(np.count_nonzero(outside & (delta > 2)) / outside_count)

    after_ssim = structural_similarity_global(clean, final)
    damage_mae = _masked_mae(clean, final, scenario.damage_mask)
    outside_mae = _masked_mae(scenario.primary, final, cv2.bitwise_not(scenario.damage_mask))
    identity, identity_engine = identity_anchor_score(final, [clean], backend=workspace.metadata.get("_identity_backend"))
    generated_fraction = float(np.count_nonzero(generated_pixels) / max(1, provenance.size))
    score, components = _score(identity, after_ssim, damage_mae, outside_mae, generated_fraction)

    landmark_nme = None
    landmarks5 = None
    backend = workspace.metadata.get("_identity_backend")
    if backend is not None:
        try:
            a = backend.analyze(clean)
            b = backend.analyze(final)
            landmarks5 = getattr(a, "landmarks5", None)
            if landmarks5 is not None and getattr(b, "landmarks5", None) is not None:
                landmark_nme = normalized_landmark_error(np.asarray(landmarks5), np.asarray(b.landmarks5))
        except Exception:
            pass

    component_metrics: dict[str, Any] = {}
    for name, mask in _component_masks_from_landmarks(clean.shape[:2], None if landmarks5 is None else np.asarray(landmarks5)).items():
        active = (mask > 0) & damage
        count = int(np.count_nonzero(active))
        if count == 0:
            continue
        component_metrics[name] = {
            "damaged_pixels": count,
            "mae": float(np.mean(np.abs(clean[active].astype(np.float32) - final[active].astype(np.float32)))),
            "reference_coverage": float(np.count_nonzero(reference_pixels & active) / count) if scenario.references else None,
        }

    cv2.imwrite(str(case_dir / "primary.png"), scenario.primary)
    cv2.imwrite(str(case_dir / "ground-truth.png"), clean)
    cv2.imwrite(str(case_dir / "damage-mask.png"), scenario.damage_mask)
    diff = cv2.absdiff(clean, final)
    cv2.imwrite(str(case_dir / "diff-heatmap.png"), cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET))

    target95_applicable = bool(scenario.recoverable and not scenario.opaque_without_evidence)
    return {
        "scenario": scenario.name,
        "reference_count": len(scenario.references),
        "recoverable": scenario.recoverable,
        "opaque_without_evidence": scenario.opaque_without_evidence,
        "psnr_before": psnr(clean, scenario.primary),
        "psnr_after": psnr(clean, final),
        "ssim_after": after_ssim,
        "damage_mae_after": damage_mae,
        "outside_region_mae": outside_mae,
        "identity_similarity": identity,
        "identity_engine": identity_engine,
        "landmark_nme": landmark_nme,
        "damage_reference_coverage": damage_reference_coverage,
        "uncovered_damage_fraction": uncovered_damage_fraction,
        "outside_damage_change_fraction": outside_damage_change_fraction,
        "generated_fraction": generated_fraction,
        "component_metrics": component_metrics,
        "conservative_recovery_score": score,
        "score_components": components,
        "target95_applicable": target95_applicable,
        "target95_passed": bool(score >= 95.0) if target95_applicable else None,
    }


def run_domain_benchmark(output: Path, *, cache: Path, limit: int = 30, size: int = 320, profile: str = "quick") -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sources, resolution_errors = resolve_and_download(cache, limit=limit)
    bootstrap = ensure_core_pretrained_models(output / "core-models", timeout_seconds=60)
    report: dict[str, Any] = {
        "format": "ConservativeFaceStudio curated female-domain benchmark",
        "version": 1,
        "selection_policy": "Curated public biographical metadata only; no visual gender classification and no morphology priors.",
        "portrait_count": len(sources),
        "requested_portrait_count": min(limit, len(CURATED_FEMALE_DOMAIN)),
        "resolution_errors": resolution_errors,
        "sources": sources,
        "cases": [],
    }
    for portrait_index, item in enumerate(sources):
        image = cv2.imread(item["local_path"], cv2.IMREAD_COLOR)
        if image is None:
            report["cases"].append({"portrait": item["key"], "error": "decode failed"})
            continue
        clean = _fit_portrait(image, size=size)
        for scenario in make_scenarios(clean, seed=20260809 + portrait_index, profile=profile):
            try:
                metrics = evaluate_domain_scenario(clean, scenario, output / item["key"], core_paths=bootstrap.paths if bootstrap.ready else None)
                metrics["portrait"] = item["key"]
                metrics["person"] = item["person"]
                report["cases"].append(metrics)
            except Exception as exc:
                report["cases"].append({"portrait": item["key"], "person": item["person"], "scenario": scenario.name, "error": str(exc)})

    valid = [row for row in report["cases"] if "conservative_recovery_score" in row]
    applicable = [row for row in valid if row.get("target95_applicable")]
    ref_cases = [row for row in valid if row.get("damage_reference_coverage") is not None]
    report["summary"] = {
        "completed_cases": len(valid),
        "error_cases": len(report["cases"]) - len(valid),
        "mean_score_recoverable": float(np.mean([row["conservative_recovery_score"] for row in applicable])) if applicable else None,
        "target95_pass_count": int(sum(row.get("target95_passed") is True for row in applicable)),
        "target95_applicable_count": len(applicable),
        "target95_pass_rate": float(sum(row.get("target95_passed") is True for row in applicable) / len(applicable)) if applicable else None,
        "mean_damage_reference_coverage": float(np.mean([row["damage_reference_coverage"] for row in ref_cases])) if ref_cases else None,
        "mean_outside_damage_change_fraction": float(np.mean([row["outside_damage_change_fraction"] for row in valid])) if valid else None,
        "source_resolution_error_count": len(resolution_errors),
    }
    (output / "female-domain-benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    fields = ["portrait", "person", "scenario", "recoverable", "reference_count", "conservative_recovery_score", "identity_similarity", "ssim_after", "damage_mae_after", "landmark_nme", "damage_reference_coverage", "uncovered_damage_fraction", "outside_damage_change_fraction", "generated_fraction", "target95_applicable", "target95_passed", "error"]
    with (output / "female-domain-benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="female-domain-benchmark")
    parser.add_argument("--cache", default=".benchmark-cache/female-domain")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--require-sources", type=int, default=24)
    args = parser.parse_args()
    report = run_domain_benchmark(Path(args.output), cache=Path(args.cache), limit=args.limit, size=args.size, profile=args.profile)
    print(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))
    if int(report.get("portrait_count", 0)) < int(args.require_sources):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
