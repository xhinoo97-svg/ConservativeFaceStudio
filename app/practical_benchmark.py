from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from app.validation import identity_anchor_score


SYMMETRY_PROVENANCE_CODE = np.uint16(65534)
GENERATED_PROVENANCE_CODE = np.uint16(65535)


@dataclass(frozen=True)
class PortraitSource:
    key: str
    filename: str
    page_url: str
    license: str = "Public domain (NASA / US Government work)"

    @property
    def download_url(self) -> str:
        return "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(self.filename)


PUBLIC_PORTRAITS: tuple[PortraitSource, ...] = (
    PortraitSource("eileen_collins", "Eileen Collins, early NASA portrait.jpg", "https://commons.wikimedia.org/wiki/File:Eileen_Collins,_early_NASA_portrait.jpg"),
    PortraitSource("mae_jemison", "Mae-jemison.jpg", "https://commons.wikimedia.org/wiki/File:Mae-jemison.jpg"),
    PortraitSource("sally_ride", "Sally Ride 1979.jpg", "https://commons.wikimedia.org/wiki/File:Sally_Ride_1979.jpg"),
    PortraitSource("buzz_aldrin", "Buzz Aldrin.jpg", "https://commons.wikimedia.org/wiki/File:Buzz_Aldrin.jpg"),
    PortraitSource("neil_armstrong", "Portrait of Neil Armstrong.jpg", "https://commons.wikimedia.org/wiki/File:Portrait_of_Neil_Armstrong.jpg"),
    PortraitSource("katherine_johnson", "Katherine Johnson at NASA, in 1966.jpg", "https://commons.wikimedia.org/wiki/File:Katherine_Johnson_at_NASA,_in_1966.jpg"),
    PortraitSource("peggy_whitson", "PeggyWhitson-NASA.jpg", "https://commons.wikimedia.org/wiki/File:PeggyWhitson-NASA.jpg"),
    PortraitSource("victor_glover", "NASA Candidate Victor J Glover.jpg", "https://commons.wikimedia.org/wiki/File:NASA_Candidate_Victor_J_Glover.jpg"),
    PortraitSource("guion_bluford", "Guion Bluford.jpg", "https://commons.wikimedia.org/wiki/File:Guion_Bluford.jpg"),
    PortraitSource("jessica_meir", "Official portrait of NASA astronaut Jessica Meir wearing a spacesuit (jsc2025e078605 alt).jpg", "https://commons.wikimedia.org/wiki/File:Official_portrait_of_NASA_astronaut_Jessica_Meir_wearing_a_spacesuit_(jsc2025e078605_alt).jpg"),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    primary: np.ndarray
    references: tuple[np.ndarray, ...]
    damage_mask: np.ndarray
    recoverable: bool
    opaque_without_evidence: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(source: PortraitSource, root: Path, *, timeout: int = 60) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{source.key}.jpg"
    if not target.is_file():
        request = urllib.request.Request(source.download_url, headers={"User-Agent": "ConservativeFaceStudio-benchmark/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        target.write_bytes(payload)
    checksum = _sha256(target)
    return target, checksum


def download_public_portraits(root: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for source in PUBLIC_PORTRAITS[: max(0, int(limit))]:
        path, checksum = _download(source, root)
        resolved.append({
            "key": source.key,
            "filename": source.filename,
            "page_url": source.page_url,
            "download_url": source.download_url,
            "license": source.license,
            "sha256": checksum,
            "local_path": str(path),
        })
    (root / "resolved-manifest.json").write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def _fit_portrait(image: np.ndarray, size: int = 384) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Ritratto non valido")
    h, w = image.shape[:2]
    scale = min(1.0, float(size) / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if h > size or w > size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return image
    y = (size - h) // 2
    x = (size - w) // 2
    canvas[y : y + h, x : x + w] = image
    return canvas


def _motion_blur(image: np.ndarray, length: int = 13) -> np.ndarray:
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def _mosaic(image: np.ndarray, factor: int = 12) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(2, w // factor), max(2, h // factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _jpeg_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 7.0, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 42])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else noisy


def _ellipse_mask(shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(center[0] * w), int(center[1] * h)), (int(axes[0] * w), int(axes[1] * h)), 0, 0, 360, 255, -1)
    return mask


def _partial_reference(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(clean)
    result[mask > 0] = clean[mask > 0]
    return result


def make_scenarios(clean: np.ndarray, *, seed: int = 20260808, profile: str = "full") -> tuple[Scenario, ...]:
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    full = np.full((h, w), 255, dtype=np.uint8)
    sticker = _ellipse_mask((h, w), (0.50, 0.50), (0.16, 0.11))
    eye_band = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(eye_band, (int(.23*w), int(.31*h)), (int(.77*w), int(.48*h)), 255, -1)
    nose = _ellipse_mask((h, w), (0.50, 0.53), (0.11, 0.15))
    mouth = _ellipse_mask((h, w), (0.50, 0.68), (0.18, 0.10))
    left = np.zeros((h, w), dtype=np.uint8)
    left[:, : w // 2 + w // 20] = 255
    right = np.zeros((h, w), dtype=np.uint8)
    right[:, w // 2 - w // 20 :] = 255

    opaque = clean.copy()
    opaque[sticker > 0] = (18, 18, 18)

    # A case marked recoverable must have real source evidence for every damaged pixel.
    # The previous component case damaged the whole sticker although eye/nose/mouth
    # references covered only part of it, making a >=95 gate demand unsupported pixels.
    component_support = cv2.bitwise_or(cv2.bitwise_or(eye_band, nose), mouth)
    component_damage = cv2.bitwise_and(sticker, component_support)
    component_opaque = clean.copy()
    component_opaque[component_damage > 0] = (18, 18, 18)

    scribble = clean.copy()
    for offset in (-18, -6, 6, 18):
        cv2.line(scribble, (int(.33*w), int(.50*h)+offset), (int(.67*w), int(.43*h)+offset), (10, 10, 10), max(4, w // 65))
    scribble_mask = np.any(scribble != clean, axis=2).astype(np.uint8) * 255
    translucent = clean.copy()
    overlay = np.zeros_like(clean)
    overlay[:] = (210, 60, 180)
    alpha_mask = _ellipse_mask((h, w), (0.50, 0.50), (0.20, 0.15))
    blended = cv2.addWeighted(clean, 0.58, overlay, 0.42, 0)
    translucent[alpha_mask > 0] = blended[alpha_mask > 0]

    all_cases = (
        Scenario("gaussian_mild_single", cv2.GaussianBlur(clean, (7, 7), 1.4), (), full, True),
        Scenario("gaussian_heavy_single", cv2.GaussianBlur(clean, (17, 17), 4.2), (), full, True),
        Scenario("motion_blur_single", _motion_blur(clean, 15), (), full, True),
        Scenario("noise_jpeg_single", _jpeg_noise(clean, rng), (), full, True),
        Scenario("mosaic_single", _mosaic(clean, 14), (), full, True),
        Scenario("translucent_single", translucent, (), alpha_mask, True),
        Scenario("opaque_sticker_single", opaque, (), sticker, False, True),
        Scenario("opaque_sticker_full_reference", opaque, (clean.copy(),), sticker, True),
        Scenario("scribble_two_partial", scribble, (_partial_reference(clean, left), _partial_reference(clean, right)), scribble_mask, True),
        Scenario(
            "component_only_references",
            component_opaque,
            (_partial_reference(clean, eye_band), _partial_reference(clean, nose), _partial_reference(clean, mouth)),
            component_damage,
            True,
        ),
    )
    if profile == "quick":
        chosen = {"gaussian_heavy_single", "opaque_sticker_single", "opaque_sticker_full_reference", "scribble_two_partial", "component_only_references"}
        return tuple(item for item in all_cases if item.name in chosen)
    return all_cases


def _masked_mae(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    active = mask > 0
    if not np.any(active):
        return 0.0
    return float(np.mean(np.abs(reference[active].astype(np.float32) - candidate[active].astype(np.float32))))


def _iou(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> float | None:
    a = reference_mask > 0
    b = candidate_mask > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return None
    return float(np.count_nonzero(a & b) / union)


def _precision_recall(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> tuple[float | None, float | None]:
    truth = reference_mask > 0
    pred = candidate_mask > 0
    tp = int(np.count_nonzero(truth & pred))
    fp = int(np.count_nonzero(~truth & pred))
    fn = int(np.count_nonzero(truth & ~pred))
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    return precision, recall


def _score(identity: float, ssim: float, damage_mae: float, outside_mae: float, generated_fraction: float) -> tuple[float, dict[str, float]]:
    identity_component = float(np.clip((identity + 1.0) * 0.5, 0.0, 1.0))
    ssim_component = float(np.clip((ssim + 1.0) * 0.5, 0.0, 1.0))
    recovery_component = float(np.clip(1.0 - damage_mae / 80.0, 0.0, 1.0))
    preservation_component = float(np.clip(1.0 - outside_mae / 30.0, 0.0, 1.0))
    provenance_component = float(np.clip(1.0 - generated_fraction / 0.05, 0.0, 1.0))
    components = {
        "identity": identity_component,
        "ssim": ssim_component,
        "damaged_region_recovery": recovery_component,
        "outside_region_preservation": preservation_component,
        "provenance_discipline": provenance_component,
    }
    score = 100.0 * (
        0.35 * identity_component
        + 0.20 * ssim_component
        + 0.30 * recovery_component
        + 0.10 * preservation_component
        + 0.05 * provenance_component
    )
    return float(score), components


def _landmark_error(runner: AutomaticPipelineRunner, clean: np.ndarray, final: np.ndarray) -> tuple[float | None, str | None]:
    backend = runner.executor.workspace.metadata.get("_identity_backend")
    if backend is None:
        return None, None
    try:
        a = backend.analyze(clean)
        b = backend.analyze(final)
        pa = getattr(a, "landmarks5", None)
        pb = getattr(b, "landmarks5", None)
        if pa is None or pb is None:
            return None, getattr(backend, "name", None)
        return normalized_landmark_error(np.asarray(pa), np.asarray(pb)), getattr(backend, "name", None)
    except Exception:
        return None, getattr(backend, "name", None)


def evaluate_scenario(clean: np.ndarray, scenario: Scenario, output_dir: Path, *, core_paths: dict[str, Path] | None = None) -> dict[str, Any]:
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
        final = cv2.resize(final, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_AREA)

    before_psnr = psnr(clean, scenario.primary)
    after_psnr = psnr(clean, final)
    after_ssim = structural_similarity_global(clean, final)
    damage_mae_before = _masked_mae(clean, scenario.primary, scenario.damage_mask)
    damage_mae_after = _masked_mae(clean, final, scenario.damage_mask)
    outside = cv2.bitwise_not(scenario.damage_mask)
    outside_mae = _masked_mae(scenario.primary, final, outside)
    identity, identity_engine = identity_anchor_score(final, [clean], backend=runner.executor.workspace.metadata.get("_identity_backend"))
    landmark_error, landmark_engine = _landmark_error(runner, clean, final)

    predicted_mask = runner.executor.workspace.metadata.get("inpaint_target_mask")
    if not isinstance(predicted_mask, np.ndarray) or predicted_mask.shape != scenario.damage_mask.shape:
        predicted_mask = np.zeros_like(scenario.damage_mask)
    mask_iou = _iou(scenario.damage_mask, predicted_mask)
    mask_precision, mask_recall = _precision_recall(scenario.damage_mask, predicted_mask)

    provenance = runner.executor.workspace.provenance_map
    if provenance is None or provenance.shape != clean.shape[:2]:
        provenance = np.zeros(clean.shape[:2], dtype=np.uint16)
    pixels = max(1, provenance.size)
    generated_fraction = float(np.count_nonzero(provenance == GENERATED_PROVENANCE_CODE) / pixels)
    symmetry_fraction = float(np.count_nonzero(provenance == SYMMETRY_PROVENANCE_CODE) / pixels)
    reference_fraction = float(np.count_nonzero((provenance > 0) & (provenance < SYMMETRY_PROVENANCE_CODE)) / pixels)
    primary_fraction = float(np.count_nonzero(provenance == 0) / pixels)

    conservative_score, score_components = _score(identity, after_ssim, damage_mae_after, outside_mae, generated_fraction)
    target95_applicable = bool(scenario.recoverable and not scenario.opaque_without_evidence)
    target95_passed = bool(conservative_score >= 95.0) if target95_applicable else None
    abstained = all(item.details.get("generated_pixels", 0) in (0, None) for item in result.results if item.block == "inpaint")
    abstention_correct = bool(abstained) if scenario.opaque_without_evidence else None

    diff = cv2.absdiff(clean, final)
    heat = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    cv2.imwrite(str(case_dir / "primary.png"), scenario.primary)
    cv2.imwrite(str(case_dir / "ground-truth.png"), clean)
    cv2.imwrite(str(case_dir / "damage-mask.png"), scenario.damage_mask)
    cv2.imwrite(str(case_dir / "diff-heatmap.png"), heat)

    return {
        "scenario": scenario.name,
        "reference_count": len(scenario.references),
        "recoverable": scenario.recoverable,
        "opaque_without_evidence": scenario.opaque_without_evidence,
        "psnr_before": before_psnr,
        "psnr_after": after_psnr,
        "psnr_delta": float(after_psnr - before_psnr) if math.isfinite(before_psnr) and math.isfinite(after_psnr) else None,
        "ssim_after": after_ssim,
        "damage_mae_before": damage_mae_before,
        "damage_mae_after": damage_mae_after,
        "damage_mae_improvement": float(damage_mae_before - damage_mae_after),
        "outside_region_mae": outside_mae,
        "identity_similarity": identity,
        "identity_engine": identity_engine,
        "landmark_nme": landmark_error,
        "landmark_engine": landmark_engine,
        "occlusion_iou": mask_iou,
        "occlusion_precision": mask_precision,
        "occlusion_recall": mask_recall,
        "primary_fraction": primary_fraction,
        "reference_fraction": reference_fraction,
        "symmetry_fraction": symmetry_fraction,
        "generated_fraction": generated_fraction,
        "abstention_correct": abstention_correct,
        "conservative_recovery_score": conservative_score,
        "score_components": score_components,
        "target95_applicable": target95_applicable,
        "target95_passed": target95_passed,
        "restoration_case": runner.executor.workspace.metadata.get("restoration_case"),
        "blocks_zip": str(result.blocks_zip),
    }


def run_public_benchmark(output: Path, *, cache: Path, limit: int = 10, size: int = 384, profile: str = "quick") -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    resolved = download_public_portraits(cache, limit=limit)
    bootstrap = ensure_core_pretrained_models(output / "core-models", timeout_seconds=60)
    report: dict[str, Any] = {
        "format": "ConservativeFaceStudio practical public-portrait benchmark",
        "version": 2,
        "profile": profile,
        "portrait_count": len(resolved),
        "metric_note": "The 0-100 conservative recovery score is composite; 95 is a target only for cases where the supplied evidence covers the benchmark damage, never a universal identity percentage.",
        "score_weights": {"identity": 0.35, "ssim": 0.20, "damaged_region_recovery": 0.30, "outside_region_preservation": 0.10, "provenance_discipline": 0.05},
        "core_models_ready": bootstrap.ready,
        "core_model_errors": bootstrap.errors,
        "sources": resolved,
        "cases": [],
    }
    for portrait_index, item in enumerate(resolved):
        image = cv2.imread(item["local_path"], cv2.IMREAD_COLOR)
        if image is None:
            report["cases"].append({"portrait": item["key"], "error": "decode failed"})
            continue
        clean = _fit_portrait(image, size=size)
        portrait_dir = output / item["key"]
        for scenario in make_scenarios(clean, seed=20260808 + portrait_index, profile=profile):
            try:
                metrics = evaluate_scenario(clean, scenario, portrait_dir, core_paths=bootstrap.paths if bootstrap.ready else None)
                metrics["portrait"] = item["key"]
                report["cases"].append(metrics)
            except Exception as exc:
                report["cases"].append({"portrait": item["key"], "scenario": scenario.name, "recoverable": scenario.recoverable, "error": str(exc)})

    valid = [item for item in report["cases"] if "conservative_recovery_score" in item]
    applicable = [item for item in valid if item.get("target95_applicable")]
    report["summary"] = {
        "completed_cases": len(valid),
        "error_cases": len(report["cases"]) - len(valid),
        "mean_score_all": float(np.mean([item["conservative_recovery_score"] for item in valid])) if valid else None,
        "mean_score_recoverable": float(np.mean([item["conservative_recovery_score"] for item in applicable])) if applicable else None,
        "target95_pass_count": int(sum(item.get("target95_passed") is True for item in applicable)),
        "target95_applicable_count": len(applicable),
        "target95_pass_rate": float(sum(item.get("target95_passed") is True for item in applicable) / len(applicable)) if applicable else None,
    }
    (output / "practical-benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    fields = ["portrait", "scenario", "recoverable", "reference_count", "conservative_recovery_score", "psnr_before", "psnr_after", "ssim_after", "damage_mae_before", "damage_mae_after", "identity_similarity", "landmark_nme", "occlusion_iou", "reference_fraction", "symmetry_fraction", "generated_fraction", "target95_applicable", "target95_passed", "abstention_correct", "error"]
    with (output / "practical-benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="practical-benchmark")
    parser.add_argument("--cache", default=".benchmark-cache/public-portraits")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()
    report = run_public_benchmark(Path(args.output), cache=Path(args.cache), limit=args.limit, size=args.size, profile=args.profile)
    print(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))
    if args.fail_on_errors and int(report.get("summary", {}).get("error_cases", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())