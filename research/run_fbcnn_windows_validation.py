from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from statistics import mean

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.fbcnn_upstream_backend import APPROVED_CHECKPOINT_SHA256, PINNED_REVISION
from app.female_domain_benchmark import resolve_and_download
from scripts.run_female_domain_benchmark_resilient import _resilient_urlopen
from research.fbcnn_degradation_matrix import FBCNN_DEVELOPMENT_PROFILES

MIN_IDENTITIES = 8
IDENTITY_THRESHOLD = 0.363
MAX_RESOURCE_FRACTION = 0.80


def _lpips(metric, torch, a_path: Path, b_path: Path) -> float:
    a = cv2.imread(str(a_path), cv2.IMREAD_COLOR)
    b = cv2.imread(str(b_path), cv2.IMREAD_COLOR)
    if a is None or b is None or a.shape != b.shape:
        raise RuntimeError("LPIPS input missing or shape mismatch")

    def to_tensor(image):
        rgb = image[:, :, ::-1].copy()
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1.0).unsqueeze(0)

    with torch.inference_mode():
        value = metric(to_tensor(a), to_tensor(b))
    return float(value.detach().cpu().reshape(-1)[0].item())


def _system_fraction(report: dict) -> float:
    values = []
    for key in ("before_load", "after_load", "after_inference", "post_unload"):
        item = report.get("resource_budget", {}).get(key)
        if isinstance(item, dict):
            value = item.get("system_ram_fraction")
            if isinstance(value, (int, float)):
                values.append(float(value))
    return max(values) if values else 0.0


def _source_candidates(cache: Path, limit: int) -> tuple[list[dict], list[dict]]:
    urllib.request.urlopen = _resilient_urlopen
    sources, errors = resolve_and_download(cache, limit=limit)
    usable = []
    for source in sources:
        path = Path(str(source.get("local_path", "")))
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape[0] < 256 or image.shape[1] < 256:
            errors.append({"key": source.get("key"), "error": "decode_or_resolution_failed"})
            continue
        usable.append(source)
    return usable, errors


def _run_case(
    *,
    source: dict,
    profile_id: str,
    output: Path,
    fbcnn_root: Path,
    checkpoint: Path,
    core_models: Path,
) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "research" / "run_fbcnn_vertical_slice.py"),
        "--input", str(source["local_path"]),
        "--expected-input-sha256", str(source["sha256"]),
        "--fbcnn-root", str(fbcnn_root),
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", APPROVED_CHECKPOINT_SHA256,
        "--source-sha", PINNED_REVISION,
        "--degradation-profile", profile_id,
        "--core-model-root", str(core_models),
        "--threads", "64",
        "--output", str(output),
    ]
    import psutil

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    monitored = psutil.Process(process.pid)
    cpu_samples: list[float] = []
    monitored.cpu_percent(interval=None)
    while process.poll() is None:
        try:
            cpu_samples.append(float(monitored.cpu_percent(interval=0.10)))
        except (psutil.Error, ProcessLookupError):
            time.sleep(0.10)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"vertical_slice_exit={process.returncode}\n"
            f"stdout={stdout[-4000:]}\n"
            f"stderr={stderr[-4000:]}"
        )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    logical = max(1, int(os.cpu_count() or 1))
    report["cpu_observation"] = {
        "sample_count": len(cpu_samples),
        "peak_process_percent_of_one_core": max(cpu_samples) if cpu_samples else 0.0,
        "mean_process_percent_of_one_core": mean(cpu_samples) if cpu_samples else 0.0,
        "peak_fraction_of_total_logical_cpu": (
            max(cpu_samples) / (100.0 * logical) if cpu_samples else 0.0
        ),
        "mean_fraction_of_total_logical_cpu": (
            mean(cpu_samples) / (100.0 * logical) if cpu_samples else 0.0
        ),
    }
    return report


def _profile_summary(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for profile in FBCNN_DEVELOPMENT_PROFILES:
        selected = [row for row in rows if row["profile_id"] == profile.profile_id and not row.get("error")]
        if not selected:
            result[profile.profile_id] = {"case_count": 0, "pass": False}
            continue
        psnr_before = [float(r["metrics"]["psnr_degraded"]) for r in selected]
        psnr_after = [float(r["metrics"]["psnr_fbcnn"]) for r in selected]
        ssim_before = [float(r["metrics"]["ssim_degraded"]) for r in selected]
        ssim_after = [float(r["metrics"]["ssim_fbcnn"]) for r in selected]
        lpips_before = [float(r["lpips_before"]) for r in selected]
        lpips_after = [float(r["lpips_after"]) for r in selected]
        identity_after = [float(r["metrics"]["sface_clean_vs_fbcnn"]) for r in selected]
        profile_pass = (
            mean(psnr_after) > mean(psnr_before)
            and mean(ssim_after) > mean(ssim_before)
            and mean(lpips_after) < mean(lpips_before)
            and min(identity_after) >= IDENTITY_THRESHOLD
            and all(r["disposition"]["decision"] == "PASS" for r in selected)
        )
        result[profile.profile_id] = {
            "case_count": len(selected),
            "mean_psnr_before": mean(psnr_before),
            "mean_psnr_after": mean(psnr_after),
            "mean_ssim_before": mean(ssim_before),
            "mean_ssim_after": mean(ssim_after),
            "mean_lpips_before": mean(lpips_before),
            "mean_lpips_after": mean(lpips_after),
            "min_sface_after": min(identity_after),
            "mean_sface_after": mean(identity_after),
            "pass": bool(profile_pass),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbcnn-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--identities", type=int, default=MIN_IDENTITIES)
    parser.add_argument("--source-limit", type=int, default=14)
    args = parser.parse_args()

    if args.identities < MIN_IDENTITIES:
        raise RuntimeError(f"Phase02 requires at least {MIN_IDENTITIES} identities")

    args.output.mkdir(parents=True, exist_ok=True)
    sources, source_errors = _source_candidates(args.cache, args.source_limit)
    if len(sources) < args.identities:
        raise RuntimeError(f"Only {len(sources)} usable sources resolved; need {args.identities}")

    selected_sources = sources[: args.identities]
    rows = []
    core_models = args.output / "core-models"
    for source in selected_sources:
        for profile in FBCNN_DEVELOPMENT_PROFILES:
            case_dir = args.output / "cases" / str(source["key"]) / profile.profile_id
            row = {
                "identity_key": source["key"],
                "person": source.get("person"),
                "source_sha256": source["sha256"],
                "source_license": source.get("license"),
                "profile_id": profile.profile_id,
                "family": profile.family,
            }
            try:
                report = _run_case(
                    source=source,
                    profile_id=profile.profile_id,
                    output=case_dir,
                    fbcnn_root=args.fbcnn_root,
                    checkpoint=args.checkpoint,
                    core_models=core_models,
                )
                row["metrics"] = report["metrics"]
                row["disposition"] = report["disposition"]
                row["timing_seconds"] = report["timing_seconds"]
                row["rss_mb"] = report["rss_mb"]
                row["cpu_observation"] = report["cpu_observation"]
                row["system_ram_fraction_peak_snapshot"] = _system_fraction(report)
                row["model"] = report["model"]
                row["provenance"] = report["provenance"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    # Every FBCNN subprocess unloads its model before LPIPS starts. The two heavy
    # networks are therefore never resident together.
    import torch
    import lpips

    metric = lpips.LPIPS(net="alex").eval().to(torch.device("cpu"))
    for row in rows:
        if row.get("error"):
            continue
        case_dir = args.output / "cases" / str(row["identity_key"]) / str(row["profile_id"])
        row["lpips_before"] = _lpips(metric, torch, case_dir / "clean_aligned.png", case_dir / "degraded_aligned.png")
        row["lpips_after"] = _lpips(metric, torch, case_dir / "clean_aligned.png", case_dir / "restored_fbcnn_color.png")
    del metric

    errors = [row for row in rows if row.get("error")]
    profile_summary = _profile_summary(rows)
    max_system_fraction = max(
        [float(row.get("system_ram_fraction_peak_snapshot", 0.0)) for row in rows if not row.get("error")] or [0.0]
    )
    max_rss_mb = max(
        [float(row.get("rss_mb", {}).get("peak_inference", 0.0)) for row in rows if not row.get("error")] or [0.0]
    )
    peak_cpu_fraction = max(
        [float(row.get("cpu_observation", {}).get("peak_fraction_of_total_logical_cpu", 0.0))
         for row in rows if not row.get("error")] or [0.0]
    )
    real_sface = all(
        bool(row.get("metrics", {}).get("sface_identity_gate_pass"))
        for row in rows
        if not row.get("error")
    )
    provenance_ok = all(
        int(row.get("provenance", {}).get("wrong_person_final_pixels", -1)) == 0
        and int(row.get("provenance", {}).get("violations", -1)) == 0
        for row in rows
        if not row.get("error")
    )
    validation_pass = bool(
        len(selected_sources) >= MIN_IDENTITIES
        and len(rows) == len(selected_sources) * len(FBCNN_DEVELOPMENT_PROFILES)
        and not errors
        and real_sface
        and provenance_ok
        and max_system_fraction <= MAX_RESOURCE_FRACTION
        and peak_cpu_fraction <= MAX_RESOURCE_FRACTION
        and all(item.get("pass") is True for item in profile_summary.values())
    )

    payload = {
        "format": "ConservativeFaceStudio FBCNN Phase02 Windows multi-identity validation",
        "schema_version": 1,
        "evidence_tier": "VALIDATION",
        "candidate_sha": os.environ.get("GITHUB_SHA"),
        "identity_count": len(selected_sources),
        "case_count": len(rows),
        "error_count": len(errors),
        "official_repository": "jiaxi-jiang/FBCNN",
        "official_revision": PINNED_REVISION,
        "checkpoint_sha256": APPROVED_CHECKPOINT_SHA256,
        "code_license": "Apache-2.0",
        "weights_license_state": "APACHE_2_PROJECT_WIDE_NO_SEPARATE_RESTRICTION_FOUND",
        "weights_license_basis": [
            "official repository LICENSE is Apache-2.0 at pinned revision",
            "official README states the project is released under Apache-2.0",
            "fbcnn_color.pth is an official v1.0 release asset",
            "no separate restrictive checkpoint terms found in official LICENSE/README/release",
        ],
        "source_errors": source_errors,
        "sources": [
            {
                "key": s.get("key"),
                "person": s.get("person"),
                "sha256": s.get("sha256"),
                "license": s.get("license"),
                "page_url": s.get("page_url"),
            }
            for s in selected_sources
        ],
        "profile_summary": profile_summary,
        "max_process_rss_mb": max_rss_mb,
        "max_system_ram_fraction_snapshot": max_system_fraction,
        "peak_process_cpu_fraction_of_total_logical_cpu": peak_cpu_fraction,
        "resource_limit_fraction": MAX_RESOURCE_FRACTION,
        "wrong_person_final_pixels": 0 if provenance_ok else None,
        "provenance_violations": 0 if provenance_ok else None,
        "validation_gate_pass": validation_pass,
        "production_qualified": False,
        "production_blockers": [
            "installed offline same-candidate product evidence not yet complete",
            "physical HP EliteBook target-hardware evidence belongs to PHASE_08/13",
        ],
        "cases": rows,
    }
    (args.output / "fbcnn-phase02-windows-validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "identity_count": payload["identity_count"],
            "case_count": payload["case_count"],
            "error_count": payload["error_count"],
            "profile_summary": payload["profile_summary"],
            "max_process_rss_mb": payload["max_process_rss_mb"],
            "max_system_ram_fraction_snapshot": payload["max_system_ram_fraction_snapshot"],
            "peak_process_cpu_fraction_of_total_logical_cpu": payload["peak_process_cpu_fraction_of_total_logical_cpu"],
            "validation_gate_pass": payload["validation_gate_pass"],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if validation_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
