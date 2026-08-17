from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app  # noqa: F401  # Apply the packaged OpenCV boot policy before importing cv2.
import cv2
import numpy as np

from app import female_domain_benchmark as benchmark
from app.face_analysis import choose_backend
from app.practical_benchmark import Scenario
from scripts.run_female_domain_benchmark_resilient import _resilient_urlopen


# Curated from public biographical metadata only. The list intentionally spans
# contemporary/vintage sources and different public portrait contexts. No image
# classifier is used to infer gender, age, skin tone, makeup, or morphology.
CURATED_FEMALE_DOMAIN = (
    benchmark.CuratedPortrait("eileen_collins", "Eileen Collins", "Eileen Collins NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("mae_jemison", "Mae Jemison", "Mae Jemison NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("sally_ride", "Sally Ride", "Sally Ride 1979 portrait", "adult woman; late-20th-century portrait"),
    benchmark.CuratedPortrait("katherine_johnson", "Katherine Johnson", "Katherine Johnson 1966 portrait", "adult woman; historical institutional portrait"),
    benchmark.CuratedPortrait("peggy_whitson", "Peggy Whitson", "Peggy Whitson NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("jessica_meir", "Jessica Meir", "Jessica Meir NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("sunita_williams", "Sunita Williams", "Sunita Williams NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("ellen_ochoa", "Ellen Ochoa", "Ellen Ochoa NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("stephanie_wilson", "Stephanie Wilson", "Stephanie Wilson NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("jeanette_epps", "Jeanette Epps", "Jeanette Epps NASA portrait", "adult woman; contemporary institutional portrait"),
    benchmark.CuratedPortrait("frida_kahlo", "Frida Kahlo", "Frida Kahlo portrait", "adult woman; vintage artist portrait"),
    benchmark.CuratedPortrait("marie_curie", "Marie Curie", "Marie Curie portrait", "adult woman; early photographic portrait"),
    benchmark.CuratedPortrait("rosa_parks", "Rosa Parks", "Rosa Parks portrait", "adult woman; historical documentary portrait"),
    benchmark.CuratedPortrait("maya_angelou", "Maya Angelou", "Maya Angelou portrait", "adult woman; literary portrait"),
    benchmark.CuratedPortrait("toni_morrison", "Toni Morrison", "Toni Morrison portrait", "adult woman; literary portrait"),
    benchmark.CuratedPortrait("audrey_hepburn", "Audrey Hepburn", "Audrey Hepburn portrait", "adult woman; vintage cinema portrait"),
    benchmark.CuratedPortrait("hedy_lamarr", "Hedy Lamarr", "Hedy Lamarr portrait", "adult woman; vintage studio portrait"),
    benchmark.CuratedPortrait("billie_holiday", "Billie Holiday", "Billie Holiday portrait", "adult woman; vintage performance portrait"),
    benchmark.CuratedPortrait("ella_fitzgerald", "Ella Fitzgerald", "Ella Fitzgerald portrait", "adult woman; vintage performance portrait"),
    benchmark.CuratedPortrait("josephine_baker", "Josephine Baker", "Josephine Baker portrait", "adult woman; vintage performance portrait"),
    benchmark.CuratedPortrait("amelia_earhart", "Amelia Earhart", "Amelia Earhart portrait", "adult woman; historical aviation portrait"),
    benchmark.CuratedPortrait("eleanor_roosevelt", "Eleanor Roosevelt", "Eleanor Roosevelt portrait", "adult woman; historical public portrait"),
    benchmark.CuratedPortrait("harriet_tubman", "Harriet Tubman", "Harriet Tubman portrait", "adult woman; 19th-century photographic portrait"),
    benchmark.CuratedPortrait("sojourner_truth", "Sojourner Truth", "Sojourner Truth portrait", "adult woman; 19th-century photographic portrait"),
    benchmark.CuratedPortrait("georgia_okeeffe", "Georgia O'Keeffe", "Georgia O'Keeffe portrait", "adult woman; artist portrait"),
    benchmark.CuratedPortrait("ruth_bader_ginsburg", "Ruth Bader Ginsburg", "Ruth Bader Ginsburg portrait glasses", "adult woman; portrait with eyewear"),
    benchmark.CuratedPortrait("shirley_chisholm", "Shirley Chisholm", "Shirley Chisholm portrait glasses", "adult woman; historical portrait with eyewear"),
    benchmark.CuratedPortrait("dorothy_vaughan", "Dorothy Vaughan", "Dorothy Vaughan NASA portrait", "adult woman; historical institutional portrait"),
    benchmark.CuratedPortrait("mary_jackson", "Mary Jackson", "Mary Jackson NASA portrait", "adult woman; historical institutional portrait"),
    benchmark.CuratedPortrait("annie_easley", "Annie Easley", "Annie Easley NASA portrait", "adult woman; historical institutional portrait"),
    benchmark.CuratedPortrait("nichelle_nichols", "Nichelle Nichols", "Nichelle Nichols portrait", "adult woman; studio/publicity portrait"),
    benchmark.CuratedPortrait("grace_hopper", "Grace Hopper", "Grace Hopper portrait", "adult woman; historical military portrait"),
    benchmark.CuratedPortrait("chavela_vargas", "Chavela Vargas", "Chavela Vargas portrait", "adult woman; performance portrait"),
    benchmark.CuratedPortrait("wangari_maathai", "Wangari Maathai", "Wangari Maathai portrait", "adult woman; contemporary documentary portrait"),
)


_BASE_MAKE_SCENARIOS = benchmark.make_scenarios
_BASE_RUN_DOMAIN_BENCHMARK = benchmark.run_domain_benchmark
_IDENTITY_GUARDRAIL_PREFIX = "Controllo identità SFace sotto soglia:"
try:
    _LANDMARK_BACKEND = choose_backend(prefer_embeddings=False)
except Exception:
    _LANDMARK_BACKEND = None


def _partial_reference(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(clean)
    result[mask > 0] = clean[mask > 0]
    return result


def _observed_component_scenario(clean: np.ndarray, fallback: Scenario) -> Scenario:
    if _LANDMARK_BACKEND is None:
        return fallback
    try:
        analysis = _LANDMARK_BACKEND.analyze(clean)
        landmarks5 = np.asarray(analysis.landmarks5, dtype=np.float32)
    except Exception:
        return fallback
    masks = benchmark._component_masks_from_landmarks(clean.shape[:2], landmarks5)
    eyes = masks["eyes_brows"]
    nose = masks["nose"]
    mouth_chin = cv2.bitwise_or(masks["philtrum"], masks["lips"])
    mouth_chin = cv2.bitwise_or(mouth_chin, masks["chin_jaw"])
    support = cv2.bitwise_or(cv2.bitwise_or(eyes, nose), mouth_chin)
    if int(np.count_nonzero(support)) < 64:
        return fallback
    primary = clean.copy()
    primary[support > 0] = (18, 18, 18)
    return Scenario(
        "component_only_references",
        primary,
        (
            _partial_reference(clean, eyes),
            _partial_reference(clean, nose),
            _partial_reference(clean, mouth_chin),
        ),
        support,
        True,
    )


def _observed_make_scenarios(clean: np.ndarray, *, seed: int = 20260808, profile: str = "full") -> tuple[Scenario, ...]:
    # Keep the production CI profile at exactly five cases per portrait (60-80
    # portraits => 300-400 cases), but alternate the severe low-quality single-image
    # case between heavy Gaussian blur and mosaic. This gives broad coverage without
    # increasing workflow time and directly exercises the V3 failure family.
    if profile == "quick":
        full = list(_BASE_MAKE_SCENARIOS(clean, seed=seed, profile="full"))
        alternating = "mosaic_single" if int(seed) % 2 else "gaussian_heavy_single"
        chosen = {
            alternating,
            "opaque_sticker_single",
            "opaque_sticker_full_reference",
            "scribble_two_partial",
            "component_only_references",
        }
        scenarios = [item for item in full if item.name in chosen]
        if len(scenarios) != 5:
            raise RuntimeError(f"Female-domain quick scenario drift: expected 5, got {len(scenarios)}")
    else:
        scenarios = list(_BASE_MAKE_SCENARIOS(clean, seed=seed, profile=profile))

    for index, scenario in enumerate(scenarios):
        if scenario.name == "component_only_references":
            scenarios[index] = _observed_component_scenario(clean, scenario)
            break
    return tuple(scenarios)


def _is_identity_guardrail_abstention(row: dict) -> bool:
    return str(row.get("error", "")).startswith(_IDENTITY_GUARDRAIL_PREFIX)


def _postprocess_guardrail_abstentions(report: dict, output: Path) -> dict:
    abstentions = 0
    for row in report.get("cases", []):
        if not _is_identity_guardrail_abstention(row):
            continue
        message = str(row.pop("error"))
        row["abstained"] = True
        row["abstention_reason"] = "identity_guardrail"
        row["abstention_detail"] = message
        row["target95_applicable"] = False
        row["target95_passed"] = None
        abstentions += 1

    summary = report.setdefault("summary", {})
    summary["abstention_count"] = abstentions
    summary["identity_guardrail_abstention_count"] = abstentions
    summary["error_cases"] = sum(bool(row.get("error")) for row in report.get("cases", []))

    output.mkdir(parents=True, exist_ok=True)
    (output / "female-domain-benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = [
        "portrait", "person", "scenario", "recoverable", "reference_count",
        "conservative_recovery_score", "identity_similarity", "ssim_after",
        "damage_mae_after", "landmark_nme", "damage_reference_coverage",
        "uncovered_damage_fraction", "outside_damage_change_fraction",
        "generated_fraction", "target95_applicable", "target95_passed",
        "abstained", "abstention_reason", "abstention_detail", "error",
    ]
    with (output / "female-domain-benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("cases", []))
    return report


def _observed_run_domain_benchmark(output: Path, *, cache: Path, limit: int = 30, size: int = 320, profile: str = "quick") -> dict:
    report = _BASE_RUN_DOMAIN_BENCHMARK(output, cache=cache, limit=limit, size=size, profile=profile)
    return _postprocess_guardrail_abstentions(report, output)


def main() -> int:
    urllib.request.urlopen = _resilient_urlopen
    benchmark.urllib.request.urlopen = _resilient_urlopen
    benchmark.CURATED_FEMALE_DOMAIN = CURATED_FEMALE_DOMAIN
    benchmark.make_scenarios = _observed_make_scenarios
    benchmark.run_domain_benchmark = _observed_run_domain_benchmark
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
