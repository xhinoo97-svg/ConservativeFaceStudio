# GPEN BFR-512 — REAL CPU VERTICAL SLICE

Evidence date: 2026-08-17

## Major phase report

CURRENT_HEAD: `eb9a47b9351e72b1773b9c4e759171dfe1ff0911` (tested research candidate before this report commit)

BRANCH: `research/paper-quality-local-v2`

PHASE: `3 — GPEN BFR-512 real vertical slice`

INHERITED_TESTS:
- Certified V1 base: historical same-candidate PASS on Windows #1195, Female-domain #463, Release Quality #13.
- Current v1.1 hotfix regressions remain red and isolated from this branch; they were not hidden or reused as evidence.
- V4 final holdout was not executed or used for tuning.

GPEN:
- status: `BENCHMARKING — REAL DEVELOPMENT-HOST CPU EXECUTION PASS`
- upstream source commit: `2c736702983368847fb544d234a22ac7cff25802`
- official checkpoint source: `GPEN-BFR-512.pth`
- checkpoint bytes: `284085738`
- observed checkpoint SHA-256: `f1002c41add95b0decad69604d80455576f7187dd99ca16bd611bcfd44c10b51`
- upstream expected SHA-256: `NOT PUBLISHED / NOT VERIFIED`
- backend: `PyTorch 2.13.0+cpu`
- batch: `1`
- threads: `4`
- model-load seconds: `2.701020954`
- measured 512x512 inference seconds after warm-up: `2.696852690`
- baseline RSS: `408.230 MB`
- peak model-load RSS: `1041.371 MB`
- RSS after model load: `775.063 MB`
- peak inference RSS: `1827.684 MB`
- RSS after unload + Python GC: `899.570 MB`
- SFace clean vs degraded: `0.959352434`
- SFace clean vs GPEN: `0.953970492`
- frozen SFace gate: `0.363`
- identity gate: `PASS`
- degraded PSNR vs clean: `32.970810 dB`
- GPEN PSNR vs clean: `28.066682 dB`
- degraded SSIM vs clean: `0.901944`
- GPEN SSIM vs clean: `0.747439`
- restored output SHA-256: `940736223306aecda7f4fab2666d8baeccb2727804867d0eaf0e74064993bb7f`
- Action run: `32063123272`
- evidence artifact: `gpen-bfr-512-real-cpu-2`, artifact id `9298948446`

QUALITY:
- The non-cherry-picked comparison shows a large increase in perceived sharpness around eyes, brows, facial edges and head-scarf texture relative to the synthetic degraded MAIN.
- The generated result is not pixel-faithful to the clean target: PSNR and SSIM both decrease substantially.
- This is evidence that GPEN can be valuable as a Paper Quality candidate but must not run blindly on mild damage and must never be represented as observed recovery.
- This single case does not establish that GPEN is the best restorer for any degradation family.

GFPGAN_1_4:
- status: `NOT STARTED AT TIME OF GPEN TEST`
- backend: `N/A`
- peak_RAM: `N/A`
- seconds_512: `N/A`
- quality: `N/A`

CODEFORMER:
- status: `NOT STARTED`
- backend: `N/A`
- peak_RAM: `N/A`
- seconds_512: `N/A`
- quality: `N/A`

FBCNN:
- status: `NOT STARTED`

DAMAGE_MASK_NET:
- status: `NOT STARTED`
- IoU/F1 per damage class: `NOT MEASURED`

REFERENCE_BANK:
- status: `V1 foundation present; Paper Quality extension NOT STARTED`
- Existing component/reference bank remains the intended basis for personalized per-component routing.

IDENTITY:
- status: `GPEN DEVELOPMENT CASE PASS`
- backend: OpenCV Zoo SFace
- measured clean-vs-GPEN cosine: `0.953970492`
- threshold: `0.363`, unchanged

WRONG_PERSON_OBSERVED_PIXELS:
- `N/A — no reference donor was used by this standalone GPEN experiment`

PROVENANCE_VIOLATIONS:
- `NOT EVALUATED in the standalone GPEN slice`; integration into the product provenance map has not yet occurred.

RUNTIME_ERRORS:
- Successful run #2: `0` runtime errors.
- Earlier attempt #1 failed before GPEN load because `onnxruntime` was absent from the isolated research environment; the only change was adding that existing CFS runtime dependency.

BEST_MODEL_BY_DEGRADATION:
- blur: `NOT ESTABLISHED`
- noise: `NOT ESTABLISHED`
- jpeg: `NOT ESTABLISHED`
- low_light: `NOT ESTABLISHED`
- mosaic: `NOT ESTABLISHED`
- scribble: `NOT ESTABLISHED`
- sticker: `NOT ESTABLISHED`
- mixed: `NOT ESTABLISHED`

FORENSIC_MODE_READY:
- `TRUE for the certified V1 baseline only`; this research branch has not weakened or replaced the forensic pipeline.

PAPER_QUALITY_MODE_READY:
- `FALSE`

WINDOWS_INSTALLER_READY:
- `FALSE for Paper Quality mode`

TARGET_HARDWARE_READY:
- `FALSE`

PROJECT_FINISHED:
- `FALSE`

NEXT_BLOCKER:
1. GPEN code/weights redistribution license and authoritative expected checkpoint digest are not sufficiently explicit for production redistribution qualification.
2. The result is only one Linux development-host CPU case, not Windows or EliteBook evidence.
3. Post-unload RSS is higher than baseline; repeated load/infer/unload cycles are required before classifying this as allocator retention or a model-lifecycle leak.
4. GPEN should be compared against GFPGAN v1.4 under the same image/degradation/alignment/SFace/report contract before any model-selection conclusion.

## Host scope

The measured host was an Azure GitHub Actions Linux runner:
- 2 physical / 4 logical CPUs
- approximately 15.615 GiB RAM
- Python 3.11.15

These numbers are **not** HP EliteBook 1030 G3 measurements and **not** Windows measurements.

## Evidence interpretation

GPEN has now genuinely produced a local CPU face restoration and passed the unchanged identity gate. This satisfies the Phase 3 vertical-slice prerequisite to begin Phase 4 research on GFPGAN v1.4. It does **not** satisfy `PAPER_QUALITY_READY`, `WINDOWS_INSTALLER_READY`, `TARGET_HARDWARE_READY`, or production `QUALIFIED` status.
