# GFPGAN v1.4 — REAL CPU VERTICAL SLICE

Evidence date: 2026-08-17

CURRENT_HEAD: `6ae82a157e803d41b3737b4b63f49bfa97927398` (tested research candidate before this report commit)

BRANCH: `research/paper-quality-local-v2`

PHASE: `4 — GFPGAN v1.4 real vertical slice`

INHERITED_TESTS:
- Certified V1 base remains the isolated base.
- v1.1 hotfix failures remain separate and unresolved; no failure was hidden.
- V4 final holdout was not executed or used.

GFPGAN_1_4:
- status: `BENCHMARKING — REAL DEVELOPMENT-HOST CPU EXECUTION PASS`
- exact upstream source commit: `7552a7791caad982045a7bbe5634bbf1cd5c8679`
- architecture: `GFPGANv1Clean`, channel multiplier `2`
- official release asset bytes: `348632874`
- observed checkpoint SHA-256: `e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad`
- upstream expected SHA-256: `NOT PUBLISHED / NOT VERIFIED`
- backend: `PyTorch 2.1.2+cpu`
- deterministic benchmark noise: `randomize_noise=False`
- batch: `1`, threads: `4`
- model-load seconds: `2.107661653`
- measured 512 inference seconds after warm-up: `2.786676334`
- baseline RSS: `389.090 MB`
- peak model-load RSS: `1132.137 MB`
- RSS after model load: `1004.336 MB`
- peak inference RSS: `1666.320 MB`
- post-unload + GC RSS: `1026.977 MB`
- SFace clean vs degraded: `0.959189982`
- SFace clean vs GFPGAN: `0.916649749`
- identity threshold: `0.363`, unchanged
- identity gate: `PASS`
- degraded PSNR: `32.970832 dB`
- GFPGAN PSNR: `30.647862 dB`
- degraded SSIM: `0.901944`
- GFPGAN SSIM: `0.860397`
- restored SHA-256: `0a033519ddf9e2da5734619ca8201eaf4c396f5fc49df0602acc5787b1db1cb8`
- accepted comparable run: `32064307147`
- evidence artifact: `gfpgan-v1-4-real-cpu-2`, id `9299323563`

GPEN:
- status: `BENCHMARKING`
- backend: `PyTorch CPU`
- peak_RAM: `1827.684 MB`
- seconds_512: `2.696853`
- quality: SFace `0.953970492`; PSNR `28.066682`; SSIM `0.747439`

DIRECT GPEN vs GFPGAN v1.4 — SAME INPUT / SAME ALIGNMENT:

| Metric | GPEN BFR-512 | GFPGAN v1.4 | Observation |
|---|---:|---:|---|
| SFace clean vs restored | 0.953970 | 0.916650 | GPEN retains more measured identity on this case |
| PSNR vs clean | 28.0667 dB | 30.6479 dB | GFPGAN is closer pixel-wise |
| SSIM vs clean | 0.747439 | 0.860397 | GFPGAN is structurally closer |
| inference 512 | 2.6969 s | 2.7867 s | essentially similar on these separate Azure CPU runs; not target-hardware evidence |
| peak inference RSS | 1827.7 MB | 1666.3 MB | GFPGAN lower by about 161.4 MB on these runs |

VISUAL QUALITY:
- GFPGAN removes the synthetic blur and produces a clean, smooth face, but suppresses some real fine texture visible in the clean target.
- GPEN reconstructs stronger apparent micro-detail and scarf/skin texture but deviates more from the actual clean pixels.
- Both are generative and therefore belong only to `GENERATED_MODEL_INFERRED` provenance when integrated.
- No production winner is selected from one face.

CODEFORMER:
- status: `NOT STARTED AT TIME OF THIS REPORT`

FBCNN:
- status: `NOT STARTED`

DAMAGE_MASK_NET:
- status: `NOT STARTED`
- IoU/F1: `NOT MEASURED`

REFERENCE_BANK:
- status: `V1 foundation present; paper-quality extension pending`

IDENTITY:
- status: `PASS on both GPEN and GFPGAN development slices with unchanged SFace threshold`

WRONG_PERSON_OBSERVED_PIXELS:
- `N/A — these blind-generator slices used no references`

PROVENANCE_VIOLATIONS:
- `NOT YET EVALUATED in product integration`; standalone generated outputs are explicitly treated as generated research candidates.

RUNTIME_ERRORS:
- accepted GFPGAN comparable run: `0`
- preliminary GFPGAN run also executed successfully but was excluded from model A/B because its interpolation/border convention differed from Phase 3.

BEST_MODEL_BY_DEGRADATION:
- blur: `NOT ESTABLISHED`
- noise: `NOT ESTABLISHED`
- jpeg: `NOT ESTABLISHED`
- low_light: `NOT ESTABLISHED`
- mosaic: `NOT ESTABLISHED`
- scribble: `NOT ESTABLISHED`
- sticker: `NOT ESTABLISHED`
- mixed: `NOT ESTABLISHED`

FORENSIC_MODE_READY: `TRUE for certified V1 baseline only`

PAPER_QUALITY_MODE_READY: `FALSE`

WINDOWS_INSTALLER_READY: `FALSE for Paper Quality mode`

TARGET_HARDWARE_READY: `FALSE`

PROJECT_FINISHED: `FALSE`

NEXT_BLOCKER:
- Phase 5 CodeFormer must be executed under the same input/alignment/SFace/report contract.
- GPEN/GFPGAN both require multi-case development/validation evidence and repeated load/unload memory-life-cycle measurement.
- No Azure-Linux result is target Windows/EliteBook acceptance evidence.
