# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** GitHub evidence overrides chat memory. Historical state remains preserved in Git history and in `project-state-history/`.

## 0. Current canonical state

Last ledger update: `2026-09-05T06:44Z` \
Technical state verified at: `2026-09-05T06:44Z` \
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_03_PAPER_QUALITY_RUNTIME_WIRING`  
PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**  
Last technical branch: `integration/final-paper-quality-local`  
Last technical HEAD: `c806f00e17aa074d906b6473fba7d5518e1afcd2` \
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
PR #2: OPEN + DRAFT + NOT_MERGED; preserve as non-certified historical candidate.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

No force-push, no history deletion, no automatic merge, no V3/V4 rerun, no fabricated evidence.

EXACT_NEXT_ACTION: on `integration/final-paper-quality-local`, constrain the installed FBCNN candidate image and `generated_mask` to the audited JPEG subroute before any future candidate selection; require `candidate_changed_outside_route_pixels=0` in regression and Windows DEVELOPMENT evidence, while keeping the candidate shadow-only and non-production. Do not enable Paper Quality by default or place the checkpoint in a distributable installer.

---

## 1. Installed-path truth

Installed entry path remains:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

The feature-flagged path now reaches a fail-closed installed bridge:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`

With `paper_quality_enabled=true`, Block 8 invokes `DamageMaskRuntime`, the damage router, model-qualification gate, `PersonalizedReferenceBank`, component selector, reference-first repair, calibrated candidate selector when configured, component-aware fusion, per-pixel provenance and the existing post-block identity rollback. The legacy Block-8 handler is not used as a silent fallback. The shipped default remains `paper_quality_enabled=false`.

Installed-worker E2E evidence on the exact published tree:

- structural desktop-entrypoint link checked from `app.__main__.main` through `MainWindow.start_pipeline`;
- dynamic execution starts at the real `PipelineWorker.run`, not at the Paper Quality helper;
- all Phase03 modules listed above are observed in the execution trace;
- identity guardrail executes and accepts only the verified observed-reference transfer;
- healthy pixels changed outside damage authority: `0`;
- wrong-person final pixels: `0`;
- provenance violations: `0`;
- generated pixels: `0`;
- the synthetic MAIN+1 DEV E2E executes no external pretrained model and proves wiring/authority only;
- Windows DEVELOPMENT run `33950404788` executes real LR-ASPP ONNX and official FBCNN through the same `PipelineWorker` path with MAIN+0;
- the FBCNN checkpoint hash is verified before load, only the JPEG subroute can load it, and the model is unloaded after inference;
- Block 8 preserves its accepted input byte-for-byte; the immutable imported MAIN remains separately auditable;
- the raw FBCNN candidate changed `3605` crop pixels, of which `3552` were outside the admitted JPEG subroute, so it correctly remained shadow-only and contributed zero final pixels.

The Windows run proves real installed-path execution, routing, lifecycle and fail-closed pixel authority on one public DEVELOPMENT portrait. It does not prove final restoration quality, offline-installer distribution, Clean Windows or physical EliteBook compatibility.

PAPER_QUALITY_RUNTIME_WIRED: **TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS**
PAPER_QUALITY_MODE_READY: **FALSE**
FBCNN_INSTALLED_PATH_STATUS: **WINDOWS_DEVELOPMENT_PASS / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED**

Published technical commits:

- `a73fa387b6d960d58ea41ca380e47c0716ce441f` — synchronize Phase02 FBCNN evidence without production promotion;
- `6f80e917ee3e9baad374d9488177cf34b2f0c48b` — install the feature-flagged Paper Quality bridge;
- `96b6aa4e858a7747bbf2ce52f51d81a9c10200e5` — prove the installed worker path and fail-closed behavior;
- `08b7d61a071ef1ca9583dae0d2bb89740184d989` — add the all-or-nothing DEVELOPMENT validation pack;
- `5449aa63ea515a3bed334f016c8bc44e4a15cfb3` — execute FBCNN only on an installed JPEG route;
- `45c228de55b4b3232b57ff0057a762e8b9bde217` — add the real Windows worker validation;
- `fe49201a7dc89c927e3685d5c1c80be97034e28d` — isolate dominant JPEG evidence inside a truthful MIXED parent route;
- `1930db5ffeecd9b3b2c6c83d20be357e4424a6d4` — pin the NumPy ABI required by PyTorch 2.1.2;
- `79d7a900b618031050760718d3b6531ec1cc3124` — install the exact torch/torchvision CPU pair;
- `c806f00e17aa074d906b6473fba7d5518e1afcd2` — preserve the accepted Block-8 context separately from immutable MAIN.

Verification on the same final tree:

- targeted exact-workflow Phase03 regressions: `81/81 PASS`;
- complete suite: `649 PASS / 2 SKIPPED`;
- Python static compilation and `git diff --check`: PASS;
- Windows installed-path run `33950404788`: SUCCESS;
- artifact `9964659445`, ZIP SHA-256 `650ea6c05d1551928842dbf8b53efcb5e6a3a92fca57cafc0e4a45774ae0e6d2`;
- Clean Windows, final installer and physical EliteBook: NOT_RUN.

Phase03 root causes corrected:

1. `AutomaticPipelineRunner` had no bridge to the already-developed Paper Quality modules.
2. The immutable-input constructor wrapper accepted only the old positional signature and dropped the new runtime keyword argument; it now forwards keyword arguments and has a regression test.
3. Block execution overwrote explicit `ABSTAIN`/`ROLLBACK` decisions with `PASS`; validated handler status is now preserved.
4. Malformed bbox/support-mask metadata could escape as dimensionality/type errors; installed validation now fails closed.
5. The real LR-ASPP output truthfully produced a MIXED parent route; a dominant-class JPEG subroute now bounds FBCNN loading without relabelling the parent evidence.
6. PyTorch 2.1.2 was incompatible with NumPy 2.x in the Windows environment; the optional CPU runtime now pins NumPy 1.x.
7. Official FBCNN imports `torchvision.models`; the exact compatible `torchvision==0.16.2+cpu` pair and a pack preflight were added.
8. Block 8 incorrectly used immutable MAIN as its transactional input, causing the identity guardrail to roll back accepted earlier context. Runtime input and immutable audit authority are now separate, and incoming provenance is retained.

---

## 2. Holdout lineage

FINAL_HOLDOUT_V3: **CONSUMED — NEVER RERUN**  
FINAL_HOLDOUT_V4: **CONSUMED_FAIL — 0/40 — NEVER RERUN**  
FINAL_HOLDOUT_V5: **NOT_CREATED**

No V3/V4 material was used by Phase02.

---

## 3. FBCNN pinned implementation

Official repository: `jiaxi-jiang/FBCNN`  
Pinned source revision: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`  
Checkpoint: `fbcnn_color.pth`  
Checkpoint bytes: `287755111`  
Checkpoint SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`  
Architecture reimplemented by CFS: **FALSE**  
Qualification device: **Windows CPU**  
CFS post-model authority: fixed conservative restoration fraction `0.25` (75% damaged input + 25% official FBCNN correction). Official source/checkpoint unchanged.

MODEL_LICENSE_STATUS: **CODE_APACHE_2_0; OFFICIAL_WEIGHT_ASSET_WITH_PROJECT_WIDE_APACHE_2_BASIS; FINAL_DISTRIBUTION_MANIFEST_STILL_REQUIRED**

---

## 4. Frozen Phase02 contract

Matrix: `8 identities x 6 profiles = 48 cases`.

Profiles: `jpeg-qf10-block-heavy`, `jpeg-qf20`, `jpeg-qf40`, `double-jpeg-qf40-qf15`, `social-resize-jpeg-qf20`, `mosquito-edges-qf12`.

Frozen evidence: PSNR, SSIM, LPIPS AlexNet, real SFace identity, RAM, CPU, source/checkpoint identity, wrong-person pixels and provenance.

Resource contract: <=80% system RAM, <=80% logical CPU observation, one heavy model at a time.

Frozen thresholds were not relaxed after observing any Phase02 run.

---

## 5. Phase02 final exact-head evidence

### Run `33800982565` — COMPLETE / SUCCESS

Candidate: `666cdbcfbdeee8f20901ccd063a4427d739bd107`  
Workflow run number: `10`  
Artifact: `fbcnn-phase02-windows-10`  
Artifact ID: `9916130291`  
Artifact ZIP SHA-256: `79b2b0269f982e4ca16d0eb37264f9d5300c767d090127e1500c9feda6926085`  
Artifact size: `140896131` bytes.

Execution integrity:
- exact candidate checkout PASS;
- Windows CPU runtime PASS;
- upstream/routing/resource regressions `64/64 PASS`;
- exact official FBCNN source revision PASS;
- exact checkpoint size/hash PASS;
- `48/48` cases completed;
- `0` runtime errors;
- `8` identities;
- case decisions **48 PASS / 0 ROLLBACK**;
- all frozen identity guardrails PASS;
- wrong-person final pixels `0`;
- provenance violations `0`;
- validation gate `TRUE`.

Identity evidence:
- no `identity_not_materially_worse` failures remain;
- worst observed identity delta is `-0.0005616`, inside the frozen `-0.01` guardrail;
- minimum post-restoration SFace across profiles remains above frozen threshold `0.363`.

Resource evidence:
- peak process CPU fraction `0.75 <= 0.80` — PASS;
- max system RAM fraction `0.3065598781 <= 0.80` — PASS;
- max process RSS `2456.40625 MB`.

Profile evidence:

| Profile | PSNR before -> after | SSIM before -> after | LPIPS before -> after | min SFace | PASS |
|---|---:|---:|---:|---:|---|
| Double JPEG 40->15 | 28.4559 -> 29.3054 | 0.83316 -> 0.85092 | 0.21940 -> 0.18870 | 0.71078 | TRUE |
| QF10 block-heavy | 27.1581 -> 28.0636 | 0.80043 -> 0.82149 | 0.27697 -> 0.23748 | 0.55205 | TRUE |
| QF20 | 29.8388 -> 30.7589 | 0.85989 -> 0.87709 | 0.16336 -> 0.13874 | 0.81921 | TRUE |
| QF40 | 32.5850 -> 33.4381 | 0.90813 -> 0.91918 | 0.09723 -> 0.08602 | 0.84854 | TRUE |
| Mosquito QF12 | 27.9413 -> 28.8520 | 0.82222 -> 0.84218 | 0.24483 -> 0.20788 | 0.57221 | TRUE |
| Social resize + QF20 | 26.9912 -> 27.2696 | 0.79982 -> 0.80695 | 0.29180 -> 0.28397 | 0.47335 | TRUE |

Both previously known LPIPS blockers (`jpeg-qf40`, `social-resize-jpeg-qf20`) are resolved without threshold changes. Every frozen profile improves PSNR, SSIM and LPIPS at profile aggregate level.

PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**.

Important scope boundary: this is Windows validation evidence for the FBCNN JPEG specialist. It does **not** by itself make the whole Paper Quality application production-ready, installer-ready, Target95-ready or physical-EliteBook-ready. The workflow report correctly keeps `production_qualified=false` because installed-offline same-candidate and physical target-hardware evidence belong to later phases.

---

## 6. Phase02 root-cause history

1. Initial harness generated 48 runtime errors before useful metrics.
2. Windows subprocess stdout/stderr pipe deadlock caused timeout; fixed at `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`.
3. FBCNN was executed after affine resampling, altering the JPEG artifact field; fixed at `50e5281c730535b416b6188c5d6bffc248652571` so order became `JPEG degradation -> FBCNN -> metric alignment`.
4. 100 ms psutil peak sampling overstated transient CPU usage; fixed at `60547ca87919254f59bf2ae21a0c0d89f57ac51e` with stabilized 1.0 s observation, frozen 80% limit unchanged.
5. Full-strength FBCNN authority caused identity drift in 6/48 cases; constrained with fixed 25% restoration authority at `d9adda184934394babd2a08b726598022c8de1aa` and contract tests at `666cdbcfbdeee8f20901ccd063a4427d739bd107`.
6. Exact-head run `33800982565` then passed all frozen Phase02 gates.

---

## 7. Current quality scoreboard

FBCNN Phase02 Windows multi-identity validation: **PASS / CLOSED** at exact candidate `666cdbcf...`.

DamageMask small U-Net: **REJECTED**, macro-F1 `0.173198`, macro-IoU `0.113028`.

LR-ASPP external DEVELOPMENT: F1 `0.716639`, IoU `0.579849`; **NOT_PRODUCTION_QUALIFIED**.

TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**  
COVERAGE: **NOT_FROZEN FOR FINAL TARGET95**  
TOTAL_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**

---

## 8. Safety invariants

- wrong-person final pixels must remain `0`;
- provenance violations must remain `0`;
- no generated pixel represented as original;
- no hidden fallback/model-name mismatch;
- no V3/V4 reuse;
- no hidden abstention used to inflate success;
- generated pixels remain `GENERATED_MODEL_INFERRED` unless evidence class explicitly says otherwise.

---

## 9. Phase sequence

PHASE_02 FBCNN JPEG: **CLOSED / PASS**.  
PHASE_03 PaperQualityRuntime wiring: **ACTIVE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS; PRODUCTION PIXEL AUTHORITY AND INSTALLER PENDING**. \
PHASE_04 DamageMask: DEFERRED.  
PHASE_05 geometry/component bank: DEFERRED.  
PHASE_06 MAIN+0–9: DEFERRED.  
PHASE_07 model competition: DEFERRED.  
PHASE_08 target hardware: DEFERRED.  
PHASE_09 UI: DEFERRED.  
PHASE_10 training: DEFERRED.  
PHASE_11 Target95: DEFERRED.  
PHASE_12 installer: DEFERRED.  
PHASE_13 release tests: DEFERRED.  
PHASE_14 independent V5: DEFERRED.

---

## 10. Release state

TARGET_HARDWARE_TEST: **NOT_RUN**  
INSTALLER_STATUS: **PARTIAL / NOT_FINAL_PAPER_QUALITY_INSTALLER**  
RELEASE_READY: **FALSE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

A GitHub Windows runner is not physical EliteBook evidence.

---

## 11. Phase03 cycle report — `2026-09-04T16:05Z`

- CURRENT_BRANCH: `integration/final-paper-quality-local`
- CURRENT_HEAD: `96b6aa4e858a7747bbf2ce52f51d81a9c10200e5`
- LAST_PUSH: fast-forward `666cdbcfbdeee8f20901ccd063a4427d739bd107 -> 96b6aa4e858a7747bbf2ce52f51d81a9c10200e5`
- ACTIVE_PHASE: `PHASE_03_PAPER_QUALITY_RUNTIME_WIRING`
- FILES_CHANGED: `app/automatic.py`, `app/execution.py`, `app/immutable_input_autoinstall.py`, `app/installed_paper_quality_runtime.py`, `app/paper_quality_runtime.py`, `app/settings.py`, `app/worker.py`, `config/default-settings.json`, `config/paper-quality-readiness.json`, `config/upstream-implementations.json`, `tests/test_fbcnn_development_evidence.py`, `tests/test_installed_paper_quality_path.py`, `tests/test_settings.py`, `tests/test_upstream_implementation_registry.py`
- WORKFLOW_RUNS: no new run observed for `96b6aa4e...` at verification time; Phase02 run `33800982565` remains the latest applicable remote model evidence
- TARGETED_TESTS: `118/118 PASS`
- FULL_SUITE_TESTS: `626 PASS / 2 SKIPPED`
- INSTALLED_PATH_TEST: `LOCAL_DEV_E2E_PASS` from real `PipelineWorker.run`; synthetic damage session, no external model claim
- ACTIVE_MODELS: `FBCNN=CANDIDATE_PHASE02_PASS`; `LR-ASPP=DEVELOPMENT_NOT_PRODUCTION`; `small U-Net=REJECTED`
- MODELS_ACTUALLY_EXECUTED: Phase03 installed-path E2E `NONE_EXTERNAL`; prior Phase02 Windows evidence executed official FBCNN
- MODEL_LICENSE_STATUS: `FBCNN_CODE_APACHE_2_0; WEIGHT_DISTRIBUTION_MANIFEST_PENDING; PRODUCT_LICENSE_PENDING`
- PAPER_QUALITY_RUNTIME_WIRED: `TRUE / LOCAL_DEV_E2E`
- FBCNN_INSTALLED_PATH_STATUS: `NOT_WIRED / NOT_EXECUTED`
- DAMAGE_MASK_STATUS: `small U-Net REJECTED`; `LR-ASPP DEVELOPMENT_NOT_PRODUCTION`; installed model-pack inference `NOT_VERIFIED`
- REFERENCE_COUNTS_TESTED: Phase03 dynamic path `MAIN+1`; full installed `MAIN+0..9` matrix `NOT_RUN`
- IDENTITY_RESULTS: post-Block-8 identity guardrail invoked and accepted verified observed-reference pixels in synthetic DEV E2E; product metric `NOT_MEASURED`
- WRONG_PERSON_FINAL_PIXELS: `0` in Phase03 synthetic E2E; product-wide final benchmark `NOT_MEASURED`
- PROVENANCE_VIOLATIONS: `0` in Phase03 synthetic E2E; product-wide final benchmark `NOT_MEASURED`
- HEALTHY_PIXELS_CHANGED: `0` outside damage authority in Phase03 synthetic E2E
- RAM_PEAK: Phase03 E2E `NOT_RECORDED AS PRODUCT EVIDENCE`; prior FBCNN Phase02 `2456.40625 MB`
- CPU_PEAK: Phase03 E2E `NOT_RECORDED AS PRODUCT EVIDENCE`; prior FBCNN Phase02 logical fraction `0.75`
- TARGET_HARDWARE_TEST: `NOT_RUN`
- TARGET95_ELIGIBLE_SUCCESS: `NOT_MEASURED`
- COVERAGE: `NOT_FROZEN FOR FINAL TARGET95`
- TOTAL_SUCCESS: `NOT_MEASURED`
- INSTALLER_STATUS: `PARTIAL / NO PAPER QUALITY MODEL PACK`
- RELEASE_READY: `FALSE`
- PROJECT_FINISHED: `FALSE`
- ROOT_CAUSE: developed Paper Quality modules were not reachable from the installed runner; the constructor wrapper also rejected the new injected-runtime keyword and execution status erased abstention/rollback semantics
- EXACT_NEXT_ACTION: implement the installed FBCNN JPEG-only backend at Block 8 with exact checkpoint hash verification, fixed `0.25` authority, truthful telemetry, one-model lifecycle/unload, and negative route tests for blur, sticker and healthy input.

---

## 12. Phase03 real-model installed-path cycle — `2026-09-05T06:44Z`

- CURRENT_BRANCH: `integration/final-paper-quality-local`
- CURRENT_HEAD: `c806f00e17aa074d906b6473fba7d5518e1afcd2`
- LAST_PUSH: fast-forward `79d7a900b618031050760718d3b6531ec1cc3124 -> c806f00e17aa074d906b6473fba7d5518e1afcd2`; complete Phase03 real-model sequence is a fast-forward from `96b6aa4e858a7747bbf2ce52f51d81a9c10200e5`
- ACTIVE_PHASE: `PHASE_03_PAPER_QUALITY_RUNTIME_WIRING`
- FILES_CHANGED: `.github/workflows/fbcnn-phase02-windows.yml`, `.github/workflows/paper-quality-installed-fbcnn-windows.yml`, `app/automatic.py`, `app/face_restorer_adapter.py`, `app/fbcnn_upstream_backend.py`, `app/installed_paper_quality_runtime.py`, `app/paper_quality_model_pack.py`, `app/worker.py`, `config/paper-quality-validation-pack.json`, `requirements-paper-quality-cpu.txt`, `scripts/run_installed_fbcnn_path_validation.py`, `tests/test_face_restorer_adapter.py`, `tests/test_fbcnn_upstream_backend.py`, `tests/test_installed_fbcnn_validation_driver.py`, `tests/test_installed_paper_quality_path.py`, `tests/test_paper_quality_model_pack.py`, `tests/test_progress_timeline.py`
- WORKFLOW_RUNS: `33896157286=FAIL` at first real route classification; `33949488127=FAIL` at NumPy/PyTorch ABI; `33949705168=FAIL` at missing torchvision; `33949966047=FAIL` at Block-8 transaction validation; `33950404788=SUCCESS` on exact HEAD; final artifact `9964659445`, ZIP SHA-256 `650ea6c05d1551928842dbf8b53efcb5e6a3a92fca57cafc0e4a45774ae0e6d2`
- TARGETED_TESTS: `81/81 PASS` on the final tree using the exact Windows workflow subset
- FULL_SUITE_TESTS: `649 PASS / 2 SKIPPED` on the final tree; Python compilation and `git diff --check` PASS
- INSTALLED_PATH_TEST: `WINDOWS_DEVELOPMENT_PASS`; real `PipelineWorker.run -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`, public-domain NASA portrait, QF10 JPEG, MAIN+0, no holdout
- ACTIVE_MODELS: `FBCNN=VALIDATION_CANDIDATE_PHASE02_PASS`; `LR-ASPP=DEVELOPMENT_NOT_PRODUCTION`; `small U-Net=REJECTED`; normal six-model application pack bootstrapped for the worker
- MODELS_ACTUALLY_EXECUTED: real `LR-ASPP ONNX` damage inference and official `FBCNN PyTorch CPU` JPEG candidate; FBCNN candidate was not fused
- MODEL_LICENSE_STATUS: `FBCNN_CODE_APACHE_2_0`; official release `v1.0` contains the exact weight asset under a repository-wide Apache-2.0 license with no separate weight terms found, but final checkpoint redistribution approval/manifest remains pending; `LR-ASPP_WEIGHT_TERMS_NOT_EXPLICIT_RESEARCH_ONLY`; product license pending
- PAPER_QUALITY_RUNTIME_WIRED: `TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS`
- FBCNN_INSTALLED_PATH_STATUS: `WINDOWS_DEVELOPMENT_PASS / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED`
- DAMAGE_MASK_STATUS: real LR-ASPP executed; dominant evidence `JPEG_ARTIFACT` with parent route `MIXED` and a bounded JPEG validation subroute; global/domain production qualification still blocked
- REFERENCE_COUNTS_TESTED: real Windows installed path `MAIN+0`; synthetic installed path `MAIN+1`; full installed `MAIN+0..9` matrix `NOT_RUN`
- IDENTITY_RESULTS: Block 8 accepted no generated final pixels, so its output equals its accepted input byte-for-byte; Phase02 multi-identity SFace gate remains PASS; product-wide identity benchmark `NOT_MEASURED`
- WRONG_PERSON_FINAL_PIXELS: `0` in the real installed-path run; product-wide final benchmark `NOT_MEASURED`
- PROVENANCE_VIOLATIONS: `0` in the real installed-path run; product-wide final benchmark `NOT_MEASURED`
- HEALTHY_PIXELS_CHANGED: `0` by Block 8; `115385` pixels already differed from immutable MAIN on entry because of preceding accepted pipeline blocks and are reported separately; raw shadow candidate changed `3605` crop pixels, `3552` outside its admitted JPEG subroute
- RAM_PEAK: process RSS `1398394880` bytes after FBCNN inference; system RAM fraction `0.2487176375`; budget `<=0.80` PASS on GitHub Windows runner
- CPU_PEAK: `NOT_RECORDED` in the Phase03 artifact; runtime budget restricted execution to `3/4` logical processors; Phase02 peak fraction remains `0.75`
- TARGET_HARDWARE_TEST: `NOT_RUN`; GitHub Windows runner is not the physical EliteBook
- TARGET95_ELIGIBLE_SUCCESS: `NOT_MEASURED`
- COVERAGE: `NOT_FROZEN FOR FINAL TARGET95`
- TOTAL_SUCCESS: `NOT_MEASURED`
- INSTALLER_STATUS: `PARTIAL / PAPER QUALITY VALIDATION PACK NOT DISTRIBUTED / FINAL OFFLINE INSTALLER NOT_RUN`
- RELEASE_READY: `FALSE`
- PROJECT_FINISHED: `FALSE`
- ROOT_CAUSE: four independently reproduced defects blocked the first real installed execution: truthful MIXED routing lacked a class-bounded model subroute; NumPy 2.x was ABI-incompatible with PyTorch 2.1.2; the FBCNN upstream import required torchvision; and Block 8 conflated immutable MAIN with its current transactional input. All four are corrected with regressions. Remaining immediate safety gap is that the raw FBCNN correction changes pixels outside the admitted JPEG subroute.
- EXACT_NEXT_ACTION: constrain the installed FBCNN validation candidate image and `generated_mask` to the JPEG subroute before selection, require zero candidate changes outside route in unit and Windows DEVELOPMENT evidence, and keep the candidate shadow-only/non-production until license, installer, Clean Windows and physical EliteBook gates pass.
