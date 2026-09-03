# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** GitHub evidence overrides chat memory. Historical state remains preserved in Git history and in `project-state-history/`.

## 0. Current canonical state

Last ledger update: `2026-09-03T22:55Z`  
Technical state verified at: `2026-09-03T22:55Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_03_PAPER_QUALITY_RUNTIME_WIRING`  
PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**  
Last technical branch: `integration/final-paper-quality-local`  
Last technical HEAD: `666cdbcfbdeee8f20901ccd063a4427d739bd107`  
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

EXACT_NEXT_ACTION: on `integration/final-paper-quality-local`, wire `PaperQualityRuntime`, `DamageMaskRuntime`, reference-bank selection/fusion, identity rollback, provenance and telemetry into the real installed path `app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner` behind the existing feature flag, then add an end-to-end test that fails if those components exist but are not actually called. Do not promote PAPER_QUALITY_MODE_READY until the real installed path executes them with evidence.

---

## 1. Installed-path truth

Installed entry path remains:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

At this ledger update, `AutomaticPipelineRunner` does **not** yet invoke `PaperQualityRuntime`.

PAPER_QUALITY_RUNTIME_WIRED: **FALSE**

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
PHASE_03 PaperQualityRuntime wiring: **ACTIVE / NOT_VERIFIED**.  
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
