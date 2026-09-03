# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** GitHub evidence overrides chat memory. Historical state remains preserved in Git history and in `project-state-history/`.

## 0. Current canonical state

Last ledger update: `2026-09-03T16:19Z`  
Technical state verified at: `2026-09-03T16:19Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_02_JPEG_FBCNN`  
PHASE_GATE: `IN_PROGRESS / NOT_VERIFIED`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`  
Last technical HEAD: `50e5281c730535b416b6188c5d6bffc248652571`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
PR #2: OPEN + DRAFT + NOT_MERGED; preserve as non-certified historical candidate.  
Current exact blocker: exact-head FBCNN Phase02 workflow run `33777869802` is validating the correction that executes FBCNN on the JPEG-damaged source pixels before affine metric alignment.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No force-push, no history deletion, no automatic merge, no V3/V4 rerun, no fabricated evidence.

EXACT_NEXT_ACTION: inspect run `33777869802`; if complete, retrieve its exact artifact and classify all 48 cases and six frozen profiles. If it fails, fix only the first evidenced remaining Phase02 root cause. Do not enter PHASE_03 until FBCNN is objectively qualified or rejected for the JPEG route.

---

## 1. Installed-path truth

Installed entry path remains:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

`AutomaticPipelineRunner` does **not** yet invoke `PaperQualityRuntime`.

PAPER_QUALITY_RUNTIME_WIRED: **FALSE**

PHASE_03 remains deferred until PHASE_02 closes.

---

## 2. Holdout lineage

FINAL_HOLDOUT_V3: **CONSUMED — NEVER RERUN**  
FINAL_HOLDOUT_V4: **CONSUMED_FAIL — 0/40 — NEVER RERUN**  
FINAL_HOLDOUT_V5: **NOT_CREATED**

No V3/V4 material is used by Phase02.

---

## 3. FBCNN pinned implementation

Official repository: `jiaxi-jiang/FBCNN`  
Pinned source revision: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`  
Checkpoint: `fbcnn_color.pth`  
Checkpoint bytes: `287755111`  
Checkpoint SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`  
Architecture reimplemented by CFS: **FALSE**  
Qualification device: **Windows CPU**

MODEL_LICENSE_STATUS: **CODE_APACHE_2_0; OFFICIAL_WEIGHT_ASSET_WITH_PROJECT_WIDE_APACHE_2_BASIS; FINAL_DISTRIBUTION_MANIFEST_STILL_REQUIRED**

Official source behavior: `main_test_fbcnn_color.py` JPEG-compresses the source image and feeds that compressed image directly to `model(img_L)`; it does not affine-align/resample the JPEG image before FBCNN. The official network predicts QF internally and returns restored pixels plus predicted QF.

---

## 4. Frozen Phase02 contract

Matrix: `8 identities x 6 profiles = 48 cases` minimum.

Profiles:
1. `jpeg-qf10-block-heavy`;
2. `jpeg-qf20`;
3. `jpeg-qf40`;
4. `double-jpeg-qf40-qf15`;
5. `social-resize-jpeg-qf20`;
6. `mosquito-edges-qf12`.

Frozen evidence includes PSNR, SSIM, LPIPS AlexNet, real SFace identity, RAM, CPU, source/checkpoint identity, wrong-person pixels and provenance.

Resource contract: <=80% system RAM, <=80% logical CPU observation, one heavy model at a time.

No thresholds were changed after observing run 6.

---

## 5. Workflow history

### Run `33540310269`

**HARNESS_FAIL**: 48/48 rows errored before useful quality metrics. Not a model-quality result.

### Run `33543534673`

**CANCELLED / 120-minute timeout** after one completed case. Root cause later identified as subprocess stdout/stderr pipe deadlock.

### Commit `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`

`fix(jpeg): prevent Windows validation pipe deadlock`

Child stdout/stderr moved from unconsumed `PIPE`s to per-case files. No model, threshold, dataset, profile or holdout change.

### Run `33770678754` — COMPLETE / FAIL

Candidate: `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`  
Artifact: `fbcnn-phase02-windows-6`, ID `9899983739`, ZIP SHA-256 `63c2a1fe6b5a0a220c6bbee64402a845508a53baeb4a63931244bed237bb2fc6`.

Execution integrity:
- exact candidate checkout PASS;
- isolated Windows CPU environment PASS;
- route/resource/upstream regressions `63/63 PASS`;
- pinned official source PASS;
- checkpoint bytes/hash PASS;
- real validation completed `48/48` cases;
- runtime error count `0`;
- identities `8`;
- wrong-person final pixels `0`;
- provenance violations `0`.

Resource evidence:
- max process RSS `1139.72265625 MB`;
- max system RAM fraction `0.2611618668`;
- observed peak process CPU fraction `0.882` (secondary unresolved resource-gate failure; >0.80).

Case decisions: `16 PASS / 32 ROLLBACK`. Every recorded case-level rollback was due to the frozen `psnr_improved` guard; identity/provenance guards did not cause those 32 rollbacks.

Aggregate profile results from exact artifact:

| Profile | PSNR before -> after | SSIM before -> after | LPIPS before -> after | min SFace | PASS |
|---|---:|---:|---:|---:|---|
| QF10 block-heavy | 27.1581 -> 27.6082 | 0.80043 -> 0.80833 | 0.27697 -> 0.27299 | 0.52991 | FALSE |
| QF20 | 29.8388 -> 30.2117 | 0.85989 -> 0.86432 | 0.16336 -> 0.16880 | 0.80728 | FALSE |
| QF40 | 32.5850 -> 32.8651 | 0.90813 -> 0.91021 | 0.09723 -> 0.10307 | 0.83307 | FALSE |
| Double JPEG 40->15 | 28.4559 -> 28.8301 | 0.83316 -> 0.83856 | 0.21940 -> 0.22222 | 0.68445 | FALSE |
| Social resize + QF20 | 26.9912 -> 27.1124 | 0.79982 -> 0.80192 | 0.29180 -> 0.30134 | 0.37434 | FALSE |
| Mosquito QF12 | 27.9413 -> 28.3746 | 0.82222 -> 0.82918 | 0.24483 -> 0.24488 | 0.56698 | FALSE |

Interpretation: PSNR and SSIM improved on average for all six profiles; LPIPS improved only for QF10 and worsened for the other five. The frozen validation gate therefore correctly remained FALSE.

---

## 6. First evidenced model-path root cause from run 6

ROOT_CAUSE: **FBCNN_WAS_EXECUTED_AFTER_AFFINE_ALIGNMENT_RESAMPLING**

The Phase02 vertical slice used this order:

`JPEG degradation -> YuNet landmarks -> affine 512 alignment/resampling -> FBCNN`

This is technically mismatched to a JPEG artifact specialist. Affine interpolation alters JPEG block boundaries, ringing and local compression structure before the model sees them. The official FBCNN evaluation path instead feeds JPEG-compressed pixels directly into the model.

Supporting run-6 evidence: for nominal QF10 cases, FBCNN's predicted input QF averaged approximately `74.33` and reached approximately `96.35`, showing that several aligned/resampled inputs no longer resembled the severe JPEG degradation presented before alignment. Similar inflation occurred across the other low-QF profiles.

This is a pipeline-order defect, not a frozen-threshold problem.

---

## 7. Correction after run 6

Technical commit: `50e5281c730535b416b6188c5d6bffc248652571`  
Commit message: `fix(jpeg): restore before affine alignment`  
Changed file: `research/run_fbcnn_vertical_slice.py`.

New model order:

`JPEG degradation -> FBCNN on original damaged pixel grid -> affine alignment only for comparable metrics`

The same landmarks/metric convention remain for clean/degraded/restored comparison. FBCNN source revision, checkpoint, damage profiles, metric thresholds, SFace threshold, LPIPS implementation, reference set and holdout policy were not changed.

New workflow: run `33777869802`, run number `7`, candidate `50e5281c730535b416b6188c5d6bffc248652571`, **QUEUED/RUNNING** at this ledger update.

Secondary known issue intentionally not bundled into this correction: run 6 observed CPU peak `0.882` versus frozen limit `0.80`. It remains a separate gate to diagnose only after classifying the pre-alignment correction.

---

## 8. Current quality scoreboard

FBCNN historical one-identity development matrix: `6/6 PASS`; insufficient for production qualification.

FBCNN Phase02 run 6: **FAIL** on frozen multi-identity quality/resource gate.

FBCNN Phase02 run 7: **IN_PROGRESS / NOT_VERIFIED**.

DamageMask small U-Net: **REJECTED**, macro-F1 `0.173198`, macro-IoU `0.113028`.

LR-ASPP external DEVELOPMENT: F1 `0.716639`, IoU `0.579849`; **NOT_PRODUCTION_QUALIFIED**.

TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**  
COVERAGE: **NOT_FROZEN FOR FINAL TARGET95**  
TOTAL_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**

---

## 9. Safety invariants

- wrong-person final pixels must remain `0`;
- provenance violations must remain `0`;
- no generated pixel represented as original;
- no hidden fallback/model-name mismatch;
- no V3/V4 reuse;
- no hidden abstention used to inflate success;
- generated pixels remain `GENERATED_MODEL_INFERRED` unless evidence class explicitly says otherwise.

---

## 10. Deferred phases

PHASE_03 PaperQualityRuntime wiring; PHASE_04 DamageMask; PHASE_05 geometry/component bank; PHASE_06 MAIN+0–9; PHASE_07 model competition; PHASE_08 target hardware; PHASE_09 UI; PHASE_10 training; PHASE_11 Target95; PHASE_12 installer; PHASE_13 release tests; PHASE_14 independent V5.

Do not execute them before Phase02 closes.

---

## 11. Release state

TARGET_HARDWARE_TEST: **NOT_RUN**  
INSTALLER_STATUS: **PARTIAL / NOT_FINAL_PAPER_QUALITY_INSTALLER**  
RELEASE_READY: **FALSE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

A GitHub Windows runner is not physical EliteBook evidence.
