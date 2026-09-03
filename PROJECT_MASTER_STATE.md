# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** GitHub evidence overrides chat memory. Historical state remains preserved in Git history and in `project-state-history/`.

## 0. Current canonical state

Last ledger update: `2026-09-03T18:20Z`  
Technical state verified at: `2026-09-03T18:20Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_02_JPEG_FBCNN`  
PHASE_GATE: `IN_PROGRESS / NOT_VERIFIED`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `50e5281c730535b416b6188c5d6bffc248652571`  
Last technical HEAD: `60547ca87919254f59bf2ae21a0c0d89f57ac51e`  
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

EXACT_NEXT_ACTION: inspect exact-head workflow run `33789818369` for candidate `60547ca87919254f59bf2ae21a0c0d89f57ac51e`. If complete, classify its resource evidence first; if the CPU measurement issue is resolved, continue with the first remaining frozen quality failure from run 7 without changing thresholds or V3/V4. Do not enter PHASE_03 until Phase02 is objectively closed.

---

## 1. Installed-path truth

Installed entry path remains:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

`AutomaticPipelineRunner` does **not** yet invoke `PaperQualityRuntime`.

PAPER_QUALITY_RUNTIME_WIRED: **FALSE**

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

Frozen thresholds were not changed after runs 6 or 7.

---

## 5. Workflow history

### Run `33540310269`

**HARNESS_FAIL**: 48/48 rows errored before useful quality metrics. Not a model-quality result.

### Run `33543534673`

**CANCELLED / timeout**. Root cause later identified as subprocess stdout/stderr pipe deadlock.

### Commit `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`

`fix(jpeg): prevent Windows validation pipe deadlock`

### Run `33770678754` — COMPLETE / FAIL

Candidate `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`. 48/48 cases completed, 0 runtime errors, 8 identities, wrong-person final pixels 0, provenance violations 0. The model was being executed after affine resampling, which distorted the JPEG artifact field before FBCNN.

### Commit `50e5281c730535b416b6188c5d6bffc248652571`

`fix(jpeg): restore before affine alignment`

New order: `JPEG degradation -> FBCNN on original damaged pixel grid -> affine alignment only for comparable metrics`.

### Run `33777869802` — COMPLETE / FAIL

Candidate: `50e5281c730535b416b6188c5d6bffc248652571`  
Artifact: `fbcnn-phase02-windows-7`, ID `9905629100`, ZIP SHA-256 `9f7a68655e3dd9f80c9611aeefbb9341b8bbbd6a8e76c0baa6ec61f639a3c7ea`.

Execution integrity:
- exact candidate checkout PASS;
- Windows CPU environment PASS;
- route/resource/upstream regressions `63/63 PASS`;
- pinned source PASS;
- exact checkpoint bytes/hash PASS;
- validation completed `48/48`;
- error count `0`;
- identity count `8`;
- wrong-person final pixels `0`;
- provenance violations `0`.

Case decisions: **42 PASS / 6 ROLLBACK**.

All six case rollbacks were caused by the frozen `identity_not_materially_worse` guard, not by threshold failure, provenance, PSNR or SSIM:
- `eileen_collins`: QF10, QF20, mosquito QF12;
- `mae_jemison`: mosquito QF12;
- `peggy_whitson`: QF10, mosquito QF12.

Aggregate profile evidence:

| Profile | PSNR before -> after | SSIM before -> after | LPIPS before -> after | min SFace | PASS |
|---|---:|---:|---:|---:|---|
| Double JPEG 40->15 | 28.4559 -> 30.5618 | 0.83316 -> 0.87784 | 0.21940 -> 0.20172 | 0.69212 | TRUE |
| QF10 block-heavy | 27.1581 -> 29.3649 | 0.80043 -> 0.85322 | 0.27697 -> 0.23752 | 0.46599 | FALSE |
| QF20 | 29.8388 -> 31.9728 | 0.85989 -> 0.90196 | 0.16336 -> 0.16062 | 0.78706 | FALSE |
| QF40 | 32.5850 -> 34.3020 | 0.90813 -> 0.93319 | 0.09723 -> 0.11464 | 0.88671 | FALSE |
| Mosquito QF12 | 27.9413 -> 30.1365 | 0.82222 -> 0.87226 | 0.24483 -> 0.21485 | 0.53546 | FALSE |
| Social resize + QF20 | 26.9912 -> 27.6699 | 0.79982 -> 0.81642 | 0.29180 -> 0.31347 | 0.46842 | FALSE |

Interpretation:
- restoring before alignment substantially improved FBCNN quality versus run 6;
- PSNR and SSIM improved for all six profiles;
- LPIPS improved for QF10, QF20, double-JPEG and mosquito profiles;
- LPIPS worsened for QF40 and social-resize profiles;
- five profiles remain not qualified because the frozen profile contract also requires every case disposition PASS; QF40 and social-resize additionally fail aggregate LPIPS improvement;
- Phase02 remains FAIL.

Resource evidence:
- max process RSS `2453.87109375 MB`;
- max system RAM fraction `0.3216690170`;
- observed peak process CPU fraction `0.924` > frozen `0.80`.

The Windows runner reports 4 logical CPUs. Each FBCNN child reports `effective_threads=3`, i.e. a 75% CPU affinity/thread budget. The run-7 first case had mean observed process CPU fraction about `0.689`, but the 100 ms psutil peak sampler reported `0.924` (369.6% of one core), exceeding the theoretical sustained 3/4-core cap. This is evidence of sub-second scheduler/timer quantization in the measurement harness rather than proof of sustained >80% CPU use.

### Commit `60547ca87919254f59bf2ae21a0c0d89f57ac51e`

`fix(jpeg): stabilize Windows CPU peak sampling`

Only the Phase02 CPU observation harness changed. The `psutil.Process.cpu_percent` observation interval is now 1.0 second instead of 0.10 second. The frozen 80% limit, model, checkpoint, identities, profiles, quality thresholds, SFace threshold, LPIPS implementation and holdout policy are unchanged. The report now records `sample_interval_seconds=1.0`.

New exact-head workflow: run `33789818369`, run number `8`, candidate `60547ca87919254f59bf2ae21a0c0d89f57ac51e`, queued/running at this ledger update.

---

## 6. Current first remaining quality blocker after resource-harness correction

Do not modify it until run 8 classifies the CPU correction.

Known run-7 quality failures are:
1. six case-level identity-retention rollbacks under the frozen `identity_not_materially_worse` guard;
2. QF40 aggregate LPIPS worsened `0.09723 -> 0.11464`;
3. social-resize aggregate LPIPS worsened `0.29180 -> 0.31347`.

No threshold relaxation is authorized.

---

## 7. Current quality scoreboard

FBCNN historical one-identity development matrix: `6/6 PASS`; insufficient for production qualification.

FBCNN Phase02 run 7: **FAIL** on frozen multi-identity quality/resource gate.

FBCNN Phase02 run 8: **IN_PROGRESS / NOT_VERIFIED**.

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

## 9. Deferred phases

PHASE_03 PaperQualityRuntime wiring; PHASE_04 DamageMask; PHASE_05 geometry/component bank; PHASE_06 MAIN+0–9; PHASE_07 model competition; PHASE_08 target hardware; PHASE_09 UI; PHASE_10 training; PHASE_11 Target95; PHASE_12 installer; PHASE_13 release tests; PHASE_14 independent V5.

Do not execute them before Phase02 closes.

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
