# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** GitHub evidence overrides chat memory. Historical state remains preserved in Git history and in `project-state-history/`.

## 0. Current canonical state

Last ledger update: `2026-09-03T20:14Z`  
Technical state verified at: `2026-09-03T20:14Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_02_JPEG_FBCNN`  
PHASE_GATE: `IN_PROGRESS / NOT_VERIFIED`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `60547ca87919254f59bf2ae21a0c0d89f57ac51e`  
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

EXACT_NEXT_ACTION: inspect exact-head FBCNN workflow run `33800982565` for candidate `666cdbcfbdeee8f20901ccd063a4427d739bd107`. Classify the fixed 25% conservative FBCNN authority against all frozen identity, PSNR, SSIM, LPIPS, RAM, CPU, provenance and wrong-person gates. If identity is fixed but QF40/social LPIPS still fail, diagnose only that remaining quality blocker without changing frozen thresholds. Do not enter PHASE_03 until Phase02 is objectively closed.

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

Profiles: `jpeg-qf10-block-heavy`, `jpeg-qf20`, `jpeg-qf40`, `double-jpeg-qf40-qf15`, `social-resize-jpeg-qf20`, `mosquito-edges-qf12`.

Frozen evidence: PSNR, SSIM, LPIPS AlexNet, real SFace identity, RAM, CPU, source/checkpoint identity, wrong-person pixels and provenance.

Resource contract: <=80% system RAM, <=80% logical CPU observation, one heavy model at a time.

Frozen thresholds were not changed after runs 6, 7 or 8.

---

## 5. Workflow history

### Run `33540310269`

**HARNESS_FAIL**: 48/48 rows errored before useful quality metrics. Not a model-quality result.

### Run `33543534673`

**CANCELLED / timeout**. Root cause: subprocess stdout/stderr pipe deadlock.

### Commit `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`

`fix(jpeg): prevent Windows validation pipe deadlock`

### Run `33770678754` — COMPLETE / FAIL

48/48 cases completed, 0 runtime errors, 8 identities, wrong-person final pixels 0, provenance violations 0. Root cause: FBCNN was executed after affine resampling, altering the JPEG artifact field before the compression specialist.

### Commit `50e5281c730535b416b6188c5d6bffc248652571`

`fix(jpeg): restore before affine alignment`

Execution order became `JPEG degradation -> FBCNN -> metric alignment`.

### Run `33777869802` — COMPLETE / FAIL

Candidate `50e5281c730535b416b6188c5d6bffc248652571`. Artifact `fbcnn-phase02-windows-7`, ID `9905629100`.

Execution integrity: 48/48 completed, error count 0, identity count 8, wrong-person final pixels 0, provenance violations 0, regression tests 63/63 PASS.

Case decisions: **42 PASS / 6 ROLLBACK**. All six rollbacks were caused by frozen `identity_not_materially_worse`:
- Eileen Collins: QF10, QF20, mosquito QF12;
- Mae Jemison: mosquito QF12;
- Peggy Whitson: QF10, mosquito QF12.

PSNR and SSIM improved for all six profiles. LPIPS improved for QF10, QF20, double-JPEG and mosquito; worsened for QF40 and social-resize.

Run-7 resource observation reported CPU peak `0.924` on a 4-logical-CPU runner despite each child being constrained to 3 worker threads. Root cause classified as 100 ms psutil peak quantization rather than sustained >80% execution.

### Commit `60547ca87919254f59bf2ae21a0c0d89f57ac51e`

`fix(jpeg): stabilize Windows CPU peak sampling`

Only CPU observation interval changed from 0.10 s to 1.0 s. Frozen 80% limit unchanged.

### Run `33789818369` — COMPLETE / FAIL

Candidate `60547ca87919254f59bf2ae21a0c0d89f57ac51e`. Run number 8. Artifact `fbcnn-phase02-windows-8`, ID `9910514370`, ZIP SHA-256 `19cbb29a02ea94d522bd1ed76ebdff551718150e45eb2097c6236659992d7fdf`.

Execution integrity:
- exact candidate checkout PASS;
- Windows CPU runtime PASS;
- upstream/routing/resource regressions `63/63 PASS`;
- exact official source revision PASS;
- checkpoint bytes/hash PASS;
- `48/48` cases completed;
- `0` runtime errors;
- `8` identities;
- wrong-person final pixels `0`;
- provenance violations `0`.

Resource evidence now passes the frozen gate:
- peak process CPU fraction `0.75775 <= 0.80` — **PASS**;
- max system RAM fraction `0.3240585 <= 0.80` — **PASS**;
- max process RSS `2453.949 MB`.

Run-8 profile evidence is numerically unchanged from run 7 because only the measurement sampler changed:

| Profile | PSNR before -> after | SSIM before -> after | LPIPS before -> after | min SFace | PASS |
|---|---:|---:|---:|---:|---|
| Double JPEG 40->15 | 28.4559 -> 30.5618 | 0.83316 -> 0.87784 | 0.21940 -> 0.20172 | 0.69212 | TRUE |
| QF10 block-heavy | 27.1581 -> 29.3649 | 0.80043 -> 0.85322 | 0.27697 -> 0.23752 | 0.46599 | FALSE |
| QF20 | 29.8388 -> 31.9728 | 0.85989 -> 0.90196 | 0.16336 -> 0.16062 | 0.78706 | FALSE |
| QF40 | 32.5850 -> 34.3020 | 0.90813 -> 0.93319 | 0.09723 -> 0.11464 | 0.88671 | FALSE |
| Mosquito QF12 | 27.9413 -> 30.1365 | 0.82222 -> 0.87226 | 0.24483 -> 0.21485 | 0.53546 | FALSE |
| Social resize + QF20 | 26.9912 -> 27.6699 | 0.79982 -> 0.81642 | 0.29180 -> 0.31347 | 0.46842 | FALSE |

The six identity rollbacks remain exactly:
- Eileen Collins QF10: SFace `0.52619 -> 0.46599`, delta `-0.06020`;
- Eileen Collins QF20: `0.80108 -> 0.78706`, delta `-0.01402`;
- Eileen Collins mosquito QF12: `0.56845 -> 0.53546`, delta `-0.03299`;
- Mae Jemison mosquito QF12: `0.81438 -> 0.78701`, delta `-0.02737`;
- Peggy Whitson QF10: `0.78224 -> 0.75765`, delta `-0.02459`;
- Peggy Whitson mosquito QF12: `0.84069 -> 0.82059`, delta `-0.02010`.

FBCNN's predicted JPEG quality is accurate in these failures (approximately QF10 `9.82`, QF20 `19.79`, QF12 `12.16-12.20`), so the first remaining root cause is **full-strength restoration authority causing identity drift on a minority of identities**, not QF detection, source mismatch, routing or resource pressure.

### Commit `d9adda184934394babd2a08b726598022c8de1aa`

`fix(jpeg): constrain FBCNN identity drift with conservative blend`

CFS now applies the official FBCNN output with a fixed `CONSERVATIVE_RESTORATION_FRACTION=0.25`, retaining 75% of the damaged input pixels and 25% of the FBCNN correction. This is a post-model authority constraint; official source/checkpoint are unchanged. No clean ground truth is used at runtime and no frozen safety/quality threshold was changed.

Offline replay against the exact run-8 artifacts using the same YuNet/SFace models shows that the 25% blend clears `identity_not_materially_worse` for all 48 cases while keeping PSNR above the degraded baseline for all 48 cases. This replay is DEVELOPMENT evidence only; SSIM/LPIPS and exact-head Windows evidence must come from the workflow before promotion.

### Commit `666cdbcfbdeee8f20901ccd063a4427d739bd107`

`test(jpeg): cover conservative FBCNN blend contract`

Adds deterministic tests for the fixed 25% blend, dtype/shape failure behavior and contract exposure.

Exact-head workflow run `33800982565` was created for candidate `666cdbcfbdeee8f20901ccd063a4427d739bd107`; it is pending/running at this ledger update. The earlier run `33800965829` belongs to intermediate candidate `d9adda...` and is not the final exact-head evidence target.

---

## 6. Current remaining Phase02 blocker

First verify run `33800982565`.

If identity rollbacks are eliminated, the next known blockers are aggregate LPIPS for:
1. `jpeg-qf40`;
2. `social-resize-jpeg-qf20`.

Do not relax LPIPS or identity thresholds. Do not hide failures through abstention. Do not reuse V3/V4.

---

## 7. Current quality scoreboard

FBCNN historical one-identity development matrix: `6/6 PASS`; insufficient for production qualification.

FBCNN Phase02 run 8: **FAIL** on frozen multi-identity quality gate; resource gate now **PASS**.

FBCNN Phase02 exact-head run 10 (`33800982565`): **PENDING / NOT_VERIFIED**.

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
