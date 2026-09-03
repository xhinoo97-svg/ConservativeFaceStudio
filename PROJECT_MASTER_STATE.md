# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory.
>
> Historical state is immutable and preserved separately:
> - through 2026-08-24: `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`;
> - exact ledger immediately before the 2026-09-03 Phase02 pipe-deadlock correction: `project-state-history/PROJECT_MASTER_STATE-through-2026-09-03-pre-pipe-fix.md`, blob `b9556884515daa7c65cef5876668e588c3af6209`.

## 0. Current canonical state

Last ledger update: `2026-09-03T15:06Z`  
Technical state verified at: `2026-09-03T15:06Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_02_JPEG_FBCNN`  
PHASE_GATE: `IN_PROGRESS / NOT_VERIFIED`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `3105218633a272a634c431b7ef26c84f9b34f226`  
Last technical HEAD: `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`  
Last technical tree: `3fdf8b8f8661c603bb00ee1f459abb6e99fef3a4`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
PR #2: OPEN + DRAFT + NOT_MERGED; preserve as non-certified historical candidate.  
Current exact blocker: FBCNN Phase02 Windows run `33770678754` for candidate `d0f14d7fa1303a35b1fe3b284f587c86986dafa4` is executing after correction of the validation subprocess deadlock.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No force-push, no history deletion, no automatic merge, no V3/V4 rerun, no fabricated evidence.

EXACT_NEXT_ACTION: inspect run `33770678754`; retrieve its exact artifact if complete; classify all 48 FBCNN cases and the six aggregate profiles. If the run fails, fix only the first evidenced Phase02 root cause. Do not enter PHASE_03 until the FBCNN gate is objectively classified.

---

## 1. Product and installed-path truth

CFS is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative evidence remains authoritative over generated inference. Paper Quality generated pixels are always `GENERATED_MODEL_INFERRED` and may never be represented as observed/original pixels.

Installed entry path remains:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

`AutomaticPipelineRunner` still runs the legacy 13-block path directly and does **not** yet invoke `PaperQualityRuntime`. Therefore:

PAPER_QUALITY_RUNTIME_WIRED: **FALSE**

This is intentionally deferred to PHASE_03 until PHASE_02 closes.

---

## 2. Branch and release map

| Branch | Purpose | Current known HEAD | Status |
|---|---|---:|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN/RELEASED |
| `integration/final-paper-quality-local` | only active final-integration branch | `d0f14d7fa1303a35b1fe3b284f587c86986dafa4` | ACTIVE / PHASE_02 |
| `hotfix/real-world-restoration-v1.1` | Track A historical candidate | `77687b3b171f4e9989fcf486834f2d8b7a52f591` | V4 CONSUMED_FAIL/FROZEN |
| `protocol/v5-certification-hardening` | protocol development | `268188c5a2540455ff804383cb583b16546b62f1` | ARCHIVED DEV |
| `research/paper-quality-local-v2` | research evidence | `6d57725aae087bb4a3144d521d91346999f9a4fd` | SUPERSEDED AS ACTIVE ARCHITECTURE |
| `meta/project-state` | canonical state only | self-SHA omitted | ACTIVE META |

GitHub branch API currently reports the integration branch and main as `protected=false`; this is a live repository configuration fact and does not relax the project rule that certified history must not be force-pushed or rewritten.

---

## 3. Holdout lineage

FINAL_HOLDOUT_V3: **CONSUMED — NEVER RERUN**  
FINAL_HOLDOUT_V4: **CONSUMED_FAIL — 0/40 — NEVER RERUN**  
FINAL_HOLDOUT_V5: **NOT_CREATED**

No V3/V4 material is used by Phase02. V5 must not be created or executed before all prerequisites are green and a candidate is frozen.

---

## 4. FBCNN pinned implementation and license

Official repository: `jiaxi-jiang/FBCNN`  
Pinned source revision: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`  
Checkpoint: `fbcnn_color.pth`  
Checkpoint bytes: `287755111`  
Checkpoint SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`  
Architecture reimplemented by CFS: **FALSE**  
Device under Phase02 qualification: **CPU**

MODEL_LICENSE_STATUS: **CODE_APACHE_2_0; WEIGHTS_PROJECT_WIDE_APACHE_2_BASIS_NO_SEPARATE_RESTRICTION_FOUND; NOT_LEGAL_ADVICE**

The exact pinned official repository contains Apache License 2.0. The checkpoint is the official v1.0 release asset. Distribution remains blocked unless the release manifest carries all required third-party notices and no later-discovered checkpoint-specific restriction conflicts with distribution.

---

## 5. Routing and production authority

FBCNN accepts only JPEG/compression/recompression context or explicit verified `jpeg_detected=True`.

Route-gated execution is implemented on the active integration branch:
- `f9a45c396cf45daf3eb9e348e98078b7396a29f1` — bind model execution to qualified damage route;
- `bc4d51da972e6fd410e92fb2584b3eaa554fc53f` — tests proving route-gated specialist execution;
- `3105218633a272a634c431b7ef26c84f9b34f226` — CI exercises route-gated FBCNN execution boundary.

This does **not** yet make FBCNN production-qualified. Production routing remains fail-closed until the Phase02 quality/resource evidence passes.

---

## 6. Frozen Phase02 validation contract

Required validation matrix: at least `8 identities x 6 profiles = 48 cases`.

Profiles:
1. `jpeg-qf10-block-heavy`;
2. `jpeg-qf20`;
3. `jpeg-qf40`;
4. `double-jpeg-qf40-qf15`;
5. `social-resize-jpeg-qf20`;
6. `mosquito-edges-qf12`.

Metrics/gates:
- PSNR before/after;
- SSIM before/after;
- LPIPS AlexNet before/after;
- real SFace identity;
- process RSS;
- system RAM fraction;
- process CPU fraction;
- exact source/checkpoint identity;
- wrong-person final pixels = 0;
- provenance violations = 0;
- aggregate profile PASS only when frozen metric/disposition rules pass.

Resource contract: <=80% system RAM, <=80% logical CPU observation, one heavy model at a time.

---

## 7. Phase02 workflow history and root causes

### Run `33540310269` — FAILED harness attempt

Candidate: `0342a2c3fa831d82073b484e8755a4aee778fcd6`.

Upstream/checkpoint bootstrap succeeded. Targeted tests succeeded. All 48 rows ended as errors before useful profile metrics. Classification: **HARNESS_FAILURE, NOT FBCNN_QUALITY_FAILURE**.

### Run `33543534673` — CANCELLED by 120-minute job timeout

Candidate: `3105218633a272a634c431b7ef26c84f9b34f226`.

Pre-validation evidence:
- Windows CPU environment installed successfully;
- upstream registry verified;
- `63 passed` targeted regression tests;
- exact FBCNN source verified;
- exact 287,755,111-byte checkpoint and SHA-256 verified.

The job entered real validation and was cancelled at the workflow 120-minute timeout. The uploaded artifact contained one completed case only: `eileen_collins/jpeg-qf10-block-heavy`.

Observed completed-case evidence:
- model load: `1.3157447 s`;
- measured 512 inference after warm-up: `12.1677914 s`;
- peak process RSS: `1100.10546875 MB`;
- peak observed system RAM fraction: `0.2383705636`;
- effective processors: `3/4`;
- SFace clean-vs-FBCNN: `0.5300669341` >= `0.363`;
- wrong-person final pixels: `0`;
- provenance violations: `0`;
- QF10 case disposition: `ROLLBACK` because PSNR changed `25.4942508 -> 25.4893449` (`-0.0049059 dB`) although SSIM improved slightly.

This single case cannot qualify or disqualify the six-profile aggregate by itself.

### Root cause identified 2026-09-03

`research/run_fbcnn_windows_validation.py::_run_case` launched each vertical-slice child with `stdout=PIPE` and `stderr=PIPE`, then polled `process.poll()` without consuming either pipe until after process exit. A child can fill a Windows anonymous pipe and block on write while the parent waits for termination, creating a producer/consumer deadlock.

ROOT_CAUSE: **WINDOWS_SUBPROCESS_PIPE_DEADLOCK_IN_VALIDATION_HARNESS**

### Correction

Technical commit: `d0f14d7fa1303a35b1fe3b284f587c86986dafa4` (`fix(jpeg): prevent Windows validation pipe deadlock`).

Correction:
- child stdout/stderr now write directly to per-case log files;
- parent continues CPU sampling while polling;
- logs remain available for failure diagnostics;
- FBCNN implementation, checkpoint, images, profile order, thresholds, metrics and routing rules were not changed.

New exact-head workflow: `33770678754`, run number `6`, **IN_PROGRESS** at this ledger update.

---

## 8. Current quality scoreboard

FBCNN historical DEVELOPMENT matrix: `6/6 PASS` on one development identity, run `32674085939`; not production qualification.

FBCNN Phase02 Windows multi-identity validation: **NOT_VERIFIED / RUNNING**.

DamageMask small U-Net: **REJECTED**, macro-F1 `0.173198`, macro-IoU `0.113028`.

LR-ASPP external DEVELOPMENT: F1 `0.716639`, IoU `0.579849`; subgroup gaps remain; **NOT_PRODUCTION_QUALIFIED**.

TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**  
COVERAGE: **NOT_FROZEN FOR FINAL TARGET95**  
TOTAL_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**

---

## 9. Safety and provenance invariants

Generated/derived region classes used by the project must remain explicit. Final release reporting must distinguish:
- ORIGINAL_PRESERVED;
- REFERENCE_DERIVED;
- GEOMETRICALLY_INFERRED;
- GENERATED_MODEL_INFERRED;
- ABSTAINED.

Current mandatory invariants:
- wrong-person final pixels = `0`;
- provenance violations = `0`;
- no healthy-pixel modification outside admitted policy;
- no silent fallback presented under another model name;
- no generated pixel represented as original;
- no hidden abstention used to inflate success.

---

## 10. Deferred phases — do not execute before Phase02 closes

PHASE_03: wire PaperQualityRuntime into the installed production path.  
PHASE_04: production damage-mask system.  
PHASE_05: geometry/component bank.  
PHASE_06: MAIN + 0–9 reference matrix.  
PHASE_07: specialist model competition.  
PHASE_08: target-computer optimization and physical-hardware evidence.  
PHASE_09: 13-block UI telemetry.  
PHASE_10: targeted training/adaptation.  
PHASE_11: frozen Target95 metrics.  
PHASE_12: single offline installer.  
PHASE_13: release acceptance.  
PHASE_14: one independent V5 holdout execution after candidate freeze.

---

## 11. Target hardware and release state

Target: HP EliteBook 1030 G3 class machine, Windows, 16 GB RAM; exact physical CPU/GPU still requires detection on the actual device.

TARGET_HARDWARE_TEST: **NOT_RUN**  
INSTALLER_STATUS: **PARTIAL / NOT_FINAL_PAPER_QUALITY_INSTALLER**  
RELEASE_READY: **FALSE**  
PROJECT_FINISHED: **FALSE**

A GitHub Windows runner is validation evidence, not physical EliteBook evidence.

---

## 12. Push journal since preserved historical ledger

### PUSH-20260901-PHASE02-ROUTING

Sequence ending at `3105218633a272a634c431b7ef26c84f9b34f226` added Windows resource handling, route-gated FBCNN model execution and tests. The exact-head run `33543534673` reached real model execution but deadlocked after the first completed case and later hit the workflow timeout. No threshold, dataset or holdout was changed.

### PUSH-20260903-001

Previous technical HEAD: `3105218633a272a634c431b7ef26c84f9b34f226`.  
New technical HEAD: `d0f14d7fa1303a35b1fe3b284f587c86986dafa4`.  
Tree: `3fdf8b8f8661c603bb00ee1f459abb6e99fef3a4`.  
Commit: `fix(jpeg): prevent Windows validation pipe deadlock`.  
Changed file: `research/run_fbcnn_windows_validation.py`.  
Cause: child-process stdout/stderr pipe deadlock under parent polling.  
Correction: file-backed per-case stdout/stderr logs.  
Automatic workflow: `33770678754`, run #6.  
Initial workflow status: IN_PROGRESS.  
No model promotion. No threshold change. No V3/V4 access. No V5 creation.

---

## 13. Session continuity rule

Every session must:
1. read this canonical ledger;
2. reconcile live branch/PR/workflow state;
3. continue the current ACTIVE_PHASE;
4. update this ledger after every significant technical push;
5. preserve exact historical evidence before any ledger compaction;
6. never infer PASS/QUALIFIED/RELEASE_READY/Target95/EliteBook/PROJECT_FINISHED without exact evidence.

If `PROJECT_FINISHED=FALSE` and there is no genuine external HARD_BLOCKER, execute `EXACT_NEXT_ACTION` rather than writing another plan.