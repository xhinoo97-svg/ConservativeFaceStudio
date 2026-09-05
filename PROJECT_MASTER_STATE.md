# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this before engineering decisions. GitHub evidence overrides chat memory. Detailed historical states remain preserved in Git history and `project-state-history/`; important failures and decisions remain summarized here.

## 0. Current canonical state

Last ledger update: `2026-09-05T09:27Z`  
Technical state verified at: `2026-09-05T09:27Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_04_DAMAGE_MASK`  
PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**  
PHASE_03_PAPER_QUALITY_RUNTIME_WIRING: **PASS / CLOSED FOR DEVELOPMENT WIRING**  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `b108def622ac1714d323020dab96709254c34b30`  
Last technical HEAD: `95aff6f3b142a94d459e4112c8eb94c0f20f0efb`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
PR #2: **OPEN / DRAFT / NOT_MERGED**; preserve as non-certified historical candidate.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Safety: no force-push, no certified-history rewrite, no auto-merge, no V3/V4 rerun, no fabricated model/metric/resource evidence.

EXACT_NEXT_ACTION: Phase04. Reconcile the existing DamageMask datasets, taxonomy, LR-ASPP artifact/training code and rejected small-U-Net evidence on `integration/final-paper-quality-local`; build the required multi-class evaluation matrix without using V3/V4; then compare the current LR-ASPP DEVELOPMENT model against the next lightweight segmentation architecture only with identity-disjoint TRAIN/VALIDATION data. Do not start a broad BFR competition before the damage-localization gate is measured.

---

## 1. Product and installed-path truth

CFS is one local Windows product with two evidence classes of operation: Conservative/Forensic reconstruction and a future Paper Quality mode. The installed desktop path is:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

With the Paper Quality feature flag enabled, the real installed path reaches:

`PipelineWorker -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`

Block 8 invokes `DamageMaskRuntime`, damage routing, model qualification, `PersonalizedReferenceBank`, component selection, reference-first repair, calibrated candidate selection when configured, component-aware fusion, per-pixel provenance and the existing identity rollback. The shipped default remains `paper_quality_enabled=false`.

The project policy is **official upstream implementation first**. When an executable paper repository exists, CFS pins and executes upstream code/checkpoints rather than recreating the architecture. CFS owns thin adapters, routing, identity/provenance firewalls, resource control, checkpoint verification, Windows/offline packaging, benchmarking and release gates. Upstream code is not assumed bug-free.

PAPER_QUALITY_RUNTIME_WIRED: **TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS**  
PAPER_QUALITY_MODE_READY: **FALSE**  
FBCNN_INSTALLED_PATH_STATUS: **WINDOWS_DEVELOPMENT_PASS / JPEG_SUBROUTE_BOUNDED / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED**

Phase03 being closed means the advanced runtime is demonstrably connected to the actual installed application path and same-HEAD regressions are green. It does **not** mean Paper Quality is production-qualified, installer-qualified or EliteBook-qualified.

---

## 2. Branch / holdout safety map

- `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`: certified V1, immutable.
- `integration/final-paper-quality-local@95aff6f3b142a94d459e4112c8eb94c0f20f0efb`: active technical branch.
- `meta/project-state`: canonical ledger branch.
- PR #2 `hotfix/real-world-restoration-v1.1`: OPEN/DRAFT/NOT_MERGED; not a certified release candidate.
- `FINAL_HOLDOUT_V3`: **CONSUMED — NEVER RERUN / NEVER TUNE**.
- `FINAL_HOLDOUT_V4`: **CONSUMED_FAIL — 0/40 — NEVER RERUN / NEVER TUNE**.
- `FINAL_HOLDOUT_V5`: **NOT_CREATED**.

No V3/V4 material may be used by current Paper Quality development.

---

## 3. Phase02 FBCNN frozen evidence

Official repository: `jiaxi-jiang/FBCNN`  
Pinned revision: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`  
Checkpoint: `fbcnn_color.pth`  
Checkpoint size: `287755111` bytes  
Checkpoint SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`  
Architecture reimplemented by CFS: **FALSE**  
Fixed CFS conservative correction fraction: `0.25`.

Phase02 exact-head Windows run: `33800982565` at `666cdbcfbdeee8f20901ccd063a4427d739bd107`.  
Matrix: `8 identities x 6 JPEG profiles = 48 cases`.  
Result: `48/48 PASS`, `0` errors, `0` rollbacks, wrong-person final pixels `0`, provenance violations `0`.  
Peak observed CPU fraction: `0.75`; max system RAM fraction `0.3065598781`; max process RSS `2456.40625 MB`.

| Profile | PSNR before -> after | SSIM before -> after | LPIPS before -> after | min SFace |
|---|---:|---:|---:|---:|
| Double JPEG 40->15 | 28.4559 -> 29.3054 | 0.83316 -> 0.85092 | 0.21940 -> 0.18870 | 0.71078 |
| QF10 block-heavy | 27.1581 -> 28.0636 | 0.80043 -> 0.82149 | 0.27697 -> 0.23748 | 0.55205 |
| QF20 | 29.8388 -> 30.7589 | 0.85989 -> 0.87709 | 0.16336 -> 0.13874 | 0.81921 |
| QF40 | 32.5850 -> 33.4381 | 0.90813 -> 0.91918 | 0.09723 -> 0.08602 | 0.84854 |
| Mosquito QF12 | 27.9413 -> 28.8520 | 0.82222 -> 0.84218 | 0.24483 -> 0.20788 | 0.57221 |
| Social resize + QF20 | 26.9912 -> 27.2696 | 0.79982 -> 0.80695 | 0.29180 -> 0.28397 | 0.47335 |

PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**. This is development/validation evidence, not installer/EliteBook qualification.

MODEL_LICENSE_STATUS: **FBCNN_CODE_APACHE_2_0; OFFICIAL_WEIGHT_ASSET_PRESENT_IN_PROJECT_RELEASE; FINAL_CHECKPOINT_REDISTRIBUTION_MANIFEST/PRODUCT_LICENSE STILL PENDING**.

---

## 4. Phase03 final same-HEAD evidence

### Run `33957871473` — COMPLETE / SUCCESS

Technical SHA: `95aff6f3b142a94d459e4112c8eb94c0f20f0efb`  
Workflow: `Paper Quality installed FBCNN Windows validation` run `#7`  
Artifact: `paper-quality-installed-fbcnn-7`  
Artifact ID: `9966988511`  
Artifact ZIP SHA-256: `fb3c07f18c9814947eab3a9499847f92387498b943d045b816815315cb77aae1`.

Same-HEAD gates:
- exact Windows checkout: PASS;
- Phase03 targeted tests: **81/81 PASS** in `5.00 s`;
- complete test suite: **651/651 PASS** in `16.02 s`;
- LR-ASPP DEVELOPMENT artifact exact SHA/size: PASS;
- official FBCNN source exact revision: PASS;
- FBCNN checkpoint exact SHA/size: PASS;
- validation pack offline inspection: PASS, `production_qualified=false`;
- normal six-model application pack bootstrap: PASS;
- licensed public DEVELOPMENT input: PASS;
- real installed Worker route: PASS;
- evidence contract: PASS.

Environment/evidence:
- Windows Server 2025 GitHub runner; **not physical EliteBook evidence**;
- Python `3.11.9`;
- `torch 2.1.2+cpu`, `torchvision 0.16.2+cpu`, `numpy 1.26.4`, `onnxruntime 1.29.0`, CUDA false;
- public-domain NASA portrait, QF10 JPEG, MAIN+0;
- real entry path: `PipelineWorker.run -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`.

Observed damage/router output:
- parent route `MIXED`;
- JPEG_ARTIFACT: `16768` pixels, fraction `0.6515387007`, mean confidence `0.7818839550`;
- SCRIBBLE: `8968` pixels, fraction `0.3484612993`, mean confidence `0.8190361261`;
- specialist route `JPEG_ARTIFACT` only.

FBCNN raw vs bounded authority:
- crop `55 x 66`;
- raw changed `3605` pixels;
- raw changed outside JPEG subroute `3552`;
- JPEG route mask `54`;
- bounded candidate changed `53`;
- bounded generated mask `53`;
- bounded changed outside route **`0`**;
- candidate `INSTALLED_PATH_VALIDATION_SHADOW`, `fused_to_final=false`;
- generated final pixels `0`;
- wrong-person final pixels `0`;
- provenance violations `0`;
- healthy pixels changed by Block 8 `0`;
- final Block-8 SHA equals accepted Block-8 input SHA.

Resource/lifecycle evidence:
- CPU affinity budget `3/4` logical processors, cap `0.8`;
- heavy-model concurrency `1`;
- process RSS after FBCNN inference `1399234560` bytes;
- system RAM fraction after inference `0.2531317382`;
- FBCNN model load `1.4613618 s`;
- inference `0.2104671 s` on this `55 x 66` crop;
- CPU utilization peak fraction itself: **NOT_RECORDED**.

Interpretation: official upstream FBCNN is a whole-crop restorer. CFS measures the raw output but grants it no candidate authority outside the classified JPEG subroute. Upstream architecture/checkpoint remain unchanged.

PHASE_03_PAPER_QUALITY_RUNTIME_WIRING: **PASS / CLOSED FOR DEVELOPMENT WIRING**.

---

## 5. DamageMask state — active Phase04

Known evidence before Phase04 expansion:
- small U-Net: **REJECTED**, macro-F1 `0.173198`, macro-IoU `0.113028`;
- LR-ASPP: **DEVELOPMENT / NOT_PRODUCTION_QUALIFIED**, F1 `0.716639`, IoU `0.579849` from prior research evidence;
- LR-ASPP exact DEVELOPMENT ONNX used by installed route SHA-256 `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`, size `12879910` bytes;
- Windows installed-path inference is real, but this is not domain qualification.

Required Phase04 evaluation classes:
`OPAQUE_STICKER`, `TRANSLUCENT_STICKER`, `EMOJI`, `TEXT`, thin/thick black/color `SCRIBBLE`, local/global blur, `MOTION_BLUR`, `DEFOCUS`, `BLOCK_MOSAIC`, `PIXELATION`, `JPEG_ARTIFACT`, `NOISE`, `MIXED_DAMAGE`, plus healthy pixels.

Required metrics: precision, recall, F1, IoU, false-positive rate on healthy pixels, false-negative rate inside sticker/scribble, per class, opacity, size, severity and facial position.

FaceMat may be investigated only if official code, checkpoint, license and Windows feasibility are verifiable. No paper-only claim becomes executable evidence.

DAMAGE_MASK_STATUS: **LR-ASPP DEVELOPMENT_NOT_PRODUCTION; SMALL_U_NET REJECTED; PHASE04 ACTIVE**.

---

## 6. Upstream model policy and serious model registry

Policy: `forbidden_when_official_executable_upstream_exists` to reimplement model architecture; integration mode `pinned_official_upstream_plus_thin_cfs_adapter`.

- FBCNN — JPEG specialist; real Windows CPU evidence; validation candidate.
- NAFNet — existing conservative deblur/denoise.
- GPEN BFR-512 — blind restoration challenger; official upstream pinned; licensing/Windows target validation pending.
- GFPGAN v1.4 — blind restoration challenger; official upstream pinned.
- CodeFormer — official upstream CPU slice exists; production license blocker.
- InstantRestore — personalized multi-reference challenger; CUDA/CPU feasibility and licensing/checkpoint gates pending.
- RefFaceInpainting — reference-guided occlusion specialist candidate; revision/checkpoint/CPU/Windows gates pending.
- RefineFIR — only promote if executable upstream/checkpoint becomes verifiable.
- OSDFace — severe one-step diffusion challenger; CPU/Windows/resource gates pending.
- RestoreFormer++, Restormer, RestorerID, RefSTAR, DMDNet, ReF-LDM and reproducible NTIRE methods — research/benchmark candidates until measured and licensed.

IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED.

---

## 7. Safety invariants

- frozen identity safety thresholds are not relaxed to make quality pass;
- wrong-person final observed pixels `0`;
- provenance violations `0`;
- healthy pixels not rewritten outside valid repair authority;
- generated pixels remain `GENERATED_MODEL_INFERRED`;
- one heavy model at a time;
- <=80% logical CPU allocation and <=80% process/system RAM;
- no hidden fallback/model-name substitution;
- no hidden abstention used to inflate success;
- no V3/V4 reuse.

---

## 8. Phase sequence

PHASE_02 FBCNN JPEG: **CLOSED / PASS**.  
PHASE_03 PaperQualityRuntime wiring: **CLOSED / PASS FOR DEVELOPMENT WIRING**.  
PHASE_04 DamageMask: **ACTIVE**.  
PHASE_05 geometry/component bank: DEFERRED.  
PHASE_06 MAIN+0–9: DEFERRED.  
PHASE_07 specialist-model competition: DEFERRED.  
PHASE_08 target hardware: DEFERRED.  
PHASE_09 UI/timeline completion: DEFERRED.  
PHASE_10 training/calibration: DEFERRED.  
PHASE_11 Target95: DEFERRED.  
PHASE_12 installer: DEFERRED.  
PHASE_13 release tests: DEFERRED.  
PHASE_14 independent V5: DEFERRED.

---

## 9. Release state

REFERENCE_COUNTS_TESTED: real Windows `MAIN+0`; synthetic installed E2E `MAIN+1`; installed `MAIN+0..9` matrix **NOT_RUN**.  
IDENTITY_RESULTS: Phase02 multi-identity SFace gate PASS; Phase03 validation-shadow grants no generated final authority; product-wide Paper Quality identity benchmark **NOT_MEASURED**.  
WRONG_PERSON_FINAL_PIXELS: `0` in current installed-path evidence.  
PROVENANCE_VIOLATIONS: `0` in current installed-path evidence.  
HEALTHY_PIXELS_CHANGED: `0` by Block 8 in current installed-path evidence.  
TARGET_HARDWARE_TEST: **NOT_RUN**.  
TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**.  
COVERAGE: **NOT_FROZEN FOR FINAL TARGET95**.  
TOTAL_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**.  
INSTALLER_STATUS: **PARTIAL / PAPER QUALITY VALIDATION PACK NOT DISTRIBUTED / FINAL OFFLINE INSTALLER NOT_RUN**.  
RELEASE_READY: **FALSE**.  
PROJECT_FINISHED: **FALSE**.

A GitHub Windows runner is not physical HP EliteBook acceptance.

---

## 10. Historical engineering record — append-preserved important failures

### Phase02 FBCNN
1. Initial harness produced 48 runtime errors before metrics.
2. Windows subprocess stdout/stderr pipe deadlock caused timeout.
3. FBCNN after affine resampling altered JPEG artifact field; corrected to `JPEG degradation -> FBCNN -> metric alignment`.
4. 100 ms CPU sampling overstated transient usage; stabilized without changing the 80% limit.
5. Full-strength FBCNN identity drift occurred in `6/48`; fixed by frozen `0.25` correction fraction without threshold relaxation.
6. Run `33800982565` passed 48/48.

### Phase03 installed runtime
1. Paper Quality modules were not reachable from the actual app runner.
2. Constructor wrapper dropped the injected runtime keyword.
3. Execution status overwrote ABSTAIN/ROLLBACK semantics.
4. Malformed geometry/support metadata needed fail-closed validation.
5. LR-ASPP produced truthful MIXED routing; a class-bounded specialist subroute was added.
6. NumPy 2.x conflicted with PyTorch 2.1.2; optional runtime pins NumPy 1.26.x.
7. Official FBCNN requires compatible torchvision; exact CPU pair was added.
8. Block 8 conflated immutable MAIN with current accepted transactional input.
9. Raw FBCNN changed pixels outside sparse JPEG route; `b108def6...` now bounds candidate authority to the route while retaining raw telemetry.
10. The old full suite belonged to a previous SHA; `95aff6f3...` made full pytest a mandatory same-HEAD Windows installed-path gate and passed `651/651`.

Historical failed workflows remain evidence: `33896157286`, `33949488127`, `33949705168`, `33949966047`. Successful installed runs: `33950404788`, `33950854805`, final same-HEAD gate `33957871473`.

---

## 11. Latest technical push record — `95aff6f3b142a94d459e4112c8eb94c0f20f0efb`

- CURRENT_BRANCH: `integration/final-paper-quality-local`
- CURRENT_HEAD: `95aff6f3b142a94d459e4112c8eb94c0f20f0efb`
- LAST_PUSH: `b108def622ac1714d323020dab96709254c34b30 -> 95aff6f3b142a94d459e4112c8eb94c0f20f0efb`
- ACTIVE_PHASE: `PHASE_04_DAMAGE_MASK` after closing Phase03 development wiring
- FILES_CHANGED: `.github/workflows/paper-quality-installed-fbcnn-windows.yml`
- WORKFLOW_RUNS: `33957871473=SUCCESS`; artifact `9966988511`; ZIP SHA-256 `fb3c07f18c9814947eab3a9499847f92387498b943d045b816815315cb77aae1`
- TARGETED_TESTS: `81/81 PASS`
- FULL_SUITE_TESTS: `651/651 PASS`
- INSTALLED_PATH_TEST: `WINDOWS_DEVELOPMENT_PASS`
- ACTIVE_MODELS: `FBCNN=VALIDATION_CANDIDATE`; `LR-ASPP=DEVELOPMENT_NOT_PRODUCTION`; `small U-Net=REJECTED`
- MODELS_ACTUALLY_EXECUTED: real `LR-ASPP ONNX`; official upstream `FBCNN PyTorch CPU`
- MODEL_LICENSE_STATUS: `FBCNN_CODE_APACHE_2_0; FINAL_WEIGHT_DISTRIBUTION_MANIFEST_PENDING; PRODUCT_LICENSE_PENDING`; LR-ASPP research-only pending explicit weight terms
- PAPER_QUALITY_RUNTIME_WIRED: `TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS`
- FBCNN_INSTALLED_PATH_STATUS: `PASS / JPEG_SUBROUTE_BOUNDED / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED`
- DAMAGE_MASK_STATUS: `LR-ASPP DEVELOPMENT_NOT_PRODUCTION; PHASE04 ACTIVE`
- REFERENCE_COUNTS_TESTED: `MAIN+0` real Windows; `MAIN+1` synthetic; `0..9 NOT_RUN`
- IDENTITY_RESULTS: no FBCNN pixels fused; Phase02 identity evidence remains PASS; product-wide metric NOT_MEASURED
- WRONG_PERSON_FINAL_PIXELS: `0`
- PROVENANCE_VIOLATIONS: `0`
- HEALTHY_PIXELS_CHANGED: `0` by Block 8
- RAM_PEAK: post-inference RSS `1399234560` bytes; system RAM fraction `0.2531317382`
- CPU_PEAK: `NOT_RECORDED`; execution restricted to `3/4` logical processors
- TARGET_HARDWARE_TEST: `NOT_RUN`
- TARGET95_ELIGIBLE_SUCCESS: `NOT_MEASURED`
- COVERAGE: `NOT_FROZEN`
- TOTAL_SUCCESS: `NOT_MEASURED`
- INSTALLER_STATUS: `PARTIAL / NOT_FINAL`
- RELEASE_READY: `FALSE`
- PROJECT_FINISHED: `FALSE`
- ROOT_CAUSE: Phase03 same-HEAD proof was incomplete because the complete suite had not been run after the FBCNN route-bounding commit; the Windows installed-path workflow now makes full pytest mandatory and all gates pass
- EXACT_NEXT_ACTION: inspect current DamageMask code/data/evidence, create the required per-damage/per-opacity/per-size/per-position evaluation matrix on legal identity-disjoint DEVELOPMENT/VALIDATION data, and benchmark LR-ASPP honestly before selecting the next lightweight segmentation challenger.
