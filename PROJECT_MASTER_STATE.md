# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this before engineering decisions. GitHub evidence overrides chat memory. Detailed historical states remain preserved in Git history and `project-state-history/`; important failures and decisions remain summarized here.

## 0. Current canonical state

Last ledger update: `2026-09-05T10:46Z`  
Technical state verified at: `2026-09-05T10:46Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_04_DAMAGE_MASK`  
PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**  
PHASE_03_PAPER_QUALITY_RUNTIME_WIRING: **PASS / CLOSED FOR DEVELOPMENT WIRING**  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `2692cd2919097e3dcd2f11eb892a2d31baa75e67`  
Last technical HEAD: `a04d11f67e98de9950747feff894e60460446174`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
PR #2: **OPEN / DRAFT / NOT_MERGED**; preserve as non-certified historical candidate.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Safety: no force-push, no certified-history rewrite, no auto-merge, no V3/V4 rerun, no fabricated model/metric/resource evidence, no threshold relaxation to manufacture a pass.

EXACT_NEXT_ACTION: Phase04. The current LR-ASPP is rejected by the frozen 8,288-case expanded DEVELOPMENT measurement. Do not tune the frozen gate and do not use V3/V4/final holdout. Build and measure the next lightweight multi-class DamageMask segmentation challenger on legal identity-disjoint DEVELOPMENT/TRAIN/VALIDATION data, using the same 1,036-case-per-identity factor matrix and Phase04 thresholds. FaceMat may be used only if official executable code, checkpoint, license and Windows feasibility are verifiable; otherwise use a verifiable official lightweight segmentation architecture and the existing legal development pipeline. Only a challenger that passes the frozen Phase04 gate may be wired as the production DamageMask candidate.

---

## 1. Product and installed-path truth

CFS is one local Windows product with Conservative/Forensic reconstruction and a future Paper Quality mode. Installed desktop path:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

With Paper Quality enabled:

`PipelineWorker -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`

Block 8 invokes `DamageMaskRuntime`, damage routing, model qualification, `PersonalizedReferenceBank`, component selection, reference-first repair, calibrated candidate selection when configured, component-aware fusion, per-pixel provenance and identity rollback. Shipped default remains `paper_quality_enabled=false`.

PAPER_QUALITY_RUNTIME_WIRED: **TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS**  
PAPER_QUALITY_MODE_READY: **FALSE**  
FBCNN_INSTALLED_PATH_STATUS: **WINDOWS_DEVELOPMENT_PASS / JPEG_SUBROUTE_BOUNDED / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED**

Phase03 closure proves wiring to the real installed application path. It does not prove Paper Quality production qualification, final installer qualification, Target95, or physical EliteBook acceptance.

---

## 2. Branch / holdout safety map

- `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`: certified V1, immutable.
- `integration/final-paper-quality-local@a04d11f67e98de9950747feff894e60460446174`: active technical branch.
- `meta/project-state`: canonical ledger branch.
- PR #2 `hotfix/real-world-restoration-v1.1`: OPEN/DRAFT/NOT_MERGED; not certified.
- `FINAL_HOLDOUT_V3`: **CONSUMED — NEVER RERUN / NEVER TUNE**.
- `FINAL_HOLDOUT_V4`: **CONSUMED_FAIL — 0/40 — NEVER RERUN / NEVER TUNE**.
- `FINAL_HOLDOUT_V5`: **NOT_CREATED**.

No V3/V4 material may be used by current Paper Quality development.

---

## 3. Phase02 FBCNN frozen evidence

Official repository: `jiaxi-jiang/FBCNN`  
Pinned revision: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`  
Checkpoint: `fbcnn_color.pth`  
Checkpoint SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`  
CFS conservative correction fraction: `0.25`.

Phase02 exact-head Windows run: `33800982565` at `666cdbcfbdeee8f20901ccd063a4427d739bd107`.  
Matrix: `8 identities x 6 JPEG profiles = 48 cases`.  
Result: `48/48 PASS`, `0` errors, `0` rollbacks, wrong-person final pixels `0`, provenance violations `0`.  
Peak observed CPU fraction `0.75`; max system RAM fraction `0.3065598781`; max process RSS `2456.40625 MB`.

PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**. Development/validation evidence only; not final installer/EliteBook qualification.

MODEL_LICENSE_STATUS: **FBCNN_CODE_APACHE_2_0; FINAL_CHECKPOINT_REDISTRIBUTION_MANIFEST/PRODUCT_LICENSE STILL PENDING**.

---

## 4. Phase03 final installed-path evidence

Final same-HEAD gate: run `33957871473`, SHA `95aff6f3b142a94d459e4112c8eb94c0f20f0efb`, Windows SUCCESS.

- targeted tests: `81/81 PASS`;
- complete suite: `651/651 PASS`;
- exact LR-ASPP DEVELOPMENT artifact: PASS;
- official FBCNN revision/checkpoint: PASS;
- real `PipelineWorker -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime` path: PASS;
- FBCNN candidate authority bounded strictly to JPEG subroute;
- generated final pixels `0` in validation shadow;
- wrong-person final pixels `0`;
- provenance violations `0`;
- healthy pixels changed by Block 8 `0`;
- one heavy model at a time and CPU affinity budget enforced.

PHASE_03_PAPER_QUALITY_RUNTIME_WIRING: **PASS / CLOSED FOR DEVELOPMENT WIRING**.

---

## 5. DamageMask state — active Phase04

### 5.1 Prior evidence

- small U-Net: **REJECTED**, macro-F1 `0.173198`, macro-IoU `0.113028`;
- LR-ASPP prior synthetic DEVELOPMENT evidence: macro-F1 `0.716639`, macro-IoU `0.579849`; this used the earlier narrower evaluation and is not production qualification;
- exact LR-ASPP DEVELOPMENT ONNX SHA-256: `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`;
- checkpoint redistribution license remains not explicit upstream/research-only pending product licensing decision.

### 5.2 Frozen expanded Phase04 matrix

The original 52-case generator did not actually measure the declared factor cross-product. It was replaced by a validated cross-factorial matrix:

- required classes: `OPAQUE_STICKER`, `TRANSLUCENT_STICKER`, `EMOJI`, `TEXT`, thin/thick black/color `SCRIBBLE`, `BLUR_LOCAL`, `BLUR_GLOBAL`, `MOTION_BLUR`, `DEFOCUS`, `BLOCK_MOSAIC`, `PIXELATION`, `JPEG_ARTIFACT`, `NOISE`, `MIXED_DAMAGE`, `HEALTHY`;
- local factors: 7 facial positions x 3 sizes x 3 severities;
- translucent sticker additionally: 3 opacity levels;
- global damage: GLOBAL position x 3 sizes x 3 severities;
- HEALTHY: one negative-control row;
- total: **1,036 cases per identity**;
- matrix contract run `33960765453`: **SUCCESS**;
- V3/V4/final holdout used: **FALSE**.

Frozen Phase04 quality gate:
- binary precision >= `0.95`;
- binary recall >= `0.90`;
- sticker F1 >= `0.90`;
- scribble F1 >= `0.90`;
- motion-blur F1 >= `0.85`;
- local-blur F1 >= `0.85`;
- critical minimum class F1 > `0`.

### 5.3 Expanded LR-ASPP Windows measurement — authoritative DEVELOPMENT rejection

Workflow run: **`33961309003` / SUCCESS**  
Technical SHA measured: `2692cd2919097e3dcd2f11eb892a2d31baa75e67`  
Evidence artifact: `phase04-expanded-damage-7`, artifact ID `9968067900`  
Recorded technical evidence: `config/phase04-expanded-lraspp-evidence.json` at `a04d11f67e98de9950747feff894e60460446174`.

Execution:
- Windows runner;
- exact frozen LR-ASPP ONNX SHA verified;
- `8` licensed public DEVELOPMENT portraits;
- `8 x 1,036 = 8,288` cases;
- completed `8,288/8,288`;
- error cases `0`;
- threshold `0.55`, frozen before measurement;
- V3 used `false`; V4 used `false`; V5 used `false`; final holdout used `false`;
- training/tuning during evaluation `false`.

Measured binary metrics:
- precision `0.4530042058`;
- recall `0.5011500617`;
- F1 `0.4758624332`;
- IoU `0.3122175081`;
- false-positive rate `0.0616876220`;
- false-negative rate `0.4988499383`.

Required group F1:
- STICKER `0.5404884889`;
- SCRIBBLE `0.1890941230`;
- MOTION_BLUR `0.4256175732`;
- BLUR_LOCAL `0.6830043335`;
- critical minimum: `SCRIBBLE_THIN_BLACK`, F1 `0.0515851607`.

Gate result: **FAIL**. Every threshold except `critical_min_f1_gt_0` failed. This is a quality-gate failure, not a workflow/infrastructure failure.

Runtime:
- `CPUExecutionProvider`;
- total ONNX inference `69.0765 s`;
- `0.0083345 s/case` average;
- output is mask logits only and cannot modify image pixels;
- wrong-person final pixels `0`;
- provenance violations `0`;
- restoration pass count `0`.

OpenCV 5 did not expose the legacy cascade namespace on this runner; all eight evaluation crops therefore used the explicit center-square fallback. This limitation is recorded and future broader real-domain qualification should use the verified product face detector or another pinned detector. It does not convert this failed 8,288-case gate into a pass and does not authorize threshold tuning.

DAMAGE_MASK_STATUS: **LR-ASPP REJECTED_FOR_PHASE04_FINAL / DEVELOPMENT_REFERENCE_ONLY; SMALL_U_NET REJECTED; NEXT_CHALLENGER_REQUIRED; PHASE04 ACTIVE**.

---

## 6. Upstream model policy and serious model registry

Policy: when official executable upstream exists, do not reimplement its architecture. Integration mode: `pinned_official_upstream_plus_thin_cfs_adapter`.

- FBCNN — JPEG specialist; real Windows CPU evidence; validation candidate.
- NAFNet — existing conservative deblur/denoise.
- GPEN BFR-512 — blind restoration challenger; licensing/Windows target validation pending.
- GFPGAN v1.4 — blind restoration challenger; official upstream pinned.
- CodeFormer — official upstream CPU slice exists; production license blocker.
- InstantRestore — personalized multi-reference challenger; CPU/license/checkpoint gates pending.
- RefFaceInpainting — reference-guided occlusion candidate; revision/checkpoint/CPU/Windows gates pending.
- FaceMat — investigate only if official executable code/checkpoint/license and Windows feasibility are verifiable.
- RestoreFormer++, Restormer, RestorerID, RefSTAR, DMDNet, ReF-LDM, OSDFace and reproducible NTIRE methods — research/benchmark candidates until measured and licensed.

IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED.

---

## 7. Safety invariants

- frozen quality/identity thresholds are not relaxed to make a result pass;
- wrong-person final observed pixels must remain `0`;
- provenance violations must remain `0`;
- healthy pixels cannot be rewritten outside valid repair authority;
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
PHASE_04 DamageMask: **ACTIVE / LR-ASPP FINAL CANDIDACY REJECTED**.  
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
IDENTITY_RESULTS: Phase02 multi-identity SFace gate PASS; product-wide Paper Quality identity benchmark **NOT_MEASURED**.  
WRONG_PERSON_FINAL_PIXELS: `0` in current installed-path/evaluation evidence.  
PROVENANCE_VIOLATIONS: `0`.  
TARGET_HARDWARE_TEST: **NOT_RUN**.  
TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**.  
COVERAGE: **NOT_FROZEN FOR FINAL TARGET95**.  
TOTAL_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**.  
INSTALLER_STATUS: **PARTIAL / FINAL OFFLINE PAPER QUALITY INSTALLER NOT_RUN**.  
RELEASE_READY: **FALSE**.  
PROJECT_FINISHED: **FALSE**.

A GitHub Windows runner is not physical HP EliteBook acceptance.

---

## 10. Historical engineering record — important failures preserved

### Phase02 FBCNN
1. Initial harness produced 48 runtime errors before metrics.
2. Windows subprocess stdout/stderr pipe deadlock caused timeout.
3. FBCNN ordering relative to affine resampling was corrected.
4. CPU sampling was stabilized without changing the 80% limit.
5. Full-strength FBCNN identity drift occurred in `6/48`; frozen `0.25` correction fraction passed without threshold relaxation.
6. Run `33800982565` passed 48/48.

### Phase03 installed runtime
1. Paper Quality modules were initially unreachable from the actual app runner.
2. Constructor wrapper dropped injected runtime keyword.
3. Status handling overwrote ABSTAIN/ROLLBACK semantics.
4. Malformed geometry/support metadata required fail-closed validation.
5. LR-ASPP truthful MIXED routing required class-bounded specialist subroute.
6. NumPy 2.x conflicted with PyTorch 2.1.2; optional runtime pins NumPy 1.26.x.
7. Raw FBCNN changed pixels outside sparse JPEG route; authority was bounded to the route.
8. Full pytest became a same-HEAD Windows gate; final Phase03 run passed `651/651`.

### Phase04 DamageMask
1. The initial expanded evaluation contract generated only `52` cases while claiming complete factor coverage. Corrected to a validated **1,036-case cross-product**.
2. Run `33960733395` failed because the old canonical test hardcoded `case_count == 52`; test contract was corrected. Run `33960765453` then passed.
3. Runs `33960800384` and `33960880610` exposed stale generator-test assumptions.
4. Diagnostic run `33960968287` preserved `143 failed / 899 passed`; root cause was an affine synthetic QA fixture that blur kernels could preserve exactly. Fixture was replaced by deterministic structured texture; algorithm and thresholds were unchanged.
5. Run `33961054893` passed generator authority but evaluator crashed before inference.
6. Diagnostic run `33961181037` isolated OpenCV 5 `cv2.CascadeClassifier` namespace incompatibility. Evaluator gained OpenCV 4/5 compatibility plus explicit recorded fallback; model and threshold unchanged.
7. Run `33961309003` completed **8,288/8,288** cases with zero runtime errors and honestly returned **DEVELOPMENT_GATE_FAIL**. LR-ASPP final Phase04 candidacy is rejected.

---

## 11. Latest technical push record — `a04d11f67e98de9950747feff894e60460446174`

- CURRENT_BRANCH: `integration/final-paper-quality-local`
- CURRENT_HEAD: `a04d11f67e98de9950747feff894e60460446174`
- LAST_PUSH: `2692cd2919097e3dcd2f11eb892a2d31baa75e67 -> a04d11f67e98de9950747feff894e60460446174`
- ACTIVE_PHASE: `PHASE_04_DAMAGE_MASK`
- PHASE04_MATRIX: `1036 cases/identity / cross-factorial / frozen`
- PHASE04_MATRIX_WORKFLOW: `33960765453=SUCCESS`
- PHASE04_WINDOWS_MEASUREMENT: `33961309003=SUCCESS_EXECUTION / DEVELOPMENT_GATE_FAIL`
- PHASE04_CASES: `8288/8288`, errors `0`
- PHASE04_LRASPP_BINARY: precision `0.453004`; recall `0.501150`; F1 `0.475862`; IoU `0.312218`
- PHASE04_REQUIRED_GROUP_F1: sticker `0.540488`; scribble `0.189094`; motion blur `0.425618`; local blur `0.683004`
- PHASE04_CRITICAL_MIN: `SCRIBBLE_THIN_BLACK F1=0.051585`
- ACTIVE_MODELS: `FBCNN=VALIDATION_CANDIDATE`; `LR-ASPP=REJECTED_FOR_PHASE04_FINAL/DEVELOPMENT_REFERENCE_ONLY`; `small U-Net=REJECTED`
- MODELS_ACTUALLY_EXECUTED: exact LR-ASPP ONNX over 8,288 cases; official upstream FBCNN PyTorch CPU in Phase02/03
- MODEL_LICENSE_STATUS: FBCNN final weight distribution/product manifest pending; LR-ASPP checkpoint redistribution terms not explicit
- PAPER_QUALITY_RUNTIME_WIRED: `TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS`
- DAMAGE_MASK_STATUS: `NEXT_CHALLENGER_REQUIRED / PHASE04 ACTIVE`
- REFERENCE_COUNTS_TESTED: `MAIN+0` real Windows; `MAIN+1` synthetic; `0..9 NOT_RUN`
- WRONG_PERSON_FINAL_PIXELS: `0`
- PROVENANCE_VIOLATIONS: `0`
- TARGET_HARDWARE_TEST: `NOT_RUN`
- TARGET95_ELIGIBLE_SUCCESS: `NOT_MEASURED`
- COVERAGE: `NOT_FROZEN FOR FINAL TARGET95`
- INSTALLER_STATUS: `PARTIAL / NOT_FINAL`
- RELEASE_READY: `FALSE`
- PROJECT_FINISHED: `FALSE`
- ROOT_CAUSE: prior LR-ASPP evidence came from a narrower synthetic taxonomy and weaker development gate. The complete Phase04 factor matrix reveals insufficient precision/recall and severe thin-scribble/localization generalization gaps. The correct action is model/dataset improvement, not threshold relaxation.
- EXACT_NEXT_ACTION: implement and benchmark the next lightweight multi-class DamageMask segmentation challenger on identity-disjoint legal development data using the same frozen 1,036-case matrix and Phase04 gate; do not consume V3/V4/final holdout and do not start broad restoration-model competition before DamageMask passes.
