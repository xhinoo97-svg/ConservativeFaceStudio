# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this before engineering decisions. GitHub evidence overrides chat memory. Detailed prior states remain preserved in Git history and `project-state-history/`. Previous full canonical blob before this reconciliation: `7650d88a0d9d4103c59ec10b56df6820789cf21e`.

## 0. Current canonical state

Last ledger update: `2026-09-05T23:10Z`  
Technical state verified at: `2026-09-05T23:10Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_04_DAMAGE_MASK`  
PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**  
PHASE_03_PAPER_QUALITY_RUNTIME_WIRING: **PASS / CLOSED FOR DEVELOPMENT WIRING**  
PHASE_04_DAMAGE_MASK_GATE: **FAIL / OPEN — NO QUALIFIED DAMAGE MASK**  
Last technical branch: `integration/final-paper-quality-local`  
Previous ledger technical HEAD: `a04d11f67e98de9950747feff894e60460446174`  
Current technical HEAD: `dca9b3534dbeffec11a7cee4b3dd8c449a5872d8`  
Current technical commit: `phase04: train DeepLab v2 on YuNet face-centric development bank`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
PR #2: **OPEN / DRAFT / NOT_MERGED**; preserve as non-certified historical candidate.  
Overall project status: `PARTIAL / FINAL PRODUCT NOT READY`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_RUNTIME_WIRED: **TRUE / WINDOWS DEVELOPMENT INSTALLED PATH PASS**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
RELEASE_READY: **FALSE**  
PROJECT_FINISHED: **FALSE**

Safety: no force-push, no certified-history rewrite, no auto-merge, no V3/V4 rerun, no hidden threshold relaxation, no fabricated evidence, no generated pixels presented as original, wrong-person final pixels must remain zero, provenance violations must remain zero.

## 1. Manager assessment

The project is **approximately 43/100 complete toward the requested final Paper Quality Windows product**. This is a weighted engineering estimate, not a release metric. Phase02 and Phase03 are genuinely complete; Phase04 is the dominant blocker. Later phases contain foundations/prototypes but are not release-qualified.

Do not interpret completion of 3 early phases as near-release. The highest-risk unfinished work remains: qualified DamageMask, geometry/component bank, MAIN+0..9 end-to-end matrix, specialist competition, physical target optimization, full truthful UI telemetry, Target95 benchmark, one offline installer, Clean Windows/EliteBook release acceptance and independent V5.

## 2. Installed-path truth

Installed desktop path:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

Paper Quality development path:

`PipelineWorker -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`

Phase03 proves that the real installed path can invoke Paper Quality runtime controls, `DamageMaskRuntime`, personalized reference bank, reference-first selection, fusion, identity/provenance checks and telemetry. Production Paper Quality remains fail-closed because Phase04 has no qualified damage-mask artifact.

## 3. Frozen Phase02 FBCNN evidence

Official upstream: `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`  
Checkpoint: `fbcnn_color.pth`  
SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`  
Conservative correction authority: `0.25`.

Run `33800982565` at `666cdbcfbdeee8f20901ccd063a4427d739bd107`: `8 identities x 6 profiles = 48 cases`, `48/48 PASS`, `0` errors, `0` rollbacks, wrong-person pixels `0`, provenance violations `0`, peak CPU fraction `0.75`, max system RAM fraction `0.30656`, max process RSS `2456.41 MB`.

PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**. This is development/validation evidence, not final installer/EliteBook qualification.

## 4. Frozen Phase03 installed-runtime evidence

Final same-HEAD Windows run `33957871473` at `95aff6f3b142a94d459e4112c8eb94c0f20f0efb`: targeted `81/81 PASS`; complete suite `651/651 PASS`; real installed Paper Quality path PASS; FBCNN authority bounded to JPEG subroute; generated final pixels `0` in validation shadow; wrong-person pixels `0`; provenance violations `0`; healthy pixels changed by Block 8 `0`; one-heavy-model/resource boundary enforced.

PHASE_03_PAPER_QUALITY_RUNTIME_WIRING: **PASS / CLOSED FOR DEVELOPMENT WIRING**.

## 5. Phase04 frozen DamageMask gate

Frozen expanded matrix: `1,036 cases per identity`, cross-factorial over damage type, position, size, severity and opacity where applicable. No V3/V4/final holdout use.

Frozen minimums:
- binary precision >= `0.95`;
- binary recall >= `0.90`;
- sticker F1 >= `0.90`;
- scribble F1 >= `0.90`;
- motion-blur F1 >= `0.85`;
- local-blur F1 >= `0.85`;
- critical minimum class F1 > `0`.

### 5.1 Rejected historical candidates

Small U-Net: macro-F1 `0.173198`, macro-IoU `0.113028` — **REJECTED**.

LR-ASPP expanded Windows measurement, run `33961309003`: `8,288/8,288` cases, `0` runtime errors. Binary precision `0.453004`, recall `0.501150`, F1 `0.475862`, IoU `0.312218`; sticker F1 `0.540488`, scribble `0.189094`, motion blur `0.425618`, local blur `0.683004`; critical thin-black scribble F1 `0.051585`. Execution succeeded but frozen quality gate failed — **REJECTED FOR FINAL PHASE04**.

### 5.2 Current DeepLab YuNet v2 evidence — authoritative latest result

Technical HEAD: `dca9b3534dbeffec11a7cee4b3dd8c449a5872d8`.  
Workflow: `33991389207` — GitHub conclusion **SUCCESS**. This means training/evaluation infrastructure succeeded; it does **not** mean the model passed the quality gate.  
Artifact: `phase04-deeplab-yunet-development-1`, ID `9976832756`, ZIP SHA-256 `e9f1a7ea3b5aa027bec137bc419f0db7f2fdf33c44dd3058269c0242fc28f33b`, size `41,156,000` bytes.

Data/training:
- 10 accepted legal public DEVELOPMENT portraits through verified YuNet;
- 6 training identities, 4 identity-disjoint validation identities;
- V3 used `false`; V4 used `false`; final holdout used `false`;
- architecture `torchvision_deeplabv3_mobilenet_v3_large`;
- two complete balanced matrix epochs;
- `2,072` training samples; all `1,036` unique matrix cases seen;
- training elapsed `222.47 s` CPU;
- loss `3.2961 -> 0.7139`, minimum `0.2632`;
- checkpoint `44,343,867` bytes.

Validation:
- `4 x 1,036 = 4,144` cases;
- error cases `0`;
- disposition: **DEVELOPMENT_GATE_FAIL**;
- binary precision `0.492098`;
- binary recall `0.451897`;
- binary F1 `0.471142`;
- IoU `0.308166`;
- sticker F1 `0.683359`;
- scribble F1 `0.271144`;
- motion-blur F1 `0.049557`;
- local-blur F1 `0.085047`;
- critical minimum: `BLUR_GLOBAL`, F1 `0.003785`.

Frozen gate checks: precision FAIL; recall FAIL; sticker FAIL; scribble FAIL; motion blur FAIL; local blur FAIL; only critical_min_f1_gt_0 PASS.

DAMAGE_MASK_STATUS: **NO PRODUCTION-QUALIFIED CANDIDATE. LR-ASPP REJECTED; SMALL U-NET REJECTED; DEEPLAB YUNET V2 REJECTED FOR FROZEN PHASE04 QUALITY.**

## 6. Phase04 root-cause decision

The latest DeepLab improved overlay-style damage relative to LR-ASPP in important areas (sticker `0.683` vs `0.540`; scribble `0.271` vs `0.189`) but collapsed on degradation-field classes (motion blur `0.0496` vs `0.4256`; local blur `0.0850` vs `0.6830`; global blur effectively near zero). Therefore another blind repetition of the same single multi-class segmentation recipe is not the highest-value next action.

ROOT_CAUSE: **the current one-head semantic segmentation formulation is trying to solve two materially different visual problems with one mask classifier: localized foreign overlays/occlusions versus distributed image-quality degradations such as blur/noise/JPEG. Current evidence shows conflicting specialization and inadequate generalization.**

Architecture decision for next challenger: keep a single downstream `DamageMaskRuntime` contract, but internally evaluate a **family-routed two-head detector**:
1. `OVERLAY/OCCLUSION HEAD`: opaque/translucent sticker, emoji, text, scribble, missing region, mosaic/pixelation where spatial mask is meaningful;
2. `DEGRADATION HEAD`: local/global/motion/defocus blur, noise and JPEG damage using degradation-specific features/model rather than forcing the overlay segmenter to learn them identically.

The family router itself must be measured and fail-closed. No image modification is authorized until the final emitted mask/class passes the same frozen Phase04 validation contract.

EXACT_NEXT_ACTION: **implement the smallest testable two-head Phase04 challenger on `integration/final-paper-quality-local`: preserve the frozen 1,036-case matrix and thresholds; reuse DeepLab only as an overlay challenger where measured; benchmark a lightweight degradation-specific head for blur/noise/JPEG; add contract tests proving the router cannot authorize healthy pixels or silently substitute a model; then run the same identity-disjoint Windows DEVELOPMENT gate. Do not proceed to Phase05 until a candidate passes Phase04.**

## 7. Remaining phase state

- PHASE_02 FBCNN JPEG: **100% / CLOSED PASS**.
- PHASE_03 installed PaperQualityRuntime wiring: **100% / CLOSED DEVELOPMENT PASS**.
- PHASE_04 DamageMask: **~35% / ACTIVE / quality blocker**.
- PHASE_05 geometry + 13-component bank: **~20% foundation only / NOT QUALIFIED**.
- PHASE_06 MAIN + 0..9 references: **~10% / full matrix NOT RUN**.
- PHASE_07 specialist model competition: **~20% discovery/partial benchmarks / NOT CLOSED**.
- PHASE_08 target-PC adaptation: **~20% resource framework exists / physical target NOT RUN**.
- PHASE_09 truthful 13-block UI/timeline: **~30% baseline UI exists / final telemetry contract incomplete**.
- PHASE_10 targeted training/calibration: **~20% development work only**.
- PHASE_11 Target95: **~15% metric infrastructure / final frozen benchmark NOT RUN**.
- PHASE_12 one offline installer: **~25% existing packaging foundation / final Paper Quality installer NOT RUN**.
- PHASE_13 release acceptance: **0% final sequence not run**.
- PHASE_14 independent V5: **0% / NOT CREATED**.

Weighted manager estimate to final requested product: **~43/100**.

## 8. Release truth

REFERENCE_COUNTS_TESTED: real Windows `MAIN+0`; synthetic installed E2E `MAIN+1`; full installed `MAIN+0..9` matrix **NOT_RUN**.  
IDENTITY_RESULTS: Phase02 multi-identity SFace PASS; product-wide Paper Quality identity benchmark **NOT_MEASURED**.  
WRONG_PERSON_FINAL_PIXELS: `0` in current qualifying evidence.  
PROVENANCE_VIOLATIONS: `0`.  
RAM_PEAK: FBCNN Phase02 max process RSS `2456.41 MB`; final product-wide peak **NOT_MEASURED**.  
TARGET_HARDWARE_TEST: **NOT_RUN ON PHYSICAL ELITEBOOK**.  
TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT ACHIEVED**.  
COVERAGE: **NOT FROZEN FOR FINAL TARGET95**.  
TOTAL_SUCCESS: **NOT MEASURED / NOT ACHIEVED**.  
INSTALLER_STATUS: **PARTIAL / FINAL OFFLINE PAPER QUALITY INSTALLER NOT RUN**.  
MODEL_LICENSE_STATUS: FBCNN code Apache-2.0; final redistribution manifest/checkpoint licensing for all packaged models remains a release gate.  
RELEASE_READY: **FALSE**.  
PROJECT_FINISHED: **FALSE**.

A GitHub Windows runner is not physical HP EliteBook acceptance.

## 9. Historical safety record

- FINAL_HOLDOUT_V3: **CONSUMED — NEVER RERUN / NEVER TUNE**.
- FINAL_HOLDOUT_V4: **CONSUMED_FAIL — 0/40 — NEVER RERUN / NEVER TUNE**.
- FINAL_HOLDOUT_V5: **NOT_CREATED**.
- PR #2 remains open/draft/non-certified.
- No threshold may be changed after seeing development/final evidence merely to manufacture PASS.
- Generated pixels must retain truthful provenance class.
- Wrong-person final pixels target remains exactly zero.

## 10. Mandatory cycle report

CURRENT_BRANCH: `integration/final-paper-quality-local`  
CURRENT_HEAD: `dca9b3534dbeffec11a7cee4b3dd8c449a5872d8`  
LAST_PUSH: `phase04: train DeepLab v2 on YuNet face-centric development bank`  
WORKFLOW_RUNS: latest Phase04 `33991389207=SUCCESS_EXECUTION / DEVELOPMENT_GATE_FAIL`  
TESTS: latest workflow all infrastructure/training/evaluation steps PASS; model frozen quality gate FAIL  
ACTIVE_MODELS: FBCNN validation candidate; NAFNet conservative baseline; LR-ASPP rejected reference; DeepLab YuNet v2 rejected frozen Phase04 challenger  
MODELS_ACTUALLY_EXECUTED: DeepLabV3-MobileNetV3 CPU on 4,144 identity-disjoint validation cases; FBCNN official CPU in Phase02/03  
PAPER_QUALITY_RUNTIME_WIRED: `TRUE / DEVELOPMENT INSTALLED PATH`  
DAMAGE_MASK_STATUS: `NO QUALIFIED CANDIDATE / TWO-HEAD CHALLENGER NEXT`  
REFERENCE_COUNTS_TESTED: `MAIN+0 real; MAIN+1 synthetic; 0..9 NOT RUN`  
WRONG_PERSON_FINAL_PIXELS: `0` in current qualifying evidence  
PROVENANCE_VIOLATIONS: `0`  
TARGET_HARDWARE_TEST: `NOT_RUN`  
TARGET95_ELIGIBLE_SUCCESS: `NOT_MEASURED`  
COVERAGE: `NOT_FROZEN FINAL`  
TOTAL_SUCCESS: `NOT_MEASURED`  
INSTALLER_STATUS: `PARTIAL / NOT FINAL`  
RELEASE_READY: `FALSE`  
PROJECT_FINISHED: `FALSE`  
ROOT_CAUSE: `single-head segmentation formulation conflicts across overlay masks and distributed degradation fields`  
EXACT_NEXT_ACTION: `build and benchmark family-routed two-head DamageMask challenger under unchanged frozen Phase04 gate`.
