# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this before engineering decisions. GitHub evidence overrides chat memory. Detailed historical states remain preserved in Git history and `project-state-history/`; important failures and decisions remain summarized here.

## 0. Current canonical state

Last ledger update: `2026-09-05T06:52Z`  
Technical state verified at: `2026-09-05T06:52Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_03_PAPER_QUALITY_RUNTIME_WIRING`  
PHASE_02_JPEG_FBCNN_GATE: **PASS / CLOSED**  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `c806f00e17aa074d906b6473fba7d5518e1afcd2`  
Last technical HEAD: `b108def622ac1714d323020dab96709254c34b30`  
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

EXACT_NEXT_ACTION: require the **complete pytest suite on the same Phase03 Windows installed-path HEAD**. Add a full-suite step to `.github/workflows/paper-quality-installed-fbcnn-windows.yml`, push one minimal CI commit, and do not advance to Phase04 until targeted tests and full suite are green on that new exact SHA. FBCNN remains validation-shadow/non-production.

---

## 1. Product and installed-path truth

CFS is one local Windows product with two evidence classes of operation: Conservative/Forensic reconstruction and a future Paper Quality mode. The installed desktop path is:

`app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`

With the Paper Quality feature flag enabled, the real installed path reaches:

`PipelineWorker -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`

Block 8 can invoke `DamageMaskRuntime`, damage routing, model qualification, `PersonalizedReferenceBank`, component selection, reference-first repair, calibrated candidate selection when configured, component-aware fusion, per-pixel provenance and the existing identity rollback. The shipped default remains `paper_quality_enabled=false`.

The project policy is **official upstream implementation first**. When an executable paper repository exists, CFS pins and executes upstream code/checkpoints rather than recreating the architecture. CFS owns thin adapters, routing, identity/provenance firewalls, resource control, checkpoint verification, Windows/offline packaging, benchmarking and release gates. Upstream code is not assumed bug-free.

PAPER_QUALITY_RUNTIME_WIRED: **TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS**  
PAPER_QUALITY_MODE_READY: **FALSE**  
FBCNN_INSTALLED_PATH_STATUS: **WINDOWS_DEVELOPMENT_PASS / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED**

---

## 2. Branch / holdout safety map

- `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`: certified V1, immutable.
- `integration/final-paper-quality-local@b108def622ac1714d323020dab96709254c34b30`: active technical branch.
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

Aggregate profile evidence:

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

## 4. Current Phase03 installed-path evidence

### Exact-head run `33950854805` — SUCCESS

Technical SHA: `b108def622ac1714d323020dab96709254c34b30`  
Workflow: `Paper Quality installed FBCNN Windows validation` run `#6`  
Artifact ID: `9964797262`  
Artifact ZIP digest: `sha256:e9b67e8ef0b56859d143bd28a2748841caec4563121154c1007986d2a01d2b91`.

Environment/evidence:
- Windows Server 2025 GitHub runner; **not physical EliteBook evidence**;
- Python `3.11.9`;
- `torch 2.1.2+cpu`, `torchvision 0.16.2+cpu`, `numpy 1.26.4`, `onnxruntime 1.29.0`, CUDA false;
- exact official FBCNN repository revision verified;
- exact FBCNN checkpoint SHA-256 and size verified before load;
- real LR-ASPP ONNX executed from frozen DEVELOPMENT artifact;
- normal six-model application pack bootstrapped;
- public-domain NASA DEVELOPMENT portrait, QF10 JPEG, MAIN+0;
- real entry path: `PipelineWorker.run -> AutomaticPipelineRunner -> InstalledPaperQualityRuntime`;
- targeted Phase03 workflow tests: `81/81 PASS`;
- **FULL_SUITE_TESTS on `b108...`: NOT_VERIFIED**. Previous `649 PASS / 2 SKIPPED` belongs to `c806...` and is not reused as same-HEAD proof.

Observed damage/router output on the DEVELOPMENT portrait:
- parent route: `MIXED`;
- JPEG_ARTIFACT: `16768` admitted pixels, fraction `0.6515387007`, mean confidence `0.7818839550`;
- SCRIBBLE: `8968` admitted pixels, fraction `0.3484612993`, mean confidence `0.8190361261`;
- specialist model route: `JPEG_ARTIFACT` only.

FBCNN raw vs bounded authority:
- face crop `55 x 66` pixels;
- raw candidate changed `3605` crop pixels;
- raw changed outside admitted JPEG route: `3552` pixels;
- JPEG route mask: `54` pixels;
- bounded candidate changed: `53` pixels;
- bounded generated mask: `53` pixels;
- bounded candidate changes outside route: **`0`**;
- route bounding applied: TRUE;
- candidate remained `INSTALLED_PATH_VALIDATION_SHADOW` and `fused_to_final=false`;
- final generated pixels: `0`;
- wrong-person final pixels: `0`;
- provenance violations: `0`;
- healthy pixels changed by Block 8: `0`;
- Block 8 output SHA equals accepted Block-8 input SHA.

Resource/lifecycle evidence:
- CPU affinity budget: `3/4` logical processors, max fraction `0.8`;
- heavy-model concurrency: `1`;
- process RSS after inference: `1399259136` bytes;
- system RAM fraction after inference: `0.2493143231`;
- model load: about `1.4796 s`;
- inference: about `0.19364 s` on the `55 x 66` crop;
- model unload boundary recorded;
- CPU_PEAK as a measured utilization fraction: **NOT_RECORDED** for this run.

Interpretation: official upstream FBCNN is a whole-crop restorer and naturally changes pixels outside a sparse class mask. CFS now measures that raw behavior truthfully but removes all candidate authority outside the audited JPEG subroute **before any future candidate selection/fusion**. No upstream weights or architecture were modified.

---

## 5. DamageMask state

- small U-Net: **REJECTED**, macro-F1 `0.173198`, macro-IoU `0.113028`.
- LR-ASPP: **DEVELOPMENT / NOT_PRODUCTION_QUALIFIED**, F1 `0.716639`, IoU `0.579849` from existing research evidence.
- Installed LR-ASPP inference has executed successfully on Windows development path but global/domain qualification is not complete.
- Phase04 must evaluate per-class precision/recall/F1/IoU, healthy false positives, damage false negatives, opacity, size and facial position for opaque/translucent sticker, emoji, text, thin/thick/color scribble, local/global blur, motion blur, defocus, mosaic, pixelation, JPEG, noise and mixed damage.

DAMAGE_MASK_STATUS: **LR-ASPP DEVELOPMENT_NOT_PRODUCTION; SMALL_U_NET REJECTED**.

---

## 6. Upstream model policy and active model registry

Policy: `forbidden_when_official_executable_upstream_exists` to reimplement model architecture; use `pinned_official_upstream_plus_thin_cfs_adapter`.

Current serious models include:
- FBCNN — JPEG specialist, real Windows CPU evidence, validation candidate;
- NAFNet — existing conservative deblur/denoise;
- GPEN BFR-512 — blind restoration challenger, official upstream pinned, licensing/Windows target validation pending;
- GFPGAN v1.4 — blind restoration challenger, official upstream pinned;
- CodeFormer — official upstream CPU slice exists, production license blocker;
- InstantRestore — personalized multi-reference challenger, CUDA/CPU feasibility and licensing/checkpoint gates pending;
- RefFaceInpainting — reference-guided occlusion specialist candidate; revision/checkpoint/CPU/Windows gates pending;
- RefineFIR — only promote if executable upstream/checkpoint becomes verifiable;
- OSDFace — severe one-step diffusion challenger, CPU/Windows/resource gates pending;
- RestoreFormer++ and other specialists — research until measured.

Implementation existence is not qualification. `QUALIFIED` requires exact checkpoint/license, real execution, identity/provenance gates, Windows/offline pack and target hardware evidence.

---

## 7. Safety invariants

- SFace safety threshold is not lowered to make quality pass;
- wrong-person final observed pixels must remain `0`;
- provenance violations must remain `0`;
- healthy pixels must not be rewritten outside valid repair authority;
- generated pixels are `GENERATED_MODEL_INFERRED`, never observed/original;
- one heavy model at a time;
- <=80% logical CPU allocation and <=80% process/system RAM;
- no hidden fallbacks or model-name substitution;
- no hidden abstention used to inflate success;
- no V3/V4 reuse.

---

## 8. Phase sequence

PHASE_02 FBCNN JPEG: **CLOSED / PASS**.  
PHASE_03 PaperQualityRuntime wiring: **ACTIVE — real Windows installed path PASS, FBCNN class-bounded shadow PASS, exact-HEAD full suite still required**.  
PHASE_04 DamageMask: **BLOCKED BY PHASE03 FULL-SUITE GATE**.  
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

REFERENCE_COUNTS_TESTED: real Windows installed path `MAIN+0`; synthetic installed E2E `MAIN+1`; installed `MAIN+0..9` matrix **NOT_RUN**.  
IDENTITY_RESULTS: Phase02 multi-identity SFace gate PASS; current installed shadow run grants no generated final authority; product-wide Paper Quality identity benchmark **NOT_MEASURED**.  
WRONG_PERSON_FINAL_PIXELS: `0` in exact current installed-path evidence; product-wide final benchmark NOT_MEASURED.  
PROVENANCE_VIOLATIONS: `0` in exact current installed-path evidence; product-wide final benchmark NOT_MEASURED.  
HEALTHY_PIXELS_CHANGED: `0` by Block 8 on exact current installed-path evidence.  
TARGET_HARDWARE_TEST: **NOT_RUN**.  
TARGET95_ELIGIBLE_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**.  
COVERAGE: **NOT_FROZEN FOR FINAL TARGET95**.  
TOTAL_SUCCESS: **NOT_MEASURED / NOT_ACHIEVED**.  
INSTALLER_STATUS: **PARTIAL / PAPER QUALITY VALIDATION PACK NOT DISTRIBUTED / FINAL OFFLINE INSTALLER NOT_RUN**.  
RELEASE_READY: **FALSE**.  
PROJECT_FINISHED: **FALSE**.

A GitHub Windows runner is not physical HP EliteBook acceptance.

---

## 10. Historical engineering record — preserved important failures

### Phase02 FBCNN
1. Initial harness produced 48 runtime errors before metrics.
2. Windows subprocess stdout/stderr pipe deadlock caused timeout; corrected.
3. FBCNN was executed after affine resampling, altering JPEG artifacts; corrected to `JPEG degradation -> FBCNN -> metric alignment`.
4. 100 ms CPU sampling overstated transient usage; observation stabilized without changing the frozen 80% limit.
5. Full-strength FBCNN identity drift occurred in `6/48`; fixed by frozen `0.25` correction fraction without threshold relaxation.
6. Exact-head run `33800982565` then passed 48/48.

### Phase03 installed runtime
1. Developed Paper Quality modules were not reachable from the actual app runner.
2. Constructor wrapper dropped the injected runtime keyword.
3. Execution status overwrote explicit ABSTAIN/ROLLBACK semantics.
4. Malformed geometry/support metadata did not fail closed cleanly.
5. Real LR-ASPP produced a truthful MIXED route; class-bounded specialist subroute was required.
6. NumPy 2.x was ABI-incompatible with PyTorch 2.1.2.
7. Official FBCNN import required the compatible torchvision package.
8. Block 8 conflated immutable MAIN with current accepted transactional input.
9. Official FBCNN raw candidate changed pixels outside the sparse JPEG subroute; `b108def622ac1714d323020dab96709254c34b30` now bounds candidate image and generated mask to route authority while retaining raw telemetry.

Historical workflow failures remain evidence and are not erased: `33896157286`, `33949488127`, `33949705168`, `33949966047`; first fully successful real installed FBCNN run `33950404788`; bounded-candidate success `33950854805`.

---

## 11. Technical push record — `b108def622ac1714d323020dab96709254c34b30`

- CURRENT_BRANCH: `integration/final-paper-quality-local`
- CURRENT_HEAD: `b108def622ac1714d323020dab96709254c34b30`
- LAST_PUSH: `c806f00e17aa074d906b6473fba7d5518e1afcd2 -> b108def622ac1714d323020dab96709254c34b30`
- ACTIVE_PHASE: `PHASE_03_PAPER_QUALITY_RUNTIME_WIRING`
- FILES_CHANGED: `app/installed_paper_quality_runtime.py`, `tests/test_installed_paper_quality_path.py`, installed FBCNN validation contract/evidence paths associated with the same tree
- WORKFLOW_RUNS: `33950854805=SUCCESS`, artifact `9964797262`, digest `e9b67e8ef0b56859d143bd28a2748841caec4563121154c1007986d2a01d2b91`
- TARGETED_TESTS: `81/81 PASS`
- FULL_SUITE_TESTS: `NOT_VERIFIED ON THIS EXACT HEAD`; prior `649 PASS / 2 SKIPPED` belongs to `c806...`
- INSTALLED_PATH_TEST: `WINDOWS_DEVELOPMENT_PASS`
- ACTIVE_MODELS: `FBCNN=VALIDATION_CANDIDATE`; `LR-ASPP=DEVELOPMENT_NOT_PRODUCTION`; `small U-Net=REJECTED`
- MODELS_ACTUALLY_EXECUTED: `LR-ASPP ONNX`, official upstream `FBCNN PyTorch CPU`
- MODEL_LICENSE_STATUS: `FBCNN_CODE_APACHE_2_0; FINAL_WEIGHT_DISTRIBUTION_MANIFEST_PENDING; PRODUCT_LICENSE_PENDING`; LR-ASPP research-only pending explicit weight terms
- PAPER_QUALITY_RUNTIME_WIRED: `TRUE / WINDOWS_DEVELOPMENT_INSTALLED_PATH_PASS`
- FBCNN_INSTALLED_PATH_STATUS: `PASS / JPEG_SUBROUTE_BOUNDED / VALIDATION_SHADOW / NOT_PRODUCTION_QUALIFIED`
- DAMAGE_MASK_STATUS: `LR-ASPP DEVELOPMENT_NOT_PRODUCTION`
- REFERENCE_COUNTS_TESTED: `MAIN+0` real Windows; `MAIN+1` synthetic wiring; `0..9 NOT_RUN`
- IDENTITY_RESULTS: no FBCNN pixels fused; Phase02 identity evidence remains valid; product-wide metric NOT_MEASURED
- WRONG_PERSON_FINAL_PIXELS: `0`
- PROVENANCE_VIOLATIONS: `0`
- HEALTHY_PIXELS_CHANGED: `0` by Block 8
- RAM_PEAK: `1399259136 bytes RSS` after inference; system RAM fraction `0.2493143231`
- CPU_PEAK: `NOT_RECORDED`; execution restricted to `3/4` logical processors
- TARGET_HARDWARE_TEST: `NOT_RUN`
- TARGET95_ELIGIBLE_SUCCESS: `NOT_MEASURED`
- COVERAGE: `NOT_FROZEN`
- TOTAL_SUCCESS: `NOT_MEASURED`
- INSTALLER_STATUS: `PARTIAL / NOT_FINAL`
- RELEASE_READY: `FALSE`
- PROJECT_FINISHED: `FALSE`
- ROOT_CAUSE: whole-crop upstream FBCNN output exceeded a sparse JPEG class mask; candidate authority is now explicitly class-bounded before selection while raw behavior remains auditable
- EXACT_NEXT_ACTION: add and run complete pytest in the same Windows installed-path workflow on the next minimal technical SHA; fix any regression before Phase04.
