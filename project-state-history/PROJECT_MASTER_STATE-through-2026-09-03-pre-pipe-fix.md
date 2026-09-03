# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Full history through 2026-08-24 is preserved byte-for-byte at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`.

## 0. Document metadata

Last ledger update: `2026-09-01T17:52Z`  
Technical state verified at: `2026-09-01T17:52Z`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
ACTIVE_PHASE: `PHASE_02_JPEG_FBCNN`  
PHASE_GATE: `IN_PROGRESS / NOT_VERIFIED`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `bc732272a91ec8ad288e415e8600bf421b1485e9`  
Last technical HEAD: `0342a2c3fa831d82073b484e8755a4aee778fcd6`  
Last technical tree: `940596f39cf668e228c9783871d50b67caa4284b`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
Current active engineering track: final integration, upstream-first Paper Quality + immutable Conservative safety  
Current exact blocker: FBCNN Phase02 same-HEAD Windows multi-identity validation run `33540310269` is in progress. Do not advance to PHASE_03 until its evidence is classified and FBCNN JPEG routing/integration gate is closed.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No force-push certified history, no auto-merge, no consumed-holdout rerun.

EXACT_NEXT_ACTION: inspect run `33540310269`; if FAIL, read the failing job/log and fix only the first real Phase02 root cause; if SUCCESS, retrieve the exact artifact/report and use it to complete FBCNN routing/integration evidence without entering PHASE_03 prematurely.

---

## 1. Executive project summary

CFS is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative Mode is evidence-faithful; Paper Quality generation is always `GENERATED_MODEL_INFERRED` and subordinate to observed MAIN/same-person evidence.

Active development is `integration/final-paper-quality-local`, based on immutable certified `main`. Direction is **UPSTREAM-FIRST**: use official executable paper/model repositories at pinned revisions; CFS adds thin adapters, exact source/checkpoint verification, identity/provenance safety, <=80% resource control, routing/fusion, Windows/offline packaging and qualification.

The real installed entry path has been reconciled: `app.__main__.main -> MainWindow -> PipelineWorker -> AutomaticPipelineRunner`. `AutomaticPipelineRunner` still executes the legacy 13-block path directly and does **not** yet invoke `PaperQualityRuntime`; that is PHASE_03 work and is intentionally not being changed during PHASE_02.

The current phase is FBCNN JPEG qualification. CFS is reusing the exact official upstream implementation and the already-successful historical benchmark code rather than reimplementing FBCNN. The new same-HEAD Windows gate expands evidence from one development identity to at least eight identities and six JPEG/recompression profiles, adds standard LPIPS, real SFace-only identity evidence, CPU/RAM telemetry, and preserves exact checkpoint/source identity.

---

## 2. Branch and release map

| Branch | Purpose | Current HEAD | Status | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN/RELEASED | certified history | preserve |
| `integration/final-paper-quality-local` | final integration | `0342a2c3fa831d82073b484e8755a4aee778fcd6` | ACTIVE / PHASE_02 | FBCNN run `33540310269` IN_PROGRESS | classify Windows multi-identity FBCNN evidence |
| `hotfix/real-world-restoration-v1.1` | Track A evidence | `77687b3b171f4e9989fcf486834f2d8b7a52f591` | V4 CONSUMED_FAIL/FROZEN | PR #2 draft/no-go | never rerun V4 |
| `protocol/v5-certification-hardening` | protocol DEV | `268188c5a2540455ff804383cb583b16546b62f1` | ARCHIVED DEV | synthetic PASS | no V5 before prerequisites |
| `research/paper-quality-local-v2` | research evidence | `6d57725aae087bb4a3144d521d91346999f9a4fd` | SUPERSEDED AS ACTIVE ARCHITECTURE | preserve evidence | port only measured winners |
| `research/face-restoration-v2` | early research | `757a3f60e2f012a1d0b1758c7280bfdd492f33df` | ARCHIVED | historical | preserve |
| `feature/block-pipeline-v1` | V1 history | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | ARCHIVED | historical | preserve |
| `release/v1-certified` | V1 release history | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | ARCHIVED | historical | preserve |
| `meta/project-state` | canonical ledger | self-SHA omitted | ACTIVE META | docs only | update after every technical push |

---

## 3. Product version roadmap

- **PRODUCT_V1 — RELEASED:** certified conservative baseline; SFace `0.363`, wrong-person observed `0`, provenance violations `0`.
- **PRODUCT_V1_1 — FAILED V4 CANDIDATE/PRESERVED:** V4 consumed and failed before case 1; no rerun.
- **PRODUCT_V2 — IMPLEMENTING/BENCHMARKING:** upstream specialists, damage routing, calibrated selection, provenance-safe fusion, 80% resource contract; FBCNN is the active PHASE_02 specialist and is not production-qualified yet.
- **PRODUCT_V3 — DESIGNING/PROTOTYPES:** MAIN + 0–9 references, 13-component authority.
- **PRODUCT_V4 — DESIGNING/PROTOTYPES:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified offline Windows product + physical EliteBook acceptance; V5 holdout does not exist.

Product versions are independent from holdout versions.

---

## 4. Holdout / benchmark lineage

FINAL_HOLDOUT_V3: **CONSUMED**, never rerun/tune.  
FINAL_HOLDOUT_V4: **CONSUMED_FAIL**, STARTED persisted, 0/40, never rerun/tune.  
FINAL_HOLDOUT_V5: **NOT_CREATED**.  
Female-domain: stress/report evidence, Target95 not achieved.  
FBCNN DEV matrix: one identity, six compression profiles PASS, historical run `32674085939`.  
FBCNN Phase02 Windows validation: at least 8 independently sourced public adult identities x 6 frozen JPEG/recompression profiles; VALIDATION only; run `33540310269` IN_PROGRESS; no final holdout material.  
DamageMaskNet U-Net: stopped model/data hypothesis.  
LR-ASPP external DEVELOPMENT: 40 identities/880 cases, aggregate pass with subgroup gaps.

---

## 5. Current global objectives

OBJ-001 V1 preserve — **PASS**.  
OBJ-002 V3/V4 consumed evidence preserve — **PASS**.  
OBJ-003 canonical ledger — **IN_PROGRESS**.  
OBJ-004 production-qualified damage mask — **IN_PROGRESS / FUTURE PHASE_04; do not execute now**.  
OBJ-005 broad BFR validation — **IN_PROGRESS / FUTURE PHASE_07; do not execute now**.  
OBJ-006 FBCNN qualification — **IN_PROGRESS / PHASE_02 / WINDOWS VALIDATION RUNNING**.  
OBJ-007 personalized reference system — **IN_PROGRESS / FUTURE PHASE_06**.  
OBJ-008 RefFace — **BLOCKED / FUTURE PHASE_07**.  
OBJ-009 Paper Quality Windows pack — **IN_PROGRESS foundation**.  
OBJ-010 physical EliteBook — **NOT_RUN / PHASE_08/13**.  
OBJ-011 upstream registry — **PASS foundation**.  
OBJ-012 exact generated-model authority — **PASS for integration contract** at `bc732272...`; model qualification remains separate.  
OBJ-013 V5 protocol — **DEV VALIDATING only; PHASE_14**.

---

## 6. Model master registry

Certified V1: YuNet, SFace, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2, constrained LaMa.

Research/upstream-first:
- **FBCNN** `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`; checkpoint `fbcnn_color.pth`, 287755111 bytes, SHA-256 `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`: **DEV_PASS / VALIDATION_RUNNING**. Code license Apache-2.0. Official repository LICENSE and README state the project is Apache-2.0; checkpoint is an official v1.0 release asset and no separate restrictive checkpoint terms were found in those official sources. This is evidence of project licensing, not legal advice.
- **GPEN** `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`: BENCHMARKING/license unresolved; not current phase.
- **GFPGAN v1.4** `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`: BENCHMARKING; not current phase.
- **CodeFormer** `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`: BENCHMARKING/BLOCKED_LICENSE; not current phase.
- DamageMaskNet small U-Net: REJECTED/STOPPED.
- LR-ASPP: DEVELOPMENT validation only, NOT_QUALIFIED.
- RefFaceInpainting: FEASIBILITY_ONLY/BLOCKED.
- InstantRestore: FEASIBILITY_ONLY/BLOCKED_HARDWARE until proven.
- Others: DISCOVERED/AUDITED/FEASIBILITY_ONLY.

IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED.

---

## 7. Current model evidence

GPEN DEV SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474`.  
GFPGAN1.4 DEV SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604`.  
FBCNN QF20 DEV PSNR `34.62->36.78`, SSIM `0.9486->0.9634`, SFace `0.9571->0.9691`, `~1.305GB`.  
FBCNN run `32674085939`: 6/6 DEV profiles PASS, one identity, artifact `9502200502`, archive SHA `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`.  
FBCNN Windows Phase02 run `33540310269`: **IN_PROGRESS / NOT_VERIFIED** at this ledger update.  
DamageMaskNet U-Net macro-F1 `0.173198`, macro-IoU `0.113028`, STOPPED.  
LR-ASPP external DEV F1 `0.716639`, IoU `0.579849`, domain gaps.

Paper figures are not CFS/Windows/EliteBook results unless reproduced.

---

## 8. 13-block architecture

1 IMPORT deterministic. 2 DEBLUR NAFNet / qualified BFR only. 3 ENHANCE FBCNN candidate for detected JPEG; **current PHASE_02**. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK no production-qualified multi-class model yet. 7 REGION_SELECT 13-component bank. 8 INPAINT observed first, qualified generated specialist second. 9 FUSION MAIN > observed ref > generated; route + qualification + exact artifact identity required. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`. 12 UPSCALE Lanczos unless measured SR qualifies. 13 EXPORT deterministic provenance/model/resource/artifact identity.

Current installed production path still bypasses `PaperQualityRuntime`; PHASE_03 will wire it only after PHASE_02 closes.

---

## 9. Photo and input contract

MAIN: smartphone/social compression, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, missing components, crop/partial, low light, mixed/unknown damage. References MAIN + 0–9 full/partial/component-only/different pose/expression/light/resolution/degraded/useless/wrong-person images. Full accepted same-person may global-anchor; partial local only; wrong-person never anchor/donor/score booster.

---

## 10. Dataset construction

Target ~300–400 identity-disjoint research/validation sources/cases with explicit domain composition; store source/license/date/identity/hash/resolution/split/degradation/severity/seed/mask/reference relationships. Final holdouts never train/tune.

For PHASE_02, FBCNN validation uses public adult portraits resolved through the existing license-filtered source resolver; each selected source records source SHA-256, page URL and license metadata. It does not use V3/V4 or V5.

---

## 11. Component-by-component reconstruction

LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Observed same-person evidence outranks generated inference.

---

## 12. Damage routing

Routes cover HEALTHY, GAUSSIAN_BLUR, MOTION_BLUR, DEFOCUS, JPEG_ARTIFACT, NOISE, PIXELATION, OCCLUSION, SCRIBBLE, TEXT_WATERMARK, MIXED, SMALL_FACE, PARTIAL_CROP. Missing/unverified evidence abstains; malformed/inconsistent evidence rolls back. Generated candidates require complete production qualification, route attestation and exact candidate repo/revision/checkpoint match before fusion.

FBCNN backend accepts only JPEG/compression/recompression contexts or explicit verified `jpeg_detected=True`; tests reject blur/mosaic/healthy routes. The route planner names FBCNN for `JPEG_ARTIFACT`, but production execution remains fail-closed until model qualification evidence is complete.

---

## 13. Decision log

**DEC-20260828-014 route/production attestation — ACCEPTED/IMPLEMENTED/TEST_PASS.**

**DEC-20260828-015 candidate artifact identity continuity — ACCEPTED/IMPLEMENTED/TEST_PASS.** Exact repo/revision/checkpoint identity is extracted from typed production evidence and compared against each generated candidate before selector/fusion. FBCNN emits its verified upstream identity explicitly.

**DEC-20260901-016 upstream benchmark reuse for FBCNN Phase02 — ACCEPTED/IMPLEMENTED.** Rather than reimplement FBCNN or its previously proven development harness, the integration branch reuses the exact historical benchmark modules/blobs that produced run `32674085939`, then adds only a Windows multi-identity orchestration/LPIPS layer. Reversal condition: reused harness is shown to be incompatible with current integration semantics or produces invalid evidence.

---

## 14. Experiment log

**EXP-20260828-034 route/qualification binding attempt 1/3 — PASS.** Run `33207031788`, targeted 46/46, full 602/602.

**EXP-20260828-035 candidate artifact identity attempt 1/3 — PASS.** Technical sequence `e617550e... -> 65d97133... -> bc732272...`; exact-head run `33208101745` SUCCESS. Targeted runner tests `50/50 PASS`; complete pytest `608/608 PASS`; success/pre-marker-failure/post-marker-failure protocol ordering PASS. Artifact `one-shot-protocol-hardening-33208101745`, ID `9700486979`, archive SHA-256 `c20dbcfb852c6c72f2099618b9e18eda905d2fb63c348d8a0f5769e4759ba8bc`. This proves the contract, not model quality/qualification.

**EXP-20260901-036 FBCNN Phase02 Windows multi-identity qualification attempt 1/3 — IN_PROGRESS / NOT_VERIFIED.**
- Technical push: `bc732272a91ec8ad288e415e8600bf421b1485e9 -> 0342a2c3fa831d82073b484e8755a4aee778fcd6`.
- Workflow: `FBCNN Phase02 Windows qualification`, run `33540310269`, same candidate SHA.
- Required matrix: >=8 identities x six frozen compression profiles (QF10, QF20, QF40, double JPEG, social resize/recompression, mosquito-edge JPEG stress).
- Metrics: PSNR, SSIM, standard LPIPS AlexNet, real SFace-only identity, process/system RAM, CPU observation, model load/inference time, provenance.
- Source/model policy: official FBCNN upstream pinned at `54d183...`; exact checkpoint hash `8b0e4ef...`; no architecture reimplementation.
- Result: pending. Do not infer PASS.

---

## 15. Quality scoreboard

DEV evidence exists; broad validation incomplete. V3/V4 consumed/forbidden. Target-PC Paper Quality NOT_RUN. SFace `0.363`, wrong-person observed `0`, provenance `0`, healthy/outside MAE `<=8.0` where frozen. Target95 not achieved.

PHASE_02 Windows validation scoreboard is `NOT_VERIFIED` until run `33540310269` finishes.

---

## 16. Target hardware

HP EliteBook 1030 G3, Windows, 16GB; exact CPU/GPU runtime detected. <=80% logical CPU, <=80% process/system RAM, one heavy model, CPU-first/no CUDA. Windows GitHub runner evidence is not physical EliteBook evidence.

---

## 17. Release safety rules

Never lower frozen safety thresholds, allow wrong-person/provenance violations, cherry-pick, relabel generated as observed, use proxy as SFace, rerun V3/V4, auto-merge, force-push certified history or fabricate evidence.

---

## 18. Provenance classes

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

---

## 19. Track A

Preserved evidence only. V4 `CONSUMED_FAIL`, 0/40, no rerun. PR #2 is not certified.

---

## 20. Track B / final integration

`ACTIVE_PHASE=PHASE_02_JPEG_FBCNN`.

Active integration uses official paper code directly where executable. FBCNN is a thin pinned-upstream adapter; no architecture copy. Historical proven FBCNN benchmark code was ported by exact Git blob identity into the active integration branch. A new Windows validation driver expands the evidence to multiple identities and LPIPS/CPU telemetry.

Do not work on PaperQualityRuntime wiring, DamageMask, face geometry, reference-count matrix, model competition, UI, training, Target95, installer, release testing or V5 until the Phase02 gate closes.

---

## 21. Current Paper Quality blocker

Run `33540310269` is the current blocker. It must produce or fail to produce real Windows CPU evidence from official FBCNN bytes.

PASS requirements include:
- exact upstream revision/checkpoint;
- >=8 usable identities;
- six compression profiles per identity;
- no runtime errors;
- real SFace gate;
- PSNR/SSIM/LPIPS improvement at profile aggregate level;
- zero wrong-person observed pixels;
- zero provenance violations;
- <=80% system RAM and bounded CPU;
- FBCNN remains unavailable for healthy/non-JPEG routes.

Even a Windows validation PASS does not create production authority by itself; installed-offline same-candidate and physical EliteBook evidence remain future gates. No threshold or dataset may be changed after seeing this run merely to manufacture a PASS.

---

## 22. Specialist model strategy

INPUT -> detect/align -> damage -> refs/identity -> qualified specialist -> candidate -> hard gates -> component fusion -> final identity/provenance -> export. Use best measured specialist, not all models.

---

## 23. Model selection policy

Multi-identity DEV/VALIDATION by damage family; identity hard gate first, then perceptual/geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/RAM/runtime. Never tune on final holdout. Official implementation preferred but not presumed target-compatible.

---

## 24. Historical record

Full history through 2026-08-24: `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`.

---

## 25. Push journal

### PUSH-20260828-001
`f33880d5... -> 75b31aab... -> 8bcc801e...`; route requires actual production qualification/attestation. Run `33207031788` SUCCESS, targeted 46/46, full 602/602.

### PUSH-20260828-002
`8bcc801e... -> e617550e...`; add generic model artifact identity parser. Initial state NOT_VERIFIED.

### PUSH-20260828-003
`e617550e... -> 65d97133...`; add exact/ambiguous artifact identity tests. Initial state NOT_VERIFIED.

### PUSH-20260828-004
- Previous: `65d97133311fff1a062be3bc821e9c3de03ec365`.
- New: `bc732272a91ec8ad288e415e8600bf421b1485e9`, tree `bec3bca903d5b236b5201c941bffe208996276de`.
- Atomic files: `app/face_restorer_adapter.py`, `app/fbcnn_upstream_backend.py`, `app/paper_quality_runtime.py`, `tests/test_paper_quality_runtime.py`, `tests/test_fbcnn_upstream_backend.py`.
- Result: generated candidate explicit upstream identity + exact qualification match before fusion.
- Exact-head workflow: run `33208101745` SUCCESS; targeted `50/50`; full `608/608`; protocol ordering PASS.
- Artifact: ID `9700486979`; SHA-256 `c20dbcfb852c6c72f2099618b9e18eda905d2fb63c348d8a0f5769e4759ba8bc`.
- No model promoted; no threshold/dataset/holdout/V5 change.

### PUSH-20260901-001
- Previous: `bc732272a91ec8ad288e415e8600bf421b1485e9`.
- New: `0342a2c3fa831d82073b484e8755a4aee778fcd6`, tree `940596f39cf668e228c9783871d50b67caa4284b`.
- Commit: `test(jpeg): qualify FBCNN across Windows identities`.
- Reused exact historical blobs: `research/run_fbcnn_vertical_slice.py`, `research/fbcnn_degradation_matrix.py`, `research/run_gfpgan14_vertical_slice.py`, `research/run_gfpgan14_vertical_slice_exact.py`, `tests/test_fbcnn_degradation_matrix.py`.
- New Phase02 orchestration: `research/run_fbcnn_windows_validation.py`.
- New same-HEAD workflow: `.github/workflows/fbcnn-phase02-windows.yml`.
- Workflow run: `33540310269`, IN_PROGRESS at ledger update.
- No model promoted; no thresholds changed; no consumed holdout accessed.

### PUSH-20260828-META-001
Prior ledger preserved byte-for-byte before reconciliation; meta-only.

---

## 26. Session continuity rule

Every session reads this ledger, reconciles GitHub, continues the current `ACTIVE_PHASE`, and updates this file after every technical push. Never infer PASS/QUALIFIED/RELEASE_READY/Target95/EliteBook/PROJECT_FINISHED without exact evidence. If `PROJECT_FINISHED=FALSE` and no HARD_BLOCKER exists, execute `EXACT_NEXT_ACTION` rather than writing a new plan.
