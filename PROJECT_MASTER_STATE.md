# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Full prior history through 2026-08-24 is preserved byte-for-byte at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`.

## 0. Document metadata

Last ledger update: `2026-08-28`  
Technical state verified at: `2026-08-28`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `e617550e368dd376bba64c7d94a3516d916032f2`  
Last technical HEAD: `65d97133311fff1a062be3bc821e9c3de03ec365`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
Current active engineering track: final integration, upstream-first Paper Quality + immutable Conservative safety  
Current exact blocker: model artifact identity parser + parser tests exist, but `RestorationCandidate`, FBCNN output and Paper Quality runtime are not yet wired to that identity. Current partial head is **NOT_VERIFIED** and cannot authorize generated pixels.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No force-push certified history, no auto-merge, no V3/V4 rerun.

---

## 1. Executive project summary

Conservative Face Studio is a local Windows face-restoration product for damaged smartphone/social-media portraits. Conservative Mode is evidence-faithful; Paper Quality Mode may generate detail only as `GENERATED_MODEL_INFERRED` after identity, provenance, geometry, model-qualification and resource gates.

Active development is `integration/final-paper-quality-local`, created from immutable certified `main`. The architecture is **UPSTREAM-FIRST**: use official executable paper/model repositories at pinned revisions rather than reimplementing their networks. CFS owns thin adapters, source/checkpoint verification, safety, <=80% resource control, routing/fusion, Windows/offline packaging and qualification.

Current milestone: make every generated candidate traceable end-to-end to one exact official repository, immutable revision and checkpoint SHA-256, and require an exact match against the production `ModelQualification` attestation before selector/fusion/export. Broad identity-disjoint validation, final Windows/EliteBook acceptance and Target95 remain incomplete.

---

## 2. Branch and release map

| Branch | Purpose | Current HEAD | Status | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN/RELEASED | certified history | preserve |
| `integration/final-paper-quality-local` | final integration | `65d97133311fff1a062be3bc821e9c3de03ec365` | ACTIVE / PARTIAL ARTIFACT BINDING | current partial head NOT_VERIFIED; prior `8bcc801e...` run `33207031788` SUCCESS | atomically wire candidate/backend/runtime + tests |
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
- **PRODUCT_V2 — IMPLEMENTING/BENCHMARKING:** upstream Paper Quality specialists, damage routing, calibrated selection, provenance-safe fusion, 80% resource contract; no research-heavy model production-qualified.
- **PRODUCT_V3 — DESIGNING/PROTOTYPES:** personalized MAIN + 0–9 references, 13-component authority.
- **PRODUCT_V4 — DESIGNING/PROTOTYPES:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified offline Windows product + physical EliteBook acceptance; V5 holdout does not exist.

---

## 4. Holdout / benchmark lineage

`FINAL_HOLDOUT_V3`: **CONSUMED**, 39/40 historical, never rerun/tune.  
`FINAL_HOLDOUT_V4`: **CONSUMED_FAIL**, STARTED persisted, runner failed before case 1, 0/40, never rerun/tune.  
`FINAL_HOLDOUT_V5`: **NOT_CREATED**.  
Female-domain: stress/report evidence, Target95 not achieved.  
FBCNN DEV matrix: one identity, six compression profiles PASS.  
DamageMaskNet small U-Net: research hypothesis stopped.  
LR-ASPP external DEVELOPMENT: 40 identities/880 cases, aggregate pass but subgroup/domain gaps.

---

## 5. Current global objectives

OBJ-001 Preserve V1 — **PASS**.  
OBJ-002 Preserve consumed V3/V4 — **PASS**.  
OBJ-003 Canonical ledger — **IN_PROGRESS**.  
OBJ-004 Production-qualified damage mask — **IN_PROGRESS**.  
OBJ-005 Broad BFR validation — **IN_PROGRESS**.  
OBJ-006 FBCNN JPEG qualification — **IN_PROGRESS / DEV_PASS only**.  
OBJ-007 Personalized reference system — **IN_PROGRESS**.  
OBJ-008 RefFace — **BLOCKED** by mask/production/target-PC gates.  
OBJ-009 Paper Quality Windows pack — **IN_PROGRESS foundation**.  
OBJ-010 Physical EliteBook acceptance — **NOT_RUN**.  
OBJ-011 Official upstream registry — **PASS foundation**.  
OBJ-012 Exact generated-model authority — **IN_PROGRESS**; route-attestation PASS at `8bcc801e...`; artifact parser/test partial at `65d97133...`.  
OBJ-013 Future V5 protocol — **DEV VALIDATING only**.

---

## 6. Model master registry

Certified V1: YuNet, SFace, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2, constrained LaMa.

Research/upstream-first:
- FBCNN `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`, checkpoint SHA-256 `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`: **DEV_PASS/BENCHMARKING**, not production-qualified.
- GPEN `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`: BENCHMARKING/license unresolved.
- GFPGAN v1.4 `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`: BENCHMARKING.
- CodeFormer `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`: BENCHMARKING/BLOCKED_LICENSE.
- DamageMaskNet small U-Net: REJECTED/STOPPED.
- LR-ASPP: DEVELOPMENT validation only, NOT_QUALIFIED.
- RefFaceInpainting: FEASIBILITY_ONLY/BLOCKED.
- InstantRestore: FEASIBILITY_ONLY/BLOCKED_HARDWARE until proven.
- Other paper models: DISCOVERED/AUDITED/FEASIBILITY_ONLY until measured.

`IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED`.

---

## 7. Current model evidence

GPEN DEV: SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474`.  
GFPGAN1.4 DEV: SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604`.  
FBCNN QF20 DEV: PSNR `34.62->36.78`, SSIM `0.9486->0.9634`, SFace `0.9571->0.9691`, `~1.305GB`.  
FBCNN matrix run `32674085939`: 6/6 DEV profiles PASS, one identity, artifact `9502200502`, SHA `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`.  
DamageMaskNet U-Net: macro-F1 `0.173198`, macro-IoU `0.113028`, STOPPED.  
LR-ASPP external DEV: 40 identities/880 cases, F1 `0.716639`, IoU `0.579849`, domain gaps remain.

Paper-reported metrics are not CFS/Windows/EliteBook measurements unless reproduced.

---

## 8. 13-block architecture

1 IMPORT deterministic immutable MAIN.  
2 DEBLUR NAFNet mild; heavy BFR only when qualified.  
3 ENHANCE FBCNN candidate only for detected JPEG.  
4 LANDMARKS YuNet/pose.  
5 ALIGN deterministic geometry.  
6 OCCLUSION_MASK damage contract; no production-qualified multi-class model yet.  
7 REGION_SELECT 13-component personalized bank.  
8 INPAINT observed reference first; generated specialist only if qualified.  
9 FUSION MAIN > observed ref > generated; route + production attestation required.  
10 FRONTALIZE geometry-only Conservative.  
11 IDENTITY_CHECK SFace `0.363`.  
12 UPSCALE Lanczos unless measured SR qualifies.  
13 EXPORT deterministic provenance/model/resource evidence; exact candidate artifact identity is current work.

---

## 9. Photo and input contract

MAIN supports smartphone/social compression, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, missing components, partial/crop, low light and mixed unknown corruption. References are MAIN + 0–9 full/partial/component-only/different pose/expression/light/resolution/degraded/useless/wrong-person images. Full accepted same-person may global-anchor; partial is component-local; wrong-person never anchors/donates/boosts score.

---

## 10. Dataset construction

Target broad research/validation bank remains ~300–400 representative identity-disjoint sources/cases with explicit domain composition. Store source/license/date/identity/hash/resolution/split/degradation/severity/seed/mask/reference relationships. Final holdouts never train/tune.

---

## 11. Component-by-component reconstruction

LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Track MAIN visibility/damage, observed refs/confidence, generated candidates, selected source/provenance, identity/geometry and unresolved state. Observed same-person evidence outranks generation.

---

## 12. Damage routing

Fail-closed routes cover HEALTHY, GAUSSIAN_BLUR, MOTION_BLUR, DEFOCUS, JPEG_ARTIFACT, NOISE, PIXELATION, OCCLUSION, SCRIBBLE, TEXT_WATERMARK, MIXED, SMALL_FACE, PARTIAL_CROP. Missing/unverified evidence abstains; malformed/inconsistent evidence rolls back. Generated routes require complete production qualification + matching attestation. Candidate artifact identity wiring remains incomplete.

---

## 13. Decision log

**DEC-20260828-014 — route/production attestation binding — ACCEPTED/IMPLEMENTED/TEST_PASS.** Run `33207031788`: targeted 46/46, full 602/602.

**DEC-20260828-015 — candidate artifact identity continuity — ACCEPTED/IN_PROGRESS.** Candidate must expose exact `repo`, immutable revision and checkpoint SHA-256 matching production evidence. Parser `app/model_artifact_identity.py` and parser regression exist; runtime wiring pending.

Earlier decisions remain in historical archive.

---

## 14. Experiment log

**EXP-20260828-034 — route/qualification binding attempt 1/3 — PASS.** `f33880d5... -> 75b31aab... -> 8bcc801e...`, run `33207031788` SUCCESS, targeted 46/46, full 602/602, artifact `9700075188`, SHA `8362ebb4ff8f9391256d2ea87c9b7380296ae7f4f0f4d7666e0df861afae4842`.

**EXP-20260828-035 — candidate artifact identity attempt 1/3 — IN_PROGRESS / NOT_VERIFIED.** `e617550e...` adds generic identity parser; `65d97133...` adds exact/ambiguous-source tests. No wiring to candidate/FBCNN/fusion yet, so no completed gate claim.

---

## 15. Quality scoreboard

DEV evidence exists; broad validation incomplete. V3/V4 consumed and forbidden. Unified real-world/target-PC Paper Quality NOT_RUN. Safety: SFace >=`0.363`, wrong-person observed `0`, provenance violations `0`, healthy/outside MAE `<=8.0` where frozen. Target95 not achieved.

---

## 16. Target hardware

HP EliteBook 1030 G3, Windows, 16GB; exact CPU/GPU runtime-detected. <=80% logical CPU, <=80% process RAM, <=80% total-system RAM, one heavy model. CPU-first/no CUDA dependency. Linux timings are not target-PC evidence.

---

## 17. Release safety rules

Never weaken SFace `0.363`, wrong-person zero, provenance zero, frozen healthy limits; never threshold-shop, cherry-pick, relabel generated as observed, use proxy as SFace, rerun V3/V4, auto-merge, force-push certified history or fabricate evidence.

---

## 18. Provenance classes

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

---

## 19. Track A — PRODUCT_V1_1

Track A is preserved evidence. V4 is `CONSUMED_FAIL`, 0/40 final cases, never rerun. PR #2 is not a certified release. Future certification requires independent V5 after prerequisites.

---

## 20. Track B / final integration

Active engineering is the integration branch. Research branches are source/evidence lines. Use official paper code directly when executable; CFS owns thin adapters and verification. FBCNN already dynamically imports official pinned upstream code and validates checkpoint size/hash before CPU load; it remains DEV only.

---

## 21. Current Paper Quality blocker

Route -> production attestation: DONE/TEST_PASS.  
Production evidence -> unique artifact identity parser: IMPLEMENTED/PARTIAL, NOT_VERIFIED.  
Candidate/FBCNN/runtime artifact match: NOT_IMPLEMENTED yet at current head.  
Production-qualified heavy model: NONE.  
Final Windows/offline/EliteBook/Target95: BLOCKED/NOT_RUN.  
V5: NOT_CREATED.

---

## 22. Specialist model strategy

INPUT -> detect/align -> damage -> references/identity -> qualified specialist -> candidate -> hard gates -> component fusion -> final identity/provenance -> export. Use the best measured specialist per damage; observed same-person evidence always outranks generation.

---

## 23. Model selection policy

Compare multiple identity-disjoint DEV/VALIDATION cases by damage family. Identity hard gate first; then perceptual quality, geometry, artifacts, healthy preservation, PSNR/SSIM/LPIPS as useful, RAM/runtime. Never tune on final holdout. Official code is preferred but not presumed bug-free or target-compatible.

---

## 24. Historical record

Full history through 2026-08-24: `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`. Immutable historical evidence.

---

## 25. Push journal — append-only from 2026-08-28

### PUSH-20260828-001
- Branch: integration.
- Technical series: `f33880d5... -> 75b31aab... -> 8bcc801e...`.
- Result: route requires actual production `ModelQualification` and matching attestation.
- Run `33207031788` SUCCESS; targeted 46/46; full 602/602; artifact `9700075188`, SHA `8362ebb4...`.
- No model/threshold/data/holdout/V5 change.

### PUSH-20260828-002
- Previous: `8bcc801e...`; new: `e617550e368dd376bba64c7d94a3516d916032f2`.
- Added `app/model_artifact_identity.py`.
- Status: NOT_VERIFIED / not wired.

### PUSH-20260828-003
- Previous: `e617550e...`; new: `65d97133311fff1a062be3bc821e9c3de03ec365`.
- Added `tests/test_model_artifact_identity.py` proving exact identity extraction and ambiguous repository fail-closed behavior.
- Status: NOT_VERIFIED until complete wiring + same-head suite.
- No model/threshold/data/holdout/V5 change.

### PUSH-20260828-META-001
Prior ledger preserved byte-for-byte before reconciliation; meta-only.

---

## 26. Session continuity rule

Every session reads this ledger, reconciles GitHub, continues the recorded blocker and updates this file after every technical push. Never infer PASS/QUALIFIED/RELEASE_READY/Target95/EliteBook/PROJECT_FINISHED without reproducible evidence.