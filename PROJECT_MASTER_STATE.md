# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Full history through 2026-08-24 is preserved byte-for-byte at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`.

## 0. Document metadata

Last ledger update: `2026-08-28`  
Technical state verified at: `2026-08-28`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `65d97133311fff1a062be3bc821e9c3de03ec365`  
Last technical HEAD: `bc732272a91ec8ad288e415e8600bf421b1485e9`  
Last technical tree: `bec3bca903d5b236b5201c941bffe208996276de`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
Current active engineering track: final integration, upstream-first Paper Quality + immutable Conservative safety  
Current exact blocker: generated-route and exact candidate artifact identity authority are now TEST_PASS. The next blocker is **real model qualification**: FBCNN is the first specialist to advance because its official upstream code/checkpoint already have reproducible DEV evidence, but it still lacks broad identity-disjoint validation, final installed-offline Windows evidence on the integration candidate, and physical EliteBook <=80% resource evidence.  
Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No force-push certified history, no auto-merge, no consumed-holdout rerun.

---

## 1. Executive project summary

CFS is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative Mode is evidence-faithful; Paper Quality generation is always `GENERATED_MODEL_INFERRED` and subordinate to observed MAIN/same-person evidence.

Active development is `integration/final-paper-quality-local`, based on immutable certified `main`. Direction is **UPSTREAM-FIRST**: use official executable paper/model repositories at pinned revisions; CFS adds thin adapters, exact source/checkpoint verification, identity/provenance safety, <=80% resource control, routing/fusion, Windows/offline packaging and qualification.

The model-authority boundary is now materially stronger: a generated route needs a complete production `ModelQualification`; the route carries its deterministic attestation; the actual `RestorationCandidate` carries exact upstream repository/revision/checkpoint SHA; and the runtime rejects any missing or mismatched artifact identity before candidate selection/fusion. This architecture test does not qualify a model. The next milestone is broad real FBCNN qualification evidence.

---

## 2. Branch and release map

| Branch | Purpose | Current HEAD | Status | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN/RELEASED | certified history | preserve |
| `integration/final-paper-quality-local` | final integration | `bc732272a91ec8ad288e415e8600bf421b1485e9` | ACTIVE / TEST_GREEN | run `33208101745` SUCCESS; targeted 50/50; full 608/608 | FBCNN multi-identity + Windows qualification |
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
- **PRODUCT_V2 — IMPLEMENTING/BENCHMARKING:** upstream specialists, damage routing, calibrated selection, provenance-safe fusion, 80% resource contract; no research-heavy model production-qualified.
- **PRODUCT_V3 — DESIGNING/PROTOTYPES:** MAIN + 0–9 references, 13-component authority.
- **PRODUCT_V4 — DESIGNING/PROTOTYPES:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified offline Windows product + physical EliteBook acceptance; V5 holdout does not exist.

---

## 4. Holdout / benchmark lineage

FINAL_HOLDOUT_V3: **CONSUMED**, never rerun/tune.  
FINAL_HOLDOUT_V4: **CONSUMED_FAIL**, STARTED persisted, 0/40, never rerun/tune.  
FINAL_HOLDOUT_V5: **NOT_CREATED**.  
Female-domain: stress/report evidence, Target95 not achieved.  
FBCNN DEV matrix: one identity, six compression profiles PASS.  
DamageMaskNet U-Net: stopped model/data hypothesis.  
LR-ASPP external DEVELOPMENT: 40 identities/880 cases, aggregate pass with subgroup gaps.

---

## 5. Current global objectives

OBJ-001 V1 preserve — **PASS**.  
OBJ-002 V3/V4 consumed evidence preserve — **PASS**.  
OBJ-003 canonical ledger — **IN_PROGRESS**.  
OBJ-004 production-qualified damage mask — **IN_PROGRESS**.  
OBJ-005 broad BFR validation — **IN_PROGRESS**.  
OBJ-006 FBCNN qualification — **IN_PROGRESS / DEV_PASS only**.  
OBJ-007 personalized reference system — **IN_PROGRESS**.  
OBJ-008 RefFace — **BLOCKED**.  
OBJ-009 Paper Quality Windows pack — **IN_PROGRESS foundation**.  
OBJ-010 physical EliteBook — **NOT_RUN**.  
OBJ-011 upstream registry — **PASS foundation**.  
OBJ-012 exact generated-model authority — **PASS for current integration contract** at `bc732272...`; model-specific production evidence still separate.  
OBJ-013 V5 protocol — **DEV VALIDATING only**.

---

## 6. Model master registry

Certified V1: YuNet, SFace, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2, constrained LaMa.

Research/upstream-first:
- **FBCNN** `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`; checkpoint SHA-256 `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`: DEV_PASS/BENCHMARKING only; next qualification target.
- **GPEN** `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`: BENCHMARKING/license unresolved.
- **GFPGAN v1.4** `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`: BENCHMARKING.
- **CodeFormer** `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`: BENCHMARKING/BLOCKED_LICENSE.
- DamageMaskNet small U-Net: REJECTED/STOPPED.
- LR-ASPP: DEVELOPMENT validation only, NOT_QUALIFIED.
- RefFaceInpainting: FEASIBILITY_ONLY/BLOCKED.
- InstantRestore: FEASIBILITY_ONLY/BLOCKED_HARDWARE until proven.
- Others: DISCOVERED/AUDITED/FEASIBILITY_ONLY.

---

## 7. Current model evidence

GPEN DEV SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474`.  
GFPGAN1.4 DEV SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604`.  
FBCNN QF20 DEV PSNR `34.62->36.78`, SSIM `0.9486->0.9634`, SFace `0.9571->0.9691`, `~1.305GB`.  
FBCNN run `32674085939`: 6/6 DEV profiles PASS, one identity, artifact `9502200502`, archive SHA `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`.  
DamageMaskNet U-Net macro-F1 `0.173198`, macro-IoU `0.113028`, STOPPED.  
LR-ASPP external DEV F1 `0.716639`, IoU `0.579849`, domain gaps.

Paper figures are not CFS/Windows/EliteBook results unless reproduced.

---

## 8. 13-block architecture

1 IMPORT deterministic. 2 DEBLUR NAFNet / qualified BFR only. 3 ENHANCE FBCNN candidate for detected JPEG. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK no production-qualified multi-class model yet. 7 REGION_SELECT 13-component bank. 8 INPAINT observed first, qualified generated specialist second. 9 FUSION MAIN > observed ref > generated; route + qualification + exact artifact identity required. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`. 12 UPSCALE Lanczos unless measured SR qualifies. 13 EXPORT deterministic provenance/model/resource/artifact identity.

---

## 9. Photo and input contract

MAIN: smartphone/social compression, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, missing components, crop/partial, low light, mixed/unknown damage. References MAIN + 0–9 full/partial/component-only/different pose/expression/light/resolution/degraded/useless/wrong-person images. Full accepted same-person may global-anchor; partial local only; wrong-person never anchor/donor/score booster.

---

## 10. Dataset construction

Target ~300–400 identity-disjoint research/validation sources/cases with explicit domain composition; store source/license/date/identity/hash/resolution/split/degradation/severity/seed/mask/reference relationships. Final holdouts never train/tune.

---

## 11. Component-by-component reconstruction

LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Observed same-person evidence outranks generated inference.

---

## 12. Damage routing

Routes cover HEALTHY, GAUSSIAN_BLUR, MOTION_BLUR, DEFOCUS, JPEG_ARTIFACT, NOISE, PIXELATION, OCCLUSION, SCRIBBLE, TEXT_WATERMARK, MIXED, SMALL_FACE, PARTIAL_CROP. Missing/unverified evidence abstains; malformed/inconsistent evidence rolls back. Generated candidates require complete production qualification, route attestation and exact candidate repo/revision/checkpoint match before fusion.

---

## 13. Decision log

**DEC-20260828-014 route/production attestation — ACCEPTED/IMPLEMENTED/TEST_PASS.**

**DEC-20260828-015 candidate artifact identity continuity — ACCEPTED/IMPLEMENTED/TEST_PASS.** Exact repo/revision/checkpoint identity is extracted from typed production evidence and compared against each generated candidate before selector/fusion. FBCNN emits its verified upstream identity explicitly.

---

## 14. Experiment log

**EXP-20260828-034 route/qualification binding attempt 1/3 — PASS.** Run `33207031788`, targeted 46/46, full 602/602.

**EXP-20260828-035 candidate artifact identity attempt 1/3 — PASS.** Technical sequence `e617550e... -> 65d97133... -> bc732272...`; exact-head run `33208101745` SUCCESS. Targeted runner tests `50/50 PASS`; complete pytest `608/608 PASS`; success/pre-marker-failure/post-marker-failure protocol ordering PASS. Artifact `one-shot-protocol-hardening-33208101745`, ID `9700486979`, archive SHA-256 `c20dbcfb852c6c72f2099618b9e18eda905d2fb63c348d8a0f5769e4759ba8bc`. This proves the contract, not model quality/qualification.

---

## 15. Quality scoreboard

DEV evidence exists; broad validation incomplete. V3/V4 consumed/forbidden. Target-PC Paper Quality NOT_RUN. SFace `0.363`, wrong-person observed `0`, provenance `0`, healthy/outside MAE `<=8.0` where frozen. Target95 not achieved.

---

## 16. Target hardware

HP EliteBook 1030 G3, Windows, 16GB; exact CPU/GPU runtime detected. <=80% logical CPU, <=80% process/system RAM, one heavy model, CPU-first/no CUDA. Linux timings are not target-PC evidence.

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

Active integration uses official paper code directly where executable. FBCNN is a thin pinned-upstream adapter; no architecture copy. The integration contract now binds generated output to exact qualified source artifacts. No research model is production-qualified yet.

---

## 21. Current Paper Quality blocker

Artifact-identity architecture is green. Next: advance **FBCNN real qualification** using official upstream bytes. Required evidence remains broad identity-disjoint JPEG/double-JPEG/social/smartphone recompression validation, licenses/redistribution confirmation, same-candidate installed-offline Windows run, and physical HP EliteBook <=80% resource measurement. Do not fabricate the physical-PC gate. V5 remains NOT_CREATED.

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

### PUSH-20260828-META-001
Prior ledger preserved byte-for-byte before reconciliation; meta-only.

---

## 26. Session continuity rule

Every session reads this ledger, reconciles GitHub, continues the current blocker and updates this file after every technical push. Never infer PASS/QUALIFIED/RELEASE_READY/Target95/EliteBook/PROJECT_FINISHED without exact evidence.