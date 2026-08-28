# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. The full prior ledger through 2026-08-24 is preserved byte-for-byte at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, historical blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`.

## 0. Document metadata

Last ledger update: `2026-08-28`  
Technical state verified at: `2026-08-28`  
Repository: `xhinoo97-svg/ConservativeFaceStudio`  
Canonical state branch: `meta/project-state`  
Last technical branch: `integration/final-paper-quality-local`  
Previous technical HEAD: `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a`  
Last technical HEAD: `e617550e368dd376bba64c7d94a3516d916032f2`  
Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`  
Current active engineering track: final integration, upstream-first Paper Quality + immutable Conservative safety  
Current exact blocker: candidate artifact-identity verifier has been added but is not yet wired into `RestorationCandidate`, official FBCNN output and the Paper Quality fusion boundary; this partial technical push is **NOT_VERIFIED** until the complete binding and same-HEAD tests pass.  
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

Conservative Face Studio is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative Mode preserves observed identity/evidence with explicit provenance; Paper Quality Mode may generate unsupported detail only as `GENERATED_MODEL_INFERRED` and only after model, identity, geometry, provenance and resource gates.

The active engineering line is `integration/final-paper-quality-local`, created from immutable certified `main`. The architecture is **UPSTREAM-FIRST**: official executable paper/model repositories are reused at pinned revisions rather than reimplemented by CFS. CFS owns thin adapters, checkpoint/revision verification, safety/provenance, routing/fusion, <=80% resource control, Windows/offline packaging and qualification.

Current milestone: complete end-to-end artifact identity continuity so generated pixels are traceable from `RestorationCandidate` -> exact official repository/revision/checkpoint -> production `ModelQualification` attestation -> authorized damage route -> fusion/export. Broad multi-identity validation, final Windows/EliteBook acceptance and Target95 remain incomplete.

---

## 2. Branch and release map

| Branch | Purpose | Current HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | certified history | preserve |
| `integration/final-paper-quality-local` | single final integration | `e617550e368dd376bba64c7d94a3516d916032f2` | ACTIVE / IMPLEMENTING | latest partial push NOT_VERIFIED; prior `8bcc801e...` run `33207031788` SUCCESS | finish artifact binding, test same HEAD |
| `hotfix/real-world-restoration-v1.1` | Track A evidence | `77687b3b171f4e9989fcf486834f2d8b7a52f591` | V4 CONSUMED_FAIL / FROZEN | PR #2 OPEN/DRAFT/NO-GO | never rerun V4 |
| `protocol/v5-certification-hardening` | future protocol DEV | `268188c5a2540455ff804383cb583b16546b62f1` | ARCHIVED DEV | synthetic PASS | no V5 until prerequisites |
| `research/paper-quality-local-v2` | advanced research evidence | `6d57725aae087bb4a3144d521d91346999f9a4fd` | SUPERSEDED AS ACTIVE ARCHITECTURE | preserve evidence | port only measured winners |
| `research/face-restoration-v2` | early research | `757a3f60e2f012a1d0b1758c7280bfdd492f33df` | SUPERSEDED / ARCHIVED | historical | preserve |
| `feature/block-pipeline-v1` | V1 history | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | ARCHIVED | historical | preserve |
| `release/v1-certified` | V1 release history | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | ARCHIVED | merged history | preserve |
| `meta/project-state` | canonical ledger | self-SHA omitted | ACTIVE META | docs only | update after every technical push |

---

## 3. Product version roadmap

**PRODUCT_V1 — RELEASED.** Certified conservative baseline; SFace `0.363`, wrong-person observed pixels `0`, provenance violations `0`.

**PRODUCT_V1_1 — FAILED AS V4 CERTIFICATION CANDIDATE / PRESERVED.** Track A candidate passed prerequisites, but V4 was consumed then failed before case 1 because of runner interface error. No rerun allowed.

**PRODUCT_V2 — IMPLEMENTING/BENCHMARKING.** Paper Quality Local: upstream models, damage routing, candidate selection, provenance-safe fusion, 80% resource contract. No research-heavy model is production-qualified yet.

**PRODUCT_V3 — DESIGNING/PROTOTYPES.** MAIN + 0–9 same-person references; 13-component authority; reference-first repair.

**PRODUCT_V4 — DESIGNING/PROTOTYPES.** Damage-specialist hybrid routing and component-aware generated fusion.

**PRODUCT_V5 — PLANNED.** Unified product + offline Windows model pack + clean installer + physical EliteBook acceptance. V5 holdout does not exist and is not authorized.

---

## 4. Holdout / benchmark lineage

- `FINAL_HOLDOUT_V3`: **CONSUMED**, 39/40 historical; mosaic SFace `0.360 < 0.363`; never rerun/tune.
- `FINAL_HOLDOUT_V4`: **CONSUMED_FAIL**, STARTED marker persisted, runner failed before case 1, 0/40; never rerun/tune.
- `FINAL_HOLDOUT_V5`: **NOT_CREATED**.
- Female-domain: stress/report evidence only unless separately frozen; historical Target95 materially below target.
- FBCNN matrix: DEVELOPMENT, one public identity, six compression profiles PASS.
- DamageMaskNet mixed-source: research TRAIN/VALIDATION; small U-Net hypothesis stopped.
- LR-ASPP external validation: DEVELOPMENT, 40 identities/880 cases; aggregate pass but subgroup/domain gaps.

Consumed holdouts are immutable evidence and forbidden for tuning.

---

## 5. Current global objectives

OBJ-001 Preserve V1 — **PASS**.  
OBJ-002 Preserve V3/V4 consumed evidence — **PASS**.  
OBJ-003 Canonical ledger — **IN_PROGRESS**.  
OBJ-004 Qualified multi-class damage mask — **IN_PROGRESS**; small U-Net failed, LR-ASPP not production-qualified.  
OBJ-005 Broad BFR selection — **IN_PROGRESS**, upstream-first.  
OBJ-006 FBCNN JPEG qualification — **IN_PROGRESS**, DEV_PASS only.  
OBJ-007 Personalized reference system — **IN_PROGRESS**.  
OBJ-008 RefFace specialist — **BLOCKED** by mask/production/target-PC gates.  
OBJ-009 Paper Quality Windows pack — **IN_PROGRESS foundation / NOT_READY**.  
OBJ-010 Physical EliteBook acceptance — **PROPOSED / NOT_RUN**.  
OBJ-011 Official upstream registry — **PASS as foundation**.  
OBJ-012 Exact generated-model authority — **IN_PROGRESS**; route-attestation binding PASS at `8bcc801e...`, candidate artifact binding partial at `e617550e...`.  
OBJ-013 Future one-shot V5 protocol — **VALIDATING DEV only**.

---

## 6. Model master registry

Certified V1: YuNet, SFace, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2, constrained LaMa.

Research/upstream-first:
- **FBCNN** — `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`; `fbcnn_color.pth`, SHA-256 `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`; DEVELOPMENT leader for JPEG; not production-qualified.
- **GPEN BFR-512** — `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`; BENCHMARKING / licensing unresolved.
- **GFPGAN v1.4** — `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`; BENCHMARKING.
- **CodeFormer** — `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`; CPU DEV evidence, production license blocker.
- **DamageMaskNet small U-Net** — REJECTED/STOPPED for mask quality.
- **LR-ASPP** — DEVELOPMENT validation evidence, not production-qualified.
- **RefFaceInpainting** — FEASIBILITY_ONLY/BLOCKED; CPU vertical slice not authorized yet.
- **InstantRestore** — pinned upstream research; CPU/EliteBook/hardware/license production evidence incomplete.
- OSDFace/RestoreFormer++/VQFR/FaceMe/others — DISCOVERED/AUDITED/FEASIBILITY_ONLY until measured.

No model is called QUALIFIED merely because code/checkpoint exists.

---

## 7. Current model evidence

Linux CPU DEVELOPMENT historical evidence only:
- GPEN: SFace `0.95397`, `~2.697s`, peak RSS `~1.828GB`, PSNR `28.07`, SSIM `0.7474` on one DEV case.
- GFPGAN1.4: SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604`.
- FBCNN QF20: PSNR `34.62 -> 36.78`, SSIM `0.9486 -> 0.9634`, SFace `0.9571 -> 0.9691`, peak RSS `~1.305GB`.
- FBCNN run `32674085939`: 6/6 DEVELOPMENT compression profiles PASS, one identity only, artifact `9502200502`, archive SHA-256 `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`.
- DamageMaskNet small U-Net: macro-F1 `0.173198`, macro-IoU `0.113028`; STOPPED.
- LR-ASPP external DEVELOPMENT: 40 identities/880 cases, F1 `0.716639`, IoU `0.579849`, but subgroup/min-class gaps remain.

Paper metrics are references, never substituted for CFS/Windows/EliteBook measurements.

---

## 8. 13-block architecture

1 IMPORT: deterministic immutable MAIN.  
2 DEBLUR: NAFNet mild; heavy BFR only through qualified routing.  
3 ENHANCE: FBCNN candidate for detected JPEG; no generic beauty pass.  
4 LANDMARKS: YuNet/pose evidence.  
5 ALIGN: deterministic similarity/affine/RANSAC.  
6 OCCLUSION_MASK: taxonomy/runtime contract; no production-qualified multi-class mask model yet.  
7 REGION_SELECT: 13-component personalized reference bank.  
8 INPAINT: observed reference first; generated specialist only if qualified.  
9 FUSION: MAIN > observed same-person reference > generated; calibrated + route/model authority required.  
10 FRONTALIZE: geometry-only Conservative.  
11 IDENTITY_CHECK: SFace `0.363`, direct/non-transitive/fail-closed.  
12 UPSCALE: Lanczos; research SR only after qualification.  
13 EXPORT: deterministic image/provenance/model/resource evidence; exact candidate artifact identity is current work.

---

## 9. Photo and input contract

MAIN: low-res smartphone/social images, JPEG/double-JPEG, defocus/motion/mixed blur, noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, partially/fully covered components, missing component, crop/partial face, low light, mixed/unknown corruption.

References: MAIN + 0–9 full/partial/component-only/different-pose/expression/light/resolution/degraded/useless/wrong-person images. Full accepted same-person may global-anchor; partial accepted references are component-local; wrong-person is never anchor/donor/score booster.

---

## 10. Dataset construction

Target broad bank: approximately 300–400 representative, identity-disjoint TRAIN/DEV/VALIDATION/FINAL_HOLDOUT sources/cases, with explicit domain/female representation rather than silent bias. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/mask/reference relationships. Final holdouts never train/tune.

---

## 11. Component-by-component reconstruction

Canonical 13: LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Track MAIN visibility/damage, observed references/confidence, generated candidates, selected source, provenance, identity/geometry and unresolved state. Observed same-person evidence outranks generation.

---

## 12. Damage routing

Fail-closed routes cover HEALTHY, GAUSSIAN_BLUR, MOTION_BLUR, DEFOCUS, JPEG_ARTIFACT, NOISE, PIXELATION, OCCLUSION, SCRIBBLE, TEXT_WATERMARK, MIXED, SMALL_FACE, PARTIAL_CROP. Missing/unverified evidence abstains; malformed/inconsistent evidence rolls back. Generated routes require a complete production qualification and matching route attestation. Candidate artifact-byte identity is the current unfinished step.

---

## 13. Decision log

**DEC-20260828-014 — Production attestation at fusion boundary — ACCEPTED/IMPLEMENTED/TEST_PASS.** A route boolean/model key is not model authority. Exact `ModelQualification` and matching deterministic attestation are required. Evidence: run `33207031788`, targeted 46/46, full 602/602.

**DEC-20260828-015 — Candidate artifact identity continuity — ACCEPTED/IN_PROGRESS.** Generated candidates must expose exact official repository, immutable revision and checkpoint SHA-256, and these must match production evidence before fusion/export. `app/model_artifact_identity.py` introduced at `e617550e...`; complete wiring/tests pending.

Earlier decisions remain in the archived ledger.

---

## 14. Experiment log

**EXP-20260828-034 — route/qualification attestation binding, attempt 1/3 — PASS.** Technical series `f33880d5... -> 75b31aab... -> 8bcc801e...`; run `33207031788` SUCCESS; targeted `46/46`, full `602/602`; artifact ID `9700075188`, SHA-256 `8362ebb4ff8f9391256d2ea87c9b7380296ae7f4f0f4d7666e0df861afae4842`. No real model/holdout qualification.

**EXP-20260828-035 — candidate artifact identity continuity, attempt 1/3 — IN_PROGRESS / NOT_VERIFIED.** `e617550e...` adds a generic production-evidence parser/verifier for one `repo:`, one `commit:` and one `checkpoint-sha256:` identity. It is not yet wired into candidates/runtime and therefore cannot be treated as a completed gate.

---

## 15. Quality scoreboard

DEV evidence exists; broad heavy-model validation is incomplete. V3/V4 are consumed and forbidden for tuning. Real-world unified product and target-PC Paper Quality remain NOT_VERIFIED/NOT_RUN. Safety targets remain SFace >=`0.363`, wrong-person observed pixels `0`, provenance violations `0`, frozen healthy/outside MAE `<=8.0` where applicable. Target95 is not achieved.

---

## 16. Target hardware

HP EliteBook 1030 G3, Windows, 16GB RAM; exact CPU/GPU runtime-detected. <=80% logical CPU, <=80% process RAM, <=80% total-system RAM, one heavy model resident. CPU-first/no CUDA dependency. OpenVINO/iGPU only after support and parity evidence. Linux timings are not EliteBook evidence.

---

## 17. Release safety rules

Never lower SFace `0.363`, allow wrong-person observed pixels, allow provenance violations, threshold-shop, cherry-pick, remove hard failures, relabel generated pixels as observed, use proxy similarity as SFace, rerun V3/V4, auto-merge, force-push certified history or fabricate evidence.

---

## 18. Provenance classes

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`. Generated pixels never become observed evidence because identity looks correct.

---

## 19. Track A — PRODUCT_V1_1

Track A is preserved evidence only. V4 is `CONSUMED_FAIL`, 0/40 final cases, no rerun. PR #2 remains non-certified. Any future final certification requires a new independent V5 lineage after all product prerequisites are green.

---

## 20. Track B / final integration — Paper Quality

Active engineering is `integration/final-paper-quality-local`. Research branches remain read-only source/evidence lines. Official executable repositories are the implementation baseline; CFS patches only integration/compatibility defects and adds safety/resource/packaging boundaries. FBCNN already follows this pattern by dynamically importing the exact official upstream network from a pinned checkout and verifying its approved checkpoint before CPU load.

---

## 21. Current Paper Quality blocker

1. Route -> production qualification attestation binding: **DONE / TEST_PASS** at `8bcc801e...`.
2. Candidate -> exact official repo/revision/checkpoint -> qualification binding: **IN_PROGRESS / NOT_VERIFIED** at `e617550e...`.
3. Production-qualified heavy model: **NONE**.
4. Broad identity-disjoint model validation, final Windows/offline model pack, physical EliteBook and Target95: **BLOCKED/NOT_RUN**.
5. V5: **NOT_CREATED/NOT_AUTHORIZED**.

---

## 22. Specialist model strategy

INPUT -> detect/align -> damage evidence -> reference analysis -> identity anchors -> qualified specialist route -> candidates -> hard identity/geometry/quality gates -> evidence-aware component fusion -> final identity/provenance -> export. Use the best measured specialist per damage, not every model. Observed same-person evidence remains first authority.

---

## 23. Model selection policy

Select model winners only across multiple identity-disjoint DEV/VALIDATION cases by damage family. Identity is a hard gate; then perceptual quality, geometry, artifacts, healthy preservation, PSNR/SSIM/LPIPS where appropriate, RAM/runtime. Never select/tune on final holdout. Official code is preferred but not assumed correct or target-PC compatible.

---

## 24. Historical record

Full append-only history through 2026-08-24 is preserved at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`, blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`. Do not rewrite it.

---

## 25. Push journal — append-only from reconciliation

### PUSH-20260828-001
- TECHNICAL BRANCH: `integration/final-paper-quality-local`.
- PREVIOUS HEAD: `f5ca07e0b5268ec2b8843f9dce93b5d6a9fdf5cd`.
- COMMITS: `f33880d5...`, `75b31aab...`, `8bcc801e...`.
- NEW HEAD/TREE: `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a` / `f3984f16751d648b7a9ffbd726c64ac50f3ad21b`.
- FILES: `app/damage_router.py`, `app/paper_quality_runtime.py`, `tests/test_paper_quality_runtime.py`.
- RESULT: exact production qualification attestation required before generated fusion.
- TESTS: run `33207031788` SUCCESS; targeted `46/46`; full `602/602`; protocol ordering PASS.
- ARTIFACT: `9700075188`, SHA-256 `8362ebb4ff8f9391256d2ea87c9b7380296ae7f4f0f4d7666e0df861afae4842`.
- MODEL/HOLDOUT EFFECT: none; no model promoted, no threshold/data/holdout/V5 change.

### PUSH-20260828-002
- TECHNICAL BRANCH: `integration/final-paper-quality-local`.
- PREVIOUS HEAD: `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a`.
- NEW HEAD: `e617550e368dd376bba64c7d94a3516d916032f2`.
- FILE ADDED: `app/model_artifact_identity.py`.
- PURPOSE: parse and fail-closed validate the unique `repo:`, `commit:` and `checkpoint-sha256:` identity from a production qualification.
- TEST/WORKFLOW STATE: **NOT_VERIFIED** at this ledger update; module is not yet wired into generated candidates or fusion.
- MODEL/HOLDOUT EFFECT: none.
- NEXT ACTION: wire explicit artifact identity into `RestorationCandidate`, FBCNN output and Paper Quality runtime; add mismatch/missing regressions; run same-HEAD targeted + full suite; update ledger again before further technical work.

### PUSH-20260828-META-001
Prior canonical ledger preserved byte-for-byte at the historical archive path before reconciliation. Meta-only; no product code/model/threshold/data/holdout changed.

---

## 26. Session continuity rule

Every session reads this ledger, reconciles current GitHub HEAD/workflow/artifacts, continues the recorded blocker, and updates this ledger after every technical push. Never infer PASS/QUALIFIED/RELEASE_READY/Target95/EliteBook/PROJECT_FINISHED without reproducible evidence.