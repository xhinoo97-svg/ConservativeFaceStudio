# Conservative Face Studio — Project Master State

> CANONICAL PROJECT LEDGER. Every new engineering/ChatGPT/Codex session must read this file before making project decisions.
>
> This is a living construction document: CURRENT STATE is editable, while important failures, decisions, experiments and technical pushes are preserved in the append-only historical sections. `IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED`.

## 0. Document metadata

- **Last ledger update:** 2026-08-19T05:46+02:00 (Europe/Rome)
- **Technical state verified at:** 2026-08-19
- **Repository:** `xhinoo97-svg/ConservativeFaceStudio`
- **Canonical state branch:** `meta/project-state`
- **Last technical branch:** `hotfix/real-world-restoration-v1.1`
- **Previous technical HEAD:** `3645c8c39653d04616167e881adaf28d2b93cd45`
- **Last technical HEAD:** `3e919f7a1cc54e1bdb00607c2bbeece1d3392724`
- **Certified base:** `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- **Current active engineering tracks:** Track A — PRODUCT_V1_1 operational stabilization; Track B — Paper Quality research
- **Current exact blockers:**
  1. **Track A:** a direct-MAIN component-transfer fix has been pushed at `3e919f7a...`; its same-HEAD CI result is **NOT_VERIFIED**. The immediately preceding head `3645c8c...` had Release Quality, Windows and Female-domain all FAIL.
  2. **Track B:** DamageMaskNet mixed-source U-Net attempt 3 was already triggered, but its final evidence remains unavailable through the current Actions interface. Do not launch attempt 4 merely to recover observability.
- **Overall project status:** PARTIAL

| Global gate | State | Scope / evidence |
|---|---|---|
| FORENSIC_MODE_READY | TRUE | Certified PRODUCT_V1 only. This does not certify V1.1. |
| PAPER_QUALITY_MODE_READY | FALSE | Real DEV evidence exists, but broad validation/Windows/target-PC gates are incomplete. |
| WINDOWS_INSTALLER_READY | PARTIAL | Historical PRODUCT_V1 package certification exists; V1.1 new-head and Paper Quality package state are not verified. |
| TARGET_HARDWARE_READY | FALSE | No real HP EliteBook 1030 G3 Paper Quality acceptance. |
| QUALITY_TARGET_ACHIEVED | FALSE | No broad identity-disjoint evidence establishes the final quality target. |
| PROJECT_FINISHED | FALSE | Unified product acceptance is incomplete. |

### Mandatory ledger update protocol

For every active technical branch:

`technical work -> tests/evidence -> commit -> push -> read exact remote HEAD -> update PROJECT_MASTER_STATE.md -> commit/push ledger`

The ledger records the technical commit SHA, never its own current commit SHA. Do not force-push certified history and do not auto-merge.

---

## 1. Executive project summary

Conservative Face Studio (CFS) is a local Windows face-restoration application for difficult real photographs: low-quality smartphone/social-media images, blur, JPEG damage, pixelation/mosaic, scribbles/stickers/black bars, opaque loss, crop/partial faces and multi-reference restoration.

CFS deliberately separates two authorities:

- **Conservative / forensic mode:** original MAIN and verified same-person observed reference evidence have priority; provenance is explicit; wrong-person contribution is zero; inadequate evidence may remain unresolved/abstain.
- **Paper Quality mode:** learned facial priors may synthesize missing information, but generated content is always `GENERATED_MODEL_INFERRED` and can never be relabeled as observed evidence.

PRODUCT_V1 is the immutable certified baseline. PRODUCT_V1_1 is an operational/safety hotfix and must be repaired without importing experimental Paper Quality generators. Track B is a real ML research program with real Linux CPU development evidence for GPEN, GFPGAN v1.4, CodeFormer and FBCNN, plus implemented research infrastructure for resource control, damage routing, personalized references, reference-first repair, hard-gated candidate selection and deterministic component fusion.

Current Track A work isolates **ranking membership** from **identity/component authority**: a single-link SFace component may rank references, but only a direct MAIN↔REF SFace edge at the frozen threshold, or a separate strict face-local same-canvas proof, may authorize an observed reference. Track B remains blocked on DamageMaskNet attempt-3 evidence recovery.

---

## 2. Branch and release map

| Branch | Purpose | Base | Current verified HEAD | Status | Last meaningful change | CI state | Merge status | Superseded by | Next gate |
|---|---|---|---|---|---|---|---|---|---|
| `main` | Certified PRODUCT_V1 | historical | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | Signed PR #1 merge; commit records Windows #1195, Female #463, Release Quality #13 | historical certified green | MERGED | none | Preserve; never rewrite. |
| `feature/block-pipeline-v1` | Original V1 implementation | pre-V1 | `5eff667373cd47c07ba14aaad2acafee6d5a61c1` | MERGED / SUPERSEDED | Original block pipeline | historical | merged into main | `main` | archive only |
| `release/v1-certified` | Certified V1 candidate | V1 feature | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | FROZEN / ARCHIVED | Same-SHA candidate certification | historical certified green | merged via PR #1 | `main` | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A operational/safety hotfix | PRODUCT_V1 | `3e919f7a1cc54e1bdb00607c2bbeece1d3392724` | VALIDATING / ACTIVE | Persist direct MAIN component-transfer authority; same-canvas evidence-key compatibility | **NOT_VERIFIED on this new HEAD**; previous `3645c8c...` had Release Quality/Windows/Female FAIL | PR #2 OPEN, DRAFT, NOT MERGED | none | targeted tests, full pytest, then same-HEAD Windows/Female/Release Quality |
| `research/face-restoration-v2` | Early degradation/dataset research | `main` | `757a3f6081b7b152cdc615a07cd99aec40fa0a1c` | SUPERSEDED AS ACTIVE ARCHITECTURE | Early degradation generator/dataset spec | NOT_VERIFIED | not merged | `research/paper-quality-local-v2` as active direction, but not a literal Git superset | preserve/port useful assets explicitly |
| `research/paper-quality-local-v2` | Advanced Paper Quality research | `main` | `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd` | ACTIVE / BENCHMARKING | BFR/JPEG evidence, 80% governor, DamageMaskNet/reference/fusion/RefFace preparation | latest research push state partly NOT_VERIFIED | not merged | none | recover DamageMaskNet attempt 3 evidence |
| `meta/project-state` | Canonical project ledger | certified `main` | intentionally not self-recorded | ACTIVE META | canonical state archive | documentation-only | not a product merge branch | none | update after every technical push |

### Research-branch reconciliation

`research/face-restoration-v2` and `research/paper-quality-local-v2` diverged from the same certified base. The advanced branch does **not** literally contain the two early research commits. Therefore the early branch is only superseded as the active architecture; its data/degradation assets remain historical material for explicit review/porting.

---

## 3. PRODUCT VERSION ROADMAP

Product versions and evaluation/holdout versions are separate namespaces.

### PRODUCT_V1 — Certified Conservative Baseline

- **State:** RELEASED
- **Objective/user purpose:** evidence-first local conservative restoration with auditability.
- **Base:** `main@2767513f...`
- **Architecture:** certified 13-block pipeline.
- **Models:** YuNet, SFace, NAFNet, face-parsing ResNet18 ONNX, head-pose MobileNetV2 ONNX, constrained LaMa and deterministic infrastructure.
- **Supported domain:** baseline blur/occlusion/reference restoration according to the frozen V1 contract.
- **Safety:** SFace `0.363`; wrong-person observed pixels `0`; provenance violations `0`; healthy-region policy preserved.
- **Windows:** historical PASS/certified.
- **EliteBook-specific acceptance:** NOT_VERIFIED.
- **Known limitation:** severe missing information cannot achieve modern generative perceptual quality without leaving the forensic evidence domain.
- **Acceptance evidence:** signed merge and historical certification runs.

### PRODUCT_V1_1 — Operational Real-World Hotfix

- **State:** VALIDATING
- **Branch/head:** `hotfix/real-world-restoration-v1.1@3e919f7a...`
- **Objective:** real-world reliability without changing V1 safety philosophy.
- **Architecture/models:** same production family; no Track B generators.
- **Blocks affected:** preflight/identity, observed reference eligibility, same-canvas bridge, repair/fusion safety, release protocol.
- **Current work:** ranking clusters remain ranking-only; direct MAIN↔REF SFace authority is now persisted separately for component transfer; strict face-local same-canvas remains a separate override.
- **Safety:** threshold remains `0.363`; no transitive A-B-C identity authority; wrong-person `0`; provenance `0`; V3 never rerun; V4 not executed early.
- **Windows/Female/Release Quality:** new-head state NOT_VERIFIED; previous exact head failed all three.
- **Next gate:** narrow regressions -> full pytest -> same-head Windows/Female/Release Quality -> candidate protocol.

### PRODUCT_V2 — Paper Quality Local

- **State:** BENCHMARKING
- **Branch:** `research/paper-quality-local-v2`
- **Objective:** damage-aware local CPU restoration with modern priors and explicit generated provenance.
- **Models under evidence:** GPEN BFR-512, GFPGAN v1.4, CodeFormer, FBCNN, NAFNet; DamageMaskNet under development.
- **Architecture:** common candidate adapter; specialist routing; hard identity gates; deterministic evidence-aware fusion; 80% resource governor.
- **Safety:** generated content always `GENERATED_MODEL_INFERRED`; SFace hard gate unchanged.
- **Resource target:** <=80% logical CPU, <=80% process/system RAM, one heavy model resident.
- **Windows/EliteBook:** NOT_RUN/NOT_READY for Paper Quality release.
- **Current blocker:** DamageMaskNet attempt-3 result recovery.

### PRODUCT_V3 — Personalized Multi-Reference Restoration

- **State:** PLANNED with enabling prototypes
- **Objective:** MAIN + 0–9 same-person references used per component, not one global best donor.
- **Architecture:** local `PersonIdentityProfile`, robust full-reference consensus, component coverage/quality, partial-reference local authority.
- **Blocks:** 7, 8, 9, 11, 13.
- **Safety:** full accepted reference may be global anchor; partial accepted reference is component-local; wrong-person is never anchor/donor/identity booster.
- **Next gate:** identity-disjoint 0/1/9-reference validation with exact provenance.

### PRODUCT_V4 — Damage-Specialist Hybrid Architecture

- **State:** PLANNED with enabling prototypes
- **Objective:** detect corruption family, choose the best specialist, score/fuse at component level.
- **Candidates:** FBCNN, NAFNet, qualified BFR, RefFaceInpainting and future measured specialists.
- **Blocks:** 2, 3, 6, 7, 8, 9, 11, 12.
- **Current enablers:** DamageMaskNet pipeline, specialist routing framework, component fusion.
- **Blocker:** segmentation/specialist qualification incomplete.

### PRODUCT_V5 — Unified Final Product

- **State:** PLANNED
- **Objective:** stable Conservative + Paper Quality + personalized multi-reference + specialist routing + offline Windows model pack + installer + target-PC acceptance.
- **Acceptance:** no known release defect, clean Windows/offline package, model hashes/licenses, real EliteBook evidence, all safety/quality gates.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

| Evaluation set | Purpose | Cases/identities | Source/legal | Split/freeze | Executed | Consumed | Tuning? | State/result |
|---|---|---|---|---|---|---|---|---|
| CALIBRATION_V1 | V1 safety calibration | historical 60 cases | frozen manifests | calibration | YES | certification evidence | only under original protocol | historical `60/60` at certified candidate |
| FINAL_HOLDOUT_V1 | PRODUCT_V1 certification | historical 40 | frozen | final | YES | YES | NO | historical `40/40` |
| FINAL_HOLDOUT_V2 | historical lineage | exact details not fully re-reconciled in this ledger pass | NOT_VERIFIED | historical | NOT_VERIFIED | NOT_VERIFIED | NO unless protocol proves otherwise | recover before claiming |
| FINAL_HOLDOUT_V3 | V1.1 historical final evaluation | 40 | frozen | final | YES | **YES** | **NO** | `39/40`; `cfsfs3-fin-020-medium_block_mosaic`; SFace `0.360 < 0.363` |
| FINAL_HOLDOUT_V4 | independent V1.1 final holdout | 40 / 20 identities; 19 female-domain + 1 control | ControlFace10K CC BY 4.0, pinned | frozen before candidate modification | **NO** | **NO** | **NO** | freeze exists; no `CONSUMED.json`; no certification request; SFace `0.363`, MAE `8.0`, wrong-person `0` |
| FINAL_HOLDOUT_V5 | future | not created | not created | future | NO | NO | NO | PLANNED |
| Female-domain | real-domain stress/safety | quick profile targets ~300–400 cases | curated/project sources | validation/stress | multiple runs | not a final holdout | quality report-only; safety hard gates | previous hotfix head: benchmark completed then report validation FAIL; exact nested reason NOT_VERIFIED |
| Paper Quality DEV | model comparison | expansion required | research sources | DEV | partial | NO | YES | real BFR/FBCNN evidence exists, insufficient for production winner |
| Paper Quality VALIDATION | independent model/router validation | expansion required | licensed research banks | VALIDATION | incomplete | NO | freezes selector after validation | incomplete |
| DamageMaskNet bank | exact synthetic masks | FairFace + ControlFace mixed bank; identity-disjoint ControlFace validation | source/license recorded | TRAIN/VALIDATION | attempt 3 triggered | NO | TRAIN/DEV only | result NOT_VERIFIED |

**Invariant:** consumed holdouts are never tuned/rerun. HOLDOUT_V3 stays consumed. HOLDOUT_V4 remains untouched until the correct one-shot sequence.

---

## 5. CURRENT GLOBAL OBJECTIVES

### OBJ-001 — Preserve certified PRODUCT_V1
- **VERSION/TRACK:** V1 / baseline
- **STATUS:** PASS
- **WHY:** immutable regression reference.
- **SUCCESS:** no history rewrite/force push.
- **EVIDENCE:** signed `main@2767513f...` merge and historical certification.
- **BLOCKER:** none.
- **NEXT:** preserve.
- **LAST UPDATED:** 2026-08-19.

### OBJ-002 — Restore PRODUCT_V1_1 operational gates
- **VERSION/TRACK:** V1.1 / A
- **STATUS:** VALIDATING
- **WHY:** real cases must work without weakening safety.
- **SUCCESS:** targeted tests + full pytest + same-HEAD Windows/Female/Release Quality, wrong-person/provenance zero.
- **CURRENT EVIDENCE:** previous `3645c8c...` Release Quality targeted suite `4 failed, 195 passed`; patch `3e919f7a...` now persists direct MAIN component-transfer evidence and same-canvas alias compatibility.
- **CURRENT BLOCKER:** new-head test/workflow outcome NOT_VERIFIED.
- **DEPENDENCIES:** preflight SFace matrix, V2 firewall, V4 direct-edge hardening, face-local same-canvas proof.
- **NEXT:** evaluate new-head targeted/full suite; fix only remaining root cause.
- **LAST UPDATED:** 2026-08-19T05:46+02:00.

### OBJ-003 — Maintain canonical project ledger
- **VERSION/TRACK:** all / meta
- **STATUS:** IN_PROGRESS
- **WHY:** remove dependence on conversation memory.
- **SUCCESS:** every technical push immediately recorded with exact SHA/evidence.
- **CURRENT EVIDENCE:** this update records `3645c8c... -> 3e919f7a...`.
- **NEXT:** update again after the next technical push.

### OBJ-004 — Resolve DamageMaskNet hypothesis
- **VERSION/TRACK:** V2/V4 / B
- **STATUS:** BLOCKED
- **SUCCESS:** recover attempt 3; PASS requires per-class IoU/F1, ONNX parity, RAM/runtime; true quality failure exhausts U-Net hypothesis.
- **EVIDENCE:** attempt 1 HTTP403, attempt 2 HTTP429, attempt 3 mixed-source triggered.
- **BLOCKER:** attempt-3 evidence not observable through current interface.
- **NEXT:** recover existing evidence without rerun.

### OBJ-005 — Select blind BFR candidates using broad evidence
- **VERSION/TRACK:** V2 / B
- **STATUS:** IN_PROGRESS
- **SUCCESS:** identity-disjoint DEV/VALIDATION comparison by identity, perceptual quality, geometry, artifacts, healthy preservation, runtime/RAM.
- **EVIDENCE:** real one-case comparable GPEN/GFPGAN and CodeFormer slice.
- **BLOCKER:** insufficient breadth; licensing for some candidates.
- **NEXT:** multi-image benchmark after current blocker.

### OBJ-006 — Qualify JPEG specialist
- **VERSION/TRACK:** V2/V4 / B
- **STATUS:** IN_PROGRESS
- **SUCCESS:** JPEG QF ranges, double JPEG, social-media recompression, resize+JPEG, JPEG+blur, Windows/EliteBook.
- **EVIDENCE:** FBCNN QF20 improves PSNR/SSIM/SFace.
- **NEXT:** broaden validation.

### OBJ-007 — Validate Personalized Reference Bank
- **VERSION/TRACK:** V3 / B
- **STATUS:** IN_PROGRESS
- **SUCCESS:** 0/1/9, full/partial/wrong/duplicate/low-quality/multi-pose tests; per-component improvement; exact provenance.
- **EVIDENCE:** framework implemented; latest workflow state NOT_VERIFIED.
- **NEXT:** broad validation after core damage gate.

### OBJ-008 — Measure RefFaceInpainting CPU feasibility
- **VERSION/TRACK:** V3/V4 / B
- **STATUS:** BLOCKED
- **SUCCESS:** same-person inpainting under 80%, SFace gate, exact healthy outside mask, hashes/runtime/RAM.
- **EVIDENCE:** manual-only workflow prepared, 0/3 attempts consumed.
- **BLOCKER:** sequence requires OBJ-004.
- **NEXT:** attempt 1/3 only after DamageMaskNet resolution.

### OBJ-009 — Paper Quality Windows model pack/installer
- **VERSION/TRACK:** V2→V5 / release
- **STATUS:** PROPOSED
- **SUCCESS:** offline verified pack, hashes/licenses, no dev Python, clean-machine execution.
- **NEXT:** after model/router qualification.

### OBJ-010 — Real HP EliteBook acceptance
- **VERSION/TRACK:** V5 / release
- **STATUS:** PROPOSED
- **SUCCESS:** real blur/JPEG/mosaic/scribble/sticker/low-light/multi-ref tests with measured CPU/RAM/backend/output hash/identity and no OOM/network/CUDA/dev tools.
- **NEXT:** after same-candidate Windows installer.

---

## 6. MODEL MASTER REGISTRY

All serious models remain listed even when rejected/blocked. Exact upstream hashes/licenses are taken from project registries/research evidence; unknown fields remain `NOT_VERIFIED` rather than guessed.

### Production / established conservative stack

**YuNet** — role: face detection/5-point geometry; upstream OpenCV Zoo; exact URL/hash in `app/model_registry.py`; CPU OpenCV DNN; Windows historically used; state **QUALIFIED for PRODUCT_V1**; blocks 4/11 support; target-PC specific state NOT_VERIFIED.

**SFace** — role: identity hard gate; OpenCV Zoo; threshold `0.363`; CPU OpenCV FaceRecognizerSF; state **QUALIFIED for PRODUCT_V1**; blocks 7/11 safety; wrong-person references never gain authority through score maximization.

**NAFNet** — role: lightweight deblur/denoise/pre-clean; project ONNX registry; CPU/OpenCV DNN; state **QUALIFIED for current conservative role**; blocks 2/3; not considered a missing-identity generator.

**Face Parsing ResNet18 ONNX** — role: 19-class semantic face parsing; `yakhyo/face-parsing`; MIT code; registry-pinned model hash; 512 ONNX Runtime CPU; state **QUALIFIED for current parsing role**; blocks 6/7/8.

**Head Pose MobileNetV2 ONNX** — role: pose geometry; CPU ONNX; state **QUALIFIED for current role**; blocks 4/10.

**LaMa ONNX** — role: residual non-identity-critical inpainting only; CPU ONNX; state **QUALIFIED only under constrained policy**; never evidence for eye/nose/lip identity structure; block 8.

### Paper Quality / specialist candidates

**GPEN BFR-512**
- role: fast blind face restoration; input 512 aligned;
- upstream: `yangxy/GPEN`; exact source/checkpoint evidence in research report;
- license/weight redistribution: unresolved for product distribution;
- CPU: real Linux DEV PASS; Windows/EliteBook NOT_RUN;
- measured DEV: SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, peak RSS `~1.828GB`;
- supported damage: severe blind face degradation/mixed blur;
- risk: generative texture/identity-detail hallucination;
- state: **BENCHMARKING / BLOCKED_LICENSE for distribution decision**;
- version/blocks: PRODUCT_V2, 2/3/8;
- evidence: `research/GPEN_VERTICAL_SLICE_REPORT.md`.

**GFPGAN v1.4**
- role: blind real-world face restoration; input 512 aligned;
- official v1.4 release asset; redistribution terms must be preserved/reverified;
- Linux CPU DEV PASS; Windows/EliteBook NOT_RUN;
- measured comparable DEV: SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, peak RSS `~1.666GB`;
- state: **BENCHMARKING**; no one-image winner claim;
- blocks 2/3/8; evidence `research/GFPGAN_V1_4_VERTICAL_SLICE_REPORT.md`.

**CodeFormer**
- role: severe restoration / controllable fidelity / potential face inpainting;
- official `w=0.5` aligned CPU slice executed;
- S-Lab terms are a production redistribution/commercial blocker unless compatible authorization exists;
- Linux CPU real slice PASS under resource governor; Windows/EliteBook NOT_RUN;
- exact PSNR/SSIM/RAM/time must be recovered from artifact, never reconstructed from memory;
- state: **BENCHMARKING / BLOCKED_LICENSE**; blocks 2/3/8.

**FBCNN**
- role: blind JPEG/double-JPEG specialist;
- real Linux CPU DEV QF20 result: PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, SFace `0.9571→0.9691`, peak RSS `~1.305GB`;
- Windows/EliteBook NOT_RUN; double-JPEG/social-media breadth incomplete;
- state: **BENCHMARKING / current JPEG leader**; block 3; evidence `research/PHASE7_FBCNN_RESULT.md`.

**DamageMaskNet small U-Net**
- role: 12-class localized damage segmentation (`HEALTHY` + 11 corruption classes);
- CFS research training; exact synthetic masks; FairFace + ControlFace bank;
- intended CPU ONNX; attempt-3 IoU/F1/parity/RAM/runtime NOT_VERIFIED;
- state: **BENCHMARKING / BLOCKED**; block 6; if attempt 3 is true model/data FAIL, U-Net hypothesis ends.

**RefFaceInpainting**
- role: same-person reference-guided large facial occlusion; 256;
- upstream `WuyangLuo/RefFaceInpainting@0f1ad75677cc8fae4ae14d878e4c6cfce9365f28`; MIT repository; official generator/ArcFace links;
- upstream CUDA-hardcoded; CFS prepared minimal CPU path loads only `UnetG + ArcFace resnet101` and patches one device allocation without changing weights/math;
- CFS runtime result: NOT_RUN; Windows/EliteBook NOT_RUN;
- state: **FEASIBILITY_ONLY / NOT_RUN**; block 8; evidence manual workflow/status report.

**InstantRestore** — role: personalized multi-reference restoration; official implementation uses two UNets + two VAEs + CLIP and CUDA/FP16 assumptions; license unresolved in current audit; CPU/Windows/EliteBook NOT_RUN; state **FEASIBILITY_ONLY / BLOCKED_HARDWARE + license audit**; future V3 challenger.

**OSDFace** — role: modern one-step severe blind challenger; official inference has CUDA/device-stream assumptions; CPU/Windows NOT_RUN; state **FEASIBILITY_ONLY / BLOCKED_HARDWARE**.

**RestoreFormer++** — role severe blind challenger; exact production checkpoint/license/backend not selected in this pass; no CFS CPU evidence; state **DISCOVERED / FEASIBILITY_ONLY**.

**VQFR** — role blind restoration challenger; no CFS CPU/Windows evidence; state **DISCOVERED / FEASIBILITY_ONLY**.

**GPEN face inpainting** — role missing-region challenger; same GPEN licensing concern, high-resolution CPU cost unverified; state **FEASIBILITY_ONLY**.

**RefineFIR** — reference-guided architecture teacher; public executable/checkpoint path inadequate for a fair CFS runtime benchmark; state **FEASIBILITY_ONLY**; useful copy-or-not concept.

**PerFuSe** and **RefIPFR** — personalized/reference architecture teachers; official executable runtime suitable for current CFS benchmark NOT_VERIFIED; state **DISCOVERED / FEASIBILITY_ONLY**.

**Real-ESRGAN** — optional Paper Quality x2 upscale challenger; no current CFS qualification; CPU cost/identity/background tradeoff unmeasured; state **FEASIBILITY_ONLY**; block 12.

### Registry integrity issue

Certified `THIRD_PARTY_MODULES.md` refers to machine-readable catalog files under `models/`, but reconciled `main/models/` contains only `README.md`. Active production catalog/download validation is in `app/model_registry.py` and companion runtime/production registries. This stale documentation must be corrected explicitly; do not invent missing manifests.

---

## 7. CURRENT MODEL EVIDENCE

### DEVELOPMENT only

| Model | Scope | SFace | PSNR | SSIM | Time | Peak RSS | Status |
|---|---|---:|---:|---:|---:|---:|---|
| GPEN BFR-512 | one aligned comparable DEV face | `0.95397` | `28.07` | `0.7474` | `~2.697s` @512 Linux CPU | `~1.828GB` | BENCHMARKING |
| GFPGAN v1.4 | same comparable DEV face | `0.91665` | `30.65` | `0.8604` | `~2.787s` | `~1.666GB` | BENCHMARKING |
| FBCNN | JPEG QF20 | `0.9571→0.9691` | `34.62→36.78` | `0.9486→0.9634` | artifact/report | `~1.305GB` | JPEG leader in DEV |
| CodeFormer w=0.5 | aligned DEV slice | real identity gate PASS | artifact required | artifact required | artifact required | artifact required | BENCHMARKING / license blocked |

These are not Windows, validation, holdout or EliteBook measurements.

---

## 8. 13-BLOCK ARCHITECTURE

| # | Block | Current function/model | Authority/provenance | Current failure/quality limit | Future change/model | Version | Implementation/test/benchmark | Resource/deps | Next action |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | IMPORT | deterministic OpenCV/Pillow, MAIN + refs | imported bytes observed | source quality varies | richer source/hash metadata; no learned import | V2/V5 | V1 RELEASED | low | preserve deterministic import |
| 2 | DEBLUR | NAFNet mild deblur | model output not observed evidence | severe blur may need facial prior | common checkpoint -> NAFNet or accepted GPEN/GFPGAN/CodeFormer, never blind chain | V2/V4 | research BENCHMARKING | one heavy model | broaden BFR validation |
| 3 | ENHANCE | conservative NAFNet/general enhancement | healthy MAIN priority | no V1 JPEG specialist | FBCNN when JPEG detected; low-light specialist only if detected/qualified | V2/V4 | FBCNN BENCHMARKING | <=80% | broaden JPEG families |
| 4 | LANDMARKS | YuNet + pose support | measured geometry only | severe damage/pose reduces confidence | MediaPipe/3DDFA only if measured | V2/V4 | V1 qualified | low/moderate | no invented landmarks |
| 5 | ALIGN | deterministic similarity/affine/RANSAC | geometry only | landmark quality | one common aligned candidate checkpoint | V2 | implemented | low | standardize A/B |
| 6 | OCCLUSION_MASK | face parsing + heuristics | repair mask, not identity | damage-family discrimination weak | DamageMaskNet 12-class + confidence/component | V2/V4 | implemented pipeline; training result NOT_VERIFIED | lightweight ONNX target | recover attempt 3 |
| 7 | REGION_SELECT | component bank/reference memory | same-person observed only | legacy ranking not sufficiently personalized | 13-component Personalized Reference Bank | V3 | prototype implemented; workflow NOT_VERIFIED | low/moderate | validate 0/1/9 refs |
| 8 | INPAINT | observed reference first; constrained LaMa residuals | observed > inferred; identity-critical LaMa prohibited | no qualified generative missing-detail specialist | RefFace/CodeFormer/GPEN candidate only in Paper mode | V2/V3/V4 | reference-first implemented; RefFace PREPARED | heavy candidate | resolve DamageMaskNet then RefFace |
| 9 | FUSION | deterministic/regional | healthy MAIN > observed same-person | component tradeoffs | deterministic component-aware fusion; generated only in remaining authority | V2/V4 | research implemented, latest workflow NOT_VERIFIED | low | validate seams/provenance |
| 10 | FRONTALIZE | geometry-only conservative | no hidden-side synthesis | unresolved hidden region | Paper mode may synthesize only as GENERATED | V4 | planned | moderate | keep modes separate |
| 11 | IDENTITY_CHECK | SFace `0.363` + hardening | wrong-person never anchor/donor | V1.1 legitimate transfer regressions under validation | direct MAIN edge + face-local bridge; robust full-ref aggregation later | V1.1/V3 | **new direct-transfer patch VALIDATING** | low/moderate | evaluate `3e919f7a...` |
| 12 | UPSCALE | Lanczos | deterministic | limited perceptual SR | optional Real-ESRGAN x2 if benchmark wins | V2/V4 | feasibility only | potentially heavy | defer |
| 13 | EXPORT | deterministic image/evidence | exact provenance | richer Paper reports needed | generated mask, component source map, selection/identity/damage/timing/RAM/hashes | V2/V5 | partial research | low | unify after router qualification |

---

## 9. PHOTO AND INPUT CONTRACT

### MAIN

Target corruption domain includes: low resolution, smartphone/social-media recompression, JPEG/double JPEG, defocus/motion/mixed blur, noise, pixelation, block mosaic, scribble, sticker, black bar, opaque mask/block, partially/fully covered eye, covered mouth/nose, missing component, crop/partial face, low light/uneven exposure and mixed/unknown real-world degradation.

MAIN remains the target canvas/pose/frame in Conservative Mode.

### REFERENCES

Contract: MAIN + 0–9 references. A reference may be full, partial, eye/mouth/nose-only, side angle, different expression/light/resolution, blurred/compressed/occluded, useless or wrong-person.

- accepted **FULL same-person** -> may be global anchor and component donor if direct/approved identity evidence and geometry allow;
- accepted **PARTIAL same-person** -> component-local authority only;
- **wrong-person** -> never anchor, observed donor or identity-score improver.

Image quality eligibility is separate from identity eligibility.

### RESEARCH / GROUND TRUTH

Store source/license, identity, original resolution/hash, partition, degradation parameters/seed, exact damage mask where generated and derivative hashes. Final holdout identities/images are never training/tuning data.

---

## 10. DATASET CONSTRUCTION

Paper Quality planning target: approximately **300–400** representative face images/cases initially, with explicitly documented female-domain proportion when intentionally emphasized. Do not silently bias.

Protected identity partitions: `TRAIN`, `DEVELOPMENT`, `VALIDATION`, `FINAL_HOLDOUT`; forbidden identity leakage across protected splits.

Per source/example record: `source, usage/license basis, download date, identity_id, file hash, original resolution, domain/gender label when intentionally used, split, degradation, severity, seed, exact mask, reference relationships`.

Current data assets:
- early `research/face-restoration-v2` dataset/degradation specification (historical/superseded active branch, not deleted);
- DamageMaskNet mixed FairFace real + ControlFace multi-view bank with identity-disjoint ControlFace validation;
- frozen V4 holdout excluded from Track B training/validation.

---

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Canonical components:
`LEFT_EYE`, `RIGHT_EYE`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`, `NOSE`, `PHILTRUM`, `MOUTH_LIPS`, `LEFT_CHEEK`, `RIGHT_CHEEK`, `CHIN`, `JAW`, `FOREHEAD`, `FACE_CONTOUR`.

Per component track: MAIN visibility/damage, best/alternate observed refs, reference confidence/coverage, generated candidate availability, selected source/provenance, identity/geometry consistency and unresolved state.

Observed same-person evidence outranks generated inference. A failed eye/nose/mouth target cannot be silently recovered later by a broad cheek/jaw donor.

---

## 12. DAMAGE ROUTING

| Damage | Detector | Primary | Secondary | Ref-first? | Generative? | Specialist | Fallback/abstention |
|---|---|---|---|---|---|---|---|
| HEALTHY | damage map/heuristics | preserve MAIN | none | N/A | no default | none | unchanged MAIN |
| DEFOCUS_BLUR | blur/damage metric | NAFNet/validated deblur | BFR if severe | when better observed component | Paper only | NAFNet/BFR | rollback/unresolved |
| MOTION_BLUR | damage metric | deblur | BFR if severe | yes | Paper only | NAFNet/BFR | rollback |
| NOISE | noise metric | NAFNet/current winner | specialist if measured | optional | only if needed | NAFNet | preserve healthy detail |
| JPEG | JPEG detector | FBCNN | face candidate only if residual damage | optional | after specialist | **FBCNN** | avoid heavy model when unnecessary |
| DOUBLE_JPEG | JPEG detector | FBCNN | BFR only if residual warrants | optional | yes | FBCNN | validation pending |
| PIXELATION | DamageMaskNet | observed component reconstruction | generated candidate | YES | Paper yes | reference bank/BFR | unresolved Conservative |
| BLOCK_MOSAIC | DamageMaskNet | observed component reconstruction | generated component | YES | Paper yes | reference bank/future specialist | unresolved Conservative |
| SCRIBBLE | DamageMaskNet | observed repair | RefFace/generated | YES | Paper yes | RefFace if qualified | unresolved Conservative |
| STICKER | DamageMaskNet | observed repair | RefFace/generated | YES | Paper yes | RefFace if qualified | unresolved Conservative |
| OPAQUE_BLOCK | DamageMaskNet | observed repair | reference-conditioned inpaint | YES | Paper yes | RefFace if qualified | safe abstention at zero evidence |
| BLACK_BAR | DamageMaskNet | observed repair | reference-conditioned inpaint | YES | Paper yes | RefFace if qualified | safe abstention Conservative |
| PARTIAL_OCCLUSION | parser/DamageMaskNet | observed repair | generated candidate | YES | Paper yes | qualified ref specialist | rollback/unresolved |
| MISSING_COMPONENT | component visibility | observed component bank | generated component | YES | Paper yes | RefFace/BFR if qualified | unresolved Conservative |
| LOW_LIGHT | exposure detector | conservative correction | Zero-DCE++ only if qualified | optional | Paper yes | Zero-DCE++ feasibility | no blanket brightening |
| MIXED_DAMAGE | multi-class map | specialist routing | minimal additional candidates | depends | Paper yes | dynamic router | never blind-chain all generators |

---

## 13. DECISION LOG — engineering outcomes, not private reasoning

### DEC-20260819-001 — Canonical meta ledger
- **Proposal:** `PROJECT_MASTER_STATE.md` on `meta/project-state` is canonical.
- **Problem:** project state was fragmented across chats/branches.
- **Evidence:** multiple diverged branches, stale V1 docs, active hotfix/research state.
- **Benefit:** auditable continuity.
- **Risk:** ledger drift if update protocol ignored.
- **Status:** ACCEPTED.

### DEC-20260819-002 — Advanced research branch is active architecture
- `research/paper-quality-local-v2` is active; early `research/face-restoration-v2` is preserved but superseded as active architecture. They are not falsely described as merged/superset.
- **Status:** ACCEPTED.

### DEC-20260819-003 — 80% total-PC resource contract
- <=80% logical CPU; <=80% process/system RAM; one heavy model resident.
- **Reversal:** only measured target-PC evidence supports a versioned change.
- **Status:** ACCEPTED.

### DEC-20260819-004 — Evidence authority outranks generated quality
- healthy MAIN > verified same-person observed reference > accepted generated candidate.
- **Status:** ACCEPTED.

### DEC-20260819-005 — DamageMaskNet mixed source bank
- Wikimedia attempt 1 HTTP403 and attempt 2 HTTP429 were acquisition failures; attempt 3 changed acquisition to FairFace + ControlFace while keeping U-Net hypothesis/hyperparameters.
- **Status:** ACCEPTED for attempt 3.

### DEC-20260819-006 — RefFace next large-occlusion specialist after DamageMaskNet gate
- selected for task specificity; generic CodeFormer-everywhere, heavier InstantRestore/OSDFace and GPEN inpainting remain alternatives/challengers.
- **Status:** ACCEPTED / BLOCKED BY SEQUENCE.

### DEC-20260819-007 — V3 consumed; V4 untouched one-shot
- V3 never rerun/tuned. V4 freeze exists but no consumption/request marker.
- **Status:** ACCEPTED safety protocol.

### DEC-20260819-008 — Separate ranking cluster from component-transfer identity authority
- **DATE:** 2026-08-19
- **PROPOSAL:** preserve the preflight connected SFace component for ranking only; persist observed-reference component-transfer authority only when the existing preflight SFace matrix contains a direct MAIN(0)↔REF edge at `>=0.363`.
- **PROBLEM:** the largest single-link cluster can exclude MAIN or propagate A-B-C even though MAIN-C is below threshold; at the same time, a legitimate direct MAIN↔REF proof was being discarded if that ref was not in the selected ranking component.
- **AFFECTED VERSION/TRACK:** PRODUCT_V1_1 / Track A.
- **AFFECTED BLOCKS:** preflight, reference eligibility/ALIGN, IDENTITY_CHECK; same-canvas evidence contract.
- **MODELS:** SFace only; no model/checkpoint/threshold change.
- **EVIDENCE:** previous same-HEAD hidden/targeted regressions expected component-transfer acceptance/persistence and a same-canvas identity-bridge list; existing V4 direct-edge tests explicitly prohibit transitive authority.
- **EXPECTED BENEFIT:** recover legitimate same-person donor eligibility without reopening reference-only/transitive trust.
- **RISKS:** treating ranking-cluster membership as authority would reintroduce wrong-person/transitive risk; explicitly prohibited by this decision.
- **ALTERNATIVES:** lower SFace threshold (rejected); trust majority/reference-only cluster (rejected); weaken face-local same-canvas proof (rejected).
- **REVERSAL:** evidence shows direct-MAIN persistence itself violates wrong-person/provenance gates.
- **STATUS:** VALIDATING at `3e919f7a...`.

---

## 14. EXPERIMENT LOG

### EXP-20260817-001 — GPEN BFR-512
- DEVELOPMENT Linux CPU, aligned 512; real inference after setup dependency fix.
- SFace `0.95397`; `~2.697s`; peak RSS `~1.828GB`; PSNR `28.07`; SSIM `0.7474`.
- **Conclusion:** BENCHMARKING, not production-qualified.

### EXP-20260817-002 — GFPGAN v1.4
- first output technically ran but alignment mismatch invalidated A/B; corrected comparable run succeeded.
- SFace `0.91665`; `~2.787s`; peak RSS `~1.666GB`; PSNR `30.65`; SSIM `0.8604`.
- **Conclusion:** BENCHMARKING; no one-image winner.

### EXP-20260817-003 — CodeFormer w=0.5 CPU
- attempt 1 packaging/import failure (`basicsr.version`) before real model inference;
- attempt 2 real CPU PASS under 80% governor;
- exact numerical artifact must be recovered before comparative restatement.
- **Conclusion:** BENCHMARKING / license blocker.

### EXP-20260817-004 — FBCNN JPEG QF20
- real CPU PASS; PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, SFace `0.9571→0.9691`, peak RSS `~1.305GB`.
- **Conclusion:** current DEV JPEG leader; broaden.

### EXP-20260818-005 — DamageMaskNet attempt 1/3
- **Result:** INFRASTRUCTURE FAIL before training, Wikimedia HTTP403.

### EXP-20260818-006 — DamageMaskNet attempt 2/3
- **Result:** INFRASTRUCTURE FAIL before training, Wikimedia HTTP429 after verified sources.

### EXP-20260818-007 — DamageMaskNet attempt 3/3
- FairFace + ControlFace acquisition; same U-Net hypothesis.
- **Result:** NOT_VERIFIED because final run evidence is inaccessible through current interface.
- **Decision:** no attempt 4 for observability.

### EXP-20260819-008 — RefFace CPU vertical slice
- **Status:** PREPARED / NOT_RUN; attempt `0/3` consumed.
- Plan: two same-identity ControlFace views, exact opaque face mask, CFS parser/YuNet/SFace, minimal CPU `UnetG + ArcFace resnet101`, 80% resource contract, exact MAIN outside mask, GENERATED provenance.

### EXP-20260819-009 — Track A direct component-transfer authority repair
- **Hypothesis:** direct preflight MAIN↔REF SFace evidence must survive ranking-component selection without allowing transitive cluster authority.
- **Attempt:** 1/3 for this specific implementation hypothesis.
- **Start HEAD:** `3645c8c39653d04616167e881adaf28d2b93cd45`.
- **Technical HEAD:** `3e919f7a1cc54e1bdb00607c2bbeece1d3392724`.
- **Changes:** `PreflightCandidate.accepted_for_component_transfer`; direct-MAIN matrix extraction; persisted `reference_component_transfer_accepted`; V2 audit accepts explicit direct-transfer evidence; same-canvas face-local bridge writes old and canonical alias keys; new regression test locks A-B-C non-transitivity.
- **Threshold/model change:** NONE; SFace remains `0.363`.
- **Result:** NOT_VERIFIED pending new-head test/CI evidence.
- **Next:** inspect targeted/full test result before attempt 2.

---

## 15. QUALITY SCOREBOARD

Keep scopes separate.

### DEV
GPEN/GFPGAN/FBCNN values are recorded in Section 7; CodeFormer exact artifact recovery required. Broad BFR validation incomplete.

### VALIDATION
DamageMaskNet attempt 3 NOT_VERIFIED; Personalized Reference Bank framework exists but latest workflow evidence NOT_VERIFIED; broad selector calibration not run.

### HOLDOUT
- V1 historical certification: 40/40.
- V3 consumed: 39/40, SFace failure `0.360 < 0.363`.
- V4: frozen, NOT_RUN, UNCONSUMED.

### REAL-WORLD
Prior hotfix head Female-domain workflow FAIL after benchmark execution at report validation. New head state NOT_VERIFIED.

### TARGET-PC
HP EliteBook 1030 G3: Paper Quality NOT_RUN.

Scoreboard dimensions to maintain: SFace/identity, PSNR, SSIM, LPIPS, NIQE where useful, healthy MAE, damage recovery, wrong-person observed pixels, provenance violations, generated/reference-supported/unresolved fraction, component geometry, runtime and RAM.

---

## 16. TARGET HARDWARE

Primary: **HP EliteBook 1030 G3, 16 GB RAM, Windows**. Exact CPU/GPU must be detected at runtime.

- CPU-first; no CUDA requirement.
- Optional OpenVINO/Intel acceleration only after actual support and output parity tests.
- <=80% logical processors.
- <=80% process RAM and <=80% whole-system RAM.
- maximum one heavy restoration model resident.

Every serious model eventually needs real target-PC: CPU/RAM/Windows/backend, model load seconds, inference seconds, peak RAM, output hash, identity result. Linux CPU numbers are never relabeled as EliteBook evidence.

---

## 17. RELEASE SAFETY RULES

Frozen/current invariants:

- SFace threshold `0.363`.
- wrong-person observed contribution `0 pixels`.
- provenance violations `0`.
- healthy/outside repair MAE `<=8.0` where the frozen release policy applies.
- independent calibration remains valid.

Forbidden: threshold shopping, cherry-picking, consumed-holdout rerun/tuning, difficult-case removal, generated-as-observed provenance, raw wrong-person score maximization, auto-merge, certified-history force push, fabricated CI/RAM/speed/hash/output.

---

## 18. PROVENANCE CLASSES

Canonical minimum:
- `MAIN_OBSERVED`
- `OBSERVED_REFERENCE`
- `SYMMETRY_INFERRED`
- `GENERATED_MODEL_INFERRED`
- `UNRESOLVED`

Identity similarity never changes provenance class.

---

## 19. TRACK A — PRODUCT_V1_1 OPERATIONAL RELEASE

### Previous exact head — preserved failure evidence

`3645c8c39653d04616167e881adaf28d2b93cd45`

- Release Quality targeted suite: **4 failed, 195 passed**.
- `test_main_source_zero_never_changes_after_preflight`: expected legitimate reference `accepted_for_component_transfer=True`, observed False.
- `test_component_transfer_acceptance_is_persisted_for_all_blocks`: expected `[True]`, observed `[False]`.
- `test_final_identity_reuses_preflight_acceptance_instead_of_requiring_main_bridge`: expected accepted identity; observed reason `main_not_in_accepted_sface_cluster`.
- `test_restores_imported_primary_when_selected_anchor_is_verified_same_canvas`: expected face-local identity bridge list `[1]`, observed `[]`.
- Windows: FAIL during Python-test stage; exact nested assertion NOT_VERIFIED in this ledger.
- Female-domain: benchmark execution completed, lightweight report validation FAIL; exact nested assertion NOT_VERIFIED.

### Current exact head

`3e919f7a1cc54e1bdb00607c2bbeece1d3392724`

Technical change:

1. `app/preflight.py`
   - adds `accepted_for_component_transfer` independently of `accepted_identity` ranking membership;
   - computes it **only** from the existing pairwise SFace matrix using a direct MAIN source-0 edge at the unchanged frozen threshold;
   - persists per-reference transfer flags and authority source IDs;
   - A-B-C transitive linking does not confer C authority.
2. `app/face_domain_guard_v2_policy.py`
   - preserves explicit direct preflight component-transfer acceptance through the existing identity-eligibility audit/firewall.
3. `app/primary_anchor_policy.py`
   - retains strict face-local identity proof;
   - writes both `identity_bridge_original_reference_indices` and `identity_bridge_matched_original_reference_indices` from the **same strict face-local list**, never the broad whole-canvas list.
4. `tests/test_preflight_component_transfer_authority.py`
   - regression locks direct-only A-B-C behavior, direct transfer persistence and same-canvas alias equality.

**Current test/CI state:** NOT_VERIFIED. Do not claim this fixes all four failures until same-head evidence exists.

**Next exact action:** read the new-head targeted test result once available. If it fails, diagnose only the remaining failure and keep this hypothesis within its 3-attempt limit. Then full pytest and same-head Windows/Female/Release Quality. No V3/V4 execution.

---

## 20. TRACK B — PAPER QUALITY

Active: `research/paper-quality-local-v2@645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd`.

Current research layers:
- 80% governor + one-heavy-model lifecycle;
- real GPEN/GFPGAN/CodeFormer/FBCNN CPU slices;
- damage taxonomy + DamageMaskNet pipeline;
- 13 components including FACE_CONTOUR;
- Personalized Reference Bank;
- reference-first observed repair with original source-index provenance;
- hard-gated candidate selector framework;
- deterministic component-aware fusion;
- 19-class CFS parser adapter;
- RefFace CPU vertical slice prepared/manual-only/NOT_RUN.

Models must eventually converge to QUALIFIED, REJECTED or a documented blocker; not remain indefinitely optional.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Recover the already-triggered DamageMaskNet attempt 3 **without rerunning/tuning**.

- **PASS:** record per-class IoU/F1, ONNX parity, RAM/runtime and source-bank hashes; decide whether U-Net remains valid.
- **INFRASTRUCTURE FAIL:** correct infrastructure only; do not silently change data/model hypothesis.
- **MODEL/DATA QUALITY FAIL:** attempt 3 exhausts the three-attempt U-Net hypothesis; stop and document a new lightweight segmentation hypothesis rather than micro-tuning.

Only after this gate: RefFace CPU attempt 1/3.

---

## 22. SPECIALIST MODEL STRATEGY

`INPUT -> detect/align -> damage classification -> reference analysis -> identity anchors -> specialist route -> candidates -> component evaluation -> identity/geometry/quality gates -> evidence-aware deterministic fusion -> final identity -> provenance/export`

- JPEG -> FBCNN candidate.
- Blur -> dedicated deblur / measured BFR, not automatic chaining.
- Opaque loss + same-person ref -> observed evidence first; RefFace-like generation only if qualified.
- Unsupported missing detail -> Paper Quality generated inference when enabled.
- Observed reference detail -> evidence-first.
- Never GPEN→GFPGAN→CodeFormer blindly.

---

## 23. MODEL SELECTION POLICY

Blind-restoration winners require multiple identity-disjoint DEV/VALIDATION images and per-degradation reporting. Measure identity, perceptual quality, geometry, artifacts, healthy preservation, PSNR/SSIM/LPIPS, runtime/RAM. Final holdouts are never used to select weights/models. Identity is a hard gate before weighted ranking. Remove complexity that does not earn measured benefit.

---

## 24. HISTORICAL RECORD — append-only

### HIST-20260815-001 — PRODUCT_V1 certified merge
- candidate `release/v1-certified@f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` -> `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`.
- merge commit records Windows #1195, Female #463, Release Quality #13.
- decision: immutable PRODUCT_V1 regression base.

### HIST-20260818-002 — Track A blocked hotfix snapshot
- branch/head `hotfix/real-world-restoration-v1.1@3645c8c39653d04616167e881adaf28d2b93cd45`.
- Release Quality FAIL (`4 failed, 195 passed` targeted); Windows FAIL; Female FAIL.
- V3 consumed/no rerun; V4 frozen/unexecuted/unconsumed.
- objective: OBJ-002.

### HIST-20260818-003 — Advanced Paper Quality research snapshot
- `research/paper-quality-local-v2@645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd`.
- real DEV model evidence + routing/fusion infrastructure; DamageMaskNet attempt 3 NOT_VERIFIED; RefFace PREPARED/NOT_RUN.
- objectives OBJ-004..OBJ-008.

### HIST-20260819-004 — Canonical ledger established
- created `meta/project-state` from certified `main@2767513f...`.
- project state no longer depends on chat memory.

### HIST-20260819-005 — Direct component-transfer authority patch
- **Timestamp:** 2026-08-19T05:46+02:00.
- **Technical branch:** `hotfix/real-world-restoration-v1.1`.
- **Previous HEAD:** `3645c8c39653d04616167e881adaf28d2b93cd45`.
- **Exact technical HEAD:** `3e919f7a1cc54e1bdb00607c2bbeece1d3392724`.
- **Files changed:** `app/preflight.py`, `app/face_domain_guard_v2_policy.py`, `app/primary_anchor_policy.py`, `tests/test_preflight_component_transfer_authority.py`.
- **Models/checkpoints:** NONE changed. SFace threshold remains `0.363`.
- **Objective affected:** OBJ-002.
- **Safety intent:** direct MAIN↔REF SFace authority survives ranking selection; ranking clusters remain non-authoritative; face-local same-canvas proof remains strict; no transitive A-B-C trust.
- **Tests:** new regression definitions committed; GitHub same-head result **NOT_VERIFIED at ledger update time**. Prior failures remain preserved above.
- **Next action:** evaluate new-head targeted/full tests before any second implementation attempt.
