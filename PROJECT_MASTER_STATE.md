# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this before any engineering decision. GitHub evidence overrides chat memory. Current-state sections are maintained; Decision, Experiment and Historical records preserve important prior outcomes.

## 0. Document metadata

- Last ledger update: `2026-08-19T05:56:46+02:00`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Previous technical HEAD: `3e919f7a1cc54e1bdb00607c2bbeece1d3392724`
- Last technical HEAD: `7c683eddad579974329bb186622aea01feccee61`
- Active tracks: Track A PRODUCT_V1_1 stabilization; Track B Paper Quality research
- Current Track A blocker: trusted-anchor implementation attempt 2/3 is pushed and **NOT_VERIFIED**. Attempt 1 failed four tracked V4 identity-policy tests.
- Current Track B blocker: recover already-triggered DamageMaskNet mixed-source attempt-3 evidence without rerunning for observability.

| Global gate | State |
|---|---|
| FORENSIC_MODE_READY | TRUE for certified PRODUCT_V1 only |
| PAPER_QUALITY_MODE_READY | FALSE |
| WINDOWS_INSTALLER_READY | PARTIAL — historical V1 only |
| TARGET_HARDWARE_READY | FALSE |
| QUALITY_TARGET_ACHIEVED | FALSE |
| PROJECT_FINISHED | FALSE |

Mandatory sequence: `technical work -> evidence/tests -> push -> exact remote SHA -> update this ledger -> push ledger`.

## 1. Executive project summary

CFS is a local Windows face-restoration system for difficult smartphone/social-media portraits. Conservative Mode treats MAIN and verified same-person observed reference information as evidence, with explicit provenance and safe abstention. Paper Quality Mode may synthesize missing information but every generated pixel remains `GENERATED_MODEL_INFERRED`.

PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is an operational/safety hotfix, isolated from Paper Quality models. Track B is an ML research program with real CPU development evidence for GPEN, GFPGAN v1.4, CodeFormer and FBCNN plus resource, damage-routing, personalized-reference and component-fusion prototypes.

Track A currently separates ranking clusters from identity authority: connected SFace clusters may rank references, but global authority must be direct whole-face SFace evidence or an explicitly safe bridge. Attempt 2 refines the difference between full-face global anchors and partial same-canvas component evidence.

## 2. Branch and release map

| Branch | Purpose | Verified HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | signed PR #1 merge; historical Windows #1195, Female #463, Release Quality #13 | preserve |
| `feature/block-pipeline-v1` | original V1 implementation | `5eff667373cd47c07ba14aaad2acafee6d5a61c1` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | certified V1 candidate | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | FROZEN / ARCHIVED | merged via PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `7c683eddad579974329bb186622aea01feccee61` | VALIDATING / ACTIVE | PR #2 OPEN/DRAFT; this exact HEAD CI NOT_VERIFIED | targeted tests then full/release gates |
| `research/face-restoration-v2` | early degradation/dataset research | `757a3f6081b7b152cdc615a07cd99aec40fa0a1c` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd` | ACTIVE / BENCHMARKING | research evidence partly NOT_VERIFIED | DamageMaskNet evidence recovery |
| `meta/project-state` | canonical ledger | intentionally not self-recorded | ACTIVE META | documentation branch | update after each technical push |

Research branches diverged from the same certified base; the advanced branch is not falsely described as a Git superset of the early branch.

## 3. PRODUCT VERSION ROADMAP

- **PRODUCT_V1 — RELEASED:** certified conservative/forensic baseline. SFace `0.363`; wrong-person observed pixels `0`; provenance violations `0`; historical Windows certification. EliteBook-specific acceptance NOT_VERIFIED.
- **PRODUCT_V1_1 — VALIDATING/BLOCKED:** operational real-world hotfix, same production model family. Current identity-policy attempt 2/3 at `7c683edd...`; no Track B generative rescue.
- **PRODUCT_V2 — BENCHMARKING:** Paper Quality Local; modern BFR candidates, damage-aware specialists, hard identity gates, generated provenance, deterministic fusion, <=80% resource contract.
- **PRODUCT_V3 — PLANNED with prototypes:** personalized MAIN + 0–9 references, per-component authority; full refs may be global anchors, partial refs local only, wrong-person never anchor/donor/score booster.
- **PRODUCT_V4 — PLANNED with prototypes:** damage-specialist hybrid routing and component-aware fusion.
- **PRODUCT_V5 — PLANNED:** unified modes + personalized references + specialists + offline Windows pack/installer + clean target-PC acceptance.

Product versions are never confused with HOLDOUT_V1…V5.

## 4. HOLDOUT / benchmark lineage

| Evaluation set | State | Tuning rule |
|---|---|---|
| CALIBRATION_V1 | historical 60/60 certified evidence | original calibration only |
| FINAL_HOLDOUT_V1 | historical 40/40 | consumed for certification; no tuning |
| FINAL_HOLDOUT_V2 | details NOT_VERIFIED in this reconciliation | do not use until recovered |
| FINAL_HOLDOUT_V3 | **CONSUMED**, 39/40; mosaic case SFace `0.360 < 0.363` | NEVER rerun/tune |
| FINAL_HOLDOUT_V4 | 40 cases / 20 identities; frozen ControlFace10K; **NOT_RUN / UNCONSUMED** | execute only via valid one-shot sequence; never tune |
| FINAL_HOLDOUT_V5 | not created | future independent final set |
| Female-domain | quick profile ~300–400 cases | quality report-only; safety hard gates |
| Paper Quality DEV | partial | model/router development allowed |
| Paper Quality VALIDATION | incomplete | independent selection validation |
| DamageMaskNet bank | FairFace + ControlFace with exact synthetic masks | TRAIN/VALIDATION only |

At attempt-1 HEAD, V3 manifest verification PASS without execution and V4 freeze/history verification PASS without execution. Frozen V4 blobs matched original freeze commit `ad564c9b1cd9514250eac08425d16c2414ead9fa`.

## 5. CURRENT GLOBAL OBJECTIVES

- **OBJ-001 Preserve PRODUCT_V1 — PASS.** Never rewrite certified history.
- **OBJ-002 Restore PRODUCT_V1_1 gates — VALIDATING.** Success: targeted tests + full pytest + same-HEAD Windows/Female/Release Quality with zero wrong-person/provenance violations. Attempt 2/3 pushed at `7c683edd...`; result NOT_VERIFIED.
- **OBJ-003 Canonical ledger — IN_PROGRESS.** This update records exact attempt-2 SHA before CI interpretation.
- **OBJ-004 DamageMaskNet — BLOCKED.** Recover attempt 3; true quality fail ends the U-Net hypothesis after three attempts.
- **OBJ-005 Broad blind-BFR selection — IN_PROGRESS.** Needs identity-disjoint DEV/VALIDATION.
- **OBJ-006 FBCNN JPEG specialist — IN_PROGRESS.** Expand JPEG families and Windows/EliteBook evidence.
- **OBJ-007 Personalized Reference Bank — IN_PROGRESS.** Validate 0/1/9, full/partial/wrong/duplicate/low-quality/multi-pose.
- **OBJ-008 RefFace CPU feasibility — BLOCKED by OBJ-004.** Attempt 0/3 consumed.
- **OBJ-009 Paper Quality Windows pack/installer — PROPOSED.** After qualification.
- **OBJ-010 HP EliteBook acceptance — PROPOSED.** Final real-PC gate.

## 6. MODEL MASTER REGISTRY

Established PRODUCT_V1 roles: **YuNet** detector; **SFace** identity gate `0.363`; **NAFNet** mild deblur/denoise; **Face Parsing ResNet18 ONNX** 19-class parser; **Head Pose MobileNetV2 ONNX**; constrained **LaMa ONNX** for non-identity-critical residuals. Their exact URLs/hashes are governed by project registry code; qualification applies only to their certified roles.

| Model | Role | State | Evidence / blocker |
|---|---|---|---|
| GPEN BFR-512 | blind face restoration | BENCHMARKING / distribution-license blocker | DEV SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474` |
| GFPGAN v1.4 | blind face restoration | BENCHMARKING | DEV SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604` |
| CodeFormer w=0.5 | severe restoration | BENCHMARKING / BLOCKED_LICENSE | real CPU slice PASS; exact artifact metrics must be reread |
| FBCNN | JPEG specialist | BENCHMARKING / DEV leader | QF20 PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, SFace `0.9571→0.9691`, `~1.305GB` |
| DamageMaskNet U-Net | 12-class damage segmentation | BENCHMARKING / BLOCKED | attempt-3 metrics NOT_VERIFIED |
| RefFaceInpainting | same-person large occlusion | FEASIBILITY_ONLY / NOT_RUN | MIT repo; minimal CPU path prepared; 0/3 attempts |
| InstantRestore | personalized multi-ref | FEASIBILITY_ONLY / BLOCKED_HARDWARE+license | 2 UNets + 2 VAEs + CLIP; CUDA/FP16 assumptions |
| OSDFace | severe blind challenger | FEASIBILITY_ONLY / BLOCKED_HARDWARE | official CUDA/device-stream path |
| RestoreFormer++ | blind challenger | FEASIBILITY_ONLY | no CFS CPU qualification |
| VQFR | blind challenger | FEASIBILITY_ONLY | no CFS CPU qualification |
| GPEN inpainting | missing-region challenger | FEASIBILITY_ONLY | license/resource unresolved |
| RefineFIR | architecture teacher | FEASIBILITY_ONLY | executable/checkpoint path inadequate |
| PerFuSe / RefIPFR | personalization teachers | FEASIBILITY_ONLY | official runnable path NOT_VERIFIED |
| Real-ESRGAN | optional x2 upscale | FEASIBILITY_ONLY | CPU/identity/background tradeoff unmeasured |

Registry documentation issue: `THIRD_PARTY_MODULES.md` references machine-readable files under `models/` not present on reconciled `main`; active registry is in `app/model_registry.py` and companion runtime/production code. Do not invent missing manifests.

## 7. CURRENT MODEL EVIDENCE

All values are DEVELOPMENT Linux CPU only.

| Model | SFace | PSNR | SSIM | Runtime | Peak RSS |
|---|---:|---:|---:|---:|---:|
| GPEN | `0.95397` | `28.07` | `0.7474` | `~2.697s` | `~1.828GB` |
| GFPGAN v1.4 | `0.91665` | `30.65` | `0.8604` | `~2.787s` | `~1.666GB` |
| FBCNN QF20 | `0.9571→0.9691` | `34.62→36.78` | `0.9486→0.9634` | evidence artifact | `~1.305GB` |
| CodeFormer | identity gate PASS | artifact required | artifact required | artifact required | artifact required |

## 8. 13-BLOCK ARCHITECTURE

1. **IMPORT:** deterministic MAIN+refs; no generator.
2. **DEBLUR:** NAFNet mild; future measured BFR candidate from common checkpoint, never blind chaining.
3. **ENHANCE:** conservative general; FBCNN for detected JPEG; low-light specialist only if qualified.
4. **LANDMARKS:** YuNet/pose support; no invented landmarks.
5. **ALIGN:** deterministic similarity/affine/RANSAC; common benchmark geometry.
6. **OCCLUSION_MASK:** parsing+heuristics; DamageMaskNet 12-class target; attempt-3 result NOT_VERIFIED.
7. **REGION_SELECT:** component bank/reference memory; V3 Personalized Reference Bank over 13 components.
8. **INPAINT:** observed refs first, constrained LaMa residuals; Paper generators only if qualified and always GENERATED.
9. **FUSION:** deterministic; healthy MAIN > observed same-person reference > accepted generated within repair authority.
10. **FRONTALIZE:** geometry-only Conservative; Paper hidden-region synthesis only as GENERATED.
11. **IDENTITY_CHECK:** SFace `0.363`; Track A direct/full/partial trusted-anchor semantics currently VALIDATING.
12. **UPSCALE:** Lanczos; Real-ESRGAN only after measured benefit.
13. **EXPORT:** deterministic provenance; future generated/component/model-selection/identity/damage/timing/RAM/hash reports.

## 9. PHOTO AND INPUT CONTRACT

MAIN domain: low-res phone/social-media, JPEG/double JPEG, defocus/motion/mixed blur, noise, pixelation/mosaic, scribble/sticker/black bar/opaque block, covered/missing components, crop/partial face, low light/uneven exposure, mixed unknown damage. MAIN defines target canvas/pose/frame in Conservative Mode.

References: MAIN + 0–9; full/partial/component-only/side-angle/different expression/light/resolution/blur/compression/occlusion/useless/wrong-person. Full accepted same-person may be global anchor; partial accepted same-person is local only; wrong-person never global anchor/donor/identity booster.

## 10. DATASET CONSTRUCTION

Initial Paper Quality target ~300–400 representative cases, with intentional female-domain proportion explicitly recorded. Identity-disjoint TRAIN / DEVELOPMENT / VALIDATION / FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain label/split/degradation/severity/seed/exact mask/reference relationships. Never tune/train on final holdout.

Current assets: early degradation-spec branch; DamageMaskNet FairFace + ControlFace bank with identity-disjoint ControlFace validation; V4 holdout excluded from Track B training.

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

13 components: LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Track visibility, damage, best/alternate refs, confidence/coverage, generated candidates, selected source/provenance, identity/geometry and unresolved state. Observed same-person evidence outranks generation.

## 12. DAMAGE ROUTING

- HEALTHY -> preserve MAIN.
- DEFOCUS/MOTION BLUR -> measured deblur/NAFNet; BFR only if needed in Paper mode.
- NOISE -> NAFNet/current measured winner.
- JPEG/DOUBLE_JPEG -> FBCNN first.
- PIXELATION/MOSAIC -> observed component reconstruction first, generated component only in Paper mode.
- SCRIBBLE/STICKER/OPAQUE/BLACK_BAR -> observed reference first; RefFace-like specialist only if qualified.
- PARTIAL_OCCLUSION/MISSING_COMPONENT -> component bank then generated fallback in Paper mode.
- LOW_LIGHT -> detected correction; Zero-DCE++ only if qualified.
- MIXED -> specialist router with minimum necessary candidates, never automatic GPEN→GFPGAN→CodeFormer chaining.

## 13. DECISION LOG

- **DEC-20260819-001 ACCEPTED:** canonical meta ledger.
- **DEC-20260819-002 ACCEPTED:** advanced research branch active; early branch preserved, not falsely merged.
- **DEC-20260819-003 ACCEPTED:** <=80% logical CPU, <=80% process/system RAM, one heavy model.
- **DEC-20260819-004 ACCEPTED:** healthy MAIN > observed same-person > accepted generated.
- **DEC-20260819-005 ACCEPTED:** DamageMaskNet attempt-3 acquisition switched to FairFace+ControlFace after 403/429 without changing U-Net hypothesis.
- **DEC-20260819-006 ACCEPTED/BLOCKED:** RefFace next large-occlusion specialist after DamageMaskNet resolution.
- **DEC-20260819-007 ACCEPTED:** V3 consumed; V4 frozen/unexecuted one-shot.
- **DEC-20260819-008 ACCEPTED DIRECTION / VALIDATING:** ranking cluster is not identity authority; direct current/preflight SFace must be preserved without transitive A-B-C rescue. Attempt 2 adds full-vs-partial and legacy-engine distinctions.

## 14. EXPERIMENT LOG

- **EXP-20260817-001 GPEN:** real Linux CPU DEV; BENCHMARKING.
- **EXP-20260817-002 GFPGAN v1.4:** comparable Linux CPU DEV; BENCHMARKING.
- **EXP-20260817-003 CodeFormer:** attempt1 packaging fail; attempt2 real CPU PASS; license blocker.
- **EXP-20260817-004 FBCNN:** real QF20 improvement; DEV JPEG leader.
- **EXP-20260818-005 DamageMaskNet 1/3:** infrastructure FAIL, Wikimedia 403.
- **EXP-20260818-006 DamageMaskNet 2/3:** infrastructure FAIL, Wikimedia 429.
- **EXP-20260818-007 DamageMaskNet 3/3:** FairFace+ControlFace, result NOT_VERIFIED; no fourth rerun for observability.
- **EXP-20260819-008 RefFace:** PREPARED / NOT_RUN; 0/3 consumed.
- **EXP-20260819-009 Track A direct-transfer hypothesis:** attempt 1/3 at `3e919f7a...` FAIL (`4 failed,102 passed`). Failures: partial same-canvas promoted globally; direct current SFace anchor lost on runtime reorder/no matrix; downstream identity wrapper lost anchor; missing `engine` label rejected otherwise structured scores.
- **EXP-20260819-010 Track A trusted-anchor refinement:** attempt 2/3, `3e919f7a... -> 7c683eddad579974329bb186622aea01feccee61`. Changes only `app/identity_anchor_v4_hardening.py` + fail-closed regressions. Rules: current flag+numeric score may survive when no valid preflight matrix; valid matrix constrains un-hardened trust to fixed direct authority; partial same-canvas with `score=None` never global; missing legacy `engine` accepted when scores are valid, but explicitly non-SFace/proxy engine rejected. SFace threshold/models unchanged. Result NOT_VERIFIED pending same-head CI.

## 15. QUALITY SCOREBOARD

DEV model values in Section 7. VALIDATION: broad BFR incomplete; DamageMaskNet attempt3 NOT_VERIFIED; reference-bank validation incomplete. HOLDOUT: V1 historical 40/40; V3 consumed 39/40; V4 frozen/unexecuted. REAL-WORLD: previous Track A head Female failed; attempt1 Female was still in progress at last check. TARGET-PC: Paper Quality NOT_RUN.

Maintain per scope: SFace/identity, PSNR, SSIM, LPIPS, NIQE where useful, healthy MAE, damage recovery, wrong-person pixels, provenance violations, generated/reference/unresolved fractions, component geometry, runtime, RAM.

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU detected at runtime. CPU-first, no CUDA requirement. Optional OpenVINO/iGPU only after actual support and parity. <=80% logical CPU, <=80% process/system RAM, one heavy model. Every serious model eventually needs real target-PC load/inference seconds, peak RAM, backend, output hash and identity result.

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed pixels `0`; provenance violations `0`; healthy/outside MAE `<=8.0` where frozen policy applies; independent calibration. No threshold/benchmark shopping, cherry-picking, consumed-holdout reruns, difficult-case deletion, generated-as-observed, raw wrong-person max-score rescue, auto-merge, force-push certified history or fabricated results.

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`. Identity similarity never changes provenance.

## 19. TRACK A — PRODUCT_V1_1

### Historical `3645c8c...`
Release Quality `4 failed,195 passed`; Windows FAIL; Female FAIL. V3 consumed. V4 frozen/unexecuted.

### Attempt 1 — `3e919f7a...`
Release Quality #124 (`32213288260`) FAIL `4 failed,102 passed`. V3/V4 verification PASS without execution. Artifact ID `9351482916`, zip SHA256 `67dc2827a9aac356d81af462f463a80a7fa8c70f466cfdc254f874022a7baf97`. Windows #1307 FAIL. Four tracked failures established the full-vs-partial and current-direct-anchor problems described in EXP-009.

### Attempt 2 — `7c683eddad579974329bb186622aea01feccee61`
Technical changes:
1. global trusted anchors require whole-face current score; face-local same-canvas `score=None` remains component-local;
2. when a valid preflight matrix exists, un-hardened trust is constrained to fixed direct authorities/direct SFace reasons;
3. when no valid preflight matrix exists, current per-slot SFace flag + finite numeric score is preserved and mapped through `runtime_source_order`, preventing loss after reorder without creating cluster trust;
4. explicit non-SFace/proxy `engine` fails closed; missing engine metadata is accepted for legacy nonempty structured score evidence; empty/missing/nonfinite scores still fail;
5. new tests explicitly protect proxy rejection and explicit SFace acceptance.

No model/checkpoint/threshold/holdout change. **CI state: NOT_VERIFIED at ledger update time.**

Next: read attempt-2 targeted result. If fail, attempt 3/3 must be a single evidence-based correction or the hypothesis stops/reassesses. If targeted PASS, run/observe full pytest and same-head Windows/Female/Release Quality. Never execute V3/V4 here.

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`: 80% governor; GPEN/GFPGAN/CodeFormer/FBCNN slices; DamageMaskNet pipeline; 13-component reference bank; reference-first repair; candidate selector; deterministic fusion; parser adapter; RefFace manual CPU workflow. Models must converge to QUALIFIED, REJECTED or documented blockers.

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt 3 without rerun/tuning. PASS -> per-class IoU/F1 + ONNX parity + RAM/runtime. Infrastructure FAIL -> fix infrastructure only. True model/data FAIL -> U-Net hypothesis exhausted, document a new lightweight architecture hypothesis. Only afterward run RefFace CPU attempt 1/3.

## 22. SPECIALIST MODEL STRATEGY

`INPUT -> detect/align -> damage classification -> reference analysis -> identity anchors -> specialist route -> candidates -> component gates -> evidence-aware fusion -> final identity -> provenance/export`.
JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference-conditioned specialist. Unsupported detail -> Paper generation only. Never blindly chain all BFR models.

## 23. MODEL SELECTION POLICY

Blind-restorer winners require multiple identity-disjoint DEV/VALIDATION cases and per-damage reporting. Identity hard gate precedes weighted quality ranking. Measure geometry, artifacts, healthy preservation, PSNR/SSIM/LPIPS, runtime/RAM. Never select on final holdout. Remove models whose incremental gain does not justify cost/dependencies/license risk.

## 24. HISTORICAL RECORD — append-only

### HIST-20260815-001 — PRODUCT_V1 certified merge
`release/v1-certified@f476c6f... -> main@2767513f...`; historical Windows #1195, Female #463, Release Quality #13.

### HIST-20260818-002 — Track A blocked snapshot
`hotfix@3645c8c...`: Release Quality FAIL (`4/195` failure/pass profile), Windows FAIL, Female FAIL; V3 consumed; V4 frozen/unexecuted.

### HIST-20260818-003 — Paper Quality snapshot
`research/paper-quality-local-v2@645862d1...`: real DEV model evidence; DamageMaskNet attempt3 NOT_VERIFIED; RefFace PREPARED/NOT_RUN.

### HIST-20260819-004 — Canonical ledger established
Created `meta/project-state` from certified main; project state no longer depends on chat memory.

### HIST-20260819-005 — Direct-transfer patch pushed
`3645c8c... -> 3e919f7a...`; preflight direct MAIN transfer authority + same-canvas alias. No threshold/model change.

### HIST-20260819-006 — Attempt-1 CI result
Exact `3e919f7a...`: Release Quality #124 FAIL `4 failed,102 passed`; Windows #1307 FAIL; Female #574 last seen in progress; V3/V4 verification-only PASS. Attempt 1/3 consumed.

### HIST-20260819-007 — Trusted-anchor attempt 2 pushed
Timestamp from Git commit: `2026-08-19T03:56:46Z`. Technical branch `hotfix/real-world-restoration-v1.1`; previous `3e919f7a1cc54e1bdb00607c2bbeece1d3392724`; exact new HEAD `7c683eddad579974329bb186622aea01feccee61`. Files: `app/identity_anchor_v4_hardening.py`, `tests/test_identity_anchor_v4_fail_closed.py`. Models/checkpoints/threshold unchanged. Objective OBJ-002. Same-head result NOT_VERIFIED at ledger update. Next action: inspect targeted CI before any attempt 3.
