# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this file before any project decision. GitHub evidence overrides chat memory. Current state is editable; Decision/Experiment/Historical records are append-only.

## 0. Document metadata

- Last ledger update: `2026-08-19T05:55+02:00`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Last technical HEAD: `3e919f7a1cc54e1bdb00607c2bbeece1d3392724`
- Active tracks: Track A PRODUCT_V1_1 stabilization; Track B Paper Quality research
- Current exact Track A blocker: attempt 1/3 at `3e919f7a...` failed Release Quality with four tracked V4 identity-policy regressions; Windows also failed; Female-domain was still `in_progress` at the last permitted check.
- Current exact Track B blocker: recover the already-triggered DamageMaskNet mixed-source attempt-3 evidence without launching attempt 4 merely for observability.

| Gate | State |
|---|---|
| FORENSIC_MODE_READY | TRUE for certified PRODUCT_V1 only |
| PAPER_QUALITY_MODE_READY | FALSE |
| WINDOWS_INSTALLER_READY | PARTIAL — V1 historical only |
| TARGET_HARDWARE_READY | FALSE |
| QUALITY_TARGET_ACHIEVED | FALSE |
| PROJECT_FINISHED | FALSE |

Mandatory sequence after every technical push:
`technical work -> tests/evidence -> commit/push -> exact remote SHA -> ledger update/push`.

## 1. Executive project summary

CFS is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative Mode prioritizes MAIN and verified same-person observed evidence with exact provenance and safe abstention. Paper Quality Mode may synthesize missing detail, but generated pixels remain `GENERATED_MODEL_INFERRED`. PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is a safety/operational hotfix. Track B is an ML research track with real CPU development evidence and specialist routing prototypes.

Current Track A direction is to separate **ranking membership** from **identity authority**. A connected SFace component may rank images, but global/component authority must come from a direct verified identity edge or another explicitly defined safe proof; transitive A-B-C agreement must not create identity authority. Current attempt 1 proved the direction needs refinement at the V4 trusted-anchor layer.

## 2. Branch and release map

| Branch | Purpose | Verified HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | signed PR #1 merge; historical Windows #1195, Female #463, Release Quality #13 | preserve |
| `feature/block-pipeline-v1` | original V1 implementation | `5eff667373cd47c07ba14aaad2acafee6d5a61c1` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | certified V1 candidate | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | FROZEN / ARCHIVED | merged via PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `3e919f7a1cc54e1bdb00607c2bbeece1d3392724` | BLOCKED / ACTIVE | PR #2 OPEN/DRAFT; Release Quality FAIL; Windows FAIL; Female last seen in progress | identity-policy attempt 2/3 |
| `research/face-restoration-v2` | early degradation/dataset research | `757a3f6081b7b152cdc615a07cd99aec40fa0a1c` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd` | ACTIVE / BENCHMARKING | research state partly NOT_VERIFIED | DamageMaskNet evidence recovery |
| `meta/project-state` | canonical ledger | intentionally not self-recorded | ACTIVE META | documentation branch | update after every technical push |

The two research branches diverged from the same certified base. The advanced branch is not a literal Git superset of the early branch; the early branch is superseded only as the active architecture.

## 3. PRODUCT VERSION ROADMAP

### PRODUCT_V1 — RELEASED
Certified conservative/forensic baseline. Safety: SFace `0.363`, wrong-person observed pixels `0`, provenance violations `0`, frozen healthy-region policy. Windows historically certified. EliteBook-specific acceptance NOT_VERIFIED.

### PRODUCT_V1_1 — IMPLEMENTING / BLOCKED
Operational real-world hotfix. Same production model family; no Track B generators. Current head `3e919f7a...`. Attempt 1/3 at the direct-transfer hypothesis failed four V4 policy tests; next action is a narrow trusted-anchor/fail-closed correction without changing SFace or provenance rules.

### PRODUCT_V2 — BENCHMARKING
Paper Quality Local: damage-aware routing, modern BFR candidates, FBCNN, explicit generated provenance, hard identity gates, deterministic fusion, <=80% CPU/system/process RAM and one heavy model at a time. Windows/EliteBook Paper Quality NOT_READY.

### PRODUCT_V3 — PLANNED with enabling prototypes
Personalized MAIN + 0–9 references. Full accepted refs may be global anchors; partial refs are component-local only; wrong-person refs are never anchor/donor/identity booster.

### PRODUCT_V4 — PLANNED with enabling prototypes
Damage-specialist hybrid: classify damage, choose specialists, score/fuse per component.

### PRODUCT_V5 — PLANNED
Unified Conservative + Paper Quality + personalized references + specialist routing + offline model pack + clean Windows installer + real target-PC acceptance.

## 4. HOLDOUT / benchmark lineage

| Set | State | Tuning rule |
|---|---|---|
| CALIBRATION_V1 | historical 60/60 at certified candidate | original calibration protocol only |
| FINAL_HOLDOUT_V1 | historical 40/40 | consumed for certification; no tuning |
| FINAL_HOLDOUT_V2 | details NOT_VERIFIED in this reconciliation | do not tune until protocol recovered |
| FINAL_HOLDOUT_V3 | **CONSUMED**, 39/40; `medium_block_mosaic`, SFace `0.360 < 0.363` | NEVER rerun/tune |
| FINAL_HOLDOUT_V4 | 40 cases / 20 identities; frozen ControlFace10K; **NOT_RUN / UNCONSUMED** | never tune; execute only through valid one-shot sequence |
| FINAL_HOLDOUT_V5 | not created | future independent final set |
| Female-domain | ~300–400 quick cases | quality report-only; safety hard gates apply |
| Paper Quality DEV | partial | model/router development allowed |
| Paper Quality VALIDATION | incomplete | independent selection validation |
| DamageMaskNet bank | FairFace + ControlFace, exact synthetic masks | TRAIN/VALIDATION only |

V4 freeze/history was reverified at `3e919f7a...`; frozen blob SHAs matched the original freeze commit `ad564c9b1cd9514250eac08425d16c2414ead9fa`. V4 was not executed.

## 5. CURRENT GLOBAL OBJECTIVES

- **OBJ-001 — Preserve PRODUCT_V1 — PASS.** Never rewrite certified history.
- **OBJ-002 — Restore PRODUCT_V1_1 gates — BLOCKED.** Success = targeted tests + full pytest + same-HEAD Windows/Female/Release Quality, zero wrong-person/provenance violations. Attempt 1/3 failed 4/106 targeted tests at `3e919f7a...`; next = trusted-anchor attempt 2/3.
- **OBJ-003 — Maintain canonical ledger — IN_PROGRESS.** This update records the attempt-1 CI result before another technical push.
- **OBJ-004 — Resolve DamageMaskNet — BLOCKED.** Recover attempt 3; PASS requires per-class IoU/F1, ONNX parity, RAM/runtime. True model/data fail exhausts U-Net hypothesis.
- **OBJ-005 — Broad blind-BFR selection — IN_PROGRESS.** Identity-disjoint DEV/VALIDATION required.
- **OBJ-006 — Qualify FBCNN JPEG specialist — IN_PROGRESS.** Expand QF/double-JPEG/social-media/resize+JPEG/JPEG+blur and Windows/EliteBook.
- **OBJ-007 — Validate Personalized Reference Bank — IN_PROGRESS.** 0/1/9, full/partial/wrong/duplicate/low-quality/multi-pose.
- **OBJ-008 — RefFace CPU feasibility — BLOCKED by OBJ-004.** Attempt 0/3 consumed.
- **OBJ-009 — Paper Quality Windows pack/installer — PROPOSED.** After model/router qualification.
- **OBJ-010 — HP EliteBook acceptance — PROPOSED.** Final target-PC gate.

## 6. MODEL MASTER REGISTRY

Established V1: **YuNet** (detector), **SFace** (identity `0.363`), **NAFNet** (mild deblur/denoise), **Face Parsing ResNet18 ONNX** (19 classes), **Head Pose MobileNetV2 ONNX**, constrained **LaMa ONNX**. These are qualified only for their certified/current roles, not every possible use.

Paper Quality / specialist states:

| Model | Role | State | Measured / blocker |
|---|---|---|---|
| GPEN BFR-512 | blind face restoration | BENCHMARKING / distribution-license blocker | DEV SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474` |
| GFPGAN v1.4 | blind face restoration | BENCHMARKING | DEV SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604` |
| CodeFormer w=0.5 | severe restoration | BENCHMARKING / BLOCKED_LICENSE | real CPU slice PASS; exact comparative artifact metrics must be reread, not guessed |
| FBCNN | JPEG specialist | BENCHMARKING / current DEV JPEG leader | QF20 PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, SFace `0.9571→0.9691`, `~1.305GB` |
| DamageMaskNet U-Net | 12-class damage mask | BENCHMARKING / BLOCKED | attempt-3 metrics NOT_VERIFIED |
| RefFaceInpainting | same-person large occlusion | FEASIBILITY_ONLY / NOT_RUN | MIT repo; prepared minimal CPU path; attempt 0/3 |
| InstantRestore | multi-ref personalized | FEASIBILITY_ONLY / BLOCKED_HARDWARE+license audit | 2 UNets + 2 VAEs + CLIP, CUDA/FP16 assumptions |
| OSDFace | severe blind challenger | FEASIBILITY_ONLY / BLOCKED_HARDWARE | official CUDA/device-stream path |
| RestoreFormer++ | blind challenger | FEASIBILITY_ONLY | no CFS CPU qualification |
| VQFR | blind challenger | FEASIBILITY_ONLY | no CFS CPU qualification |
| GPEN inpainting | missing-region challenger | FEASIBILITY_ONLY | license/resource unresolved |
| RefineFIR | architecture teacher | FEASIBILITY_ONLY | executable/checkpoint path inadequate |
| PerFuSe / RefIPFR | personalization teachers | FEASIBILITY_ONLY | executable official runtime NOT_VERIFIED |
| Real-ESRGAN | optional x2 upscale | FEASIBILITY_ONLY | CPU/identity/background tradeoff unmeasured |

Registry note: certified `THIRD_PARTY_MODULES.md` refers to files under `models/` that are not present on reconciled `main`; active registry logic is in `app/model_registry.py` and related runtime/production registries. Do not invent missing manifests.

## 7. CURRENT MODEL EVIDENCE

All values below are DEVELOPMENT Linux CPU evidence, never Windows/EliteBook qualification.

| Model | SFace | PSNR | SSIM | Runtime | Peak RSS |
|---|---:|---:|---:|---:|---:|
| GPEN | `0.95397` | `28.07` | `0.7474` | `~2.697s` | `~1.828GB` |
| GFPGAN v1.4 | `0.91665` | `30.65` | `0.8604` | `~2.787s` | `~1.666GB` |
| FBCNN QF20 | `0.9571→0.9691` | `34.62→36.78` | `0.9486→0.9634` | evidence artifact | `~1.305GB` |
| CodeFormer | identity gate PASS | artifact required | artifact required | artifact required | artifact required |

## 8. 13-BLOCK ARCHITECTURE

| # | Block | Current | Future / status |
|---:|---|---|---|
| 1 | IMPORT | deterministic MAIN+refs | retain; richer source/hash metadata |
| 2 | DEBLUR | NAFNet mild | common checkpoint -> measured NAFNet/BFR candidate; never blind-chain |
| 3 | ENHANCE | conservative general | FBCNN for detected JPEG; low-light specialist only if qualified |
| 4 | LANDMARKS | YuNet/pose support | hard-pose alternatives only if measured |
| 5 | ALIGN | deterministic similarity/affine/RANSAC | common benchmark alignment |
| 6 | OCCLUSION_MASK | parsing + heuristics | DamageMaskNet 12-class; attempt-3 result NOT_VERIFIED |
| 7 | REGION_SELECT | component bank/reference memory | Personalized Reference Bank over 13 components |
| 8 | INPAINT | observed refs first; constrained LaMa residuals | Paper-mode RefFace/BFR only if qualified; GENERATED provenance |
| 9 | FUSION | deterministic/regional | healthy MAIN > observed ref > accepted generated within authority |
| 10 | FRONTALIZE | geometry-only Conservative | Paper synthesis only as GENERATED |
| 11 | IDENTITY_CHECK | SFace `0.363`, V2/V4 hardening | Track A trusted-anchor semantics currently BLOCKED |
| 12 | UPSCALE | Lanczos | optional Real-ESRGAN x2 only after benchmark |
| 13 | EXPORT | deterministic provenance/evidence | add generated/component/model-selection/identity/damage/timing/RAM/hash reports |

## 9. PHOTO AND INPUT CONTRACT

MAIN target domain: low-resolution phone/social-media images, JPEG/double JPEG, blur/noise, pixelation/mosaic, scribbles/stickers/black bars/opaque blocks, covered/missing eye/nose/mouth, crop/partial face, low light/uneven exposure, mixed unknown damage. MAIN is target canvas/pose/frame in Conservative Mode.

References: MAIN + 0–9; full/partial/eye-only/mouth-only/nose-only/side-angle/different expression/light/resolution/blur/compression/occlusion/useless/wrong-person. Full accepted same-person may be a global anchor; partial accepted same-person is component-local; wrong-person is never global anchor, donor or identity-score improver.

## 10. DATASET CONSTRUCTION

Paper Quality planning target: ~300–400 representative cases initially, with explicit female-domain percentage when intentionally emphasized. Protected identity partitions: TRAIN / DEVELOPMENT / VALIDATION / FINAL_HOLDOUT. Per example store source/license/date/identity/hash/resolution/domain label/split/degradation/severity/seed/mask/reference relationships. Never train/tune on final holdout.

Current banks: early `research/face-restoration-v2` degradation specification; DamageMaskNet FairFace + ControlFace source bank with identity-disjoint ControlFace validation; V4 holdout excluded from Track B training/validation.

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Canonical 13: `LEFT_EYE`, `RIGHT_EYE`, `LEFT_EYEBROW`, `RIGHT_EYEBROW`, `NOSE`, `PHILTRUM`, `MOUTH_LIPS`, `LEFT_CHEEK`, `RIGHT_CHEEK`, `CHIN`, `JAW`, `FOREHEAD`, `FACE_CONTOUR`.

Per component track MAIN visibility/damage, best/alternate observed refs, reference confidence/coverage, generated candidates, selected source/provenance, identity/geometry consistency and unresolved state. Observed same-person evidence outranks generation.

## 12. DAMAGE ROUTING

| Damage | Primary | Secondary / specialist | Conservative fallback |
|---|---|---|---|
| HEALTHY | preserve MAIN | none | unchanged |
| DEFOCUS/MOTION BLUR | NAFNet / measured deblur | accepted BFR if severe in Paper mode | rollback/unresolved |
| NOISE | NAFNet/current winner | specialist if measured | preserve healthy |
| JPEG/DOUBLE_JPEG | FBCNN | BFR only for residual facial damage | avoid unnecessary generation |
| PIXELATION/MOSAIC | observed component reconstruction | generated component in Paper mode | unresolved |
| SCRIBBLE/STICKER/OPAQUE/BLACK_BAR | observed reference first | RefFace-like specialist if qualified | unresolved/safe abstention |
| PARTIAL_OCCLUSION/MISSING_COMPONENT | observed component bank | generated component if Paper mode | unresolved |
| LOW_LIGHT | detected conservative correction | Zero-DCE++ only if qualified | no blanket brightening |
| MIXED | multi-class specialist router | minimum needed candidates | never chain all generators |

## 13. DECISION LOG

- **DEC-20260819-001 ACCEPTED:** canonical `meta/project-state/PROJECT_MASTER_STATE.md`.
- **DEC-20260819-002 ACCEPTED:** advanced Paper Quality branch is active architecture; early research branch preserved, not falsely merged.
- **DEC-20260819-003 ACCEPTED:** <=80% logical CPU, <=80% process/system RAM, max one heavy model.
- **DEC-20260819-004 ACCEPTED:** healthy MAIN > verified observed same-person reference > accepted generated candidate.
- **DEC-20260819-005 ACCEPTED:** DamageMaskNet attempt 3 changed acquisition to FairFace+ControlFace after Wikimedia 403/429 while preserving U-Net hypothesis.
- **DEC-20260819-006 ACCEPTED/BLOCKED:** RefFace is next large-occlusion specialist after DamageMaskNet resolution.
- **DEC-20260819-007 ACCEPTED:** V3 consumed; V4 remains frozen/unexecuted one-shot.
- **DEC-20260819-008 ACCEPTED DIRECTION / IMPLEMENTATION REASSESS:** ranking component is not identity authority; direct MAIN identity evidence must survive ranking selection; no transitive A-B-C trust. Attempt 1 exposed further trusted-anchor distinctions to fix.

## 14. EXPERIMENT LOG

- **EXP-20260817-001 GPEN:** real Linux CPU DEV; result in Section 7; BENCHMARKING.
- **EXP-20260817-002 GFPGAN v1.4:** comparable real Linux CPU DEV; BENCHMARKING.
- **EXP-20260817-003 CodeFormer:** attempt 1 packaging fail, attempt 2 real CPU PASS; license blocker.
- **EXP-20260817-004 FBCNN:** real JPEG QF20 improvement; current DEV JPEG leader.
- **EXP-20260818-005 DamageMaskNet 1/3:** infrastructure FAIL, Wikimedia 403 before training.
- **EXP-20260818-006 DamageMaskNet 2/3:** infrastructure FAIL, Wikimedia 429 before training completion.
- **EXP-20260818-007 DamageMaskNet 3/3:** FairFace+ControlFace; result NOT_VERIFIED; no attempt 4 for observability.
- **EXP-20260819-008 RefFace:** PREPARED / NOT_RUN; 0/3 attempts consumed.
- **EXP-20260819-009 Track A direct-transfer hypothesis, attempt 1/3:** `3645c8c... -> 3e919f7a...`; no model/threshold change. **FAIL**: Release Quality `4 failed, 102 passed`. Failure families: partial same-canvas incorrectly became global trusted source; direct current SFace source was lost after runtime reorder/no preflight matrix; identity wrapper consequently supplied too few anchors; nonempty legacy score evidence was rejected solely because `engine` metadata was absent. Windows also FAIL. Female last seen in progress. Next = attempt 2/3 at trusted-anchor/fail-closed semantics.

## 15. QUALITY SCOREBOARD

DEV: model values in Section 7. VALIDATION: broad BFR incomplete; DamageMaskNet attempt 3 NOT_VERIFIED; reference-bank broad validation incomplete. HOLDOUT: V1 historical 40/40; V3 consumed 39/40; V4 frozen/not-run. REAL-WORLD: Track A current-head Female still not final at last check. TARGET-PC: Paper Quality NOT_RUN.

Maintain per scope: identity/SFace, PSNR, SSIM, LPIPS, NIQE where useful, healthy MAE, damage recovery, wrong-person pixels, provenance violations, generated/reference-supported/unresolved fraction, component geometry, runtime, RAM.

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows. Detect exact CPU/GPU at runtime. CPU-first, no CUDA requirement. Optional acceleration only after actual support and parity validation. Resource contract <=80% logical CPU, <=80% process/system RAM, one heavy model. Every serious model eventually needs target-PC load time, inference time, peak RAM, backend, output hash and identity result.

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed pixels `0`; provenance violations `0`; healthy/outside MAE `<=8.0` where frozen protocol applies; independent calibration. Forbidden: threshold-shopping, cherry-picking, consumed-holdout reruns, deleting hard failures, generated-as-observed, wrong-person max-score rescue, auto-merge, force-push certified history, fabricated results.

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`. Passing identity never changes provenance class.

## 19. TRACK A — PRODUCT_V1_1

### Historical head `3645c8c...`
Release Quality `4 failed, 195 passed`; Windows FAIL; Female FAIL. Old failures centered on missing component-transfer persistence, reuse of preflight acceptance and same-canvas bridge evidence.

### Attempt-1 head `3e919f7a...`
Technical intent: persist direct MAIN↔REF transfer authority separately from ranking clusters and expose compatible strict same-canvas bridge keys. No threshold/model change.

Exact Release Quality #124 (`32213288260`) evidence:
- V3 manifests verification PASS; V3 not executed.
- V4 freeze/history verification PASS; V4 not executed.
- targeted suite: **4 failed, 102 passed**; full pytest skipped.
- failure 1: `test_partial_same_canvas_sheet_never_becomes_global_identity_anchor` expected trusted `{1}`, got `{1,2}`. Partial same-canvas source with no SFace score was promoted globally.
- failure 2: `test_final_identity_anchors_resolve_original_sources_after_runtime_reordering` expected source `[3]`, got `[]`. Current direct verified SFace source was lost when preflight matrix was absent/reordered.
- failure 3: `test_identity_wrapper_restores_runtime_refs_when_identity_handler_raises` expected immutable MAIN + source1 anchors, received only immutable MAIN; same direct-source loss root cause.
- failure 4: `test_nonempty_proxy_or_sface_scores_remain_valid_evidence` rejected nonempty structured scores solely because `engine` was absent. Empty scores and explicit proxy must still fail closed, but legacy structured nonempty evidence cannot require an engine label that older valid handlers did not emit.
- Release Quality artifact ID `9351482916`, zip SHA256 `67dc2827a9aac356d81af462f463a80a7fa8c70f466cfdc254f874022a7baf97`.
- Windows #1307 also FAIL; inspect only if its failure is not explained by the same tracked suite.
- Female #574 was still in progress at the last check; do not poll immediately.

**Attempt 2/3 exact action:**
1. do not unconditionally add same-canvas sources to global trusted anchors; partial same-canvas remains component-local;
2. preserve already-current direct SFace trusted sources even when no preflight matrix is available, with original-source mapping preserved across runtime reorder;
3. accept nonempty structured score evidence when `engine` is absent for legacy compatibility, but reject explicit proxy/non-SFace engines and empty/missing scores;
4. run tracked targeted tests, then full pytest and same-head release gates.

No V3/V4 execution.

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`. Real research layers: 80% governor, GPEN/GFPGAN/CodeFormer/FBCNN slices, damage taxonomy/DamageMaskNet, 13-component reference bank, reference-first repair, candidate selector, deterministic component fusion, parser adapter, RefFace manual CPU workflow. Models must converge to QUALIFIED, REJECTED or documented blockers.

## 21. CURRENT PAPER QUALITY BLOCKER

Recover already-triggered DamageMaskNet attempt 3 without rerun/tuning. PASS -> IoU/F1 per class, ONNX parity, RAM/runtime. Infrastructure FAIL -> fix infrastructure only. True model/data FAIL -> U-Net hypothesis exhausted; start a new documented lightweight segmentation hypothesis. Only then RefFace attempt 1/3.

## 22. SPECIALIST MODEL STRATEGY

`INPUT -> detect/align -> damage classification -> reference analysis -> identity anchors -> specialist route -> candidates -> component gates -> evidence-aware fusion -> final identity -> provenance/export`.
JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference-conditioned specialist. Unsupported missing detail -> Paper Quality generation only. Never blindly chain GPEN→GFPGAN→CodeFormer.

## 23. MODEL SELECTION POLICY

Blind-restorer winners require multiple identity-disjoint DEV/VALIDATION cases and per-degradation reporting. Identity is a hard gate before weighted ranking. Measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM. Never select from final holdout. Remove complexity that does not earn measured benefit.

## 24. HISTORICAL RECORD — append-only

### HIST-20260815-001 — PRODUCT_V1 certified merge
`release/v1-certified@f476c6f...` -> `main@2767513f...`; merge records Windows #1195, Female #463, Release Quality #13. Certified base frozen.

### HIST-20260818-002 — Track A blocked snapshot
`hotfix@3645c8c...`: Release Quality FAIL (`4 failed,195 passed`), Windows FAIL, Female FAIL; V3 consumed; V4 frozen/unexecuted.

### HIST-20260818-003 — Paper Quality snapshot
`research/paper-quality-local-v2@645862d1...`: real DEV model evidence; DamageMaskNet attempt 3 NOT_VERIFIED; RefFace PREPARED/NOT_RUN.

### HIST-20260819-004 — Canonical ledger established
Created `meta/project-state` from certified main; repository state no longer depends on chat memory.

### HIST-20260819-005 — Track A direct-transfer patch pushed
`3645c8c39653d04616167e881adaf28d2b93cd45 -> 3e919f7a1cc54e1bdb00607c2bbeece1d3392724`. Files: `app/preflight.py`, `app/face_domain_guard_v2_policy.py`, `app/primary_anchor_policy.py`, new direct-transfer regression test. SFace/model weights unchanged. At push time CI was NOT_VERIFIED.

### HIST-20260819-006 — Track A attempt-1 CI result
On exact technical HEAD `3e919f7a...`: Release Quality #124 FAIL with `4 failed,102 passed`; V3/V4 verification PASS without execution; Windows #1307 FAIL; Female #574 still in progress at last check. Four failures narrowed to partial same-canvas global trust, loss of current direct-SFace anchor under reorder/no matrix, downstream anchor-list consequence, and overly strict missing-engine rejection. Attempt 1/3 consumed. Next technical action is attempt 2/3 with no threshold/model change.
