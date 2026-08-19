# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current state is maintained; important decisions, experiments, failures and technical pushes are preserved.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Previous technical HEAD: `60b796581feb7a9c6fecd3a20a95759da4e48aa5`
- Last technical HEAD: `5240eaecb8943244f5bf7276a0905489d261318b`
- Active Paper Quality HEAD: `research/paper-quality-local-v2@d4f09f2ba7d6862f2be818c0421073007a26f885`
- Track A identity/source/provenance targeted suite: latest exact evidence `108/108 PASS` on Release Quality #132 at `60b79658...`.
- Current Track A gate: exact-head CI on `5240eaec...` is **IN_PROGRESS**. This HEAD changes only the legacy IMPORT/preflight ordering test so it no longer depends on the unrelated V4 biometric firewall. SFace `0.363`, face-local identity, provenance, models and holdouts are unchanged.
- Current Track B direction: **UPSTREAM-FIRST**. Official executable paper/model repositories are the architecture baseline; CFS owns only thin adapters, identity/provenance safety, resource control, checkpoint/hash verification, Windows/offline packaging and qualification tests.
- Current Track B blocker retained: recover already-triggered DamageMaskNet attempt-3 evidence without rerun for observability. Upstream qualification work proceeds independently and does not alter V1.1.

FORENSIC_MODE_READY: **TRUE for certified PRODUCT_V1 only**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL — historical PRODUCT_V1 only**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No auto-merge or certified-history force push.

---

## 1. Executive project summary

CFS is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative Mode preserves MAIN pose/composition/geometry and uses verified observed evidence with explicit provenance. Paper Quality Mode may generate unsupported detail only as `GENERATED_MODEL_INFERRED`.

PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is an operational/safety hotfix isolated from Track B generative models. Track B contains real CPU model evidence and damage/reference/fusion research.

Identity authority is no longer the current blocker. The same-canvas architecture separates **whole-canvas sameness**, which is only geometry/canvas evidence, from the stricter face-local identity bridge. V4 also remains fail-closed: histogram/proxy similarity is not identity authority, direct SFace evidence cannot be propagated transitively, and wrong-person same-canvas donors are blocked.

Paper Quality no longer treats local CFS reimplementation of published architectures as the default. When an official executable upstream exists, the official repository is pinned and reused; CFS patches only compatibility/integration defects and must still prove the result under its own safety and target-PC gates.

---

## 2. Branch and release map

| Branch | Purpose | HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f...` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | V1 history | `5eff6673...` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate history | `f476c6f0...` | FROZEN / ARCHIVED | merged PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `5240eaec...` | VALIDATING / ACTIVE | PR #2 OPEN/DRAFT; Windows #1316, Release #133, Female #583 in progress at ledger update | finish same-head prerequisites; never run V4 early |
| `research/face-restoration-v2` | early data/degradation research | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `d4f09f2b...` | ACTIVE / BENCHMARKING | upstream-first registry/bootstrap added; qualification partly NOT_VERIFIED | validate registry/tests, then qualify pinned upstream models sequentially |
| `meta/project-state` | canonical ledger | self-SHA omitted | ACTIVE META | docs only | update after every technical push |

---

## 3. PRODUCT VERSION ROADMAP

- **PRODUCT_V1 — RELEASED:** certified conservative baseline, SFace `0.363`, wrong-person observed `0`, provenance violations `0`.
- **PRODUCT_V1_1 — VALIDATING/BLOCKED:** operational hotfix; exact same-head prerequisites are not all green yet.
- **PRODUCT_V2 — BENCHMARKING:** Paper Quality Local with measured BFR/JPEG specialists, hard gates, generated provenance, 80% resource contract and pinned official upstream implementations.
- **PRODUCT_V3 — PLANNED with prototypes:** personalized MAIN+0–9 refs, per-component authority.
- **PRODUCT_V4 — PLANNED with prototypes:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified modes + offline model pack + installer + clean Windows + real HP EliteBook acceptance.

Product and holdout versions remain separate.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

CALIBRATION_V1 historical 60/60. FINAL_HOLDOUT_V1 historical 40/40. FINAL_HOLDOUT_V2 details not fully re-reconciled. FINAL_HOLDOUT_V3 **CONSUMED** 39/40; mosaic SFace `0.360<0.363`; NEVER rerun/tune. FINAL_HOLDOUT_V4 frozen 40 cases/20 identities, **NOT_RUN/UNCONSUMED**, one-shot only. V5 not created. Female-domain ~300–400 stress cases. Paper Quality DEV/VALIDATION separate. DamageMaskNet bank FairFace+ControlFace TRAIN/VALIDATION only.

V3 is verification-only. V4 may be executed once only after frozen request protocol and all exact same-head prerequisites pass. No V4 request or CONSUMED marker has been authorized by this ledger update.

---

## 5. CURRENT GLOBAL OBJECTIVES

OBJ-001 Preserve V1 — PASS.  
OBJ-002 Restore V1.1 gates — VALIDATING on exact HEAD `5240eaec...`.  
OBJ-003 Canonical ledger — IN_PROGRESS.  
OBJ-004 DamageMaskNet — BLOCKED on existing attempt3 evidence.  
OBJ-005 Broad BFR selection — IN_PROGRESS, now upstream-first.  
OBJ-006 FBCNN JPEG qualification — IN_PROGRESS, official upstream pinned.  
OBJ-007 Personalized Reference Bank validation — IN_PROGRESS.  
OBJ-008 RefFace CPU — BLOCKED by OBJ-004, 0/3 attempts.  
OBJ-009 Paper Quality Windows pack — PROPOSED.  
OBJ-010 HP EliteBook acceptance — PROPOSED.  
OBJ-011 Official upstream implementation registry — IMPLEMENTED, qualification tests pending on target environments.

---

## 6. MODEL MASTER REGISTRY

Certified roles: YuNet, SFace `0.363`, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research: GPEN BENCHMARKING/license blocker; GFPGAN1.4 BENCHMARKING; CodeFormer BENCHMARKING/BLOCKED_LICENSE; FBCNN BENCHMARKING/current DEV JPEG leader; DamageMaskNet BENCHMARKING/BLOCKED; RefFace FEASIBILITY_ONLY/NOT_RUN; InstantRestore/OSDFace hardware-blocked feasibility; RestoreFormer++/VQFR/GPEN-inpainting/RefineFIR/PerFuSe/RefIPFR/Real-ESRGAN feasibility until measured.

Official repository registry now exists at `config/upstream-implementations.json` on the Paper Quality branch. Pinned and verified repository baselines at creation: GPEN `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`; GFPGAN `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`; CodeFormer `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`; FBCNN `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`; InstantRestore `snap-research/InstantRestore@05891bf7d30ab7290c501272de7a1a4a51b21b4f`. RefineFIR, RefFaceInpainting, OSDFace and RestoreFormer++ repositories are registered but remain `NOT_VERIFIED` until a revision/checkpoint/license/runtime qualification is completed.

Registry documentation/package-manifest mismatch from V1 remains separate; never invent missing manifests/hashes.

---

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEV only: GPEN SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, `~1.828GB`; GFPGAN1.4 SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`; FBCNN QF20 SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, `~1.305GB`; CodeFormer real CPU slice PASS, exact metrics artifact-required.

These CFS measurements are distinct from paper-reported metrics. An upstream repository being official does not make its paper numbers reproduced on the HP EliteBook; target-PC results remain NOT_RUN until measured.

---

## 8. 13-BLOCK ARCHITECTURE

1 IMPORT deterministic. 2 DEBLUR NAFNet/measured BFR later. 3 ENHANCE FBCNN for JPEG. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK parser + DamageMaskNet target. 7 REGION_SELECT component/reference bank. 8 INPAINT observed first, Paper generation only as GENERATED. 9 FUSION MAIN > observed ref > generated. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`, direct/non-transitive and proxy-fail-closed. 12 UPSCALE Lanczos/optional measured SR. 13 EXPORT deterministic provenance/model/resource evidence.

---

## 9. PHOTO AND INPUT CONTRACT

MAIN supports low-res phone/social-media, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black bar/opaque loss, missing components, crop/partial, low light, mixed damage. References MAIN+0–9 full/partial/component-only/angle/expression/light/resolution/degraded/useless/wrong-person. Full accepted same-person may global-anchor; partial local only; wrong-person never anchor/donor/score booster.

---

## 10. DATASET CONSTRUCTION

Paper Quality target initially ~300–400 representative cases with explicit female-domain percentage. Identity-disjoint TRAIN/DEV/VALIDATION/FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/mask/reference relationships. Never train/tune on final holdout.

---

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

13 components: left/right eye, left/right eyebrow, nose, philtrum, mouth/lips, left/right cheek, chin, jaw, forehead, face contour. Track MAIN visibility/damage, refs/confidence, generated candidates, selected source/provenance, identity/geometry, unresolved state. Observed same-person outranks generation.

---

## 12. DAMAGE ROUTING

HEALTHY preserve MAIN. BLUR NAFNet/measured BFR. JPEG FBCNN. PIXELATION/MOSAIC observed component first then Paper generation. SCRIBBLE/STICKER/OPAQUE/BLACK_BAR observed reference then qualified reference specialist. PARTIAL/MISSING component bank then Paper fallback. LOW_LIGHT specialist only when detected. MIXED minimal specialist set; never blind-chain generators.

---

## 13. DECISION LOG

DEC-001 canonical ledger ACCEPTED. DEC-002 active Paper Quality branch ACCEPTED. DEC-003 <=80% CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet bank ACCEPTED. DEC-006 RefFace after DamageMaskNet ACCEPTED/BLOCKED. DEC-007 V3 consumed/V4 untouched ACCEPTED. DEC-008 ranking cluster != identity authority ACCEPTED/CLOSED after 3/3. DEC-009 localized-damage same-canvas edge isolation ACCEPTED as architectural separation: broad canvas is not identity authority; face-local/SFace remains the identity gate. **DEC-010 official-upstream-first model integration — ACCEPTED:** if official executable paper/model code exists, reuse and pin that source rather than reimplementing the architecture. Upstream code is not assumed bug-free; CFS changes are limited to compatibility/adapters/safety/resource/package integration and must pass independent tests.

---

## 14. EXPERIMENT LOG

DamageMaskNet 1/3 403 infra fail, 2/3 429 infra fail, 3/3 NOT_VERIFIED; no attempt4 for observability. RefFace PREPARED/NOT_RUN.

Identity hypothesis attempts 1–3 closed; targeted suite reached 108/108 PASS at `9b8810ce...` and again on Release Quality #132 at `60b79658...`.

**EXP-20260819-012 DEC-009 attempt1:** `9b8810ce... -> 2fcaeb1b...`; strict Lab + local mismatch dilation + minimum stable-edge support. Release Quality #129 targeted `1 failed,107 passed`. Sole failure `test_shared_background_cannot_become_identity_bridge_when_face_region_differs`: broad same-canvas expected TRUE but matcher returned FALSE because attempt1's stable-edge minimum rejected a shared canvas whose only informative edges were inside the deliberately changed face. Separate face-local test expects FALSE and remains the identity safety boundary. Artifact `9364721505`, digest `37a71c20ca44af86d2a4e6f839246f0d34ba9287563f8082e6641f747978eb0c`.

**EXP-20260819-013 DEC-009 attempt2:** `2fcaeb1b... -> 49af8cb1...`; strict global Lab rule retained, attempt-1 stable-edge survival requirement removed, face-local identity/SFace/provenance unchanged.

**EXP-20260819-014 Female #580 on `49af8cb1...`:** 76 resolved portraits, 380 executed cases, 125 runtime errors, 22 safe abstentions, 233 completed restorations. Target95 report-only `21/179 = 11.7%`. Error root classes: 77 proxy-not-SFace authority, 41 no biometric anchor, 4 no usable SFace comparisons, only 3 true below-threshold SFace failures. `mosaic_single` was `38/38` runtime error and has zero references by benchmark design. Interpretation: the dominant blocker is lack/propagation of real SFace evidence on severe no-reference cases, not a reason to lower `0.363` or re-enable proxy authority.

**EXP-20260819-015 infrastructure repair series:** `49af8cb1... -> c56e7fbf... -> 60b79658... -> 5240eaec...`. `c56e7fbf...` attempted to isolate an IMPORT test with synthetic SFace evidence; `60b79658...` made V4 frozen-blob history verification shallow-clone safe while preserving exact blob pins. Release Quality #132 on `60b79658...` proved V3 verify-only PASS, V4 freeze/blob verification PASS, targeted identity/source/provenance `108/108 PASS`; full pytest `545 passed, 1 failed`, solely the IMPORT/preflight ordering test still crossing the V4 identity wrapper. `5240eaec...` scopes that test directly to IMPORT-before-preflight ordering, with no production identity change. Same-head CI is IN_PROGRESS at this ledger update.

**EXP-20260819-016 upstream-first Track B:** `research/paper-quality-local-v2` advanced to `d4f09f2b...`. Added machine-readable official-upstream registry, offline validator, pinned detached-checkout bootstrap, and tests. Pinned initial baselines: GPEN, GFPGAN, CodeFormer, FBCNN, InstantRestore. Unpinned specialist repositories remain NOT_VERIFIED and cannot bootstrap. No Paper model was promoted to production by this change.

---

## 15. QUALITY SCOREBOARD

DEV evidence exists; broad validation incomplete. V3 consumed 39/40. V4 frozen/unexecuted. Target-PC Paper Quality NOT_RUN. Female #580 Target95 report-only `11.7%`; therefore QUALITY_TARGET_ACHIEVED remains FALSE. Maintain DEV/VALIDATION/HOLDOUT/REAL-WORLD/TARGET-PC separately.

---

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU runtime-detected. CPU-first/no CUDA. <=80% logical CPU, <=80% process/system RAM, one heavy model. Optional acceleration only after support/parity evidence.

---

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed `0 pixels`; provenance violations `0`; healthy/outside MAE `<=8.0` where frozen policy applies. No threshold shopping, cherry-picking, consumed-holdout rerun, hard-case deletion, generated-as-observed, wrong-person score rescue, proxy-as-SFace authority, auto-merge, force-push certified history or fabricated evidence.

---

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

---

## 19. TRACK A — PRODUCT_V1_1

Previous safety architecture established immutable MAIN, direct/non-transitive SFace trust, face-local same-canvas identity bridge, wrong-person pixel-donor firewall, exact-head CI and V4 one-shot authority.

At `60b796581feb7a9c6fecd3a20a95759da4e48aa5`, Release Quality #132: V3 verification-only PASS; V4 freeze + pinned blob/origin verification PASS; targeted identity/source/provenance `108/108 PASS`; full pytest `545 passed, 1 failed`. The sole failure was a legacy test whose stated purpose is immutable IMPORT ordering but which ran the full pipeline into the V4 biometric firewall.

Current exact technical HEAD `5240eaecb8943244f5bf7276a0905489d261318b`: only `tests/test_automatic.py` was changed from the `60b...` state, scoping `test_preflight_cannot_mutate_true_import_snapshot` to `IMPORT -> preflight` ordering. Production code, SFace threshold, V4 manifests and holdout protocol were not changed. Exact-head Windows #1316, Release Quality #133 and Female #583 were all IN_PROGRESS at ledger update.

Next exact action: wait for those already-selected same-head runs. On any failure, inspect that exact run; do not blind-rerun. If Windows + Release + Female all PASS on the same candidate, only then the V4 certification-request protocol may become eligible. If V4 is eventually consumed FAIL, never rerun it; freeze V5 before tuning.

---

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@d4f09f2ba7d6862f2be818c0421073007a26f885`: real CPU BFR/JPEG evidence, 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, hard-gated selector, deterministic fusion, parser adapter, RefFace workflow, plus official-upstream registry/bootstrap.

`config/upstream-implementations.json` encodes the upstream-first contract. `scripts/verify_upstream_implementation_registry.py` rejects malformed/unpinned qualification states. `scripts/bootstrap_pinned_upstream.py` checks out the exact official Git revision detached and research-only; it refuses unpinned models and requires explicit research acceptance. `tests/test_upstream_implementation_registry.py` covers policy, exact FBCNN pin, explicit research opt-in and unpinned refusal. These tests are committed but not yet claimed PASS on the target PC.

PDF constraints remain: separate global identity from local texture; use correspondence between matching regions; region-adaptive identity guidance for severe BFR; MAIN preserves pose/composition/expression/geometry; unsupported detail remains conservative. Paper-reported metrics and CFS-reproduced metrics remain separate.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1, ONNX parity, RAM/runtime. Infrastructure fail -> infrastructure-only. True model/data fail -> U-Net hypothesis ends. Then RefFace attempt1/3.

In parallel, upstream qualification may proceed without altering that observability rule: first FBCNN/GPEN/GFPGAN/CodeFormer on exact pinned source, then InstantRestore if CPU/Windows feasibility is credible. RefineFIR/RefFaceInpainting/OSDFace/RestoreFormer++ remain NOT_VERIFIED until exact revision/checkpoint/license/runtime evidence exists.

---

## 22. SPECIALIST MODEL STRATEGY

input -> detect/align -> damage -> reference/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference specialist. Never blindly chain generators.

Architecture implementation rule: official executable upstream first; CFS thin adapter second. A fork/patch is justified only by a demonstrated compatibility, CPU/Windows, packaging, API or safety integration defect and must preserve a traceable diff from the pinned upstream revision.

---

## 23. MODEL SELECTION POLICY

Select winners on multiple identity-disjoint DEV/VALIDATION cases per damage; identity hard gate first; measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM; never select/tune using final holdout. Official paper metrics are context, not acceptance evidence. A model enters CFS only after its exact source/checkpoint, license, adapter behavior, wrong-person/provenance behavior and target-hardware resource use are measured.

---

## 24. HISTORICAL RECORD — append-only

- HIST-20260815-001 certified V1 merge `f476c6f... -> 2767513f...`, historical Windows/Female/Release Quality certification.
- HIST-20260818-002 Track A blocked `3645c8c...`; V3 consumed; V4 frozen/unexecuted.
- HIST-20260818-003 Track B `645862d1...` snapshot; DamageMaskNet attempt3 NOT_VERIFIED; RefFace NOT_RUN.
- HIST-20260819-004 canonical meta ledger established.
- HIST-20260819-005..010 identity attempts 1–3 + evidence; behavioral hypothesis closed.
- HIST-20260819-011..013 protocol-only source-contract corrections.
- HIST-20260819-014 `9b8810ce...` targeted 108 PASS/full pytest 3 fail; DEC-009 created.
- HIST-20260819-015 technical push `9b8810ce... -> 2fcaeb1b...`, DEC-009 attempt1 plus test-fixture alignment.
- HIST-20260819-016 Release Quality #129 on `2fcaeb1b...`: V3/V4 verify-only PASS; targeted `1 failed,107 passed`; artifact `9364721505`, digest `37a71c20ca44af86d2a4e6f839246f0d34ba9287563f8082e6641f747978eb0c`; DEC-009 attempt1 consumed.
- HIST-20260819-017 technical push `2fcaeb1b... -> 49af8cb1...`; DEC-009 attempt2; face-local identity/SFace/provenance unchanged.
- HIST-20260819-018 Female #580 exact artifact on `49af8cb1...`: 380 cases, 125 runtime errors, 22 safe abstentions, Target95 `21/179=11.7%`; dominant failure is lack of real SFace evidence, not threshold shopping.
- HIST-20260819-019 `c56e7fbf...` fixture attempt; `60b79658...` shallow-safe V4 blob history verification. Release #132: targeted 108/108 PASS, V4 freeze PASS, full pytest 545/546 with sole IMPORT-ordering fixture failure.
- HIST-20260819-020 `5240eaec...` scopes the IMPORT snapshot test to its actual ordering invariant; same-head CI in progress at ledger update.
- HIST-20260819-021 Track B upstream-first series `a7ffced0... -> b8da2286... -> 2978be94... -> d4f09f2b...`: official registry, validation, exact detached bootstrap and offline policy tests. No research model promoted to release.
