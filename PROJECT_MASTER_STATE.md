# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current state is maintained; important decisions, experiments, failures and technical pushes are preserved.

## OWNER DASHBOARD

- CURRENT PRODUCT VERSION: `PRODUCT_V1_1` operational hotfix; `PRODUCT_V1` remains the immutable certified release.
- CURRENT ACTIVE BRANCH: `hotfix/real-world-restoration-v1.1` for Track A; `research/paper-quality-local-v2` for isolated Track B research.
- CURRENT TECHNICAL HEAD: Track A `5240eaecb8943244f5bf7276a0905489d261318b`; Track B `1591fa3cebaa44a3ef80a6a2178ed85fd1ae2d66`.
- CURRENT PHASE: Track A failure classification and operational recovery; Track B benchmark/research infrastructure.
- CURRENT MAIN OBJECTIVE: eliminate runtime-error outcomes without weakening identity, provenance or holdout safety.
- WHAT WAS JUST COMPLETED: exact GitHub reconciliation of PR #2, all remote branches, same-HEAD Actions #1316/#133/#583 and the current Paper Quality branch.
- WHAT IS BEING WORKED ON: convert unsupported/no-biometric-evidence restoration paths into explicit conservative rollback or predeclared abstention outcomes instead of exceptions.
- WHAT IS BLOCKING PROGRESS: Track A Windows practical benchmark has 70 runtime errors and Female-domain #583 has 122; Release Quality #133 therefore cannot restore the same-HEAD Windows model pack. Track B DamageMaskNet attempt 3 remains evidence-unrecoverable from the available Actions index and must not be rerun merely for observability.
- WHAT MODEL IS CURRENTLY BEING TESTED: Track A uses the certified local YuNet/SFace/NAFNet/parser/pose/LaMa pack; Track B's newest measured candidate integration is the official pinned FBCNN adapter, still `NOT_VERIFIED` for its new workflow.
- WHY THAT MODEL: SFace is the frozen identity authority; FBCNN is the current DEV JPEG specialist leader and is isolated from V1.1.
- CURRENT BEST MODEL PER DAMAGE TYPE: mild blur/denoise NAFNet; JPEG FBCNN in DEV only; severe blind face GPEN in DEV identity evidence; opaque/reference-supported loss observed same-person component transfer; unqualified classes preserve MAIN/rollback/abstain.
- CURRENT QUALITY RESULT: Female #583 Target95 report-only `22/182`; quality target not achieved.
- CURRENT SAFETY RESULT: targeted identity/source/provenance suite PASS and full pytest `546/546` on Release #133 before the external prerequisite failure; SFace `0.363`, wrong-person `0` and provenance rules unchanged. End-to-end runtime gate FAIL.
- CURRENT WINDOWS STATUS: #1316 FAIL at practical public-portrait benchmark; installer/build steps did not run.
- CURRENT ELITEBOOK STATUS: `NOT_RUN` for PRODUCT_V1_1 and Paper Quality.
- NEXT EXACT STEP: implement and test structured fail-closed rollback/abstention for no-anchor/no-real-SFace cases on current Track A without treating proxy evidence as identity authority; do not execute V3 or V4.
- ESTIMATED PROJECT COMPLETION STATE: overall `43%` — engineering estimate based on independent release, research, personalized-reference, Windows and physical-PC gates below.

Completion estimates: V1.1 operational `72%` (runtime outcome handling, same-HEAD CI, installer and target-PC acceptance remain); Paper Quality V2 `38%` (DamageMaskNet evidence, specialist validation, broader identity-disjoint evaluation, Windows/offline pack remain); personalized restoration `35%` (prototype authority/routing exists, scientific validation and target-PC qualification remain); Windows productization `62%` (V1 historical package exists, V1.1 same-HEAD package and physical EliteBook acceptance remain); overall `43%` (unified PRODUCT_V5 and independent quality/target-hardware evidence remain).

## 0. Document metadata

- Updated: `2026-08-23`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last Track A branch: `hotfix/real-world-restoration-v1.1`
- Track A current HEAD: `5240eaecb8943244f5bf7276a0905489d261318b`
- Active Paper Quality HEAD: `research/paper-quality-local-v2@1591fa3cebaa44a3ef80a6a2178ed85fd1ae2d66`
- Track A identity/source/provenance targeted suite: latest exact evidence `108/108 PASS` on Release Quality #132 at `60b79658...`.
- Current Track A gate: exact-head Windows #1316, Release Quality #133 and Female-domain #583 on `5240eaec...` are all **FAIL**. Windows stopped at the practical benchmark with 70 runtime errors. Female executed 380 cases with 122 runtime errors and Target95 report-only `22/182`. Release Quality passed V3/V4 verification, the targeted suite and full pytest `546/546`, then failed because the exact-head Windows prerequisite failed. SFace `0.363`, face-local identity, provenance, models and holdouts are unchanged. V4 is not authorized to run.
- Current Track B direction: **UPSTREAM-FIRST**. Official executable paper/model repositories are the architecture baseline; CFS owns thin adapters, identity/provenance safety, resource control, checkpoint/hash verification, Windows/offline packaging and qualification tests.
- Current Track B FBCNN state: official source pinned; thin upstream adapter and fail-closed tests committed; real CPU upstream-adapter workflow wired. New adapter/workflow evidence is **NOT_VERIFIED** until the corresponding Actions run completes and its artifact is inspected.
- Current Track B blocker retained: recover already-triggered DamageMaskNet attempt-3 evidence without rerun for observability. Upstream qualification proceeds independently and does not alter V1.1.

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

Identity authority is no longer the current blocker. The same-canvas architecture separates **whole-canvas sameness**, which is only geometry/canvas evidence, from the stricter face-local identity bridge. V4 remains fail-closed: histogram/proxy similarity is not identity authority, direct SFace evidence cannot be propagated transitively, and wrong-person same-canvas donors are blocked.

Paper Quality does not reimplement a published architecture when an official executable upstream exists. The official repository is pinned and reused; CFS patches only compatibility/integration defects and adds safety/resource/package boundaries. Official code is not assumed bug-free and paper-reported quality is not treated as CFS/EliteBook evidence.

---

## 2. Branch and release map

| Branch | Purpose | HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f...` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | V1 history | `5eff6673...` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate history | `f476c6f0...` | FROZEN / ARCHIVED | merged PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `5240eaec...` | BLOCKED / ACTIVE | PR #2 OPEN/DRAFT/MERGEABLE; Windows #1316, Release #133, Female #583 FAIL | remove runtime-error outcomes without weakening safety; never run V4 early |
| `research/face-restoration-v2` | early data/degradation research | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `1591fa3c...` | ACTIVE / BENCHMARKING | FBCNN upstream adapter plus six commits of per-block progress/timeline/ETA telemetry; no workflow run indexed on the latest HEAD | recover existing DamageMaskNet attempt-3 evidence; keep FBCNN evidence NOT_VERIFIED |
| `meta/project-state` | canonical ledger | self-SHA omitted | ACTIVE META | docs only | update after every meaningful technical push |

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

V3 is verification-only. V4 may execute once only after frozen request protocol and all exact same-head prerequisites pass. No V4 request or CONSUMED marker has been authorized by this ledger update.

---

## 5. CURRENT GLOBAL OBJECTIVES

OBJ-001 Preserve V1 — PASS.  
OBJ-002 Restore V1.1 gates — VALIDATING on exact HEAD `5240eaec...`.  
OBJ-003 Canonical ledger — IN_PROGRESS.  
OBJ-004 DamageMaskNet — BLOCKED on existing attempt3 evidence.  
OBJ-005 Broad BFR selection — IN_PROGRESS, upstream-first.  
OBJ-006 FBCNN JPEG qualification — IN_PROGRESS; official source + thin adapter committed, new runtime evidence NOT_VERIFIED.  
OBJ-007 Personalized Reference Bank validation — IN_PROGRESS.  
OBJ-008 RefFace CPU — BLOCKED by OBJ-004, 0/3 attempts.  
OBJ-009 Paper Quality Windows pack — PROPOSED.  
OBJ-010 HP EliteBook acceptance — PROPOSED.  
OBJ-011 Official upstream implementation registry — IMPLEMENTED; runtime qualification remains per-model.  
OBJ-012 Official upstream adapters — IN_PROGRESS; FBCNN is first concrete implementation.

---

## 6. MODEL MASTER REGISTRY

Certified roles: YuNet, SFace `0.363`, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research: GPEN BENCHMARKING/license blocker; GFPGAN1.4 BENCHMARKING; CodeFormer BENCHMARKING/BLOCKED_LICENSE; FBCNN BENCHMARKING/current DEV JPEG leader; DamageMaskNet BENCHMARKING/BLOCKED; RefFace FEASIBILITY_ONLY/NOT_RUN; InstantRestore/OSDFace hardware-blocked feasibility; RestoreFormer++/VQFR/GPEN-inpainting/RefineFIR/PerFuSe/RefIPFR/Real-ESRGAN feasibility until measured.

Official repository registry: `config/upstream-implementations.json`. Initial pinned source baselines: GPEN `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`; GFPGAN `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`; CodeFormer `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`; FBCNN `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`; InstantRestore `snap-research/InstantRestore@05891bf7d30ab7290c501272de7a1a4a51b21b4f`. RefineFIR, RefFaceInpainting, OSDFace and RestoreFormer++ remain `NOT_VERIFIED` until revision/checkpoint/license/runtime qualification.

FBCNN upstream contract at `6ea5d113...`: official repository source only; Apache-2.0 code; `fbcnn_color.pth` official v1.0 asset; exact source revision fixed. CFS `app/fbcnn_upstream_backend.py` dynamically imports the official `models/network_fbcnn.py`, requires `.cfs-upstream.json`, requires an explicit 64-hex checkpoint SHA-256, rejects non-JPEG/recompression routes, runs CPU-only and marks model-modified pixels `GENERATED_MODEL_INFERRED`. CFS does **not** contain a copied `class FBCNN` architecture.

Registry documentation/package-manifest mismatch from V1 remains separate; never invent missing manifests/hashes.

---

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEV historical evidence: GPEN SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, `~1.828GB`; GFPGAN1.4 SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`; FBCNN QF20 SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, `~1.305GB`; CodeFormer real CPU slice PASS, exact metrics artifact-required.

The historical FBCNN workflow already used official source SHA `54d18319...`, CPU-only PyTorch and the official `fbcnn_color.pth` asset with expected byte size `287755111`, and emitted the checkpoint SHA-256 into evidence. The new upstream-adapter workflow is intended to reproduce this route through the common CFS backend and recover a digest suitable for later registry pinning; until that new run/artifact is inspected its new evidence is NOT_VERIFIED.

These CFS measurements are distinct from paper-reported metrics. An official repository does not make paper numbers reproduced on the HP EliteBook; target-PC results remain NOT_RUN until measured.

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

DEC-001 canonical ledger ACCEPTED. DEC-002 active Paper Quality branch ACCEPTED. DEC-003 <=80% CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet bank ACCEPTED. DEC-006 RefFace after DamageMaskNet ACCEPTED/BLOCKED. DEC-007 V3 consumed/V4 untouched ACCEPTED. DEC-008 ranking cluster != identity authority ACCEPTED/CLOSED after 3/3. DEC-009 localized-damage same-canvas edge isolation ACCEPTED: broad canvas is not identity authority; face-local/SFace remains the identity gate. **DEC-010 official-upstream-first model integration — ACCEPTED:** if official executable paper/model code exists, reuse and pin that source rather than reimplementing the architecture. Upstream code is not assumed bug-free; CFS changes are limited to compatibility/adapters/safety/resource/package integration and must pass independent tests. **DEC-011 FBCNN first upstream-adapter qualification — ACCEPTED:** preserve the official PyTorch network and official color checkpoint path; CFS only supplies pinned checkout verification, checkpoint digest gate, JPEG routing, resource/provenance boundary and evidence workflow.

---

## 14. EXPERIMENT LOG

DamageMaskNet 1/3 403 infra fail, 2/3 429 infra fail, 3/3 NOT_VERIFIED; no attempt4 for observability. RefFace PREPARED/NOT_RUN.

Identity hypothesis attempts 1–3 closed; targeted suite reached 108/108 PASS at `9b8810ce...` and again on Release Quality #132 at `60b79658...`.

**EXP-20260819-012 DEC-009 attempt1:** `9b8810ce... -> 2fcaeb1b...`; strict Lab + local mismatch dilation + minimum stable-edge support. Release Quality #129 targeted `1 failed,107 passed`. Sole failure separated broad same-canvas geometry evidence from face-local identity authority. Artifact `9364721505`, digest `37a71c20ca44af86d2a4e6f839246f0d34ba9287563f8082e6641f747978eb0c`.

**EXP-20260819-013 DEC-009 attempt2:** `2fcaeb1b... -> 49af8cb1...`; strict global Lab rule retained, attempt-1 stable-edge survival requirement removed, face-local identity/SFace/provenance unchanged.

**EXP-20260819-014 Female #580 on `49af8cb1...`:** 76 resolved portraits, 380 executed cases, 125 runtime errors, 22 safe abstentions, 233 completed restorations. Target95 report-only `21/179 = 11.7%`. Error root classes: 77 proxy-not-SFace authority, 41 no biometric anchor, 4 no usable SFace comparisons, only 3 true below-threshold SFace failures. `mosaic_single` was `38/38` runtime error and has zero references by benchmark design. Dominant blocker is lack/propagation of real SFace evidence on severe no-reference cases, not a reason to lower `0.363` or re-enable proxy authority.

**EXP-20260819-015 infrastructure repair series:** `49af8cb1... -> c56e7fbf... -> 60b79658... -> 5240eaec...`. Release Quality #132 on `60b79658...`: V3 verify-only PASS, V4 freeze/blob verification PASS, targeted identity/source/provenance `108/108 PASS`; full pytest `545 passed, 1 failed`, solely the IMPORT/preflight ordering test. `5240eaec...` scopes that test directly to IMPORT-before-preflight ordering, with no production identity change.

**EXP-20260823-018 exact-HEAD Track A reconciliation:** Windows #1316 FAIL after `546 passed`, validation and smoke checks because the practical public-portrait benchmark produced 70 runtime errors, dominated by severe multi-reference cases with no usable biometric anchor and no SFace/proxy comparison. Female #583 resolved 76/80 portraits, executed 380 cases, completed 237 restorations, produced 122 runtime errors, and reported Target95 `22/182`. Release Quality #133 verified consumed V3 without execution, verified frozen V4 without execution, passed the targeted suite and full pytest `546/546`, then failed when Windows #1316 completed FAIL. Classification: production runtime outcome handling, not test-suite instability and not permission to weaken SFace/provenance.

**EXP-20260823-019 Paper Quality reconciliation:** remote Track B advanced `6ea5d113... -> 1591fa3c...` with per-block progress, live model timeline and ETA telemetry plus tests/workflow validation. No workflow run is indexed for the latest HEAD. These commits improve observability only and do not qualify a restoration model or resolve DamageMaskNet attempt 3.

**EXP-20260819-016 upstream-first Track B:** `research/paper-quality-local-v2` advanced through `a7ffced0... -> b8da2286... -> 2978be94... -> d4f09f2b...`. Added machine-readable official-upstream registry, offline validator, pinned detached-checkout bootstrap and tests. Pinned GPEN, GFPGAN, CodeFormer, FBCNN and InstantRestore. Unpinned specialists remain NOT_VERIFIED and cannot bootstrap. No Paper model promoted to production.

**EXP-20260819-017 FBCNN upstream-adapter integration:** `d4f09f2b... -> dfaf7bd1... -> 0ae6d420... -> 6ea5d113...`. Added `app/fbcnn_upstream_backend.py` with exact official source enforcement, explicit checkpoint SHA-256 firewall, CPU-only inference, JPEG-only routing and generated provenance. Added fail-closed tests for wrong repo/revision/hash/route. Then atomically refactored `research/run_fbcnn_vertical_slice.py` and `.github/workflows/research-fbcnn-vertical-slice.yml` so the benchmark uses the pinned CFS bootstrap + thin official backend rather than local model-loading/inference duplication. Workflow also validates registry/tests, checks official checkpoint byte size `287755111`, discovers and records its SHA-256, enforces SFace identity gate and uploads evidence. Runtime result/artifact **NOT_VERIFIED** at this ledger update.

---

## 15. QUALITY SCOREBOARD

DEV evidence exists; broad validation incomplete. V3 consumed 39/40. V4 frozen/unexecuted. Target-PC Paper Quality NOT_RUN. Female #583 Target95 report-only `22/182 = 12.1%`; therefore QUALITY_TARGET_ACHIEVED remains FALSE. Maintain DEV/VALIDATION/HOLDOUT/REAL-WORLD/TARGET-PC separately.

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

Current exact technical HEAD `5240eaecb8943244f5bf7276a0905489d261318b`: only `tests/test_automatic.py` changed from `60b...`, scoping `test_preflight_cannot_mutate_true_import_snapshot` to `IMPORT -> preflight` ordering. Production code, SFace threshold, V4 manifests and holdout protocol were not changed. Exact-head Windows #1316, Release Quality #133 and Female #583 all completed FAIL.

Next exact action: implement explicit structured rollback/abstention outcomes for unsupported no-anchor/no-real-SFace runtime cases, using immutable MAIN/checkpoint output and preserving rejected/partial diagnostics. Do not turn proxy evidence into SFace authority. Re-run targeted/full tests and only then push a new Track A candidate. V3 remains verification-only and V4 remains unexecuted.

---

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@1591fa3cebaa44a3ef80a6a2178ed85fd1ae2d66`: real CPU BFR/JPEG evidence, 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, hard-gated selector, deterministic fusion, parser adapter, RefFace workflow, official-upstream registry/bootstrap, first concrete thin upstream backend (FBCNN), and per-block progress/model timeline/ETA telemetry.

`config/upstream-implementations.json` encodes the upstream-first contract. `scripts/verify_upstream_implementation_registry.py` rejects malformed/unpinned qualification states. `scripts/bootstrap_pinned_upstream.py` checks out exact official Git revisions detached/research-only and refuses unpinned models. `tests/test_upstream_implementation_registry.py` covers registry policy. `tests/test_fbcnn_upstream_backend.py` covers FBCNN source/hash/route fail-closed behavior. These new tests are committed but must not be called PASS until Actions evidence exists.

FBCNN integration now uses official code directly. `research/run_fbcnn_vertical_slice.py` no longer owns a duplicate FBCNN network loader/inference implementation; it calls `FBCNNUpstreamBackend`. The associated workflow bootstraps the exact official checkout and official release checkpoint, then measures real CPU identity/quality/resource evidence. Its first upstream-adapter result is pending/NOT_VERIFIED.

PDF constraints remain: separate global identity from local texture; use correspondence between matching regions; region-adaptive identity guidance for severe BFR; MAIN preserves pose/composition/expression/geometry; unsupported detail remains conservative. Paper-reported metrics and CFS-reproduced metrics remain separate.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1, ONNX parity, RAM/runtime. Infrastructure fail -> infrastructure-only. True model/data fail -> U-Net hypothesis ends. Then RefFace attempt1/3.

In parallel: complete FBCNN upstream-adapter evidence, pin the observed official checkpoint digest only after artifact verification, then broaden FBCNN validation across single JPEG, double-JPEG/non-aligned and social/smartphone recompression. Next upstream heavy candidates remain GPEN/GFPGAN/CodeFormer, then InstantRestore if CPU/Windows feasibility is credible. RefineFIR/RefFaceInpainting/OSDFace/RestoreFormer++ remain NOT_VERIFIED until exact revision/checkpoint/license/runtime evidence exists.

---

## 22. SPECIALIST MODEL STRATEGY

input -> detect/align -> damage -> reference/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference specialist. Never blindly chain generators.

Architecture implementation rule: official executable upstream first; CFS thin adapter second. A fork/patch is justified only by a demonstrated compatibility, CPU/Windows, packaging, API or safety integration defect and must preserve a traceable diff from the pinned upstream revision.

---

## 23. MODEL SELECTION POLICY

Select winners on multiple identity-disjoint DEV/VALIDATION cases per damage; identity hard gate first; measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM; never select/tune using final holdout. Official paper metrics are context, not acceptance evidence. A model enters CFS only after exact source/checkpoint, license, adapter behavior, wrong-person/provenance behavior and target-hardware resource use are measured.

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
- HIST-20260819-020 `5240eaec...` scopes the IMPORT snapshot test to its actual ordering invariant; exact same-head CI remained in progress at the latest check.
- HIST-20260819-021 Track B upstream-first series `a7ffced0... -> b8da2286... -> 2978be94... -> d4f09f2b...`: official registry, validation, exact detached bootstrap and offline policy tests. No research model promoted to release.
- HIST-20260819-022 Track B FBCNN series `d4f09f2b... -> dfaf7bd1... -> 0ae6d420... -> 6ea5d113...`: thin pinned official backend, fail-closed tests and atomic benchmark/workflow refactor. Runtime evidence NOT_VERIFIED at ledger update; no research model promoted to release.
- HIST-20260823-023 exact-head Track A Actions reconciled: Windows #1316 FAIL, Female #583 FAIL, Release #133 FAIL downstream of Windows; V3 not executed and V4 not executed.
- HIST-20260823-024 Track B reconciled at `1591fa3c...`; telemetry work recorded, model qualification state unchanged.

---

## 25. PUSH JOURNAL — append-only

### PUSH-20260823-001

- DATE/TIME UTC: `2026-08-23`, exact ledger commit time supplied by Git.
- TECHNICAL BRANCH: reconciliation across `hotfix/real-world-restoration-v1.1` and `research/paper-quality-local-v2`; no technical push created by this entry.
- PREVIOUS HEAD: Track A `5240eaec...`; Track B `6ea5d113...` as previously recorded.
- NEW REMOTE HEAD: Track A unchanged `5240eaec...`; Track B observed `1591fa3c...`.
- COMMITS INCLUDED: Track B telemetry series `ce10155...` through `1591fa3...`.
- FILES ADDED/MODIFIED/REMOVED: ledger reconciliation only on meta branch; technical file details remain in the referenced commits.
- OBJECTIVES/VERSIONS/BLOCKS/MODELS/DATASETS AFFECTED: OBJ-002 status changed to BLOCKED; OBJ-004 remains BLOCKED; PRODUCT_V1_1 runtime outcome handling; no model/dataset/holdout change.
- TESTS RUN/RESULT: remote Release Quality #133 targeted suite PASS and full pytest `546/546`; remote Windows #1316 test step PASS; end-to-end practical gate FAIL.
- WORKFLOWS/RESULT: Windows #1316 FAIL; Female #583 FAIL; Release Quality #133 FAIL; latest Track B HEAD has no indexed workflow run.
- BENCHMARK RESULT: Windows practical 70 runtime errors; Female 380 executed, 237 completed, 122 runtime errors, Target95 `22/182` report-only.
- KNOWN FAILURES/NEW RISKS: fail-closed biometric exceptions are being counted as runtime crashes; any correction must preserve zero wrong-person contribution and must not grant proxy identity authority.
- DECISIONS/STATUS TRANSITIONS: Track A VALIDATING -> BLOCKED; V4 remains NOT_RUN/UNCONSUMED.
- NEXT EXACT ACTION: implement structured conservative rollback/abstention on current Track A, test locally, push exact SHA, then append a second journal entry with that technical result.

---

## 26. SESSION START/END CONTINUITY RULE

Every session reads this ledger, reconciles GitHub, and continues the recorded blocker. Every legitimate technical push is followed by its exact remote SHA and evidence here. Ledger commits never attempt to record their own SHA.
