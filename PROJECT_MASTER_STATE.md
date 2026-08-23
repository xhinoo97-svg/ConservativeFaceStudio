# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current state is maintained; important decisions, experiments, failures and technical pushes are preserved.

## OWNER DASHBOARD

- CURRENT PRODUCT VERSION: `PRODUCT_V1_1` candidate is V4 NO-GO/CONSUMED_FAIL; `PRODUCT_V1` remains the immutable certified release.
- CURRENT ACTIVE BRANCH: `hotfix/real-world-restoration-v1.1` for Track A; `research/paper-quality-local-v2` for isolated Track B research.
- CURRENT TECHNICAL HEAD: Track A branch `77687b3b171f4e9989fcf486834f2d8b7a52f591`; evaluated candidate `b6ce7ebde87d4ce84e5849664716dc3e822ad762`; Track B `1591fa3cebaa44a3ef80a6a2178ed85fd1ae2d66`.
- CURRENT PHASE: Track A V4 one-shot consumed FAIL before the first case; candidate `b6ce7ebd...` is NO-GO and may not be rerun. Track B benchmark/research infrastructure.
- CURRENT MAIN OBJECTIVE: preserve the consumed V4 record and design an independent V5 certification lineage without weakening identity, provenance or holdout safety.
- WHAT WAS JUST COMPLETED: V4 Final Certification #1 consumed the one-shot authority for candidate `b6ce7ebd...`. Pre-consumption gates passed, including calibration 60/60; after the persistent STARTED marker, the runner failed before case 1 with `TypeError: build_freeze() missing 1 required positional argument: 'contract_payload'`. Final marker state is `CONSUMED_FAIL`.
- WHAT IS BEING WORKED ON: Track A failure evidence is frozen and reconciled; no retry, rerun or candidate repair is allowed from V4. Track B research continues independently.
- WHAT IS BLOCKING PROGRESS: V4 is irrevocably consumed with 0/40 cases completed, so `b6ce7ebd...` cannot be certified. Any future certification requires a new independently frozen, identity-disjoint V5 candidate/holdout lineage. Track B DamageMaskNet attempt 3 remains evidence-unrecoverable from the available Actions index.
- WHAT MODEL IS CURRENTLY BEING TESTED: Track A uses the certified local YuNet/SFace/NAFNet/parser/pose/LaMa pack; Track B's newest measured candidate integration is the official pinned FBCNN adapter, still `NOT_VERIFIED` for its new workflow.
- WHY THAT MODEL: SFace is the frozen identity authority; FBCNN is the current DEV JPEG specialist leader and is isolated from V1.1.
- CURRENT BEST MODEL PER DAMAGE TYPE: mild blur/denoise NAFNet; JPEG FBCNN in DEV only; severe blind face GPEN in DEV identity evidence; opaque/reference-supported loss observed same-person component transfer; unqualified classes preserve MAIN/rollback/abstain.
- CURRENT QUALITY RESULT: V4 final quality was not measured because 0/40 cases ran. Female #584 remains recoverable mean `81.3055`, Target95 report-only `21/304 = 6.91%`; quality target not achieved.
- CURRENT SAFETY RESULT: V4 pre-consumption checks passed targeted `110/110`, full pytest `547/547`, and calibration `60/60` with zero errors, provenance violations and wrong-person pixels. Final-holdout safety metrics are NOT_MEASURED because the runner failed before case 1. SFace `0.363` and frozen guardrails were unchanged.
- CURRENT WINDOWS STATUS: #1317 `SUCCESS`; exact-HEAD installer, portable package, release metadata, production-model updates and validation artifacts published. Physical EliteBook acceptance remains `NOT_RUN`.
- CURRENT ELITEBOOK STATUS: `NOT_RUN` for PRODUCT_V1_1 and Paper Quality.
- NEXT EXACT STEP: preserve `CONSUMED_FAIL`; do not rerun V4 or modify/delete its marker. Define a new V5 protocol with an independently frozen identity-disjoint holdout and repair the generic runner interface before freezing any new candidate.
- ESTIMATED PROJECT COMPLETION STATE: overall `43%` — engineering estimate based on the consumed V4 NO-GO plus independent research, Windows and physical-PC gates below.

Completion estimates: V1.1 operational `76%` (exact-head runtime, release, calibration and packaging gates pass, but certification candidate is V4 NO-GO; new V5 and target-PC acceptance remain); Paper Quality V2 `38%`; personalized restoration `35%`; Windows productization `80%` (CI artifacts pass; physical EliteBook acceptance remains); overall `43%`.

## 0. Document metadata

- Updated: `2026-08-23`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last Track A branch: `hotfix/real-world-restoration-v1.1`
- Track A current branch HEAD: `77687b3b171f4e9989fcf486834f2d8b7a52f591`; evaluated candidate: `b6ce7ebde87d4ce84e5849664716dc3e822ad762`
- Active Paper Quality HEAD: `research/paper-quality-local-v2@1591fa3cebaa44a3ef80a6a2178ed85fd1ae2d66`
- Track A identity/source/provenance targeted suite: latest exact evidence `108/108 PASS` on Release Quality #134 at `b6ce7ebd...`.
- Current Track A gate: Windows #1317, Release #134 and Female #584 remain SUCCESS on `b6ce7ebd...`. V4 Final Certification #1 (`32656139686`) is FAIL; request `d847798e...`, STARTED marker `d03d97c6...`, final disposition `77687b3b...`. V4 is `CONSUMED_FAIL`; 0/40 cases executed; no rerun.
- Current Track B direction: **UPSTREAM-FIRST**. Official executable paper/model repositories are the architecture baseline; CFS owns thin adapters, identity/provenance safety, resource control, checkpoint/hash verification, Windows/offline packaging and qualification tests.
- Current Track B FBCNN state: official source pinned; thin upstream adapter and fail-closed tests committed; real CPU upstream-adapter workflow wired. New adapter/workflow evidence is **NOT_VERIFIED** until the corresponding Actions run completes and its artifact is inspected.
- Current Track B blocker retained: recover already-triggered DamageMaskNet attempt-3 evidence without rerun for observability. Upstream qualification proceeds independently and does not alter V1.1.

FORENSIC_MODE_READY: **TRUE for certified PRODUCT_V1 only**
PAPER_QUALITY_MODE_READY: **FALSE**
WINDOWS_INSTALLER_READY: **CI-READY for PRODUCT_V1_1; physical EliteBook acceptance NOT_RUN**
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
| `hotfix/real-world-restoration-v1.1` | Track A | branch `77687b3b...`; candidate `b6ce7ebd...` | V4 CONSUMED_FAIL / NO-GO | PR #2 OPEN/DRAFT; prerequisites PASS; V4 #1 FAIL before case 1 | preserve marker/evidence; no V4 rerun; create independent V5 lineage |
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

CALIBRATION_V1 historical 60/60. FINAL_HOLDOUT_V1 historical 40/40. FINAL_HOLDOUT_V2 details not fully re-reconciled. FINAL_HOLDOUT_V3 **CONSUMED** 39/40; mosaic SFace `0.360<0.363`; NEVER rerun/tune. FINAL_HOLDOUT_V4 **CONSUMED_FAIL** with 0/40 cases completed: marker was written before execution and runner then failed on the generic `build_freeze` interface before case 1; NEVER rerun/tune. V5 not created. Female-domain ~300–400 stress cases. Paper Quality DEV/VALIDATION separate. DamageMaskNet bank FairFace+ControlFace TRAIN/VALIDATION only.

V3 is verification-only. V4 is consumed and closed. Its marker must not be modified or deleted, and no retry, job rerun or second request is permitted. Any future candidate requires a new independently frozen identity-disjoint V5 holdout.

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

**EXP-20260823-020 exact-HEAD Track A rollback qualification:** `b6ce7ebde87d4ce84e5849664716dc3e822ad762`, tree `51369b538b20c6f58853f59afa3b0fd43dc07919`. Windows #1317 SUCCESS with full pytest `547/547`, practical benchmark `120/120` completed and zero runtime errors, extended practical matrix `82/82` completed, successful installer/portable packaging and five artifacts. Female #584 SUCCESS: 76/80 portraits resolved, 380/380 cases completed, zero runtime errors, recoverable mean `81.3055`, Target95 report-only `21/304 = 6.91%`; four source-resolution errors reduced the resolved portrait set but did not create runtime case errors. Release Quality #134 SUCCESS: targeted `108/108`, full pytest `547/547`, V3 and frozen V4 manifests verification-only, real local-model smoke PASS, V4 candidate calibration `60/60` with zero errors, same-HEAD candidate freeze created. V4 final holdout explicitly remained unexecuted. Classification: operational rollback fix verified without threshold, model, provenance, workflow, dataset or holdout changes; quality target remains unmet and report-only.

**EXP-20260823-021 V4 one-shot final certification:** request commit `d847798efda95890febb1bf9ae5c9832727ccb43` was the single child of candidate `b6ce7ebd...` and added only `release/v4-certification-request.json`. V4 Final Certification #1, run `32656139686`, attempt 1, verified the request, detached candidate, immutable freeze/history, targeted regressions `110/110`, full pytest `547/547`, Windows #1317, Female #584, real model pack and CPU runtime. Recalibration passed 60/60 with zero errors, zero provenance violations and zero wrong-person pixels; Target95 remained report-only `10/51`. Commit `d03d97c677a10a51d5333e8cf6a6de573cf42161` persisted state STARTED before execution. The V4 runner then failed before case 1 because `run_face_smartphone_baseline.run_baseline()` called the V4 `build_freeze()` adapter with one argument while V4 requires `contract_payload`; exact error `TypeError: build_freeze() missing 1 required positional argument: 'contract_payload'`. No final-holdout baseline or gate was produced, so completed cases are 0/40 and V4 wrong-person/provenance/rollback/abstention/Target95 metrics are NOT_MEASURED. Final marker commit `77687b3b171f4e9989fcf486834f2d8b7a52f591` records `CONSUMED_FAIL`; artifact `9497756063`, archive SHA-256 `62ad4233872464ac30e67d2adf761396d21f511fcd45936b5d33b214f335c1d0`. Candidate classification: **NO-GO**. No rerun or retry is allowed; this failure occurred after consumption even though before the first case.

**EXP-20260823-019 Paper Quality reconciliation:** remote Track B advanced `6ea5d113... -> 1591fa3c...` with per-block progress, live model timeline and ETA telemetry plus tests/workflow validation. No workflow run is indexed for the latest HEAD. These commits improve observability only and do not qualify a restoration model or resolve DamageMaskNet attempt 3.

**EXP-20260819-016 upstream-first Track B:** `research/paper-quality-local-v2` advanced through `a7ffced0... -> b8da2286... -> 2978be94... -> d4f09f2b...`. Added machine-readable official-upstream registry, offline validator, pinned detached-checkout bootstrap and tests. Pinned GPEN, GFPGAN, CodeFormer, FBCNN and InstantRestore. Unpinned specialists remain NOT_VERIFIED and cannot bootstrap. No Paper model promoted to production.

**EXP-20260819-017 FBCNN upstream-adapter integration:** `d4f09f2b... -> dfaf7bd1... -> 0ae6d420... -> 6ea5d113...`. Added `app/fbcnn_upstream_backend.py` with exact official source enforcement, explicit checkpoint SHA-256 firewall, CPU-only inference, JPEG-only routing and generated provenance. Added fail-closed tests for wrong repo/revision/hash/route. Then atomically refactored `research/run_fbcnn_vertical_slice.py` and `.github/workflows/research-fbcnn-vertical-slice.yml` so the benchmark uses the pinned CFS bootstrap + thin official backend rather than local model-loading/inference duplication. Workflow also validates registry/tests, checks official checkpoint byte size `287755111`, discovers and records its SHA-256, enforces SFace identity gate and uploads evidence. Runtime result/artifact **NOT_VERIFIED** at this ledger update.

---

## 15. QUALITY SCOREBOARD

DEV evidence exists; broad validation incomplete. V3 is consumed 39/40. V4 is `CONSUMED_FAIL` with 0/40 cases completed because the runner failed after the persistent STARTED marker but before case 1; no V4 quality or safety metric was measured. Target-PC Paper Quality NOT_RUN. Female #584 Target95 remains report-only `21/304 = 6.91%`; therefore QUALITY_TARGET_ACHIEVED remains FALSE. Maintain DEV/VALIDATION/HOLDOUT/REAL-WORLD/TARGET-PC separately.

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

Evaluated candidate `b6ce7ebde87d4ce84e5849664716dc3e822ad762`, tree `51369b538b20c6f58853f59afa3b0fd43dc07919`, passed Windows #1317, Release #134, Female #584 and the V4 workflow's pre-consumption suite/calibration. Request `d847798e...` triggered the single V4 run. The workflow persisted STARTED and then failed before case 1 at the generic baseline/V4 freeze adapter boundary. Branch HEAD `77687b3b...` contains only the request plus immutable STARTED/final `CONSUMED_FAIL` marker history above the candidate. No code, model, threshold, manifest, contract or candidate SHA was changed by the request/certification sequence.

Next exact action: preserve V4 `CONSUMED_FAIL` and its artifacts; never rerun or retry it. For future certification, define an independently frozen identity-disjoint V5 lineage, repair and test the runner interface before candidate freeze, then evaluate a new candidate without consulting V4 hidden outputs. PR #2 must not merge as certified.

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
- HIST-20260823-025 Track A `5240eaec... -> b6ce7ebd...`: structured immutable-MAIN identity rollback; exact tree `51369b53...`; local targeted `33/33 PASS`, full pytest `547/547 PASS`; exact-head CI started; V3/V4 not executed by this push.
- HIST-20260823-026 Track A exact-head qualification complete on `b6ce7ebd...`: Windows #1317 SUCCESS, Release #134 SUCCESS, Female #584 SUCCESS; female `380/380` cases and zero runtime errors; Release calibration `60/60`, exact V4 candidate freeze created; final V4 holdout NOT_RUN/UNCONSUMED.
- HIST-20260823-027 V4 one-shot consumed FAIL: request `d847798e...`, run `32656139686`, STARTED `d03d97c6...`, final marker `77687b3b...`; pre-gates/calibration PASS, then runner interface TypeError before case 1; 0/40, no rerun, candidate NO-GO; artifact `9497756063`, SHA-256 `62ad4233...`.

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

### PUSH-20260823-002

- DATE/TIME UTC: `2026-08-23`; exact technical commit time supplied by GitHub.
- TECHNICAL BRANCH: `hotfix/real-world-restoration-v1.1`.
- PREVIOUS HEAD: `5240eaecb8943244f5bf7276a0905489d261318b`.
- NEW REMOTE HEAD: `b6ce7ebde87d4ce84e5849664716dc3e822ad762`.
- COMMITS INCLUDED: `Rollback failed identity checks without runtime crashes`.
- FILES ADDED: none.
- FILES MODIFIED: `app/automatic.py`, `app/female_domain_benchmark.py`, `app/history.py`, `app/practical_benchmark.py`, `tests/test_automatic.py`.
- FILES REMOVED: none.
- OBJECTIVES/VERSIONS/BLOCKS/MODELS/DATASETS AFFECTED: OBJ-002; PRODUCT_V1_1; Block 11 product-boundary outcome and benchmark reporting. No model, threshold, workflow, benchmark manifest, dataset or holdout changed.
- TESTS RUN/RESULT: targeted `33/33 PASS`; full pytest `547/547 PASS`; `compileall` and `git diff --check` PASS.
- WORKFLOWS TRIGGERED/RESULT: Windows #1317 `SUCCESS`; Release Quality #134 `SUCCESS`; Female-domain #584 `SUCCESS`, all on exact remote SHA `b6ce7ebd...`.
- BENCHMARK RESULT: Windows practical `120/120` completed with zero runtime errors; extended matrix `82/82` completed. Female-domain resolved 76/80 portraits and completed 380/380 cases with zero runtime errors, recoverable mean `81.3055`, Target95 report-only `21/304 = 6.91%`. Release calibration admitted `60/60` with zero errors and froze the same candidate tree.
- ARTIFACTS: Windows installer `9496542126` digest `f61b5775...`; portable `9496542964` digest `8249b1f7...`; release metadata `9496543197`; production-model updates `9496543913`; validation `9496548494`. Female evidence `9496763776` digest `08aa6528...`; lightweight report `9496764084` digest `e8ffca4f...`. Release evidence `9496826181` digest `51ac88db...`.
- KNOWN FAILURES/NEW RISKS: rollback remains a safety/no-recovery outcome, never a restoration PASS. Female Target95 remains materially below target and report-only; PRODUCT quality is not certified. Physical EliteBook acceptance remains NOT_RUN.
- DECISIONS CREATED: none; implements the already-recorded structured fail-closed direction.
- STATUS TRANSITIONS: Track A BLOCKED -> VALIDATING -> EXACT-HEAD GATES PASS; V4 candidate calibrated/frozen, V4 final holdout remains NOT_RUN/UNCONSUMED.
- NEXT EXACT ACTION: preserve the exact candidate freeze. Do not execute V4 final without its separate explicit one-shot request/authorization protocol.

### PUSH-20260823-003

- DATE/TIME UTC: `2026-08-23`.
- TECHNICAL BRANCH: `hotfix/real-world-restoration-v1.1`.
- PREVIOUS HEAD: candidate `b6ce7ebde87d4ce84e5849664716dc3e822ad762`.
- REQUEST COMMIT: `d847798efda95890febb1bf9ae5c9832727ccb43`, immediate child of candidate; message `Request one-shot V4 final certification for b6ce7eb`.
- WORKFLOW MARKER COMMITS: STARTED `d03d97c677a10a51d5333e8cf6a6de573cf42161`; final `CONSUMED_FAIL` / current branch HEAD `77687b3b171f4e9989fcf486834f2d8b7a52f591`.
- FILES ADDED: request commit added only `release/v4-certification-request.json`; workflow added `benchmarks/face-smartphone-v4-final-holdout/CONSUMED.json`.
- FILES MODIFIED: workflow final-disposition commit modified only `CONSUMED.json` from STARTED to CONSUMED_FAIL.
- FILES REMOVED: none.
- CODE/MODELS/THRESHOLDS/MANIFESTS/CONTRACTS/CANDIDATE CHANGED: none.
- WORKFLOW/RESULT: V4 Final Certification #1, run `32656139686`, attempt 1, `FAILURE`; no rerun/retry.
- PRE-CONSUMPTION RESULT: immutable freeze PASS; targeted `110/110`; full pytest `547/547`; Windows #1317 and Female #584 resolved SUCCESS; model/runtime PASS; calibration `60/60`, errors `0`, provenance invalid `0`, wrong-person final pixels `0`, restoration passes `60`, Target95 report-only `10/51`.
- CONSUMPTION/FINAL RESULT: marker persisted before execution; state `CONSUMED_FAIL`; V4 runner failed before case 1 with `TypeError: build_freeze() missing 1 required positional argument: 'contract_payload'`; final gate skipped; completed cases `0/40`; final-holdout metrics NOT_MEASURED.
- ARTIFACT: `v4-final-certification-32656139686`, ID `9497756063`, archive SHA-256 `62ad4233872464ac30e67d2adf761396d21f511fcd45936b5d33b214f335c1d0`.
- INSTALLER EVIDENCE RETAINED: Windows artifact `ConservativeFaceStudio-Setup-Windows-x64`, ID `9496542126`, archive SHA-256 `f61b577553d33ffea63fae2bdd072c5953eb789399e89b4bf0dd38e929043212`, internal EXE SHA-256 `52784ae70f215e34fb81f8b56bd885bfb1ba34a7573e795d6d6b5be3bc3e2999`; clean Windows and user PC NOT_RUN.
- CLASSIFICATION: candidate `b6ce7ebd...` NO-GO; V4 closed/consumed; OPERATIONAL_RELEASE_READY FALSE; QUALITY_COMPLETE FALSE.
- NEXT EXACT ACTION: preserve all V4 evidence and marker; do not merge PR #2 as certified; do not rerun V4. Create a new independent V5 holdout/candidate lineage only after the generic runner interface is repaired and tested before freeze.

---

## 26. SESSION START/END CONTINUITY RULE

Every session reads this ledger, reconciles GitHub, and continues the recorded blocker. Every legitimate technical push is followed by its exact remote SHA and evidence here. Ledger commits never attempt to record their own SHA.
