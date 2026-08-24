# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current state is maintained; important decisions, experiments, failures and technical pushes are preserved.

## OWNER DASHBOARD

- CURRENT PRODUCT VERSION: `PRODUCT_V1_1` candidate is V4 NO-GO/CONSUMED_FAIL; `PRODUCT_V1` remains the immutable certified release.
- CURRENT ACTIVE BRANCH: `hotfix/real-world-restoration-v1.1` preserves consumed Track A evidence; `protocol/v5-certification-hardening` contains DEV-only runner hardening; `research/paper-quality-local-v2` is the active isolated Track B research branch.
- CURRENT TECHNICAL HEAD: Track A branch `77687b3b171f4e9989fcf486834f2d8b7a52f591`; evaluated candidate `b6ce7ebde87d4ce84e5849664716dc3e822ad762`; protocol hardening `268188c5a2540455ff804383cb583b16546b62f1`; Track B `0d4b5fa2d4fad36c9fde484d2ebdac0ae6a61053`.
- CURRENT PHASE: V4 remains permanently consumed FAIL/NO-GO. The generic future one-shot runner is hardened on synthetic DEV fixtures; Track B is qualifying the official FBCNN compression specialist and replacing an inadequate DamageMaskNet hypothesis.
- CURRENT MAIN OBJECTIVE: preserve consumed V4 evidence, prove future protocol ordering before any V5 freeze, and advance Paper Quality only through measured DEVELOPMENT/VALIDATION evidence.
- WHAT WAS JUST COMPLETED: official LR-ASPP run `32675225785` passed the frozen DEVELOPMENT mask gate: macro-F1 `0.711144`, macro-IoU `0.569570`, minimum class F1 `0.423585`; artifact/checkpoint/ONNX hashes are verified. It is not production qualified.
- WHAT IS BEING WORKED ON: expand evaluation of the frozen LR-ASPP artifact to a substantially larger identity-disjoint DEVELOPMENT/VALIDATION bank without retraining. RefFace remains NOT_RUN.
- WHAT IS BLOCKING PROGRESS: V4 cannot certify the Track A candidate. Paper Quality lacks a qualified multi-class damage mask, identity-disjoint multi-identity validation, Windows/offline qualification and the frozen 300–400 identity benchmark.
- WHAT MODEL IS CURRENTLY BEING TESTED: official torchvision LR-ASPP/MobileNetV3 at `pytorch/vision@c6f39778...`; DEVELOPMENT gate PASS on two validation identities, broader identity-disjoint evaluation pending.
- WHY THAT MODEL: the stopped small U-Net failed six damage classes at F1 zero; LR-ASPP is an official lightweight semantic-segmentation architecture with real CPU/ONNX feasibility.
- CURRENT BEST MODEL PER DAMAGE TYPE: mild blur/denoise NAFNet; JPEG FBCNN in DEV only; severe blind face GPEN in DEV identity evidence; opaque/reference-supported loss observed same-person component transfer; unqualified classes preserve MAIN/rollback/abstain.
- CURRENT QUALITY RESULT: FBCNN public DEV matrix 6/6 PASS on one identity. LR-ASPP DEVELOPMENT mask gate PASS on two validation identities (`F1 0.711144`, `IoU 0.569570`, minimum class F1 `0.423585`). Neither is Target95/production evidence. V4 final quality remains NOT_MEASURED; Female #584 Target95 report-only remains `21/304 = 6.91%`.
- CURRENT SAFETY RESULT: V4 pre-consumption checks passed targeted `110/110`, full pytest `547/547`, and calibration `60/60` with zero errors, provenance violations and wrong-person pixels. Final-holdout safety metrics are NOT_MEASURED because the runner failed before case 1. SFace `0.363` and frozen guardrails were unchanged.
- CURRENT WINDOWS STATUS: #1317 `SUCCESS`; exact-HEAD installer, portable package, release metadata, production-model updates and validation artifacts published. Physical EliteBook acceptance remains `NOT_RUN`.
- CURRENT ELITEBOOK STATUS: `NOT_RUN` for PRODUCT_V1_1 and Paper Quality.
- NEXT EXACT STEP: evaluate the frozen LR-ASPP checkpoint/ONNX without retraining on a substantially larger identity-disjoint DEVELOPMENT/VALIDATION bank; keep RefFace blocked and do not create/execute V5.
- ESTIMATED PROJECT COMPLETION STATE: overall `43%` — engineering estimate based on the consumed V4 NO-GO plus independent research, Windows and physical-PC gates below.

Completion estimates: V1.1 operational `76%` (exact-head runtime, release, calibration and packaging gates pass, but certification candidate is V4 NO-GO; new V5 and target-PC acceptance remain); Paper Quality V2 `38%`; personalized restoration `35%`; Windows productization `80%` (CI artifacts pass; physical EliteBook acceptance remains); overall `43%`.

## 0. Document metadata

- Updated: `2026-08-23`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last Track A branch: `hotfix/real-world-restoration-v1.1`
- Track A current branch HEAD: `77687b3b171f4e9989fcf486834f2d8b7a52f591`; evaluated candidate: `b6ce7ebde87d4ce84e5849664716dc3e822ad762`
- Protocol hardening HEAD: `protocol/v5-certification-hardening@268188c5a2540455ff804383cb583b16546b62f1`
- Active Paper Quality HEAD: `research/paper-quality-local-v2@0d4b5fa2d4fad36c9fde484d2ebdac0ae6a61053`
- Track A identity/source/provenance targeted suite: latest exact evidence `108/108 PASS` on Release Quality #134 at `b6ce7ebd...`.
- Current Track A gate: Windows #1317, Release #134 and Female #584 remain SUCCESS on `b6ce7ebd...`. V4 Final Certification #1 (`32656139686`) is FAIL; request `d847798e...`, STARTED marker `d03d97c6...`, final disposition `77687b3b...`. V4 is `CONSUMED_FAIL`; 0/40 cases executed; no rerun.
- Current Track B direction: **UPSTREAM-FIRST**. Official executable paper/model repositories are the architecture baseline; CFS owns thin adapters, identity/provenance safety, resource control, checkpoint/hash verification, Windows/offline packaging and qualification tests.
- Current protocol state: generic `build_freeze` supports both legacy one-argument and generic cases+contract providers by signature binding before execution; synthetic one-shot success and failure-before/after-marker ordering is verified. No V5 exists and no final holdout was accessed.
- Current Track B FBCNN state: **DEVELOPMENT MATRIX PASS, NOT PRODUCTION QUALIFIED**. Run `32674085939`, artifact `9502200502`, SHA-256 `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`; 6/6 profiles PASS.
- Current Track B DamageMaskNet state: attempt 3 evidence recovered from run `32087249287`, artifact `9307331508`, SHA-256 `e3b7aa05...`. Infrastructure/export/parity PASS; macro-F1 `0.173198` and macro-IoU `0.113028`, with six damage classes at F1 zero. Small U-Net hypothesis stopped as MODEL/DATA QUALITY FAIL.

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
| `protocol/v5-certification-hardening` | future protocol DEV hardening only | `268188c5...` | TEST_PASS / no V5 | run `32673504579` SUCCESS; 558/558; artifact `9502021996` | transfer by traceable commit to a future candidate only after quality prerequisites |
| `research/face-restoration-v2` | early data/degradation research | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `0d4b5fa2...` | ACTIVE / BENCHMARKING | LR-ASPP run `32675225785` SUCCESS; frozen DEV gate PASS; artifact `9502642834` verified | expand identity-disjoint validation; resolve checkpoint licensing |
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
OBJ-004 DamageMaskNet — MODEL/DATA QUALITY FAIL; small U-Net hypothesis STOPPED.
OBJ-005 Broad BFR selection — IN_PROGRESS, upstream-first.  
OBJ-006 FBCNN JPEG qualification — DEVELOPMENT MATRIX PASS 6/6; multi-identity/Windows/offline qualification IN_PROGRESS.
OBJ-007 Personalized Reference Bank validation — IN_PROGRESS.  
OBJ-008 RefFace CPU — BLOCKED by OBJ-004, 0/3 attempts.  
OBJ-009 Paper Quality Windows pack — PROPOSED.  
OBJ-010 HP EliteBook acceptance — PROPOSED.  
OBJ-011 Official upstream implementation registry — IMPLEMENTED; runtime qualification remains per-model.  
OBJ-012 Official upstream adapters — IN_PROGRESS; FBCNN is first concrete implementation.
OBJ-013 Future one-shot protocol hardening — DEV TEST_PASS; V5 NOT_CREATED/NOT_AUTHORIZED.

---

## 6. MODEL MASTER REGISTRY

Certified roles: YuNet, SFace `0.363`, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research: GPEN BENCHMARKING/license blocker; GFPGAN1.4 BENCHMARKING; CodeFormer BENCHMARKING/BLOCKED_LICENSE; FBCNN DEVELOPMENT_MATRIX_PASS/current compression leader; DamageMaskNet small U-Net STOPPED_MODEL_DATA_FAIL; LR-ASPP DEVELOPMENT_MASK_ADEQUACY_PASS/NOT_PRODUCTION_QUALIFIED; RefFace FEASIBILITY_ONLY/NOT_RUN; InstantRestore/OSDFace hardware-blocked feasibility; RestoreFormer++/VQFR/GPEN-inpainting/RefineFIR/PerFuSe/RefIPFR/Real-ESRGAN feasibility until measured.

Official repository registry: `config/upstream-implementations.json`. Initial pinned source baselines: GPEN `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802`; GFPGAN `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679`; CodeFormer `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`; FBCNN `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`; InstantRestore `snap-research/InstantRestore@05891bf7d30ab7290c501272de7a1a4a51b21b4f`. RefineFIR, RefFaceInpainting, OSDFace and RestoreFormer++ remain `NOT_VERIFIED` until revision/checkpoint/license/runtime qualification.

FBCNN upstream contract at `6ea5d113...`: official repository source only; Apache-2.0 code; `fbcnn_color.pth` official v1.0 asset; exact source revision fixed. CFS `app/fbcnn_upstream_backend.py` dynamically imports the official `models/network_fbcnn.py`, requires `.cfs-upstream.json`, requires an explicit 64-hex checkpoint SHA-256, rejects non-JPEG/recompression routes, runs CPU-only and marks model-modified pixels `GENERATED_MODEL_INFERRED`. CFS does **not** contain a copied `class FBCNN` architecture.

Registry documentation/package-manifest mismatch from V1 remains separate; never invent missing manifests/hashes.

---

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEV historical evidence: GPEN SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, `~1.828GB`; GFPGAN1.4 SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`; FBCNN QF20 SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, `~1.305GB`; CodeFormer real CPU slice PASS, exact metrics artifact-required.

FBCNN evidence is now recovered and extended. The official source is `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd`, Apache-2.0; official checkpoint `fbcnn_color.pth` is `287755111` bytes with SHA-256 `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`. Run `32674085939` executed six public DEV profiles: all 6 PASS; errors 0; rollbacks 0; abstentions 0; wrong-person pixels 0; provenance violations 0. Artifact `9502200502`, archive SHA-256 `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`. This does not qualify production because only one public identity was used and Windows/offline/EliteBook were not tested.

DamageMaskNet attempt 3 produced real checkpoint SHA-256 `e3b05272782aded20f209ddd39a3ac847cf4f3a90e5e3f02b63cae90474e2b7d` and ONNX SHA-256 `64e032d8693edc55d69a0a77d8665034d4edbeff43a93b6a622c4639a0d018c7`; ONNX argmax parity was exact and first CPU inference `0.01282s`. Mask quality failed: macro-F1 `0.173198`, macro-IoU `0.113028`; BLUR, MOTION_BLUR, PIXELATION, BLOCK_MOSAIC, JPEG_ARTIFACT and STICKER all had F1 zero.

LR-ASPP run `32675225785`, exact Track B `2b775b81...`, used official `pytorch/vision@c6f39778...` and passed the frozen DEVELOPMENT mask gate. Artifact `9502642834`, archive SHA-256 `0bef114cfeed95ebcceb81ce8f5dfc43c3fdb37bca82c69a346ed6219c137a11`; trained checkpoint `d510e6991cca582c3696b6b9132bf3fdb7948e240f4bf136440d8b75046910f4`; ONNX `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`. ONNX argmax parity exact; CPU first call `0.00675s`. Validation is only two identities, visual boundary/class errors remain and checkpoint redistribution license is not explicit; production and RefFace remain blocked.

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

DEC-001 canonical ledger ACCEPTED. DEC-002 active Paper Quality branch ACCEPTED. DEC-003 <=80% CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet bank ACCEPTED. DEC-006 RefFace after adequate DamageMaskNet/replacement mask ACCEPTED/BLOCKED. DEC-007 V3/V4 consumed and immutable ACCEPTED. DEC-008 ranking cluster != identity authority ACCEPTED/CLOSED after 3/3. DEC-009 localized-damage same-canvas edge isolation ACCEPTED. **DEC-010 official-upstream-first model integration — ACCEPTED. DEC-011 FBCNN thin upstream adapter — ACCEPTED; DEVELOPMENT matrix PASS, production qualification pending. DEC-012 future one-shot lifecycle — ACCEPTED on synthetic DEV only:** preflight completes before consumption, marker is immediately before first case access, and failures after marker are terminal. **DEC-013 DamageMaskNet small U-Net — STOPPED:** recovered attempt 3 is a model/data quality failure; no tuning or rerun of this hypothesis.

---

## 14. EXPERIMENT LOG

DamageMaskNet 1/3 403 infra fail, 2/3 429 infra fail, 3/3 MODEL/DATA QUALITY FAIL; small U-Net hypothesis stopped. Historical later path-triggered runs already existed before this audit and were not launched or used for tuning. RefFace PREPARED/NOT_RUN because mask quality is inadequate.

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

**EXP-20260823-022 protocol hardening DEV:** branch `protocol/v5-certification-hardening@268188c5...`, direct child of Track A `77687b3b...`; generic freeze signature adapter, callback-driven one-shot lifecycle, synthetic DEV runner, failure injection and exact entrypoint tests. Local and remote full suite `558/558`; run `32673504579` SUCCESS; artifact `9502021996`, SHA-256 `e1265e92164a83b5d5a066d4fdd6635df3d023af4ebf4d9895a618ece78c6930`. No V5 or holdout created/executed.

**EXP-20260823-023 DamageMask attempt 3 recovery:** run `32087249287` and artifact `9307331508` were recovered without rerun. Archive SHA-256 `e3b7aa05...`; checkpoint/ONNX hashes above; export, loader, CPU inference and parity PASS. Per-class mask quality failed materially, so the hypothesis is STOPPED and RefFace remains gated.

**EXP-20260823-024 FBCNN compression DEV matrix:** Track B `7dfeb0a8...`, run `32674085939` SUCCESS. Six profiles completed and all passed identity, PSNR, SSIM, provenance and wrong-person guardrails. QF10 block-heavy PSNR `30.8117->33.6059`, SFace `0.8504->0.8954`; QF20 `34.6184->36.7801`, `0.9571->0.9691`; QF40 `38.6756->39.7644`, `0.9837->0.9877`; double-JPEG `32.6772->35.1038`, `0.9074->0.9371`; social recompression `33.5442->35.1728`, `0.9513->0.9561`; mosquito stress `31.8589->34.1065`, `0.8900->0.9139`. Single-identity DEV only; production NOT_QUALIFIED.

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

Active `research/paper-quality-local-v2@0d4b5fa2d4fad36c9fde484d2ebdac0ae6a61053`: real CPU BFR/JPEG evidence, 80% governor, Personalized Reference Bank, reference-first repair, hard-gated selector, deterministic fusion, parser adapter, gated RefFace workflow, official-upstream registry/bootstrap, FBCNN DEV matrix and documented LR-ASPP DEVELOPMENT result.

`config/upstream-implementations.json` encodes the upstream-first contract. Registry/bootstrap and FBCNN source/hash/route/matrix tests pass locally and in run `32674085939`. Local full Track B pytest is `557/557 PASS`; test isolation no longer leaves a fake `app.reference_inpainting` module in the shared process.

FBCNN integration uses official code directly and has public DEV evidence for five compression families. DamageMask runtime workflow `32674085927` is SUCCESS. Progress timeline workflow `32674575985` is SUCCESS after declaring its ONNX Runtime dependency; all seven timeline contract tests passed and no artifact is expected from that contract-only workflow.

LR-ASPP source is pinned to `pytorch/vision@c6f39778e636ec40a69bdbc74386818c57a65af3` (`v0.16.2`), BSD-3-Clause code. Run `32675225785` passed the frozen mask thresholds macro-F1 `0.70`, macro-IoU `0.55` and every damage class F1 `0.35`. Checkpoint redistribution license is not explicit, so production qualification remains blocked regardless of DEV quality.

PDF constraints remain: separate global identity from local texture; use correspondence between matching regions; region-adaptive identity guidance for severe BFR; MAIN preserves pose/composition/expression/geometry; unsupported detail remains conservative. Paper-reported metrics and CFS-reproduced metrics remain separate.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Evaluate the frozen LR-ASPP artifact on a substantially larger identity-disjoint DEVELOPMENT/VALIDATION bank without retraining. The small U-Net is stopped; do not tune it. RefFace remains NOT_RUN until broader mask quality and compatible licensing pass.

In parallel: broaden the now-passing FBCNN DEV matrix to multiple identity-disjoint DEVELOPMENT/VALIDATION identities, then Windows/offline/EliteBook. Next upstream heavy candidates remain GPEN/GFPGAN/CodeFormer, then InstantRestore if CPU/Windows feasibility is credible. RefineFIR/RefFaceInpainting/OSDFace/RestoreFormer++ remain NOT_VERIFIED until exact revision/checkpoint/license/runtime evidence exists.

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
- HIST-20260823-028 protocol hardening branch `268188c5...`: synthetic one-shot lifecycle and generic freeze adapter remote PASS; V5 not created.
- HIST-20260823-029 Track B `1591fa3c... -> 7dfeb0a8...`: DamageMask attempt 3 recovered/failed quality; FBCNN six-profile DEV matrix PASS; runtime CI repaired; timeline CI still missing ONNX Runtime.
- HIST-20260823-030 Track B `94b37f13...`: timeline workflow dependency declared; run `32674575985` SUCCESS; no model, data, threshold or holdout change.
- HIST-20260823-031 Track B `2b775b81...`: official LR-ASPP comparison prepared and launched once on DEVELOPMENT; local loader/export smoke and 563/563 tests PASS; remote result pending.
- HIST-20260824-032 LR-ASPP run `32675225785` SUCCESS; frozen DEVELOPMENT mask gate PASS; artifact/checkpoint/ONNX verified; production/RefFace remain blocked by validation scale and checkpoint licensing.

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

### PUSH-20260823-004

- DATE/TIME UTC: `2026-08-23`.
- TECHNICAL BRANCH: `protocol/v5-certification-hardening`.
- BASE/PREVIOUS HEAD: `77687b3b171f4e9989fcf486834f2d8b7a52f591`.
- NEW REMOTE HEAD: `268188c5a2540455ff804383cb583b16546b62f1`.
- COMMITS INCLUDED: `Harden generic one-shot certification runner`.
- FILES MODIFIED/ADDED: generic frozen-benchmark adapter; generic one-shot lifecycle; synthetic DEV runner; production baseline entrypoint integration; unit/E2E tests; DEV-only workflow.
- TESTS: targeted `11/11 PASS`; full pytest local and remote `558/558 PASS`; compile and diff checks PASS.
- WORKFLOW: `Protocol V5 hardening DEV`, run `32673504579`, SUCCESS; every job step passed.
- ARTIFACT: `protocol-v5-hardening-32673504579`, ID `9502021996`, archive SHA-256 `e1265e92164a83b5d5a066d4fdd6635df3d023af4ebf4d9895a618ece78c6930`.
- PROTOCOL RESULT: success path writes STARTED immediately before first synthetic case access; injected pre-marker failure remains PRECONSUMPTION_FAIL with no marker/case access; injected post-marker failure is terminal CONSUMED_FAIL with no case access.
- HOLDOUT/V5 EFFECT: none. V3/V4 evidence unchanged; no V5 request, benchmark, candidate or holdout was created or executed.
- NEXT EXACT ACTION: transfer the tested fix to any future candidate by traceable commit only after quality prerequisites; continue Track B research meanwhile.

### PUSH-20260823-005

- DATE/TIME UTC: `2026-08-23`.
- TECHNICAL BRANCH: `research/paper-quality-local-v2`.
- PREVIOUS HEAD: `1591fa3cebaa44a3ef80a6a2178ed85fd1ae2d66`.
- NEW REMOTE HEAD: `7dfeb0a855f3f7c0840693bb2c03c25bc498d4eb`.
- COMMITS INCLUDED: `30eaf364...` research CI isolation; `ca985fe9...` recovered-model audit; `7dfeb0a8...` FBCNN compression matrix.
- FILES MODIFIED/ADDED: three research workflows, Paper Quality status, DEV requirements, FBCNN matrix/runner/summarizer, FBCNN tests and reference-first test isolation.
- TESTS: targeted `36/36 PASS`; final full local pytest `557/557 PASS`; compile and diff checks PASS.
- WORKFLOWS: FBCNN matrix run `32674085939` SUCCESS; DamageMask runtime run `32674085927` SUCCESS; progress-timeline run `32674085920` FAILURE on missing `onnxruntime` in its narrow environment.
- FBCNN RESULT: 6/6 public DEV profiles PASS; 0 errors; 0 rollbacks; 0 abstentions; 0 wrong-person pixels; 0 provenance violations. This is not production/Target95 evidence.
- FBCNN ARTIFACT: `fbcnn-compression-dev-matrix-4`, ID `9502200502`, archive SHA-256 `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`.
- DAMAGEMASK ATTEMPT 3: recovered without rerun from run `32087249287`, artifact `9307331508`; real checkpoint/ONNX and CPU parity PASS, mask quality MODEL/DATA FAIL. Small U-Net hypothesis STOPPED; RefFace remains NOT_RUN.
- NEXT EXACT ACTION: add ONNX Runtime to the progress-timeline workflow and verify the exact new HEAD; then benchmark the next lightweight mask architecture without changing the frozen taxonomy/acceptance contract.

### PUSH-20260823-006

- DATE/TIME UTC: `2026-08-23`.
- TECHNICAL BRANCH: `research/paper-quality-local-v2`.
- PREVIOUS HEAD: `7dfeb0a855f3f7c0840693bb2c03c25bc498d4eb`.
- NEW REMOTE HEAD: `94b37f131d0bef5aca1081d7baaef46f9e4d6bf7`.
- COMMIT INCLUDED: `Declare timeline runtime dependency`.
- FILE MODIFIED: `.github/workflows/research-progress-timeline.yml` only; adds the demonstrated `onnxruntime>=1.20,<2` dependency.
- TESTS: local `tests/test_progress_timeline.py` `7/7 PASS`; remote workflow test step PASS.
- WORKFLOW: `Research progress timeline`, run `32674575985`, SUCCESS; all job steps PASS; no artifact is defined for this contract-only workflow.
- EFFECT: infrastructure-only repair. No model, checkpoint, data, mask contract, threshold, V3/V4 evidence or holdout changed.
- NEXT EXACT ACTION: benchmark the next lightweight damage-localization architecture against the unchanged DEVELOPMENT contract; keep RefFace blocked until adequate mask quality is demonstrated.

### PUSH-20260823-007

- DATE/TIME UTC: `2026-08-23`.
- TECHNICAL BRANCH: `research/paper-quality-local-v2`.
- PREVIOUS HEAD: `94b37f131d0bef5aca1081d7baaef46f9e4d6bf7`.
- NEW REMOTE HEAD: `2b775b8186ac974f568b3644c59350cc1f12181a`.
- COMMIT INCLUDED: `Add official LRASPP damage mask comparison`.
- FILES MODIFIED/ADDED: LR-ASPP source/checkpoint contract, thin official adapter, trainer/offline loader/export evidence, six contract tests, dedicated push-only workflow and Paper Quality status.
- UPSTREAM: `pytorch/vision@c6f39778e636ec40a69bdbc74386818c57a65af3`, code BSD-3-Clause; MobileNetV3 backbone `8738ca79...`, `22139423` bytes; checkpoint license not explicit/research-only.
- TESTS: contract `6/6 PASS`; full local pytest `563/563 PASS`; real loader shape `1x12x192x192`, 308 tensors loaded; synthetic full-entrypoint smoke checkpoint reload drift `0`, ONNX argmax parity exact and max logit drift `2.03e-6`.
- WORKFLOW: `Research DamageMask LRASPP comparison`, run `32675225785`, in progress at this ledger update.
- SAFETY: unchanged DEVELOPMENT taxonomy/source-bank; no V3/V4/final holdout; stopped U-Net not relaunched; RefFace not executed; model emits mask logits and cannot introduce person pixels.
- NEXT EXACT ACTION: monitor run `32675225785`, verify artifact hashes and classify the frozen quality gate without retuning.

### PUSH-20260824-008

- DATE/TIME UTC: `2026-08-24`.
- TECHNICAL BRANCH: `research/paper-quality-local-v2`.
- PREVIOUS HEAD: `2b775b8186ac974f568b3644c59350cc1f12181a`.
- NEW REMOTE HEAD: `0d4b5fa2d4fad36c9fde484d2ebdac0ae6a61053`.
- COMMIT INCLUDED: `Record LRASPP development mask result`.
- FILES MODIFIED/ADDED: Paper Quality status and `research/LRASPP_DAMAGE_MASK_RESULT.md`; evidence-only documentation, no model/data/threshold/workflow change.
- WORKFLOW RESULT: run `32675225785` SUCCESS; every step PASS; exact training time `715.378s`.
- ARTIFACT: `damage-mask-lraspp-dev-1`, ID `9502642834`, archive SHA-256 `0bef114cfeed95ebcceb81ce8f5dfc43c3fdb37bca82c69a346ed6219c137a11`.
- MODEL ARTIFACTS: checkpoint SHA-256 `d510e6991cca582c3696b6b9132bf3fdb7948e240f4bf136440d8b75046910f4`; ONNX SHA-256 `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`.
- QUALITY: macro-F1 `0.711144`; macro-IoU `0.569570`; minimum class F1 `0.423585`; all three frozen DEVELOPMENT checks PASS. Two validation identities only; Target95 NOT_MEASURED.
- RUNTIME/SAFETY: offline reload drift `0`; ONNX argmax exact, max logit drift `3.81e-5`; ONNX CPU first call `0.00675s`; mask-only model, wrong-person pixels `0`, provenance violations `0`.
- CLASSIFICATION: DEVELOPMENT_MASK_ADEQUACY_PASS_NOT_PRODUCTION_QUALIFIED. RefFace NOT_RUN; Windows/EliteBook NOT_RUN; checkpoint redistribution license not explicit.
- NEXT EXACT ACTION: evaluate the frozen checkpoint/ONNX without retraining on a substantially larger identity-disjoint DEVELOPMENT/VALIDATION bank.

---

## 26. SESSION START/END CONTINUITY RULE

Every session reads this ledger, reconciles GitHub, and continues the recorded blocker. Every legitimate technical push is followed by its exact remote SHA and evidence here. Ledger commits never attempt to record their own SHA.
