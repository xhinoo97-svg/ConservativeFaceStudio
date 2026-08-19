# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current-state sections are maintained; decisions, experiments and important pushes/failures are preserved.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Last technical HEAD: `7c683eddad579974329bb186622aea01feccee61`
- Active tracks: A — PRODUCT_V1_1 stabilization; B — Paper Quality research
- Exact Track A blocker: identity-policy hypothesis has consumed **2/3** implementation attempts. Attempt 2 reduced Release Quality targeted failures to **1 failed, 107 passed**. Remaining failure is a legacy `main_bridged_cross_reference_cluster` flag surviving when the preflight matrix is absent.
- Exact Track B blocker: recover already-triggered DamageMaskNet attempt-3 evidence without launching attempt 4 for observability.

FORENSIC_MODE_READY: **TRUE for PRODUCT_V1 only**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL — historical V1 only**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory protocol: `technical push -> exact SHA -> ledger update` before another technical push.

## 1. Executive project summary

CFS is a local Windows face-restoration system for damaged smartphone/social-media portraits. Conservative Mode uses MAIN and verified same-person observed evidence with explicit provenance; Paper Quality Mode may synthesize missing detail but generated pixels remain `GENERATED_MODEL_INFERRED`.

PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is an operational/safety hotfix isolated from experimental Paper Quality models. Track B contains real CPU model evidence and research routing/reference/fusion infrastructure.

The current Track A design separates **ranking clusters** from **identity authority**. Single-link SFace components can rank images but cannot create transitive identity authority. Attempt 2 correctly fixed partial same-canvas global trust, runtime-reordered direct anchors and legacy missing-engine compatibility; one legacy cluster-only flag still requires explicit fail-closed rejection when matrix evidence is unavailable.

## 2. Branch and release map

| Branch | Purpose | Verified HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f...` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | original V1 branch | `5eff6673...` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate | `f476c6f0...` | FROZEN / ARCHIVED | merged via PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `7c683edd...` | BLOCKED / ACTIVE | PR #2 OPEN/DRAFT; Release Quality FAIL `1/107`; Windows/Female attempt-2 runs were still in progress at the check that triggered this ledger update | final identity attempt 3/3 |
| `research/face-restoration-v2` | early data/degradation research | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `645862d1...` | ACTIVE / BENCHMARKING | partly NOT_VERIFIED | DamageMaskNet evidence recovery |
| `meta/project-state` | canonical ledger | self-SHA intentionally omitted | ACTIVE META | docs only | update after every technical push |

The two research branches diverged from the same certified base; the advanced branch is not falsely described as containing the early commits.

## 3. PRODUCT VERSION ROADMAP

- **PRODUCT_V1 — RELEASED:** certified conservative baseline; SFace `0.363`, wrong-person observed `0`, provenance violations `0`; historical Windows certification.
- **PRODUCT_V1_1 — IMPLEMENTING/BLOCKED:** operational real-world safety hotfix. Current identity-policy implementation is at final attempt 3/3 boundary; no Track B generators.
- **PRODUCT_V2 — BENCHMARKING:** Paper Quality Local; damage-aware specialists, BFR candidates, generated provenance, hard identity gates, deterministic fusion, 80% resource contract.
- **PRODUCT_V3 — PLANNED with prototypes:** MAIN + 0–9 personalized references, per-component authority.
- **PRODUCT_V4 — PLANNED with prototypes:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified modes + offline Windows pack/installer + real EliteBook acceptance.

Product versions are separate from HOLDOUT versions.

## 4. HOLDOUT / BENCHMARK LINEAGE

- CALIBRATION_V1: historical 60/60 certified evidence.
- FINAL_HOLDOUT_V1: historical 40/40; no tuning.
- FINAL_HOLDOUT_V2: exact current details NOT_VERIFIED; do not use.
- FINAL_HOLDOUT_V3: **CONSUMED**, 39/40; `medium_block_mosaic`, SFace `0.360 < 0.363`; NEVER rerun/tune.
- FINAL_HOLDOUT_V4: 40 cases/20 identities, frozen ControlFace10K, **NOT_RUN / UNCONSUMED**; execute only through valid one-shot protocol.
- FINAL_HOLDOUT_V5: not created.
- Female-domain: ~300–400-case stress profile; safety hard gates, quality report-only.
- Paper Quality DEV/VALIDATION: incomplete but active.
- DamageMaskNet bank: FairFace + ControlFace, exact synthetic masks, TRAIN/VALIDATION only.

Attempt-2 Release Quality reverified V3 manifests and V4 freeze/history without executing either holdout.

## 5. CURRENT GLOBAL OBJECTIVES

- OBJ-001 Preserve PRODUCT_V1 — **PASS**.
- OBJ-002 Restore PRODUCT_V1_1 gates — **BLOCKED / final implementation attempt available**. Attempt 2 = `1 failed,107 passed`; remaining failure is cluster-only legacy flag without matrix.
- OBJ-003 Canonical ledger — **IN_PROGRESS**; current CI evidence recorded before attempt 3.
- OBJ-004 DamageMaskNet — **BLOCKED** pending attempt-3 evidence recovery.
- OBJ-005 Blind BFR selection — **IN_PROGRESS**; broad identity-disjoint DEV/VALIDATION required.
- OBJ-006 FBCNN JPEG qualification — **IN_PROGRESS**.
- OBJ-007 Personalized Reference Bank validation — **IN_PROGRESS**.
- OBJ-008 RefFace CPU feasibility — **BLOCKED** by DamageMaskNet sequencing; 0/3 attempts consumed.
- OBJ-009 Paper Quality Windows pack — **PROPOSED**.
- OBJ-010 Real EliteBook acceptance — **PROPOSED**.

## 6. MODEL MASTER REGISTRY

Certified-role stack: YuNet, SFace, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research states:
- GPEN BFR-512 — **BENCHMARKING / licensing blocker**; DEV SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474`.
- GFPGAN v1.4 — **BENCHMARKING**; SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604`.
- CodeFormer w=0.5 — **BENCHMARKING / BLOCKED_LICENSE**; real CPU slice PASS, exact comparative artifact metrics must be reread.
- FBCNN — **BENCHMARKING / current DEV JPEG leader**; QF20 PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, SFace `0.9571→0.9691`, peak RSS `~1.305GB`.
- DamageMaskNet U-Net — **BENCHMARKING / BLOCKED**, attempt-3 result NOT_VERIFIED.
- RefFaceInpainting — **FEASIBILITY_ONLY / NOT_RUN**, official MIT repo, minimal CPU path prepared.
- InstantRestore — **FEASIBILITY_ONLY / BLOCKED_HARDWARE+license**.
- OSDFace — **FEASIBILITY_ONLY / BLOCKED_HARDWARE**.
- RestoreFormer++, VQFR, GPEN inpainting, RefineFIR, PerFuSe, RefIPFR, Real-ESRGAN — **FEASIBILITY_ONLY/DISCOVERED** until measured.

Registry documentation issue remains: `THIRD_PARTY_MODULES.md` references missing `models/` manifests; actual active registry is code-based. Never invent files/hashes.

## 7. CURRENT MODEL EVIDENCE

Development Linux CPU only: GPEN/GFPGAN/FBCNN values above; CodeFormer exact artifact required. None of these are Windows/EliteBook qualification.

## 8. 13-BLOCK ARCHITECTURE

1 IMPORT deterministic. 2 DEBLUR NAFNet -> measured BFR candidates later. 3 ENHANCE FBCNN for JPEG. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK parser + future verified DamageMaskNet. 7 REGION_SELECT component bank -> Personalized Reference Bank. 8 INPAINT observed refs first, Paper generators only as GENERATED. 9 FUSION healthy MAIN > observed ref > generated. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`, current Track A trusted-anchor work. 12 UPSCALE Lanczos, optional benchmarked SR. 13 EXPORT deterministic provenance plus future model/damage/resource reports.

## 9. PHOTO AND INPUT CONTRACT

MAIN: low-res smartphone/social-media, JPEG/double JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black bar/opaque loss, covered/missing components, crop/partial face, low-light/uneven exposure, mixed damage. MAIN defines conservative canvas/pose/frame.

References: MAIN + 0–9; full/partial/component-only/angle/expression/light/resolution/blur/compression/occlusion/useless/wrong-person. Full accepted same-person may be global anchor; partial is local only; wrong-person never anchor/donor/identity-score improver.

## 10. DATASET CONSTRUCTION

Initial Paper Quality target ~300–400 representative cases with explicit female-domain proportion. Identity-disjoint TRAIN/DEV/VALIDATION/FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/mask/reference relationships. Never tune/train on final holdout.

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Track LEFT/RIGHT EYE, LEFT/RIGHT EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT/RIGHT CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Per component: MAIN visibility/damage, best/alternate refs, confidence, generated candidate, selected source/provenance, identity/geometry, unresolved state. Observed same-person evidence outranks generation.

## 12. DAMAGE ROUTING

HEALTHY preserve MAIN; BLUR -> NAFNet/measured deblur then Paper BFR only if needed; NOISE -> current measured denoiser; JPEG/DOUBLE_JPEG -> FBCNN; PIXELATION/MOSAIC -> observed component first then Paper generation; SCRIBBLE/STICKER/OPAQUE/BLACK_BAR -> observed reference first then qualified RefFace-like specialist; MISSING/PARTIAL_OCCLUSION -> component bank then Paper generated fallback; LOW_LIGHT -> detected specialist only; MIXED -> minimum necessary specialist candidates, never blind generator chaining.

## 13. DECISION LOG

DEC-20260819-001 canonical meta ledger — ACCEPTED.  
DEC-20260819-002 advanced Paper Quality branch active, early branch preserved — ACCEPTED.  
DEC-20260819-003 <=80% CPU/process/system RAM, one heavy model — ACCEPTED.  
DEC-20260819-004 evidence authority order — ACCEPTED.  
DEC-20260819-005 mixed FairFace+ControlFace DamageMaskNet bank — ACCEPTED.  
DEC-20260819-006 RefFace next large-occlusion specialist after DamageMaskNet — ACCEPTED/BLOCKED.  
DEC-20260819-007 V3 consumed, V4 untouched one-shot — ACCEPTED.  
DEC-20260819-008 ranking cluster != identity authority — ACCEPTED DIRECTION. Final implementation distinction: explicit cluster-promotion reasons must fail closed without direct matrix evidence; direct current SFace flags may survive missing matrix.

## 14. EXPERIMENT LOG

EXP-20260817-001 GPEN — DEV PASS/BENCHMARKING.  
EXP-20260817-002 GFPGAN1.4 — DEV PASS/BENCHMARKING.  
EXP-20260817-003 CodeFormer — packaging fail then real CPU PASS; license blocked.  
EXP-20260817-004 FBCNN — QF20 DEV improvement.  
EXP-20260818-005 DamageMaskNet 1/3 — infrastructure FAIL 403.  
EXP-20260818-006 DamageMaskNet 2/3 — infrastructure FAIL 429.  
EXP-20260818-007 DamageMaskNet 3/3 — result NOT_VERIFIED; no attempt4 for observability.  
EXP-20260819-008 RefFace — PREPARED/NOT_RUN.  
EXP-20260819-009 Track A attempt1/3 `3e919f7a...` — FAIL `4 failed,102 passed`.  
EXP-20260819-010 Track A attempt2/3 `7c683edd...` — **FAIL `1 failed,107 passed`**. Fixed three of four attempt-1 families. Remaining `test_missing_preflight_matrix_fails_closed_for_cluster_promoted_flag`: source1 reason `direct_sface` must survive; source2 reason `main_bridged_cross_reference_cluster` must be revoked even though its flag+score exist and matrix is absent. Release Quality artifact ID `9351701070`, zip SHA256 `d5a6a9e7766796ed8b43fb40f5230bf6448b31c1655076dd234553a41cf5a97c`. Final attempt 3/3 is justified as one narrow reason-aware fail-closed correction.

## 15. QUALITY SCOREBOARD

DEV model evidence exists; broad validation incomplete. V3 holdout consumed 39/40. V4 frozen/unexecuted. Target-PC Paper Quality NOT_RUN. Maintain metrics separately by DEV, VALIDATION, HOLDOUT, REAL-WORLD, TARGET-PC.

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU runtime-detected. CPU-first, no CUDA requirement. <=80% logical CPU, <=80% process/system RAM, one heavy model. Optional acceleration only after support/parity evidence.

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed `0`; provenance violations `0`; healthy/outside MAE `<=8.0` where frozen policy applies. No threshold-shopping, cherry-picking, consumed-holdout reruns, hard-case deletion, generated-as-observed, wrong-person max-score rescue, auto-merge, force-push certified history or fabricated evidence.

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

## 19. TRACK A — PRODUCT_V1_1

Historical `3645c8c...`: Release Quality FAIL `4 failed,195 passed`; Windows/Female FAIL.

Attempt1 `3e919f7a...`: Release Quality FAIL `4 failed,102 passed`; established partial/global/reorder/missing-engine distinctions.

Attempt2 `7c683edd...`: Release Quality #125 FAIL only `1 failed,107 passed`. V3/V4 verification PASS with no execution. The sole failure is legacy cluster-only flag survival when no preflight matrix exists. Direct reason `direct_sface` must remain; `main_bridged_cross_reference_cluster` must be explicitly revoked. Windows/Female attempt-2 runs were not queried after this Release Quality failure because they were still in progress.

**Final attempt 3/3 exact action:** modify only V4 hardening reason filtering so current flag+numeric score is preserved without matrix **unless its existing reason explicitly identifies cluster promotion** (`main_bridged_cross_reference_cluster` / `same_canvas_bridged_cross_reference_cluster`). Apply the same rule in bridge and downstream trusted-source paths. No threshold/model/preflight/holdout change. Then targeted tests. If still FAIL, stop this hypothesis and reassess instead of attempt4.

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`. Real CPU BFR/JPEG evidence plus 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, candidate selector, deterministic fusion, parser adapter, RefFace manual workflow. Models must converge to qualified/rejected/documented blockers.

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1 + ONNX parity + RAM/runtime. Infrastructure FAIL -> fix infrastructure only. Model/data FAIL -> U-Net hypothesis ends. Then RefFace attempt1/3.

## 22. SPECIALIST MODEL STRATEGY

`input -> align -> damage -> references/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance`. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference specialist. Never blindly chain all generators.

## 23. MODEL SELECTION POLICY

Select BFR winners only on multiple identity-disjoint DEV/VALIDATION cases, per degradation. Identity hard gate precedes ranking. Measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM. Never select on final holdout.

## 24. HISTORICAL RECORD — append-only

- HIST-20260815-001: PRODUCT_V1 certified merge `f476c6f... -> 2767513f...`, historical Windows #1195/Female #463/Release Quality #13.
- HIST-20260818-002: Track A blocked `3645c8c...`, all three release workflows FAIL; V3 consumed; V4 frozen/unexecuted.
- HIST-20260818-003: Paper Quality `645862d1...`, real DEV evidence; DamageMaskNet attempt3 NOT_VERIFIED; RefFace NOT_RUN.
- HIST-20260819-004: `meta/project-state` canonical ledger created.
- HIST-20260819-005: Track A attempt1 technical push `3645c8c... -> 3e919f7a...`, no threshold/model change.
- HIST-20260819-006: attempt1 CI `4 failed,102 passed`, Windows FAIL, V3/V4 verification-only PASS.
- HIST-20260819-007: attempt2 technical push `3e919f7a... -> 7c683edd...`, trusted-anchor/full-vs-partial/legacy-engine refinement, no threshold/model change.
- **HIST-20260819-008:** exact attempt2 Release Quality #125 result: `1 failed,107 passed`; sole failure `test_missing_preflight_matrix_fails_closed_for_cluster_promoted_flag`; artifact `9351701070`, digest `d5a6a9e7766796ed8b43fb40f5230bf6448b31c1655076dd234553a41cf5a97c`. Attempt2/3 consumed; one final reason-aware implementation attempt remains.
