# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current state is maintained here; important decisions, experiments, failures and technical pushes are preserved.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Previous technical HEAD: `7c683eddad579974329bb186622aea01feccee61`
- Last technical HEAD: `84640fb7cee273bb0f5dcb19b4f9a1f3908e583e`
- Track A state: identity-policy implementation hypothesis has used its **3rd and final evidence-based attempt**; result NOT_VERIFIED. No attempt 4 is allowed under the same hypothesis.
- Track B blocker: recover the already-triggered DamageMaskNet attempt-3 evidence without rerun for observability.

FORENSIC_MODE_READY: **TRUE for certified PRODUCT_V1 only**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL — historical V1 only**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence after every technical push: technical evidence -> exact technical SHA -> ledger update. No auto-merge or certified-history force push.

## 1. Executive project summary

CFS is a local Windows face-restoration application for difficult phone/social-media portraits. Conservative Mode treats MAIN and verified same-person observed pixels as evidence with strict provenance; Paper Quality Mode may generate missing detail only as `GENERATED_MODEL_INFERRED`.

PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is a safety/operational hotfix. Track B is a real ML research track with CPU model evidence and damage/reference/fusion infrastructure.

Track A currently enforces: ranking cluster != identity authority; global identity requires whole-face direct evidence or a separately verified safe bridge; partial same-canvas evidence remains local; explicit cluster-promotion flags fail closed. Attempt 3 only adds the final reason-aware cluster rejection required by attempt-2 evidence.

## 2. Branch and release map

| Branch | Purpose | HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f...` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | V1 implementation history | `5eff6673...` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate | `f476c6f0...` | FROZEN / ARCHIVED | merged PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `84640fb7...` | VALIDATING / ACTIVE | PR #2 OPEN/DRAFT; exact-head result NOT_VERIFIED | evaluate final attempt; if fail, stop/reassess hypothesis |
| `research/face-restoration-v2` | early research | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `645862d1...` | ACTIVE / BENCHMARKING | partly NOT_VERIFIED | DamageMaskNet evidence recovery |
| `meta/project-state` | canonical ledger | self-SHA omitted | ACTIVE META | docs only | update after each technical push |

## 3. PRODUCT VERSION ROADMAP

- PRODUCT_V1 — **RELEASED:** certified forensic/conservative baseline.
- PRODUCT_V1_1 — **VALIDATING:** operational hotfix; current identity hypothesis at final attempt 3/3.
- PRODUCT_V2 — **BENCHMARKING:** Paper Quality Local, modern priors/specialists, generated provenance, hard gates, 80% resource contract.
- PRODUCT_V3 — **PLANNED with prototypes:** personalized MAIN + 0–9 refs, component-level authority.
- PRODUCT_V4 — **PLANNED with prototypes:** damage-specialist hybrid routing/fusion.
- PRODUCT_V5 — **PLANNED:** unified modes + offline Windows pack/installer + real target-PC acceptance.

Product versions are separate from evaluation/holdout versions.

## 4. HOLDOUT / benchmark lineage

- CALIBRATION_V1 historical 60/60.
- FINAL_HOLDOUT_V1 historical 40/40; no tuning.
- FINAL_HOLDOUT_V2 details NOT_VERIFIED in this pass; do not use.
- FINAL_HOLDOUT_V3 **CONSUMED**, 39/40, mosaic SFace `0.360 < 0.363`; NEVER rerun/tune.
- FINAL_HOLDOUT_V4 40 cases/20 identities, frozen ControlFace10K, **NOT_RUN/UNCONSUMED**; one-shot protocol only.
- FINAL_HOLDOUT_V5 not created.
- Female-domain ~300–400-case stress profile; safety hard gates, quality report-only.
- Paper Quality DEV/VALIDATION incomplete.
- DamageMaskNet FairFace+ControlFace bank TRAIN/VALIDATION only.

V3/V4 are verification-only during current Track A work; neither is executed.

## 5. CURRENT GLOBAL OBJECTIVES

- OBJ-001 Preserve V1 — **PASS**.
- OBJ-002 Restore V1.1 gates — **VALIDATING / 3rd attempt pushed**. If attempt 3 fails, stop this implementation hypothesis and reassess rather than attempt4.
- OBJ-003 Canonical ledger — **IN_PROGRESS**.
- OBJ-004 DamageMaskNet — **BLOCKED** pending existing attempt-3 evidence.
- OBJ-005 Broad blind-BFR selection — **IN_PROGRESS**.
- OBJ-006 FBCNN JPEG qualification — **IN_PROGRESS**.
- OBJ-007 Personalized Reference Bank validation — **IN_PROGRESS**.
- OBJ-008 RefFace CPU — **BLOCKED** by DamageMaskNet; 0/3 runs consumed.
- OBJ-009 Paper Quality Windows pack — **PROPOSED**.
- OBJ-010 Real HP EliteBook acceptance — **PROPOSED**.

## 6. MODEL MASTER REGISTRY

Certified roles: YuNet detector; SFace identity gate `0.363`; NAFNet deblur/denoise; Face Parsing ResNet18 ONNX; Head Pose MobileNetV2 ONNX; constrained LaMa ONNX.

Research: GPEN BFR-512 **BENCHMARKING/license blocker**; GFPGAN v1.4 **BENCHMARKING**; CodeFormer **BENCHMARKING/BLOCKED_LICENSE**; FBCNN **BENCHMARKING/current DEV JPEG leader**; DamageMaskNet **BENCHMARKING/BLOCKED**; RefFace **FEASIBILITY_ONLY/NOT_RUN**; InstantRestore and OSDFace **FEASIBILITY_ONLY/BLOCKED_HARDWARE**; RestoreFormer++, VQFR, GPEN-inpainting, RefineFIR, PerFuSe, RefIPFR, Real-ESRGAN remain feasibility/research until measured.

Registry documentation mismatch remains: certified `THIRD_PARTY_MODULES.md` references absent `models/` machine-readable catalogs; actual active registry code is under `app/`. Never invent manifests/hashes.

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEVELOPMENT only:
- GPEN: SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, peak RSS `~1.828GB`.
- GFPGAN v1.4: SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`.
- FBCNN QF20: SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, `~1.305GB`.
- CodeFormer: real aligned CPU slice PASS; exact comparative metrics must be read from artifact.

## 8. 13-BLOCK ARCHITECTURE

1 IMPORT deterministic. 2 DEBLUR NAFNet + future measured BFR candidate. 3 ENHANCE FBCNN for JPEG. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK parser + DamageMaskNet target. 7 REGION_SELECT component/reference bank. 8 INPAINT observed refs first, Paper generators only as GENERATED. 9 FUSION healthy MAIN > observed ref > generated. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`, final Track A hypothesis validation. 12 UPSCALE Lanczos/optional measured SR. 13 EXPORT deterministic provenance + future resource/model reports.

## 9. PHOTO AND INPUT CONTRACT

MAIN target: low-res phone/social-media, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, covered/missing components, crop/partial face, low-light/uneven exposure, mixed damage. MAIN remains Conservative target canvas/pose/frame.

References: MAIN + 0–9 full/partial/component-only/angle/expression/light/resolution/blur/compression/occlusion/useless/wrong-person. Full accepted same-person may be global anchor; partial same-person local only; wrong-person never anchor/donor/identity booster.

## 10. DATASET CONSTRUCTION

Paper Quality initial target ~300–400 representative cases with explicit female-domain percentage. Identity-disjoint TRAIN/DEV/VALIDATION/FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/exact mask/reference relationships. Never tune/train on final holdout.

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Track 13 components: LEFT/RIGHT EYE, LEFT/RIGHT EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT/RIGHT CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Track MAIN visibility/damage, observed refs, confidence, generated candidate, selected source/provenance, identity/geometry and unresolved state. Observed same-person outranks generation.

## 12. DAMAGE ROUTING

HEALTHY preserve MAIN; BLUR -> NAFNet/measured deblur then Paper BFR if needed; JPEG -> FBCNN; PIXELATION/MOSAIC -> observed component first then Paper generation; SCRIBBLE/STICKER/OPAQUE/BLACK_BAR -> observed reference first then qualified reference specialist; PARTIAL/MISSING -> component bank then Paper fallback; LOW_LIGHT -> detected specialist only; MIXED -> minimal specialist candidates, never blind generator chaining.

## 13. DECISION LOG

DEC-001 canonical ledger ACCEPTED. DEC-002 advanced research branch active ACCEPTED. DEC-003 <=80% CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet bank ACCEPTED. DEC-006 RefFace next after DamageMaskNet ACCEPTED/BLOCKED. DEC-007 V3 consumed/V4 untouched ACCEPTED. DEC-008 ranking cluster != identity authority ACCEPTED; final implementation rule explicitly excludes reasons `main_bridged_cross_reference_cluster` and `same_canvas_bridged_cross_reference_cluster` when direct matrix evidence is absent.

## 14. EXPERIMENT LOG

- EXP GPEN/GFPGAN/FBCNN/CodeFormer: real DEV evidence as recorded above.
- DamageMaskNet attempt1/3 403 infrastructure fail; attempt2/3 429 infrastructure fail; attempt3/3 result NOT_VERIFIED, no fourth observability rerun.
- RefFace PREPARED/NOT_RUN, 0/3.
- Track A identity hypothesis attempt1/3 `3e919f7a...`: FAIL `4 failed,102 passed`.
- Track A attempt2/3 `7c683edd...`: FAIL `1 failed,107 passed`; sole failure cluster-promoted flag survives with missing matrix.
- **Track A attempt3/3 `84640fb7cee273bb0f5dcb19b4f9a1f3908e583e`: NOT_VERIFIED.** One-file change in `app/identity_anchor_v4_hardening.py`; adds explicit `_CLUSTER_ONLY_REASONS`, blocks their missing-matrix fallback in both bridge and downstream trusted-source paths, preserves non-cluster current direct whole-face SFace evidence. No model/checkpoint/threshold/preflight/holdout change.

## 15. QUALITY SCOREBOARD

DEV model evidence exists; broad validation incomplete. V3 holdout consumed 39/40. V4 frozen/unexecuted. Target-PC Paper Quality NOT_RUN. Keep DEV/VALIDATION/HOLDOUT/REAL-WORLD/TARGET-PC metrics separate.

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU runtime detected. CPU-first; no CUDA requirement. <=80% logical CPU, <=80% process/system RAM, one heavy model. Optional acceleration only after support/parity evidence.

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed `0`; provenance violations `0`; frozen healthy/outside MAE `<=8.0`. No threshold-shopping, cherry-picking, consumed-holdout reruns, hard-case deletion, generated-as-observed, wrong-person score rescue, auto-merge, certified-history force-push or fabricated evidence.

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

## 19. TRACK A — PRODUCT_V1_1

Historical `3645c8c...`: Release Quality `4 failed,195 passed`, Windows/Female FAIL.

Attempt1 `3e919f7a...`: Release Quality `4 failed,102 passed`; fixed/pinpointed direct transfer, partial same-canvas, runtime reorder and engine metadata problems.

Attempt2 `7c683edd...`: Release Quality #125 `1 failed,107 passed`; sole failure `test_missing_preflight_matrix_fails_closed_for_cluster_promoted_flag`. Artifact ID `9351701070`, digest `d5a6a9e7766796ed8b43fb40f5230bf6448b31c1655076dd234553a41cf5a97c`.

Attempt3 `84640fb7...`: final allowed implementation attempt. Exact rule: missing-matrix fallback can preserve current numeric whole-face direct flags **only when existing reason is not an explicit cluster-promotion reason**. Direct `direct_sface` survives; cluster promotion fails closed. No threshold/model/holdout changes. **Result NOT_VERIFIED.**

If attempt3 targeted suite fails, STOP this hypothesis and document reassessment/new hypothesis instead of attempt4. If PASS, proceed to full pytest and same-head Windows/Female/Release Quality; V3/V4 remain unexecuted.

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`. Real model evidence + 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, hard-gated selector, deterministic fusion, parser adapter and RefFace manual workflow. Models must converge to QUALIFIED, REJECTED or documented blockers.

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1 + ONNX parity + RAM/runtime. Infrastructure FAIL -> infrastructure only. True model/data FAIL -> U-Net hypothesis ends. Then RefFace attempt1/3.

## 22. SPECIALIST MODEL STRATEGY

`input -> detect/align -> damage -> reference/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance`. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference specialist. Never blindly chain all generators.

## 23. MODEL SELECTION POLICY

Select BFR/specialist winners on multiple identity-disjoint DEV/VALIDATION cases, per damage. Identity is a hard gate before ranking. Measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM. Never select on final holdout.

## 24. HISTORICAL RECORD — append-only

- HIST-20260815-001: certified V1 merge `f476c6f... -> 2767513f...`; historical Windows #1195/Female #463/Release Quality #13.
- HIST-20260818-002: Track A `3645c8c...` all three release workflows FAIL; V3 consumed; V4 frozen/unexecuted.
- HIST-20260818-003: Track B `645862d1...` real DEV evidence; DamageMaskNet attempt3 NOT_VERIFIED; RefFace NOT_RUN.
- HIST-20260819-004: canonical `meta/project-state` ledger established.
- HIST-20260819-005: attempt1 push `3645c8c... -> 3e919f7a...`.
- HIST-20260819-006: attempt1 CI `4 failed,102 passed`.
- HIST-20260819-007: attempt2 push `3e919f7a... -> 7c683edd...`.
- HIST-20260819-008: attempt2 CI `1 failed,107 passed`; artifact `9351701070` / `d5a6a9e...`.
- **HIST-20260819-009:** final identity-hypothesis attempt3 technical push `7c683eddad579974329bb186622aea01feccee61 -> 84640fb7cee273bb0f5dcb19b4f9a1f3908e583e`. One file: `app/identity_anchor_v4_hardening.py`. No model/checkpoint/threshold/holdout change. Result NOT_VERIFIED at ledger update. No attempt4 under this hypothesis.
