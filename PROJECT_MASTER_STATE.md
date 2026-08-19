# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** GitHub evidence overrides chat memory. Read before engineering. Current state is maintained; important decisions, experiments, failures and pushes are preserved.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Previous technical HEAD: `84640fb7cee273bb0f5dcb19b4f9a1f3908e583e`
- Last technical HEAD: `ab69f18aae3c486710f93aff73bab65fd97641a4`
- Track A identity-authority implementation hypothesis: **CLOSED after 3/3 evidence-based behavioral attempts.**
- Current Track A blocker: exact-head CI for the documentation/protocol-only correction at `ab69f18a...` is **NOT_VERIFIED**. The correction restored the required non-transitive invariant sentence and changed no executable logic.
- Current Track B blocker: recover DamageMaskNet attempt-3 evidence without rerun for observability.

FORENSIC_MODE_READY: **TRUE for certified PRODUCT_V1 only**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL — historical V1 only**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory: technical push -> exact SHA -> ledger update. No auto-merge/force-push certified history.

## 1. Executive project summary

CFS is a local Windows face-restoration system for damaged phone/social-media portraits. Conservative Mode uses MAIN and verified same-person observed evidence with explicit provenance; Paper Quality Mode may generate missing information only as `GENERATED_MODEL_INFERRED`.

PRODUCT_V1 is certified/immutable. PRODUCT_V1_1 is a safety/operational hotfix isolated from Paper Quality. Track B contains real CPU research evidence and specialist/reference/fusion infrastructure.

Track A’s three-attempt identity hypothesis converged behaviorally: ranking cluster != identity authority; direct whole-face evidence is preserved; partial same-canvas stays component-local; explicit cluster promotions fail closed; legacy structured SFace scores without an `engine` label remain compatible while explicit proxy engines are rejected. The only remaining failure at `84640fb7...` was a static source-text contract. Technical HEAD `ab69f18a...` restores that exact invariant wording only.

## 2. Branch and release map

| Branch | HEAD | State | CI / next gate |
|---|---|---|---|
| `main` | `2767513f...` | FROZEN / RELEASED | preserve |
| `feature/block-pipeline-v1` | `5eff6673...` | MERGED / SUPERSEDED | archive |
| `release/v1-certified` | `f476c6f0...` | FROZEN / ARCHIVED | preserve |
| `hotfix/real-world-restoration-v1.1` | `ab69f18a...` | VALIDATING / ACTIVE | exact-head CI NOT_VERIFIED; previous `84640fb7...` had `1 failed,107 passed`, sole failure static invariant text |
| `research/face-restoration-v2` | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | preserve assets |
| `research/paper-quality-local-v2` | `645862d1...` | ACTIVE / BENCHMARKING | DamageMaskNet evidence recovery |
| `meta/project-state` | self-SHA omitted | ACTIVE META | update after each technical push |

## 3. PRODUCT VERSION ROADMAP

PRODUCT_V1 RELEASED conservative baseline. PRODUCT_V1_1 VALIDATING operational hotfix. PRODUCT_V2 BENCHMARKING Paper Quality Local. PRODUCT_V3 PLANNED personalized multi-reference. PRODUCT_V4 PLANNED damage-specialist hybrid. PRODUCT_V5 PLANNED unified offline Windows product. Product labels are separate from holdout labels.

## 4. HOLDOUT / benchmark lineage

CALIBRATION_V1 historical 60/60. FINAL_HOLDOUT_V1 historical 40/40. FINAL_HOLDOUT_V2 details NOT_VERIFIED. FINAL_HOLDOUT_V3 **CONSUMED** 39/40, mosaic SFace `0.360<0.363`, never rerun/tune. FINAL_HOLDOUT_V4 frozen 40 cases/20 identities, **NOT_RUN/UNCONSUMED**, one-shot only. V5 not created. Female-domain ~300–400 stress cases. Paper Quality DEV/VALIDATION separate. DamageMaskNet bank FairFace+ControlFace TRAIN/VALIDATION only.

V3 and V4 remain verification-only during current Track A work; neither may be executed until the release sequence explicitly permits it.

## 5. CURRENT GLOBAL OBJECTIVES

- OBJ-001 Preserve V1 — PASS.
- OBJ-002 Restore V1.1 — VALIDATING. Identity behavior hypothesis closed; protocol-only invariant correction pushed at `ab69f18a...`; exact-head gates pending verification.
- OBJ-003 Canonical ledger — IN_PROGRESS.
- OBJ-004 DamageMaskNet — BLOCKED pending existing attempt3 evidence.
- OBJ-005 Broad BFR selection — IN_PROGRESS.
- OBJ-006 FBCNN JPEG qualification — IN_PROGRESS.
- OBJ-007 Personalized Reference Bank validation — IN_PROGRESS.
- OBJ-008 RefFace CPU — BLOCKED by DamageMaskNet; 0/3 attempts.
- OBJ-009 Paper Quality Windows pack — PROPOSED.
- OBJ-010 real HP EliteBook acceptance — PROPOSED.

## 6. MODEL MASTER REGISTRY

Certified-role models: YuNet, SFace `0.363`, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research states: GPEN BENCHMARKING/license blocker; GFPGAN1.4 BENCHMARKING; CodeFormer BENCHMARKING/BLOCKED_LICENSE; FBCNN BENCHMARKING/current DEV JPEG leader; DamageMaskNet BENCHMARKING/BLOCKED; RefFace FEASIBILITY_ONLY/NOT_RUN; InstantRestore/OSDFace hardware-blocked feasibility; RestoreFormer++/VQFR/GPEN-inpaint/RefineFIR/PerFuSe/RefIPFR/Real-ESRGAN feasibility until measured.

Registry documentation mismatch: certified `THIRD_PARTY_MODULES.md` references absent `models/` machine-readable catalogs; actual active registry is under `app/`. Never invent hashes/manifests.

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEV only: GPEN SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, `~1.828GB`; GFPGAN1.4 SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`; FBCNN QF20 SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, `~1.305GB`; CodeFormer real CPU slice PASS, exact comparative metrics must be read from artifact.

## 8. 13-BLOCK ARCHITECTURE

1 IMPORT deterministic; 2 DEBLUR NAFNet/BFR candidate; 3 ENHANCE FBCNN for JPEG; 4 LANDMARKS YuNet/pose; 5 ALIGN deterministic; 6 OCCLUSION_MASK parsing + DamageMaskNet target; 7 REGION_SELECT component/reference bank; 8 INPAINT observed first, generated Paper fallback; 9 FUSION MAIN > observed ref > generated; 10 FRONTALIZE geometry-only Conservative; 11 IDENTITY_CHECK SFace `0.363`, non-transitive direct-anchor policy behavior converged; 12 UPSCALE Lanczos/optional measured SR; 13 EXPORT deterministic provenance/resources/model reports.

## 9. PHOTO AND INPUT CONTRACT

MAIN: low-res smartphone/social-media, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, covered/missing components, crop/partial, low-light, mixed. References MAIN+0–9: full/partial/component-only/angle/expression/light/resolution/degraded/useless/wrong-person. Full accepted same-person may global-anchor; partial is local; wrong-person never anchor/donor/identity booster.

## 10. DATASET CONSTRUCTION

Initial Paper Quality target ~300–400 representative cases with explicit female-domain percentage. Identity-disjoint TRAIN/DEV/VALIDATION/FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/mask/reference relationships. Never tune/train on final holdout.

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

13 components: left/right eye, left/right eyebrow, nose, philtrum, mouth/lips, left/right cheek, chin, jaw, forehead, face contour. Track visibility/damage, observed refs/confidence, generated candidates, selected source/provenance, identity/geometry and unresolved. Observed same-person outranks generation.

## 12. DAMAGE ROUTING

HEALTHY preserve MAIN; BLUR NAFNet/measured BFR; JPEG FBCNN; PIXELATION/MOSAIC observed component first then Paper generation; SCRIBBLE/STICKER/OPAQUE/BLACK_BAR observed ref then qualified reference specialist; MISSING/PARTIAL component bank then Paper fallback; LOW_LIGHT specialist only when detected; MIXED minimum necessary specialist set, never blind chaining.

## 13. DECISION LOG

DEC-001 canonical ledger ACCEPTED. DEC-002 advanced research branch active ACCEPTED. DEC-003 <=80% CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet bank ACCEPTED. DEC-006 RefFace after DamageMaskNet ACCEPTED/BLOCKED. DEC-007 V3 consumed/V4 untouched ACCEPTED. DEC-008 ranking cluster != identity authority **ACCEPTED; identity implementation hypothesis CLOSED after 3 attempts**. Residual static-contract restoration is explicitly not a fourth behavioral attempt.

## 14. EXPERIMENT LOG

GPEN/GFPGAN/FBCNN real DEV evidence; CodeFormer packaging fail then CPU PASS. DamageMaskNet attempts: 1/3 403 infrastructure, 2/3 429 infrastructure, 3/3 NOT_VERIFIED; no attempt4 for observability. RefFace PREPARED/NOT_RUN.

Track A identity hypothesis:
- attempt1 `3e919f7a...`: `4 failed,102 passed`;
- attempt2 `7c683edd...`: `1 failed,107 passed`, cluster-only missing-matrix failure;
- attempt3 `84640fb7...`: `1 failed,107 passed`, sole failure static `tests/test_v4_direct_edge_protocol.py::test_v4_hardening_forbids_transitive_component_authority`, requiring exact sentence `Newly trusted references never become new`; no dynamic identity assertion failed. Artifact `9351774422`, digest `c010fb733dc58fc9725aa7b27745f0ef704c8d7ffe5862cc1371f2260461bbe0`.
- protocol/documentation correction `ab69f18aae3c486710f93aff73bab65fd97641a4`: restored the exact required sentence in the module docstring while leaving executable logic unchanged. Result NOT_VERIFIED at this ledger update.

## 15. QUALITY SCOREBOARD

DEV evidence exists; broad validation incomplete. V3 consumed 39/40. V4 frozen/unexecuted. Paper Quality target-PC NOT_RUN. Keep DEV/VALIDATION/HOLDOUT/REAL-WORLD/TARGET-PC metrics separate.

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows, exact CPU/GPU runtime-detected. CPU-first/no CUDA requirement. <=80% logical CPU, <=80% process/system RAM, one heavy model. Optional acceleration only with support/parity evidence.

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed pixels `0`; provenance violations `0`; frozen healthy/outside MAE `<=8.0`. No threshold shopping, cherry-picking, consumed-holdout rerun, hard-case deletion, generated-as-observed, wrong-person score rescue, auto-merge, force-push certified history or fabricated evidence.

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

## 19. TRACK A — PRODUCT_V1_1

Historical `3645c8c...`: Release Quality `4 failed,195 passed`, Windows/Female FAIL.
Attempt1 `3e919f7a...`: `4 failed,102 passed`.
Attempt2 `7c683edd...`: `1 failed,107 passed` behavioral cluster-only issue.
Attempt3 `84640fb7...`: `1 failed,107 passed`; all remaining failure evidence was static source-text contract only.

Current `ab69f18a...`: restored the exact static sentence `Newly trusted references never become new` in the identity-hardening docstring; executable logic unchanged. **Exact-head CI NOT_VERIFIED.**

If targeted identity/protocol suite is green, proceed to full pytest and same-head Windows/Female/Release Quality. Any new behavioral identity failure requires a new documented hypothesis, not continuation of the closed 3/3 hypothesis. V3/V4 remain unexecuted.

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`: real CPU BFR/JPEG evidence, 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, selector, deterministic fusion, parser adapter, RefFace manual workflow. Models must converge to qualified/rejected/documented blocker.

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1+ONNX parity+RAM/runtime; infrastructure fail -> infrastructure only; model/data fail -> U-Net hypothesis ends. Then RefFace attempt1/3.

## 22. SPECIALIST MODEL STRATEGY

input -> detect/align -> damage -> reference/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed first then qualified reference specialist. Never blindly chain generators.

## 23. MODEL SELECTION POLICY

Select winners on identity-disjoint DEV/VALIDATION per degradation; identity hard gate first; measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM; never select using final holdout.

## 24. HISTORICAL RECORD — append-only

- HIST-20260815-001 certified V1 merge `f476c6f... -> 2767513f...` with historical Windows #1195/Female #463/Release Quality #13.
- HIST-20260818-002 Track A `3645c8c...` three release workflows FAIL; V3 consumed; V4 frozen/unexecuted.
- HIST-20260818-003 Track B `645862d1...` research snapshot; DamageMaskNet attempt3 NOT_VERIFIED; RefFace NOT_RUN.
- HIST-20260819-004 canonical meta ledger established.
- HIST-20260819-005 identity attempt1 push `3645c8c... -> 3e919f7a...`.
- HIST-20260819-006 attempt1 CI `4 failed,102 passed`.
- HIST-20260819-007 identity attempt2 push `3e919f7a... -> 7c683edd...`.
- HIST-20260819-008 attempt2 CI `1 failed,107 passed`, artifact `9351701070`, digest `d5a6a9e...`.
- HIST-20260819-009 identity attempt3 push `7c683edd... -> 84640fb7...`.
- HIST-20260819-010 attempt3 result: Release Quality #126 (`32214190714`) `1 failed,107 passed`; sole failure static source-contract phrase; artifact `9351774422`, digest `c010fb733dc58fc9725aa7b27745f0ef704c8d7ffe5862cc1371f2260461bbe0`; identity behavior hypothesis closed at 3/3.
- **HIST-20260819-011:** protocol/documentation-only technical push `84640fb7cee273bb0f5dcb19b4f9a1f3908e583e -> ab69f18aae3c486710f93aff73bab65fd97641a4`. File: `app/identity_anchor_v4_hardening.py`. Change: restore exact static invariant sentence `Newly trusted references never become new` in module docstring. Executable behavior/models/checkpoints/thresholds/holdouts unchanged. CI result NOT_VERIFIED at ledger update.
