# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current state is maintained; important decisions, experiments, failures and technical pushes are preserved.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Previous technical HEAD: `2fcaeb1b10696894b6f8a412c9643b2965529ccc`
- Last technical HEAD: `49af8cb1dccb55ad1a326ca08ce6d3ba83e8c95e`
- Track A identity-authority hypothesis: CLOSED, targeted identity/source/provenance previously 108/108 PASS.
- Current Track A gate: DEC-009 same-canvas localized-damage attempt **2/3** is IMPLEMENTED and **NOT_VERIFIED**. Attempt 2 removes only the attempt-1 stable-edge survival requirement from the broad same-canvas matcher; SFace, face-local identity, provenance, models and holdouts are unchanged.
- Current Track B blocker: recover already-triggered DamageMaskNet attempt-3 evidence without rerun for observability.

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

Identity authority is no longer the current blocker. The remaining same-canvas issue is architectural separation: **whole-canvas sameness is evidence of shared source geometry/canvas, not identity**. A shared background with a changed face may legitimately pass the broad canvas matcher but must fail the stricter face-local identity bridge. Attempt 2/3 therefore removes attempt 1's stable-edge survival threshold while preserving the strict global Lab canvas gate and the separate face-local identity gate.

---

## 2. Branch and release map

| Branch | Purpose | HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f...` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | V1 history | `5eff6673...` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate history | `f476c6f0...` | FROZEN / ARCHIVED | merged PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `49af8cb1...` | VALIDATING / ACTIVE | PR #2 OPEN/DRAFT; exact-head CI NOT_VERIFIED | DEC-009 attempt2/3 validation |
| `research/face-restoration-v2` | early data/degradation research | `757a3f60...` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Track B | `645862d1...` | ACTIVE / BENCHMARKING | partly NOT_VERIFIED | DamageMaskNet evidence recovery |
| `meta/project-state` | canonical ledger | self-SHA omitted | ACTIVE META | docs only | update after every technical push |

---

## 3. PRODUCT VERSION ROADMAP

- **PRODUCT_V1 — RELEASED:** certified conservative baseline, SFace `0.363`, wrong-person observed `0`, provenance violations `0`.
- **PRODUCT_V1_1 — VALIDATING/BLOCKED:** operational hotfix; current DEC-009 same-canvas hypothesis attempt 2/3 is pushed and awaiting exact-head evidence.
- **PRODUCT_V2 — BENCHMARKING:** Paper Quality Local with measured BFR/JPEG specialists, hard gates, generated provenance, 80% resource contract.
- **PRODUCT_V3 — PLANNED with prototypes:** personalized MAIN+0–9 refs, per-component authority.
- **PRODUCT_V4 — PLANNED with prototypes:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified modes + offline model pack + installer + clean Windows + real HP EliteBook acceptance.

Product and holdout versions remain separate.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

CALIBRATION_V1 historical 60/60. FINAL_HOLDOUT_V1 historical 40/40. FINAL_HOLDOUT_V2 details not fully re-reconciled. FINAL_HOLDOUT_V3 **CONSUMED** 39/40; mosaic SFace `0.360<0.363`; NEVER rerun/tune. FINAL_HOLDOUT_V4 frozen 40 cases/20 identities, **NOT_RUN/UNCONSUMED**, one-shot only. V5 not created. Female-domain ~300–400 stress cases. Paper Quality DEV/VALIDATION separate. DamageMaskNet bank FairFace+ControlFace TRAIN/VALIDATION only.

V3/V4 remain verification-only in Track A.

---

## 5. CURRENT GLOBAL OBJECTIVES

OBJ-001 Preserve V1 — PASS.  
OBJ-002 Restore V1.1 gates — VALIDATING on DEC-009 attempt2/3.  
OBJ-003 Canonical ledger — IN_PROGRESS.  
OBJ-004 DamageMaskNet — BLOCKED on existing attempt3 evidence.  
OBJ-005 Broad BFR selection — IN_PROGRESS.  
OBJ-006 FBCNN JPEG qualification — IN_PROGRESS.  
OBJ-007 Personalized Reference Bank validation — IN_PROGRESS.  
OBJ-008 RefFace CPU — BLOCKED by OBJ-004, 0/3 attempts.  
OBJ-009 Paper Quality Windows pack — PROPOSED.  
OBJ-010 HP EliteBook acceptance — PROPOSED.

---

## 6. MODEL MASTER REGISTRY

Certified roles: YuNet, SFace `0.363`, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research: GPEN BENCHMARKING/license blocker; GFPGAN1.4 BENCHMARKING; CodeFormer BENCHMARKING/BLOCKED_LICENSE; FBCNN BENCHMARKING/current DEV JPEG leader; DamageMaskNet BENCHMARKING/BLOCKED; RefFace FEASIBILITY_ONLY/NOT_RUN; InstantRestore/OSDFace hardware-blocked feasibility; RestoreFormer++/VQFR/GPEN-inpainting/RefineFIR/PerFuSe/RefIPFR/Real-ESRGAN feasibility until measured.

Registry documentation mismatch remains; never invent missing manifests/hashes.

---

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEV only: GPEN SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, `~1.828GB`; GFPGAN1.4 SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`; FBCNN QF20 SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, `~1.305GB`; CodeFormer real CPU slice PASS, exact metrics artifact-required.

---

## 8. 13-BLOCK ARCHITECTURE

1 IMPORT deterministic. 2 DEBLUR NAFNet/measured BFR later. 3 ENHANCE FBCNN for JPEG. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK parser + DamageMaskNet target. 7 REGION_SELECT component/reference bank. 8 INPAINT observed first, Paper generation only as GENERATED. 9 FUSION MAIN > observed ref > generated. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`, direct/non-transitive policy targeted green before DEC-009. 12 UPSCALE Lanczos/optional measured SR. 13 EXPORT deterministic provenance/model/resource evidence.

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

DEC-001 canonical ledger ACCEPTED. DEC-002 active Paper Quality branch ACCEPTED. DEC-003 <=80% CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet bank ACCEPTED. DEC-006 RefFace after DamageMaskNet ACCEPTED/BLOCKED. DEC-007 V3 consumed/V4 untouched ACCEPTED. DEC-008 ranking cluster != identity authority ACCEPTED/CLOSED after 3/3. **DEC-009 localized-damage same-canvas edge isolation — attempt1 FAIL; attempt2 IMPLEMENTED/VALIDATING:** keep strict global Lab gate and local mismatch edge exclusion, but remove the requirement that a fixed proportion of global edges survive. Broad same-canvas is not identity authority; face-local bridge remains the identity gate. If fewer than 64 stable edges remain, edge consistency is non-authoritative and the already-passed strict photometric canvas rule governs broad match.

---

## 14. EXPERIMENT LOG

DamageMaskNet 1/3 403 infra fail, 2/3 429 infra fail, 3/3 NOT_VERIFIED; no attempt4 for observability. RefFace PREPARED/NOT_RUN.

Identity hypothesis attempts 1–3 closed; targeted suite reached 108/108 PASS at `9b8810ce...`.

**EXP-20260819-012 DEC-009 attempt1:** `9b8810ce... -> 2fcaeb1b...`; strict Lab + local mismatch dilation + minimum stable-edge support. Release Quality #129 targeted `1 failed,107 passed`. Sole failure `test_shared_background_cannot_become_identity_bridge_when_face_region_differs`: broad same-canvas expected TRUE but matcher returned FALSE because attempt1's stable-edge minimum rejected a shared canvas whose only informative edges were inside the deliberately changed face. Separate face-local test expects FALSE and remains the identity safety boundary. Artifact `9364721505`, digest `37a71c20ca44af86d2a4e6f839246f0d34ba9287563f8082e6641f747978eb0c`.

**EXP-20260819-013 DEC-009 attempt2:** technical push `2fcaeb1b10696894b6f8a412c9643b2965529ccc -> 49af8cb1dccb55ad1a326ca08ce6d3ba83e8c95e`. Only `app/primary_anchor_policy.py` changed. Strict global Lab median/p90 remains unchanged. Local mismatch (`delta_map > 0.035`, <=10% of comparable pixels) remains excluded from secondary gradient testing through a 5x5 dilation. The attempt-1 `max(64,35% raw edges)` survival requirement was removed. If >=64 stable edges remain, their edge consistency is still validated; otherwise broad canvas matching relies on the already-passed strict photometric rule and never becomes identity authority. Face-local identity verifier, SFace threshold, provenance, models, checkpoints and holdouts are unchanged. Result **NOT_VERIFIED** at ledger update.

---

## 15. QUALITY SCOREBOARD

DEV evidence exists; broad validation incomplete. V3 consumed 39/40. V4 frozen/unexecuted. Target-PC Paper Quality NOT_RUN. Maintain DEV/VALIDATION/HOLDOUT/REAL-WORLD/TARGET-PC separately.

---

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU runtime-detected. CPU-first/no CUDA. <=80% logical CPU, <=80% process/system RAM, one heavy model. Optional acceleration only after support/parity evidence.

---

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed `0 pixels`; provenance violations `0`; healthy/outside MAE `<=8.0` where frozen policy applies. No threshold shopping, cherry-picking, consumed-holdout rerun, hard-case deletion, generated-as-observed, wrong-person score rescue, auto-merge, force-push certified history or fabricated evidence.

---

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

---

## 19. TRACK A — PRODUCT_V1_1

Previous `9b8810ce...`: targeted 108 PASS; full pytest 3 failed/543 passed.

DEC-009 attempt1 `2fcaeb1b...`: Release Quality #129 stopped at targeted suite with **1 failed,107 passed**; V3/V4 verification PASS without execution. Sole failure established broad same-canvas vs face-local identity separation.

Current exact technical HEAD `49af8cb1dccb55ad1a326ca08ce6d3ba83e8c95e`: DEC-009 attempt2/3 changes only broad same-canvas localized edge handling. **Exact-head CI NOT_VERIFIED.**

Next exact action: inspect one same-head status. If targeted PASS, proceed to full pytest/model-pack/calibration/Windows/Female gates. If DEC-009 behavior fails, only attempt3/3 remains. V3/V4 remain unexecuted.

---

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`: real CPU BFR/JPEG evidence, 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, hard-gated selector, deterministic fusion, parser adapter, RefFace manual workflow.

PDF constraints: separate global identity from local texture; use correspondence between matching regions; region-adaptive identity guidance for severe BFR; MAIN preserves pose/composition/expression/geometry; unsupported detail remains conservative. These guide PRODUCT_V3/V4, not V1.1 model substitution.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1, ONNX parity, RAM/runtime. Infrastructure fail -> infrastructure-only. True model/data fail -> U-Net hypothesis ends. Then RefFace attempt1/3.

---

## 22. SPECIALIST MODEL STRATEGY

input -> detect/align -> damage -> reference/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference specialist. Never blindly chain generators.

---

## 23. MODEL SELECTION POLICY

Select winners on multiple identity-disjoint DEV/VALIDATION cases per damage; identity hard gate first; measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM; never select/tune using final holdout.

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
- **HIST-20260819-017:** technical push `2fcaeb1b10696894b6f8a412c9643b2965529ccc -> 49af8cb1dccb55ad1a326ca08ce6d3ba83e8c95e`. DEC-009 attempt2/3. File changed: `app/primary_anchor_policy.py` only. Removed attempt-1 stable-edge survival threshold while preserving strict Lab canvas rule, face-local identity verifier, SFace `0.363`, provenance and all holdout/model state. Exact-head result NOT_VERIFIED at ledger update.
