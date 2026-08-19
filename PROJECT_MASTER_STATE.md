# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current-state sections are maintained; important decisions, experiments, failures and technical pushes are append-only historical evidence.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Last technical HEAD: `9b8810ce5e53daa92d4f13bcbd8b23c6a25df105`
- Active engineering: Track A PRODUCT_V1_1 operational stabilization; Track B Paper Quality research.
- Exact Track A blocker: Release Quality #128 has **targeted identity/provenance 108/108 PASS**, then full pytest **3 failed, 543 passed**. Two failures are stale/synthetic test-contract issues; one is a real same-canvas mosaic-detection defect. A new narrow same-canvas localized-damage hypothesis is accepted before implementation.
- Exact Track B blocker: recover the already-triggered DamageMaskNet mixed-source attempt-3 evidence; do not run attempt 4 merely for observability.

FORENSIC_MODE_READY: **TRUE for certified PRODUCT_V1 only**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL — historical PRODUCT_V1 only**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> evidence/tests -> commit/push -> exact remote SHA -> ledger update/push`. No auto-merge. No force-push of certified history.

---

## 1. Executive project summary

Conservative Face Studio is a local Windows face-restoration system for low-quality, compressed, blurred, pixelated, mosaicked, scribbled, sticker-covered and otherwise damaged portraits. Conservative Mode protects MAIN pose/composition/geometry and uses only verified observed evidence with exact provenance. Paper Quality Mode may synthesize unsupported missing detail, but generated pixels remain `GENERATED_MODEL_INFERRED` and never become observed evidence.

PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is an operational/safety hotfix and remains isolated from experimental Track B generators. Track B contains real CPU model evidence and research infrastructure for damage routing, personalized references, reference-first restoration and component-aware fusion.

The V1.1 identity-authority hypothesis is now closed: the complete targeted identity/source/provenance suite is green. The current engineering issue is no longer identity thresholding. It is the correct detection of **same-canvas references when a local mosaic introduces many artificial gradient edges but most of the canvas is unchanged**, plus two stale test fixtures that still encode pre-V4 assumptions.

---

## 2. Branch and release map

| Branch | Purpose | Verified HEAD | State | CI / merge state | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | V1 implementation history | `5eff667373cd47c07ba14aaad2acafee6d5a61c1` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate history | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | FROZEN / ARCHIVED | merged via PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `9b8810ce5e53daa92d4f13bcbd8b23c6a25df105` | BLOCKED / ACTIVE | PR #2 OPEN/DRAFT; Release Quality FAIL after targeted PASS/full pytest FAIL; Windows FAIL; Female-domain last seen in progress | same-canvas hypothesis attempt 1/3 + stale-test correction |
| `research/face-restoration-v2` | early dataset/degradation research | `757a3f6081b7b152cdc615a07cd99aec40fa0a1c` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Paper Quality research | `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd` | ACTIVE / BENCHMARKING | partly NOT_VERIFIED | recover DamageMaskNet attempt 3 |
| `meta/project-state` | canonical ledger | self-SHA intentionally omitted | ACTIVE META | documentation-only | update after every technical push |

The two research branches diverged from the same certified base. The advanced branch is not falsely described as a literal Git superset of the early branch.

---

## 3. PRODUCT VERSION ROADMAP

### PRODUCT_V1 — RELEASED
Certified conservative/forensic baseline. Safety: SFace `0.363`; wrong-person observed pixels `0`; provenance violations `0`; frozen healthy-region policy. Historical Windows qualification exists; EliteBook-specific acceptance is NOT_VERIFIED.

### PRODUCT_V1_1 — IMPLEMENTING / BLOCKED
Operational real-world hotfix. Same production model family as V1; no Paper Quality generator rescue. Identity targeted suite is now PASS. Current blocker is full pytest with one real same-canvas mosaic matcher issue plus two test fixtures that must be aligned to the hardened safety contract without weakening production behavior.

### PRODUCT_V2 — BENCHMARKING
Paper Quality Local: damage-aware specialist routing, modern BFR candidates, hard identity gates, generated provenance, deterministic component fusion, <=80% logical CPU/process/system RAM, one heavy model resident.

### PRODUCT_V3 — PLANNED with enabling prototypes
Personalized MAIN + 0–9 same-person references, ranked per component. Full valid reference may be a global anchor; partial reference is component-local; wrong-person reference is never anchor/donor/identity-score improver.

### PRODUCT_V4 — PLANNED with enabling prototypes
Damage-specialist hybrid architecture: localized damage classification -> specialist candidates -> component scoring/fusion.

### PRODUCT_V5 — PLANNED
Unified final Windows product: Conservative + Paper Quality + personalized multi-reference + specialist routing + offline model pack + one-click installer + clean Windows + real HP EliteBook acceptance.

Product version labels are separate from HOLDOUT_V1…V5.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

- **CALIBRATION_V1:** historical 60/60 certified evidence.
- **FINAL_HOLDOUT_V1:** historical 40/40 certification; no tuning.
- **FINAL_HOLDOUT_V2:** details not fully re-reconciled in this pass; do not use for new tuning claims.
- **FINAL_HOLDOUT_V3:** **CONSUMED**, 39/40; `cfsfs3-fin-020-medium_block_mosaic`, SFace `0.360 < 0.363`; NEVER rerun/tune.
- **FINAL_HOLDOUT_V4:** frozen independent 40 cases / 20 identities on pinned ControlFace10K; **NOT_RUN / UNCONSUMED**; one-shot only after valid candidate sequence.
- **FINAL_HOLDOUT_V5:** not created.
- **Female-domain:** quick profile ~300–400 cases; safety hard gates, quality target report-only.
- **Paper Quality DEV/VALIDATION:** active but incomplete, identity-disjoint expansion required.
- **DamageMaskNet bank:** FairFace + ControlFace mixed bank, exact synthetic masks, TRAIN/VALIDATION only.

All current Release Quality runs verify V3/V4 manifests without executing either holdout.

---

## 5. CURRENT GLOBAL OBJECTIVES

- **OBJ-001 Preserve PRODUCT_V1 — PASS.** Never rewrite certified history.
- **OBJ-002 Restore PRODUCT_V1_1 operational gates — IN_PROGRESS/BLOCKED.** Targeted identity/provenance is PASS 108/108; full pytest currently 3 failed/543 passed.
- **OBJ-003 Maintain canonical ledger — IN_PROGRESS.** Every technical push recorded.
- **OBJ-004 Resolve DamageMaskNet — BLOCKED.** Recover existing attempt-3 result; no observability rerun.
- **OBJ-005 Broad blind-BFR selection — IN_PROGRESS.** Multiple identity-disjoint DEV/VALIDATION cases required.
- **OBJ-006 Qualify FBCNN JPEG specialist — IN_PROGRESS.** Expand QF/double-JPEG/social-media/resize+JPEG/JPEG+blur and Windows/EliteBook.
- **OBJ-007 Validate Personalized Reference Bank — IN_PROGRESS.** 0/1/9, full/partial/wrong/duplicate/low-quality/multi-pose.
- **OBJ-008 RefFace CPU feasibility — BLOCKED by OBJ-004.** Attempt 0/3 consumed.
- **OBJ-009 Paper Quality Windows model pack/installer — PROPOSED.** After model/router qualification.
- **OBJ-010 Real HP EliteBook acceptance — PROPOSED.** Final physical-machine gate.

---

## 6. MODEL MASTER REGISTRY

Certified/current-role stack: YuNet detector, SFace identity gate `0.363`, NAFNet mild deblur/denoise, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research states:
- GPEN BFR-512 — BENCHMARKING / distribution-license blocker.
- GFPGAN v1.4 — BENCHMARKING.
- CodeFormer w=0.5 — BENCHMARKING / BLOCKED_LICENSE.
- FBCNN — BENCHMARKING / current DEV JPEG leader.
- DamageMaskNet small U-Net — BENCHMARKING / BLOCKED pending attempt-3 evidence.
- RefFaceInpainting — FEASIBILITY_ONLY / NOT_RUN.
- InstantRestore — FEASIBILITY_ONLY / BLOCKED_HARDWARE + license audit.
- OSDFace — FEASIBILITY_ONLY / BLOCKED_HARDWARE.
- RestoreFormer++, VQFR, GPEN-inpainting, RefineFIR, PerFuSe, RefIPFR, Real-ESRGAN — feasibility/discovered until measured.

Registry documentation issue remains: certified `THIRD_PARTY_MODULES.md` references machine-readable files under `models/` not present on reconciled `main`; active production registry logic is in `app/`. Do not invent missing manifests or hashes.

---

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEVELOPMENT only:
- GPEN: SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, peak RSS `~1.828GB`.
- GFPGAN v1.4: SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`.
- FBCNN QF20: SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, peak RSS `~1.305GB`.
- CodeFormer: real aligned CPU slice PASS; exact comparative metrics must be reread from artifact before quoting.

No Linux DEV number is Windows/EliteBook evidence.

---

## 8. 13-BLOCK ARCHITECTURE

1. IMPORT deterministic MAIN+refs. 2. DEBLUR NAFNet, future measured BFR candidate. 3. ENHANCE FBCNN for detected JPEG. 4. LANDMARKS YuNet/pose. 5. ALIGN deterministic similarity/affine/RANSAC. 6. OCCLUSION_MASK parser + DamageMaskNet target. 7. REGION_SELECT component/reference bank. 8. INPAINT observed reference first; Paper generation only as GENERATED. 9. FUSION healthy MAIN > observed same-person reference > accepted generated. 10. FRONTALIZE geometry-only Conservative. 11. IDENTITY_CHECK SFace `0.363`; direct/non-transitive policy targeted suite PASS. 12. UPSCALE Lanczos, optional measured SR. 13. EXPORT deterministic provenance plus future model/damage/resource reports.

---

## 9. PHOTO AND INPUT CONTRACT

MAIN target: low-res smartphone/social-media, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, covered/missing components, crop/partial face, low-light/uneven exposure, mixed unknown damage. MAIN preserves target canvas/pose/frame/expression in Conservative Mode.

References: MAIN + 0–9 full/partial/component-only/side-angle/different expression/light/resolution/degraded/useless/wrong-person. Full accepted same-person may be global anchor; partial is component-local; wrong-person never global anchor, observed donor or identity-score improver.

---

## 10. DATASET CONSTRUCTION

Initial Paper Quality target ~300–400 representative cases, with explicit female-domain percentage when intentionally emphasized. Identity-disjoint TRAIN / DEVELOPMENT / VALIDATION / FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/exact mask/reference relationships. Never tune/train on final holdout.

---

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Track: LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Per component record MAIN visibility/damage, best/alternate refs, confidence/coverage, generated candidates, selected source/provenance, identity/geometry and unresolved state. Observed same-person evidence outranks generation.

---

## 12. DAMAGE ROUTING

HEALTHY -> preserve MAIN. BLUR -> NAFNet/measured deblur then Paper BFR only if needed. JPEG/DOUBLE_JPEG -> FBCNN first. PIXELATION/MOSAIC -> observed component first then Paper generated candidate. SCRIBBLE/STICKER/OPAQUE/BLACK_BAR -> observed reference first then qualified reference-conditioned specialist. PARTIAL/MISSING -> component bank then Paper fallback. LOW_LIGHT -> detected specialist only. MIXED -> minimum necessary specialist candidates; never blind-chain all BFR models.

---

## 13. DECISION LOG

- **DEC-20260819-001:** canonical `meta/project-state` ledger — ACCEPTED.
- **DEC-20260819-002:** advanced Paper Quality branch is active architecture; early branch preserved but not falsely merged — ACCEPTED.
- **DEC-20260819-003:** <=80% logical CPU, <=80% process/system RAM, one heavy model resident — ACCEPTED.
- **DEC-20260819-004:** healthy MAIN > verified observed same-person > accepted generated — ACCEPTED.
- **DEC-20260819-005:** DamageMaskNet attempt-3 acquisition switched to FairFace+ControlFace after 403/429 without changing U-Net hypothesis — ACCEPTED.
- **DEC-20260819-006:** RefFace next large-occlusion specialist after DamageMaskNet gate — ACCEPTED/BLOCKED BY SEQUENCE.
- **DEC-20260819-007:** V3 consumed; V4 frozen/unexecuted one-shot — ACCEPTED.
- **DEC-20260819-008:** ranking cluster != identity authority; identity behavioral hypothesis CLOSED after 3/3 attempts with targeted suite now PASS — ACCEPTED.
- **DEC-20260819-009 — localized-damage same-canvas edge isolation:** ACCEPTED for attempt 1/3. Problem: a localized mosaic changes only a small photometric fraction but creates a large number of artificial gradient edges, causing `_same_canvas_match` to reject an otherwise identical canvas when occlusion detection has no seed. Proposed rule: preserve the existing strict global Lab agreement; only when the photometric mismatch is a small local fraction, dilate that mismatch region and exclude those local corruption boundaries from the secondary edge-consistency check. Require sufficient stable edge evidence to remain. Do **not** loosen face-local identity proof, SFace, provenance, or wrong-person rules. Reversal: unrelated/same-background distractor tests or safety gates regress.

---

## 14. EXPERIMENT LOG

Historical model experiments: GPEN/GFPGAN/FBCNN real DEV evidence; CodeFormer packaging fail then real CPU PASS. DamageMaskNet 1/3 HTTP403 infrastructure fail, 2/3 HTTP429 infrastructure fail, 3/3 result NOT_VERIFIED; no fourth observability rerun. RefFace PREPARED/NOT_RUN.

Track A identity hypothesis: attempt1 `3e919f7a...` -> 4 failed/102 passed; attempt2 `7c683edd...` -> 1 failed/107 passed; attempt3 `84640fb7...` -> one static-contract failure only. Protocol-only wording corrections `ab69f18a...` and `9b8810ce...` changed no executable logic.

**EXP-20260819-011 — full-pytest reconciliation after identity targeted PASS:** Release Quality #128 exact HEAD `9b8810ce...`: targeted suite **108 passed**; full pytest **3 failed,543 passed**. Failures:
1. `test_preflight_cannot_mutate_true_import_snapshot`: synthetic autorun fixture reaches legacy `lab-histogram-proxy`; hardened V4 correctly rejects explicit proxy. Resolution must update the synthetic test fixture to provide deterministic SFace-like test evidence, not weaken production fail-closed behavior.
2. `test_v2_firewall_reads_v4_same_canvas_override_at_runtime`: stale fixture uses only broad `matched_original_reference_indices`; final V4 hardening requires explicit face-local identity bridge. Resolution must update the fixture/expectation to current strict contract, not restore broad same-canvas global trust.
3. `test_mosaic_same_canvas_identity_bridge_does_not_require_occlusion_seed`: true production logic defect. Mosaic localized corruption creates many gradient differences; current edge p90 rejects same canvas even though robust Lab comparison passes. DEC-009 attempt 1/3 will address only localized edge contamination while retaining strict global/face-local gates.

Release Quality artifact: `9364260439`, digest `ea77ae3fc9c570c53040cbb7ca4415460f24621be3112b560403d90f9cf1adef`.

---

## 15. QUALITY SCOREBOARD

DEV model evidence exists; broad validation incomplete. HOLDOUT: V1 historical 40/40; V3 consumed 39/40; V4 frozen/unexecuted. REAL-WORLD current same-head Female-domain is not yet final at this ledger update. TARGET-PC Paper Quality NOT_RUN. Maintain SFace, PSNR, SSIM, LPIPS, NIQE where useful, healthy MAE, damage recovery, wrong-person pixels, provenance violations, generated/reference/unresolved fractions, geometry, runtime and RAM separately by scope.

---

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU detected at runtime. CPU-first; no CUDA requirement. <=80% logical CPU, <=80% process/system RAM, one heavy model resident. Optional acceleration only after support and output-parity evidence.

---

## 17. RELEASE SAFETY RULES

SFace threshold `0.363`; wrong-person observed contribution `0 pixels`; provenance violations `0`; frozen healthy/outside MAE `<=8.0` where applicable; independent calibration. No threshold-shopping, cherry-picking, consumed-holdout rerun, difficult-case deletion, generated-as-observed provenance, wrong-person score rescue, auto-merge, force-push certified history or fabricated evidence.

---

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

---

## 19. TRACK A — PRODUCT_V1_1

Current exact HEAD `9b8810ce5e53daa92d4f13bcbd8b23c6a25df105`:
- V3 manifest verification PASS; no V3 execution.
- V4 freeze/history verification PASS; no V4 execution.
- targeted identity/source/partial/MAIN/provenance suite: **108/108 PASS**.
- full pytest: **3 failed,543 passed**.
- Windows #1311: FAIL; detailed independent packaging cause not yet established because Release Quality already shows full-test blockers.
- Female-domain #578: last seen `in_progress`; do not poll repeatedly.

Exact next technical action: implement DEC-009 attempt1/3 in `_same_canvas_match` with bounded localized photometric-outlier exclusion for the edge check; update only stale synthetic/binding test fixtures to the already-enforced V4 contract. Do not alter SFace/hardening/holdouts/models. Then ledger update, targeted relevant tests, full pytest and same-head release gates.

---

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`. Real CPU BFR/JPEG evidence; 80% governor; DamageMaskNet pipeline; Personalized Reference Bank; reference-first repair; candidate selector; deterministic component fusion; parser adapter; RefFace manual CPU workflow.

PDF-derived constraints verified: reference-guided inpainting should separate global identity from local component texture/control; cross-image information should be aggregated only between corresponding regions; severe BFR needs region-adaptive identity guidance rather than uniform generative pressure; MAIN preserves pose/composition/expression/geometry while references clarify identity/damage; unsupported fine detail must stay conservative. These support PRODUCT_V3/V4 research but do not qualify new models for V1.1.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1 + ONNX parity + RAM/runtime. Infrastructure fail -> infrastructure-only correction. True model/data quality fail -> U-Net hypothesis ends and a new documented lightweight segmentation architecture is required. Then RefFace attempt1/3.

---

## 22. SPECIALIST MODEL STRATEGY

`input -> detect/align -> damage -> references/identity -> specialist route -> candidates -> hard gates -> component fusion -> final identity -> provenance/export`. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid ref -> observed evidence first then qualified reference specialist. Never blindly chain all generators.

---

## 23. MODEL SELECTION POLICY

Select model winners only from multiple identity-disjoint DEV/VALIDATION cases per degradation. Identity hard gate precedes quality ranking. Measure geometry, artifacts, healthy preservation, PSNR/SSIM/LPIPS, runtime/RAM. Never select/tune using final holdout. Remove complexity that does not earn measured benefit.

---

## 24. HISTORICAL RECORD — append-only

- HIST-20260815-001: certified V1 merge `release/v1-certified@f476c6f... -> main@2767513f...`; historical Windows #1195/Female #463/Release Quality #13.
- HIST-20260818-002: Track A `3645c8c...`, Release Quality/Windows/Female all FAIL; V3 consumed; V4 frozen/unexecuted.
- HIST-20260818-003: Track B `645862d1...`, real DEV model evidence; DamageMaskNet attempt3 NOT_VERIFIED; RefFace NOT_RUN.
- HIST-20260819-004: canonical `meta/project-state` ledger established.
- HIST-20260819-005..010: identity hypothesis attempts 1–3 and their CI evidence; behavioral hypothesis closed after 3/3.
- HIST-20260819-011: protocol-only `84640fb7... -> ab69f18a...`, first static invariant restored.
- HIST-20260819-012: `ab69f18a...` Release Quality #127 remained one static failure; Windows stopped at tests.
- HIST-20260819-013: protocol-only `ab69f18a... -> 9b8810ce...`, complete static direct-edge wording restored; no executable logic change.
- **HIST-20260819-014:** exact `9b8810ce...` Release Quality #128: targeted identity/provenance **108 passed**; full pytest **3 failed,543 passed**. Artifact `9364260439`, digest `ea77ae3fc9c570c53040cbb7ca4415460f24621be3112b560403d90f9cf1adef`. Failure triage recorded in EXP-011. New DEC-009 same-canvas localized-damage hypothesis accepted before implementation.
