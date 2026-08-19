# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read before every engineering decision. GitHub evidence overrides chat memory. Current-state sections are maintained; important decisions, experiments, failures and technical pushes are preserved.

## 0. Document metadata

- Updated: `2026-08-19`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Last technical branch: `hotfix/real-world-restoration-v1.1`
- Previous technical HEAD: `9b8810ce5e53daa92d4f13bcbd8b23c6a25df105`
- Last technical HEAD: `2fcaeb1b10696894b6f8a412c9643b2965529ccc`
- Active engineering: Track A PRODUCT_V1_1 operational stabilization; Track B Paper Quality research.
- Track A current gate: same-canvas localized-damage hypothesis attempt **1/3** pushed; exact-head CI is **NOT_VERIFIED**.
- Track B blocker: recover already-triggered DamageMaskNet attempt-3 evidence; do not rerun merely for observability.

FORENSIC_MODE_READY: **TRUE for certified PRODUCT_V1 only**  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL — historical PRODUCT_V1 only**  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> evidence/tests -> commit/push -> exact remote SHA -> ledger update/push`. No auto-merge or certified-history force push.

---

## 1. Executive project summary

Conservative Face Studio is a local Windows face-restoration system for difficult smartphone/social-media portraits. Conservative Mode preserves MAIN pose/composition/geometry and uses verified observed evidence with exact provenance. Paper Quality Mode may generate unsupported missing detail, but generated pixels remain `GENERATED_MODEL_INFERRED`.

PRODUCT_V1 is certified and immutable. PRODUCT_V1_1 is an operational/safety hotfix isolated from Track B generators. Track B contains real CPU model evidence plus damage-routing, personalized-reference and component-fusion research.

The V1.1 identity-authority hypothesis is closed and the targeted identity/source/provenance suite is green. The current attempt addresses a separate same-canvas problem: localized mosaic corruption can create many artificial gradient edges even when the rest of the canvas is identical. Attempt 1 keeps the strict global color agreement, excludes only a small dilated local mismatch region from the secondary edge test, and requires substantial stable edge evidence to remain.

---

## 2. Branch and release map

| Branch | Purpose | Verified HEAD | State | CI / merge | Next gate |
|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | historical certified green | preserve |
| `feature/block-pipeline-v1` | V1 implementation history | `5eff667373cd47c07ba14aaad2acafee6d5a61c1` | MERGED / SUPERSEDED | historical | archive |
| `release/v1-certified` | V1 candidate history | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | FROZEN / ARCHIVED | merged via PR #1 | preserve |
| `hotfix/real-world-restoration-v1.1` | Track A | `2fcaeb1b10696894b6f8a412c9643b2965529ccc` | VALIDATING / ACTIVE | PR #2 OPEN/DRAFT; exact-head CI NOT_VERIFIED | relevant tests -> full pytest -> same-head release gates |
| `research/face-restoration-v2` | early dataset/degradation research | `757a3f6081b7b152cdc615a07cd99aec40fa0a1c` | SUPERSEDED AS ACTIVE ARCHITECTURE | not merged | preserve useful assets |
| `research/paper-quality-local-v2` | advanced Paper Quality | `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd` | ACTIVE / BENCHMARKING | partly NOT_VERIFIED | DamageMaskNet attempt-3 recovery |
| `meta/project-state` | canonical ledger | self-SHA intentionally omitted | ACTIVE META | documentation-only | update after every technical push |

Research branches diverged from the same certified base; the advanced branch is not falsely described as a literal Git superset of the early branch.

---

## 3. PRODUCT VERSION ROADMAP

- **PRODUCT_V1 — RELEASED:** certified conservative/forensic baseline; SFace `0.363`, wrong-person observed `0`, provenance violations `0`.
- **PRODUCT_V1_1 — VALIDATING:** operational hotfix. Identity targeted suite previously reached 108/108 PASS; current same-canvas attempt is separate and does not modify identity threshold/model policy.
- **PRODUCT_V2 — BENCHMARKING:** Paper Quality Local, damage-aware specialists, BFR candidates, generated provenance, hard gates, deterministic fusion, 80% resource contract.
- **PRODUCT_V3 — PLANNED with prototypes:** personalized MAIN + 0–9 references, per-component authority.
- **PRODUCT_V4 — PLANNED with prototypes:** damage-specialist hybrid routing/fusion.
- **PRODUCT_V5 — PLANNED:** unified modes, offline Windows model pack, one-click installer, clean Windows and real target-PC acceptance.

Product versions remain separate from holdout versions.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

CALIBRATION_V1 historical 60/60. FINAL_HOLDOUT_V1 historical 40/40. FINAL_HOLDOUT_V2 details not fully re-reconciled. FINAL_HOLDOUT_V3 **CONSUMED** 39/40, mosaic SFace `0.360 < 0.363`, NEVER rerun/tune. FINAL_HOLDOUT_V4 frozen independent 40 cases/20 identities, **NOT_RUN / UNCONSUMED**, one-shot only. V5 not created. Female-domain quick profile ~300–400 cases. Paper Quality DEV/VALIDATION separate. DamageMaskNet bank FairFace+ControlFace TRAIN/VALIDATION only.

V3/V4 are verification-only during current Track A work; neither is executed.

---

## 5. CURRENT GLOBAL OBJECTIVES

- OBJ-001 Preserve PRODUCT_V1 — PASS.
- OBJ-002 Restore PRODUCT_V1_1 gates — VALIDATING. Same-canvas attempt1 pushed at `2fcaeb1b...`, result NOT_VERIFIED.
- OBJ-003 Canonical ledger — IN_PROGRESS.
- OBJ-004 DamageMaskNet — BLOCKED pending existing attempt3 evidence.
- OBJ-005 Broad BFR selection — IN_PROGRESS.
- OBJ-006 FBCNN JPEG qualification — IN_PROGRESS.
- OBJ-007 Personalized Reference Bank validation — IN_PROGRESS.
- OBJ-008 RefFace CPU feasibility — BLOCKED by OBJ-004, 0/3 attempts consumed.
- OBJ-009 Paper Quality Windows pack/installer — PROPOSED.
- OBJ-010 HP EliteBook real acceptance — PROPOSED.

---

## 6. MODEL MASTER REGISTRY

Certified/current-role stack: YuNet, SFace `0.363`, NAFNet, Face Parsing ResNet18 ONNX, Head Pose MobileNetV2 ONNX, constrained LaMa ONNX.

Research states: GPEN BENCHMARKING/license blocker; GFPGAN v1.4 BENCHMARKING; CodeFormer BENCHMARKING/BLOCKED_LICENSE; FBCNN BENCHMARKING/current DEV JPEG leader; DamageMaskNet BENCHMARKING/BLOCKED; RefFace FEASIBILITY_ONLY/NOT_RUN; InstantRestore/OSDFace hardware-blocked feasibility; RestoreFormer++/VQFR/GPEN-inpaint/RefineFIR/PerFuSe/RefIPFR/Real-ESRGAN feasibility until measured.

Registry documentation mismatch remains: certified `THIRD_PARTY_MODULES.md` references absent machine-readable files under `models/`; actual active registry logic is under `app/`. Do not invent manifests/hashes.

---

## 7. CURRENT MODEL EVIDENCE

Linux CPU DEVELOPMENT only: GPEN SFace `0.95397`, PSNR `28.07`, SSIM `0.7474`, `~2.697s`, `~1.828GB`; GFPGAN v1.4 SFace `0.91665`, PSNR `30.65`, SSIM `0.8604`, `~2.787s`, `~1.666GB`; FBCNN QF20 SFace `0.9571→0.9691`, PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, peak RSS `~1.305GB`; CodeFormer real CPU slice PASS, exact comparative metrics must be reread from artifact.

---

## 8. 13-BLOCK ARCHITECTURE

1 IMPORT deterministic. 2 DEBLUR NAFNet/future measured BFR. 3 ENHANCE FBCNN for JPEG. 4 LANDMARKS YuNet/pose. 5 ALIGN deterministic. 6 OCCLUSION_MASK parser + DamageMaskNet target. 7 REGION_SELECT component/reference bank. 8 INPAINT observed reference first, Paper generation only as GENERATED. 9 FUSION healthy MAIN > observed same-person ref > accepted generated. 10 FRONTALIZE geometry-only Conservative. 11 IDENTITY_CHECK SFace `0.363`, direct/non-transitive targeted suite green. 12 UPSCALE Lanczos/optional measured SR. 13 EXPORT deterministic provenance + future model/damage/resource reports.

---

## 9. PHOTO AND INPUT CONTRACT

MAIN: low-res phone/social-media, JPEG/double-JPEG, blur/noise, pixelation/mosaic, scribble/sticker/black-bar/opaque loss, covered/missing components, crop/partial, low light, mixed damage. References: MAIN + 0–9 full/partial/component-only/angle/expression/light/resolution/degraded/useless/wrong-person. Full accepted same-person may global-anchor; partial remains local; wrong-person never anchor/donor/identity booster.

---

## 10. DATASET CONSTRUCTION

Initial Paper Quality target ~300–400 representative cases with explicit female-domain proportion. Identity-disjoint TRAIN/DEV/VALIDATION/FINAL_HOLDOUT. Store source/license/date/identity/hash/resolution/domain/split/degradation/severity/seed/exact mask/reference relationships. Never tune/train on final holdout.

---

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Track LEFT/RIGHT EYE, LEFT/RIGHT EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT/RIGHT CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR. Record MAIN visibility/damage, observed refs/confidence, generated candidates, selected source/provenance, identity/geometry and unresolved state. Observed same-person evidence outranks generation.

---

## 12. DAMAGE ROUTING

HEALTHY preserve MAIN; BLUR -> NAFNet/measured deblur then Paper BFR only if needed; JPEG -> FBCNN; PIXELATION/MOSAIC -> observed component first then Paper generation; SCRIBBLE/STICKER/OPAQUE/BLACK_BAR -> observed ref then qualified reference specialist; PARTIAL/MISSING -> component bank then Paper fallback; LOW_LIGHT -> detected specialist only; MIXED -> minimum necessary candidates, never blind generator chaining.

---

## 13. DECISION LOG

DEC-001 canonical ledger ACCEPTED. DEC-002 advanced Paper Quality branch active ACCEPTED. DEC-003 <=80% logical CPU/process/system RAM + one heavy model ACCEPTED. DEC-004 evidence authority order ACCEPTED. DEC-005 mixed DamageMaskNet source bank ACCEPTED. DEC-006 RefFace after DamageMaskNet ACCEPTED/BLOCKED. DEC-007 V3 consumed/V4 untouched ACCEPTED. DEC-008 ranking cluster != identity authority ACCEPTED; identity behavioral hypothesis CLOSED after 3/3 and targeted suite PASS. **DEC-009 localized-damage same-canvas edge isolation ACCEPTED, attempt1/3 now IMPLEMENTED/VALIDATING.**

Attempt1 rule: strict global Lab median/p90 is unchanged. If photometric mismatches are local and <=10% of comparable pixels, only a 5x5-dilated mismatch neighborhood is excluded from the secondary gradient comparison. At least `max(64, 35% of original edge count)` stable edges must remain. Face-local identity proof, SFace threshold, provenance and wrong-person rules remain unchanged.

---

## 14. EXPERIMENT LOG

Historical model evidence as above. DamageMaskNet attempts: 1/3 HTTP403 infrastructure fail, 2/3 HTTP429 infrastructure fail, 3/3 result NOT_VERIFIED; no attempt4 for observability. RefFace PREPARED/NOT_RUN.

Identity Track A hypothesis: attempts 1–3 closed; targeted suite at `9b8810ce...` reached 108/108 PASS. Release Quality #128 full pytest was 3 failed/543 passed.

**EXP-20260819-012 — DEC-009 same-canvas attempt 1/3:** technical push `9b8810ce5e53daa92d4f13bcbd8b23c6a25df105 -> 2fcaeb1b10696894b6f8a412c9643b2965529ccc`.
- `app/primary_anchor_policy.py`: keep strict global Lab same-canvas gate; exclude only bounded localized photometric corruption from secondary edge consistency; require stable-edge support.
- `tests/test_automatic.py`: immutable-import plumbing fixture now supplies explicit deterministic SFace-like embeddings instead of depending on the production-forbidden histogram proxy. Production fail-closed behavior is not weakened.
- `tests/test_identity_anchor_v4_binding.py`: positive fixture now uses explicit strict face-local identity-bridge fields; negative test proves broad same-canvas evidence alone does not override V2 rejection.
- Models/checkpoints/SFace threshold/holdouts/provenance classes: unchanged.
- Result: NOT_VERIFIED at ledger update.

---

## 15. QUALITY SCOREBOARD

DEV model evidence exists; broad validation incomplete. HOLDOUT V1 historical 40/40; V3 consumed 39/40; V4 frozen/unexecuted. REAL-WORLD Female-domain current-head result not yet known. TARGET-PC Paper Quality NOT_RUN. Keep all scopes separate.

---

## 16. TARGET HARDWARE

HP EliteBook 1030 G3, 16GB Windows; exact CPU/GPU runtime-detected. CPU-first/no CUDA requirement. <=80% logical CPU, <=80% process/system RAM, one heavy model. Optional acceleration only after support/output-parity evidence.

---

## 17. RELEASE SAFETY RULES

SFace `0.363`; wrong-person observed contribution `0 pixels`; provenance violations `0`; frozen healthy/outside MAE `<=8.0` where applicable. No threshold-shopping, cherry-picking, consumed-holdout reruns, difficult-case removal, generated-as-observed, wrong-person score rescue, auto-merge, force-push certified history or fabricated evidence.

---

## 18. PROVENANCE CLASSES

`MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

---

## 19. TRACK A — PRODUCT_V1_1

Previous exact HEAD `9b8810ce...`: V3/V4 verification PASS without execution; targeted suite **108/108 PASS**; full pytest **3 failed,543 passed**; Release Quality artifact `9364260439`, digest `ea77ae3fc9c570c53040cbb7ca4415460f24621be3112b560403d90f9cf1adef`.

Current exact HEAD `2fcaeb1b10696894b6f8a412c9643b2965529ccc`: DEC-009 attempt1 plus two stale-test-fixture corrections. No SFace/model/threshold/holdout/provenance weakening. **Exact-head CI NOT_VERIFIED.**

Next exact action: inspect relevant same-head test result. If relevant/full pytest PASS, continue to model-pack/calibration and same-head Windows/Female/Release Quality. If same-canvas behavior fails, only attempts 2/3 and 3/3 remain under DEC-009. V3/V4 remain unexecuted.

---

## 20. TRACK B — PAPER QUALITY

Active `research/paper-quality-local-v2@645862d1...`: real CPU BFR/JPEG evidence, 80% governor, DamageMaskNet pipeline, Personalized Reference Bank, reference-first repair, hard-gated selector, deterministic fusion, parser adapter, RefFace manual workflow.

PDF-derived constraints verified: reference-guided inpainting should separate global identity from local component texture; correspondence should limit information exchange to matching regions; severe BFR benefits from region-adaptive identity guidance; MAIN preserves pose/composition/expression/geometry; references clarify identity/damaged detail; unsupported fine detail stays conservative. These support PRODUCT_V3/V4 research, not V1.1 model substitution.

---

## 21. CURRENT PAPER QUALITY BLOCKER

Recover DamageMaskNet attempt3 without rerun/tuning. PASS -> per-class IoU/F1 + ONNX parity + RAM/runtime. Infrastructure fail -> infrastructure-only correction. True model/data fail -> U-Net hypothesis ends; document a new lightweight architecture. Then RefFace attempt1/3.

---

## 22. SPECIALIST MODEL STRATEGY

`input -> detect/align -> damage -> reference/identity -> specialist -> candidates -> hard gates -> component fusion -> final identity/provenance`. JPEG -> FBCNN. Blur -> measured deblur/BFR. Opaque loss + valid reference -> observed evidence first then qualified reference specialist. Never blindly chain all generators.

---

## 23. MODEL SELECTION POLICY

Select winners on multiple identity-disjoint DEV/VALIDATION cases per damage family. Identity hard gate precedes quality ranking. Measure geometry/artifacts/healthy preservation/PSNR/SSIM/LPIPS/runtime/RAM. Never select/tune using final holdout.

---

## 24. HISTORICAL RECORD — append-only

- HIST-20260815-001: certified V1 merge `f476c6f... -> 2767513f...`, historical Windows #1195/Female #463/Release Quality #13.
- HIST-20260818-002: Track A `3645c8c...`, Release Quality/Windows/Female FAIL; V3 consumed; V4 frozen/unexecuted.
- HIST-20260818-003: Track B `645862d1...`, real DEV evidence; DamageMaskNet attempt3 NOT_VERIFIED; RefFace NOT_RUN.
- HIST-20260819-004: canonical meta ledger established.
- HIST-20260819-005..010: identity attempts 1–3 and evidence; behavioral hypothesis closed.
- HIST-20260819-011..013: protocol-only source-contract restoration, no executable behavior change.
- HIST-20260819-014: `9b8810ce...` Release Quality #128 targeted 108 PASS; full pytest 3 failed/543 passed; failure triage and DEC-009 created.
- **HIST-20260819-015:** technical push `9b8810ce5e53daa92d4f13bcbd8b23c6a25df105 -> 2fcaeb1b10696894b6f8a412c9643b2965529ccc`. Files: `app/primary_anchor_policy.py`, `tests/test_automatic.py`, `tests/test_identity_anchor_v4_binding.py`. Same-canvas hypothesis attempt1/3; stale test fixtures aligned to hardened V4 contract. Models/checkpoints/thresholds/holdouts unchanged. Exact-head result NOT_VERIFIED at ledger update.
