# Conservative Face Studio — Project Master State

> **CANONICAL PROJECT LEDGER.** Read this file before every engineering decision. GitHub evidence overrides chat memory. The complete prior ledger through 2026-08-24 is preserved byte-for-byte at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md` (historical blob `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`). Do not rewrite or delete that archive.

## 0. Document metadata

- Last ledger update: `2026-08-28`
- Technical state verified at: `2026-08-28`
- Repository: `xhinoo97-svg/ConservativeFaceStudio`
- Canonical state branch: `meta/project-state`
- Last technical branch: `integration/final-paper-quality-local`
- Previous technical HEAD: `f5ca07e0b5268ec2b8843f9dce93b5d6a9fdf5cd`
- Last technical HEAD: `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a`
- Last technical tree: `f3984f16751d648b7a9ffbd726c64ac50f3ad21b`
- Certified base: `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- Current active engineering track: single final integration line, upstream-first Paper Quality + preserved Conservative safety
- Current exact blocker: no research-heavy restoration model has complete production evidence. Generated-route authorization is now bound to a real `ModelQualification` attestation, but the next boundary must also bind the actual `RestorationCandidate` to the exact official repository revision and checkpoint hash that were qualified.
- Overall project status: `PARTIAL`

FORENSIC_MODE_READY: **TRUE** — certified PRODUCT_V1 only  
PAPER_QUALITY_MODE_READY: **FALSE**  
WINDOWS_INSTALLER_READY: **PARTIAL** — historical/exact-head Track A CI evidence exists; final unified Paper Quality installer and physical target-PC acceptance do not  
TARGET_HARDWARE_READY: **FALSE**  
QUALITY_TARGET_ACHIEVED: **FALSE**  
PROJECT_FINISHED: **FALSE**

Mandatory sequence: `technical work -> tests/evidence -> push -> exact remote SHA -> ledger update`. No force-push of certified history, no auto-merge, no consumed-holdout rerun.

---

## 1. Executive project summary

Conservative Face Studio is a local Windows face-restoration application for damaged smartphone/social-media portraits. Its final product is intentionally different from generic blind face enhancement: it separates an evidence-faithful Conservative Mode from a Paper Quality Mode that may generate unsupported detail only when that detail is explicitly labeled `GENERATED_MODEL_INFERRED`.

The immutable certified V1 remains the safety reference. The current product engineering line is `integration/final-paper-quality-local`, created from immutable `main`, and selectively integrates tested operational/safety foundations plus upstream-first Paper Quality components. Official paper/model implementations are reused directly when executable; CFS owns thin adapters, exact revision/checkpoint verification, identity/provenance firewalls, resource limits, deterministic routing/fusion, Windows/offline packaging and release qualification.

Current maturity: the integration architecture is substantially built and has a green same-HEAD full test suite, but no new heavy research model is production-qualified. The biggest technical limitation is end-to-end evidence binding from a production qualification to the exact model bytes that generated a candidate. The biggest quality limitation is that broad identity-disjoint validation and Target95-quality evidence remain incomplete; Paper Quality quality claims cannot be inferred from paper figures or one-identity DEVELOPMENT tests.

Current milestone: make every generated candidate cryptographically/auditably traceable to the exact qualified official repository revision and checkpoint, then continue real multi-identity validation and Windows/EliteBook qualification model by model.

---

## 2. Branch and release map

| Branch | Purpose | Base | Current verified HEAD | Status | CI state | Merge status | Superseded by | Next gate |
|---|---|---|---|---|---|---|---|---|
| `main` | certified PRODUCT_V1 | historical | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | historical certified green | main | none | preserve |
| `integration/final-paper-quality-local` | single final product integration | exact certified `main` | `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a` | ACTIVE / IMPLEMENTING | run `33207031788` SUCCESS; targeted `46/46`, full `602/602` | not merged | none | bind candidate bytes to qualification evidence |
| `hotfix/real-world-restoration-v1.1` | Track A operational/safety evidence | `main` | `77687b3b171f4e9989fcf486834f2d8b7a52f591` | FROZEN NO-GO EVIDENCE LINE | prereqs historically green; V4 final consumed FAIL | PR #2 OPEN/DRAFT/NOT MERGED | integration line for future product work | never rerun V4 |
| `protocol/v5-certification-hardening` | future one-shot protocol DEV | Track A history | `268188c5a2540455ff804383cb583b16546b62f1` | ARCHIVED DEV EVIDENCE | synthetic protocol PASS | not merged | integration contains selected protocol work | no V5 until quality prerequisites |
| `research/paper-quality-local-v2` | advanced research/evidence line | `main` | `6d57725aae087bb4a3144d521d91346999f9a4fd` | SUPERSEDED AS ACTIVE ARCHITECTURE / PRESERVE EVIDENCE | research evidence exists | not merged | `integration/final-paper-quality-local` | preserve/port only measured winners |
| `research/face-restoration-v2` | early dataset/degradation research | `main` | `757a3f60e2f012a1d0b1758c7280bfdd492f33df` | SUPERSEDED / ARCHIVED | historical | not merged | advanced research/integration | preserve useful assets |
| `feature/block-pipeline-v1` | V1 history | historical | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | MERGED / ARCHIVED | historical | superseded | main | preserve |
| `release/v1-certified` | V1 release history | historical | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / ARCHIVED | historical | merged | main | preserve |
| `meta/project-state` | canonical ledger | metadata only | self-SHA intentionally not recorded | ACTIVE META | docs only | n/a | none | update after every technical push |

---

## 3. PRODUCT VERSION ROADMAP

### PRODUCT_V1 — RELEASED
- Objective/user purpose: conservative/forensic restoration baseline.
- Base: certified `main@2767513f...`.
- Architecture/models: 13-block conservative pipeline; YuNet, SFace, NAFNet, face parsing, head pose, constrained LaMa and deterministic geometry/fusion.
- Safety: SFace `0.363`, wrong-person observed contribution `0`, provenance violations `0`.
- Windows: historically certified. EliteBook Paper Quality target evidence is separate.

### PRODUCT_V1_1 — FAILED AS V4 CERTIFICATION CANDIDATE / PRESERVED
- Objective: operational real-world hotfix without weakening V1 safety.
- Evaluated candidate: `b6ce7ebd...`; Track A final state `77687b3b...`.
- V4: `CONSUMED_FAIL`, 0/40 cases completed because the runner failed after STARTED marker but before case 1.
- Acceptance: NO-GO. V4 can never be rerun.

### PRODUCT_V2 — IMPLEMENTING / BENCHMARKING
- Objective: Paper Quality Local with modern priors and specialist routing.
- Base/branch: active integration from immutable main.
- Models under evidence: official FBCNN, GPEN, GFPGAN1.4, CodeFormer; LR-ASPP damage-mask replacement remains development-only.
- Resource target: <=80% logical CPU, <=80% process/system RAM, one heavy model at a time.
- Windows/EliteBook: final unified evidence NOT_VERIFIED/NOT_RUN.

### PRODUCT_V3 — DESIGNING / PROTOTYPES
- Objective: Personalized Multi-Reference Restoration, MAIN + 0–9 same-person references, per-component authority.
- Completed foundation: 13-component bank, global-vs-local reference authority, reference-first routing.
- Production qualification: FALSE.

### PRODUCT_V4 — DESIGNING / PROTOTYPES
- Objective: Damage-Specialist Hybrid Architecture with class-specific routes and component-aware fusion.
- Completed foundation: fail-closed damage router, candidate selector, deterministic fusion, route authorization.
- Production qualification: FALSE.

### PRODUCT_V5 — PLANNED
- Objective: unified Conservative + Paper Quality + personalized + specialist routing + single offline Windows installer + physical EliteBook acceptance.
- V5 holdout: DOES NOT EXIST.
- Creation/execution is forbidden until all prerequisites are independently green.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

| Evaluation set | State | Tuning allowed? | Key evidence |
|---|---|---:|---|
| CALIBRATION_V1 | historical | historical only | prior certified calibration |
| FINAL_HOLDOUT_V1 | historical | no | certified V1 evidence |
| FINAL_HOLDOUT_V2 | historical / not fully reconciled here | no unless explicit lineage proves otherwise | preserve records |
| FINAL_HOLDOUT_V3 | **CONSUMED** | **NO** | 39/40; mosaic SFace `0.360 < 0.363`; never rerun/tune |
| FINAL_HOLDOUT_V4 | **CONSUMED_FAIL** | **NO** | STARTED persisted; runner failed before case 1; 0/40; never rerun/tune |
| FINAL_HOLDOUT_V5 | **NOT_CREATED** | NO | future identity-disjoint one-shot only after prerequisites |
| Female-domain | stress/evidence | development/reporting only unless explicitly frozen | historical 300–400-case runs; Target95 not achieved |
| FBCNN DEV matrix | DEVELOPMENT | yes, within DEV rules | one identity, six compression profiles PASS |
| DamageMaskNet mixed-source | TRAIN/VALIDATION research | yes, non-holdout | small U-Net hypothesis stopped |
| LR-ASPP external DEV validation | DEVELOPMENT | yes | 40 identities / 880 cases, aggregate gate pass but subgroup gaps |

Consumed V3/V4 data are immutable evidence and cannot be renamed, rerun or used for tuning.

---

## 5. CURRENT GLOBAL OBJECTIVES

- **OBJ-001 Preserve certified V1 — PASS.** Immutable `main` remains unchanged.
- **OBJ-002 Preserve Track A/V4 evidence — PASS.** V4 remains `CONSUMED_FAIL`; no rerun.
- **OBJ-003 Canonical ledger — IN_PROGRESS.** Current state reconciled here; historical prior ledger archived byte-for-byte.
- **OBJ-004 DamageMaskNet replacement — IN_PROGRESS.** Small U-Net STOPPED; LR-ASPP development evidence exists but production/domain/license gates remain.
- **OBJ-005 Broad BFR selection — IN_PROGRESS.** Upstream-first GPEN/GFPGAN/CodeFormer evidence must expand beyond small DEV slices.
- **OBJ-006 FBCNN JPEG qualification — IN_PROGRESS.** DEV leader; production evidence incomplete.
- **OBJ-007 Personalized Reference Bank — IN_PROGRESS.** Foundation tested; final production runtime/validation incomplete.
- **OBJ-008 RefFace specialist — BLOCKED.** No production-qualified damage mask and no target-PC evidence.
- **OBJ-009 Paper Quality Windows pack — PROPOSED/IN_PROGRESS foundation.** No final unified installed-offline qualification yet.
- **OBJ-010 Physical HP EliteBook acceptance — PROPOSED.** NOT_RUN for final Paper Quality product.
- **OBJ-011 Official upstream registry — PASS as architecture foundation.** Runtime qualification remains per model.
- **OBJ-012 Exact generated-model authority — IN_PROGRESS.** Route-plan and production attestation are bound; next bind candidate repository/revision/checkpoint bytes.
- **OBJ-013 Future V5 protocol — VALIDATING DEV only.** Synthetic protocol green; no V5 object/holdout.

---

## 6. MODEL MASTER REGISTRY

| Model | Role | Upstream / revision | Checkpoint | License state | CPU/Windows state | Current status | Why / blocker |
|---|---|---|---|---|---|---|---|
| YuNet | detection/5pt | OpenCV Zoo | production pinned | accepted V1 | certified V1 | QUALIFIED V1 | preserve |
| SFace | identity | OpenCV Zoo | production pinned | accepted V1 | certified V1 | QUALIFIED V1 | threshold frozen `0.363` |
| NAFNet | mild deblur/denoise | existing OpenCV Zoo/CFS asset | production pinned | accepted V1 | certified V1 | QUALIFIED V1 | not facial-prior generator |
| Face Parsing ResNet18 ONNX | semantic parsing | yakhyo/face-parsing | pinned V1 | MIT evidence | V1 | QUALIFIED V1 | used as semantic support |
| Head Pose MobileNetV2 | pose | pinned V1 | pinned V1 | recorded | V1 | QUALIFIED V1 | geometry only |
| LaMa | residual non-identity-critical inpaint | OpenCV asset | pinned V1 | recorded | V1 | QUALIFIED/FALLBACK V1 | never identity evidence |
| FBCNN | JPEG/recompression specialist | `jiaxi-jiang/FBCNN@54d1831927506b3247e2d4d245abb4f4dab1a1cd` | `fbcnn_color.pth`, SHA `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8` | code Apache-2.0; final weights redistribution evidence still must be release-audited | Linux CPU DEV PASS; final Windows/EliteBook incomplete | DEV_PASS / BENCHMARKING | current JPEG leader; one-identity matrix insufficient |
| GPEN BFR-512 | severe blind BFR | `yangxy/GPEN@2c736702983368847fb544d234a22ac7cff25802` | official research checkpoint | redistribution/license blocker | Linux CPU DEV evidence | BENCHMARKING / BLOCKED_LICENSE | strong identity DEV evidence, broad validation absent |
| GFPGAN v1.4 | blind BFR | `TencentARC/GFPGAN@7552a7791caad982045a7bbe5634bbf1cd5c8679` | official v1.4 | code Apache-2.0; asset terms tracked separately | Linux CPU DEV evidence | BENCHMARKING | broad validation/Windows/EliteBook absent |
| CodeFormer | severe BFR/inpainting | `sczhou/CodeFormer@b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b` | official | S-Lab/non-commercial blocker | Linux CPU DEV PASS | BLOCKED_LICENSE / BENCHMARKING | cannot production-qualify while license incompatible |
| DamageMaskNet small U-Net | damage segmentation | CFS-trained hypothesis | hashes preserved in archive/history | internal research | CPU/export parity PASS | REJECTED / STOPPED | macro-F1 `0.173198`; six classes F1 zero |
| LR-ASPP damage mask | damage segmentation replacement | official `pytorch/vision` architecture at pinned revision | CFS-trained DEV checkpoint | code BSD-3-Clause; checkpoint redistribution not production-cleared | CPU/ONNX DEV evidence | VALIDATION_PASS aggregate / NOT_QUALIFIED | subgroup/domain gaps + license/production evidence |
| RefFaceInpainting | reference-guided occlusion | official repository audited historically | official checkpoint audit incomplete for production | MIT repo; full production evidence incomplete | CPU vertical slice NOT_RUN | FEASIBILITY_ONLY / BLOCKED | waits for qualified mask + target-PC feasibility |
| InstantRestore | personalized multi-reference | `snap-research/InstantRestore@05891bf7d30ab7290c501272de7a1a4a51b21b4f` | official research weights | upstream/license audit incomplete | CUDA-oriented; CPU/EliteBook not qualified | BLOCKED_HARDWARE / FEASIBILITY_ONLY | scientifically close to use case but too heavy until proven |
| OSDFace / RestoreFormer++ / VQFR / FaceMe / others | challengers/teachers | preserve upstream audits | varies | varies | NOT_VERIFIED for target product | DISCOVERED/AUDITED/FEASIBILITY_ONLY | benchmark only when justified |

`IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED` remains mandatory.

---

## 7. CURRENT MODEL EVIDENCE

Historical Linux CPU DEVELOPMENT only:
- GPEN BFR-512: SFace `0.95397`, `~2.697s`, peak RSS `~1.828GB`, PSNR `28.07`, SSIM `0.7474` on one DEV case.
- GFPGAN v1.4: SFace `0.91665`, `~2.787s`, peak RSS `~1.666GB`, PSNR `30.65`, SSIM `0.8604` on the same style of DEV evidence.
- FBCNN QF20 historical slice: PSNR `34.62 -> 36.78`, SSIM `0.9486 -> 0.9634`, SFace `0.9571 -> 0.9691`, peak RSS `~1.305GB`.
- FBCNN public compression matrix run `32674085939`: 6/6 DEV profiles PASS, one identity only, zero runtime/provenance/wrong-person failures; artifact `9502200502`, archive SHA-256 `365251ee8b17dc31099569d328e52439fd6440e869f0ddbe16c4cb4116112842`.
- CodeFormer: real CPU aligned slice PASS at official `w=0.5`; exact comparative metrics remain artifact-bound and are not reconstructed from memory.
- DamageMaskNet small U-Net: export/parity infrastructure PASS; macro-F1 `0.173198`, macro-IoU `0.113028`; hypothesis STOPPED.
- LR-ASPP external DEVELOPMENT: 40 identities, 880 cases; overall F1 `0.716639`, IoU `0.579849`; subgroup/min-class weaknesses remain and production is not qualified.

Paper-reported metrics are scientific references only. They are not CFS, Windows or EliteBook measurements unless reproduced by a recorded CFS experiment.

---

## 8. 13-BLOCK ARCHITECTURE

| # | Block | Current function / authority | Current/primary model | Future/specialist direction | Status / next action |
|---:|---|---|---|---|---|
| 1 | IMPORT | immutable observed MAIN | deterministic OpenCV/Pillow | unchanged | stable |
| 2 | DEBLUR | mild conservative deblur | NAFNet | measured GPEN/GFPGAN/CodeFormer candidates only when damage/router says so | model selection incomplete |
| 3 | ENHANCE | conservative enhancement/JPEG route | FBCNN candidate for JPEG; NAFNet general | FBCNN only when compression detected | FBCNN DEV only |
| 4 | LANDMARKS | face/5pt/pose evidence | YuNet + pose | hard-pose specialist only if measured | stable foundation |
| 5 | ALIGN | deterministic geometry | affine/similarity/RANSAC | no generator | stable foundation |
| 6 | OCCLUSION_MASK | damage evidence contract | diagnostic taxonomy; no production mask model yet | LR-ASPP or better qualified lightweight segmenter | production blocker |
| 7 | REGION_SELECT | 13-component observed authority/ranking | component bank + personalized selector | multi-reference quality/pose aware selection | tested foundation |
| 8 | INPAINT | observed evidence first | reference-first route / constrained LaMa residual | RefFace or other qualified specialist only after mask/model gates | blocked for research specialist |
| 9 | FUSION | MAIN > observed reference > generated | deterministic component fusion | generated candidate only with calibrated + production authority | attestation binding now green |
| 10 | FRONTALIZE | geometry-only Conservative | pose transforms | Paper generation only if separately authorized | no hidden-side invention in Conservative |
| 11 | IDENTITY_CHECK | final biometric safety | SFace | optional second backend only after independent qualification | threshold frozen `0.363` |
| 12 | UPSCALE | deterministic safe upscale | Lanczos | optional measured SR | no unqualified global SR |
| 13 | EXPORT | deterministic evidence/report | CFS exporter | include model attestation, hashes, resource/timing | candidate-byte binding next |

---

## 9. PHOTO AND INPUT CONTRACT

MAIN target: low resolution, smartphone/social compression, JPEG/double-JPEG, defocus/motion/mixed blur, noise, pixelation/block mosaic, scribble, sticker, black bar, opaque mask, partially/fully covered eye/mouth/nose, missing component, crop/partial face, low light/uneven exposure, combined/unknown real-world corruption.

References: MAIN + 0–9 images; full face, partial face, component-only, different pose/expression/light/resolution, blurred/compressed/partially occluded/useless/wrong-person references.

Identity eligibility is separate from image quality. Accepted full same-person references may be global anchors; accepted partial references are component-local only; wrong-person references are never anchors, never pixel donors and never score boosters.

---

## 10. DATASET CONSTRUCTION

Large Paper Quality research/validation target remains approximately 300–400 representative face sources/cases with explicit domain composition. TRAIN / DEVELOPMENT / VALIDATION / FINAL_HOLDOUT must be identity-disjoint where required. Store source, license/usage basis, date, identity ID, hash, original resolution, domain metadata, split, degradation/severity/seed/mask and reference relationships. Final holdouts are never training/tuning data.

The stopped DamageMaskNet bank and LR-ASPP DEVELOPMENT evidence remain research datasets, not final certification data.

---

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Canonical components: LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE, PHILTRUM, MOUTH_LIPS, LEFT_CHEEK, RIGHT_CHEEK, CHIN, JAW, FOREHEAD, FACE_CONTOUR.

For every component track MAIN visibility/damage, best/alternate observed reference, confidence, generated candidate availability, selected source, provenance, identity/geometry consistency and unresolved state. Observed same-person evidence outranks generated inference.

---

## 12. DAMAGE ROUTING

Current fail-closed route catalog covers HEALTHY, GAUSSIAN_BLUR, MOTION_BLUR, DEFOCUS, JPEG_ARTIFACT, NOISE, PIXELATION, OCCLUSION, SCRIBBLE, TEXT_WATERMARK, MIXED, SMALL_FACE and PARTIAL_CROP, mapped from the frozen damage taxonomy and explicit verified secondary evidence where needed.

Routing principles:
- HEALTHY -> preserve MAIN.
- JPEG/recompression -> FBCNN only after production qualification.
- mild blur -> NAFNet; severe blind restoration candidates remain validation-gated.
- mosaic/pixelation -> observed component evidence first; generation only through a qualified route.
- sticker/scribble/opaque/black-bar/missing component -> observed reference first; reference-guided generator only if separately qualified.
- malformed/unverified routing evidence -> ROLLBACK/ABSTAIN, never model guessing.

Every generated candidate now requires route-plan model key + production attestation agreement before fusion. Next step is binding that attestation to the exact repository revision/checkpoint recorded by the candidate itself.

---

## 13. DECISION LOG — current decisions

### DEC-20260828-014 — Bind generated route authority to production attestation
- DATE: 2026-08-28
- PROPOSAL: a `DamageRoutePlan` alone is insufficient authority; generated pixels require the exact `ModelQualification` production attestation that created the route.
- PROBLEM: a manually constructed/altered route with `qualified_for_execution=True` could otherwise bypass model-gate evidence at the fusion boundary.
- AFFECTED VERSION: PRODUCT_V2/V4/V5 planning.
- BLOCKS: 8/9/13.
- MODELS: all future generated specialists.
- EVIDENCE: same-HEAD run `33207031788`, targeted 46/46 and full 602/602 PASS.
- EXPECTED BENEFIT: fail-closed model authority, auditable generation, no arbitrary production boolean.
- RISKS: interface becomes stricter; callers must supply the actual qualification object.
- ALTERNATIVES: trust `selected_model_key`; rejected because model name alone does not prove evidence.
- REVERSAL CONDITION: only if a stronger cryptographically/auditably equivalent production authority replaces it.
- STATUS: ACCEPTED / IMPLEMENTED / TEST_PASS.

### DEC-20260828-015 — Upstream model bytes must be bound to qualification
- DATE: 2026-08-28
- PROPOSAL: extend the candidate contract so generated output identifies official repository, exact revision and checkpoint SHA-256, and compare those values with typed qualification evidence before fusion/export.
- PROBLEM: current attestation binds route to model evidence, but candidate currently exposes only model key/version generically; FBCNN already reports upstream identity in `quality_metrics` but the generic contract does not enforce it.
- STATUS: ACCEPTED / NEXT IMPLEMENTATION.

Earlier decisions and their full historical detail are preserved in the archived ledger file referenced at the top of this document.

---

## 14. EXPERIMENT LOG — current experiment

### EXP-20260828-034 — Production-attestation route binding
- HYPOTHESIS: generated Paper Quality pixels must not be fusible from a route-plan boolean/model key alone.
- MODEL: model-independent contract; synthetic `ref_face_inpainting` qualification fixture.
- ATTEMPT: 1/3.
- DATASET: unit/synthetic protocol DEV only; no final holdout.
- BACKEND/OS: GitHub Actions Ubuntu 24.04, Python 3.11.16.
- TECHNICAL COMMITS: `f33880d5aa05bf576c1f36766046a4b570ec8517` -> `75b31aabb1c52c77d8af6eed270b1ef5c64dcf1c` -> `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a`.
- WORKFLOW: `Final integration one-shot protocol DEV`, run `33207031788`, attempt 1.
- RESULT: SUCCESS.
- TESTS: targeted `46/46`; full pytest `602/602`; synthetic success/pre-marker failure/post-marker failure ordering PASS.
- ARTIFACT: `one-shot-protocol-hardening-33207031788`, ID `9700075188`, archive SHA-256 `8362ebb4ff8f9391256d2ea87c9b7380296ae7f4f0f4d7666e0df861afae4842`.
- CONCLUSION: route/model attestation bypass is closed at the orchestration boundary. This does not qualify a model or measure image quality.
- NEXT ACTION: candidate upstream-byte binding.

---

## 15. QUALITY SCOREBOARD

- DEV: model evidence listed in Section 7; FBCNN current compression leader in limited DEV evidence.
- VALIDATION: LR-ASPP has identity-disjoint external DEVELOPMENT-style validation but domain gaps; heavy BFR broad validation incomplete.
- HOLDOUT: V3 consumed 39/40; V4 consumed fail 0/40 after marker; both forbidden for tuning/reuse.
- REAL-WORLD: incomplete for final unified product.
- TARGET-PC: NOT_RUN for final Paper Quality product.
- SFace release threshold: `0.363` unchanged.
- wrong-person observed pixels target: `0`.
- provenance violations target: `0`.
- healthy/outside MAE frozen limit where applicable: `<=8.0`.
- Target95: NOT_ACHIEVED / not currently valid as final product claim.

---

## 16. TARGET HARDWARE

Primary target: HP EliteBook 1030 G3, Windows, 16 GB RAM. Exact CPU/GPU must be runtime-detected.

Resource contract:
- <=80% logical CPU;
- <=80% process RAM;
- <=80% total-system RAM;
- maximum one heavy restoration model resident at once;
- CPU-first, no CUDA requirement;
- optional OpenVINO/iGPU only after real support + quality parity evidence.

No Linux CPU time is allowed to be relabeled as EliteBook evidence. Final target-hardware status remains FALSE/NOT_RUN.

---

## 17. RELEASE SAFETY RULES

Never weaken: SFace `0.363`; wrong-person observed contribution `0`; provenance violations `0`; frozen healthy/outside limits. Never threshold-shop, benchmark-shop, cherry-pick, remove difficult failures, relabel generated pixels as observed, use proxy similarity as SFace authority, rerun V3/V4, auto-merge, force-push certified history or fabricate tests/metrics/RAM/speed/output.

---

## 18. PROVENANCE CLASSES

Minimum canonical classes: `MAIN_OBSERVED`, `OBSERVED_REFERENCE`, `SYMMETRY_INFERRED`, `GENERATED_MODEL_INFERRED`, `UNRESOLVED`.

Generated identity-correct-looking pixels remain generated. They never become observed evidence.

---

## 19. TRACK A — PRODUCT_V1_1

Track A is preserved as operational/safety evidence, not the active architecture branch. Candidate `b6ce7ebd...` passed its historical exact-HEAD prerequisite gates, but V4 certification consumed and failed after the STARTED marker due a runner interface error before case 1. Final Track A branch state `77687b3b...` records `CONSUMED_FAIL`. No rerun/retry is permitted and PR #2 must not be presented as a certified release.

Future final certification must use a new independent V5 lineage after quality and target-PC prerequisites are green.

---

## 20. TRACK B / FINAL INTEGRATION — PAPER QUALITY

The active engineering line is now `integration/final-paper-quality-local@8bcc801e...`; research branches are evidence/source lines.

Direction: **UPSTREAM-FIRST**. When official executable code exists, use it directly at a pinned revision. CFS should not re-create GPEN/GFPGAN/CodeFormer/FBCNN/RefFace/etc. architectures merely to own the source. Compatibility patches must be minimal and documented. Every model still has to pass CFS-specific identity/provenance, broad validation, resource, Windows/offline and target-PC gates.

The current integration contains model-independent foundations plus a disabled official FBCNN adapter. FBCNN imports the official upstream network module from a detached checkout and verifies its approved checkpoint before load. It remains DEVELOPMENT/BENCHMARKING, not production-qualified.

---

## 21. CURRENT PAPER QUALITY BLOCKER

The previous DamageMaskNet attempt-3 observability blocker is resolved: evidence was recovered and the small U-Net hypothesis was stopped for model/data quality.

The current blocker is **production model identity continuity**:
1. route selection is bound to a complete production `ModelQualification` attestation — DONE/Test PASS;
2. actual generated candidate must be bound to exact official repo/revision/checkpoint evidence — NEXT;
3. no real heavy research model currently satisfies all production gates — BLOCKED by validation/license/Windows/EliteBook evidence;
4. no V5 may be created before these prerequisites are green.

---

## 22. SPECIALIST MODEL STRATEGY

`INPUT -> detect/align -> damage evidence -> reference analysis -> identity authority -> specialist route -> candidate generation -> hard identity/geometry/quality gates -> component-aware fusion -> final identity/provenance -> export`.

Use the best verified specialist per damage, not the largest model collection. FBCNN is the current JPEG challenger; NAFNet covers mild restoration; observed same-person components remain first choice for information loss; RefFace/InstantRestore and blind BFR models remain gated until real evidence justifies them.

---

## 23. MODEL SELECTION POLICY

Do not select GPEN/GFPGAN/CodeFormer/etc. from one attractive image. Compare multiple identity-disjoint DEV/VALIDATION cases by damage family with identity as hard gate, then perceptual quality, geometry, artifact rate, healthy-region preservation, PSNR/SSIM/LPIPS where applicable, RAM and runtime. Final holdouts never tune selection.

An official repository is a source-of-implementation guarantee, not a quality guarantee. The exact official revision/checkpoint used must be recorded and verified.

---

## 24. HISTORICAL RECORD

The full append-only ledger through 2026-08-24, including DEC/EXP/HIST/PUSH entries and detailed V3/V4/FBCNN/LR-ASPP/Track A evidence, is preserved byte-for-byte at:

`project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md`

Historical blob SHA: `d92bbf605609f7b1f5360009cfed4ec4a392b9a9`.

This archive is authoritative historical evidence and must not be rewritten to make the project appear cleaner. This current file maintains present state plus new append-only entries from the reconciliation onward.

---

## 25. PUSH JOURNAL — append-only from 2026-08-28

### PUSH-20260828-001

- DATE/TIME: `2026-08-28`.
- TECHNICAL BRANCH: `integration/final-paper-quality-local`.
- PREVIOUS TECHNICAL HEAD: `f5ca07e0b5268ec2b8843f9dce93b5d6a9fdf5cd`.
- TECHNICAL COMMITS: `f33880d5aa05bf576c1f36766046a4b570ec8517`, `75b31aabb1c52c77d8af6eed270b1ef5c64dcf1c`, `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a`.
- NEW REMOTE HEAD: `8bcc801e1621cbc1d1cc4f317c9cd4088444ba3a`.
- NEW TREE: `f3984f16751d648b7a9ffbd726c64ac50f3ad21b`.
- FILES MODIFIED: `app/damage_router.py`, `app/paper_quality_runtime.py`, `tests/test_paper_quality_runtime.py`.
- OBJECTIVES: OBJ-012 generated-model production authority.
- MODELS AFFECTED: none promoted/activated; contract applies to all future generated models.
- DATASETS/HOLDOUTS: none changed/accessed; V3/V4 untouched; V5 not created.
- TESTS/WORKFLOW: exact-head run `33207031788` SUCCESS; targeted `46/46`; full `602/602`; synthetic one-shot ordering PASS.
- ARTIFACT: ID `9700075188`; archive SHA-256 `8362ebb4ff8f9391256d2ea87c9b7380296ae7f4f0f4d7666e0df861afae4842`.
- RESULT: a generated route is no longer authorized by a model key/boolean alone; the runtime requires the actual production qualification and matching deterministic attestation digest.
- NEXT ACTION: bind generated candidate source repository + revision + checkpoint SHA-256 to the qualification evidence before any generated fusion/export.

### PUSH-20260828-META-001

- META BRANCH: `meta/project-state`.
- ACTION: prior ledger blob preserved byte-for-byte at `project-state-history/PROJECT_MASTER_STATE-through-2026-08-24.md` before current-state reconciliation; no technical branch/content was changed by the archive operation.

---

## 26. SESSION START/END CONTINUITY RULE

Every new session must:
1. read this file first;
2. reconcile active GitHub branch HEADs, workflows and artifacts;
3. read the historical archive only when deeper prior evidence is needed;
4. continue the current blocker rather than an obsolete chat-memory task;
5. after every meaningful technical push, obtain the exact remote HEAD and update this ledger before the next technical push.

No session may infer `PASS`, `QUALIFIED`, `RELEASE_READY`, Target95 achievement, Windows/EliteBook readiness or project completion without exact reproducible evidence.