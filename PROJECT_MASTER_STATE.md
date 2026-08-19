# Conservative Face Studio — Project Master State

> CANONICAL PROJECT LEDGER. Read this file before making project decisions in a new ChatGPT/Codex/engineering session.
>
> This ledger records verified repository state, planning definitions, measured evidence, unresolved uncertainty, important failures and the next exact engineering action. It is intentionally separate from certified production history.

## 0. Document metadata

- **Last ledger update:** 2026-08-19T05:14+02:00 (Europe/Rome)
- **Technical state verified at:** 2026-08-19
- **Repository:** `xhinoo97-svg/ConservativeFaceStudio`
- **Canonical state branch:** `meta/project-state`
- **Last technical branch:** `research/paper-quality-local-v2`
- **Last technical HEAD:** `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd`
- **Certified base:** `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- **Current active engineering tracks:** Track A — V1.1 operational stabilization; Track B — Paper Quality research
- **Current exact blockers:**
  1. Track A: same-HEAD hotfix CI is red because current identity/preflight hardening breaks legitimate same-person component-transfer / same-canvas cases in targeted regressions; Windows and Female-domain are also red on the same hotfix HEAD.
  2. Track B: DamageMaskNet mixed-source U-Net attempt 3 was triggered but its final evidence cannot currently be recovered through the available Actions interface; do not launch attempt 4 merely for observability.
- **Overall project status:** PARTIAL

| Global gate | State | Scope / evidence |
|---|---|---|
| FORENSIC_MODE_READY | TRUE | Certified PRODUCT_V1 only; does not imply V1.1 readiness. |
| PAPER_QUALITY_MODE_READY | FALSE | Research components exist; validation/Windows/target-PC gates incomplete. |
| WINDOWS_INSTALLER_READY | PARTIAL | Historical PRODUCT_V1 installer/package certification exists; current V1.1 and Paper Quality builds are not release-ready. |
| TARGET_HARDWARE_READY | FALSE | No verified real HP EliteBook 1030 G3 acceptance. |
| QUALITY_TARGET_ACHIEVED | FALSE | Development evidence is promising but no broad identity-disjoint validation establishes the final target. |
| PROJECT_FINISHED | FALSE | Unified product acceptance is incomplete. |

### Mandatory ledger update protocol

Every future technical push to an active branch must follow:

`technical work -> tests -> commit -> push -> read exact remote HEAD -> update this ledger -> commit/push ledger`

For every technical push, append a Historical Record entry containing technical branch, previous HEAD, exact new HEAD, timestamp, tests/workflows/results, affected models/objectives and next action. Do not try to record the ledger commit's own SHA inside the same commit.

---

## 1. Executive project summary

Conservative Face Studio (CFS) is a local Windows face-restoration application designed for difficult real photographs, especially low-quality smartphone/social-media images and photographs with blur, JPEG damage, pixelation, mosaics, stickers, scribbles, opaque coverage or missing facial detail.

Its defining architectural distinction is that **evidence-faithful restoration and generative restoration are separate authorities**:

- **Conservative / forensic mode** prioritizes original MAIN pixels and verified same-person reference pixels, tracks provenance, rejects wrong-person contribution and abstains when evidence is inadequate.
- **Paper Quality mode** may use modern learned facial priors to improve perceptual quality, but every generated pixel remains `GENERATED_MODEL_INFERRED`; generation may never be represented as observed evidence.

PRODUCT_V1 is a certified conservative baseline. PRODUCT_V1.1 is an operational hotfix that is currently blocked by real regressions and must not import experimental Paper Quality models to hide them. The advanced Paper Quality research track has real CPU development evidence for GPEN, GFPGAN v1.4, CodeFormer and FBCNN, plus implemented research infrastructure for resource control, damage routing, personalized references, reference-first repair, candidate selection and component-aware fusion.

The biggest technical limitation is that the advanced research architecture is not yet qualified end-to-end on Windows/EliteBook and DamageMaskNet attempt 3 evidence remains unresolved. The biggest quality limitation is insufficient broad identity-disjoint validation to choose per-damage winners and calibrate the candidate selector without cherry-picking.

**Current next milestone:** restore auditable state first; then stabilize Track A regressions independently and recover DamageMaskNet attempt 3 evidence before executing RefFaceInpainting attempt 1/3.

---

## 2. Branch and release map

| Branch | Purpose | Base | Current verified HEAD | Status | Last meaningful change | CI state | Merge state | Superseded by | Next gate |
|---|---|---|---|---|---|---|---|---|---|
| `main` | Certified PRODUCT_V1 | historical | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | FROZEN / RELEASED | Signed merge of PR #1; commit records Windows #1195, Female #463, Release Quality #13 certification | Historical certified green | merged | none | Do not modify/rewrite. |
| `feature/block-pipeline-v1` | Original V1 implementation branch | pre-V1 | `5eff667373cd47c07ba14aaad2acafee6d5a61c1` | MERGED / SUPERSEDED | Block pipeline implementation | historical | merged into main as one parent | `main` | Archive as history. |
| `release/v1-certified` | Certified V1 candidate | V1 feature branch | `f476c6f04b57b658fd152a0a82e5b50cb5afbdbc` | FROZEN / ARCHIVED | Candidate that passed V1 certification before merge | historical certified green | merged via PR #1 | `main` | Preserve. |
| `hotfix/real-world-restoration-v1.1` | Track A operational/safety hotfix | PRODUCT_V1 | `3645c8c39653d04616167e881adaf28d2b93cd45` | BLOCKED / ACTIVE | Identity/preflight, V4 protocol, Windows/female qualification hardening | Release Quality FAIL; Windows FAIL; Female-domain FAIL on exact HEAD | PR #2 OPEN, DRAFT, NOT MERGED | none | Fix inherited identity/preflight regressions without weakening thresholds; rerun same-HEAD gates. |
| `research/face-restoration-v2` | Early degradation/dataset research | `main` | `757a3f6081b7b152cdc615a07cd99aec40fa0a1c` | SUPERSEDED AS ACTIVE ARCHITECTURE | Early degradation generator and dataset specification | NOT_VERIFIED | not merged | `research/paper-quality-local-v2` as active direction, but **not** a Git superset | Preserve/port useful dataset assets explicitly if needed. |
| `research/paper-quality-local-v2` | Advanced Track B Paper Quality research | `main` | `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd` | ACTIVE / BENCHMARKING | Resource governor, BFR/JPEG evidence, DamageMaskNet/reference/fusion research, RefFace preparation, truth-state report | latest research push-run state partly NOT_VERIFIED | not merged | none | Recover DamageMaskNet attempt 3 evidence. |
| `meta/project-state` | Canonical project state / technical archive | certified `main` | self-referential ledger SHA intentionally not recorded here | ACTIVE META | Canonical ledger creation | documentation-only | not for product merge | none | Update after every technical push. |

### Research branch reconciliation

`research/face-restoration-v2` and `research/paper-quality-local-v2` diverged from the same certified base. The advanced branch does **not** literally contain the two commits from the early branch. Therefore the early branch is superseded only as the **active architecture**, not falsely described as merged. Its dataset/degradation concepts remain historical material that may be explicitly ported after review.

---

## 3. PRODUCT VERSION ROADMAP

Product version labels are planning/release definitions. They are separate from dataset/holdout labels.

### PRODUCT_V1 — Certified Conservative Baseline

- **Status:** RELEASED
- **Objective:** evidence-first conservative/forensic restoration with auditable provenance.
- **User-visible purpose:** safely restore damaged faces without representing generated guesses as observed identity evidence.
- **Base:** `main@2767513f95dde2d417e7c6f1faf2357149a1a32f`
- **Architecture:** 13-block deterministic/evidence-first pipeline.
- **Core models:** YuNet, SFace, NAFNet, face-parsing ResNet18 ONNX, head-pose MobileNetV2 ONNX, LaMa ONNX where policy permits.
- **Blocks modified:** baseline all 13.
- **Datasets:** V1 calibration/final/female benchmark lineage.
- **Supported photo types:** baseline blur/occlusion/reference workflows according to V1 release contract.
- **Known limitations:** perceptual restoration quality below modern generative BFR on severe information loss; real-world robustness motivated V1.1.
- **Safety guarantees:** frozen SFace `0.363`; wrong-person observed contribution `0`; provenance violations `0`; healthy-region policy preserved.
- **Resource target:** CPU/local/offline production baseline.
- **Windows status:** historically certified.
- **EliteBook status:** NOT_VERIFIED as a specific real-machine acceptance campaign.
- **Completed:** release certification and merge.
- **Blocker:** none for historical release; immutable baseline.
- **Next gate:** none; preserve as regression reference.

### PRODUCT_V1_1 — Operational Real-World Hotfix

- **Status:** IMPLEMENTING / BLOCKED
- **Objective:** improve runtime robustness and real-world restoration safety without weakening V1 evidence philosophy.
- **User-visible purpose:** safer/reliable operation on difficult phone/reference cases.
- **Base/branch:** `hotfix/real-world-restoration-v1.1` from PRODUCT_V1.
- **Architecture:** same 13-block conservative architecture with identity/preflight/provenance hardening.
- **Models:** same production model family; no Track B generative rescue.
- **Blocks most affected:** preflight/identity, reference eligibility, regional repair/fusion, release protocol.
- **Datasets:** inherited calibration, female-domain, consumed V3 history, frozen V4 protocol.
- **Known limitations:** current exact HEAD has targeted identity regressions and three red release workflows.
- **Safety guarantees:** SFace remains `0.363`; wrong-person pixels `0`; provenance violations `0`; no V3 tuning; no early V4.
- **Windows status:** FAIL on current HEAD.
- **EliteBook status:** NOT_RUN.
- **Current work:** repair legitimate same-person transfer/bridge semantics while remaining fail-closed for wrong-person/transitive identity.
- **Next gate:** targeted regressions -> full pytest -> same-HEAD Windows/Female/Release Quality -> only then candidate/final protocol.

### PRODUCT_V2 — Paper Quality Local

- **Status:** BENCHMARKING
- **Objective:** modern local CPU-first perceptual restoration with damage-aware specialist routing and explicit generated provenance.
- **User-visible purpose:** substantially better-looking restoration when Conservative Mode cannot recover missing information.
- **Base/branch:** `research/paper-quality-local-v2` from certified PRODUCT_V1.
- **Architecture:** common candidate adapter, damage routing, hard identity gates, deterministic evidence-aware fusion, 80% resource governor.
- **Models evaluated:** GPEN BFR-512, GFPGAN v1.4, CodeFormer, FBCNN, existing NAFNet; DamageMaskNet under development.
- **Blocks modified/prototyped:** 2, 3, 6, 8, 9, 11, 13 plus shared resource/runtime infrastructure.
- **Datasets:** DEVELOPMENT research source bank; DamageMaskNet FairFace + ControlFace source bank; no final-holdout tuning.
- **Supported photos:** blur, mixed degradation, JPEG, severe blind facial degradation, information-loss cases with generated fallback.
- **Known limitations:** no broad validation/model winner; licensing blockers for some research models; Windows/EliteBook not qualified.
- **Safety:** generated pixels tagged `GENERATED_MODEL_INFERRED`; SFace hard gate unchanged.
- **Resource target:** <=80% logical CPU, <=80% process/system RAM, one heavy model at a time.
- **Next gate:** resolve DamageMaskNet attempt 3; then broaden identity-disjoint BFR/JPEG validation and calibrate selector on DEV/VALIDATION only.

### PRODUCT_V3 — Personalized Multi-Reference Restoration

- **Status:** PLANNED with enabling research prototypes
- **Objective:** MAIN + 0–9 same-person references as structured identity/component guidance, not one whole-face donor.
- **User-visible purpose:** use the best observed eye/nose/mouth/etc. from different valid photos of the same person.
- **Base:** future validated PRODUCT_V2 architecture.
- **Architecture:** `PersonIdentityProfile`, component bank, per-component quality/coverage, full-vs-partial identity authority, robust consensus embedding.
- **Models:** SFace identity; reference-conditioned specialist candidates only after qualification.
- **Blocks targeted:** 7, 8, 9, 11, 13.
- **Known limitations:** prototypes are not a released PRODUCT_V3; broad reference-bank validation is pending.
- **Safety:** partial references component-local only; wrong-person never global anchor/donor/identity booster.
- **Next gate:** validate reference bank and reference-first reconstruction over identity-disjoint multi-reference cases.

### PRODUCT_V4 — Damage-Specialist Hybrid Architecture

- **Status:** PLANNED with enabling research prototypes
- **Objective:** route each corruption family to the most specialized acceptable engine and fuse at component level.
- **Architecture:** DamageMaskNet/detector -> specialist route -> component candidate scoring -> deterministic fusion.
- **Models considered:** FBCNN, NAFNet, GPEN/GFPGAN/CodeFormer, RefFaceInpainting, possible future qualified lightweight specialists.
- **Blocks targeted:** 2, 3, 6, 7, 8, 9, 11, 12.
- **Known limitations:** DamageMaskNet not yet verified; RefFace prepared but NOT_RUN; other specialists not qualified.
- **Next gate:** finish PRODUCT_V2/V3 evidence and specialist qualification.

### PRODUCT_V5 — Unified Final Product

- **Status:** PLANNED
- **Objective:** stable Conservative Mode + Paper Quality Mode + personalized references + specialist routing + offline Windows model pack + clean installer + target-PC acceptance.
- **User-visible purpose:** one production application choosing the best verified local path for each person/damage while preserving evidence semantics.
- **Acceptance:** all product gates, clean Windows, offline package, model hashes/licenses, real HP EliteBook acceptance, no known release defects.

---

## 4. HOLDOUT / BENCHMARK LINEAGE

| Evaluation set | Purpose | Identity/case state | Source/legal state | Split/freeze | Executed | Consumed | Tuning allowed? | Relevant result/state |
|---|---|---|---|---|---|---|---|---|
| CALIBRATION_V1 | V1 safety calibration | 60 cases historically reported | frozen project manifests | historical | YES | certification evidence | only according to original calibration protocol | historical `60/60` at certified candidate. |
| FINAL_HOLDOUT_V1 | PRODUCT_V1 certification | 40 cases historically reported | frozen project manifests | final | YES | YES for certification semantics | NO after certification | historical `40/40` at certified candidate. |
| FINAL_HOLDOUT_V2 | historical project lineage | exact current manifest/result details not re-reconciled in this ledger pass | NOT_VERIFIED | historical | NOT_VERIFIED | NOT_VERIFIED | NO unless protocol explicitly says otherwise | recover before making claims. |
| FINAL_HOLDOUT_V3 | V1.1 historical final evaluation | 40 cases | frozen | final | YES | **YES** | **NO** | `39/40`; failure `cfsfs3-fin-020-medium_block_mosaic`, SFace `0.360 < 0.363`. Never rerun for tuning/certification. |
| FINAL_HOLDOUT_V4 | independent V1.1 final holdout | 40 cases, 20 identities; 19 female-domain + 1 control | ControlFace10K CC BY 4.0 source pinned; no V1/V2/V3 collision recorded in freeze | frozen before candidate change | **NO** | **NO** | **NO before execution; never tune on it** | `freeze.json` exists; `CONSUMED.json` absent; certification request absent. SFace `0.363`, outside MAE `8.0`, wrong-person `0`. |
| FINAL_HOLDOUT_V5 | future independent final set | not created | not created | future | NO | NO | NO | PLANNED only. |
| Female-domain benchmark | real-domain safety/quality stress | current quick profile designed for 300–400 cases | curated/project sources | validation/stress | multiple historical runs | no final-holdout semantics | report-only quality metrics; safety gates apply | current hotfix run benchmark execution reached completion but lightweight report validation failed; exact nested assertion NOT_VERIFIED here. |
| Paper Quality DEV | model comparison / calibration development | identity-disjoint expansion required | research source bank | DEVELOPMENT | partial | NO | YES | GPEN/GFPGAN/CodeFormer/FBCNN evidence exists, currently too small for production winner selection. |
| Paper Quality VALIDATION | independent model/router validation | to be expanded | open-source licensed banks | VALIDATION | partial/not comprehensive | NO | selector thresholds/weights frozen after this stage | broad validation incomplete. |
| DamageMaskNet research bank | exact synthetic damage masks on face sources | mixed FairFace real + ControlFace multi-view/identity-disjoint validation | source/license recorded in research bank | TRAIN/VALIDATION | attempt 3 triggered | NO | TRAIN/DEV only | final attempt-3 evidence currently NOT_VERIFIED. |

### Holdout invariants

- Consumed holdouts are never reused for tuning.
- HOLDOUT_V3 remains consumed even though it failed 1/40.
- HOLDOUT_V4 must not be executed until the correct same-candidate pre-final sequence authorizes one-shot execution.
- Never rename or mutate an old holdout to manufacture independence.

---

## 5. CURRENT GLOBAL OBJECTIVES

### OBJ-001 — Preserve certified PRODUCT_V1
- **VERSION:** PRODUCT_V1
- **TRACK:** Release baseline
- **STATUS:** PASS
- **WHY:** provide immutable safe regression baseline.
- **SUCCESS:** `main` remains at certified history unless an explicit future release is merged through normal review.
- **EVIDENCE:** signed merge `2767513f...`; historical certification runs recorded in commit.
- **BLOCKER:** none.
- **NEXT:** never force-push/rewrite.
- **LAST UPDATED:** 2026-08-19.

### OBJ-002 — Restore PRODUCT_V1_1 operational gates
- **VERSION:** PRODUCT_V1_1
- **TRACK:** A
- **STATUS:** BLOCKED
- **WHY:** current hotfix must work in real cases without weakening safety.
- **SUCCESS:** targeted tests + full pytest + same-HEAD Windows + Female-domain + Release Quality pass, wrong-person/provenance remain zero.
- **EVIDENCE:** current HEAD `3645c8c...`; Release Quality targeted suite `4 failed, 195 passed`.
- **BLOCKER:** legitimate same-person transfer/identity bridge is rejected by current preflight/hardening semantics.
- **NEXT:** inspect current hotfix identity/preflight code and fix root cause without changing SFace `0.363`.
- **LAST UPDATED:** 2026-08-19.

### OBJ-003 — Maintain canonical project ledger
- **VERSION:** all
- **TRACK:** meta
- **STATUS:** IN_PROGRESS
- **WHY:** eliminate dependence on chat memory.
- **SUCCESS:** this file updated after every technical push with exact remote SHA/evidence.
- **BLOCKER:** none.
- **NEXT:** append ledger entry immediately after next technical push.
- **LAST UPDATED:** 2026-08-19.

### OBJ-004 — Resolve DamageMaskNet hypothesis
- **VERSION:** PRODUCT_V2/PRODUCT_V4 enabling research
- **TRACK:** B
- **STATUS:** BLOCKED
- **WHY:** damage-aware routing requires reliable localized damage classification.
- **SUCCESS:** recover attempt-3 result; if PASS, per-class IoU/F1 + ONNX parity + RAM/runtime; if model/data FAIL, stop U-Net hypothesis after 3 attempts.
- **EVIDENCE:** attempts 1/2 infrastructure fail; attempt 3 mixed-source triggered.
- **BLOCKER:** final attempt-3 evidence not observable in current interface.
- **NEXT:** recover existing evidence without rerunning/tuning.
- **LAST UPDATED:** 2026-08-19.

### OBJ-005 — Select blind face-restoration candidates using broad evidence
- **VERSION:** PRODUCT_V2
- **TRACK:** B
- **STATUS:** IN_PROGRESS
- **SUCCESS:** identity-disjoint DEV/VALIDATION comparison of GPEN/GFPGAN/CodeFormer by identity, quality, geometry, artifacts, runtime/RAM.
- **CURRENT EVIDENCE:** one comparable development case plus real CPU execution.
- **BLOCKER:** insufficient sample size; licensing for some candidates.
- **NEXT:** multi-image benchmark after immediate blocker resolution.

### OBJ-006 — Qualify JPEG specialist
- **VERSION:** PRODUCT_V2/4
- **TRACK:** B
- **STATUS:** IN_PROGRESS
- **SUCCESS:** FBCNN gains on JPEG QF ranges, double JPEG, social-media recompression, resize+JPEG, JPEG+blur without identity regression; Windows/EliteBook acceptance.
- **EVIDENCE:** QF20 DEV improvement in PSNR/SSIM/SFace.
- **NEXT:** broaden JPEG validation.

### OBJ-007 — Validate Personalized Reference Bank
- **VERSION:** PRODUCT_V3
- **TRACK:** B
- **STATUS:** IN_PROGRESS
- **SUCCESS:** 0/1/9 refs, full/partial/wrong/duplicate/low-quality/multi-pose tests with per-component improvement and exact provenance.
- **EVIDENCE:** framework implemented; latest workflow state NOT_VERIFIED.
- **NEXT:** broad validation after core damage gate.

### OBJ-008 — Measure RefFaceInpainting CPU feasibility
- **VERSION:** PRODUCT_V3/4
- **TRACK:** B
- **STATUS:** BLOCKED
- **SUCCESS:** real same-identity reference inpainting output under 80% budget, identity gate, exact healthy outside mask, hashes/runtime/RAM.
- **EVIDENCE:** manual-only vertical slice prepared; official MIT source pinned; core CUDA allocation patch isolated.
- **BLOCKER:** sequencing requires DamageMaskNet gate resolution.
- **NEXT:** execute attempt 1/3 only after OBJ-004 resolves.

### OBJ-009 — Build Paper Quality Windows model pack/installer
- **VERSION:** PRODUCT_V2→V5
- **TRACK:** B/release
- **STATUS:** PROPOSED
- **SUCCESS:** offline verified model pack, hashes/licenses, no dev Python, clean-machine package execution.
- **NEXT:** only after model/router qualification.

### OBJ-010 — Real HP EliteBook 1030 G3 acceptance
- **VERSION:** PRODUCT_V5
- **TRACK:** release
- **STATUS:** PROPOSED
- **SUCCESS:** real CPU/RAM/Windows/backend measurements on blur, JPEG, mosaic, scribbled eye, sticker mouth, low-light, multi-reference; no OOM/crash/network/CUDA/dev tools.
- **NEXT:** after same-candidate Windows installer is ready.

---

## 6. MODEL MASTER REGISTRY

`IMPLEMENTED != TESTED != BENCHMARKED != QUALIFIED != RELEASED`.

Where a digest/license/runtime field has not been re-read in this reconciliation, it is explicitly left `NOT_VERIFIED` or points to the authoritative repository registry/evidence rather than being guessed.

| Model | Project role / damage | Upstream / checkpoint | License / redistribution | Input | CPU / Windows / backend | Measured evidence | Current state | Why selected / not selected | Version / blocks | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| YuNet | face detection / landmarks source | OpenCV Zoo; exact URL/hash in `app/model_registry.py` | registry-pinned; re-read manifest for exact terms before redistribution change | detector native | CPU OpenCV DNN; production-used | historical V1 | QUALIFIED for PRODUCT_V1 | lightweight deterministic detector | V1+, Blocks 4/11 support | `app/model_registry.py`, V1 release evidence |
| SFace | identity hard gate | OpenCV Zoo; exact URL/hash in registry | registry-pinned | aligned face | CPU OpenCV FaceRecognizerSF | threshold frozen `0.363` | QUALIFIED for PRODUCT_V1 | safety identity backend | V1+, Block 11 | registry/tests/release evidence |
| NAFNet | mild deblur / denoise pre-clean | project ONNX registry | exact hash in registry | tiled/general image | CPU/OpenCV DNN | production V1 evidence; Paper Quality challenger role | QUALIFIED for current conservative role | lightweight, not final facial-prior generator | V1+, Blocks 2/3 | registry/release tests |
| Face Parsing ResNet18 ONNX | 19-class facial parsing | `yakhyo/face-parsing`; model hash registry-pinned | MIT code; asset terms follow upstream/registry | 512 | ONNX Runtime CPU | production active; Paper Quality adapter prepared | QUALIFIED for V1 parsing role | enables component/RefFace maps | Blocks 6/7/8 | registry + parser adapter/tests |
| Head Pose MobileNetV2 ONNX | pose geometry | registry-pinned | exact terms/hash in registry | face crop | CPU ONNX | production V1 evidence | QUALIFIED for current role | geometry, not generation | Blocks 4/10 | registry/tests |
| LaMa ONNX | residual non-identity-critical inpainting | registry-pinned | exact terms/hash in registry | masked image | CPU ONNX | V1 constrained use | QUALIFIED only for policy-limited role | must never become evidence for identity-critical facial structure | Block 8 | registry/policy/tests |
| Real-ESRGAN | optional Paper Quality upscale | official project; exact checkpoint not selected | NOT_VERIFIED for chosen production weight | variable/x2 candidate | CPU likely expensive; Windows NOT_RUN | no CFS qualification | DISCOVERED / FEASIBILITY_ONLY | optional, not default CPU background SR | PRODUCT_V2/4, Block 12 | model matrix |
| GPEN BFR-512 | fast blind face restoration | `yangxy/GPEN`, BFR-512; source commit/checkpoint recorded in research report | redistribution/license clarity unresolved | 512 aligned | real Linux CPU works; Windows NOT_RUN | SFace `0.95397`; `2.697s`; peak RSS `~1.828GB`; PSNR `28.07`; SSIM `0.7474` on one DEV case | BENCHMARKING / BLOCKED_LICENSE for distribution decision | strong identity on measured case; generative texture can diverge pixel-wise | PRODUCT_V2, Blocks 2/3/8 | `research/GPEN_VERTICAL_SLICE_REPORT.md` |
| GFPGAN v1.4 | blind face restoration | official GFPGAN v1.4 release asset | exact redistribution terms must be preserved/reverified before product pack | 512 aligned | real Linux CPU works; Windows NOT_RUN | SFace `0.91665`; `2.787s`; peak RSS `~1.666GB`; PSNR `30.65`; SSIM `0.8604` one DEV case | BENCHMARKING | more conservative/pixel-close than GPEN on first comparable case; not enough evidence to choose winner | PRODUCT_V2, Blocks 2/3/8 | `research/GFPGAN_V1_4_VERTICAL_SLICE_REPORT.md` |
| CodeFormer | severe restoration / controllable fidelity / candidate inpainting | official CodeFormer, `w=0.5` development slice | S-Lab terms block unrestricted commercial redistribution/use unless compatible authorization exists | 512 aligned | Linux CPU slice PASS; Windows NOT_RUN | exact comparative metrics must be read from stored artifact; do not reconstruct from memory | BENCHMARKING / BLOCKED_LICENSE | powerful quality/fidelity control; license and CPU cost matter | PRODUCT_V2, Blocks 2/3/8 | CodeFormer research workflow/artifact |
| FBCNN | JPEG/double-JPEG specialist | official FBCNN research source/checkpoint | exact redistribution terms to reverify before pack | image/JPEG path | real Linux CPU works; Windows NOT_RUN | QF20: PSNR `34.62→36.78`; SSIM `0.9486→0.9634`; SFace `0.9571→0.9691`; peak RSS `~1.305GB` | BENCHMARKING / current JPEG leader | specialist gain without unnecessary face generation | PRODUCT_V2/4, Block 3 | `research/PHASE7_FBCNN_RESULT.md` |
| DamageMaskNet small U-Net | localized damage segmentation | CFS-trained research model | CFS code; source images retain their licenses | research 256/ONNX target | CPU/ONNX design; final attempt evidence NOT_VERIFIED | per-class IoU/F1 NOT_VERIFIED | BENCHMARKING / BLOCKED | exact synthetic masks; hypothesis stops after 3 model attempts if quality failure confirmed | PRODUCT_V2/4, Block 6 | DamageMaskNet research scripts/status report |
| RefFaceInpainting | same-person reference-guided large occlusion | `WuyangLuo/RefFaceInpainting@0f1ad756...`; official generator + ArcFace links | MIT repository; observed checkpoint hashes must be recorded on run | 256 | upstream CUDA-hardcoded; CFS CPU minimal path prepared, NOT_RUN | none yet | FEASIBILITY_ONLY / NOT_RUN | highly specialized to sticker/black-bar/missing-region with reference | PRODUCT_V3/4, Block 8 | manual research workflow + status doc |
| InstantRestore | personalized multi-reference restoration | official research repo/checkpoints | repository licensing unresolved in current audit | diffusion/SD-style | official implementation CUDA/FP16, 2 UNets + 2 VAEs + CLIP | no CPU CFS run | FEASIBILITY_ONLY / BLOCKED_HARDWARE + license audit | scientifically close to MAIN+multi-ref use case but likely high resource cost | future V3 challenger | `research/INSTANTRESTORE_FEASIBILITY.md` |
| OSDFace | one-step severe blind restoration challenger | official research implementation | exact redistribution terms NOT_VERIFIED | model-specific | official inference uses CUDA streams/device assumptions | no CPU CFS run | FEASIBILITY_ONLY / BLOCKED_HARDWARE | modern severe blind challenger; port cost must justify gain | future V2/4 challenger | model matrix/status doc |
| RestoreFormer++ | severe blind restoration challenger | official project | NOT_VERIFIED in this pass | model-specific | CPU/Windows NOT_RUN | none in CFS | DISCOVERED / FEASIBILITY_ONLY | benchmark only after current specialist sequence | future V2/4 | model matrix |
| VQFR | blind face restoration challenger | official project | NOT_VERIFIED | model-specific | CPU/Windows NOT_RUN | none in CFS | DISCOVERED / FEASIBILITY_ONLY | only promote if quality/resource gain justifies cost | future V2/4 | model matrix |
| GPEN face inpainting | generative face inpainting challenger | GPEN official | same GPEN licensing concern | typically 1024 variant/path | CPU cost NOT_VERIFIED | none | FEASIBILITY_ONLY | potential missing-region specialist after RefFace | V4, Block 8 | model matrix |
| RefineFIR | reference-guided architecture teacher | public paper/repo | executable checkpoint path currently inadequate | research | NOT_RUN | none | FEASIBILITY_ONLY | copy-or-not concept useful; runtime not benchmarkable now | V3 teacher | status doc |
| PerFuSe | personalized restoration teacher/challenger | research paper/project | executable official runtime NOT_VERIFIED | research | NOT_RUN | none | DISCOVERED / FEASIBILITY_ONLY | smartphone/photo-library personalization concept | future V3 | status/model research |
| RefIPFR | personalized/reference restoration teacher | research | executable official runtime NOT_VERIFIED | research | NOT_RUN | none | DISCOVERED / FEASIBILITY_ONLY | reference-conditioned design ideas | future V3 | status/model research |

### Registry integrity note

`THIRD_PARTY_MODULES.md` on certified `main` refers to machine-readable files under `models/`, but the reconciled `main/models/` directory contains only `README.md`. The actual active production model catalog/download validation is implemented in `app/model_registry.py` and companion production/runtime registry code. This documentation mismatch must be corrected in a future documentation-only technical change; do not invent missing manifest files.

---

## 7. CURRENT MODEL EVIDENCE

### DEVELOPMENT — measured, not production qualification

| Model | Case / scope | Identity | PSNR | SSIM | Runtime | Peak RSS | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| GPEN BFR-512 | one comparable aligned DEV face | SFace `0.95397` | `28.07` | `0.7474` | `~2.697s` @512 Linux CPU | `~1.828GB` | strong identity / sharper generative texture; not a production winner from one image. |
| GFPGAN v1.4 | same comparable aligned DEV face | SFace `0.91665` | `30.65` | `0.8604` | `~2.787s` Linux CPU | `~1.666GB` | closer pixel/structure on this case; not enough evidence for winner. |
| FBCNN | JPEG QF20 DEV | SFace `0.9571→0.9691` | `34.62→36.78` | `0.9486→0.9634` | see evidence artifact | `~1.305GB` | current measured JPEG leader; broaden before qualification. |
| CodeFormer w=0.5 | aligned DEV slice | real SFace gate PASS | exact artifact required | exact artifact required | exact artifact required | exact artifact required | do not restate guessed metrics. |

No Linux DEV number above is a Windows or EliteBook result.

---

## 8. 13-BLOCK ARCHITECTURE

| # | Block | Current certified function/model | Current authority / provenance | Main current limitation | Proposed / research change | Version target | Implementation / test / benchmark state | Resource / dependency | Next action |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | IMPORT | deterministic OpenCV/Pillow import; MAIN + refs | input bytes are observed | input quality varies | retain deterministic import; richer source/hash metadata | V2/V5 | baseline RELEASED | low | no generator here. |
| 2 | DEBLUR | NAFNet mild deblur under conservative policy | generated/model output cannot become observed evidence | severe facial blur may need facial prior | route from common checkpoint to NAFNet or accepted GPEN/GFPGAN/CodeFormer candidate, never blind chain | V2/V4 | candidate infrastructure BENCHMARKING | one heavy model at a time | broaden blind-restorer benchmark. |
| 3 | ENHANCE | NAFNet/general conservative enhancement | healthy MAIN has priority | no specialist JPEG route in V1 | FBCNN for severe JPEG; Zero-DCE++ only if low-light detector proves need | V2/V4 | FBCNN BENCHMARKING; low-light specialist not qualified | <=80% | expand JPEG families. |
| 4 | LANDMARKS | YuNet/production face analysis; pose model support | measured geometry only | hard pose/partial damage can reduce confidence | optional MediaPipe/3DDFA only if benchmarked; never invent landmarks | V2/4 | current V1 qualified | low/moderate | preserve deterministic geometry. |
| 5 | ALIGN | deterministic similarity/partial affine/RANSAC | no identity synthesis | limited by landmarks | retain; common aligned checkpoint for candidate A/B | V2 | implemented in research slices | low | standardize benchmark alignment. |
| 6 | OCCLUSION_MASK | face parsing + existing occlusion heuristics | mask describes repair authority, not identity evidence | weak explicit classification of mosaic/JPEG/etc. | DamageMaskNet 12-class localized segmentation + exact confidence/component mapping | V2/V4 | IMPLEMENTED research; training result NOT_VERIFIED | lightweight ONNX target | recover attempt-3 evidence. |
| 7 | REGION_SELECT | `component_bank`, `reference_memory`, observed geometry/reliability | only same-person valid observed reference can donate; exact source index | legacy whole-reference logic insufficient for best-per-component personalization | Personalized Reference Bank; rank 13 components independently; robust full-ref consensus | V3 | IMPLEMENTED research; workflow NOT_VERIFIED | low/moderate | validate 0/1/9, partial/wrong/low-quality cases. |
| 8 | INPAINT | observed-reference repair first; constrained LaMa for suitable residuals | observed same-person > inferred; identity-critical LaMa prohibited as evidence | no qualified high-quality generative missing-detail specialist | reference-first wrapper; RefFace/CodeFormer/GPEN candidates only in Paper mode, GENERATED provenance | V2/3/4 | reference-first IMPLEMENTED; RefFace PREPARED NOT_RUN | potentially heavy | resolve DamageMaskNet then RefFace attempt 1/3. |
| 9 | FUSION | deterministic/regional fusion | MAIN/observed reference authority dominates | whole-face candidates can hide component tradeoffs | deterministic component-aware fusion: healthy MAIN > observed ref > accepted generated within authority | V2/4 | IMPLEMENTED research; latest workflow NOT_VERIFIED | low | validate seams/geometry/provenance. |
| 10 | FRONTALIZE | geometry-only conservative frontalization | no hidden-side synthesis in conservative mode | hidden regions remain unresolved | Paper mode may synthesize hidden region only as GENERATED | V4 | planning | moderate | keep modes separate. |
| 11 | IDENTITY_CHECK | SFace hard gate `0.363`; conservative reference/identity policies | wrong-person no anchor/donor; partial local only | Track A current hardening rejects some legitimate same-person cases | robust accepted-full-reference aggregation; optional second backend only after benchmark | V1.1/V3 | Track A BLOCKED; research framework implemented | low/moderate | fix current preflight/bridge regressions without lowering 0.363. |
| 12 | UPSCALE | Lanczos conservative | deterministic | limited perceptual SR | optional Real-ESRGAN x2 only if CPU/identity/background benchmark justifies | V2/4 | FEASIBILITY_ONLY | potentially heavy | defer until restoration winners. |
| 13 | EXPORT | deterministic final image + project evidence | exact provenance mandatory | research needs richer generated/source/timing reports | export generated mask, per-component source, model-selection, identity, damage, timings, RAM, hashes | V2/V5 | partially implemented research | low | unify after router qualification. |

---

## 9. PHOTO AND INPUT CONTRACT

### MAIN photo

CFS is intended to handle, independently or in combination:

- low resolution; smartphone compression; old social-media downloads;
- JPEG artifacts; double JPEG; recompression; resize+JPEG;
- defocus blur; motion blur; mixed blur; noise;
- pixelation; block mosaic;
- scribbles; stickers; black bars; opaque masks/blocks;
- partially/fully covered eye; covered mouth; covered nose; missing facial component;
- crop; partial face; low light; uneven exposure;
- mixed/unknown real-world degradation.

MAIN defines the target canvas/pose/frame in conservative policy. A better-looking reference may be a donor/analysis anchor but may not silently replace MAIN geometry.

### REFERENCE photos

Input contract: MAIN + **0–9 references**.

References may be full face, partial face, eye-only, mouth-only, nose-only, side angle, different expression/lighting/resolution, blurred/compressed/occluded, unusable or wrong-person distractors.

Identity eligibility is separate from image quality:

- accepted **FULL same-person** reference -> may become a global identity anchor and component donor according to geometry/quality;
- accepted **PARTIAL same-person** reference -> component-local authority only, never global identity anchor;
- **wrong-person** reference -> never global anchor, never observed-pixel donor, never allowed to improve final identity score.

### GROUND TRUTH / RESEARCH photos

Research clean images must have source/license/provenance and file hash. Synthetic degradation must store seed, parameters and exact mask where applicable. Identity partitions must be explicit. Final holdouts are excluded from training/tuning.

---

## 10. DATASET CONSTRUCTION

### Large Paper Quality bank target

- target scale: approximately **300–400 identity/case examples** initially, expanded when statistical evidence requires;
- target domain: real intended face-restoration usage, with explicit female-domain proportion rather than silent bias;
- partition by identity: TRAIN / DEVELOPMENT / VALIDATION / FINAL_HOLDOUT;
- forbidden identity leakage across protected partitions.

For every source/example record:

`source, license/usage basis, download date, identity_id, original hash, resolution, domain/gender label if intentionally used, split, degradation family, severity, seed, exact mask, reference relationships, generated derivative hashes`.

### Current research banks

- `research/face-restoration-v2`: early dataset/degradation specification (~400-case planning, historical 50/50 domain concept); branch superseded as active architecture but assets preserved.
- DamageMaskNet current bank: FairFace real faces + ControlFace10K multi-view identities, synthetic exact masks, ControlFace identity-disjoint validation.
- V4 holdout is **not** part of Paper Quality training/validation.

---

## 11. COMPONENT-BY-COMPONENT RECONSTRUCTION

Canonical components:

1. LEFT_EYE
2. RIGHT_EYE
3. LEFT_EYEBROW
4. RIGHT_EYEBROW
5. NOSE
6. PHILTRUM
7. MOUTH_LIPS
8. LEFT_CHEEK
9. RIGHT_CHEEK
10. CHIN
11. JAW
12. FOREHEAD
13. FACE_CONTOUR

For each component, the runtime/profile must be able to record:

`MAIN visibility, damage class/confidence, best observed reference, alternates, reference confidence, generated candidates, selected source, provenance, identity consistency, geometry consistency, unresolved state`.

Authority invariant: observed same-person information outranks generated inference. A failed high-priority semantic component is not silently filled by a broad cheek/jaw donor.

---

## 12. DAMAGE ROUTING

| Damage | Detector | Primary route | Secondary route | Reference-first? | Generative allowed in Paper mode? | Specialist candidate | Fallback / abstention |
|---|---|---|---|---|---|---|---|
| HEALTHY | DamageMaskNet/heuristics | preserve MAIN | none | N/A | NO by default | none | unchanged MAIN |
| DEFOCUS_BLUR | damage detector/blur metric | NAFNet or validated deblur | blind face candidate if severe | if observed component is better | YES | NAFNet + BFR challenger | conservative unresolved/rollback if gates fail |
| MOTION_BLUR | detector/blur metric | NAFNet/deblur candidate | blind face candidate if severe | yes where useful | YES | NAFNet/BFR | rollback/abstain |
| NOISE | noise metric | NAFNet-SIDD/current winner | specialist after benchmark | optional | YES only if necessary | NAFNet | preserve healthy detail |
| JPEG | JPEG detector | FBCNN | then face candidate only if residual facial damage warrants | optional | YES after specialist | **FBCNN** current leader | no unnecessary heavy model |
| DOUBLE_JPEG | JPEG detector | FBCNN candidate | BFR only if needed | optional | YES | FBCNN | validation pending |
| PIXELATION | DamageMaskNet | observed component reconstruction | generated candidate | YES | YES | reference bank + BFR | unresolved/abstain Conservative |
| BLOCK_MOSAIC | DamageMaskNet | observed component reconstruction | generated face/component candidate | YES | YES | reference bank / future specialist | unresolved if no evidence Conservative |
| SCRIBBLE | DamageMaskNet exact/learned mask | observed reference repair | RefFace/generative component | YES | YES | RefFace if qualified | conservative unresolved |
| STICKER | DamageMaskNet | observed reference repair | RefFace/generative component | YES | YES | RefFace if qualified | conservative unresolved |
| OPAQUE_BLOCK | DamageMaskNet | observed reference repair | reference-conditioned inpainting | YES | YES | RefFace if qualified | safe abstention when zero evidence |
| BLACK_BAR | DamageMaskNet | observed reference repair | reference-conditioned inpainting | YES | YES | RefFace if qualified | safe abstention Conservative |
| PARTIAL_OCCLUSION | DamageMaskNet/parser | observed repair where valid | generated candidate if Paper mode | YES | YES | RefFace/other qualified | unresolved/rollback |
| MISSING_COMPONENT | DamageMaskNet/component visibility | observed component bank | generated component | YES | YES | RefFace/CodeFormer/etc. after qualification | unresolved Conservative |
| LOW_LIGHT | detector/exposure | deterministic/light specialist only if genuinely low light | Zero-DCE++ candidate after audit | optional | YES if qualified | Zero-DCE++ feasibility | no blanket brightening |
| MIXED_DAMAGE | multi-label damage map/router | specialist sequence from common checkpoint | minimal additional models | depends on component | YES | dynamic router | never blindly chain all BFR models |

---

## 13. DECISION LOG — NEW DIRECTIONS

### DEC-20260819-001 — Canonical meta ledger
- **DATE:** 2026-08-19
- **PROPOSAL:** maintain canonical `PROJECT_MASTER_STATE.md` on `meta/project-state`.
- **PROBLEM:** project state had become dependent on chat memory and branch-local reports.
- **AFFECTED:** all versions/tracks.
- **EVIDENCE:** multiple branches, stale V1 release docs, diverged research branches, current hotfix/research state.
- **BENEFIT:** auditable continuity and exact SHA history.
- **RISK:** stale ledger if update protocol is ignored.
- **ALTERNATIVE:** branch-local reports only; rejected as fragmented.
- **REVERSAL CONDITION:** only if replaced by an equally canonical machine+human state system with migration.
- **STATUS:** ACCEPTED.

### DEC-20260819-002 — Advanced research branch is active architecture
- **PROPOSAL:** treat `research/paper-quality-local-v2` as active Paper Quality architecture; preserve early `research/face-restoration-v2` as superseded research history.
- **EVIDENCE:** early branch ~2 commits of dataset/degradation work; advanced branch ~83 commits of model/runtime architecture; branches diverged and are not a literal merge.
- **RISK:** useful early assets could be forgotten.
- **MITIGATION:** explicit review/port, never fake merge status.
- **STATUS:** ACCEPTED.

### DEC-20260819-003 — 80% whole-PC budget + one heavy model
- **PROPOSAL:** <=80% logical CPU, <=80% process/system RAM, one heavy model resident at a time.
- **PROBLEM:** target is 16GB Windows laptop, CPU-first.
- **BENEFIT:** preserves Windows/UI headroom and avoids simultaneous model memory spikes.
- **REVERSAL:** only measured target-PC evidence can justify a versioned change.
- **STATUS:** ACCEPTED.

### DEC-20260819-004 — Evidence authority outranks generated quality
- **PROPOSAL:** healthy MAIN > verified same-person observed reference > accepted generated candidate.
- **PROBLEM:** generative BFR may look better while hallucinating identity/detail.
- **STATUS:** ACCEPTED; core provenance rule.

### DEC-20260819-005 — Replace Wikimedia-dependent DamageMaskNet acquisition with mixed source bank
- **PROBLEM:** attempt 1 HTTP 403, attempt 2 HTTP 429 before training.
- **CHOSEN:** FairFace real + ControlFace multi-view source bank with hashes/splits; U-Net hypothesis unchanged for attempt 3.
- **RISK:** dataset-domain shift; must validate per class.
- **STATUS:** ACCEPTED for attempt 3.

### DEC-20260819-006 — RefFace is next large-occlusion specialist after DamageMaskNet gate
- **PROBLEM:** sticker/black-bar/missing component with same-person reference needs specialized reference-conditioned generation.
- **EVIDENCE:** official RefFace code/checkpoints, MIT repo; prepared minimal CPU path strips inference-useless discriminators and patches one device-hardcoded allocation without changing weights/architecture.
- **ALTERNATIVES:** generic CodeFormer everywhere, InstantRestore, OSDFace, GPEN inpainting.
- **WHY CHOSEN:** more task-specialized and plausibly lighter than diffusion/multi-UNet alternatives.
- **REVERSAL:** identity/resource/quality failure in measured attempt series.
- **STATUS:** ACCEPTED / BLOCKED BY SEQUENCE.

### DEC-20260819-007 — V3 consumed; V4 one-shot remains untouched
- **PROBLEM:** prevent benchmark leakage/tuning.
- **EVIDENCE:** V3 39/40 and consumed; V4 freeze exists, no consumed/request marker.
- **STATUS:** ACCEPTED safety protocol.

---

## 14. EXPERIMENT LOG

### EXP-20260817-001 — GPEN BFR-512 vertical slice
- **HYPOTHESIS:** GPEN can provide useful paper-quality CPU restoration while preserving identity.
- **ATTEMPTS:** 2 effective setup attempts; first blocked by missing environment dependency before model execution; second real inference PASS.
- **DATASET/SPLIT:** DEVELOPMENT, no final holdout.
- **BACKEND:** Linux CPU, batch 1, aligned 512.
- **RESULT:** real output; SFace `0.95397`, `~2.697s`, `~1.828GB`, PSNR `28.07`, SSIM `0.7474`.
- **CONCLUSION:** BENCHMARKING, not qualified; broaden data/license/Windows evidence.
- **ARTIFACT:** `research/GPEN_VERTICAL_SLICE_REPORT.md` + workflow artifact.

### EXP-20260817-002 — GFPGAN v1.4 vertical slice
- **HYPOTHESIS:** v1.4 offers strong real-world restoration in same common aligned comparison.
- **ATTEMPTS:** initial technical run succeeded but alignment mismatch invalidated A/B; corrected comparable run succeeded.
- **RESULT:** SFace `0.91665`, `~2.787s`, `~1.666GB`, PSNR `30.65`, SSIM `0.8604`.
- **CONCLUSION:** BENCHMARKING; one case cannot choose winner.

### EXP-20260817-003 — CodeFormer w=0.5 CPU vertical slice
- **HYPOTHESIS:** official `w=0.5` severe-restoration candidate can run locally under resource governor.
- **ATTEMPT 1:** packaging/import failure before model inference (`basicsr.version`).
- **ATTEMPT 2:** real CPU inference PASS under 80% governor.
- **METRICS:** exact comparative metrics must be read from stored artifact before restating.
- **CONCLUSION:** BENCHMARKING; license blocker remains.

### EXP-20260817-004 — FBCNN JPEG QF20
- **HYPOTHESIS:** specialist JPEG cleanup improves fidelity/identity before any generative face model.
- **ATTEMPT:** real Linux CPU PASS.
- **RESULT:** PSNR `34.62→36.78`, SSIM `0.9486→0.9634`, SFace `0.9571→0.9691`, peak RSS `~1.305GB`.
- **CONCLUSION:** current JPEG leader in DEV; broader JPEG family validation required.

### EXP-20260818-005 — DamageMaskNet U-Net attempt 1/3
- **HYPOTHESIS:** lightweight U-Net can segment exact synthetic target damages.
- **RESULT:** INFRASTRUCTURE FAIL before training: Wikimedia HTTP 403.
- **DECISION:** fix acquisition only; preserve hypothesis.

### EXP-20260818-006 — DamageMaskNet U-Net attempt 2/3
- **RESULT:** INFRASTRUCTURE FAIL before training: sources began verifying, then Wikimedia HTTP 429.
- **DECISION:** do not change U-Net; move acquisition to mixed open-source bank.

### EXP-20260818-007 — DamageMaskNet U-Net attempt 3/3
- **DATA:** FairFace + ControlFace mixed source bank; identity-disjoint ControlFace validation; same U-Net hypothesis/hyperparameters.
- **RESULT:** **NOT_VERIFIED** — run was triggered but final evidence is not exposed through the available Actions interface.
- **DECISION:** do not launch attempt 4 for observability. Recover existing evidence. If actual model/data quality failure, U-Net hypothesis is exhausted and must stop/reassess.

### EXP-20260819-008 — RefFaceInpainting CPU vertical slice
- **STATUS:** PREPARED / NOT_RUN.
- **ATTEMPT:** 0/3 consumed.
- **INPUT PLAN:** two ControlFace views of same identity, exact opaque facial mask, CFS parser/YuNet/SFace.
- **BACKEND:** CPU, minimal `UnetG + ArcFace resnet101`, one device-neutral allocation patch.
- **ACCEPTANCE:** 80% resource contract, SFace >=0.363, exact MAIN outside mask, GENERATED provenance, hashes/runtime/RAM.
- **NEXT:** run attempt 1 only after DamageMaskNet gate resolution.

---

## 15. QUALITY SCOREBOARD

Never combine these scopes into one overall number.

### DEVELOPMENT

- GPEN, GFPGAN, FBCNN measurements: see Section 7.
- CodeFormer real slice PASS; exact numerical artifact recovery required for comparison.
- wrong-person/provenance for production safety remain governed by hard gates; Paper Quality broad DEV scoreboard not complete.

### VALIDATION

- Broad identity-disjoint blind-restorer validation: INCOMPLETE.
- DamageMaskNet validation: NOT_VERIFIED for attempt 3.
- Personalized reference bank validation: framework exists; latest workflow result NOT_VERIFIED.

### HOLDOUT

- V1 final historical: certified 40/40.
- V3: consumed `39/40`, failure SFace `0.360 < 0.363`.
- V4: frozen, NOT_RUN, UNCONSUMED.

### REAL-WORLD

- Female-domain historical and current stress runs exist.
- Current hotfix same-HEAD female workflow: FAIL at lightweight report validation after benchmark execution.
- No claim of current hotfix real-world readiness.

### TARGET-PC

- HP EliteBook 1030 G3: NOT_RUN for Paper Quality acceptance.

Metrics to maintain per scope: SFace/identity, PSNR, SSIM, LPIPS, NIQE when useful, healthy MAE, recovery per damage family, wrong-person observed pixels, provenance violations, generated/reference-supported/unresolved fractions, component geometry drift, runtime and peak RAM.

---

## 16. TARGET HARDWARE

Primary deployment target:

- **Machine family:** HP EliteBook 1030 G3
- **RAM:** 16 GB
- **OS:** Windows
- **Exact CPU/GPU:** detect at runtime; do not assume SKU.
- **Architecture:** CPU-first.
- **Optional acceleration:** OpenVINO / compatible Intel iGPU / other backend only after actual support + output parity benchmark.
- **CUDA:** not required and must not be required.
- **Resource contract:** <=80% logical processors, <=80% process RAM, <=80% whole-system RAM, max one heavy restoration model resident.

Every serious model must eventually record real target-PC load time, inference time, peak RAM, backend, output hash and identity result. Linux CPU timing is never relabeled as target-hardware evidence.

---

## 17. RELEASE SAFETY RULES

Frozen/current safety invariants unless an explicitly approved versioned policy supersedes them with evidence:

- **SFace threshold:** `0.363`.
- **Wrong-person observed contribution:** `0 pixels`.
- **Provenance violations:** `0`.
- **Healthy/outside repair MAE:** `<=8.0` where the existing frozen release protocol applies.
- Calibration remains independent.

Forbidden shortcuts:

- threshold-shopping;
- benchmark-shopping/cherry-picking;
- rerunning/tuning consumed holdouts;
- deleting difficult cases;
- representing generated pixels as observed evidence;
- using raw wrong-person references to improve identity score;
- auto-merge/force-push certified history;
- fabricated Actions results, RAM, speed, hashes or output images.

---

## 18. PROVENANCE CLASSES

Minimum canonical provenance vocabulary:

- `MAIN_OBSERVED`
- `OBSERVED_REFERENCE`
- `SYMMETRY_INFERRED`
- `GENERATED_MODEL_INFERRED`
- `UNRESOLVED`

Generated content never changes class merely because it passes identity or resembles a reference.

---

## 19. TRACK A — PRODUCT_V1_1 OPERATIONAL RELEASE

Current exact head: `3645c8c39653d04616167e881adaf28d2b93cd45`.

Same-HEAD state:

- **Release Quality:** FAIL. Targeted regressions: `4 failed, 195 passed`.
  - `test_main_source_zero_never_changes_after_preflight`: legitimate reference expected component-transfer acceptance but got False.
  - `test_component_transfer_acceptance_is_persisted_for_all_blocks`: expected `[True]`, got `[False]`.
  - `test_final_identity_reuses_preflight_acceptance_instead_of_requiring_main_bridge`: expected accepted identity but got reason `main_not_in_accepted_sface_cluster`.
  - `test_restores_imported_primary_when_selected_anchor_is_verified_same_canvas`: expected identity-bridge matched reference `[1]`, got `[]`.
- **Windows:** FAIL on same HEAD; full Python-test stage failed. Exact nested assertion/count must be re-read before claiming more detail.
- **Female-domain:** FAIL on same HEAD. Benchmark execution reached completion; lightweight 300–400-case report validation failed. Exact nested assertion is NOT_VERIFIED in this reconciliation.

Immediate policy:

1. inspect current `app/preflight.py`, `app/identity_anchor_v4_policy.py`, `app/identity_anchor_v4_hardening.py`, `app/primary_anchor_policy.py`, `app/face_resilience_binding_policy.py` and the four failing tests;
2. distinguish legitimate same-person evidence from reference-only/transitive/wrong-person rescue;
3. fix root cause without lowering SFace or weakening fail-closed behavior;
4. targeted tests -> full pytest -> same-HEAD Windows/Female/Release Quality;
5. no V3 execution; no V4 execution until pre-final sequence is valid.

---

## 20. TRACK B — PAPER QUALITY

Active branch: `research/paper-quality-local-v2@645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd`.

This is a real ML research track, not documentation-only. Model states must converge to `QUALIFIED`, `REJECTED`, or a documented blocker after sufficient evidence rather than remaining indefinitely optional.

Current implemented/researched layers:

- 80% resource governor + one-heavy-model lifecycle;
- GPEN/GFPGAN/CodeFormer/FBCNN vertical slices;
- damage taxonomy and DamageMaskNet pipeline;
- 13-component component bank including FACE_CONTOUR;
- Personalized Reference Bank;
- reference-first repair with original source-index provenance;
- hard-gated candidate selector framework;
- deterministic component-aware fusion;
- yakhyo 19-class parser adapter;
- RefFace CPU vertical slice prepared but manual-only/NOT_RUN.

---

## 21. CURRENT PAPER QUALITY BLOCKER

**DamageMaskNet attempt 3 evidence recovery.**

Do not trigger attempt 4 merely because evidence is inconvenient to retrieve.

Classify recovered attempt 3 as:

### A. PASS
Read and record:
- IoU per class;
- F1 per class;
- ONNX parity;
- RAM;
- runtime;
- source-bank hashes/split evidence.

Then decide whether lightweight U-Net remains valid and scale DEVELOPMENT/VALIDATION appropriately.

### B. INFRASTRUCTURE FAIL
Correct acquisition/infrastructure only. Do not silently alter model/data hypothesis.

### C. MODEL/DATA QUALITY FAIL
If attempt 3 genuinely reaches training/evaluation and fails quality, the three-attempt U-Net hypothesis is consumed. Stop it and create a new documented hypothesis for the next lightweight architecture (e.g. MobileNetV3 segmentation head / lightweight DeepLab), rather than micro-tuning indefinitely.

After this blocker is resolved, run RefFaceInpainting CPU vertical slice **attempt 1/3**, provided prerequisites still hold.

---

## 22. SPECIALIST MODEL STRATEGY

Long-term routing:

`INPUT -> detect/align -> damage classification -> reference analysis -> identity anchors -> specialist route -> candidates -> component evaluation -> identity/geometry/quality hard gates -> evidence-aware deterministic fusion -> final identity -> provenance/export`

Principles:

- JPEG -> FBCNN candidate.
- Blur -> dedicated deblur or appropriate BFR, not automatic generator chaining.
- Opaque occlusion + valid same-person reference -> observed-reference reconstruction first; RefFace-like specialist only if qualified.
- Unsupported missing detail -> Paper Quality generation only when enabled; mark GENERATED.
- Observed reference detail -> evidence-first.
- Never run GPEN -> GFPGAN -> CodeFormer serially simply because all exist.

---

## 23. MODEL SELECTION POLICY

For GPEN/GFPGAN/CodeFormer and future blind restorers:

- use multiple identity-disjoint DEVELOPMENT/VALIDATION faces;
- evaluate per degradation family, not a single average;
- measure identity, geometry, artifact rate, component fidelity, healthy-region preservation, PSNR/SSIM/LPIPS and runtime/RAM;
- keep final holdout untouched while selecting weights/models;
- identity remains a hard gate before weighted quality ranking;
- record why the winner won;
- remove models whose measured incremental benefit does not justify RAM/runtime/dependency/license cost.

---

## 24. HISTORICAL RECORD — append-only important states/pushes

### HIST-20260815-001 — PRODUCT_V1 certified merge
- **Technical branch/candidate:** `release/v1-certified@f476c6f04b57b658fd152a0a82e5b50cb5afbdbc`
- **Resulting main:** `2767513f95dde2d417e7c6f1faf2357149a1a32f`
- **Evidence:** merge commit records Windows #1195, Female-domain #463, Release Quality #13 certification.
- **Models/objectives:** baseline production stack; OBJ-001.
- **Decision:** freeze as certified PRODUCT_V1 regression base.

### HIST-20260818-002 — Track A current blocked hotfix state
- **Branch:** `hotfix/real-world-restoration-v1.1`
- **Technical HEAD:** `3645c8c39653d04616167e881adaf28d2b93cd45`
- **Tests/workflows:** Release Quality FAIL (`4 failed, 195 passed` targeted); Windows FAIL; Female-domain FAIL.
- **Affected objective:** OBJ-002.
- **V3:** consumed, no rerun.
- **V4:** frozen/unexecuted/unconsumed.
- **Next:** fix legitimate identity/preflight bridge regression without safety relaxation.

### HIST-20260818-003 — Advanced Paper Quality research snapshot
- **Branch:** `research/paper-quality-local-v2`
- **Technical HEAD:** `645862d1b8ff3c1d7abe7df6cee0e17e4f2d68dd`
- **State:** real DEV model evidence + research routing/fusion infrastructure; DamageMaskNet attempt 3 NOT_VERIFIED; RefFace PREPARED/NOT_RUN.
- **Models:** GPEN, GFPGAN v1.4, CodeFormer, FBCNN, DamageMaskNet, RefFace and feasibility candidates.
- **Objectives:** OBJ-004 through OBJ-008.
- **Next:** recover DamageMaskNet attempt 3 evidence without rerun.

### HIST-20260819-004 — Canonical ledger established
- **Technical/meta action:** created `meta/project-state` from certified `main@2767513f95dde2d417e7c6f1faf2357149a1a32f` and established this root ledger.
- **Purpose:** make repository state self-contained for future engineers/sessions.
- **Tests:** documentation-only action; repository state reconciled before creation.
- **Next engineering action:** Track A inspect/fix exact current identity/preflight regressions; in parallel, recover existing Track B DamageMaskNet attempt-3 evidence when observable. Every subsequent technical push must be followed by a ledger update.
