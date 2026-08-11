# Conservative Face Studio — Release State

## Verified repository state

- Branch: `feature/block-pipeline-v1`
- Pull request: `#1` → `main`
- PR HEAD before the final failure-injection batch: `e54abd54216a527d4dbc57a4d4e1c50c0e6fa788`
- Local content-equivalent base commit: `68084d379d35db49bfb845aaa61e4e5a3624db8b`
- Base `main` SHA: `5eff667373cd47c07ba14aaad2acafee6d5a61c1`
- Tested workflow PR HEAD: `e54abd54216a527d4dbc57a4d4e1c50c0e6fa788`
- Tested PR merge SHA: `c1c15af445669ef700876599d5fac506b6511014`
- The merge SHA is recorded separately from PR HEAD and was emitted as `GITHUB_SHA` by the verified Windows workflow.
- `PRODUCT_COMPLETE_PRE_TUNING`: **TRUE**
- `ARCHITECTURE_FROZEN`: **TRUE at the production runtime represented by PR HEAD**
- `RELEASE_READY`: **FALSE**
- TARGET95 policy: **REPORT ONLY**
- Current required gate: **FINAL_FAILURE_INJECTION_CI**

## Verified workflow evidence

- Windows build `#1181` (`31471837461`): **SUCCESS** on tested PR HEAD and merge SHA.
- Female-domain benchmark `#449` (`31471837454`): **SUCCESS** on the same tested PR HEAD.
- Female-domain report: **379 completed, 0 runtime errors**; TARGET95 remains report-only.
- The Windows validation artifact reports `product_complete_pre_tuning = true` only after the
  full test, runtime, model, EXE, portable, offline, ZIP, updater, installer, clean-install and
  installed-application gates pass.
- Windows artifacts are present and unexpired:
  - `ConservativeFaceStudio-Validation` (`9094599248`)
  - `ConservativeFaceStudio-Production-Model-Updates` (`9094589479`)
  - `ConservativeFaceStudio-Release-Metadata` (`9094587606`)
  - `ConservativeFaceStudio-Portable-Windows-x64` (`9094587194`)
  - `ConservativeFaceStudio-Setup-Windows-x64` (`9094585097`)

The stale `#1176` / `#444` failures and the intermediate Windows `#1178`-`#1180`
portability failures are historical and are not current blockers.
`product-audit.json` remains a static source/module inventory; it is not edited to impersonate
packaging or workflow evidence.

## Architecture freeze

Production runtime architecture is frozen at PR HEAD. Until a measured regression proves a
minimal production change necessary, do not add runtime blocks or features, restructure modules,
redesign APIs/GUI/workflows, or alter model routing. Benchmark and evaluation tooling for the
current quality gate must remain isolated from production runtime.

Permanent contracts remain unchanged:

- MAIN/source0 is the target, canvas, primary geometry and provenance source.
- MAIN plus `0..9` references remains supported.
- References are optional observed evidence; abstention is valid.
- Immutable originals, source eligibility, protected regions and rollback remain active.
- Generated/model-inferred pixels never become original MAIN/reference evidence.
- Partial references may use component-local alignment without requiring a global transform.
- Normal restoration remains offline and uses checksum-verified local production weights.

## Frozen face-smartphone benchmark

The supplied real examples have been inspected as domain references. They show opaque graphic
occlusions, rectangular blur, mosaic/pixelation, black paint/scribble, physical masks and mixed
smartphone degradation concentrated on facial regions.

The frozen benchmark keeps two sets separate:

- **Set A — controlled ground truth:** real clean reusable photographs, controlled synthetic face
  damage only, fixed source/mask checksums, fixed seeds/references and a frozen 60/40 split.
- **Set B — real-world challenge:** actual damaged images used only for conservative visual,
  provenance, abstention, artifact and catastrophic-failure review. No quantitative ground truth
  is invented.

The `face-smartphone-v1` Set A freeze contains exactly 100 primary cases: 60 calibration and 40
untouched holdout. All 100 target the face, all ordinary controlled masks have 100% overlap with
the declared facial restoration domain, mask checksums are unique, reference counts cover `0..9`,
and calibration/holdout source identities are disjoint. Frozen digests:

- case manifest: `74162d364727c8fdcd8b5242b48ba3f05dd99e12c3c8aebfe65fcae58e08f537`
- source manifest: `1bba20dcd587972b54e3b63d4e250d10e47c46b548eb144c8ae71375bf2c63a0`
- contract: `a14d2d3e45f34ec565442fa92e8ae32d6a776d780f7d09fe35a060df255a0973`

The source manifest records the original reuse license and attribution for every clean image. Set B
contains only the supplied qualitative real-damage examples; those files are not committed and no
quantitative ground truth is claimed for them.

## Frozen unchanged-production baseline

The unchanged production pipeline at `368f00c122520f471e7ef310f9daf8781b51f111` ran with the six
checksum-verified production weights. It completed the 40-case fast subset and the full 60-case
calibration split without touching holdout:

- calibration completed: **60/60**
- runtime errors: **0**
- hard guardrail passes: **55/60**
- provenance-invalid cases: **0**
- TARGET95: **report only** (`13/51` applicable cases reported, not used as a gate)
- full local baseline report SHA256:
  `a1a0aab21dd08ba034d2dcca2a2511703290e789e5e23c836ac43490105c5f74`

The five guardrail misses were reviewed individually. Four contain final pixels from an explicitly
wrong-person full-face reference (17,606 to 36,753 pixels); severe MAIN damage caused primary face
detection to abstain, and the later reference-derived landmark path treated every geometrically
aligned reference as partial evidence instead of retaining the preflight identity-cluster veto. The
fifth miss (`partial_face_crop`) has zero wrong-person final pixels and is a separate preservation
case.

## RefSel fast qualification

RefSel is **DEFERRED_LEGAL_ARTIFACT**. The official RefSTAR repository currently exposes no code
license and does not publish an immutable, checksum-addressed RefSel checkpoint or redistribution
terms. No checkpoint was downloaded, no adapter was created, no dependencies were added, and no
production routing or weight was changed. This valid stop outcome does not block the base release.

## Calibration candidate and untouched holdout

The isolated `face-domain-guard-v1` candidate changed only existing source-eligibility and facial
intervention boundaries; it changed no model weights, model routing, TARGET95 policy, benchmark
case, mask, source or split. On the complete 60-case calibration split it completed **60/60** with
zero runtime/provenance errors, **60/60** hard-guardrail passes and zero wrong-person final pixels.
The candidate was therefore admitted to the single untouched holdout run.

The untouched holdout completed **40/40** with zero runtime errors, zero provenance violations and
zero wrong-person final pixels, but passed **39/40** frozen hard guardrails. Case
`cfsfs1-hol-079-small_block_mosaic` had undamaged-region MAE `8.464879989624023`, above the frozen
`8.0` limit. The candidate is therefore **NO-GO** and its production changes have been removed from
the working tree. TARGET95 remained report-only (`9/34` applicable passes) and was not used in the
decision. Immutable report evidence:

- calibration candidate summary: `benchmarks/face-smartphone-v1/calibration-candidate-summary.json`
- holdout summary: `benchmarks/face-smartphone-v1/holdout-summary.json`
- raw holdout report SHA256: `58c64c6e400e2266d389f046e9dd23bdef744a5a6df199aec5bcb8d50d1775e6`

## Current blocker

The first calibration candidate failed one frozen holdout undamaged-region preservation guardrail.
The holdout has now been consumed and must not be used for candidate tuning. Lowering the frozen
threshold or adapting the rejected candidate to the observed holdout case is prohibited.

Remaining sequence:

`FINAL FAILURE-INJECTION CI → HOLDOUT PROTOCOL DECISION → FINAL REGRESSION`

## Final failure-injection candidate

The final evaluation-only gate covers 19 mandatory scenarios: missing/corrupt weights, checksum
failure, model smoke failure, updater rollback, no network, no GPU, GPU inference failure, CPU
fallback, wrong/unreadable references, 0/9 references, tiny MAIN, EXIF orientation, Unicode Windows
paths, low disk, restoration crash, stale lock and restart/recovery. Local evidence is **19/19 PASS**
and the complete local suite is **416/416 PASS**. One functional import defect found by this gate was
fixed minimally: product image loading now uses a Unicode-safe byte read before OpenCV decode while
retaining EXIF orientation. No production model, coefficient, quality threshold, TARGET95 rule or
restoration routing changed. CI evidence for this final batch is still required.

## Next exact action

Push the single final failure-injection batch and let both workflows finish without interruption.
If CI is green, preserve the NO-GO record and do not tune against the consumed holdout. Release still
requires a product decision: retain the unchanged baseline or authorize a new independently frozen
holdout for a new candidate.
