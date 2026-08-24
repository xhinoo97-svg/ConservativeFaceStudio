# Final integration — Phase 0 repository inventory

## Outcome

Phase 0 is complete on `integration/final-paper-quality-local` created at exact immutable base
`main@2767513f95dde2d417e7c6f1faf2357149a1a32f` (tree `d444f3191b58f3213263a40480bd8e861a903b72`).
The baseline full suite passes `434/434`. This is an inventory and regression baseline, not a
Paper Quality, Target95, Windows or release qualification.

The machine-readable companion is `config/final-integration-inventory.json`.

## Reconciled remote truth

| Line | Exact SHA / state | Decision |
|---|---|---|
| `main` | `2767513f95dde2d417e7c6f1faf2357149a1a32f` | immutable integration base |
| Track A | `77687b3b171f4e9989fcf486834f2d8b7a52f591` | preserve consumed evidence; selective source only |
| evaluated Track A candidate | `b6ce7ebde87d4ce84e5849664716dc3e822ad762` | V4 NO-GO; never reinterpret |
| Track B | `6d57725aae087bb4a3144d521d91346999f9a4fd` | selective research source only |
| protocol hardening | `268188c5a2540455ff804383cb583b16546b62f1` | DEV-tested source; no V5 created |
| canonical ledger | `3aa66028e9f26563ee8264c4d397df26c25c6268` | governing state/rules |
| PR #2 | OPEN, DRAFT, not merged | NO-GO; do not merge as certified |
| V3 / V4 | CONSUMED / CONSUMED_FAIL | no retry, rerun, tuning or reuse |
| V5 | NOT_CREATED / NOT_AUTHORIZED | blocked by quality/product gates |

The source deltas are intentionally too broad for an unreviewed merge: Track A changes 73 files,
the protocol line 80 files, and Track B 87 files versus `main`. Each group must therefore be
selected by origin commit/path, reviewed, tested and recorded.

## What is actually in the repository

Git contains `models/README.md` but no neural weight file. Production weights are obtained during
the verified build/model-pack process, checked against fixed SHA-256 values, exercised by smoke
tests, and staged by `scripts/stage_production_models.ps1`. Consequently, “declared ACTIVE” and
“weight present in this checkout” are distinct states.

The base code declares and wires six production-pack roles:

| Model | Declared | Runtime role | Git weight | Installer contract | Base integration CI |
|---|---|---|---|---|---|
| YuNet | ACTIVE | face detection / five landmarks | absent | staged with hash | NOT_RUN |
| SFace | ACTIVE | direct identity guardrail | absent | staged with hash | NOT_RUN |
| OpenCV NAFNet ONNX | ACTIVE | deblur | absent | staged with hash | NOT_RUN |
| Face Parsing ResNet18 ONNX | ACTIVE | semantic support | absent | staged with hash | NOT_RUN |
| Head Pose MobileNetV2 ONNX | ACTIVE | pose gate / 2D warp | absent | staged with hash | NOT_RUN |
| OpenCV LaMa ONNX | FALLBACK | small generated inpaint residual | absent | staged with hash | NOT_RUN |

Historical artifacts demonstrate these roles on earlier exact candidates, but no Windows,
installer, offline or physical-HP result exists yet for this new branch. Earlier evidence cannot
be relabelled as same-candidate evidence.

FBCNN and LR-ASPP are absent from the base product pipeline and installer. FBCNN has real Track B
DEV inference evidence but only a one-identity public matrix. LR-ASPP has a real frozen checkpoint,
ONNX parity and 40-identity synthetic/external DEVELOPMENT evidence, but its binary precision is
only `0.673451`, domain robustness is not qualified, and checkpoint redistribution licensing is
not explicit. Neither is production-qualified here.

GPEN, GFPGAN, CodeFormer, RestoreFormer, MediaPipe Face Landmarker, 3DDFA and other advanced
entries are catalog/testing declarations in this base; a catalog row is not a wired, installed,
measured backend.

## 13-block measured wiring

| # | Block | Actual base behavior | Phase-0 status |
|---:|---|---|---|
| 1 | Import | deterministic immutable input setup | wired |
| 2 | Deblur | NAFNet when verified pack exists; classical fallback | wired |
| 3 | Enhance | deterministic enhancer called with automatic `blend=0.0` | **NO-OP DEFECT** |
| 4 | Face/Landmarks | YuNet five-point geometry | wired, not dense |
| 5 | Align References | five-point/RANSAC/ORB deterministic transforms | wired |
| 6 | Detect Damage | face parsing + heuristics/reference consensus | wired, not multi-class router |
| 7 | Select Best Regions | observed-reference specific memory | wired |
| 8 | Repair/Inpaint | verified observed references first; tiny LaMa residual | wired |
| 9 | Fusion | deterministic observed-pixel fusion with provenance | wired |
| 10 | Pose | MobileNetV2 gate plus conservative supported 2D transform | wired |
| 11 | Identity Check | SFace normal path; proxy fallback still exists | wired, proxy must remain non-authoritative |
| 12 | Upscale | deterministic Lanczos | wired |
| 13 | Export | image, provenance sidecar and blocks ZIP | wired |

The automatic pipeline currently forces `parameters["blend"] = 0.0` for ENHANCE while the block is
reported as executed. This must become an honest `SKIPPED` decision or a real, gated operation;
it must not be renamed as FBCNN/JPEG work before FBCNN is actually loaded and invoked.

## UI and resource truth

The worker runs outside the UI thread and the progress bar has 13 positions. Its event contract is
only `(index, name)`. It does not yet report exact model/checkpoint, model load state, weighted
progress, elapsed/ETA, live CPU/RAM, component donor, or explicit PASS/SKIPPED/ROLLBACK/ABSTAIN/ERROR.
There is no demonstrated safe cancellation contract. Handler-local engine dictionaries retain
loaded engines, so “one heavy model at a time” is text in the UI but is not yet proven by unload
instrumentation or an acceptance test.

## Audit limitations

`scripts/audit_product_modules.py` reports 262 source/model entries as `PRODUCTION_READY`, 8 as
`FUNCTIONAL_BUT_UNVERIFIED`, and 21 as `OPTIONAL_RESEARCH`; however most source files are assigned
`PRODUCTION_READY` by a default path rule. This report is useful as a catalog but does not prove
runtime reachability, model loading, inference, installer inclusion or Windows operation. The new
inventory therefore records each of those dimensions separately.

## Gate and next action

Classification: `PHASE0_INVENTORY_COMPLETE_BASELINE_TEST_PASS_NOT_YET_MEASURED`.

Next exact action: selectively integrate valid Track A operational/safety changes while excluding
all V4 request, freeze, manifest, marker, workflow and execution artifacts. Run a targeted suite
and then the full suite before integrating the generic future protocol as a separate traceable
group. No advanced model is added before the branch full suite is green.
