# SPECIALIST_MODEL_ROUTING_V2.md

Status: 2026-08-18
Branch: `research/paper-quality-local-v2`

## Non-negotiable execution budget

Paper Quality mode uses `app/resource_budget.py` with a maximum total-PC resource fraction of **0.80**.

- CPU affinity: at most `floor(logical_processors * 0.80)` logical processors.
- Heavy models: **one at a time**.
- Process working-set/RSS ceiling: no more than `0.80 * total_physical_RAM` when measurable.
- Whole-system physical RAM: CFS refuses a heavy-model load/inference when projected **total PC RAM use** would exceed 80%.
- Every heavy route performs checks before load and after load/inference/unload.
- OpenMP, MKL, OpenBLAS, NumExpr, OpenCV and PyTorch thread counts are capped.
- No quality or identity gate is weakened to stay inside the budget. A model that cannot qualify under the budget is rejected for the target PC.

On the intended 16 GiB EliteBook, the nominal 80% physical-RAM ceiling is approximately 12.8 GiB for **all running software combined**, preserving at least ~3.2 GiB for Windows/UI/other processes under normal measurable conditions.

## Selection philosophy

Use the most specialized model that wins on the actual degradation family. Do not run every model and do not chain generative restorers destructively.

Every heavy candidate starts from a common aligned checkpoint and returns a `RestorationCandidate`. Identity is a hard gate before perceptual ranking. Observed same-person evidence has higher evidentiary authority than any generated candidate.

## Primary specialist map

| Damage / task | Primary route to qualify | Secondary / challenger | Why |
|---|---|---|---|
| Mild deblur / denoise | Existing NAFNet | task-specific NAFNet challenger only if measured | Lightweight existing production path; no need for a facial generator on mild damage. |
| JPEG / double-JPEG / social recompression | **FBCNN** | NAFNet/SwinIR only if validation wins | FBCNN is directly specialized and already improved PSNR, SSIM and SFace on the first real CFS JPEG slice. |
| Genuine low light | Zero-DCE++ research specialist | later lightweight low-light challenger | Run only after low-light detection; licensing can block shipping. |
| Sticker / scribble / mosaic / missing component with valid same-person evidence | **Observed per-component reconstruction** | RefineFIR-inspired copy-or-not scoring | Real same-person evidence outranks generated content and best matches the product's identity goal. |
| Personalized severe degradation with several references | **InstantRestore benchmark candidate** | CFS component bank + RefineFIR; FaceMe only if needed | InstantRestore is specifically designed for a degraded face plus a small set of same-person references in one forward pass. It still must prove CPU/Windows/80%-budget feasibility. |
| Fine facial detail from one strong reference | **RefineFIR benchmark/concept** | current component-bank scoring | Explicitly addresses whether identity-specific details should be copied from the reference. |
| Large opaque facial occlusion with one strong reference | **RefFaceInpainting benchmark candidate** | CodeFormer inpainting | Specialized for large missing facial regions with identity and component texture control. |
| Fast severe blind facial restoration | **GPEN BFR-512** | GFPGAN v1.4 | GPEN currently has the strongest measured SFace among the first CFS blind slices. |
| Fidelity-biased blind facial restoration | **GFPGAN v1.4** | CodeFormer | Best first-case PSNR/SSIM among GPEN/GFPGAN while remaining below identity threshold risk. |
| Controllable blind / inpainting prior | **CodeFormer** | GFPGAN | Useful fidelity knob and official aligned-face/inpainting paths; licensing remains visible. |
| Extreme blind damage without adequate reference evidence | **OSDFace research challenger** | GPEN/GFPGAN/CodeFormer | Modern one-step diffusion with explicit identity loss; only promoted if real CPU gain justifies cost. |
| Damage classification / localization | **DamageMaskNet trained for CFS** | deterministic heuristics + face parsing | Exact synthetic masks make this one of the few models worth training specifically for CFS. |
| Identity verification | **Frozen SFace hard gate** | independent ArcFace/AdaFace evaluator after calibration | Safety threshold is never lowered; secondary embeddings are diagnostic/consensus only. |
| Face-only upscale | Real-ESRGAN x2 only after restoration | SwinIR if target validation wins | No global background SR by default on CPU. |

## Personalized routing for MAIN + 0..9 references

For every identity-critical component independently:

1. Build accepted identity anchors from valid **full-face** references only.
2. Keep partial references component-local.
3. Rank observed component evidence by visibility, sharpness, damage, pose/geometric fit, exposure and agreement with other accepted references.
4. If adequate observed evidence exists, reconstruct from the best same-person component source(s).
5. If observed evidence is insufficient in Paper Quality mode, run the most specialized learned candidate **one at a time**:
   - personalized multi-reference severe case -> InstantRestore first after qualification;
   - one-reference fine-detail case -> RefineFIR route;
   - large missing area -> RefFaceInpainting;
   - blind severe residual -> GPEN/GFPGAN/CodeFormer, then OSDFace only if needed.
6. Apply SFace hard gate and geometry/healthy-boundary checks.
7. Mark all selected generated pixels `GENERATED_MODEL_INFERRED`.
8. Unload the heavy model before another heavy candidate may be loaded.

A final face may legitimately use:

- left eye from REF3 observed evidence,
- right eye from REF5 observed evidence,
- nose from MAIN,
- mouth from an accepted generated InstantRestore/CodeFormer candidate,

provided geometry, identity, seams and provenance all pass.

## Dynamic stop policy

Do not execute all models merely to compare them in production.

Example severe route:

`COMMON CHECKPOINT -> specialist A -> hard gates -> score`

If accepted with sufficient calibrated quality margin: **STOP**.

Only when rejected/ambiguous:

`unload A -> specialist B -> hard gates -> score -> unload B`

Never:

`GPEN -> GFPGAN -> CodeFormer -> OSDFace`

as a destructive image chain.

## Scientific-best vs EliteBook-best

A newer model is not automatically the shipping winner.

InstantRestore, OSDFace, FaceMe and other diffusion-based systems remain `RESEARCH` until they satisfy all of:

1. official source and checkpoint provenance;
2. code/weights license audit;
3. real CPU execution;
4. Windows execution;
5. <=80% CPU/process RAM/**whole-system RAM** policy;
6. no OOM or persistent memory leak after unload;
7. frozen identity hard gate;
8. measurable improvement on the exact degradation family;
9. acceptable seconds-per-512-face on the real HP EliteBook 1030 G3.

If they fail, a smaller GPEN/GFPGAN/FBCNN/NAFNet/reference-first route is the better production model even if a paper reports stronger GPU results.

## 2026 paper-only teachers

- **PerFuSe (CVPRW 2026):** highly relevant personalized full-image modular fusion and smartphone evaluation; no official executable repository/checkpoint found in the 2026-08-18 audit, so no runtime is claimed.
- **Reference-Guided Identity Preserving Face Restoration (2025):** composite reference context, hard-example identity loss and training-free multi-reference adaptation; current public repository contains paper material rather than an executable model release.
- **BioDDM (CVPRW 2026):** use biometric-subspace guidance as an identity-ranking concept, not a default CPU diffusion dependency.

These works can influence CFS architecture, but cannot be called installed, compatible or benchmarked until code and weights actually exist and execute.

## Qualification order from current state

1. Complete DamageMaskNet mixed-source DEVELOPMENT slice and ONNX parity.
2. Scale DamageMaskNet development/validation data toward 300–400 identity-disjoint sources; compare another lightweight segmentation architecture only if the U-Net hypothesis is inadequate.
3. Connect damage class/confidence to existing component-bank per-component routing.
4. Run **InstantRestore** as the first new learned personalized vertical slice under the 80% governor.
5. Benchmark RefineFIR and RefFaceInpainting on their specialist reference/occlusion cases.
6. Run the broader degradation matrix for GPEN/GFPGAN/CodeFormer/FBCNN using the common adapter.
7. Benchmark OSDFace only on severe blind cases that remain unresolved.
8. Promote only measured winners per damage family.

Final holdouts remain untouched until development/validation choices are frozen.
