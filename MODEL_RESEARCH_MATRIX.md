# MODEL_RESEARCH_MATRIX.md

Status date: 2026-08-18

Research branch base: certified V1 `2767513f95dde2d417e7c6f1faf2357149a1a32f`.

`NOT VERIFIED` means no authoritative statement has been established for ConservativeFaceStudio. Published paper quality is not treated as target-PC compatibility.

## Qualification rules

- `RESEARCH`: scientifically relevant, not yet executed in CFS.
- `BENCHMARKING`: official implementation/checkpoint has produced or is producing measured CFS evidence, but release gates remain open.
- `QUALIFIED`: real local execution, license, identity, quality, Windows and target-product gates passed.
- `REJECTED`: measured, legal or dependency evidence makes the model unsuitable for the product.
- Every Paper Quality heavy model must obey `app/resource_budget.py`: <=80% CPU affinity, <=80% process RAM, <=80% total system RAM, and one heavy model at a time.
- ONNX/OpenVINO is `NOT VERIFIED` unless supplied upstream or converted by CFS with numerical/identity/visual parity evidence.
- No final holdout is used to choose models.

## Measured Tier A candidates

| Model | Official source | Task | License / checkpoint | CFS evidence so far | Main risk | Target | State |
|---|---|---|---|---|---|---|---|
| **GPEN BFR-512** | CVPR 2021; `yangxy/GPEN` | Fast blind face restoration with embedded GAN prior | Official checkpoint available; code/weights redistribution terms still not sufficiently explicit for release qualification | Real Linux CPU 512 slice PASS; SFace `0.95397`; ~`2.697 s`; peak RSS ~`1.828 GB`; one development case only | Generative detail can diverge from ground truth; licensing unresolved; Windows/EliteBook not measured | Blocks 2/8 | **BENCHMARKING** |
| **GFPGAN v1.4** | CVPR 2021; `TencentARC/GFPGAN` | Blind face restoration with generative facial prior | Apache-2.0 code; exact asset terms still recorded separately | Apples-to-apples Linux CPU slice PASS; SFace `0.91665`; ~`2.787 s`; peak RSS ~`1.666 GB`; PSNR/SSIM higher than GPEN on the first development case | Identity change under strong generative prior; one case is insufficient | Blocks 2/8 | **BENCHMARKING** |
| **CodeFormer** | NeurIPS 2022; `sczhou/CodeFormer` | Codebook/Transformer restoration with fidelity control and face-inpainting model | S-Lab License 1.0 / non-commercial constraints remain a production blocker | Real CPU vertical slice PASS at official aligned-face `w=0.5` after packaging fix; 80% governor active; comparison evidence corrected | License; fidelity-realness trade-off; Windows/EliteBook not measured | Blocks 2/8 | **BENCHMARKING** |
| **FBCNN** | ICCV 2021; `jiaxi-jiang/FBCNN` | Blind JPEG artifact removal | Apache-2.0; official color checkpoint | Real CPU QF=20 slice PASS. PSNR `34.62 -> 36.78 dB`; SSIM `0.9486 -> 0.9634`; SFace `0.9571 -> 0.9691`; peak RSS ~`1.305 GB` | Specialist only; must qualify double-JPEG/social/smartphone recompression and Windows | Block 3 before BFR only when JPEG is detected | **BENCHMARKING / current JPEG leader** |
| **NAFNet** | ECCV 2022; `megvii-research/NAFNet` | Mild deblur/denoise | Existing CFS ONNX production asset | Existing certified V1 path remains authoritative | Not a facial-prior generator | Blocks 2/3 | **Existing V1 component** |
| **Zero-DCE++** | TPAMI 2021; `Li-Chongyi/Zero-DCE_extension` | Low-light enhancement | Non-commercial/academic restrictions in upstream licensing | Not yet CFS-benchmarked | Exposure/colour drift; license; only route on genuine low light | Block 3 | **RESEARCH** |

## Highest-priority personalized / severe specialists

| Model | Official source / paper | Specialization | Upstream runtime evidence | Why it matters to CFS | Target-PC concern | Priority / state |
|---|---|---|---|---|---|---|
| **InstantRestore** | SIGGRAPH 2025; official `snap-research/InstantRestore` | **Single-step personalized face restoration** using shared-image attention and ~4 same-person references; no per-identity fine-tune at inference | Official code and pretrained checkpoints are published; upstream setup is CUDA-oriented PyTorch | Closest published model to the actual CFS use case: degraded MAIN + several photos of the same person; direct competitor to our reference-bank route | Diffusion backbone may still be too slow/heavy on EliteBook CPU; must run one real 512 CPU slice under 80% before any promotion | **P0 personalized challenger / RESEARCH** |
| **OSDFace** | CVPR 2025; official `jkwang28/OSDFace` | One-step diffusion blind face restoration with visual representation embedder/VQ prior and explicit recognition identity loss | Upstream released inference code + pretrained models in Dec 2025; documented environment is PyTorch 2.4/CUDA 12.1-oriented | Strong severe-blind challenger where no reference evidence exists | CPU/RAM/dependency/Windows feasibility unknown; likely materially heavier than GPEN/GFPGAN | **P1 severe-blind challenger / RESEARCH** |
| **RefineFIR** | WACV 2025; official `RefineFIR/RefineFIR` | **Single-reference fine-detail restoration**, explicit copy-or-not behavior | Paper and official code/data link published | Very aligned with CFS component philosophy: copy wrinkles/moles/eye/nose/mouth detail only when semantically consistent | Runtime/license/checkpoint audit still required | **P0 architectural + benchmark candidate** |
| **RefFaceInpainting** | TCSVT 2023; official `WuyangLuo/RefFaceInpainting` | Large facial occlusion inpainting with separate identity + component-texture control from a reference | Official code/checkpoint; upstream tested PyTorch 1.10.1 on RTX3090; MIT repository license | Highly specialized for sticker/black-bar/large missing facial regions when a good same-person reference exists | Old GPU-oriented stack, dependency age, CPU performance unknown | **P1 occlusion specialist / RESEARCH** |
| **FaceMe** | AAAI 2025; official `modyu-liu/FaceMe` | Personalized diffusion restoration using identity features from one/few/arbitrary refs | Official inference/training code; checkpoint links; RealVisXL/ControlNet-scale stack; Pi-Lab License 1.0 | Strong identity-conditioning comparison for CFS consensus reference profile | Likely too heavy for CPU-only EliteBook, but must be measured rather than assumed | **P2 personalized challenger / RESEARCH** |
| **RestoreFormer++** | TPAMI 2023; official `wzhouxiff/RestoreFormerPlusPlus` | Reconstruction-oriented dictionary + spatial cross-attention | Apache-2.0 code, official checkpoint | Useful non-diffusion severe BFR challenger if newer models cannot meet CPU budget | Transformer/dictionary memory and older research dependencies | **P2 / RESEARCH** |

## New 2025–2026 architecture teachers

| Work | Verified status | Useful concept | CFS decision |
|---|---|---|---|
| **PerFuSe — Personalized Full-Image Restoration via Modular Fusion** (CVPRW 2026) | CVF paper/supplement available. No official implementation/checkpoint was found in the repository audit on 2026-08-18. | Separates personalized and non-personalized regions; multiple generative modules; mask-guided/context-aware fusion; uses a subject photo library; evaluated on real smartphone images | **TEACHER only** until official executable code/checkpoints exist. Extract modular mask/fusion/reference-selection ideas; do not fabricate a runtime. |
| **Reference-Guided Identity Preserving Face Restoration** (2025, Zhou et al.) | Paper available; `cdluminate/RefIPFR` currently contains paper material only, not an executable model release | Composite high/low-level reference context; hard-example identity loss; training-free multi-reference adaptation at inference | **TEACHER / high-priority concept**. Do not schedule model integration until official code/weights are actually released. |
| **BioDDM** (CVPRW 2026) | Paper-level research candidate | Biometric-subspace/identity guidance for diffusion restoration | Extract identity-guidance ideas only; iterative diffusion is not a default CPU target. |
| **NTIRE 2026 face restoration challenge methods** | Challenge evidence, generally unconstrained compute | Modern identity/perceptual evaluation and current failure modes | Use for metric/routing ideas, not as automatic production dependencies. |

## Additional architecture teachers

| Model | Useful idea | Default decision |
|---|---|---|
| DMDNet | Identity-specific + generic component dictionaries | Reuse concept in CFS component/reference memory, not full legacy stack |
| DFDNet | Multi-scale eye/nose/mouth dictionaries and confidence transfer | Teacher |
| ASFFNet | Multi-exemplar feature alignment and adaptive fusion | Teacher |
| ReF-LDM | Flexible multi-reference CacheKV aggregation | Teacher; multi-step LDM too costly by default |
| DifFace / DR2 | Robustness to unknown severe degradation | Teacher/benchmark only |

## Production-oriented ranking by actual CFS damage

This is a **qualification order**, not a claim that every model will ship.

1. **Observed same-person component evidence** — always first for sticker/scribble/mosaic/missing eye/nose/mouth when geometry and identity are valid.
2. **FBCNN** — JPEG/double-JPEG/social recompression pre-clean only when detected.
3. **NAFNet** — mild blur/noise/deblur.
4. **InstantRestore** — first personalized learned challenger after DamageMaskNet/reference-bank routing, because its input contract most closely matches MAIN + several references.
5. **RefineFIR** — fine identity-detail/reference copy-or-not challenger.
6. **RefFaceInpainting** — large occlusion with a strong reference.
7. **GPEN / GFPGAN / CodeFormer** — blind generative candidates when observed reference evidence cannot solve the component.
8. **OSDFace** — severe blind challenger if its one-step diffusion runtime fits <=80% total-PC resources and provides measured quality gain.
9. **FaceMe / RestoreFormer++** — benchmark only if the preceding specialists leave a measurable quality gap.
10. **PerFuSe / RefIPFR / BioDDM** — architecture teachers until an official executable implementation exists and target-PC feasibility can be measured.

## Non-negotiable identity/provenance policy

- SFace frozen safety threshold remains unchanged.
- Wrong-person references have zero observed-pixel and identity-anchor authority.
- Partial references remain component-local.
- Generated candidates are `GENERATED_MODEL_INFERRED`, never `OBSERVED_REFERENCE` or `ORIGINAL_RECOVERED`.
- Healthy MAIN pixels are not rewritten merely because a generator looks sharper.
- Candidate scoring is done from a common checkpoint; heavy generators are not chained destructively.
- Final model selection is calibrated only on DEVELOPMENT/VALIDATION, never final holdout.

## Immediate execution order

1. Complete DamageMaskNet mixed-source DEVELOPMENT vertical slice and ONNX parity.
2. Build the larger 300–400 source development/validation bank and compare lightweight segmentation architectures only if the current U-Net hypothesis fails/underperforms.
3. Connect DamageMaskNet output to the existing per-component reference bank.
4. Benchmark **InstantRestore** as the first learned personalized model under the same 80% governor.
5. Benchmark RefineFIR / RefFaceInpainting on their specialist cases.
6. Re-run GPEN/GFPGAN/CodeFormer/FBCNN over the broader degradation validation matrix through the common adapter.
7. Benchmark OSDFace severe cases only after the personalized/reference-first path is established.
8. Promote only measured winners per degradation family.

Final holdouts remain untouched until the research/validation candidate is frozen.
