# MODEL_RESEARCH_MATRIX.md

Status date: 2026-08-17

Research branch base: certified V1 `2767513f95dde2d417e7c6f1faf2357149a1a32f`.

`NOT VERIFIED` means no authoritative statement was found in the paper/official repository during this audit. It is intentionally not inferred from third-party ports.

## Qualification rules

- `RESEARCH`: scientifically relevant, not yet run in ConservativeFaceStudio.
- `BENCHMARKING`: official implementation/checkpoint is being measured.
- `QUALIFIED`: real local execution, identity/quality/backend/license checks and target-product gates passed.
- `REJECTED`: measured or legal/dependency evidence makes it unsuitable for production.
- ONNX/OpenVINO support is `NOT VERIFIED` unless supplied officially or converted with parity tests by this project.
- Windows/CPU claims distinguish upstream claims from ConservativeFaceStudio qualification.

## Tier A — required qualification candidates

| Model | Paper / official source | Task / architecture | Params / FLOPs | License audit | Checkpoint | CPU / Windows | ONNX / OpenVINO | Known limitations / identity risk | Target blocks | Priority / state |
|---|---|---|---|---|---|---|---|---|---|---|
| GPEN BFR-512 | CVPR 2021, *GAN Prior Embedded Network for Blind Face Restoration in the Wild*; https://github.com/yangxy/GPEN ; https://arxiv.org/abs/2105.06070 | Blind face restoration; GAN prior embedded directly in restoration network, 512 aligned face path | NOT VERIFIED authoritatively in current audit | **Code/weights license NOT VERIFIED**: repository exposes no clear top-level license in audited file tree; README states released weights are not the authors' best model due to commercial issues | Official `GPEN-BFR-512.pth` linked by upstream README | Upstream explicitly states CPU works without `--use_cuda` and Windows works without compiling CUDA; **CFS Windows/EliteBook not yet measured** | No official ONNX/OpenVINO artifact found; conversion requires parity test | Generative prior can invent identity detail; checkpoint redistribution cannot be qualified until license clarified | 2, 8; candidate source for 9 | **P0 / BENCHMARKING next** |
| GFPGAN v1.4 | CVPR 2021, *Towards Real-World Blind Face Restoration with Generative Facial Prior*; https://github.com/TencentARC/GFPGAN | Blind face restoration with pretrained generative facial prior | NOT VERIFIED | Apache-2.0 for GFPGAN code, subject to listed third-party licenses; weights terms must be recorded with downloaded asset | Official v1.4 model in upstream model zoo; upstream says v1.4 gives slightly more detail and better identity than v1.3 | Clean implementation avoids CUDA extensions; CFS CPU/Windows execution **not yet measured** | No official ONNX/OpenVINO qualification found | Upstream itself notes generative versions can alter identity; must A/B against v1.3 and SFace hard gate | 2, 8; candidate source for 9 | **P1 / RESEARCH** |
| CodeFormer | NeurIPS 2022, *Towards Robust Blind Face Restoration with Codebook Lookup Transformer*; https://github.com/sczhou/CodeFormer | VQ codebook + Transformer; blind restoration, fidelity control, aligned-face inpainting | NOT VERIFIED | S-Lab License 1.0, non-commercial use; checkpoint redistribution/use must comply | Official `codeformer.pth`; official inpainting checkpoint also released | Official inference contains CPU path; CFS Windows/CPU performance not measured | No official ONNX/OpenVINO qualification found; Transformer/VQ conversion must be proven | Fidelity-realness tradeoff; strong prior can hallucinate identity; license limits production scenarios | 2, 8; candidate source for 9 | **P2 / RESEARCH** |
| FBCNN | ICCV 2021, *Towards Flexible Blind JPEG Artifacts Removal*; https://github.com/jiaxi-jiang/FBCNN | Blind JPEG artifact removal; quality-factor predictor + restoration network | NOT VERIFIED | Apache-2.0 | Official color/grayscale pretrained models in upstream | PyTorch implementation; CFS CPU/Windows not yet measured | No official ONNX/OpenVINO qualification found | Specialist only; must not pre-clean healthy/non-JPEG faces | 3 before BFR when JPEG severe | **P1 / RESEARCH** |
| NAFNet | ECCV 2022, *Simple Baselines for Image Restoration*; https://github.com/megvii-research/NAFNet | Activation-free image restoration network; denoise/deblur | Authoritative counts depend on task/config; do not substitute a single number | Upstream license to be re-audited against exact checkpoint already used by CFS | Existing CFS verified OpenCV-Zoo NAFNet ONNX | Already part of V1 production path; exact V1 qualification remains authoritative only for existing checkpoint | Existing ONNX path is already integrated; OpenVINO parity not yet established | General restoration, not an identity-aware facial generator | 2, 3 | **Existing V1 component / preserve** |
| Zero-DCE++ | TPAMI 2021, *Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation*; https://github.com/Li-Chongyi/Zero-DCE_extension | Lightweight zero-reference low-light curve estimation | NOT VERIFIED in audited upstream README | CC BY-NC 4.0 / academic-research-only statement in upstream | Upstream `Epoch99.pth` snapshot | Upstream environment is old PyTorch/CUDA; CFS CPU/Windows not measured | No official ONNX/OpenVINO qualification found | Can shift exposure/colour; route only after real low-light detection | 3 | **P2 / RESEARCH; licensing restricts shipping** |

## Tier B — benchmark, do not automatically ship

| Model | Official source / paper | Core idea | License / deployment evidence | Main risk / expected cost | Target | State |
|---|---|---|---|---|---|---|
| RestoreFormer++ | TPAMI 2023; https://github.com/wzhouxiff/RestoreFormerPlusPlus ; https://arxiv.org/abs/2308.07228 | Reconstruction-oriented HQ dictionary + fully-spatial multi-head cross-attention + extended degradation model | Apache-2.0 code; Linux/GPU-oriented research stack; CPU/Windows/ONNX not qualified | Transformer/dictionary memory and dependency cost; identity must be measured | 2/8 candidate | RESEARCH |
| VQFR v2 | ECCV 2022 Oral; https://github.com/TencentARC/VQFR ; https://arxiv.org/abs/2205.06803 | Vector-quantized detail dictionary + parallel decoder + texture warping | Apache-2.0; upstream lists Linux/GPU as optional but uses research extensions; CFS CPU/Windows not qualified | Custom extension/deformable-conv complexity; quality can outpace fidelity | 2/8 candidate | RESEARCH |
| GPEN face inpainting | https://github.com/yangxy/GPEN | GPEN generative inpainting, upstream 1024 path | Same unresolved GPEN license issue; upstream checkpoint available | 1024 path likely costly on 16 GB CPU; identity-critical hallucination risk | 8 | RESEARCH; only after BFR-512 |
| RefFaceInpainting | TCSVT 2023, *Reference-Guided Large-Scale Face Inpainting with Identity and Texture Control*; https://github.com/WuyangLuo/RefFaceInpainting ; https://arxiv.org/abs/2303.07014 | Dual identity/texture control with Half-AdaIN and component-wise style injection | Tested upstream with PyTorch 1.10.1 / RTX3090; license/checkpoint redistribution NOT VERIFIED in current audit | Old GPU-oriented stack; large-mask generative risk; potentially valuable for personalized severe occlusion | 8 | RESEARCH |
| SwinIR | ICCV Workshops 2021, *SwinIR: Image Restoration Using Swin Transformer*; https://github.com/JingyunLiang/SwinIR | Residual Swin Transformer blocks for SR/denoise/JPEG | Apache-2.0; upstream publishes params/FLOPs for specific SR configs, not assumed for our task | Heavier than FBCNN/NAFNet; only useful if measured quality gain outweighs CPU cost | 3/12 | RESEARCH |

## Teacher / architecture-extraction models

These are architectural teachers first. They are not production dependencies unless a later benchmark explicitly promotes them.

| Model | Primary source | Useful idea to extract | Why not ship by default | State |
|---|---|---|---|---|
| DMDNet | https://github.com/csxmli2016/DMDNet | Dual generic + identity-specific memory dictionaries; component-level matching | CC BY-NC-SA; legacy stack; our existing component/reference bank can implement the idea more cheaply | TEACHER |
| DFDNet | ECCV 2020; https://github.com/csxmli2016/DFDNet | Multi-scale dictionaries for left/right eyes, nose, mouth; confidence-aware dictionary feature transfer | CC BY-NC-SA; generic dictionary is less personalized than our real-reference bank | TEACHER |
| ASFFNet / ASFFNet512 | CVPR 2020; https://github.com/csxmli2016/ASFFNet512 | Multi-exemplar selection, feature alignment/illumination normalization, adaptive spatial feature fusion | CC BY-NC-SA; GPU research implementation; idea fits Block 7/9 without full model | TEACHER |
| RefineFIR | WACV 2025, *Copy or Not? Reference-Based Face Image Restoration with Fine Details*; https://github.com/RefineFIR/RefineFIR | Explicit copy-or-not objective; copy identity-specific fine detail only when semantically consistent | Runtime/license/weights not yet audited; use decision concept first | TEACHER |
| InstantRestore | SIGGRAPH 2025; https://github.com/snap-research/InstantRestore ; https://arxiv.org/abs/2412.06753 | Single-step personalized diffusion, shared-image attention, ~4 refs, landmark-attention identity supervision | Tested upstream in CUDA Docker; diffusion backbone unsuitable for target CPU until proven otherwise | TEACHER |
| FaceMe | AAAI 2025; https://github.com/modyu-liu/FaceMe ; https://arxiv.org/abs/2501.05177 | Consensus/personal identity embeddings from arbitrary reference count as diffusion conditioning | Stable-Diffusion/ControlNet-scale stack; Pi-Lab license; target CPU cost expected high but must be measured before any claim | TEACHER |
| ReF-LDM | NeurIPS 2024; https://github.com/ChiWeiHsiao/ref-ldm ; https://arxiv.org/abs/2412.05043 | CacheKV for flexible multi-reference aggregation; identity-disjoint FFHQ-Ref split | Non-commercial research license; 50-step LDM inference in official demo; too costly for current target unless a lightweight derivative is built | TEACHER |
| OSDFace | CVPR 2025; https://github.com/jkwang28/OSDFace ; https://arxiv.org/abs/2411.17163 | One-step diffusion; visual tokenizer + VQ dictionary + explicit face-recognition identity loss | Official environment is PyTorch 2.4/CUDA 12.1-oriented; CFS CPU/Windows not qualified | TEACHER |
| DifFace | TPAMI 2024; https://github.com/zsyOAOA/DifFace ; https://arxiv.org/abs/2212.06512 | Diffused-error contraction; graceful handling of unseen severe degradation; adjustable fidelity/realness | Iterative diffusion is expensive on target CPU; use robustness concepts/benchmark only | TEACHER |
| DR2 | CVPR 2023, *Diffusion-Based Robust Degradation Remover for Blind Face Restoration*; paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_DR2_Diffusion-Based_Robust_Degradation_Remover_for_Blind_Face_Restoration_CVPR_2023_paper.html ; code referenced by paper ecosystem: https://github.com/Kaldwin0106/DR2_Drgradation_Remover | Convert arbitrary degradation into a degradation-invariant coarse prediction before enhancement | Iterative diffusion cost; official code/license/checkpoint status must be independently verified before any integration | TEACHER |

## Immediate engineering conclusions

1. **Do not install ten models.** The first real vertical slice is GPEN BFR-512 only.
2. The existing CFS component bank and up-to-nine-reference memory already implement the correct product skeleton for personalized per-component routing; extend their scoring rather than replacing them.
3. Paper Quality mode needs a new generated provenance class `GENERATED_MODEL_INFERRED`; existing conservative observed provenance remains authoritative.
4. GPEN BFR-512 can be benchmarked because the upstream explicitly documents CPU and Windows operation, but it cannot become a redistributable `QUALIFIED` production dependency until code/weights licensing is explicitly resolved.
5. GFPGAN v1.4 is the next model only after GPEN produces a real CFS image and actual CPU RAM/time/identity evidence.
6. CodeFormer is technically attractive for severe restoration/inpainting but its S-Lab non-commercial license must remain visible in qualification state.
7. FBCNN is the preferred first JPEG specialist because its task exactly matches single/double/real-world JPEG artifacts and its code is Apache-2.0.
8. Diffusion systems are teacher/benchmark candidates, not default EliteBook dependencies.

## Primary-source audit set

- GPEN: https://github.com/yangxy/GPEN ; https://arxiv.org/abs/2105.06070
- GFPGAN: https://github.com/TencentARC/GFPGAN
- CodeFormer: https://github.com/sczhou/CodeFormer
- FBCNN: https://github.com/jiaxi-jiang/FBCNN
- NAFNet: https://github.com/megvii-research/NAFNet
- Zero-DCE++: https://github.com/Li-Chongyi/Zero-DCE_extension
- RestoreFormer++: https://github.com/wzhouxiff/RestoreFormerPlusPlus
- VQFR: https://github.com/TencentARC/VQFR
- RefFaceInpainting: https://github.com/WuyangLuo/RefFaceInpainting
- SwinIR: https://github.com/JingyunLiang/SwinIR
- DMDNet: https://github.com/csxmli2016/DMDNet
- DFDNet: https://github.com/csxmli2016/DFDNet
- ASFFNet512: https://github.com/csxmli2016/ASFFNet512
- RefineFIR paper: https://openaccess.thecvf.com/content/WACV2025/html/Chong_Copy_or_Not_Reference-Based_Face_Image_Restoration_with_Fine_Details_WACV_2025_paper.html
- InstantRestore: https://github.com/snap-research/InstantRestore
- FaceMe: https://github.com/modyu-liu/FaceMe
- ReF-LDM: https://github.com/ChiWeiHsiao/ref-ldm
- OSDFace: https://github.com/jkwang28/OSDFace
- DifFace: https://github.com/zsyOAOA/DifFace
- DR2 paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_DR2_Diffusion-Based_Robust_Degradation_Remover_for_Blind_Face_Restoration_CVPR_2023_paper.html
