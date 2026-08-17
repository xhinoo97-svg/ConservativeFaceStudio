# SPECIALIST_MODEL_ROUTING_V2.md

Status: 2026-08-17
Branch: `research/paper-quality-local-v2`

## Non-negotiable execution budget

Paper Quality mode uses `app/resource_budget.py` with a maximum total-PC resource fraction of **0.80**.

- CPU affinity: at most `floor(logical_processors * 0.80)` logical processors.
- Heavy models: **exactly one at a time**.
- Physical-RAM process ceiling: `0.80 * total_physical_RAM` when the OS can report it.
- Every heavy model route performs pre-load, post-load, post-inference and post-unload checks.
- OpenMP, MKL, OpenBLAS, NumExpr, OpenCV and PyTorch thread counts are capped.
- No quality gate may be weakened to stay inside the budget. If a model cannot qualify under the budget, it is rejected for the target PC.

For the intended 16 GiB machine, the nominal process ceiling is approximately 12.8 GiB, leaving at least 20% physical RAM outside the application budget.

## Selection philosophy

Use the most specialized model that wins on the actual degradation family. Do not run every model and do not chain generative restorers destructively.

Every heavy candidate starts from a common 512 aligned checkpoint and returns a `RestorationCandidate`. Identity remains a hard gate before perceptual ranking.

## Primary specialist map

| Damage / task | Primary specialist to qualify | Secondary / challenger | Why |
|---|---|---|---|
| Mild deblur / denoise | Existing NAFNet | task-specific NAFNet checkpoint only if benchmarked | Lightweight, already compatible with the local architecture; preserve for mild cases. |
| Severe blind facial degradation | **OSDFace** research challenger | GPEN BFR-512, GFPGAN v1.4, CodeFormer | OSDFace is face-specific one-step diffusion (CVPR 2025) and is scientifically stronger than treating a general diffusion model as the default. It must still prove CPU/RAM/Windows feasibility under the 80% cap before promotion. |
| Fast severe facial restoration | **GPEN BFR-512** | GFPGAN v1.4 | GPEN already produced a real CPU result with strong SFace identity on the first development case; license/Windows/target-hardware gates remain open. |
| Fidelity-biased blind facial restoration | **GFPGAN v1.4** | CodeFormer high-fidelity setting | GFPGAN v1.4 produced the best PSNR/SSIM of the first GPEN/GFPGAN A/B case and lower measured peak RSS than GPEN. More cases are required. |
| Controllable quality/fidelity restoration | **CodeFormer** | GFPGAN v1.4 | Explicit fidelity control and aligned-face restoration; also provides a dedicated official face-inpainting model. License constrains production qualification. |
| JPEG / double-JPEG / social recompression | **FBCNN** | NAFNet/SwinIR only if validation wins | FBCNN is directly specialized for blind JPEG artifact removal; route it only when JPEG severity warrants pre-cleaning. |
| Genuine low light | **Zero-DCE++** research specialist | a later lightweight low-light challenger if licensing blocks shipping | Extremely lightweight and task-specific; non-commercial licensing must remain a production blocker unless resolved. |
| Sticker / scribble / opaque block with usable same-person evidence | **Observed per-component reference reconstruction** | RefineFIR-inspired copy-or-not scoring | Real same-person evidence outranks any generator. Selection occurs independently per eye/brow/nose/philtrum/mouth/etc. |
| Large facial occlusion with one strong same-person reference | **RefFaceInpainting** research challenger | CodeFormer inpainting | Reference-guided face inpainting is specialized for large missing regions with identity/texture control. Must prove CPU/Windows/weight availability. |
| Severe degradation with multiple same-person references | **InstantRestore** research challenger | RefineFIR concepts + component bank | InstantRestore is explicitly personalized and uses a small set of references with shared-image attention. It is evaluated as a one-step personalized route, not assumed deployable on the EliteBook. |
| Fine identity detail from references | **RefineFIR-inspired copy-or-not component route** | current component bank | Prefer lightweight extraction of the paper's reference-copy decision logic before considering its full research stack. |
| Damage classification / localization | **DamageMaskNet trained for CFS** | face parsing + deterministic heuristics | Exact synthetic corruption masks allow a target-domain model for scribble/sticker/mosaic/black-bar/blur/JPEG classes. |
| Identity verification | **Frozen SFace gate** | ArcFace/InsightFace as independent secondary evaluator after calibration | SFace threshold is not lowered. Wrong-person and partial references cannot become global anchors. |
| Face-only x2 upscale | **Real-ESRGAN x2 only after restoration** | SwinIR if it wins target-hardware validation | Upscale is post-restoration and bounded to avoid spending CPU/RAM on healthy background by default. |

## Important distinction: scientific best vs target-machine best

A newer research model is not automatically the production winner.

For example, OSDFace is a face-specialized one-step diffusion model and InstantRestore is personalized, but both remain `RESEARCH` until they satisfy:

1. official source and checkpoint provenance;
2. code/weights license audit;
3. real CPU execution;
4. Windows execution;
5. <=80% total-PC resource budget;
6. no OOM / leak;
7. identity hard gate;
8. measured improvement on the relevant degradation family;
9. acceptable seconds-per-512-face on the real HP EliteBook 1030 G3.

If they fail those gates, the target-machine best can remain GPEN/GFPGAN/CodeFormer or a reference-first component route.

## Sequential heavy-model policy

For one damaged face/component:

`CHECKPOINT -> specialist A -> score -> unload -> specialist B only if required -> score -> unload -> select/fuse`

Never:

`GPEN -> GFPGAN -> CodeFormer -> diffusion`

The latter compounds hallucination and violates the common-checkpoint comparison design.

## Per-component preference for the real use case

The user normally provides MAIN + 0..9 references. For identity-critical damage the router preference is:

1. same-person observed component with adequate geometry/quality;
2. component-bank reconstruction;
3. personalized reference model if qualified;
4. blind face prior candidate;
5. explicit unresolved/abstain in Conservative mode.

A valid Paper Quality output can therefore use different sources/models for left eye, right eye, nose and mouth, provided identity, seams, geometry and provenance all pass.

## Qualification order from current state

1. Complete CodeFormer `w=0.5` vertical slice under the 80% resource budget.
2. Build common `FaceRestorerAdapter` with mandatory resource-budget hooks.
3. Re-run GPEN/GFPGAN/CodeFormer through the common adapter under the same 80% cap.
4. Integrate FBCNN as the first true degradation specialist.
5. Build DamageMaskNet dataset/training/inference path.
6. Connect personalized per-component reference routing.
7. Benchmark RefFaceInpainting for large occlusion.
8. Benchmark OSDFace as severe blind-restoration challenger.
9. Benchmark InstantRestore on the actual multi-reference protocol.
10. Promote only measured winners; reject models that do not justify RAM/time/dependency cost.

Final holdouts remain untouched until the research/validation candidate is frozen.
