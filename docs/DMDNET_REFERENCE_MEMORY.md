# DMDNet-inspired specific reference memory

Reference: Xiaoming Li et al., *Learning Dual Memory Dictionaries for Blind Face Restoration* (TPAMI 2022 / arXiv:2210.08160), official implementation: https://github.com/csxmli2016/DMDNet.

## Why it is relevant

DMDNet separates a generic dictionary from a specific dictionary built from multiple high-quality images of the same identity. Its strongest idea for ConservativeFaceStudio is the specific-memory path: all available same-identity references can contribute component information instead of selecting only one exemplar. The paper also uses multi-scale dictionary transforms and confidence-weighted fusion.

## Strict-mode adaptation

ConservativeFaceStudio does **not** copy the DMDNet reconstruction network into strict mode. `app/reference_memory.py` independently adapts the architecture idea with these constraints:

- no generic dictionary from other identities;
- no GAN/perceptual prior in strict mode;
- all replacement pixels come from user-provided same-identity photographs;
- references are ranked per facial component at multiple scales;
- top references must agree before transfer;
- local reference quality must exceed the primary image;
- replacement area is capped per region;
- the generic face remainder has a much smaller replacement cap than eyes/nose/mouth;
- every transferred pixel keeps a source index in the provenance map;
- every pipeline block is still protected by the identity guardrail and can be rolled back.

Current strict component memory uses left eye, right eye, nose, mouth, and a conservative remainder-of-face region. This intentionally mirrors the component-oriented spirit of DMDNet while retaining exact observed-pixel provenance.

## Upstream failure modes accounted for

The official DMDNet README and issue tracker report sensitivity to landmark/component localization and practical dependence on 512x512-oriented processing. ConservativeFaceStudio therefore requires valid face geometry before strict memory transfer and abstains when landmarks/bounding boxes are unavailable. Input photographs are not globally stretched to 512x512 for this strict algorithm.

## Optional original DMDNet backend

The official `DMDNet.pth` checkpoint is registered as an optional research backend in `app/model_registry.py`, but is disabled by default and is not bundled into strict processing. It can later be used for controlled A/B evaluation against the observed-pixel memory engine.

The upstream project declares CC BY-NC-SA 4.0 and the CelebRef-HQ dataset has additional non-commercial research restrictions. Keep upstream code, checkpoint, and dataset handling separate from the core strict implementation and preserve attribution/terms when used.
