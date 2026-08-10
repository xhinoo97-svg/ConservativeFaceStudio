from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BlockKind(str, Enum):
    IMPORT = "import"
    DEBLUR = "deblur"
    ENHANCE = "enhance"
    LANDMARKS = "landmarks"
    ALIGN = "align"
    OCCLUSION_MASK = "occlusion_mask"
    REGION_SELECT = "region_select"
    INPAINT = "inpaint"
    FUSION = "fusion"
    FRONTALIZE = "frontalize"
    IDENTITY_CHECK = "identity_check"
    UPSCALE = "upscale"
    EXPORT = "export"


@dataclass(frozen=True)
class BlockSpec:
    key: str
    title: str
    kind: BlockKind
    required_modules: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    optional: bool = False
    generative: bool = False


@dataclass
class PipelineState:
    blocks: tuple[BlockSpec, ...]
    current_index: int = 0
    accepted: set[str] = field(default_factory=set)
    skipped: set[str] = field(default_factory=set)
    history: list[str] = field(default_factory=list)
    redo_stack: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def current(self) -> BlockSpec:
        return self.blocks[self.current_index]

    @property
    def complete(self) -> bool:
        return self.current_index >= len(self.blocks) - 1 and self.current.key in self.accepted

    def accept_current(self) -> None:
        key = self.current.key
        self.accepted.add(key)
        self.skipped.discard(key)
        self.history.append(key)
        self.redo_stack.clear()

    def skip_current(self) -> None:
        if not self.current.optional:
            raise ValueError(f"Il blocco obbligatorio non può essere saltato: {self.current.key}")
        key = self.current.key
        self.skipped.add(key)
        self.accepted.discard(key)
        self.history.append(key)
        self.redo_stack.clear()

    def advance(self) -> BlockSpec:
        key = self.current.key
        if key not in self.accepted and key not in self.skipped:
            raise RuntimeError(f"Accetta o salta il blocco prima di continuare: {key}")
        if self.current_index < len(self.blocks) - 1:
            self.current_index += 1
        return self.current

    def retreat(self) -> BlockSpec:
        if self.current_index > 0:
            self.current_index -= 1
        return self.current

    def undo_last_decision(self) -> BlockSpec:
        if not self.history:
            raise RuntimeError("Nessuna decisione da annullare")
        key = self.history.pop()
        index = next(index for index, block in enumerate(self.blocks) if block.key == key)
        if key in self.accepted:
            decision = "accepted"
            self.accepted.remove(key)
        elif key in self.skipped:
            decision = "skipped"
            self.skipped.remove(key)
        else:
            raise RuntimeError(f"Cronologia incoerente per il blocco: {key}")
        self.redo_stack.append((key, decision, self.current_index))
        self.current_index = index
        return self.current

    def redo_last_decision(self) -> BlockSpec:
        if not self.redo_stack:
            raise RuntimeError("Nessuna decisione da ripristinare")
        key, decision, previous_index = self.redo_stack.pop()
        index = next(index for index, block in enumerate(self.blocks) if block.key == key)
        self.current_index = index
        if decision == "accepted":
            self.accepted.add(key)
            self.skipped.discard(key)
        elif decision == "skipped":
            if not self.current.optional:
                raise RuntimeError(f"Decisione di salto non valida per: {key}")
            self.skipped.add(key)
            self.accepted.discard(key)
        else:
            raise RuntimeError(f"Decisione sconosciuta: {decision}")
        self.history.append(key)
        self.current_index = min(previous_index, len(self.blocks) - 1)
        return self.current

    def reset_from_current(self) -> None:
        later = {block.key for block in self.blocks[self.current_index :]}
        self.accepted.difference_update(later)
        self.skipped.difference_update(later)
        self.history = [key for key in self.history if key not in later]
        self.redo_stack.clear()


def default_pipeline() -> tuple[BlockSpec, ...]:
    return (
        BlockSpec("import", "Importa foto e riferimenti", BlockKind.IMPORT),
        BlockSpec("deblur", "Deblur, denoise e nitidezza conservativa", BlockKind.DEBLUR, depends_on=("import",)),
        BlockSpec("enhance", "Contrasto locale e recupero qualità", BlockKind.ENHANCE, depends_on=("deblur",)),
        BlockSpec("landmarks", "Rilevamento volto e landmark", BlockKind.LANDMARKS, required_modules=("face_detection", "landmarks", "identity_guardrail"), depends_on=("enhance",)),
        BlockSpec("align", "Allineamento e confronto multi-foto", BlockKind.ALIGN, required_modules=("landmarks", "reference_alignment"), depends_on=("landmarks",)),
        BlockSpec("occlusion_mask", "Rilevamento coperture e consenso multi-foto", BlockKind.OCCLUSION_MASK, required_modules=("occlusion_detection", "face_parsing"), depends_on=("align",)),
        BlockSpec("region_select", "Selezione delle regioni migliori", BlockKind.REGION_SELECT, required_modules=("component_bank", "reference_fusion"), depends_on=("align", "occlusion_mask")),
        # Core block: it is mandatory. With no damage it must return a successful
        # no-op/abstention; with damage it must use observed donors first and then a
        # verified residual fallback instead of being silently skipped.
        BlockSpec("inpaint", "Riparazione coperture da foto reali e residuo verificato", BlockKind.INPAINT, required_modules=("reference_fusion", "inpainting_fallback"), depends_on=("occlusion_mask", "region_select"), optional=False),
        BlockSpec("fusion", "Fusione conservativa delle regioni", BlockKind.FUSION, required_modules=("reference_fusion", "provenance"), depends_on=("region_select", "inpaint")),
        BlockSpec("frontalize", "Normalizzazione posa 2D senza sintesi", BlockKind.FRONTALIZE, depends_on=("fusion",), optional=True),
        BlockSpec("identity_check", "Controllo identità e coerenza", BlockKind.IDENTITY_CHECK, required_modules=("identity_guardrail",), depends_on=("fusion",)),
        BlockSpec("upscale", "Upscale finale", BlockKind.UPSCALE, depends_on=("identity_check",), optional=True),
        BlockSpec("export", "Esporta risultato e report", BlockKind.EXPORT, depends_on=("identity_check",)),
    )


def validate_pipeline(blocks: tuple[BlockSpec, ...]) -> None:
    if not blocks:
        raise ValueError("La pipeline non può essere vuota")
    keys = [block.key for block in blocks]
    if len(keys) != len(set(keys)):
        raise ValueError("La pipeline contiene chiavi duplicate")

    positions = {key: index for index, key in enumerate(keys)}
    for block in blocks:
        for dependency in block.depends_on:
            if dependency not in positions:
                raise ValueError(f"Dipendenza mancante per {block.key}: {dependency}")
            if positions[dependency] >= positions[block.key]:
                raise ValueError(f"Dipendenza fuori ordine per {block.key}: {dependency}")

    if blocks[0].kind is not BlockKind.IMPORT:
        raise ValueError("Il primo blocco deve importare le immagini")
    if blocks[-1].kind is not BlockKind.EXPORT:
        raise ValueError("L'ultimo blocco deve esportare il risultato")
