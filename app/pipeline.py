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

    def skip_current(self) -> None:
        if not self.current.optional:
            raise ValueError(f"Il blocco obbligatorio non può essere saltato: {self.current.key}")
        key = self.current.key
        self.skipped.add(key)
        self.accepted.discard(key)
        self.history.append(key)

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

    def reset_from_current(self) -> None:
        later = {block.key for block in self.blocks[self.current_index :]}
        self.accepted.difference_update(later)
        self.skipped.difference_update(later)
        self.history = [key for key in self.history if key not in later]


def default_pipeline() -> tuple[BlockSpec, ...]:
    return (
        BlockSpec("import", "Importa foto e riferimenti", BlockKind.IMPORT),
        BlockSpec("deblur", "Deblur, denoise e nitidezza conservativa", BlockKind.DEBLUR, depends_on=("import",)),
        BlockSpec("enhance", "Contrasto locale e recupero qualità", BlockKind.ENHANCE, depends_on=("deblur",)),
        BlockSpec("landmarks", "Rilevamento volto e landmark", BlockKind.LANDMARKS, required_modules=("landmarks", "insightface"), depends_on=("enhance",)),
        BlockSpec("align", "Allineamento e confronto multi-foto", BlockKind.ALIGN, required_modules=("landmarks", "insightface"), depends_on=("landmarks",)),
        BlockSpec("occlusion_mask", "Rilevamento coperture e maschera", BlockKind.OCCLUSION_MASK, required_modules=("face_parsing",), depends_on=("align",)),
        BlockSpec("region_select", "Selezione delle regioni migliori", BlockKind.REGION_SELECT, required_modules=("reference_fusion", "insightface"), depends_on=("align", "occlusion_mask")),
        BlockSpec("inpaint", "Rimozione coperture", BlockKind.INPAINT, required_modules=("lama",), depends_on=("occlusion_mask",), optional=True, generative=True),
        BlockSpec("fusion", "Fusione conservativa delle regioni", BlockKind.FUSION, required_modules=("reference_fusion",), depends_on=("region_select",)),
        BlockSpec("frontalize", "Frontalizzazione 3D", BlockKind.FRONTALIZE, required_modules=("3ddfa",), depends_on=("fusion",), optional=True),
        BlockSpec("identity_check", "Controllo identità e coerenza", BlockKind.IDENTITY_CHECK, required_modules=("insightface",), depends_on=("fusion",)),
        BlockSpec("upscale", "Upscale finale", BlockKind.UPSCALE, required_modules=("realesrgan",), depends_on=("identity_check",), optional=True),
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
