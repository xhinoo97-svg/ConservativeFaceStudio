from __future__ import annotations

_INSTALLED = False


def install_automatic_integrity_policy() -> None:
    """Prevent mandatory block failures from being hidden as successful autoruns."""
    global _INSTALLED
    if _INSTALLED:
        return

    from app.automatic import AutomaticPipelineRunner
    from app.pipeline import BlockKind

    original_run = AutomaticPipelineRunner.run

    def validated_run(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        workspace = self.executor.workspace
        block_by_key = {block.key: block for block in self.executor.pipeline.blocks}
        unexpected: list[str] = []
        for item in result.results:
            if not bool(item.details.get("skipped", False)):
                continue
            block = block_by_key.get(item.block)
            if block is None:
                unexpected.append(f"{item.block}: blocco sconosciuto")
                continue
            if block.optional:
                continue
            if (
                not workspace.references
                and block.kind in {BlockKind.ALIGN, BlockKind.REGION_SELECT, BlockKind.FUSION}
            ):
                continue
            unexpected.append(f"{block.key}: {item.details.get('reason', 'errore non specificato')}")
        if unexpected:
            raise RuntimeError(
                "Automatic pipeline incompleta: blocchi obbligatori saltati: " + "; ".join(unexpected)
            )
        return result

    AutomaticPipelineRunner.run = validated_run
    _INSTALLED = True
