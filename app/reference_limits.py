from __future__ import annotations

MAX_PROJECT_IMAGES = 10
MAX_REFERENCE_IMAGES = MAX_PROJECT_IMAGES - 1


def validate_reference_count(count: int) -> int:
    """Validate the product contract: one primary plus at most nine references."""
    value = int(count)
    if value < 0:
        raise ValueError("Numero di riferimenti non valido")
    if value > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Sono supportate al massimo {MAX_PROJECT_IMAGES} immagini totali "
            f"(1 principale + {MAX_REFERENCE_IMAGES} riferimenti)"
        )
    return value
