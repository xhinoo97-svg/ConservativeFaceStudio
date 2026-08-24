from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.component_bank import ComponentCoverage
from app.personalized_reference_bank import (
    COMPONENTS,
    ComponentCandidate,
    PersonalizedReferenceBank,
)


@dataclass(frozen=True)
class PersonalizedComponentSelection:
    component: str
    selected_source_indices: tuple[int, ...]
    candidates: tuple[ComponentCandidate, ...]
    observed_coverage_by_source: Mapping[int, float]

    @property
    def best_source_index(self) -> int | None:
        return self.selected_source_indices[0] if self.selected_source_indices else None


def _coverage_map(values: Sequence[ComponentCoverage]) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in values:
        if item.component not in COMPONENTS:
            continue
        source = int(item.source_index)
        coverage = float(item.coverage)
        if source <= 0 or not item.usable or coverage <= 0.0:
            continue
        result[source] = max(result.get(source, 0.0), coverage)
    return result


def select_personalized_components(
    bank: PersonalizedReferenceBank,
    component_bank: Mapping[str, Sequence[ComponentCoverage]],
    *,
    max_sources_per_component: int = 3,
) -> dict[str, PersonalizedComponentSelection]:
    """Intersect identity-safe ranking with actually observed component support.

    The existing component bank remains the geometric/evidence authority. The
    Personalized Reference Bank supplies identity-safe quality ordering. A source must
    appear in both systems for the same component. This function never invents missing
    component support and never upgrades a partial reference to a global anchor.
    """
    limit = int(max_sources_per_component)
    if not 1 <= limit <= 9:
        raise ValueError("max_sources_per_component must be in 1..9")

    unknown = set(component_bank) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown component-bank keys: {sorted(unknown)}")

    selections: dict[str, PersonalizedComponentSelection] = {}
    for component in COMPONENTS:
        observed = _coverage_map(component_bank.get(component, ()))
        ranked = tuple(
            candidate
            for candidate in bank.ranked(component)
            if candidate.source_index in observed
        )
        # `bank.ranked` is already deterministically ordered by quality score then
        # original source index. Geometric coverage is a hard eligibility gate here,
        # not double-counted as another arbitrary ranking term.
        chosen = tuple(candidate.source_index for candidate in ranked[:limit])
        selections[component] = PersonalizedComponentSelection(
            component=component,
            selected_source_indices=chosen,
            candidates=ranked,
            observed_coverage_by_source={
                source: observed[source]
                for source in chosen
            },
        )
    return selections
