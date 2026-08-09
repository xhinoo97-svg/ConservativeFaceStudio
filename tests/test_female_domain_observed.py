from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts import run_female_domain_benchmark_observed as observed


class _FakeBackend:
    def analyze(self, image):
        return SimpleNamespace(
            landmarks5=np.array(
                [[90.0, 95.0], [165.0, 92.0], [128.0, 130.0], [105.0, 165.0], [151.0, 164.0]],
                dtype=np.float32,
            )
        )


def test_curated_domain_is_broad_not_nasa_only():
    assert len(observed.CURATED_FEMALE_DOMAIN) >= 30
    notes = " ".join(item.domain_note.lower() for item in observed.CURATED_FEMALE_DOMAIN[:30])
    assert "vintage" in notes
    assert "19th-century" in notes
    assert "eyewear" in notes
    assert sum("nasa" in item.search_query.lower() for item in observed.CURATED_FEMALE_DOMAIN[:30]) < 15


def test_component_damage_is_exact_union_of_observed_partial_references(monkeypatch):
    monkeypatch.setattr(observed, "_LANDMARK_BACKEND", _FakeBackend())
    clean = np.full((256, 256, 3), 127, dtype=np.uint8)
    fallback = observed.Scenario(
        "component_only_references",
        clean.copy(),
        (clean.copy(),),
        np.zeros((256, 256), dtype=np.uint8),
        True,
    )
    scenario = observed._observed_component_scenario(clean, fallback)
    assert scenario is not fallback
    assert len(scenario.references) == 3
    support = np.any(np.stack([np.any(ref != 0, axis=2) for ref in scenario.references]), axis=0)
    damage = scenario.damage_mask > 0
    assert np.array_equal(support, damage)
    assert np.all(scenario.primary[damage] == 18)
    assert np.array_equal(scenario.primary[~damage], clean[~damage])
