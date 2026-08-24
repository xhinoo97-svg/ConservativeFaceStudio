from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "config" / "final-integration-inventory.json"


def _payload() -> dict[str, object]:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_final_integration_inventory_is_bound_to_immutable_main() -> None:
    payload = _payload()
    assert payload["branch"] == "integration/final-paper-quality-local"
    assert payload["base_sha"] == "2767513f95dde2d417e7c6f1faf2357149a1a32f"
    assert payload["base_tree_sha"] == "d444f3191b58f3213263a40480bd8e861a903b72"
    remote = payload["remote_state"]
    assert remote["v3"] == "CONSUMED"
    assert remote["v4"] == "CONSUMED_FAIL_0_OF_40"
    assert remote["v5"] == "NOT_CREATED_NOT_AUTHORIZED"


def test_final_integration_inventory_does_not_promote_unverified_capabilities() -> None:
    payload = _payload()
    blocks = payload["pipeline_blocks"]
    assert [item["key"] for item in blocks] == [
        "import",
        "deblur",
        "enhance",
        "landmarks",
        "align",
        "occlusion_mask",
        "region_select",
        "inpaint",
        "fusion",
        "frontalize",
        "identity_check",
        "upscale",
        "export",
    ]
    assert len(payload["production_models"]) == 6
    assert all(item["weight_in_git"] is False for item in payload["production_models"])
    blockers = set(payload["known_blockers"])
    assert "ENHANCE_AUTOMATIC_BLEND_ZERO_NOOP" in blockers
    assert "NO_MULTICLASS_DAMAGE_ROUTER_IN_BASE" in blockers
    assert "TARGET95_NOT_MEASURED" in blockers
    assert payload["classification"].endswith("NOT_YET_MEASURED")
