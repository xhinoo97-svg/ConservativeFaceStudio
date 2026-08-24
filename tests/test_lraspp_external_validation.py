from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _load_builder():
    sys.path.insert(0, str(RESEARCH))
    try:
        path = RESEARCH / "build_controlface_mask_validation_bank.py"
        spec = importlib.util.spec_from_file_location("controlface_external_builder_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(RESEARCH))


def test_controlface_member_parser_extracts_explicit_identity_domains() -> None:
    builder = _load_builder()
    parsed = builder.parse_controlface_member(
        "ControlFace10k/Indian/female/50/identity-abc-123/r3_g0_a50_o5.png"
    )
    assert parsed == ("abc-123", "Indian", "female", "50")
    assert builder.parse_controlface_member("ControlFace10k/no-identity.png") is None


def test_balanced_selection_excludes_prior_identities_and_covers_every_stratum() -> None:
    builder = _load_builder()
    groups = []
    for race in builder.RACES:
        for sex in builder.SEXES:
            for index in range(8):
                identity = f"{race}-{sex}-{index}"
                groups.append(
                    builder.IdentityGroup(
                        identity=identity,
                        race=race,
                        sex=sex,
                        age=("25", "50", "65")[index % 3],
                        members=(f"{identity}.png",),
                    )
                )
    excluded = {"controlface:African-female-0", "controlface:Asian-male-1"}
    selected = builder.select_balanced_identities(
        groups,
        excluded_identity_keys=excluded,
        identities_per_stratum=5,
    )
    assert len(selected) == 40
    assert len({item.identity for item in selected}) == 40
    assert not {f"controlface:{item.identity}" for item in selected} & excluded
    for race in builder.RACES:
        for sex in builder.SEXES:
            stratum = [item for item in selected if item.race == race and item.sex == sex]
            assert len(stratum) == 5
            assert {item.age for item in stratum} == {"25", "50", "65"}


def test_external_evaluator_is_frozen_checkpoint_only() -> None:
    source = (RESEARCH / "evaluate_lraspp_external_validation.py").read_text(encoding="utf-8")
    assert "from_trained_checkpoint" in source
    assert "optimizer" not in source.lower()
    assert "train_epoch" not in source
    assert "development_gate(" in source
    assert '"retrained_or_tuned": False' in source
    assert '"final_holdout_used": False' in source
    assert '"refface_execution_authorized": False' in source


def test_validation_builder_freezes_no_tuning_and_identity_disjoint_contract() -> None:
    source = (RESEARCH / "build_controlface_mask_validation_bank.py").read_text(encoding="utf-8")
    assert '"training_or_tuning_authorized": False' in source
    assert '"identity_disjoint_from_prior_lraspp_bank": True' in source
    assert '"final_holdout_used": False' in source
    assert "selected_keys & excluded" in source


def test_workflow_downloads_exact_prior_artifact_and_cannot_be_manually_rerun() -> None:
    workflow = (ROOT / ".github/workflows/research-lraspp-external-validation.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" not in workflow
    assert "9502642834" in workflow
    assert "0bef114cfeed95ebcceb81ce8f5dfc43c3fdb37bca82c69a346ed6219c137a11" in workflow
    assert "d510e6991cca582c3696b6b9132bf3fdb7948e240f4bf136440d8b75046910f4" in workflow
    assert "708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9" in workflow
    assert "train_damage_mask_lraspp.py" not in workflow
    assert "benchmarks/face-smartphone" not in workflow.lower()
