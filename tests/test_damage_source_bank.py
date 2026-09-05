from __future__ import annotations

from collections import defaultdict

from research.build_damage_source_bank import _identity_from_controlface_path, _pick_identities


def test_controlface_path_parser_uses_declared_sex_and_identity_directory() -> None:
    assert _identity_from_controlface_path("ControlFace10K/asian/female/young/identity-123e4567-e89b/face_r1_g2_o3.png") == ("female", "123e4567-e89b")
    assert _identity_from_controlface_path("ControlFace10K/black/male/adult/identity-xyz/0001.jpg") == ("male", "xyz")


def test_controlface_path_parser_rejects_incomplete_paths() -> None:
    assert _identity_from_controlface_path("female/no-identity/image.png") is None
    assert _identity_from_controlface_path("identity-abc/no-sex/image.png") is None
    assert _identity_from_controlface_path("female/identity-/image.png") is None


def test_pick_identities_is_deterministic_disjoint_and_female_heavy() -> None:
    groups = {
        "female": defaultdict(list, {f"f-{index:03d}": [f"female/{index}.png"] for index in range(20)}),
        "male": defaultdict(list, {f"m-{index:03d}": [f"male/{index}.png"] for index in range(10)}),
    }
    train_a, val_a = _pick_identities(groups)
    train_b, val_b = _pick_identities(groups)
    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 14
    assert len(val_a) == 2
    assert sum(sex == "female" for sex, _ in train_a) == 10
    assert sum(sex == "male" for sex, _ in train_a) == 4
    assert all(sex == "female" for sex, _ in val_a)
    assert {identity for _, identity in train_a}.isdisjoint({identity for _, identity in val_a})
