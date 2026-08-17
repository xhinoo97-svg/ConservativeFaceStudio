from __future__ import annotations

from scripts.discover_face_smartphone_v4_sources import (
    DATASET_REVISION,
    FEMALE_IDENTITY_COUNT,
    FEMALE_RACE_QUOTAS,
    LICENSE_NAME,
    RACES,
    TOTAL_IDENTITIES,
    _parse_member,
)


def test_controlface_v4_contract_constants_are_frozen() -> None:
    assert DATASET_REVISION == "a03589de1a9e028b2d16fa1eb0e019a6930e817c"
    assert LICENSE_NAME == "CC BY 4.0"
    assert FEMALE_IDENTITY_COUNT == 19
    assert TOTAL_IDENTITIES == 20
    assert sum(FEMALE_RACE_QUOTAS.values()) == 19
    assert tuple(FEMALE_RACE_QUOTAS) == RACES


def test_parse_female_controlface_member_from_documented_hierarchy() -> None:
    parsed = _parse_member(
        "controlface/African/female/24/identity-014aecae-0dcc-4141-90ea-9c1476226341/"
        "r0_g0_a24_o0_cabc123.png"
    )
    assert parsed is not None
    assert parsed["race"] == "African"
    assert parsed["gender"] == "female"
    assert parsed["identity"] == "identity-014aecae-0dcc-4141-90ea-9c1476226341"
    assert parsed["orientation"] == 0


def test_parse_male_controlface_member_from_documented_hierarchy() -> None:
    parsed = _parse_member(
        "controlface/Indian/male/51/identity-12345678-1234-1234-1234-123456789abc/"
        "r3_g1_a51_o2_cff09aa.png"
    )
    assert parsed is not None
    assert parsed["race"] == "Indian"
    assert parsed["gender"] == "male"
    assert parsed["orientation"] == 2


def test_parser_rejects_non_image_or_missing_identity() -> None:
    assert _parse_member("controlface/African/female/24/README.txt") is None
    assert _parse_member("controlface/African/female/24/r0_g0_a24_o0_cabc123.png") is None


def test_parser_fails_closed_on_gender_or_race_mismatch() -> None:
    try:
        _parse_member(
            "controlface/African/female/24/identity-x/"
            "r0_g1_a24_o0_cabc123.png"
        )
    except RuntimeError as exc:
        assert "gender" in str(exc).lower()
    else:
        raise AssertionError("gender mismatch must fail closed")

    try:
        _parse_member(
            "controlface/Asian/female/24/identity-x/"
            "r0_g0_a24_o0_cabc123.png"
        )
    except RuntimeError as exc:
        assert "race" in str(exc).lower()
    else:
        raise AssertionError("race mismatch must fail closed")
