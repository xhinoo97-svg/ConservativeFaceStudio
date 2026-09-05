from research.phase04_damage_evaluation import (
    GLOBAL_DAMAGE_TYPES,
    POSITIONS,
    SEVERITIES,
    SIZES,
    TRANSLUCENT_OPACITIES,
    build_matrix,
    matrix_payload,
    validate_matrix,
)


def test_phase04_matrix_is_complete_cross_product():
    rows = build_matrix()
    validation = validate_matrix(rows)

    assert validation["passed"] is True
    assert len(rows) == 1036
    assert len({row.case_id for row in rows}) == len(rows)


def test_phase04_local_damage_spans_every_position_size_and_severity():
    rows = build_matrix()
    subset = [row for row in rows if row.damage_type == "SCRIBBLE_THIN_BLACK"]

    assert len(subset) == len(POSITIONS) * len(SIZES) * len(SEVERITIES)
    assert {row.position for row in subset} == set(POSITIONS)
    assert {row.size for row in subset} == set(SIZES)
    assert {row.severity for row in subset} == set(SEVERITIES)


def test_phase04_translucent_sticker_spans_every_opacity_factor():
    rows = build_matrix()
    subset = [row for row in rows if row.damage_type == "TRANSLUCENT_STICKER"]

    assert len(subset) == len(POSITIONS) * len(SIZES) * len(SEVERITIES) * len(TRANSLUCENT_OPACITIES)
    assert {row.opacity for row in subset} == set(TRANSLUCENT_OPACITIES)


def test_phase04_global_damage_does_not_fake_facial_position():
    rows = build_matrix()
    for damage_type in GLOBAL_DAMAGE_TYPES:
        subset = [row for row in rows if row.damage_type == damage_type]
        assert len(subset) == len(SIZES) * len(SEVERITIES)
        assert {row.position for row in subset} == {"GLOBAL"}


def test_phase04_payload_records_matrix_validation_and_holdout_safety():
    payload = matrix_payload()

    assert payload["case_count"] == 1036
    assert payload["coverage"]["cross_factorial"] is True
    assert payload["matrix_validation"]["passed"] is True
    assert payload["v3_used"] is False
    assert payload["v4_used"] is False
    assert payload["final_holdout_used"] is False
