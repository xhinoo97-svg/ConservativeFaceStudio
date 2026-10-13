import math

import pytest

from app.pretrained_face_handlers import reliability_aware_identity_flags


def _candidate(source: int, accepted: bool):
    return {"source_index": source, "accepted_identity": accepted}


def test_cluster_cannot_authorize_references_below_direct_sface_threshold() -> None:
    flags, reasons = reliability_aware_identity_flags(
        [0.999, 0.16, -0.10, 0.25],
        [_candidate(0, True), _candidate(1, True), _candidate(2, True), _candidate(3, False), _candidate(4, True)],
        [0, 1, 2, 3, 4],
    )
    assert flags == [True, False, False, False]
    assert reasons[1] == "rejected"
    assert reasons[2] == "rejected"


def test_reference_only_cluster_can_never_override_damaged_main() -> None:
    flags, reasons = reliability_aware_identity_flags(
        [0.10, 0.12, 0.11],
        [_candidate(0, False), _candidate(1, True), _candidate(2, True), _candidate(3, True)],
        [0, 1, 2, 3],
    )
    assert flags == [False, False, False]
    assert reasons == ["rejected", "rejected", "rejected"]


def test_wrong_person_outside_bridged_cluster_stays_rejected() -> None:
    flags, _ = reliability_aware_identity_flags(
        [0.99, -0.2],
        [_candidate(0, True), _candidate(1, True), _candidate(2, False)],
        [0, 1, 2],
    )
    assert flags == [True, False]


@pytest.mark.parametrize("score", [None, math.nan, math.inf, -math.inf, -1.1, 1.1, 0.362999])
def test_invalid_or_below_threshold_score_has_no_donor_authority(score) -> None:
    flags, reasons = reliability_aware_identity_flags(
        [0.99, score],
        [_candidate(0, True), _candidate(1, True), _candidate(2, True)],
        [0, 1, 2],
    )
    assert flags == [True, False]
    assert reasons == ["direct_sface", "rejected"]
    assert all(type(flag) is bool for flag in flags)


def test_direct_threshold_is_inclusive_and_independent_of_cluster_metadata() -> None:
    for candidates, order in [(None, []), ([{"source_index": "invalid"}], [7, 2, 1])]:
        flags, reasons = reliability_aware_identity_flags([0.363, 1.0], candidates, order)
        assert flags == [True, True]
        assert reasons == ["direct_sface", "direct_sface"]
