from app.pretrained_face_handlers import reliability_aware_identity_flags


def _candidate(source: int, accepted: bool):
    return {"source_index": source, "accepted_identity": accepted}


def test_cluster_may_rescue_same_person_only_with_main_and_direct_bridge() -> None:
    flags, reasons = reliability_aware_identity_flags(
        [0.999, 0.16, -0.10, 0.25],
        [_candidate(0, True), _candidate(1, True), _candidate(2, True), _candidate(3, False), _candidate(4, True)],
        [0, 1, 2, 3, 4],
    )
    assert flags == [True, True, False, True]
    assert reasons[1] == "main_bridged_cross_reference_cluster"
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
