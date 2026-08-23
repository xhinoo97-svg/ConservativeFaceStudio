from __future__ import annotations

import pytest

from scripts.face_smartphone_abstention import is_identity_safety_failure


@pytest.mark.parametrize(
    "message",
    [
        "Controllo identità V4 senza evidenza strutturata SFace",
        "Controllo identità V4 senza confronti SFace utilizzabili",
        "Controllo identità senza confronto SFace reale: il fallback proxy non è autorità V4",
    ],
)
def test_v4_fail_closed_identity_messages_are_safety_failures(message: str) -> None:
    assert is_identity_safety_failure(message) is True


def test_unrelated_runtime_failure_is_not_identity_abstention_marker() -> None:
    assert is_identity_safety_failure("unexpected tensor shape") is False
    assert is_identity_safety_failure("CUDA allocation failure") is False
