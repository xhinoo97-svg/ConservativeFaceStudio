from __future__ import annotations

"""Uniform adapter for frozen benchmark case, contract, and freeze providers.

Frozen benchmark modules are immutable evidence providers.  Older providers build
their freeze from the case payload and read the contract file internally, while
newer providers require the already-loaded contract payload explicitly.  The
generic runner must support both without retrying a failed call or interpreting a
``TypeError`` raised inside provider code as a signature mismatch.
"""

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol


class FrozenBenchmarkProvider(Protocol):
    BENCHMARK_ROOT: Path

    def build_cases(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FrozenBenchmarkPayloads:
    cases_payload: dict[str, Any]
    contract_payload: dict[str, Any]
    freeze_payload: dict[str, Any]


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def load_contract_payload(provider: FrozenBenchmarkProvider | ModuleType) -> dict[str, Any]:
    contract_path = Path(provider.BENCHMARK_ROOT) / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Frozen benchmark contract missing: {contract_path}")
    return _mapping(json.loads(contract_path.read_text(encoding="utf-8")), name="contract_payload")


def build_freeze_payload(
    provider: FrozenBenchmarkProvider | ModuleType,
    cases_payload: Mapping[str, Any],
    contract_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Call a provider exactly once using the signature it declares.

    Signature binding happens before provider execution.  A provider exception is
    therefore never caught and retried with a different argument list.
    """

    cases = _mapping(cases_payload, name="cases_payload")
    contract = _mapping(contract_payload, name="contract_payload")
    builder = getattr(provider, "build_freeze", None)
    if not callable(builder):
        raise TypeError("Frozen benchmark provider has no callable build_freeze")

    signature = inspect.signature(builder)
    try:
        signature.bind(cases, contract)
    except TypeError as two_argument_error:
        try:
            signature.bind(cases)
        except TypeError as one_argument_error:
            raise TypeError(
                "build_freeze must accept (cases_payload) or "
                "(cases_payload, contract_payload)"
            ) from one_argument_error
        return _mapping(builder(cases), name="freeze_payload")
    return _mapping(builder(cases, contract), name="freeze_payload")


def prepare_frozen_benchmark(
    provider: FrozenBenchmarkProvider | ModuleType,
) -> FrozenBenchmarkPayloads:
    """Load and validate every frozen manifest before any source/case access."""

    cases = _mapping(provider.build_cases(), name="cases_payload")
    contract = load_contract_payload(provider)
    freeze = build_freeze_payload(provider, cases, contract)
    return FrozenBenchmarkPayloads(
        cases_payload=cases,
        contract_payload=contract,
        freeze_payload=freeze,
    )
