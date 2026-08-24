from __future__ import annotations

"""Exercise the future one-shot lifecycle using synthetic DEV fixtures only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.one_shot_certification import OneShotCallbacks, execute_once
from scripts.frozen_benchmark_adapter import build_freeze_payload


DEV_CASE_IDS = ("dev-gaussian-001", "dev-jpeg-001", "dev-occlusion-001")


class _DevFreezeProvider:
    @staticmethod
    def build_freeze(cases_payload: dict[str, Any], contract_payload: dict[str, Any]) -> dict[str, Any]:
        canonical = json.dumps(
            {"cases": cases_payload, "contract": contract_payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "benchmark_id": cases_payload["benchmark_id"],
            "case_count": len(cases_payload["cases"]),
            "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_dev(output: Path, *, inject_failure: str | None = None) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty DEV evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    events: list[str] = []
    case_payloads = {case_id: {"case_id": case_id, "pixels": "synthetic"} for case_id in DEV_CASE_IDS}

    def preflight() -> dict[str, Any]:
        cases = {
            "benchmark_id": "cfs-protocol-hardening-dev-v1",
            "cases": [{"case_id": case_id} for case_id in DEV_CASE_IDS],
        }
        contract = {
            "benchmark_id": cases["benchmark_id"],
            "case_count": len(DEV_CASE_IDS),
            "fixture_policy": "SYNTHETIC_DEV_ONLY",
        }
        freeze = build_freeze_payload(_DevFreezeProvider, cases, contract)
        if freeze["case_count"] != contract["case_count"]:
            raise RuntimeError("DEV freeze/contract case-count mismatch")
        events.append("preflight_complete")
        return {"cases": cases, "contract": contract, "freeze": freeze}

    def persist_started(prepared: dict[str, Any]) -> None:
        events.append("marker_started")
        _write_json(
            output / "CONSUMED.json",
            {
                "state": "STARTED",
                "benchmark_id": prepared["cases"]["benchmark_id"],
                "fixture_policy": "SYNTHETIC_DEV_ONLY",
            },
        )

    def execute_cases(prepared: dict[str, Any]) -> dict[str, Any]:
        completed = []
        for item in prepared["cases"]["cases"]:
            case_id = str(item["case_id"])
            events.append(f"case_access:{case_id}")
            payload = case_payloads[case_id]
            completed.append({"case_id": payload["case_id"], "decision": "PASS"})
        return {"accepted": len(completed) == len(DEV_CASE_IDS), "completed": completed}

    def upload_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
        events.append("artifact_upload")
        _write_json(output / "events.json", events)
        _write_json(output / "evidence.json", evidence)
        archive = output / "protocol-hardening-evidence.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(output / "events.json", "events.json")
            bundle.write(output / "evidence.json", "evidence.json")
            marker = output / "CONSUMED.json"
            if marker.is_file():
                bundle.write(marker, "CONSUMED.json")
        return {
            "path": archive.name,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }

    def persist_final(state: str, evidence: dict[str, Any]) -> None:
        events.append("final_disposition")
        marker = json.loads((output / "CONSUMED.json").read_text(encoding="utf-8"))
        marker.update(
            {
                "state": state,
                "error": evidence.get("error"),
                "artifact_sha256": evidence["artifact"]["sha256"],
            }
        )
        _write_json(output / "CONSUMED.json", marker)
        _write_json(output / "events.json", events)

    result = execute_once(
        OneShotCallbacks(
            preflight=preflight,
            persist_started=persist_started,
            execute_cases=execute_cases,
            upload_evidence=upload_evidence,
            persist_final=persist_final,
        ),
        inject_failure=inject_failure,
    )
    _write_json(output / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inject-failure", choices=("before_marker", "after_marker"))
    args = parser.parse_args(argv)
    result = run_dev(Path(args.output), inject_failure=args.inject_failure)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] == "CONSUMED_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
