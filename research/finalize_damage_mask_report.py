from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = list(manifest.get("sources", []))
    train = [row for row in sources if row.get("dataset_split") == "train"]
    validation = [row for row in sources if row.get("dataset_split") == "validation"]
    if len(validation) != 2:
        raise RuntimeError(f"Expected two held-out source-bank identities, got {len(validation)}")
    if sources[-2:] != validation:
        raise RuntimeError("Mixed manifest ordering does not match trainer split contract")

    train_identity_keys = {str(row.get("identity_key")) for row in train}
    validation_identity_keys = {str(row.get("identity_key")) for row in validation}
    if train_identity_keys & validation_identity_keys:
        raise RuntimeError("Identity leakage in mixed DamageMaskNet source bank")

    data = report.setdefault("data", {})
    data["manifest"] = str(args.manifest)
    data["source_bank_version"] = manifest.get("version")
    data["source_bank_counts"] = manifest.get("counts")
    data["source_bank_datasets"] = manifest.get("datasets")
    data["real_face_training_present"] = any(
        row.get("dataset_split") == "train" and not bool(row.get("synthetic_identity"))
        for row in sources
    )
    data["explicit_identity_validation"] = all(
        bool(row.get("synthetic_identity")) and str(row.get("identity_key", "")).startswith("controlface:")
        for row in validation
    )
    data["identity_disjoint"] = True
    data["final_holdout_used"] = False
    data["limitations"] = [
        "Development vertical slice uses a small mixed FairFace + ControlFace10K source bank, not the final 300-400 identity bank",
        "FairFace real images are used only in training; held-out validation identities come from explicit ControlFace10K identity folders",
        "This run proves mixed-domain training/export behavior; it does not qualify DamageMaskNet for production",
    ]
    report["qualification_scope"] = "development_only_mixed_open_source_bank_not_final_holdout"
    report["production_qualified"] = False

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "real_face_training_present": data["real_face_training_present"],
        "explicit_identity_validation": data["explicit_identity_validation"],
        "counts": manifest.get("counts"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
