#!/usr/bin/env python3
"""Pre-outcome provenance contract for representation calibration seed 32001."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import yaml


REPO = Path("/admin/home/ryan.kim/nt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.matched_cancer_stage_20260730.objectives import (  # noqa: E402
    cancer_stage_loss,
)
from tools.matched_cancer_stage_20260730.receipts import (  # noqa: E402
    atomic_write_receipt,
    build_receipt,
    file_identity,
    require_regular_file,
    verify_receipt,
)


RECEIPT_SCHEMA = "matched-cancer-stage-provenance/v1"
CONFIG = (
    REPO
    / "configs_vendor/matched_cancer_stage_20260730/"
    "calibration_seed32001_contract.yaml"
)
BASE = (
    REPO
    / "configs_vendor/matched_cancer_stage_20260730/"
    "calibration_seed32001_base.yaml"
)
LOCK = (
    REPO
    / "results/matched_cancer_stage_20260730/"
    "CALIBRATION_SEED32001_LOCK.md"
)
DRIVER = (
    REPO
    / "tools/matched_cancer_stage_20260730/"
    "calibration_two_slot.sbatch"
)
EXCLUSIONS = (
    REPO
    / "configs_vendor/matched_stage_union_20260730/"
    "exclude_union_target_hospitals.txt"
)
POPULATION = (
    REPO
    / "configs_vendor/matched_cancer_stage_20260730/"
    "population_cancer_race.csv"
)
FINO_META = Path("/data/ryan.kim/nanopath_parquet_fairness/fino_meta.json")
RUNTIME_SOURCES = {
    "package_init": REPO / "tools/matched_cancer_stage_20260730/__init__.py",
    "calibration_contract": (
        REPO
        / "tools/matched_cancer_stage_20260730/calibration_contract.py"
    ),
    "receipts": REPO / "tools/matched_cancer_stage_20260730/receipts.py",
    "config_builder": (
        REPO / "tools/matched_cancer_stage_20260730/config_builder.py"
    ),
    "manifest_builder": (
        REPO / "tools/matched_cancer_stage_20260730/manifest_builder.py"
    ),
    "completion_receipt": (
        REPO / "tools/matched_cancer_stage_20260730/completion_receipt.py"
    ),
    "replay": REPO / "tools/matched_cancer_stage_20260730/replay.py",
    "objectives": REPO / "tools/matched_cancer_stage_20260730/objectives.py",
    "shared_fair_supcon": (
        REPO / "tools/matched_stage_union_20260730/objectives.py"
    ),
    "instrumentation": (
        REPO / "tools/matched_stage_union_20260730/instrumentation.py"
    ),
    "train": REPO / "vendor/matched_stage_train_20260730/train.py",
    "dataloader": REPO / "vendor/matched_stage_train_20260730/dataloader.py",
    "model": REPO / "vendor/matched_stage_train_20260730/model.py",
    "probe": REPO / "vendor/matched_stage_train_20260730/probe.py",
    "calibration_driver": DRIVER,
}


def require_file(path: Path) -> Path:
    return require_regular_file(path)


def identity(path: Path) -> dict[str, Any]:
    return file_identity(path)


def identity_paths(value: Mapping[str, Any]) -> Iterator[str]:
    if set(value) == {"canonical_path", "bytes", "sha256"}:
        yield str(value["canonical_path"])
        return
    for nested in value.values():
        if isinstance(nested, Mapping):
            yield from identity_paths(nested)


def validate() -> dict[str, Any]:
    contract = yaml.safe_load(require_file(CONFIG).read_text())
    expected = {
        "schema": "matched-cancer-stage-calibration/v1",
        "study_id": "matched_cancer_stage_20260730",
        "scenario": "brca_luad_black_white_calibration_seed32001",
        "representation_seed": 32001,
        "population": {"cancer_ids": [2, 15], "race_ids": [2, 4]},
        "replay": {"steps": 781, "batch_size": 128, "seed": 52001},
        "data_order_seed": 62001,
        "adapter": {
            "init_seed": 72001,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "fair_weight": 0.1,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(
                f"calibration contract {key}={contract.get(key)!r}, "
                f"expected {value!r}"
            )
    if contract.get("base_config") != str(BASE.relative_to(REPO)):
        raise ValueError("calibration base-config path drift")
    if contract.get("exclude_barcodes_file") != str(
        EXCLUSIONS.relative_to(REPO)
    ):
        raise ValueError("calibration exclusion path drift")
    if contract.get("arms") != {
        "B": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.0},
        "P": {"slot1_fair_weight": 0.1, "slot2_fair_weight": 0.0},
        "H": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.1},
    }:
        raise ValueError("B/P/H calibration timing-arm contract drift")
    replay = contract["replay"]
    if int(replay["steps"]) * int(replay["batch_size"]) != 99_968:
        raise ValueError("calibration replay exposure is not exactly 99,968")

    base = yaml.safe_load(require_file(BASE).read_text())
    expected_base = {
        ("train", "seed"): 32001,
        ("train", "batch_size"): 128,
        ("train", "max_train_samples"): 99_968,
        ("data", "tissue_thresh"): 0.0,
        ("probe", "enabled"): False,
        ("fino", "enabled"): False,
    }
    for path, expected_value in expected_base.items():
        section, key = path
        actual = base.get(section, {}).get(key)
        if actual != expected_value:
            raise ValueError(
                f"calibration base {section}.{key}={actual!r}, "
                f"expected {expected_value!r}"
            )

    parameters = list(inspect.signature(cancer_stage_loss).parameters)
    if parameters != ["h", "cancer_id", "race_id", "fair_weight"]:
        raise ValueError(f"cancer-stage loss API drift: {parameters}")

    metadata = json.loads(require_file(FINO_META).read_text())
    rows = {
        row["patient_barcode"]: row
        for row in csv.DictReader(require_file(POPULATION).open())
    }
    exclusions = set(require_file(EXCLUSIONS).read_text().splitlines())
    counts: dict[str, int] = {}
    eligible = []
    for patient, row in rows.items():
        if (
            patient in exclusions
            or row["cancer_type"] not in {"BRCA", "LUAD"}
            or row["race"] not in {"black or african american", "white"}
            or patient not in metadata["discrete"]["cancer"]
            or patient not in metadata["discrete"]["race"]
        ):
            continue
        cancer_id = int(metadata["discrete"]["cancer"][patient])
        race_id = int(metadata["discrete"]["race"][patient])
        if cancer_id not in {2, 15} or race_id not in {2, 4}:
            raise ValueError("calibration metadata ID mapping drift")
        eligible.append(patient)
        stratum = f"{row['cancer_type']}/{row['race']}"
        counts[stratum] = counts.get(stratum, 0) + 1
    expected_counts = {
        "BRCA/black or african american": 56,
        "BRCA/white": 521,
        "LUAD/black or african american": 11,
        "LUAD/white": 111,
    }
    if counts != expected_counts or len(eligible) != 699:
        raise ValueError(
            f"eligible calibration population drift: {counts}, "
            f"n={len(eligible)}"
        )

    representation_text = "\n".join(
        require_file(path).read_text().lower() for path in (CONFIG, BASE)
    )
    forbidden = ("tp53", "molecular_labels", "condition_on_label")
    if any(token in representation_text for token in forbidden):
        raise ValueError(
            "calibration representation config contains a downstream-label token"
        )

    identities = {
        "calibration_lock": identity(LOCK),
        "contract_config": identity(CONFIG),
        "base_config": identity(BASE),
        "exclusions": identity(EXCLUSIONS),
        "cancer_race_population": identity(POPULATION),
        "fino_meta": identity(FINO_META),
        "runtime_sources": {
            role: identity(path) for role, path in RUNTIME_SOURCES.items()
        },
    }
    return build_receipt(
        schema=RECEIPT_SCHEMA,
        study_id=contract["study_id"],
        scenario=contract["scenario"],
        identities=identities,
        fields={
            "status": "valid",
            "representation_seed": contract["representation_seed"],
            "replay_presentations": 99_968,
            "eligible_patients": len(eligible),
            "strata": counts,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--write-receipt",
        type=Path,
        help="atomically persist and re-verify the calibration receipt",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = validate()
    if args.write_receipt is not None:
        destination = args.write_receipt.resolve()
        if str(destination) in set(identity_paths(result["identities"])):
            raise ValueError(
                f"receipt destination would overwrite a bound input: "
                f"{destination}"
            )
        atomic_write_receipt(destination, result)
        verify_receipt(
            destination,
            expected_schema=RECEIPT_SCHEMA,
            expected_study_id=result["study_id"],
            expected_scenario=result["scenario"],
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
