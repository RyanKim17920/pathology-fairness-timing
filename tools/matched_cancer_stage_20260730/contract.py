#!/usr/bin/env python3
"""Pre-outcome integrity checks for the cancer-only straight-stage study."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO = Path("/admin/home/ryan.kim/nt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.matched_cancer_stage_20260730.objectives import cancer_stage_loss
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    require_regular_file,
    verify_receipt,
)


RECEIPT_SCHEMA = "matched-cancer-stage-provenance/v1"
CONFIG = (
    REPO / "configs_vendor/matched_cancer_stage_20260730/smoke_contract.yaml"
)
DESIGN = REPO / "results/matched_cancer_stage_20260730/DESIGN_LOCK.md"
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
    "contract": REPO / "tools/matched_cancer_stage_20260730/contract.py",
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
    "population_manifest_builder": (
        REPO
        / "tools/matched_cancer_stage_20260730/"
        "population_manifest_builder.py"
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
    "smoke_driver": (
        REPO / "tools/matched_cancer_stage_20260730/smoke_two_slot.sbatch"
    ),
}


def require_file(path: Path) -> Path:
    return require_regular_file(path)


def identity(path: Path) -> dict[str, Any]:
    return file_identity(path)


def validate() -> dict[str, Any]:
    contract = yaml.safe_load(require_file(CONFIG).read_text())
    expected = {
        "schema": "matched-cancer-stage-smoke/v1",
        "study_id": "matched_cancer_stage_20260730",
        "scenario": "brca_luad_black_white",
        "population": {"cancer_ids": [2, 15], "race_ids": [2, 4]},
        "replay": {"steps": 4, "batch_size": 8, "seed": 50730},
        "data_order_seed": 60730,
        "adapter": {
            "init_seed": 70730,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "fair_weight": 0.1,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(
                f"smoke contract {key}={contract.get(key)!r}, expected {value!r}"
            )
    if contract.get("arms") != {
        "B": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.0},
        "P": {"slot1_fair_weight": 0.1, "slot2_fair_weight": 0.0},
        "H": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.1},
    }:
        raise ValueError("B/P/H timing-arm contract drift")

    parameters = list(inspect.signature(cancer_stage_loss).parameters)
    if parameters != ["h", "cancer_id", "race_id", "fair_weight"]:
        raise ValueError(f"cancer-stage loss API drift: {parameters}")

    meta = json.loads(require_file(FINO_META).read_text())
    rows = {
        row["patient_barcode"]: row
        for row in csv.DictReader(require_file(POPULATION).open())
    }
    exclusions = set(require_file(EXCLUSIONS).read_text().splitlines())
    eligible = []
    counts: dict[str, int] = {}
    for patient, row in rows.items():
        if (
            patient in exclusions
            or row["cancer_type"] not in {"BRCA", "LUAD"}
            or row["race"] not in {"black or african american", "white"}
            or patient not in meta["discrete"]["cancer"]
            or patient not in meta["discrete"]["race"]
        ):
            continue
        cancer_id = int(meta["discrete"]["cancer"][patient])
        race_id = int(meta["discrete"]["race"][patient])
        if cancer_id not in {2, 15} or race_id not in {2, 4}:
            raise ValueError("metadata ID mapping drift")
        eligible.append(patient)
        key = f"{row['cancer_type']}/{row['race']}"
        counts[key] = counts.get(key, 0) + 1
    expected_counts = {
        "BRCA/black or african american": 56,
        "BRCA/white": 521,
        "LUAD/black or african american": 11,
        "LUAD/white": 111,
    }
    if counts != expected_counts or len(eligible) != 699:
        raise ValueError(
            f"eligible representation population drift: {counts}, n={len(eligible)}"
        )

    config_text = require_file(CONFIG).read_text().lower()
    forbidden = ("tp53", "molecular_labels", "condition_on_label")
    if any(token in config_text for token in forbidden):
        raise ValueError("representation config contains a downstream-label token")

    identities = {
        "design": identity(DESIGN),
        "contract_config": identity(CONFIG),
        "base_config": identity(REPO / contract["base_config"]),
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
            "eligible_patients": len(eligible),
            "strata": counts,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--write-receipt",
        type=Path,
        help="atomically persist and re-verify the validated provenance receipt",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = validate()
    if args.write_receipt is not None:
        destination = args.write_receipt.resolve()
        source_paths = {
            record["canonical_path"]
            for role in result["identities"].values()
            for record in (
                role.values()
                if isinstance(role, dict) and "canonical_path" not in role
                else (role,)
            )
        }
        if str(destination) in source_paths:
            raise ValueError(
                f"receipt destination would overwrite a bound input: {destination}"
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
