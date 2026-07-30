#!/usr/bin/env python3
"""Strict, seed-generic provenance contract for fixed-48 calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

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


PLAN_SCHEMA = "matched-cancer-fixed48-calibration-plan/v1"
SEED_PLAN_SCHEMA = "matched-cancer-fixed48-seed-plan/v1"
SEED_CONTRACT_SCHEMA = "matched-cancer-stage-calibration/v1"
# Kept API-compatible with the frozen legacy config builder. The receipt's
# fixed48 plan/source topology and study/scenario make it unambiguous.
RECEIPT_SCHEMA = "matched-cancer-stage-provenance/v1"
STUDY_ID = "matched_cancer_fixed48_20260730"
SEEDS = tuple(range(32001, 32049))
RUN_ORDER = ("slot1_plain", "slot1_fair", "B", "H", "P")
BASE_TEMPLATE_SEMANTIC_SHA256 = (
    "389ef0f62dbd09bb4c431712374bccffb8acecf9f25493cb3c8b7d6e0ac04e3e"
)
FROZEN_EFFECTIVE_FINO = {
    "enabled": True,
    "objective": "contrastive-two-condition",
    "method": "contrastive",
    "gamma_max": 0.7,
    "contrastive_temp": 0.2,
    "contrastive_weight": 0.1,
    "contrastive_condition_on": "cancer",
    "dose_logging": True,
    "race_weight": "none",
    "race_resample": False,
    "discrete": [["cancer", 1], ["race", -1]],
    "continuous": [],
}
PLAN = (
    REPO
    / "configs_vendor/matched_cancer_fixed48_20260730/"
    "calibration_contract.yaml"
)
SEED_PLAN = (
    REPO
    / "configs_vendor/matched_cancer_fixed48_20260730/"
    "calibration_seed_plan.yaml"
)
BASE_TEMPLATE = (
    REPO
    / "configs_vendor/matched_cancer_fixed48_20260730/"
    "calibration_base_template.yaml"
)
FINO_META = Path("/data/ryan.kim/nanopath_parquet_fairness/fino_meta.json")
PRETRAINED_CHECKPOINT = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/"
    "control/torch_home/hub/checkpoints/"
    "dinov2_vits14_reg4_pretrain.pth"
)
PRETRAINED_IDENTITY = {
    "canonical_path": str(PRETRAINED_CHECKPOINT),
    "bytes": 88_291_785,
    "sha256": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
}

RUNTIME_SOURCES = {
    "package_init": (
        REPO / "tools/matched_cancer_fixed48_20260730/__init__.py"
    ),
    "contract": REPO / "tools/matched_cancer_fixed48_20260730/contract.py",
    "runner": REPO / "tools/matched_cancer_fixed48_20260730/runner.py",
    "independent_auditor": (
        REPO / "tools/matched_cancer_fixed48_20260730/auditor.py"
    ),
    "calibration_driver": (
        REPO
        / "tools/matched_cancer_fixed48_20260730/"
        "calibration_one_seed.sbatch"
    ),
    "legacy_config_builder": (
        REPO / "tools/matched_cancer_stage_20260730/config_builder.py"
    ),
    "legacy_manifest_builder": (
        REPO / "tools/matched_cancer_stage_20260730/manifest_builder.py"
    ),
    "legacy_completion_receipt": (
        REPO / "tools/matched_cancer_stage_20260730/completion_receipt.py"
    ),
    "legacy_replay": (
        REPO / "tools/matched_cancer_stage_20260730/replay.py"
    ),
    "legacy_objectives": (
        REPO / "tools/matched_cancer_stage_20260730/objectives.py"
    ),
    "legacy_receipts": (
        REPO / "tools/matched_cancer_stage_20260730/receipts.py"
    ),
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
}


def _load_yaml(path: Path) -> dict[str, Any]:
    source = require_regular_file(path)
    value = yaml.safe_load(source.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {source}")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], where: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def validate_plan(
    plan: Mapping[str, Any], seed_plan: Mapping[str, Any]
) -> tuple[dict[int, dict[str, int]], dict[str, Any]]:
    """Validate every frozen invariant and return the exact seed lookup."""
    _exact_keys(
        plan,
        {
            "schema",
            "study_id",
            "scenario_prefix",
            "seed_plan",
            "base_template",
            "exclude_barcodes_file",
            "population_file",
            "output_namespace",
            "population",
            "replay",
            "adapter",
            "fair_weight",
            "arms",
        },
        "calibration plan",
    )
    expected_plan = {
        "schema": PLAN_SCHEMA,
        "study_id": STUDY_ID,
        "scenario_prefix": "brca_luad_black_white_calibration_seed",
        "seed_plan": str(SEED_PLAN.relative_to(REPO)),
        "base_template": str(BASE_TEMPLATE.relative_to(REPO)),
        "exclude_barcodes_file": (
            "configs_vendor/matched_stage_union_20260730/"
            "exclude_union_target_hospitals.txt"
        ),
        "population_file": (
            "configs_vendor/matched_cancer_stage_20260730/"
            "population_cancer_race.csv"
        ),
        "output_namespace": (
            "/data/ryan.kim/nanopath/reruns/"
            "matched_cancer_fixed48_20260730/calibration"
        ),
        "population": {"cancer_ids": [2, 15], "race_ids": [2, 4]},
        "replay": {"steps": 781, "batch_size": 128},
        "adapter": {"lr": 0.001, "weight_decay": 0.0001},
        "fair_weight": 0.1,
        "arms": {
            "B": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.0},
            "P": {"slot1_fair_weight": 0.1, "slot2_fair_weight": 0.0},
            "H": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.1},
        },
    }
    if dict(plan) != expected_plan:
        raise ValueError("calibration plan differs from the frozen fixed48 plan")

    _exact_keys(
        seed_plan,
        {"schema", "study_id", "legacy_seed32001", "seeds"},
        "seed plan",
    )
    if seed_plan["schema"] != SEED_PLAN_SCHEMA:
        raise ValueError("seed-plan schema drift")
    if seed_plan["study_id"] != STUDY_ID:
        raise ValueError("seed-plan study drift")
    expected_legacy = {
        "disposition": "systems_only_excluded_from_inference",
        "reusable": False,
        "rerun_in_fixed48_namespace": True,
    }
    if seed_plan["legacy_seed32001"] != expected_legacy:
        raise ValueError("legacy seed32001 disposition drift")
    rows = seed_plan["seeds"]
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise ValueError("seed plan must contain exactly 48 rows")
    lookup: dict[int, dict[str, int]] = {}
    keys = {
        "representation_seed",
        "replay_seed",
        "data_order_seed",
        "adapter_init_seed",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"seed row {index} must be a mapping")
        _exact_keys(row, keys, f"seed row {index}")
        if any(type(row[key]) is not int for key in keys):
            raise ValueError(f"seed row {index} values must be integers")
        seed = row["representation_seed"]
        expected_seed = SEEDS[index]
        expected_row = {
            "representation_seed": expected_seed,
            "replay_seed": expected_seed + 20_000,
            "data_order_seed": expected_seed + 30_000,
            "adapter_init_seed": expected_seed + 40_000,
        }
        if row != expected_row or seed in lookup:
            raise ValueError(
                f"seed row {index} differs from frozen mapping {expected_row}"
            )
        lookup[seed] = dict(row)
    if tuple(lookup) != SEEDS:
        raise ValueError("seed range/order differs from 32001..32048")
    if plan["replay"]["steps"] * plan["replay"]["batch_size"] != 99_968:
        raise ValueError("per-run exposure must be exactly 99,968")
    return lookup, expected_legacy


def load_and_validate_plan() -> tuple[
    dict[str, Any], dict[int, dict[str, int]], dict[str, Any]
]:
    plan = _load_yaml(PLAN)
    seed_plan = _load_yaml(SEED_PLAN)
    lookup, legacy = validate_plan(plan, seed_plan)
    return plan, lookup, legacy


def validate_attempt_root(
    root: Path | str,
    *,
    seed: int,
    output_namespace: Path | str,
) -> Path:
    """Require an exact fixed48 ``seed_N/attempt_NN`` output location."""
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("representation seed must be in 32001..32048")
    namespace = Path(output_namespace).resolve()
    candidate = Path(root)
    if candidate.is_symlink():
        raise ValueError("attempt root may not be a symlink")
    resolved = candidate.resolve()
    if resolved.parent.parent != namespace:
        raise ValueError("attempt root is outside the frozen output namespace")
    if resolved.parent.name != f"seed_{seed}":
        raise ValueError("attempt root seed directory mismatch")
    if re.fullmatch(r"attempt_[0-9]{2,}", resolved.name) is None:
        raise ValueError("attempt root must be named attempt_NN")
    return resolved


def _validate_base_template(base: Mapping[str, Any]) -> None:
    """Require semantic equality to every field in the frozen base template."""
    try:
        payload = json.dumps(
            dict(base),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"base template is not canonical JSON: {error}") from error
    if hashlib.sha256(payload).hexdigest() != BASE_TEMPLATE_SEMANTIC_SHA256:
        raise ValueError(
            "base template differs from the complete frozen semantic mapping"
        )


def validate_effective_representation_metadata(
    value: Mapping[str, Any],
) -> None:
    """Allow only the frozen pixel/cancer/race representation metadata schema."""
    if not isinstance(value, Mapping):
        raise ValueError("effective representation config must be a mapping")
    base = _load_yaml(BASE_TEMPLATE)
    _validate_base_template(base)
    expected_top_level = set(base) | {"matched_stage"}
    if set(value) != expected_top_level:
        raise ValueError("effective representation top-level keys differ")

    expected_data = dict(base["data"])
    expected_data["include_discrete"] = {
        "cancer": [2, 15],
        "race": [2, 4],
    }
    if value.get("data") != expected_data:
        raise ValueError(
            "representation data metadata must be exactly pixel inputs plus "
            "cancer/race include_discrete"
        )
    if value.get("model") != base["model"]:
        raise ValueError("representation model mapping differs")
    if value.get("dino") != base["dino"]:
        raise ValueError("representation DINO mapping differs")
    if value.get("probe") != base["probe"]:
        raise ValueError("representation probe mapping differs")
    if value.get("fino") != FROZEN_EFFECTIVE_FINO:
        raise ValueError(
            "representation fairness metadata must be exactly cancer/race"
        )

    stage = value.get("matched_stage")
    if not isinstance(stage, Mapping):
        raise ValueError("matched_stage mapping is missing")
    expected_stage_keys = {
        "enabled",
        "mode",
        "study_id",
        "scenario",
        "contract_receipt",
        "effective_config_receipt",
        "replay_manifest",
        "adapter_init_seed",
        "fair_weight",
        "adapter_lr",
        "adapter_weight_decay",
        "data_order_seed",
        "encoder_checkpoint",
        "encoder_checkpoint_sha256",
        "expected_encoder_state_sha256",
        "parent_completion_receipt",
        "replay",
    }
    if set(stage) != expected_stage_keys:
        raise ValueError("matched_stage keys differ from the frozen schema")
    if set(stage.get("replay", {})) != {
        "cancer_ids",
        "race_ids",
        "steps",
        "seed",
    }:
        raise ValueError("matched_stage replay keys differ from frozen schema")
    if stage["replay"]["cancer_ids"] != [2, 15]:
        raise ValueError("matched_stage cancer IDs differ")
    if stage["replay"]["race_ids"] != [2, 4]:
        raise ValueError("matched_stage race IDs differ")


def _validate_population(plan: Mapping[str, Any]) -> dict[str, int]:
    population = REPO / str(plan["population_file"])
    exclusions_path = REPO / str(plan["exclude_barcodes_file"])
    metadata = json.loads(require_regular_file(FINO_META).read_text())
    rows = {
        row["patient_barcode"]: row
        for row in csv.DictReader(require_regular_file(population).open())
    }
    exclusions = set(require_regular_file(exclusions_path).read_text().splitlines())
    counts: dict[str, int] = {}
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
        key = f"{row['cancer_type']}/{row['race']}"
        counts[key] = counts.get(key, 0) + 1
    expected = {
        "BRCA/black or african american": 56,
        "BRCA/white": 521,
        "LUAD/black or african american": 11,
        "LUAD/white": 111,
    }
    if counts != expected:
        raise ValueError(f"eligible calibration population drift: {counts}")
    return counts


def _exclusive_yaml(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(dict(value), handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize(
    *, seed: int, root: Path | str
) -> tuple[Path, Path, Path]:
    """Materialize one immutable seed base, contract, and provenance receipt."""
    plan, lookup, legacy = load_and_validate_plan()
    if seed not in lookup:
        raise ValueError("representation seed must be in 32001..32048")
    attempt_root = validate_attempt_root(
        root, seed=seed, output_namespace=plan["output_namespace"]
    )
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise ValueError("attempt root must already be a real directory")
    record = lookup[seed]
    base = _load_yaml(BASE_TEMPLATE)
    _validate_base_template(base)
    base["train"]["seed"] = seed
    base["project"]["name"] = f"matched-cancer-fixed48-seed{seed}-base"
    base["project"]["output_dir"] = str(attempt_root)
    base_path = attempt_root / "configs" / "calibration_base.yaml"
    contract_path = attempt_root / "configs" / "calibration_contract.yaml"
    receipt_path = attempt_root / "CALIBRATION_CONTRACT_RECEIPT.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {receipt_path}")
    _exclusive_yaml(base_path, base)

    scenario = f"{plan['scenario_prefix']}{seed}"
    seed_contract = {
        "schema": SEED_CONTRACT_SCHEMA,
        "study_id": STUDY_ID,
        "scenario": scenario,
        "representation_seed": seed,
        "base_config": str(base_path),
        "exclude_barcodes_file": str(plan["exclude_barcodes_file"]),
        "population": dict(plan["population"]),
        "replay": {
            "steps": int(plan["replay"]["steps"]),
            "batch_size": int(plan["replay"]["batch_size"]),
            "seed": record["replay_seed"],
        },
        "data_order_seed": record["data_order_seed"],
        "adapter": {
            "init_seed": record["adapter_init_seed"],
            "lr": float(plan["adapter"]["lr"]),
            "weight_decay": float(plan["adapter"]["weight_decay"]),
        },
        "fair_weight": float(plan["fair_weight"]),
        "arms": dict(plan["arms"]),
    }
    _exclusive_yaml(contract_path, seed_contract)

    parameters = list(inspect.signature(cancer_stage_loss).parameters)
    if parameters != ["h", "cancer_id", "race_id", "fair_weight"]:
        raise ValueError(f"cancer-stage loss API drift: {parameters}")
    counts = _validate_population(plan)
    if file_identity(PRETRAINED_CHECKPOINT) != PRETRAINED_IDENTITY:
        raise ValueError("frozen DINOv2 pretrained checkpoint identity drift")
    identities = {
        "fixed_plan": file_identity(PLAN),
        "seed_plan": file_identity(SEED_PLAN),
        "base_template": file_identity(BASE_TEMPLATE),
        "materialized_base": file_identity(base_path),
        "materialized_contract": file_identity(contract_path),
        "exclusions": file_identity(REPO / plan["exclude_barcodes_file"]),
        "cancer_race_population": file_identity(REPO / plan["population_file"]),
        "fino_meta": file_identity(FINO_META),
        "pretrained_checkpoint": file_identity(PRETRAINED_CHECKPOINT),
        "runtime_sources": {
            role: file_identity(path) for role, path in RUNTIME_SOURCES.items()
        },
    }
    receipt = build_receipt(
        schema=RECEIPT_SCHEMA,
        study_id=STUDY_ID,
        scenario=scenario,
        identities=identities,
        fields={
            "status": "valid",
            "representation_seed": seed,
            "replay_seed": record["replay_seed"],
            "data_order_seed": record["data_order_seed"],
            "adapter_init_seed": record["adapter_init_seed"],
            "replay_presentations": 99_968,
            "eligible_patients": sum(counts.values()),
            "strata": counts,
            "legacy_seed32001_disposition": legacy,
        },
    )
    atomic_write_receipt(receipt_path, receipt)
    verify_receipt(
        receipt_path,
        expected_schema=RECEIPT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario,
    )
    return base_path, contract_path, receipt_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-plan")
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--seed", type=int, required=True)
    materialize_parser.add_argument("--root", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-plan":
        _, lookup, legacy = load_and_validate_plan()
        print(
            json.dumps(
                {
                    "status": "fixed48_calibration_plan_valid",
                    "seeds": list(lookup),
                    "legacy_seed32001": legacy,
                },
                sort_keys=True,
            )
        )
        return 0
    base, contract, receipt = materialize(seed=args.seed, root=args.root)
    print(
        json.dumps(
            {
                "status": "fixed48_calibration_contract_materialized",
                "seed": args.seed,
                "base": str(base),
                "contract": str(contract),
                "receipt": str(receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
