#!/usr/bin/env python3
"""Strict configuration and artifact checks for the matched-stage study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


REPO = Path("/admin/home/ryan.kim/nt")
CONFIG_DIR = REPO / "configs_vendor/matched_stage_union_20260730"
TARGETS = CONFIG_DIR / "targets.tsv"
EXCLUSIONS = CONFIG_DIR / "exclude_union_target_hospitals.txt"
PREREG = REPO / "results/matched_stage_union_20260730/PREREGISTRATION.md"
CORRECTION = (
    REPO
    / "results/matched_stage_union_20260730/"
    "PREACTIVATION_EXCLUSION_CORRECTION.md"
)
DRIVER_CORRECTION = (
    REPO
    / "results/matched_stage_union_20260730/"
    "CALIBRATION_DRIVER_CORRECTION.md"
)
INSTRUMENTATION_CORRECTION = (
    REPO
    / "results/matched_stage_union_20260730/"
    "CALIBRATION_INSTRUMENTATION_CORRECTION.md"
)
TRAIN_SOURCE = REPO / "vendor/matched_stage_train_20260730/train.py"
INSTRUMENTATION_SOURCE = (
    REPO / "tools/matched_stage_union_20260730/instrumentation.py"
)
OBJECTIVES_SOURCE = (
    REPO / "tools/matched_stage_union_20260730/objectives.py"
)
TRAIN_METADATA = Path(
    "/data/ryan.kim/nanopath_parquet_fairness/fino_meta.json"
)
EXPECTED_SPLIT_SEED = 7777
EXPECTED_PRETRAIN_WEIGHT = 0.1
EXPECTED_SAMPLE_CAP = 100_000


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        fail(f"{label} must be a nonempty regular non-symlink file: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(require_file(path, "YAML").read_text())
    if not isinstance(payload, dict):
        fail(f"YAML root must be a mapping: {path}")
    return payload


def read_targets() -> list[dict[str, str]]:
    with require_file(TARGETS, "target matrix").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [row["target"] for row in rows] != ["brca", "luad", "ucec", "coad"]:
        fail("target matrix must contain brca, luad, ucec, coad in frozen order")
    return rows


def validate_exclusions() -> dict[str, int]:
    exclusion_rows = require_file(EXCLUSIONS, "union exclusion").read_text().splitlines()
    if exclusion_rows != sorted(set(exclusion_rows)) or not exclusion_rows:
        fail("union exclusion must be nonempty, unique, and sorted")
    held = set(exclusion_rows)
    counts: dict[str, int] = {}
    target_tss: set[str] = set()
    all_target_patients: set[str] = set()
    for row in read_targets():
        folds_path = REPO / row["folds_csv"]
        with require_file(folds_path, "fold metadata").open(newline="") as handle:
            target_records = [
                record
                for record in csv.DictReader(handle)
                if record["fold"].strip() == row["hospital_fold"]
            ]
        target_patients = [
            record["patient_barcode"].strip() for record in target_records
        ]
        target_tss.update(record["tss"].strip() for record in target_records)
        all_target_patients.update(target_patients)
        expected = int(row["expected_patients"])
        if len(target_patients) != expected or len(set(target_patients)) != expected:
            fail(
                f"{row['target']}: expected {expected} unique target patients, "
                f"found {len(target_patients)} rows/{len(set(target_patients))} unique"
            )
        missing = sorted(set(target_patients) - held)
        if missing:
            fail(f"{row['target']}: {len(missing)} target patients absent from exclusion")
        counts[row["target"]] = expected
    metadata = json.loads(require_file(TRAIN_METADATA, "training metadata").read_text())
    all_training_patients = set(metadata["discrete"]["cancer"])
    hospital_patients = {
        patient
        for patient in all_training_patients
        if patient.split("-")[1] in target_tss
    }
    expected_union = all_target_patients | hospital_patients
    missing_hospital = sorted(expected_union - held)
    extra = sorted(held - expected_union)
    if missing_hospital or extra:
        fail(
            "union exclusion is not exact: "
            f"{len(missing_hospital)} required patients missing, "
            f"{len(extra)} extra patients present"
        )
    return counts


def validate_base_config(path: Path, arm: str) -> dict[str, Any]:
    cfg = read_yaml(path)
    if int(cfg["data"]["split_seed"]) != EXPECTED_SPLIT_SEED:
        fail(f"{path}: split seed drift")
    if Path(cfg["data"]["exclude_barcodes_file"]).resolve() != EXCLUSIONS.resolve():
        fail(f"{path}: wrong union exclusion")
    if int(cfg["train"]["max_train_samples"]) != EXPECTED_SAMPLE_CAP:
        fail(f"{path}: calibration sample cap drift")
    if bool(cfg["probe"]["enabled"]):
        fail(f"{path}: probe must be disabled")
    fino = cfg.get("fino") or {}
    if arm == "plain":
        if bool(fino.get("enabled")):
            fail(f"{path}: plain arm enables fairness")
    elif arm == "faircon":
        expected = {
            "enabled": True,
            "objective": "contrastive-demographics",
            "method": "contrastive",
            "contrastive_temp": 0.2,
            "contrastive_weight": EXPECTED_PRETRAIN_WEIGHT,
            "dose_logging": True,
            "race_weight": "none",
            "race_resample": False,
            "discrete": [["race", -1]],
            "continuous": [],
        }
        for key, wanted in expected.items():
            if fino.get(key) != wanted:
                fail(f"{path}: fino.{key}={fino.get(key)!r}, expected {wanted!r}")
    else:
        fail(f"unknown arm: {arm}")
    return cfg


def validate_contract() -> dict[str, Any]:
    require_file(PREREG, "preregistration")
    require_file(CORRECTION, "preactivation correction")
    require_file(DRIVER_CORRECTION, "calibration driver correction")
    require_file(
        INSTRUMENTATION_CORRECTION,
        "calibration instrumentation correction",
    )
    require_file(TRAIN_SOURCE, "training source")
    require_file(INSTRUMENTATION_SOURCE, "instrumentation source")
    require_file(OBJECTIVES_SOURCE, "shared objective source")
    counts = validate_exclusions()
    plain = CONFIG_DIR / "calibration_plain.yaml"
    fair = CONFIG_DIR / "calibration_faircon.yaml"
    validate_base_config(plain, "plain")
    validate_base_config(fair, "faircon")
    source = (
        TRAIN_SOURCE.read_text()
        + "\n"
        + INSTRUMENTATION_SOURCE.read_text()
        + "\n"
        + OBJECTIVES_SOURCE.read_text()
    )
    required_fragments = (
        'relation="different"',
        "dose_logging",
        "gradient_dose_diagnostic",
        '"dose_fair_main_grad_ratio"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        fail(f"training source lacks contract fragments: {missing}")
    return {
        "targets": counts,
        "exclusion_rows": len(EXCLUSIONS.read_text().splitlines()),
        "study_inputs": {
            "targets": {
                "canonical_path": str(TARGETS.resolve()),
                "bytes": TARGETS.stat().st_size,
                "sha256": sha256_file(TARGETS),
            },
            "exclusions": {
                "canonical_path": str(EXCLUSIONS.resolve()),
                "bytes": EXCLUSIONS.stat().st_size,
                "sha256": sha256_file(EXCLUSIONS),
            },
            "training_metadata": {
                "canonical_path": str(TRAIN_METADATA.resolve()),
                "bytes": TRAIN_METADATA.stat().st_size,
                "sha256": sha256_file(TRAIN_METADATA),
            },
            "folds": {
                row["target"]: {
                    "canonical_path": str((REPO / row["folds_csv"]).resolve()),
                    "bytes": (REPO / row["folds_csv"]).stat().st_size,
                    "sha256": sha256_file(REPO / row["folds_csv"]),
                }
                for row in read_targets()
            },
        },
        "training_sha256": sha256_file(TRAIN_SOURCE),
        "instrumentation_sha256": sha256_file(INSTRUMENTATION_SOURCE),
        "shared_objective": {
            "canonical_path": str(OBJECTIVES_SOURCE.resolve()),
            "bytes": OBJECTIVES_SOURCE.stat().st_size,
            "sha256": sha256_file(OBJECTIVES_SOURCE),
        },
        "preregistration_sha256": sha256_file(PREREG),
        "correction_sha256": sha256_file(CORRECTION),
        "driver_correction_sha256": sha256_file(DRIVER_CORRECTION),
        "instrumentation_correction_sha256": sha256_file(
            INSTRUMENTATION_CORRECTION
        ),
    }


def make_config(args: argparse.Namespace) -> None:
    base = Path(args.base).resolve()
    destination = Path(args.output).resolve()
    stage = Path(args.stage).resolve()
    cfg = validate_base_config(base, args.arm)
    seed = int(args.seed)
    if seed not in range(31001, 31005):
        fail("calibration seed must be 31001..31004")
    cfg["project"]["name"] = f"matched-stage-calibration-{args.arm}-seed-{seed}"
    cfg["project"]["output_dir"] = str(stage)
    cfg["train"]["seed"] = seed
    cfg["train"]["resume"] = None
    cfg["probe"]["enabled"] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.replace(temporary, destination)


def read_training_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(require_file(path, "metrics").read_text().splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"{path}:{line_number}: invalid JSON: {error}")
        if "total" in row:
            rows.append(row)
    if not rows:
        fail(f"{path}: no training metric rows")
    return rows


def validate_fm(args: argparse.Namespace) -> None:
    stage = Path(args.stage).resolve()
    if not stage.is_dir() or stage.is_symlink():
        fail(f"stage must be a real directory: {stage}")
    effective_path = stage / "effective_config.yaml"
    summary_path = stage / "summary.json"
    metrics_path = stage / "metrics.jsonl"
    checkpoint_path = stage / "latest.pt"
    for path, label in (
        (effective_path, "effective config"),
        (summary_path, "summary"),
        (metrics_path, "metrics"),
        (checkpoint_path, "checkpoint"),
    ):
        require_file(path, label)
    effective = read_yaml(effective_path)
    seed = int(args.seed)
    if int(effective["train"]["seed"]) != seed:
        fail("effective seed mismatch")
    if Path(effective["project"]["output_dir"]).resolve() != stage:
        fail("effective output path mismatch")
    validate_base_config(effective_path, args.arm)
    summary = json.loads(summary_path.read_text())
    if summary.get("stop_reason") != "max_train_samples":
        fail(f"unexpected stop reason: {summary.get('stop_reason')!r}")
    batch_size = int(effective["train"]["batch_size"])
    expected_presentations = (
        EXPECTED_SAMPLE_CAP // batch_size
    ) * batch_size
    if int(summary.get("tile_presentations", -1)) != expected_presentations:
        fail(
            "training did not reach the largest full-batch presentation "
            f"count under the calibration cap ({expected_presentations})"
        )
    rows = read_training_rows(metrics_path)
    steps_completed = int(summary.get("steps_completed", -1))
    log_every = int(effective["train"]["log_every"])
    expected_dose_steps = {1, *range(log_every, steps_completed + 1, log_every)}
    if args.arm == "faircon":
        dose_rows = [row for row in rows if "dose_fair_main_grad_ratio" in row]
        actual_dose_steps = {int(row.get("step", -1)) for row in dose_rows}
        if len(dose_rows) != len(expected_dose_steps) or (
            actual_dose_steps != expected_dose_steps
        ):
            fail(
                "fair arm dose schedule mismatch: "
                f"expected {sorted(expected_dose_steps)}, "
                f"found {sorted(actual_dose_steps)}"
            )
        if len(dose_rows) != len(rows):
            fail("every ordinary fair-arm training row must contain diagnostics")
        for row in dose_rows:
            required = (
                "main",
                "fair",
                "dose_main_grad_norm",
                "dose_fair_grad_norm",
                "dose_fair_main_grad_ratio",
                "dose_grad_cosine",
            )
            if any(key not in row for key in required):
                fail("incomplete fair-arm dose row")
            if not all(
                math.isfinite(float(row[key]))
                for key in required
            ):
                fail("nonfinite fair-arm dose diagnostic")
            if float(row["dose_fair_grad_norm"]) <= 0:
                fail("zero effective fair gradient")
    elif any("dose_fair_main_grad_ratio" in row for row in rows):
        fail("plain arm unexpectedly emitted gradient-dose diagnostics")
    manifest = {
        "schema": "matched-stage-calibration-fm/v1",
        "arm": args.arm,
        "seed": seed,
        "contract": validate_contract(),
        "files": {
            str(path.relative_to(stage)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (effective_path, summary_path, metrics_path, checkpoint_path)
        },
    }
    manifest_path = stage / "manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=stage, prefix=".manifest.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, manifest_path)
    print(
        json.dumps(
            {
                "status": "valid",
                "arm": args.arm,
                "seed": seed,
                "dose_rows": sum(
                    "dose_fair_main_grad_ratio" in row for row in rows
                ),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    make = sub.add_parser("make-config")
    make.add_argument("--base", required=True)
    make.add_argument("--output", required=True)
    make.add_argument("--stage", required=True)
    make.add_argument("--arm", choices=("plain", "faircon"), required=True)
    make.add_argument("--seed", type=int, required=True)
    validate = sub.add_parser("validate-fm")
    validate.add_argument("--stage", required=True)
    validate.add_argument("--arm", choices=("plain", "faircon"), required=True)
    validate.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        print(json.dumps(validate_contract(), indent=2, sort_keys=True))
    elif args.command == "make-config":
        make_config(args)
    elif args.command == "validate-fm":
        validate_fm(args)
    else:
        fail(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
