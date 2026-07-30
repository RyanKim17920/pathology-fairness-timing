#!/usr/bin/env python3
"""Canonical prediction exporter and collector for the independent verifier."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from tools import reliable_fairness_head as reliable
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)


ROW_SCHEMA = "matched-cancer-diagnostic-prediction/v1"
EXPORT_SCHEMA = "matched-cancer-diagnostic-export/v1"
COLLECTION_SCHEMA = "matched-cancer-diagnostic-collection/v1"
FIELDS = frozenset({
    "schema", "fm_seed", "arm", "cancer", "head_seed", "patient_id",
    "y_true", "race", "fold", "role", "outer_fold", "inner_fold",
    "probability",
})
FM_SEEDS = tuple(range(32001, 32049))
ARMS = ("B", "P", "H")
CANCERS = ("BRCA", "LUAD")
COHORT_SIZES = {"BRCA": 328, "LUAD": 281}
HEAD_SEEDS = (42001, 42002, 42003, 42004)
TASK_IDS = {"BRCA": "brca_tp53", "LUAD": "luad_tp53"}


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"sealed JSONL destination exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(
                    row, ensure_ascii=False, allow_nan=False,
                    sort_keys=True, separators=(",", ":"),
                ) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            value = json.loads(line, object_pairs_hook=unique_object)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def prediction_row(
    row: Mapping[str, Any],
    *,
    fm_seed: int,
    arm: str,
    cancer: str,
    head_seed: int,
) -> dict[str, Any]:
    """Map one richer reliable-fairness row into the exact sealed 13 fields."""
    role = row["prediction_role"]
    original = int(row["original_fold"])
    if role == "outer_test":
        outer_fold, inner_fold = int(row["outer_fold"]), None
    elif role == "inner_calibration":
        outer_fold = int(row["calibration_outer_fold"])
        inner_fold = int(row["inner_fold"])
    else:
        raise ValueError("invalid nested prediction role")
    output = {
        "schema": ROW_SCHEMA,
        "fm_seed": int(fm_seed),
        "arm": arm,
        "cancer": cancer,
        "head_seed": int(head_seed),
        "patient_id": str(row["patient_id"]),
        "y_true": int(row["y_true"]),
        "race": row["race"],
        "fold": original,
        "role": role,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "probability": float(row["y_score"]),
    }
    validate_prediction_row(output)
    return output


def validate_prediction_row(row: Mapping[str, Any]) -> None:
    def is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    if set(row) != FIELDS or row.get("schema") != ROW_SCHEMA:
        raise ValueError("prediction row fields/schema differ")
    if (
        not is_int(row["fm_seed"])
        or row["fm_seed"] not in FM_SEEDS
        or row["arm"] not in ARMS
    ):
        raise ValueError("invalid seed/arm")
    if (
        row["cancer"] not in CANCERS
        or not is_int(row["head_seed"])
        or row["head_seed"] not in HEAD_SEEDS
    ):
        raise ValueError("invalid cancer/head seed")
    if not isinstance(row["patient_id"], str) or not row["patient_id"]:
        raise ValueError("invalid patient ID")
    if (
        row["race"] not in ("Black", "White")
        or not is_int(row["y_true"])
        or row["y_true"] not in (0, 1)
    ):
        raise ValueError("invalid race/outcome")
    if (
        not is_int(row["fold"])
        or row["fold"] not in range(5)
        or not is_int(row["outer_fold"])
        or row["outer_fold"] not in range(5)
    ):
        raise ValueError("invalid fold")
    if row["role"] == "outer_test":
        if row["outer_fold"] != row["fold"] or row["inner_fold"] is not None:
            raise ValueError("invalid outer row")
    elif row["role"] == "inner_calibration":
        if (
            row["outer_fold"] == row["fold"]
            or not is_int(row["inner_fold"])
            or row["inner_fold"] != row["fold"]
        ):
            raise ValueError("invalid inner row")
    else:
        raise ValueError("invalid role")
    probability = row["probability"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0 <= float(probability) <= 1
    ):
        raise ValueError("invalid probability")


def export_cell(
    *,
    nested_predictions: str | Path,
    diagnostic_receipt: str | Path,
    destination: str | Path,
    fm_seed: int,
    arm: str,
    cancer: str,
    head_seed: int,
    deployment_gate_receipt: str | Path,
    loader_root_receipt: str | Path,
    expected_cohort_size: int | None = None,
) -> Path:
    from tools.matched_cancer_diagnostic_20260730.deployment import (
        verify_gate,
        verify_loader_result,
    )

    gate = verify_gate(deployment_gate_receipt)
    loader_root_path = Path(loader_root_receipt).resolve()
    loader_root = verify_loader_result(
        loader_root_path, deployment_gate_receipt
    )
    if gate["representation_seed"] != fm_seed:
        raise ValueError("export fm_seed is not deployment-gate-bound")
    source = Path(nested_predictions).resolve()
    cell_receipt_path = Path(diagnostic_receipt).resolve()
    cell_receipt = verify_receipt(
        cell_receipt_path,
        expected_schema="matched-cancer-adapter-diagnostic/v1",
    )
    if cell_receipt.get("arm") != arm or cell_receipt.get("head_seed") != head_seed:
        raise ValueError("diagnostic receipt coordinate mismatch")
    if cell_receipt.get("task_id") != TASK_IDS[cancer]:
        raise ValueError("diagnostic task is not cancer-bound")
    if (
        cell_receipt.get("study_id") != gate["study_id"]
        or cell_receipt.get("scenario") != gate["scenario"]
    ):
        raise ValueError("diagnostic receipt differs from deployment gate")
    completion = cell_receipt.get("identities", {}).get("completion_receipt")
    if completion != gate["identities"]["completion_receipts"][arm]:
        raise ValueError("diagnostic representation is not deployment-gate-bound")
    diagnostic_root = verify_receipt(
        loader_root["identities"]["diagnostics"][cancer]["canonical_path"],
        expected_schema="matched-cancer-adapter-diagnostic-root/v1",
        expected_study_id=gate["study_id"],
        expected_scenario=gate["scenario"],
    )
    bound_cell = diagnostic_root["identities"]["cells"][arm][str(head_seed)]
    if file_identity(cell_receipt_path) != bound_cell:
        raise ValueError("diagnostic cell is not cancer-loader-root-bound")
    cohort_receipt = verify_receipt(
        loader_root["identities"]["cohorts"][cancer]["canonical_path"],
        expected_schema="matched-cancer-diagnostic-cohort/v1",
    )
    if cell_receipt["identities"].get("cohort_source") != cohort_receipt[
        "identities"
    ].get("cohort_records"):
        raise ValueError("diagnostic cell cohort source is not loader-root-bound")
    if (
        file_identity(source)
        != cell_receipt.get("identities", {}).get("predictions")
    ):
        raise ValueError("nested predictions are not diagnostic-receipt-bound")
    rich = _load_jsonl(source)
    reliable._validate_nested_prediction_records(rich)
    rows = [
        prediction_row(
            row, fm_seed=fm_seed, arm=arm, cancer=cancer,
            head_seed=head_seed,
        )
        for row in rich
    ]
    count = len({row["patient_id"] for row in rows})
    expected = COHORT_SIZES[cancer] if expected_cohort_size is None else int(
        expected_cohort_size
    )
    if count != expected or len(rows) != 5 * expected:
        raise ValueError(
            f"{cancer} exported cohort/row count {count}/{len(rows)} "
            f"!= {expected}/{5 * expected}"
        )
    requested = Path(destination)
    if requested.resolve() in {source, cell_receipt_path}:
        raise ValueError("export destination aliases a sealed input")
    output = _atomic_jsonl(requested.resolve(), rows)
    receipt = build_receipt(
        schema=EXPORT_SCHEMA,
        study_id=cell_receipt["study_id"],
        scenario=cell_receipt["scenario"],
        identities={
            "diagnostic_receipt": file_identity(cell_receipt_path),
            "deployment_gate_receipt": file_identity(
                deployment_gate_receipt
            ),
            "loader_root_receipt": file_identity(loader_root_path),
            "nested_predictions": file_identity(source),
            "exported_predictions": file_identity(output),
            "exporter": file_identity(Path(__file__)),
        },
        fields={
            "fm_seed": fm_seed, "arm": arm, "cancer": cancer,
            "head_seed": head_seed, "patient_count": count,
            "row_count": len(rows), "row_schema": ROW_SCHEMA,
        },
    )
    receipt_path = atomic_write_receipt(
        output.with_suffix(output.suffix + ".receipt.json"), receipt
    )
    verify_receipt(receipt_path, expected_schema=EXPORT_SCHEMA)
    return output


def collect_exports(
    exports: Sequence[str | Path],
    *,
    destination: str | Path,
    expected_fm_seeds: Sequence[int],
    cohort_sizes: Mapping[str, int] = COHORT_SIZES,
) -> Path:
    all_rows: list[dict[str, Any]] = []
    identities = {}
    observed = set()
    semantic = set()
    study_scenario: tuple[str, str] | None = None
    deployment_gate_sha256: str | None = None
    loader_root_sha256: str | None = None
    for index, value in enumerate(exports):
        path = Path(value).resolve()
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        receipt = verify_receipt(receipt_path, expected_schema=EXPORT_SCHEMA)
        if file_identity(path) != receipt["identities"]["exported_predictions"]:
            raise ValueError("export receipt does not bind prediction file")
        identities[str(index)] = {
            "predictions": file_identity(path),
            "receipt": file_identity(receipt_path),
        }
        context = (receipt["study_id"], receipt["scenario"])
        if study_scenario is None:
            study_scenario = context
        elif context != study_scenario:
            raise ValueError("export study/scenario contexts differ")
        gate_identity = receipt.get("identities", {}).get(
            "deployment_gate_receipt"
        )
        if not isinstance(gate_identity, Mapping):
            raise ValueError("export lacks deployment-gate ancestry")
        if deployment_gate_sha256 is None:
            deployment_gate_sha256 = gate_identity.get("sha256")
        elif gate_identity.get("sha256") != deployment_gate_sha256:
            raise ValueError("exports descend from different deployment gates")
        loader_identity = receipt.get("identities", {}).get(
            "loader_root_receipt"
        )
        if not isinstance(loader_identity, Mapping):
            raise ValueError("export lacks loader-root ancestry")
        if loader_root_sha256 is None:
            loader_root_sha256 = loader_identity.get("sha256")
        elif loader_identity.get("sha256") != loader_root_sha256:
            raise ValueError("exports descend from different loader roots")
        rows = _load_jsonl(path)
        receipt_coordinate = (
            receipt.get("fm_seed"), receipt.get("arm"),
            receipt.get("cancer"), receipt.get("head_seed"),
        )
        row_coordinates = set()
        for row in rows:
            validate_prediction_row(row)
            coordinate = (
                row["fm_seed"], row["arm"], row["cancer"], row["head_seed"]
            )
            row_coordinates.add(coordinate)
            observed.add(coordinate)
            key = (
                *coordinate, row["patient_id"], row["role"], row["outer_fold"]
            )
            if key in semantic:
                raise ValueError("duplicate semantic prediction row")
            semantic.add(key)
            all_rows.append(row)
        if row_coordinates != {receipt_coordinate}:
            raise ValueError("export receipt and row coordinates differ")
    expected = {
        (seed, arm, cancer, head)
        for seed in expected_fm_seeds for arm in ARMS
        for cancer in CANCERS for head in HEAD_SEEDS
    }
    if observed != expected:
        raise ValueError("export collection coordinate matrix is incomplete")
    for coordinate in expected:
        selected = [
            row for row in all_rows
            if (
                row["fm_seed"], row["arm"], row["cancer"], row["head_seed"]
            ) == coordinate
        ]
        n = int(cohort_sizes[coordinate[2]])
        if len(selected) != 5 * n:
            raise ValueError(f"wrong row count for coordinate {coordinate}")
        by_patient: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            by_patient.setdefault(row["patient_id"], []).append(row)
        if len(by_patient) != n:
            raise ValueError(f"wrong patient count for coordinate {coordinate}")
        for patient_rows in by_patient.values():
            roles = [row["role"] for row in patient_rows]
            outer_folds = {row["outer_fold"] for row in patient_rows}
            if (
                len(patient_rows) != 5
                or roles.count("outer_test") != 1
                or roles.count("inner_calibration") != 4
                or outer_folds != set(range(5))
            ):
                raise ValueError("patient lacks exact nested row topology")
    metadata: dict[tuple[int, str, str], tuple[Any, ...]] = {}
    cross_seed_metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
    patient_sets: dict[tuple[int, str], set[str]] = {}
    for row in all_rows:
        population = (row["fm_seed"], row["cancer"])
        patient_sets.setdefault(population, set()).add(row["patient_id"])
        key = (*population, row["patient_id"])
        value = (row["y_true"], row["race"], row["fold"])
        if key in metadata and metadata[key] != value:
            raise ValueError("patient metadata drifts across coordinates")
        metadata[key] = value
        cross_seed_key = (row["cancer"], row["patient_id"])
        if (
            cross_seed_key in cross_seed_metadata
            and cross_seed_metadata[cross_seed_key] != value
        ):
            raise ValueError("patient metadata drifts across FM seeds")
        cross_seed_metadata[cross_seed_key] = value
    for seed in expected_fm_seeds:
        for cancer in CANCERS:
            if len(patient_sets.get((seed, cancer), set())) != int(
                cohort_sizes[cancer]
            ):
                raise ValueError("patient set drifts across coordinates")
    for cancer in CANCERS:
        reference: set[str] | None = None
        for seed in expected_fm_seeds:
            current = patient_sets[(seed, cancer)]
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError("patient IDs drift across FM seeds")
    all_rows.sort(key=lambda row: (
        row["fm_seed"], row["cancer"], row["arm"], row["head_seed"],
        row["patient_id"], row["role"], row["outer_fold"],
    ))
    requested = Path(destination)
    if requested.resolve() in {Path(value).resolve() for value in exports}:
        raise ValueError("collection destination aliases an export")
    output = _atomic_jsonl(requested.resolve(), all_rows)
    receipt = build_receipt(
        schema=COLLECTION_SCHEMA,
        study_id=study_scenario[0] if study_scenario else "",
        scenario=study_scenario[1] if study_scenario else "",
        identities={
            "exports": identities,
            "collected_predictions": file_identity(output),
            "exporter": file_identity(Path(__file__)),
        },
        fields={
            "fm_seeds": list(expected_fm_seeds),
            "row_schema": ROW_SCHEMA,
            "row_count": len(all_rows),
            "combination_count": len(expected),
        },
    )
    receipt_path = atomic_write_receipt(
        output.with_suffix(output.suffix + ".receipt.json"), receipt
    )
    verify_receipt(receipt_path, expected_schema=COLLECTION_SCHEMA)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--nested-predictions", required=True)
    export.add_argument("--diagnostic-receipt", required=True)
    export.add_argument("--destination", required=True)
    export.add_argument("--fm-seed", type=int, required=True)
    export.add_argument("--arm", choices=ARMS, required=True)
    export.add_argument("--cancer", choices=CANCERS, required=True)
    export.add_argument("--head-seed", type=int, choices=HEAD_SEEDS, required=True)
    export.add_argument("--deployment-gate-receipt", required=True)
    export.add_argument("--loader-root-receipt", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--exports", nargs="+", required=True)
    collect.add_argument("--destination", required=True)
    collect.add_argument("--fm-seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export_cell(
            nested_predictions=args.nested_predictions,
            diagnostic_receipt=args.diagnostic_receipt,
            destination=args.destination,
            fm_seed=args.fm_seed, arm=args.arm, cancer=args.cancer,
            head_seed=args.head_seed,
            deployment_gate_receipt=args.deployment_gate_receipt,
            loader_root_receipt=args.loader_root_receipt,
        )
    else:
        collect_exports(
            args.exports, destination=args.destination,
            expected_fm_seeds=args.fm_seeds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
