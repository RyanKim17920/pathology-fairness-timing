"""Per-seed exporter: schema validation only, never statistical analysis."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from tools import reliable_fairness_head as reliable
from tools.matched_cancer_diagnostic_20260730 import exporter as legacy
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_contract import (
    ARMS,
    CANCERS,
    COHORT_SIZES,
    COLLECTION_SCHEMA,
    EXPORT_SCHEMA,
    HEAD_SEEDS,
    ROW_SCHEMA,
    STUDY_ID,
    TASK_IDS,
    validate_seed,
)
from .diag_deployment import verify_gate
from .diag_loader import verify_loader_result


DIAGNOSTIC_CELL_SCHEMA = "matched-cancer-adapter-diagnostic/v1"
DIAGNOSTIC_ROOT_SCHEMA = "matched-cancer-adapter-diagnostic-root/v1"


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"sealed JSONL destination exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
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
        temporary.unlink(missing_ok=True)
    return path


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = legacy._load_jsonl(path)
    for row in rows:
        legacy.validate_prediction_row(row)
        probability = row["probability"]
        if not math.isfinite(float(probability)):
            raise ValueError("non-finite prediction")
    return rows


def _verify_runtime_imports(gate: Mapping[str, Any]) -> None:
    checks = {
        "legacy_exporter": Path(legacy.__file__ or "").resolve(),
        "reliable_fairness_head": Path(reliable.__file__ or "").resolve(),
    }
    for name, path in checks.items():
        if file_identity(path) != gate["identities"]["sources"][name]:
            raise ValueError(f"runtime import {name} was redirected")


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
) -> Path:
    seed = validate_seed(fm_seed)
    if arm not in ARMS or cancer not in CANCERS or head_seed not in HEAD_SEEDS:
        raise ValueError("export coordinate differs from fixed contract")
    gate_path = Path(deployment_gate_receipt).resolve()
    gate = verify_gate(gate_path)
    _verify_runtime_imports(gate)
    if gate["representation_seed"] != seed:
        raise ValueError("export seed is not gate-bound")
    loader_path = Path(loader_root_receipt).resolve()
    loader = verify_loader_result(loader_path, gate_path)
    cell_path = Path(diagnostic_receipt).resolve()
    cell = verify_receipt(
        cell_path,
        expected_schema=DIAGNOSTIC_CELL_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=gate["scenario"],
    )
    root = verify_receipt(
        loader["identities"]["diagnostics"][cancer]["canonical_path"],
        expected_schema=DIAGNOSTIC_ROOT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=gate["scenario"],
    )
    bound = root["identities"]["cells"][arm][str(head_seed)]
    source = Path(nested_predictions).resolve()
    if (
        file_identity(cell_path) != bound
        or cell.get("arm") != arm
        or cell.get("head_seed") != head_seed
        or cell.get("task_id") != TASK_IDS[cancer]
        or cell.get("identities", {}).get("completion_receipt")
        != gate["identities"]["completion_receipts"][arm]
        or cell.get("identities", {}).get("predictions")
        != file_identity(source)
    ):
        raise ValueError("diagnostic cell coordinate/ancestry differs")
    rich = legacy._load_jsonl(source)
    reliable._validate_nested_prediction_records(rich)
    rows = [
        legacy.prediction_row(
            row, fm_seed=seed, arm=arm, cancer=cancer,
            head_seed=head_seed,
        )
        for row in rich
    ]
    n = COHORT_SIZES[cancer]
    if len(rows) != 5 * n or len({row["patient_id"] for row in rows}) != n:
        raise ValueError("exported patient/row count differs")
    output = _atomic_jsonl(Path(destination).resolve(), rows)
    receipt = build_receipt(
        schema=EXPORT_SCHEMA,
        study_id=STUDY_ID,
        scenario=gate["scenario"],
        identities={
            "diagnostic_receipt": file_identity(cell_path),
            "deployment_gate_receipt": file_identity(gate_path),
            "loader_root_receipt": file_identity(loader_path),
            "nested_predictions": file_identity(source),
            "exported_predictions": file_identity(output),
            "exporter": file_identity(Path(__file__)),
        },
        fields={
            "fm_seed": seed,
            "arm": arm,
            "cancer": cancer,
            "head_seed": head_seed,
            "patient_count": n,
            "row_count": len(rows),
            "row_schema": ROW_SCHEMA,
        },
    )
    atomic_write_receipt(
        output.with_suffix(output.suffix + ".receipt.json"), receipt
    )
    return output


def verify_collection(
    path: str | Path,
    *,
    expected_fm_seed: int,
    deployment_gate_receipt: str | Path | None = None,
    loader_root_receipt: str | Path | None = None,
) -> dict[str, Any]:
    seed = validate_seed(expected_fm_seed)
    source = Path(path).resolve()
    receipt_path = source.with_suffix(source.suffix + ".receipt.json")
    receipt = verify_receipt(
        receipt_path,
        expected_schema=COLLECTION_SCHEMA,
        expected_study_id=STUDY_ID,
    )
    if (
        receipt.get("fm_seed") != seed
        or receipt.get("fm_seeds") != [seed]
        or receipt.get("row_schema") != ROW_SCHEMA
        or receipt.get("row_count") != 36_540
        or receipt.get("combination_count") != 24
        or receipt.get("analysis_performed") is not False
        or receipt["identities"].get("collected_predictions")
        != file_identity(source)
    ):
        raise ValueError("per-seed collection semantic contract differs")
    identities = receipt.get("identities", {})
    if set(identities) != {
        "deployment_gate_receipt", "loader_root_receipt", "exports",
        "collected_predictions", "exporter",
    } or len(identities["exports"]) != 24:
        raise ValueError("per-seed collection identity topology differs")
    gate_path = Path(
        deployment_gate_receipt
        or identities["deployment_gate_receipt"]["canonical_path"]
    ).resolve()
    loader_path = Path(
        loader_root_receipt
        or identities["loader_root_receipt"]["canonical_path"]
    ).resolve()
    if (
        file_identity(gate_path) != identities["deployment_gate_receipt"]
        or file_identity(loader_path) != identities["loader_root_receipt"]
    ):
        raise ValueError("per-seed collection gate/loader identity differs")
    gate = verify_gate(gate_path)
    _verify_runtime_imports(gate)
    loader = verify_loader_result(loader_path, gate_path)
    if gate["representation_seed"] != seed or receipt["scenario"] != gate[
        "scenario"
    ]:
        raise ValueError("per-seed collection seed/scenario differs")
    expected = {
        (seed, arm, cancer, head)
        for arm in ARMS for cancer in CANCERS for head in HEAD_SEEDS
    }
    observed = set()
    export_rows: list[dict[str, Any]] = []
    metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
    patient_sets: dict[str, set[str]] = {}
    for key, bound in identities["exports"].items():
        if not isinstance(bound, Mapping) or set(bound) != {
            "predictions", "receipt"
        }:
            raise ValueError("collection export binding topology differs")
        export_path = Path(bound["predictions"]["canonical_path"]).resolve()
        export_receipt_path = Path(
            bound["receipt"]["canonical_path"]
        ).resolve()
        if (
            file_identity(export_path) != bound["predictions"]
            or file_identity(export_receipt_path) != bound["receipt"]
        ):
            raise ValueError("collection export identity drift")
        export_receipt = verify_receipt(
            export_receipt_path,
            expected_schema=EXPORT_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        coordinate = (
            export_receipt.get("fm_seed"), export_receipt.get("arm"),
            export_receipt.get("cancer"), export_receipt.get("head_seed"),
        )
        diagnostic_root = verify_receipt(
            loader["identities"]["diagnostics"][coordinate[2]][
                "canonical_path"
            ],
            expected_schema=DIAGNOSTIC_ROOT_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        bound_cell = diagnostic_root["identities"]["cells"][
            coordinate[1]
        ][str(coordinate[3])]
        cell_identity = export_receipt["identities"].get(
            "diagnostic_receipt"
        )
        if cell_identity != bound_cell:
            raise ValueError("collection export diagnostic cell differs")
        cell = verify_receipt(
            cell_identity["canonical_path"],
            expected_schema=DIAGNOSTIC_CELL_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        nested_identity = export_receipt["identities"].get(
            "nested_predictions"
        )
        if (
            not isinstance(nested_identity, Mapping)
            or nested_identity != file_identity(
                nested_identity["canonical_path"]
            )
        ):
            raise ValueError("collection nested prediction identity drift")
        if (
            key != "|".join(map(str, coordinate))
            or coordinate in observed
            or export_receipt["identities"].get(
                "deployment_gate_receipt"
            ) != file_identity(gate_path)
            or export_receipt["identities"].get(
                "loader_root_receipt"
            ) != file_identity(loader_path)
            or export_receipt["identities"].get(
                "exported_predictions"
            ) != file_identity(export_path)
            or nested_identity
            != cell.get("identities", {}).get("predictions")
            or export_receipt["identities"].get("exporter")
            != file_identity(Path(__file__))
        ):
            raise ValueError("collection export coordinate/ancestry differs")
        rows = _load_rows(export_path)
        _validate_coordinate_rows(rows, coordinate)
        observed.add(coordinate)
        for row in rows:
            metadata_key = (row["cancer"], row["patient_id"])
            value = (row["y_true"], row["race"], row["fold"])
            if metadata_key in metadata and metadata[metadata_key] != value:
                raise ValueError("patient metadata drifts across exports")
            metadata[metadata_key] = value
            patient_sets.setdefault(row["cancer"], set()).add(
                row["patient_id"]
            )
        export_rows.extend(rows)
    if observed != expected:
        raise ValueError("collection export coordinate matrix differs")
    for cancer, n in COHORT_SIZES.items():
        if len(patient_sets.get(cancer, set())) != n:
            raise ValueError(f"{cancer} patient set differs")
    collected_rows = _load_rows(source)
    if len(collected_rows) != 36_540:
        raise ValueError("collection row count differs")
    canonical = lambda row: json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if sorted(map(canonical, collected_rows)) != sorted(
        map(canonical, export_rows)
    ):
        raise ValueError("collected rows differ from sealed exports")
    return receipt


def _validate_coordinate_rows(
    rows: Sequence[Mapping[str, Any]],
    coordinate: tuple[int, str, str, int],
) -> None:
    if {
        (row["fm_seed"], row["arm"], row["cancer"], row["head_seed"])
        for row in rows
    } != {coordinate}:
        raise ValueError("export rows differ from receipt coordinate")
    n = COHORT_SIZES[coordinate[2]]
    if len(rows) != 5 * n:
        raise ValueError("export coordinate row count differs")
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    if len(by_patient) != n:
        raise ValueError("export coordinate patient count differs")
    for patient_rows in by_patient.values():
        if (
            len(patient_rows) != 5
            or sum(row["role"] == "outer_test" for row in patient_rows) != 1
            or sum(
                row["role"] == "inner_calibration" for row in patient_rows
            ) != 4
            or {row["outer_fold"] for row in patient_rows} != set(range(5))
        ):
            raise ValueError("patient lacks exact nested row topology")


def collect_exports(
    exports: Sequence[str | Path],
    *,
    destination: str | Path,
    expected_fm_seed: int,
    deployment_gate_receipt: str | Path,
    loader_root_receipt: str | Path,
) -> Path:
    seed = validate_seed(expected_fm_seed)
    gate_path = Path(deployment_gate_receipt).resolve()
    loader_path = Path(loader_root_receipt).resolve()
    gate = verify_gate(gate_path)
    _verify_runtime_imports(gate)
    verify_loader_result(loader_path, gate_path)
    if gate["representation_seed"] != seed:
        raise ValueError("collection seed is not gate-bound")
    if len(exports) != 24:
        raise ValueError("per-seed collection requires exactly 24 exports")
    all_rows: list[dict[str, Any]] = []
    identities: dict[str, Any] = {}
    observed: set[tuple[Any, ...]] = set()
    semantic: set[tuple[Any, ...]] = set()
    metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
    patient_sets: dict[str, set[str]] = {}
    for value in exports:
        path = Path(value).resolve()
        export_receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        export_receipt = verify_receipt(
            export_receipt_path,
            expected_schema=EXPORT_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        coordinate = (
            export_receipt.get("fm_seed"), export_receipt.get("arm"),
            export_receipt.get("cancer"), export_receipt.get("head_seed"),
        )
        if (
            export_receipt["identities"].get("deployment_gate_receipt")
            != file_identity(gate_path)
            or export_receipt["identities"].get("loader_root_receipt")
            != file_identity(loader_path)
            or export_receipt["identities"].get("exported_predictions")
            != file_identity(path)
            or coordinate in observed
        ):
            raise ValueError("export ancestry or coordinate differs")
        observed.add(coordinate)
        key = "|".join(map(str, coordinate))
        identities[key] = {
            "predictions": file_identity(path),
            "receipt": file_identity(export_receipt_path),
        }
        rows = _load_rows(path)
        if {
            (row["fm_seed"], row["arm"], row["cancer"], row["head_seed"])
            for row in rows
        } != {coordinate}:
            raise ValueError("export rows differ from receipt coordinate")
        for row in rows:
            row_key = (
                *coordinate, row["patient_id"], row["role"],
                row["outer_fold"],
            )
            if row_key in semantic:
                raise ValueError("duplicate semantic prediction row")
            semantic.add(row_key)
            meta_key = (row["cancer"], row["patient_id"])
            meta = (row["y_true"], row["race"], row["fold"])
            if meta_key in metadata and metadata[meta_key] != meta:
                raise ValueError("patient metadata drifts across coordinates")
            metadata[meta_key] = meta
            patient_sets.setdefault(row["cancer"], set()).add(
                row["patient_id"]
            )
            all_rows.append(row)
    expected = {
        (seed, arm, cancer, head)
        for arm in ARMS for cancer in CANCERS for head in HEAD_SEEDS
    }
    if observed != expected:
        raise ValueError("per-seed coordinate matrix is incomplete")
    for cancer, n in COHORT_SIZES.items():
        if len(patient_sets.get(cancer, set())) != n:
            raise ValueError(f"{cancer} patient set differs")
    if len(all_rows) != 36_540:
        raise ValueError("per-seed collection row count differs")
    all_rows.sort(key=lambda row: (
        row["cancer"], row["arm"], row["head_seed"], row["patient_id"],
        row["role"], row["outer_fold"],
    ))
    output = _atomic_jsonl(Path(destination).resolve(), all_rows)
    receipt = build_receipt(
        schema=COLLECTION_SCHEMA,
        study_id=STUDY_ID,
        scenario=gate["scenario"],
        identities={
            "deployment_gate_receipt": file_identity(gate_path),
            "loader_root_receipt": file_identity(loader_path),
            "exports": identities,
            "collected_predictions": file_identity(output),
            "exporter": file_identity(Path(__file__)),
        },
        fields={
            "fm_seed": seed,
            "fm_seeds": [seed],
            "row_schema": ROW_SCHEMA,
            "row_count": len(all_rows),
            "combination_count": len(expected),
            "analysis_performed": False,
        },
    )
    atomic_write_receipt(
        output.with_suffix(output.suffix + ".receipt.json"), receipt
    )
    verify_collection(
        output,
        expected_fm_seed=seed,
        deployment_gate_receipt=gate_path,
        loader_root_receipt=loader_path,
    )
    return output
