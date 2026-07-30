"""Gate-bound loader for the fixed BRCA/LUAD matched diagnostic.

The module has no import-time data discovery. Real metadata, outcomes, and tile
files are opened only from the authorization manifest after deployment gate
verification.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.matched_cancer_diagnostic_20260730.deployment import (
    load_contract,
    verify_gate,
)
from tools.matched_cancer_diagnostic_20260730.runner import (
    PatientRecord,
    load_frozen_representation,
    run_paired_diagnostic,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    load_receipt,
    topology_sha256,
    verify_receipt,
)


SOURCE_SCHEMA = "matched-cancer-diagnostic-source-bundle/v1"
TILE_LEDGER_SCHEMA = "matched-cancer-diagnostic-tile-ledger/v1"
COHORT_SCHEMA = "matched-cancer-diagnostic-cohort/v1"
LOADER_ROOT_SCHEMA = "matched-cancer-diagnostic-loader-root/v1"
SPLIT_SEED = 288_850_999
TARGET_FOLD = "target"
TILE_VIEW_SCHEMA = "matched-cancer-diagnostic-tile-view/v1"


def verify_tile_view_receipt(
    path: str | Path,
    requested_cancer_directories: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Verify receipt topology and exact hardlink-view membership without labels."""
    if not requested_cancer_directories:
        raise ValueError("at least one requested cancer view is required")
    receipt_path = Path(path).resolve()
    receipt = load_receipt(receipt_path)
    expected_fields = {
        "schema", "study_id", "scenario", "identities", "topology_sha256",
        "destination_root", "source_root", "eligible_races", "target_fold",
        "expected_patient_counts", "patient_counts", "slide_counts",
        "file_count", "cohort_summaries", "fold_columns_read",
        "parquet_columns_read", "outcomes_opened", "inventory",
    }
    if set(receipt) != expected_fields or receipt.get("schema") != TILE_VIEW_SCHEMA:
        raise ValueError("tile-view receipt fields/schema differ")
    if (
        receipt.get("study_id") != "matched_cancer_stage_20260730"
        or receipt.get("scenario")
        != "brca_luad_black_white_calibration_seed32001"
        or receipt.get("eligible_races") != ["Black", "White"]
        or receipt.get("target_fold") != "target"
        or receipt.get("fold_columns_read")
        != ["patient_barcode", "fold", "race"]
        or receipt.get("parquet_columns_read") != ["slide_path"]
        or receipt.get("outcomes_opened") is not False
    ):
        raise ValueError("tile-view semantic contract differs")
    expected_counts = {"BRCA": 328, "LUAD": 281}
    expected_summaries = {
        "BRCA": {
            "target_rows": 334,
            "eligible_rows": 328,
            "excluded_races": {
                "asian": 5,
                "american indian or alaska native": 1,
            },
            "race_counts": {"Black": 118, "White": 210},
        },
        "LUAD": {
            "target_rows": 281,
            "eligible_rows": 281,
            "excluded_races": {},
            "race_counts": {"Black": 40, "White": 241},
        },
    }
    if (
        receipt.get("expected_patient_counts") != expected_counts
        or receipt.get("patient_counts") != expected_counts
        or receipt.get("cohort_summaries") != expected_summaries
    ):
        raise ValueError("tile-view cohort contract differs")
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "frozen_folds", "source_parquets", "view_parquets"
    }:
        raise ValueError("tile-view identity topology differs")
    if receipt.get("topology_sha256") != topology_sha256(identities):
        raise ValueError("tile-view topology SHA-256 differs")
    inventory = receipt.get("inventory")
    source_files = identities["source_parquets"]
    view_files = identities["view_parquets"]
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != set(source_files)
        or set(inventory) != set(view_files)
        or receipt.get("file_count") != len(inventory)
    ):
        raise ValueError("tile-view inventory key/count topology differs")
    destination_root = Path(receipt["destination_root"]).resolve()
    if destination_root != receipt_path.parent:
        raise ValueError("tile-view receipt is outside its destination root")

    for cancer, requested in requested_cancer_directories.items():
        if cancer not in {"BRCA", "LUAD"}:
            raise ValueError(f"unsupported tile-view cancer {cancer!r}")
        directory = Path(requested)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"{cancer} tile-view directory is invalid")
        directory = directory.resolve(strict=True)
        if directory != destination_root / cancer:
            raise ValueError(f"{cancer} tile-view directory differs from receipt")
        keys = {
            key for key, row in inventory.items()
            if row.get("cancer") == cancer
        }
        expected_paths = {
            Path(view_files[key]["canonical_path"]).resolve() for key in keys
        }
        actual_paths = {
            candidate.resolve()
            for candidate in directory.rglob("*.parquet")
            if candidate.is_file() and not candidate.is_symlink()
        }
        if actual_paths != expected_paths:
            raise ValueError(f"{cancer} tile-view files differ from receipt")
        for key in keys:
            row = inventory[key]
            view_identity = view_files[key]
            source_identity = source_files[key]
            view = Path(view_identity["canonical_path"])
            source = Path(source_identity["canonical_path"])
            expected_relative = Path(str(row["destination_relative"]))
            if (
                expected_relative.parts[0] != cancer
                or destination_root / expected_relative != view
                or row.get("source_basename") != source.name
                or view.name != source.name
                or view_identity["bytes"] != source_identity["bytes"]
                or view_identity["sha256"] != source_identity["sha256"]
            ):
                raise ValueError(f"tile-view record {key} path/identity differs")
            if (
                not view.is_file() or view.is_symlink()
                or not source.is_file() or source.is_symlink()
            ):
                raise ValueError(f"tile-view record {key} is not regular")
            view_stat = view.stat()
            source_stat = source.stat()
            if (
                view_stat.st_size != view_identity["bytes"]
                or source_stat.st_size != source_identity["bytes"]
                or view_stat.st_ino != row.get("view_inode")
                or source_stat.st_ino != row.get("source_inode")
                or view_stat.st_dev != source_stat.st_dev
                or view_stat.st_ino != source_stat.st_ino
                or view_stat.st_nlink < 2
                or source_stat.st_nlink < 2
            ):
                raise ValueError(f"tile-view record {key} hardlink differs")
        if receipt.get("slide_counts", {}).get(cancer) != len(keys):
            raise ValueError(f"{cancer} tile-view slide count differs")
    return receipt


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"cohort destination exists: {path}")
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


def _source_bundle(
    path: str | Path,
    *,
    cancer: str,
    task: str,
    cohort_contract: Mapping[str, Any],
    expected_study_id: str,
    expected_scenario: str,
    expected_amendment_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = verify_receipt(
        path,
        expected_schema=SOURCE_SCHEMA,
        expected_study_id=expected_study_id,
        expected_scenario=expected_scenario,
    )
    if set(receipt.get("identities", {})) != {
        "demographics_csv", "molecular_csv", "frozen_folds_csv",
        "estimand_amendment",
    }:
        raise ValueError("source-bundle identity topology differs")
    if (
        expected_amendment_identity is not None
        and receipt["identities"]["estimand_amendment"]
        != dict(expected_amendment_identity)
    ):
        raise ValueError("source-bundle estimand amendment differs")
    expected = {
        "cancer": cancer,
        "task": task,
        "target_fold": TARGET_FOLD,
        "expected_target_rows": cohort_contract["expected_target_rows"],
        "expected_eligible_patients": cohort_contract[
            "expected_eligible_patients"
        ],
        "expected_exclusions_by_race": cohort_contract[
            "expected_exclusions_by_race"
        ],
        "expected_race_counts": cohort_contract["expected_race_counts"],
        "split_seed": SPLIT_SEED,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"source-bundle {key} drift")
    return receipt


def _tile_ledger(
    path: str | Path,
    tile_directory: str | Path,
    *,
    cancer: str,
    tile_view_receipt: str | Path,
    expected_frozen_folds_identity: Mapping[str, Any],
    expected_patient_count: int,
    expected_study_id: str,
    expected_scenario: str,
) -> dict[str, Any]:
    directory = Path(tile_directory)
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve() != directory.resolve(strict=True)
    ):
        raise ValueError("authorized tile directory is invalid")
    tile_view_path = Path(tile_view_receipt).resolve()
    tile_view = verify_tile_view_receipt(
        tile_view_path, {cancer: directory}
    )
    if tile_view["identities"]["frozen_folds"].get(cancer) != dict(
        expected_frozen_folds_identity
    ):
        raise ValueError("tile view uses a different frozen-fold source")
    production_count = {"BRCA": 328, "LUAD": 281}[cancer]
    if (
        expected_patient_count == production_count
        and tile_view["patient_counts"].get(cancer) != expected_patient_count
    ):
        raise ValueError("tile-view eligible patient count differs")
    receipt = verify_receipt(
        path,
        expected_schema=TILE_LEDGER_SCHEMA,
        expected_study_id=expected_study_id,
        expected_scenario=expected_scenario,
    )
    identities = receipt.get("identities", {})
    if set(identities) != {"files", "tile_view_receipt"} or not isinstance(
        identities["files"], Mapping
    ):
        raise ValueError("tile-ledger identity topology differs")
    if identities["tile_view_receipt"] != file_identity(tile_view_path):
        raise ValueError("tile ledger is not bound to requested tile view")
    if receipt.get("tile_directory") != str(directory.resolve()):
        raise ValueError("tile-ledger directory differs")
    if receipt.get("cancer") != cancer:
        raise ValueError("tile-ledger cancer differs")
    recorded = {
        Path(identity["canonical_path"]).resolve()
        for identity in identities["files"].values()
    }
    discovered = {
        path.resolve()
        for path in directory.rglob("*.parquet")
        if path.is_file() and not path.is_symlink()
    }
    if recorded != discovered or receipt.get("file_count") != len(recorded):
        raise ValueError("tile-ledger inventory differs from tile directory")
    view_identities = tile_view["identities"]["view_parquets"]
    expected_view_identities = {
        key: identity
        for key, identity in view_identities.items()
        if Path(identity["canonical_path"]).parent == directory.resolve()
    }
    if identities["files"] != expected_view_identities:
        raise ValueError("tile ledger differs from tile-view receipt identities")
    return receipt


def _target_rows(path: Path) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {
            "patient_barcode", "fold", "race"
        }.issubset(
            reader.fieldnames
        ):
            raise ValueError("frozen folds CSV lacks patient_barcode/fold/race")
        for line_number, row in enumerate(reader, 2):
            patient = str(row.get("patient_barcode", "")).strip()
            fold = str(row.get("fold", "")).strip()
            if not patient or not fold:
                raise ValueError(f"frozen folds CSV row {line_number} is incomplete")
            if patient in output:
                raise ValueError(f"duplicate frozen-fold patient {patient!r}")
            if fold == TARGET_FOLD:
                output[patient] = {
                    str(key): "" if value is None else str(value)
                    for key, value in row.items()
                }
    return output


def _optional_age(row: Mapping[str, Any]) -> float | None:
    for key in ("age", "age_at_diagnosis", "age_at_index"):
        value = row.get(key)
        if value not in ("", None):
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(result):
                return result
    return None


def _canonical_race(raw_value: Any) -> str:
    raw = str(raw_value).strip().lower()
    if raw in {"white"}:
        return "White"
    if raw in {"black", "black or african american"}:
        return "Black"
    if raw == "asian":
        return "Asian"
    if (
        "american indian" in raw
        or "alaska native" in raw
        or raw in {"aian", "american indian or alaska native"}
    ):
        return "American Indian or Alaska Native"
    raise ValueError(f"unexpected frozen-cohort race value {raw!r}")


def _payload_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tss(patient: str, fold_row: Mapping[str, Any], demo: Mapping[str, Any]) -> str:
    for source in (fold_row, demo):
        for key in ("tss", "tissue_source_site"):
            value = str(source.get(key, "")).strip()
            if value:
                return value
    pieces = patient.split("-")
    if len(pieces) >= 2 and pieces[0] == "TCGA" and pieces[1]:
        return pieces[1]
    raise ValueError(f"patient {patient!r} lacks tissue-source-site metadata")


def prepare_cohort(
    *,
    cancer: str,
    cohort_contract: Mapping[str, Any],
    authorized: Mapping[str, str],
    destination: str | Path,
    load_demographics_fn: Callable[..., Mapping[str, Mapping[str, Any]]] | None = None,
    build_task_cohort_fn: Callable[..., Any] | None = None,
    sensitive_of_fn: Callable[..., Any] | None = None,
    collect_tiles_fn: Callable[..., Any] | None = None,
    tile_view_receipt: str | Path | None = None,
    expected_study_id: str = "matched_cancer_stage_20260730",
    expected_scenario: str = (
        "brca_luad_black_white_calibration_seed32001"
    ),
    expected_amendment_identity: Mapping[str, Any] | None = None,
) -> tuple[list[PatientRecord], list[tuple[Any, bytes]], Path, Path]:
    """Build one exact target cohort; dependency hooks are synthetic-test only."""
    if set(authorized) != {"patient_records", "tile_source", "cohort_ledger"}:
        raise ValueError(f"{cancer} authorized path roles differ")
    task = str(cohort_contract["task"])
    expected_raw = int(cohort_contract["expected_target_rows"])
    expected_n = int(cohort_contract["expected_eligible_patients"])
    source_path = Path(authorized["patient_records"]).resolve()
    source = _source_bundle(
        source_path,
        cancer=cancer,
        task=task,
        cohort_contract=cohort_contract,
        expected_study_id=expected_study_id,
        expected_scenario=expected_scenario,
        expected_amendment_identity=expected_amendment_identity,
    )
    source_ids = source["identities"]
    folds_path = Path(
        source_ids["frozen_folds_csv"]["canonical_path"]
    ).resolve()

    if (
        load_demographics_fn is None
        or build_task_cohort_fn is None
        or sensitive_of_fn is None
        or collect_tiles_fn is None
    ):
        from tools import fairness_eval
        from tools import post_hoc_debias

        load_demographics_fn = (
            load_demographics_fn or fairness_eval.load_demographics
        )
        build_task_cohort_fn = (
            build_task_cohort_fn or post_hoc_debias.build_task_cohort
        )
        sensitive_of_fn = sensitive_of_fn or post_hoc_debias.sensitive_of
        collect_tiles_fn = collect_tiles_fn or post_hoc_debias.collect_tiles

    demo = dict(load_demographics_fn(
        source_ids["demographics_csv"]["canonical_path"], "patient_barcode"
    ))
    molecular = dict(load_demographics_fn(
        source_ids["molecular_csv"]["canonical_path"], "patient_barcode"
    ))
    label_of, _, task_cohort = build_task_cohort_fn(
        task, demo, molecular, None, None, lambda _: None
    )
    sensitive = sensitive_of_fn(demo)
    frozen = _target_rows(folds_path)
    raw_target = sorted(set(frozen) & set(task_cohort))
    if len(raw_target) != expected_raw:
        raise ValueError(
            f"{cancer} raw target intersection N={len(raw_target)} "
            f"!= {expected_raw}"
        )
    raw_races = {
        patient: _canonical_race(frozen[patient].get("race"))
        for patient in raw_target
    }
    exclusions: dict[str, int] = {}
    target = []
    for patient in raw_target:
        category = raw_races[patient]
        if category in {"Black", "White"}:
            target.append(patient)
        else:
            exclusions[category] = exclusions.get(category, 0) + 1
    if exclusions != dict(cohort_contract["expected_exclusions_by_race"]):
        raise ValueError(
            f"{cancer} exclusion breakdown {exclusions} differs from contract"
        )
    if len(target) != expected_n:
        raise ValueError(
            f"{cancer} eligible Black/White N={len(target)} != {expected_n}"
        )
    races = {patient: str(raw_races[patient]) for patient in target}
    for patient, race in raw_races.items():
        sensitive.setdefault(patient, {})["race"] = race
    race_counts = {
        race: sum(races[patient] == race for patient in target)
        for race in ("Black", "White")
    }
    if race_counts != dict(cohort_contract["expected_race_counts"]):
        raise ValueError(
            f"{cancer} eligible race counts {race_counts} differ from contract"
        )
    labels = np.asarray([int(label_of[patient]) for patient in target], dtype=int)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{cancer} target does not contain both binary outcomes")
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SPLIT_SEED
    )
    outer_fold: dict[str, int] = {}
    for fold, (_, validation) in enumerate(splitter.split(target, labels)):
        for index in validation:
            outer_fold[target[int(index)]] = fold
    if set(outer_fold.values()) != set(range(5)):
        raise ValueError(f"{cancer} deterministic outer folds are incomplete")

    patients = []
    for patient in target:
        demo_row = demo.get(patient, {})
        fold_row = frozen[patient]
        patients.append(PatientRecord(
            patient_id=patient,
            y_true=int(label_of[patient]),
            race=str(races[patient]),
            tss=_tss(patient, fold_row, demo_row),
            outer_fold=outer_fold[patient],
            sex=sensitive.get(patient, {}).get("sex"),
            age=_optional_age(demo_row),
            site=(
                str(fold_row.get("site") or fold_row.get("hospital") or "").strip()
                or None
            ),
        ))

    tile_directory = Path(authorized["tile_source"]).resolve()
    tile_ledger_path = Path(authorized["cohort_ledger"]).resolve()
    if tile_view_receipt is None:
        raise ValueError("tile-view receipt is required")
    _tile_ledger(
        tile_ledger_path,
        tile_directory,
        cancer=cancer,
        tile_view_receipt=tile_view_receipt,
        expected_frozen_folds_identity=source_ids["frozen_folds_csv"],
        expected_patient_count=expected_n,
        expected_study_id=expected_study_id,
        expected_scenario=expected_scenario,
    )
    task_tiles, pool_tiles = collect_tiles_fn(
        str(tile_directory),
        set(target),
        sensitive,
        "race",
        "task_only",
        2**31 - 1,
        0,
        0,
        lambda _: None,
    )
    if pool_tiles:
        raise ValueError("task-only tile collection unexpectedly returned a pool")
    tile_patients = {str(patient) for patient, _ in task_tiles}
    if tile_patients != set(target):
        missing = len(set(target) - tile_patients)
        extra = len(tile_patients - set(target))
        raise ValueError(
            f"{cancer} tile coverage differs (missing={missing}, extra={extra})"
        )
    if any(not isinstance(payload, (bytes, bytearray, memoryview)) or not payload
           for _, payload in task_tiles):
        raise ValueError(f"{cancer} tile collection contains invalid bytes")

    destination_path = Path(destination).resolve()
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            f"{cancer} cohort destination exists: {destination_path}"
        )
    rows = [
        {
            "patient_id": patient.patient_id,
            "y_true": patient.y_true,
            "race": patient.race,
            "tss": patient.tss,
            "outer_fold": patient.outer_fold,
            "sex": patient.sex,
            "age": patient.age,
            "site": patient.site,
        }
        for patient in patients
    ]
    cohort_path = _atomic_jsonl(destination_path / "cohort.jsonl", rows)
    receipt = build_receipt(
        schema=COHORT_SCHEMA,
        study_id=source["study_id"],
        scenario=source["scenario"],
        identities={
            "source_bundle": file_identity(source_path),
            "tile_ledger": file_identity(tile_ledger_path),
            "cohort_records": file_identity(cohort_path),
            "loader": file_identity(Path(__file__)),
        },
        fields={
            "cancer": cancer,
            "task": task,
            "patient_count": len(patients),
            "raw_target_count": len(raw_target),
            "eligible_patient_count": len(patients),
            "tile_count": len(task_tiles),
            "split_seed": SPLIT_SEED,
            "fold_counts": {
                str(fold): sum(p.outer_fold == fold for p in patients)
                for fold in range(5)
            },
            "exclusions_by_race": exclusions,
            "race_counts": race_counts,
            "eligible_patient_ids_sha256": _payload_sha256(
                [patient.patient_id for patient in patients]
            ),
            "fold_sha256": _payload_sha256([
                {
                    "patient_id": patient.patient_id,
                    "outer_fold": patient.outer_fold,
                }
                for patient in patients
            ]),
        },
    )
    receipt_path = atomic_write_receipt(
        destination_path / "COHORT_RECEIPT.json", receipt
    )
    verify_receipt(receipt_path, expected_schema=COHORT_SCHEMA)
    return patients, list(task_tiles), cohort_path, receipt_path


def load(
    *,
    contract: Mapping[str, Any],
    cancers: Sequence[str],
    authorized_paths: Mapping[str, Any],
    output_root: str | Path,
    gate_receipt: str | Path,
) -> Path:
    """Run the exact B/P/H diagnostic for both gate-bound cancers."""
    gate = verify_gate(gate_receipt)
    bound_contract = load_contract(
        gate["identities"]["deployment_contract"]["canonical_path"]
    )
    if dict(contract) != bound_contract:
        raise ValueError("loader contract differs from deployment gate")
    if tuple(cancers) != ("BRCA", "LUAD"):
        raise ValueError("loader requires exact BRCA/LUAD order")
    if authorized_paths.get("schema") != (
        "matched-cancer-diagnostic-authorization/v1"
    ):
        raise ValueError("loader authorization schema differs")
    cohorts = authorized_paths.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {"BRCA", "LUAD"}:
        raise ValueError("loader authorization cancer topology differs")
    tile_view_receipt = authorized_paths.get("tile_view_receipt")
    if not isinstance(tile_view_receipt, str) or not tile_view_receipt:
        raise ValueError("loader authorization lacks tile-view receipt")
    root = Path(output_root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"loader output root exists: {root}")
    root.mkdir(parents=True)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        representations = {
            arm: load_frozen_representation(
                gate["identities"]["completion_receipts"][arm]["canonical_path"],
                device=device,
                expected_study_id=gate["study_id"],
                expected_scenario=gate["scenario"],
            )
            for arm in ("B", "P", "H")
        }
        cohort_receipts: dict[str, dict[str, Any]] = {}
        diagnostic_receipts: dict[str, dict[str, Any]] = {}
        for cancer in cancers:
            patients, tiles, cohort_source, cohort_receipt = prepare_cohort(
                cancer=cancer,
                cohort_contract=contract["cohorts"][cancer],
                authorized=cohorts[cancer],
                destination=root / "cohorts" / cancer,
                tile_view_receipt=tile_view_receipt,
                expected_study_id=gate["study_id"],
                expected_scenario=gate["scenario"],
                expected_amendment_identity=gate["identities"][
                    "estimand_amendment"
                ],
            )
            diagnostic = run_paired_diagnostic(
                representations=representations,
                tiles=tiles,
                patients=patients,
                task_id=contract["cohorts"][cancer]["task"],
                cohort_source=cohort_source,
                output_root=root / "diagnostics" / cancer,
                cache_dir=root / "cache",
            )
            cohort_receipts[cancer] = file_identity(cohort_receipt)
            diagnostic_receipts[cancer] = file_identity(diagnostic)
    except Exception:
        # Partial output is deliberately left as audit evidence.
        raise

    receipt = build_receipt(
        schema=LOADER_ROOT_SCHEMA,
        study_id=gate["study_id"],
        scenario=gate["scenario"],
        identities={
            "deployment_gate": file_identity(gate_receipt),
            "cohorts": cohort_receipts,
            "diagnostics": diagnostic_receipts,
            "loader": file_identity(Path(__file__)),
        },
        fields={
            "status": "complete",
            "representation_seed": gate["representation_seed"],
            "cancers": list(cancers),
            "arms": ["B", "P", "H"],
        },
    )
    result = atomic_write_receipt(root / "LOADER_ROOT_RECEIPT.json", receipt)
    verify_receipt(result, expected_schema=LOADER_ROOT_SCHEMA)
    return result
