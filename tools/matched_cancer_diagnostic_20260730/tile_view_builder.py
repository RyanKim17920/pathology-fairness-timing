#!/usr/bin/env python3
"""Build outcome-blind, cohort-only hardlink views of a flat parquet store.

Only three columns are requested from the frozen-fold CSVs and only the
``slide_path`` column in parquet row group zero is read.  In particular, this
builder never requests a molecular label or an image-payload column.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)


SCHEMA = "matched-cancer-diagnostic-tile-view/v1"
CANCERS = ("BRCA", "LUAD")
FOLD_COLUMNS = ("patient_barcode", "fold", "race")
PARQUET_COLUMNS = ("slide_path",)
TARGET_FOLD = "target"
DEFAULT_EXPECTED_PATIENT_COUNTS = {"BRCA": 328, "LUAD": 281}
_PATIENT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _regular(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a non-symlink directory: {path}")
    return path.resolve(strict=True)


def _canonical_race(value: Any) -> str | None:
    raw = str(value).strip().lower()
    if raw == "white":
        return "White"
    if raw in {"black", "black or african american"}:
        return "Black"
    return None


def _load_target_patients(path: Path) -> tuple[set[str], dict[str, Any]]:
    """Read only patient/fold/race and return strict Black/White target IDs."""
    source = _regular(path, label="frozen-fold CSV")
    table = pacsv.read_csv(
        source,
        convert_options=pacsv.ConvertOptions(
            include_columns=list(FOLD_COLUMNS),
            strings_can_be_null=False,
        ),
    )
    if tuple(table.column_names) != FOLD_COLUMNS:
        raise ValueError(
            f"frozen-fold CSV must expose exactly {FOLD_COLUMNS!r}"
        )
    columns = {
        name: table[name].to_pylist()
        for name in FOLD_COLUMNS
    }
    seen: set[str] = set()
    eligible: set[str] = set()
    target_rows = 0
    race_counts = {"Black": 0, "White": 0}
    excluded_races: dict[str, int] = {}
    for row_number, (patient_raw, fold_raw, race_raw) in enumerate(
        zip(
            columns["patient_barcode"],
            columns["fold"],
            columns["race"],
            strict=True,
        ),
        2,
    ):
        patient = str(patient_raw).strip().upper()
        fold = str(fold_raw).strip()
        if not patient or not fold or _PATIENT_RE.fullmatch(patient) is None:
            raise ValueError(
                f"frozen-fold row {row_number} has invalid patient/fold"
            )
        if patient in seen:
            raise ValueError(f"duplicate frozen-fold patient {patient!r}")
        seen.add(patient)
        if fold != TARGET_FOLD:
            continue
        target_rows += 1
        race = _canonical_race(race_raw)
        if race is None:
            excluded = str(race_raw).strip() or "<empty>"
            excluded_races[excluded] = excluded_races.get(excluded, 0) + 1
            continue
        eligible.add(patient)
        race_counts[race] += 1
    return eligible, {
        "target_rows": target_rows,
        "eligible_rows": len(eligible),
        "race_counts": race_counts,
        "excluded_races": dict(sorted(excluded_races.items())),
    }


def _patient_from_slide_path(value: Any, source: Path) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: row-group-0 slide_path is empty/non-string")
    slide_path = value.strip()
    matches = {match.upper() for match in _PATIENT_RE.findall(slide_path)}
    if len(matches) != 1:
        raise ValueError(
            f"{source}: slide_path must contain exactly one TCGA patient ID"
        )
    return next(iter(matches)), slide_path


def _scan_flat_source(
    source_root: Path,
    patient_to_cancer: Mapping[str, str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    slide_paths: dict[str, Path] = {}
    source_names: dict[str, Path] = {}
    for candidate in sorted(source_root.iterdir(), key=lambda path: path.name):
        if candidate.suffix.lower() != ".parquet":
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(
                f"source parquet is not a regular non-symlink file: {candidate}"
            )
        source = candidate.resolve(strict=True)
        parquet = pq.ParquetFile(source)
        if "slide_path" not in parquet.schema_arrow.names:
            raise ValueError(f"{source}: missing required slide_path column")
        if parquet.num_row_groups < 1:
            raise ValueError(f"{source}: parquet has no row groups")
        column = parquet.read_row_group(0, columns=list(PARQUET_COLUMNS))[
            "slide_path"
        ]
        if len(column) < 1:
            raise ValueError(f"{source}: row group zero is empty")
        patient, slide_path = _patient_from_slide_path(column[0].as_py(), source)
        cancer = patient_to_cancer.get(patient)
        if cancer is None:
            continue
        if slide_path in slide_paths:
            raise ValueError(
                "duplicate selected slide_path in "
                f"{slide_paths[slide_path]} and {source}"
            )
        if source.name in source_names:
            raise ValueError(
                "duplicate selected parquet filename in "
                f"{source_names[source.name]} and {source}"
            )
        slide_paths[slide_path] = source
        source_names[source.name] = source
        selected.append(
            {
                "cancer": cancer,
                "patient_id": patient,
                "slide_path": slide_path,
                "source": source,
            }
        )
    if not selected:
        raise ValueError("no cohort-relevant parquet files were found")
    return selected


def _predicted_identity(source_identity: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "canonical_path": str(path),
        "bytes": source_identity["bytes"],
        "sha256": source_identity["sha256"],
    }


def _cleanup_owned_directory(path: Path) -> None:
    if path.exists() and path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)


def build_tile_views(
    *,
    source_root: str | Path,
    frozen_folds: Mapping[str, str | Path],
    destination_root: str | Path,
    expected_patient_counts: Mapping[str, int] | None = None,
    study_id: str = "matched_cancer_stage_20260730",
    scenario: str = "brca_luad_black_white_calibration_seed32001",
) -> Path:
    """Create BRCA/LUAD hardlink views and a fully bound integrity receipt."""
    if set(frozen_folds) != set(CANCERS):
        raise ValueError("frozen_folds must contain exactly BRCA and LUAD")
    expected = dict(
        DEFAULT_EXPECTED_PATIENT_COUNTS
        if expected_patient_counts is None
        else expected_patient_counts
    )
    if set(expected) != set(CANCERS) or any(
        not isinstance(value, int) or value < 1 for value in expected.values()
    ):
        raise ValueError("expected patient counts must be positive BRCA/LUAD ints")

    source = _directory(Path(source_root), label="flat parquet source root")
    requested = Path(destination_root).absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"tile-view destination exists: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = _directory(requested.parent, label="destination parent")

    fold_paths = {
        cancer: _regular(Path(frozen_folds[cancer]), label=f"{cancer} folds")
        for cancer in CANCERS
    }
    patients: dict[str, set[str]] = {}
    cohort_summaries: dict[str, dict[str, Any]] = {}
    for cancer in CANCERS:
        patients[cancer], cohort_summaries[cancer] = _load_target_patients(
            fold_paths[cancer]
        )
        if len(patients[cancer]) != expected[cancer]:
            raise ValueError(
                f"{cancer} eligible patient count {len(patients[cancer])} "
                f"!= expected {expected[cancer]}"
            )
    overlap = patients["BRCA"] & patients["LUAD"]
    if overlap:
        raise ValueError(
            f"cross-cancer frozen-fold patient overlap: {sorted(overlap)!r}"
        )
    patient_to_cancer = {
        patient: cancer
        for cancer in CANCERS
        for patient in patients[cancer]
    }
    selected = _scan_flat_source(source, patient_to_cancer)
    covered = {
        cancer: {
            row["patient_id"] for row in selected if row["cancer"] == cancer
        }
        for cancer in CANCERS
    }
    for cancer in CANCERS:
        missing = patients[cancer] - covered[cancer]
        if missing:
            raise ValueError(
                f"{cancer} missing parquet coverage for {len(missing)} patients: "
                f"{sorted(missing)!r}"
            )

    stage = Path(
        tempfile.mkdtemp(prefix=f".{requested.name}.building.", dir=parent)
    )
    published = False
    try:
        for cancer in CANCERS:
            (stage / cancer).mkdir()
        source_identities: dict[str, dict[str, Any]] = {}
        view_identities: dict[str, dict[str, Any]] = {}
        inventory: dict[str, dict[str, Any]] = {}
        slide_counts = {cancer: 0 for cancer in CANCERS}
        for index, row in enumerate(selected):
            key = f"{index:08d}"
            source_path = row["source"]
            relative = Path(row["cancer"]) / source_path.name
            staged_view = stage / relative
            final_view = requested / relative
            os.link(source_path, staged_view)
            source_stat = os.stat(source_path, follow_symlinks=False)
            view_stat = os.stat(staged_view, follow_symlinks=False)
            if (
                not stat.S_ISREG(source_stat.st_mode)
                or not stat.S_ISREG(view_stat.st_mode)
                or (source_stat.st_dev, source_stat.st_ino)
                != (view_stat.st_dev, view_stat.st_ino)
            ):
                raise ValueError(f"hardlink identity mismatch for {source_path}")
            source_identity = file_identity(source_path)
            source_identities[key] = source_identity
            view_identities[key] = _predicted_identity(
                source_identity, final_view
            )
            inventory[key] = {
                "cancer": row["cancer"],
                "patient_id": row["patient_id"],
                "slide_path": row["slide_path"],
                "source_basename": source_path.name,
                "destination_relative": relative.as_posix(),
                "source_device": source_stat.st_dev,
                "source_inode": source_stat.st_ino,
                "view_device": view_stat.st_dev,
                "view_inode": view_stat.st_ino,
            }
            slide_counts[row["cancer"]] += 1

        receipt = build_receipt(
            schema=SCHEMA,
            study_id=study_id,
            scenario=scenario,
            identities={
                "frozen_folds": {
                    cancer: file_identity(fold_paths[cancer])
                    for cancer in CANCERS
                },
                "source_parquets": source_identities,
                "view_parquets": view_identities,
            },
            fields={
                "source_root": str(source),
                "destination_root": str(requested),
                "target_fold": TARGET_FOLD,
                "eligible_races": ["Black", "White"],
                "fold_columns_read": list(FOLD_COLUMNS),
                "parquet_columns_read": list(PARQUET_COLUMNS),
                "outcomes_opened": False,
                "expected_patient_counts": expected,
                "patient_counts": {
                    cancer: len(covered[cancer]) for cancer in CANCERS
                },
                "slide_counts": slide_counts,
                "cohort_summaries": cohort_summaries,
                "file_count": len(selected),
                "inventory": inventory,
            },
        )
        atomic_write_receipt(stage / "TILE_VIEW_RECEIPT.json", receipt)
        # Recheck immediately before publication.  This prevents replacing a
        # destination that appeared while the (potentially long) source scan
        # and hashing pass was in progress.
        if requested.exists() or requested.is_symlink():
            raise FileExistsError(
                f"tile-view destination appeared during build: {requested}"
            )
        os.replace(stage, requested)
        published = True
        output = requested / "TILE_VIEW_RECEIPT.json"
        verify_tile_view_receipt(output)
        return output
    except Exception:
        _cleanup_owned_directory(requested if published else stage)
        raise


def verify_tile_view_receipt(path: str | Path) -> dict[str, Any]:
    """Revalidate bytes, exact inventory, and source/view inode equality."""
    receipt_path = _regular(Path(path), label="tile-view receipt")
    receipt = verify_receipt(receipt_path, expected_schema=SCHEMA)
    expected_topology = {
        "schema", "study_id", "scenario", "identities", "topology_sha256",
        "source_root", "destination_root", "target_fold", "eligible_races",
        "fold_columns_read", "parquet_columns_read", "outcomes_opened",
        "expected_patient_counts", "patient_counts", "slide_counts",
        "cohort_summaries", "file_count", "inventory",
    }
    if set(receipt) != expected_topology:
        raise ValueError("tile-view receipt field topology drift")
    if (
        receipt["target_fold"] != TARGET_FOLD
        or receipt["eligible_races"] != ["Black", "White"]
        or receipt["fold_columns_read"] != list(FOLD_COLUMNS)
        or receipt["parquet_columns_read"] != list(PARQUET_COLUMNS)
        or receipt["outcomes_opened"] is not False
    ):
        raise ValueError("tile-view outcome-blind contract drift")
    identities = receipt["identities"]
    if set(identities) != {
        "frozen_folds", "source_parquets", "view_parquets"
    } or set(identities["frozen_folds"]) != set(CANCERS):
        raise ValueError("tile-view identity topology drift")
    source_ids = identities["source_parquets"]
    view_ids = identities["view_parquets"]
    inventory = receipt["inventory"]
    if (
        not isinstance(inventory, Mapping)
        or set(source_ids) != set(view_ids)
        or set(source_ids) != set(inventory)
        or receipt["file_count"] != len(inventory)
    ):
        raise ValueError("tile-view inventory topology drift")

    source_root = _directory(Path(receipt["source_root"]), label="source root")
    destination_root = _directory(
        Path(receipt["destination_root"]), label="destination root"
    )
    if receipt_path.parent != destination_root:
        raise ValueError("tile-view receipt is outside its destination root")
    expected_children = {"BRCA", "LUAD", "TILE_VIEW_RECEIPT.json"}
    if {child.name for child in destination_root.iterdir()} != expected_children:
        raise ValueError("tile-view destination has unexpected root entries")

    patient_sets: dict[str, set[str]] = {}
    for cancer in CANCERS:
        fold_path = Path(
            identities["frozen_folds"][cancer]["canonical_path"]
        )
        patient_sets[cancer], _ = _load_target_patients(fold_path)
        if len(patient_sets[cancer]) != receipt["expected_patient_counts"][cancer]:
            raise ValueError(f"{cancer} frozen-fold patient count drift")
    if patient_sets["BRCA"] & patient_sets["LUAD"]:
        raise ValueError("cross-cancer frozen-fold patient overlap")

    discovered: set[Path] = set()
    observed_patients = {cancer: set() for cancer in CANCERS}
    observed_slides = {cancer: 0 for cancer in CANCERS}
    observed_slide_paths: set[str] = set()
    observed_sources: set[Path] = set()
    for key in sorted(inventory):
        row = inventory[key]
        if not isinstance(row, Mapping) or set(row) != {
            "cancer", "patient_id", "slide_path", "source_basename",
            "destination_relative", "source_device", "source_inode",
            "view_device", "view_inode",
        }:
            raise ValueError(f"tile-view inventory row {key} drift")
        cancer = row["cancer"]
        patient = row["patient_id"]
        if cancer not in CANCERS or patient not in patient_sets[cancer]:
            raise ValueError(f"tile-view inventory row {key} cohort contamination")
        source_path = Path(source_ids[key]["canonical_path"])
        view_path = Path(view_ids[key]["canonical_path"])
        if source_path.parent != source_root:
            raise ValueError(f"tile-view source {key} is not in flat source root")
        expected_view = destination_root / cancer / source_path.name
        if (
            view_path != expected_view
            or row["source_basename"] != source_path.name
            or row["destination_relative"]
            != (Path(cancer) / source_path.name).as_posix()
        ):
            raise ValueError(f"tile-view path mapping {key} drift")
        if source_path in observed_sources or row["slide_path"] in observed_slide_paths:
            raise ValueError("duplicate source or slide_path in tile-view inventory")
        observed_sources.add(source_path)
        observed_slide_paths.add(row["slide_path"])
        for candidate, label in (
            (source_path, "source parquet"),
            (view_path, "view parquet"),
        ):
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"{label} is not regular/non-symlink: {candidate}")
        source_stat = os.stat(source_path, follow_symlinks=False)
        view_stat = os.stat(view_path, follow_symlinks=False)
        if (
            (source_stat.st_dev, source_stat.st_ino)
            != (view_stat.st_dev, view_stat.st_ino)
            or row["source_device"] != source_stat.st_dev
            or row["source_inode"] != source_stat.st_ino
            or row["view_device"] != view_stat.st_dev
            or row["view_inode"] != view_stat.st_ino
        ):
            raise ValueError(f"tile-view hardlink identity {key} drift")
        discovered.add(view_path)
        observed_patients[cancer].add(patient)
        observed_slides[cancer] += 1

    actual_files: set[Path] = set()
    for cancer in CANCERS:
        cancer_root = _directory(
            destination_root / cancer, label=f"{cancer} view"
        )
        for candidate in cancer_root.iterdir():
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"unexpected tile-view entry: {candidate}")
            if candidate.suffix.lower() != ".parquet":
                raise ValueError(f"non-parquet tile-view entry: {candidate}")
            actual_files.add(candidate.resolve(strict=True))
    if actual_files != discovered:
        raise ValueError("tile-view on-disk inventory differs from receipt")
    for cancer in CANCERS:
        if observed_patients[cancer] != patient_sets[cancer]:
            raise ValueError(f"{cancer} tile-view patient coverage drift")
        if (
            receipt["patient_counts"][cancer] != len(observed_patients[cancer])
            or receipt["slide_counts"][cancer] != observed_slides[cancer]
        ):
            raise ValueError(f"{cancer} tile-view counts drift")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--brca-frozen-folds", required=True)
    parser.add_argument("--luad-frozen-folds", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--brca-expected-patients", type=int, default=328)
    parser.add_argument("--luad-expected-patients", type=int, default=281)
    args = parser.parse_args(argv)
    result = build_tile_views(
        source_root=args.source_root,
        frozen_folds={
            "BRCA": args.brca_frozen_folds,
            "LUAD": args.luad_frozen_folds,
        },
        destination_root=args.destination_root,
        expected_patient_counts={
            "BRCA": args.brca_expected_patients,
            "LUAD": args.luad_expected_patients,
        },
    )
    print(json.dumps({"tile_view_receipt": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
