#!/usr/bin/env python3
"""Download, validate, and receipt the datasets used by this project.

The default ``all`` command prepares the public pretraining tiles and TCGA
clinical metadata.  Pass ``--download-downstream`` to additionally fetch the
much larger downstream TCGA-12K tile mirror.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from build_fino_metadata import build as build_fino_metadata
from pathology_fairness.data_contracts import (
    DATASETS,
    DOWNSTREAM_REVISION,
    PRETRAINING_REVISION,
    RECEIPT_SCHEMA,
    dataset_files,
    local_inventory,
)


PRETRAINING_REPO = DATASETS["pretraining"]["repo"]
PRETRAINING_SHARDS = DATASETS["pretraining"]["expected_files"]
DOWNSTREAM_REPO = DATASETS["downstream"]["repo"]
DOWNSTREAM_FILES = DATASETS["downstream"]["expected_files"]
DOWNSTREAM_ROWS = DATASETS["downstream"]["expected_rows"]
DOWNSTREAM_MANIFEST_SHA256 = DATASETS["downstream"]["manifest_sha256"]
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
FOLD_SEED = 1337


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize(value):
    """Normalize unordered API objects before hashing them for provenance."""
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return value


def _parquet_files(root: Path, dataset: str) -> list[Path]:
    return dataset_files(root, dataset)


def validate_tiles(root: Path, dataset: str, deep: bool = False) -> dict:
    """Validate file completeness, Parquet schemas, and optional sample values."""
    import pyarrow.parquet as pq

    root = root.resolve()
    files = _parquet_files(root, dataset)
    if not files:
        raise FileNotFoundError(f"no {dataset} Parquet files found under {root}")
    if dataset == "pretraining":
        expected = [f"shard-{index:05d}.parquet" for index in range(PRETRAINING_SHARDS)]
        observed = [path.name for path in files]
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise ValueError(
                f"pretraining shard set is incomplete: found={len(observed)} "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    inventory = local_inventory(root, dataset)
    spec = DATASETS[dataset]
    if len(files) != spec["expected_files"]:
        raise ValueError(
            f"{dataset} file set is incomplete: found={len(files)} "
            f"expected={spec['expected_files']}"
        )
    if inventory["manifest_sha256"] != spec["manifest_sha256"]:
        raise ValueError(
            f"{dataset} file manifest does not match pinned revision: "
            f"observed={inventory['manifest_sha256']} expected={spec['manifest_sha256']}"
        )
    if inventory["total_bytes"] != spec["expected_bytes"]:
        raise ValueError(
            f"{dataset} byte total does not match pinned revision: "
            f"observed={inventory['total_bytes']} expected={spec['expected_bytes']}"
        )

    expected_schema = spec["schema"]
    total_rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        observed_schema = {
            field.name: str(field.type) for field in parquet.schema_arrow
        }
        if observed_schema != expected_schema:
            raise ValueError(
                f"{path}: schema {observed_schema} does not match {expected_schema}"
            )
        if parquet.metadata.num_rows <= 0:
            raise ValueError(f"{path}: empty Parquet file")
        if deep:
            payload_columns = (
                ["path", "jpeg"] if dataset == "pretraining"
                else ["slide_path", "image_bytes"]
            )
            sample = parquet.read_row_group(0, columns=payload_columns).slice(0, 1)
            if sample.num_rows != 1 or any(sample.column(i)[0].as_py() in (None, b"", "")
                                           for i in range(sample.num_columns)):
                raise ValueError(f"{path}: invalid first-row sample")
        total_rows += parquet.metadata.num_rows

    if total_rows != spec["expected_rows"]:
        raise ValueError(
            f"{dataset} row count does not match pinned revision: "
            f"observed={total_rows} expected={spec['expected_rows']}"
        )
    return {
        "schema": RECEIPT_SCHEMA,
        "dataset": dataset,
        "source": {
            "repo_id": spec["repo"],
            "revision": spec["revision"],
            "repo_type": "dataset",
            "lfs_manifest_sha256": spec["lfs_manifest_sha256"],
        },
        "local": {
            **inventory,
            "total_rows": total_rows,
            "schema": expected_schema,
            "deep_sample_validation": bool(deep),
        },
    }


def download_tiles(root: Path, dataset: str, workers: int, deep: bool = False) -> dict:
    """Download a pinned public dataset snapshot, resume safely, then validate."""
    from huggingface_hub import snapshot_download

    spec = DATASETS[dataset]
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"downloading {dataset} tiles: {spec['repo']}@{spec['revision']} -> {root}",
        flush=True,
    )
    snapshot_download(
        repo_id=spec["repo"],
        repo_type="dataset",
        revision=spec["revision"],
        local_dir=str(root),
        allow_patterns=spec["patterns"],
        max_workers=workers,
    )
    receipt = validate_tiles(root, dataset, deep=deep)
    _atomic_json(root / "DATASET_RECEIPT.json", receipt)
    print(
        f"validated {receipt['local']['file_count']} files and "
        f"{receipt['local']['total_rows']:,} rows",
        flush=True,
    )
    return receipt


def _gdc_url() -> str:
    filters = {
        "op": "=",
        "content": {"field": "project.program.name", "value": "TCGA"},
    }
    query = urllib.parse.urlencode({
        "filters": json.dumps(filters, separators=(",", ":")),
        "fields": ",".join([
            "submitter_id",
            "project.project_id",
            "demographic.race",
            "demographic.sex_at_birth",
            "demographic.days_to_birth",
            "diagnoses.primary_diagnosis",
        ]),
        "size": 20000,
        "format": "JSON",
    })
    return f"{GDC_CASES_ENDPOINT}?{query}"


def download_gdc_cases() -> list[dict]:
    request = urllib.request.Request(
        _gdc_url(), headers={"User-Agent": "pathology-fairness-timing/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    hits = payload.get("data", {}).get("hits", [])
    total = int(payload.get("data", {}).get("pagination", {}).get("total", -1))
    if not hits or total != len(hits):
        raise RuntimeError(f"incomplete GDC response: received={len(hits)} expected={total}")
    return hits


def _brca_classification(diagnoses: list[dict]) -> tuple[int | None, str]:
    names = {str(row.get("primary_diagnosis", "")).strip().lower()
             for row in diagnoses}
    ductal = any("duct" in name and "carcinoma" in name for name in names)
    lobular = any("lobular" in name and "carcinoma" in name for name in names)
    if ductal and not lobular:
        return 0, "IDC"
    if lobular and not ductal:
        return 1, "ILC"
    return None, "ambiguous" if ductal and lobular else "unclassified"


def _brca_label(diagnoses: list[dict]) -> int | None:
    return _brca_classification(diagnoses)[0]


def clinical_rows(cases: list[dict]) -> list[dict]:
    """Flatten public GDC cases and derive the three downstream task labels."""
    output = []
    for case in cases:
        patient_id = str(case.get("submitter_id", "")).strip()
        project_id = str((case.get("project") or {}).get("project_id", ""))
        if not patient_id.startswith("TCGA-") or not project_id.startswith("TCGA-"):
            continue
        cancer = project_id.removeprefix("TCGA-")
        demographic = case.get("demographic") or {}
        raw_race = str(demographic.get("race", "")).strip()
        normalized_race = raw_race.lower()
        race = {
            "white": "White",
            "black or african american": "Black",
            "asian": "Asian",
        }.get(normalized_race, "")
        if race:
            race_status = "mapped"
        elif not normalized_race:
            race_status = "missing"
        elif normalized_race in {"not reported", "unknown", "not allowed to collect"}:
            race_status = "not_reported"
        else:
            race_status = "unsupported_category"
        days_to_birth = demographic.get("days_to_birth")
        age = ""
        if isinstance(days_to_birth, (int, float)):
            age = round(abs(float(days_to_birth)) / 365.25, 4)
        labels = {"nsclc": None, "glioma": None, "brca": None}
        diagnosis_names = sorted({
            str(item.get("primary_diagnosis", "")).strip()
            for item in (case.get("diagnoses") or [])
            if str(item.get("primary_diagnosis", "")).strip()
        })
        brca_subtype = ""
        if cancer in {"LUAD", "LUSC"}:
            labels["nsclc"] = 0 if cancer == "LUAD" else 1
        if cancer in {"LGG", "GBM"}:
            labels["glioma"] = 0 if cancer == "LGG" else 1
        if cancer == "BRCA":
            labels["brca"], brca_subtype = _brca_classification(
                case.get("diagnoses") or []
            )
        row = {
            "patient_barcode": patient_id,
            "cancer": cancer,
            "cancer_type": cancer,
            "race": race,
            "race_gdc": raw_race,
            "race_status": race_status,
            "gender": str(demographic.get("sex_at_birth", "")).strip(),
            "age_years": age,
            "primary_diagnoses": "|".join(diagnosis_names),
            "brca_subtype": brca_subtype,
        }
        for task, label in labels.items():
            row[f"label_{task}"] = "" if label is None else label
            row[f"fold_{task}"] = ""
        output.append(row)
    return sorted(output, key=lambda row: row["patient_barcode"])


def assign_folds(rows: list[dict], task: str, n_splits: int = 5,
                 seed: int = FOLD_SEED) -> None:
    label_column = f"label_{task}"
    fold_column = f"fold_{task}"
    selected = [index for index, row in enumerate(rows) if row[label_column] != ""]
    labels = [int(rows[index][label_column]) for index in selected]
    if len(set(labels)) != 2:
        raise ValueError(f"{task}: expected two label classes")
    rng = random.Random(seed)
    for label in sorted(set(labels)):
        members = [index for index in selected
                   if int(rows[index][label_column]) == label]
        if len(members) < n_splits:
            raise ValueError(
                f"{task}: class {label} has {len(members)} patients, fewer than "
                f"n_splits={n_splits}"
            )
        by_race: dict[str, list[int]] = {}
        for index in members:
            by_race.setdefault(rows[index]["race"] or "__missing__", []).append(index)
        strata: list[list[int]] = []
        sparse: list[int] = []
        for race in sorted(by_race):
            if len(by_race[race]) >= n_splits:
                strata.append(by_race[race])
            else:
                sparse.extend(by_race[race])
        if sparse:
            strata.append(sparse)
        for stratum in strata:
            rng.shuffle(stratum)
            offset = rng.randrange(n_splits)
            for position, index in enumerate(stratum):
                rows[index][fold_column] = (position + offset) % n_splits


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def validate_clinical_receipt(metadata_dir: Path, holdout_task: str,
                              fino_path: Path) -> dict:
    receipt_path = metadata_dir / "METADATA_RECEIPT.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError(f"missing regular metadata receipt {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("holdout_task") != holdout_task
            or int(receipt.get("fold_seed", -1)) != FOLD_SEED):
        raise ValueError("metadata receipt does not match the requested study contract")
    paths = {
        "demographics_csv": metadata_dir / "tcga_demographics.csv",
        "holdout_file": metadata_dir / "downstream_holdout.txt",
        "fino_meta": fino_path,
    }
    for role, path in paths.items():
        artifact = (receipt.get("outputs") or {}).get(role) or {}
        if (not path.is_file() or path.is_symlink()
                or artifact.get("filename") != path.name
                or artifact.get("sha256") != _sha256_file(path)):
            raise ValueError(f"metadata artifact no longer matches receipt: {role}")
    return receipt


def prepare_clinical(metadata_dir: Path, holdout_task: str,
                     fino_path: Path | None = None,
                     refresh: bool = False) -> dict:
    fino_path = fino_path or metadata_dir.parent / "pretraining_tiles" / "fino_meta.json"
    receipt_path = metadata_dir / "METADATA_RECEIPT.json"
    existing = [
        metadata_dir / "tcga_demographics.csv",
        metadata_dir / "downstream_holdout.txt",
        fino_path,
        receipt_path,
    ]
    if not refresh and receipt_path.exists():
        return validate_clinical_receipt(metadata_dir, holdout_task, fino_path)
    if not refresh and any(path.exists() for path in existing):
        raise FileExistsError(
            "refusing to overwrite incomplete or unreceipted metadata; move it "
            "aside or pass --refresh-metadata explicitly"
        )
    cases = download_gdc_cases()
    source_sha256 = _sha256_bytes(
        json.dumps(_canonicalize(cases), sort_keys=True,
                   separators=(",", ":")).encode()
    )
    rows = clinical_rows(cases)
    for task in ("nsclc", "glioma", "brca"):
        assign_folds(rows, task)

    demographics_path = metadata_dir / "tcga_demographics.csv"
    write_csv(demographics_path, rows)
    holdout = sorted(
        row["patient_barcode"] for row in rows
        if row[f"label_{holdout_task}"] != ""
    )
    holdout_path = metadata_dir / "downstream_holdout.txt"
    _atomic_text(holdout_path, "\n".join(holdout) + "\n")

    fino = build_fino_metadata(
        rows, "patient_barcode", discrete=["cancer", "race"], continuous=[]
    )
    _atomic_json(fino_path, fino)
    counts = {
        task: sum(row[f"label_{task}"] != "" for row in rows)
        for task in ("nsclc", "glioma", "brca")
    }
    task_classes = {
        task: {
            str(label): sum(row[f"label_{task}"] == label for row in rows)
            for label in (0, 1)
        }
        for task in ("nsclc", "glioma", "brca")
    }
    brca_subtypes = {
        subtype: sum(row["brca_subtype"] == subtype for row in rows)
        for subtype in ("IDC", "ILC", "ambiguous", "unclassified")
    }
    race_availability = {
        cohort: {
            status: sum(
                row["race_status"] == status
                and (cohort == "all_tcga" or row[f"label_{cohort}"] != "")
                for row in rows
            )
            for status in ("mapped", "missing", "not_reported", "unsupported_category")
        }
        for cohort in ("all_tcga", "nsclc", "glioma", "brca")
    }
    race_source_counts = {}
    for row in rows:
        value = row["race_gdc"] or "<missing>"
        race_source_counts[value] = race_source_counts.get(value, 0) + 1
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "endpoint": GDC_CASES_ENDPOINT,
            "query_url": _gdc_url(),
            "program": "TCGA",
            "canonical_response_sha256": source_sha256,
        },
        "fold_seed": FOLD_SEED,
        "patients": len(rows),
        "task_patients": counts,
        "task_class_counts": task_classes,
        "brca_subtype_counts": brca_subtypes,
        "race_availability_counts": race_availability,
        "race_source_counts": dict(sorted(race_source_counts.items())),
        "holdout_task": holdout_task,
        "holdout_patients": len(holdout),
        "outputs": {
            "demographics_csv": {
                "filename": demographics_path.name,
                "sha256": _sha256_file(demographics_path),
            },
            "holdout_file": {
                "filename": holdout_path.name,
                "sha256": _sha256_file(holdout_path),
            },
            "fino_meta": {
                "filename": fino_path.name,
                "sha256": _sha256_file(fino_path),
            },
        },
    }
    _atomic_json(metadata_dir / "METADATA_RECEIPT.json", receipt)
    return receipt


def _single_slide_path(parquet_path: Path) -> str:
    """Read one validated slide identifier, preferring Parquet statistics."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    column_index = parquet.schema_arrow.get_field_index("slide_path")
    values = set()
    for row_group in range(parquet.num_row_groups):
        column = parquet.metadata.row_group(row_group).column(column_index)
        statistics = column.statistics
        if statistics and statistics.has_min_max and statistics.min == statistics.max:
            value = statistics.min
            values.add(value.decode() if isinstance(value, bytes) else str(value))
        else:
            values.update(
                str(value) for value in parquet.read_row_group(
                    row_group, columns=["slide_path"]
                ).column(0).to_pylist()
            )
        if len(values) > 1:
            raise ValueError(f"{parquet_path}: contains more than one slide_path")
    if len(values) != 1:
        raise ValueError(f"{parquet_path}: has no slide_path")
    return next(iter(values))


def prepare_cohort_receipt(downstream_dir: Path, metadata_dir: Path) -> dict:
    """Crosswalk pinned slides to every labeled task cohort and receipt coverage."""
    from pathology_fairness.data_contracts import validate_dataset_receipt

    dataset_identity = validate_dataset_receipt(downstream_dir, "downstream")
    metadata_path = metadata_dir / "tcga_demographics.csv"
    metadata_receipt_path = metadata_dir / "METADATA_RECEIPT.json"
    metadata_receipt = json.loads(metadata_receipt_path.read_text())
    expected_demographics_sha = (
        (metadata_receipt.get("outputs") or {}).get("demographics_csv") or {}
    ).get("sha256")
    if (metadata_receipt.get("schema") != RECEIPT_SCHEMA
            or not metadata_path.is_file()
            or expected_demographics_sha != _sha256_file(metadata_path)):
        raise ValueError("demographics do not match the metadata receipt")
    with metadata_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    slide_paths = [_single_slide_path(path) for path in _parquet_files(
        downstream_dir.resolve(), "downstream"
    )]
    if len(set(slide_paths)) != len(slide_paths):
        raise ValueError("downstream mirror contains duplicate slide paths")
    slide_patients = []
    for slide_path in slide_paths:
        match = re.match(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", Path(slide_path).stem)
        if not match:
            raise ValueError(f"cannot parse TCGA patient from slide path: {slide_path}")
        slide_patients.append(match.group(1))
    tiled_patients = set(slide_patients)

    tasks = {}
    for task in ("brca", "nsclc", "glioma"):
        labeled = sorted(
            row["patient_barcode"] for row in rows if row[f"label_{task}"] != ""
        )
        available = sorted(set(labeled) & tiled_patients)
        missing = sorted(set(labeled) - tiled_patients)
        available_set = set(available)
        tasks[task] = {
            "labeled_patients": len(labeled),
            "patients_with_tiles": len(available),
            "coverage_fraction": (
                len(available) / len(labeled) if labeled else None
            ),
            "missing_patients": len(missing),
            "missing_patient_ids": missing,
            "missing_patient_ids_sha256": _sha256_bytes(
                ("\n".join(missing) + "\n").encode()
            ),
            "slides": sum(patient in available_set for patient in slide_patients),
        }
    receipt = {
        "schema": "pathology-fairness-cohort/v1",
        "inputs": {
            "downstream_dataset_receipt_sha256": dataset_identity["receipt_sha256"],
            "demographics_sha256": _sha256_file(metadata_path),
            "metadata_receipt_sha256": _sha256_file(
                metadata_receipt_path
            ),
            "metadata_response_sha256": metadata_receipt["source"][
                "canonical_response_sha256"
            ],
        },
        "tile_patients": len(tiled_patients),
        "slides": len(slide_paths),
        "tasks": tasks,
    }
    _atomic_json(metadata_dir / "COHORT_RECEIPT.json", receipt)
    return receipt


def _workers(value: int | None) -> int:
    return value or min(16, os.cpu_count() or 8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser(
        "all", help="prepare pretraining tiles and clinical metadata"
    )
    all_parser.add_argument("--pretraining-dir", type=Path,
                            default=Path("data/pretraining_tiles"))
    all_parser.add_argument("--downstream-dir", type=Path,
                            default=Path("data/downstream_tiles"))
    all_parser.add_argument("--metadata-dir", type=Path,
                            default=Path("data/metadata"))
    all_parser.add_argument("--holdout-task", choices=["brca", "nsclc", "glioma"],
                            default="brca")
    all_parser.add_argument("--download-downstream", action="store_true")
    all_parser.add_argument("--refresh-metadata", action="store_true",
                            help="replace an existing validated clinical snapshot")
    all_parser.add_argument("--workers", type=int, default=None)
    all_parser.add_argument("--deep-validate", action="store_true")

    tiles_parser = subparsers.add_parser("tiles", help="download one pinned tile dataset")
    tiles_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    tiles_parser.add_argument("--dest", type=Path, required=True)
    tiles_parser.add_argument("--workers", type=int, default=None)
    tiles_parser.add_argument("--deep-validate", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate local tile data")
    validate_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    validate_parser.add_argument("--dir", type=Path, required=True)
    validate_parser.add_argument("--deep", action="store_true")

    crosswalk_parser = subparsers.add_parser(
        "crosswalk", help="receipt labeled-patient coverage in downstream tiles"
    )
    crosswalk_parser.add_argument("--downstream-dir", type=Path,
                                  default=Path("data/downstream_tiles"))
    crosswalk_parser.add_argument("--metadata-dir", type=Path,
                                  default=Path("data/metadata"))

    clinical_parser = subparsers.add_parser(
        "clinical", help="download public GDC clinical metadata and build folds"
    )
    clinical_parser.add_argument("--metadata-dir", type=Path,
                                 default=Path("data/metadata"))
    clinical_parser.add_argument("--holdout-task", choices=["brca", "nsclc", "glioma"],
                                 default="brca")
    clinical_parser.add_argument("--fino-out", type=Path,
                                 default=Path("data/pretraining_tiles/fino_meta.json"))
    clinical_parser.add_argument("--refresh-metadata", action="store_true",
                                 help="replace an existing validated clinical snapshot")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "tiles":
        download_tiles(args.dest, args.dataset, _workers(args.workers),
                       deep=args.deep_validate)
    elif args.command == "validate":
        receipt = validate_tiles(args.dir, args.dataset, deep=args.deep)
        _atomic_json(args.dir / "DATASET_RECEIPT.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.command == "crosswalk":
        receipt = prepare_cohort_receipt(args.downstream_dir, args.metadata_dir)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.command == "clinical":
        receipt = prepare_clinical(
            args.metadata_dir, args.holdout_task, args.fino_out,
            refresh=args.refresh_metadata,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        download_tiles(args.pretraining_dir, "pretraining", _workers(args.workers),
                       deep=args.deep_validate)
        clinical_receipt = prepare_clinical(
            args.metadata_dir, args.holdout_task,
            args.pretraining_dir / "fino_meta.json",
            refresh=args.refresh_metadata,
        )
        if args.download_downstream:
            download_tiles(args.downstream_dir, "downstream", _workers(args.workers),
                           deep=args.deep_validate)
            prepare_cohort_receipt(args.downstream_dir, args.metadata_dir)
        print(json.dumps(clinical_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
