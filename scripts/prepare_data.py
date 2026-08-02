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
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from build_fino_metadata import build as build_fino_metadata


PRETRAINING_REPO = "medarc/nanopath"
PRETRAINING_REVISION = "96a5b33456fd948a0f1c90ee6901d748bde39111"
PRETRAINING_SHARDS = 200
DOWNSTREAM_REPO = "medarc/TCGA-12K-parquet"
DOWNSTREAM_REVISION = "0d5c21631c1375ea9d2fd72355572b9838f7f2dd"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
FOLD_SEED = 1337
RECEIPT_SCHEMA = "pathology-fairness-data/v1"

DATASETS = {
    "pretraining": {
        "repo": PRETRAINING_REPO,
        "revision": PRETRAINING_REVISION,
        "patterns": ["shard-*.parquet"],
        "required_columns": {"path", "jpeg"},
    },
    "downstream": {
        "repo": DOWNSTREAM_REPO,
        "revision": DOWNSTREAM_REVISION,
        "patterns": ["1/*.parquet", "2/*.parquet"],
        "required_columns": {"slide_path", "image_bytes"},
    },
}


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


def _parquet_files(root: Path, dataset: str) -> list[Path]:
    patterns = DATASETS[dataset]["patterns"]
    return sorted({path for pattern in patterns for path in root.glob(pattern)})


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

    required = DATASETS[dataset]["required_columns"]
    total_rows = 0
    total_bytes = 0
    schemas = set()
    for path in files:
        parquet = pq.ParquetFile(path)
        columns = tuple(parquet.schema_arrow.names)
        missing_columns = required - set(columns)
        if missing_columns:
            raise ValueError(f"{path}: missing columns {sorted(missing_columns)}")
        if parquet.metadata.num_rows <= 0:
            raise ValueError(f"{path}: empty Parquet file")
        if deep:
            sample = parquet.read_row_group(0, columns=sorted(required)).slice(0, 1)
            if sample.num_rows != 1 or any(sample.column(i)[0].as_py() in (None, b"", "")
                                           for i in range(sample.num_columns)):
                raise ValueError(f"{path}: invalid first-row sample")
        schemas.add(columns)
        total_rows += parquet.metadata.num_rows
        total_bytes += path.stat().st_size
    if len(schemas) != 1:
        raise ValueError(f"{dataset} shards have {len(schemas)} distinct schemas")

    spec = DATASETS[dataset]
    return {
        "schema": RECEIPT_SCHEMA,
        "dataset": dataset,
        "source": {
            "repo_id": spec["repo"],
            "revision": spec["revision"],
            "repo_type": "dataset",
        },
        "local": {
            "file_count": len(files),
            "total_rows": total_rows,
            "total_bytes": total_bytes,
            "columns": list(next(iter(schemas))),
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


def _brca_label(diagnoses: list[dict]) -> int | None:
    names = {str(row.get("primary_diagnosis", "")).strip().lower()
             for row in diagnoses}
    ductal = any("duct" in name and "carcinoma" in name for name in names)
    lobular = any("lobular" in name and "carcinoma" in name for name in names)
    if ductal == lobular:
        return None
    return 0 if ductal else 1


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
        race = {
            "white": "White",
            "black or african american": "Black",
            "asian": "Asian",
        }.get(str(demographic.get("race", "")).strip().lower(), "")
        days_to_birth = demographic.get("days_to_birth")
        age = ""
        if isinstance(days_to_birth, (int, float)):
            age = round(abs(float(days_to_birth)) / 365.25, 4)
        labels = {"nsclc": None, "glioma": None, "brca": None}
        if cancer in {"LUAD", "LUSC"}:
            labels["nsclc"] = 0 if cancer == "LUAD" else 1
        if cancer in {"LGG", "GBM"}:
            labels["glioma"] = 0 if cancer == "LGG" else 1
        if cancer == "BRCA":
            labels["brca"] = _brca_label(case.get("diagnoses") or [])
        row = {
            "patient_barcode": patient_id,
            "cancer": cancer,
            "cancer_type": cancer,
            "race": race,
            "gender": str(demographic.get("sex_at_birth", "")).strip(),
            "age_years": age,
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
        members = [index for index in selected if int(rows[index][label_column]) == label]
        if len(members) < n_splits:
            raise ValueError(
                f"{task}: class {label} has {len(members)} patients, fewer than "
                f"n_splits={n_splits}"
            )
        rng.shuffle(members)
        for position, index in enumerate(members):
            rows[index][fold_column] = position % n_splits


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


def prepare_clinical(metadata_dir: Path, holdout_task: str,
                     fino_path: Path | None = None) -> dict:
    cases = download_gdc_cases()
    source_sha256 = _sha256_bytes(
        json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
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
    fino_path = fino_path or metadata_dir.parent / "pretraining_tiles" / "fino_meta.json"
    _atomic_json(fino_path, fino)
    counts = {
        task: sum(row[f"label_{task}"] != "" for row in rows)
        for task in ("nsclc", "glioma", "brca")
    }
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
        "holdout_task": holdout_task,
        "holdout_patients": len(holdout),
        "outputs": {
            "demographics_csv": {
                "path": str(demographics_path.resolve()),
                "sha256": _sha256_file(demographics_path),
            },
            "holdout_file": {
                "path": str(holdout_path.resolve()),
                "sha256": _sha256_file(holdout_path),
            },
            "fino_meta": {
                "path": str(fino_path.resolve()),
                "sha256": _sha256_file(fino_path),
            },
        },
    }
    _atomic_json(metadata_dir / "METADATA_RECEIPT.json", receipt)
    return receipt


def _workers(value: int | None) -> int:
    return value or min(16, os.cpu_count() or 8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser("all", help="prepare the complete study inputs")
    all_parser.add_argument("--pretraining-dir", type=Path,
                            default=Path("data/pretraining_tiles"))
    all_parser.add_argument("--downstream-dir", type=Path,
                            default=Path("data/downstream_tiles"))
    all_parser.add_argument("--metadata-dir", type=Path,
                            default=Path("data/metadata"))
    all_parser.add_argument("--holdout-task", choices=["brca", "nsclc", "glioma"],
                            default="brca")
    all_parser.add_argument("--download-downstream", action="store_true")
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

    clinical_parser = subparsers.add_parser(
        "clinical", help="download public GDC clinical metadata and build folds"
    )
    clinical_parser.add_argument("--metadata-dir", type=Path,
                                 default=Path("data/metadata"))
    clinical_parser.add_argument("--holdout-task", choices=["brca", "nsclc", "glioma"],
                                 default="brca")
    clinical_parser.add_argument("--fino-out", type=Path,
                                 default=Path("data/pretraining_tiles/fino_meta.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "tiles":
        download_tiles(args.dest, args.dataset, _workers(args.workers),
                       deep=args.deep_validate)
    elif args.command == "validate":
        receipt = validate_tiles(args.dir, args.dataset, deep=args.deep)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    elif args.command == "clinical":
        receipt = prepare_clinical(args.metadata_dir, args.holdout_task, args.fino_out)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        download_tiles(args.pretraining_dir, "pretraining", _workers(args.workers),
                       deep=args.deep_validate)
        clinical_receipt = prepare_clinical(
            args.metadata_dir, args.holdout_task, args.pretraining_dir / "fino_meta.json"
        )
        if args.download_downstream:
            download_tiles(args.downstream_dir, "downstream", _workers(args.workers),
                           deep=args.deep_validate)
        print(json.dumps(clinical_receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
