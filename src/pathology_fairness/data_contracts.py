"""Pinned public-data contracts and lightweight local receipt verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "pathology-fairness-data/v3"
PRETRAINING_REVISION = "96a5b33456fd948a0f1c90ee6901d748bde39111"
DOWNSTREAM_REVISION = "0d5c21631c1375ea9d2fd72355572b9838f7f2dd"

DATASETS: dict[str, dict[str, Any]] = {
    "pretraining": {
        "repo": "medarc/nanopath",
        "revision": PRETRAINING_REVISION,
        "patterns": ["shard-*.parquet"],
        "expected_files": 200,
        "expected_rows": 4_000_000,
        "expected_bytes": 119_752_861_203,
        "manifest_sha256": (
            "886fae55b1f0f69990c80f2199b163ae3e3beeaab259aa2b246174a2b93514a4"
        ),
        "lfs_manifest_sha256": (
            "320f4c3f976522f3e332db9389a8c31884338ef259c690bb7d37628d81f23c13"
        ),
        "schema": {"path": "string", "jpeg": "binary"},
    },
    "downstream": {
        "repo": "medarc/TCGA-12K-parquet",
        "revision": DOWNSTREAM_REVISION,
        "patterns": ["1/*.parquet", "2/*.parquet"],
        "expected_files": 11_368,
        "expected_rows": 24_985_184,
        "expected_bytes": 703_084_899_067,
        "manifest_sha256": (
            "0099d87fcc7162ac678021d71d5ffe3cfb0db40de2719fc834884cf00a6ea8e1"
        ),
        "lfs_manifest_sha256": (
            "a2e0a420a00ee5b09e1b1ac6545da8c5aad19e60d92248745de62dd8ff030a29"
        ),
        "schema": {
            "task_id": "string",
            "slide_path": "string",
            "x": "int32",
            "y": "int32",
            "level": "int32",
            "tile_size": "int32",
            "level_downsample": "float",
            "image_dtype": "string",
            "image_bytes": "binary",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_files(root: Path, dataset: str) -> list[Path]:
    root = Path(root)
    return sorted({
        path
        for pattern in DATASETS[dataset]["patterns"]
        for path in root.glob(pattern)
        if path.is_file()
    })


def local_inventory(root: Path, dataset: str) -> dict[str, Any]:
    """Summarize names, sizes, and modification state without reading payloads."""
    root = Path(root).resolve()
    files = dataset_files(root, dataset)
    rows = [
        (path.relative_to(root).as_posix(), path.stat().st_size)
        for path in files
    ]
    names = "".join(f"{name}\n" for name, _ in rows).encode()
    inventory = "".join(f"{name}\t{size}\n" for name, size in rows).encode()
    stat_inventory = "".join(
        f"{path.relative_to(root).as_posix()}\t{path.stat().st_size}\t"
        f"{path.stat().st_mtime_ns}\n"
        for path in files
    ).encode()
    return {
        "file_count": len(rows),
        "manifest_sha256": hashlib.sha256(names).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "stat_inventory_sha256": hashlib.sha256(stat_inventory).hexdigest(),
        "total_bytes": sum(size for _, size in rows),
    }


def local_content_manifest(root: Path, dataset: str) -> str:
    """Hash every local file using the Hugging Face LFS manifest format."""
    root = Path(root).resolve()
    payload = []
    for path in dataset_files(root, dataset):
        relative = path.relative_to(root).as_posix()
        payload.append(
            f"{relative}\t{path.stat().st_size}\t{sha256_file(path)}\n"
        )
    return hashlib.sha256("".join(payload).encode()).hexdigest()


def validate_dataset_receipt(root: Path, dataset: str) -> dict[str, Any]:
    """Fail closed if a receipt or the current local inventory is incomplete."""
    root = Path(root).resolve()
    receipt_path = root / "DATASET_RECEIPT.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError(
            f"missing regular dataset receipt {receipt_path}; run "
            f"`python scripts/prepare_data.py tiles --dataset {dataset} "
            f"--dest {root}`"
        )
    receipt = json.loads(receipt_path.read_text())
    spec = DATASETS[dataset]
    source = receipt.get("source") or {}
    local = receipt.get("local") or {}
    required = {
        "schema": RECEIPT_SCHEMA,
        "dataset": dataset,
        "repo_id": spec["repo"],
        "revision": spec["revision"],
        "lfs_manifest_sha256": spec["lfs_manifest_sha256"],
        "file_count": spec["expected_files"],
        "total_rows": spec["expected_rows"],
        "total_bytes": spec["expected_bytes"],
        "manifest_sha256": spec["manifest_sha256"],
        "content_manifest_sha256": spec["lfs_manifest_sha256"],
    }
    observed = {
        "schema": receipt.get("schema"),
        "dataset": receipt.get("dataset"),
        "repo_id": source.get("repo_id"),
        "revision": source.get("revision"),
        "lfs_manifest_sha256": source.get("lfs_manifest_sha256"),
        "file_count": local.get("file_count"),
        "total_rows": local.get("total_rows"),
        "total_bytes": local.get("total_bytes"),
        "manifest_sha256": local.get("manifest_sha256"),
        "content_manifest_sha256": local.get("content_manifest_sha256"),
    }
    if observed != required:
        raise ValueError(
            f"{dataset} DATASET_RECEIPT.json does not match the pinned contract"
        )
    contracted_files = set(dataset_files(root, dataset))
    recursive_parquets = {path for path in root.rglob("*.parquet") if path.is_file()}
    unexpected_parquets = sorted(recursive_parquets - contracted_files)
    if unexpected_parquets:
        relative = [path.relative_to(root).as_posix() for path in unexpected_parquets]
        raise ValueError(
            f"{dataset} root contains Parquet files outside the pinned manifest: "
            f"{relative[:5]}"
        )
    current = local_inventory(root, dataset)
    for field in (
        "file_count", "manifest_sha256", "inventory_sha256",
        "stat_inventory_sha256", "total_bytes",
    ):
        if current[field] != local.get(field):
            raise ValueError(
                f"{dataset} local inventory no longer matches its receipt: {field}"
            )
    return {
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_schema": RECEIPT_SCHEMA,
        "revision": spec["revision"],
        "lfs_manifest_sha256": spec["lfs_manifest_sha256"],
        **current,
    }
