#!/usr/bin/env python3
"""Reproducible stability-study runner built on :mod:`post_hoc_debias`.

It installs narrowly scoped runtime safeguards which:

* separate the fixed outer-split seed from the downstream-head seed;
* restore the head RNG after method-specific auxiliary modules are constructed;
* use an atomic, validated, content-addressed cache;
* attach complete source/cache identities to the result; and
* annotate OOF predictions with the exact outer fold and available site fields.

The model fitting and prediction calculations are delegated to the core runner.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from pathology_fairness.data_contracts import (
    RECEIPT_SCHEMA,
    validate_dataset_receipt,
)

if __package__:
    from . import post_hoc_debias as implementation
else:
    import post_hoc_debias as implementation


RUNNER_SCHEMA = "reliable-fairness-head/v2"
CACHE_SCHEMA = "reliable-fairness-embedding-cache/v2"
FROZEN_SPLIT_SEED = 1337
GENERIC_EXTERNAL_CORE_TASKS = frozenset({"dcis_duke", "cptac_gbm"})

_HEAD_SEED = 1337
_SPLIT_SEED = FROZEN_SPLIT_SEED
_CACHE_EVENTS: list[dict[str, Any]] = []
_CHECKPOINT_IDENTITY: dict[str, Any] = {}
_CACHE_CONTRACT: dict[str, Any] = {}
_OUTER_FOLDS: dict[str, int] = {}
_NESTED_PREDICTIONS = False
_NESTED_TRAINING_AUDIT: list[dict[str, Any]] = []
_ORIGINAL_BUILD_HEAD = implementation.build_head
_ORIGINAL_TRAIN_AND_EVAL = implementation.train_and_eval


class CacheIntegrityError(RuntimeError):
    """An existing content-addressed cache entry failed validation."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def checkpoint_content_identity(checkpoint: str | os.PathLike[str] | None) -> str:
    """Return a cache-tag identity based on exact checkpoint bytes."""
    global _CHECKPOINT_IDENTITY
    if not checkpoint:
        _CHECKPOINT_IDENTITY = {
            "kind": "random-init",
            "sha256": None,
            "bytes": 0,
        }
    else:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"checkpoint must be an existing regular non-symlink file: {path}"
            )
        _CHECKPOINT_IDENTITY = {
            "kind": "checkpoint",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return "checkpoint-content=" + _canonical_json(_CHECKPOINT_IDENTITY)


def ordered_tile_payload_identity(
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
) -> str:
    """Hash barcode and payload bytes in their exact input order."""
    return _ordered_tile_evidence(tiles)["ordered_tiles_sha256"]


def _ordered_tile_evidence(
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
) -> dict[str, Any]:
    """Return a compact ledger sufficient to reproduce the ordered digest."""
    digest = hashlib.sha256()
    digest.update(b"reliable-fairness-ordered-tiles-v1\0")
    barcodes: list[str] = []
    payload_sha256: list[str] = []
    payload_sizes: list[int] = []
    for index, (barcode, payload) in enumerate(tiles):
        barcode_text = str(barcode)
        barcode_bytes = barcode_text.encode("utf-8")
        try:
            payload_data = bytes(payload)
        except Exception as error:
            raise TypeError(f"tile {index} payload is not bytes-like") from error
        payload_digest = hashlib.sha256(payload_data)
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(barcode_bytes).to_bytes(8, "big"))
        digest.update(barcode_bytes)
        digest.update(len(payload_data).to_bytes(8, "big"))
        digest.update(payload_digest.digest())
        barcodes.append(barcode_text)
        payload_sha256.append(payload_digest.hexdigest())
        payload_sizes.append(len(payload_data))
    return {
        "input_barcodes": np.asarray(barcodes, dtype=np.str_),
        "payload_sha256": np.asarray(payload_sha256, dtype=np.str_),
        "payload_bytes": np.asarray(payload_sizes, dtype=np.int64),
        "ordered_tiles_sha256": digest.hexdigest(),
    }


def _ordered_digest_from_evidence(
    input_barcodes: np.ndarray,
    payload_sha256: np.ndarray,
    payload_bytes: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"reliable-fairness-ordered-tiles-v1\0")
    for index, (barcode, payload_hash, byte_count) in enumerate(
        zip(input_barcodes, payload_sha256, payload_bytes, strict=True)
    ):
        barcode_bytes = str(barcode).encode("utf-8")
        payload_hash_text = str(payload_hash)
        if len(payload_hash_text) != 64:
            raise CacheIntegrityError("tile payload evidence has invalid SHA-256")
        try:
            payload_digest = bytes.fromhex(payload_hash_text)
        except ValueError as error:
            raise CacheIntegrityError(
                "tile payload evidence has non-hex SHA-256"
            ) from error
        count = int(byte_count)
        if count < 0:
            raise CacheIntegrityError("tile payload evidence has negative bytes")
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(barcode_bytes).to_bytes(8, "big"))
        digest.update(barcode_bytes)
        digest.update(count.to_bytes(8, "big"))
        digest.update(payload_digest)
    return digest.hexdigest()


def _preprocessing_source_identity() -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    core_path = Path(implementation.__file__).resolve()
    fairness_path = Path(implementation.fe.__file__).resolve()
    backbone_path = (Path(implementation.fe.WORKTREE) / "model.py").resolve()
    if not backbone_path.is_file() or backbone_path.is_symlink():
        raise FileNotFoundError(
            f"backbone implementation must be a regular non-symlink file: "
            f"{backbone_path}"
        )
    return {
        "runner_sha256": sha256_file(runner_path),
        "runner_bytes": runner_path.stat().st_size,
        "core_sha256": sha256_file(core_path),
        "core_bytes": core_path.stat().st_size,
        "fairness_eval_sha256": sha256_file(fairness_path),
        "fairness_eval_bytes": fairness_path.stat().st_size,
        "backbone_model_sha256": sha256_file(backbone_path),
        "backbone_model_bytes": backbone_path.stat().st_size,
    }


def _array_hash_update(digest: Any, array: np.ndarray) -> None:
    canonical = np.ascontiguousarray(array)
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))


def _entry_sha256(
    metadata: dict[str, Any],
    embeddings: np.ndarray,
    barcodes: np.ndarray,
    keep_mask: np.ndarray,
    input_barcodes: np.ndarray,
    payload_sha256: np.ndarray,
    payload_bytes: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"reliable-fairness-cache-entry-v2\0")
    digest.update(_canonical_json(metadata).encode("utf-8"))
    _array_hash_update(digest, embeddings)
    encoded = np.asarray([str(value) for value in barcodes], dtype=np.str_)
    _array_hash_update(digest, encoded)
    _array_hash_update(digest, np.asarray(keep_mask, dtype=np.bool_))
    _array_hash_update(
        digest, np.asarray([str(value) for value in input_barcodes], dtype=np.str_)
    )
    _array_hash_update(
        digest, np.asarray([str(value) for value in payload_sha256], dtype=np.str_)
    )
    _array_hash_update(digest, np.asarray(payload_bytes, dtype=np.int64))
    return digest.hexdigest()


def _cache_metadata(
    tag: str,
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = evidence or _ordered_tile_evidence(tiles)
    ordered_tiles_sha256 = evidence["ordered_tiles_sha256"]
    source = {
        "checkpoint_sha256": _CHECKPOINT_IDENTITY.get("sha256"),
        "checkpoint_bytes": _CHECKPOINT_IDENTITY.get("bytes", 0),
        "ordered_tiles_sha256": ordered_tiles_sha256,
        "tile_count": len(tiles),
        "tag_sha256": hashlib.sha256(tag.encode("utf-8")).hexdigest(),
        "study_cache_contract": dict(_CACHE_CONTRACT),
        **_preprocessing_source_identity(),
    }
    cache_key = hashlib.sha256(
        (CACHE_SCHEMA + "\0" + _canonical_json(source)).encode("utf-8")
    ).hexdigest()
    return {
        "schema": CACHE_SCHEMA,
        "cache_key": cache_key,
        "source_identity": source,
    }


def _read_validated_cache(
    path: Path,
    expected: dict[str, Any],
    expected_barcodes: np.ndarray,
    expected_evidence: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        with np.load(path, allow_pickle=False) as stored:
            required = {
                "emb", "barcodes", "keep_mask", "input_barcodes",
                "payload_sha256", "payload_bytes", "metadata_json",
                "entry_sha256",
            }
            if set(stored.files) != required:
                raise CacheIntegrityError(
                    f"{path}: cache members {sorted(stored.files)} != {sorted(required)}"
                )
            embeddings = np.asarray(stored["emb"])
            barcodes = np.asarray(stored["barcodes"])
            keep_mask = np.asarray(stored["keep_mask"])
            input_barcodes = np.asarray(stored["input_barcodes"])
            payload_sha256 = np.asarray(stored["payload_sha256"])
            payload_bytes = np.asarray(stored["payload_bytes"])
            metadata = json.loads(str(stored["metadata_json"].item()))
            claimed = str(stored["entry_sha256"].item())
    except CacheIntegrityError:
        raise
    except Exception as error:
        raise CacheIntegrityError(f"{path}: unreadable cache entry: {error}") from error

    if metadata != expected:
        raise CacheIntegrityError(f"{path}: source metadata mismatch")
    if keep_mask.dtype != np.bool_ or keep_mask.shape != (len(expected_barcodes),):
        raise CacheIntegrityError(f"{path}: invalid keep mask")
    if embeddings.ndim != 2 or embeddings.shape[0] != int(keep_mask.sum()):
        raise CacheIntegrityError(f"{path}: invalid embedding shape {embeddings.shape}")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise CacheIntegrityError(f"{path}: embeddings must be floating point")
    if not np.isfinite(embeddings).all():
        raise CacheIntegrityError(f"{path}: embeddings contain non-finite values")
    normalized_barcodes = np.asarray([str(v) for v in barcodes], dtype=np.str_)
    normalized_input = np.asarray(
        [str(value) for value in input_barcodes], dtype=np.str_
    )
    normalized_payload_sha = np.asarray(
        [str(value) for value in payload_sha256], dtype=np.str_
    )
    normalized_payload_bytes = np.asarray(payload_bytes)
    if normalized_input.shape != expected_barcodes.shape or not np.array_equal(
        normalized_input, expected_barcodes
    ):
        raise CacheIntegrityError(f"{path}: input-barcode evidence mismatch")
    expected_payload_sha = expected_evidence["payload_sha256"]
    expected_payload_bytes = expected_evidence["payload_bytes"]
    if (
        normalized_payload_sha.shape != expected_payload_sha.shape
        or not np.array_equal(normalized_payload_sha, expected_payload_sha)
        or normalized_payload_bytes.dtype != np.int64
        or normalized_payload_bytes.shape != expected_payload_bytes.shape
        or not np.array_equal(normalized_payload_bytes, expected_payload_bytes)
    ):
        raise CacheIntegrityError(f"{path}: tile-payload evidence mismatch")
    reproduced_ordered = _ordered_digest_from_evidence(
        normalized_input, normalized_payload_sha, normalized_payload_bytes
    )
    if (
        reproduced_ordered
        != expected["source_identity"]["ordered_tiles_sha256"]
        or reproduced_ordered != expected_evidence["ordered_tiles_sha256"]
    ):
        raise CacheIntegrityError(f"{path}: ordered-tile evidence digest mismatch")
    if not np.array_equal(normalized_barcodes, expected_barcodes[keep_mask]):
        raise CacheIntegrityError(f"{path}: ordered barcode payload mismatch")
    actual = _entry_sha256(
        metadata, embeddings, normalized_barcodes, keep_mask,
        normalized_input, normalized_payload_sha, normalized_payload_bytes,
    )
    if claimed != actual:
        raise CacheIntegrityError(f"{path}: entry digest mismatch")
    return embeddings, normalized_barcodes.astype(object), actual


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some network filesystems do not support directory fsync. The file was
        # still fsynced and atomically renamed.
        pass


@contextmanager
def _exclusive_cache_key(directory: Path, cache_key: str) -> Iterable[None]:
    """Serialize first publication without ever replacing a valid entry."""
    lock_directory = directory / ".reliable_cache_locks"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{cache_key}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validated_cached_embed(
    tag: str,
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
    embed_fn: Callable[[Sequence[tuple[Any, bytes]]], tuple[np.ndarray, np.ndarray]],
    cache_dir: str | os.PathLike[str],
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray]:
    """Content-address and validate an embedding cache before every reuse."""
    if not tiles:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)
    evidence = _ordered_tile_evidence(tiles)
    metadata = _cache_metadata(tag, tiles, evidence=evidence)
    key = metadata["cache_key"]
    safe_role = re.sub(r"[^A-Za-z0-9_.-]", "_", tag.split("|", 1)[0])[:64]
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"emb_{safe_role}_{key}.npz"
    expected_barcodes = np.asarray([str(bc) for bc, _ in tiles], dtype=np.str_)

    action = "hit"
    if path.exists():
        embeddings, barcodes, entry_sha = _read_validated_cache(
            path, metadata, expected_barcodes, evidence
        )
        log(f"  [validated cache hit] {path}")
    else:
        with _exclusive_cache_key(directory, key):
            # A concurrent worker may have published while this worker waited.
            if path.exists():
                embeddings, barcodes, entry_sha = _read_validated_cache(
                    path, metadata, expected_barcodes, evidence
                )
                log(f"  [validated cache hit after lock] {path}")
            else:
                embeddings_all, keep = embed_fn(tiles)
                embeddings_all = np.asarray(embeddings_all)
                keep = np.asarray(keep, dtype=bool)
                if keep.shape != (len(tiles),):
                    raise CacheIntegrityError(
                        f"embed keep mask shape {keep.shape} != ({len(tiles)},)"
                    )
                embeddings = np.asarray(embeddings_all[keep])
                barcodes = expected_barcodes[keep]
                if embeddings.ndim != 2 or embeddings.shape[0] != len(barcodes):
                    raise CacheIntegrityError(
                        f"embed output shape {embeddings.shape} is inconsistent "
                        "with keep mask"
                    )
                if not np.issubdtype(embeddings.dtype, np.floating):
                    raise CacheIntegrityError("embed output must be floating point")
                if not np.isfinite(embeddings).all():
                    raise CacheIntegrityError(
                        "embed output contains non-finite values"
                    )
                entry_sha = _entry_sha256(
                    metadata,
                    embeddings,
                    barcodes,
                    keep,
                    evidence["input_barcodes"],
                    evidence["payload_sha256"],
                    evidence["payload_bytes"],
                )
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w+b",
                        dir=directory,
                        prefix=".reliable_emb_",
                        suffix=".npz.tmp",
                        delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                        np.savez(
                            handle,
                            emb=embeddings,
                            barcodes=np.asarray(barcodes, dtype=np.str_),
                            keep_mask=keep,
                            input_barcodes=evidence["input_barcodes"],
                            payload_sha256=evidence["payload_sha256"],
                            payload_bytes=evidence["payload_bytes"],
                            metadata_json=np.asarray(_canonical_json(metadata)),
                            entry_sha256=np.asarray(entry_sha),
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    _fsync_directory(directory)
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
                # Validate published bytes, never trust the in-memory arrays.
                embeddings, barcodes, entry_sha = _read_validated_cache(
                    path, metadata, expected_barcodes, evidence
                )
                action = "write"
                log(
                    f"  [validated cache write] {path} "
                    f"({embeddings.shape[0]} tiles)"
                )

    file_identity = {
        "logical_role": "pool" if tag.startswith("pool-") else "task",
        "cache_key": key,
        "cache_file_sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "entry_sha256": entry_sha,
        "action": action,
        "source_identity": metadata["source_identity"],
    }
    _CACHE_EVENTS.append(file_identity)
    return embeddings, barcodes


def _reset_head_rng() -> None:
    torch = implementation._torch()
    torch.manual_seed(_HEAD_SEED)
    np.random.seed(_HEAD_SEED)


def _patched_build_head(*args: Any, **kwargs: Any) -> Any:
    model = _ORIGINAL_BUILD_HEAD(*args, **kwargs)
    # Auxiliary-module construction consumes a method-dependent number of random
    # draws. Restore the stream before optimizer/minibatch/dropout activity.
    _reset_head_rng()
    return model


def _patient_set_hash(patients: Iterable[str]) -> str:
    payload = _canonical_json(sorted({str(patient) for patient in patients}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nested_dump_records(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Produce outer-test and honest nested inner-calibration predictions."""
    import inspect

    signature = inspect.signature(_ORIGINAL_TRAIN_AND_EVAL)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    outer_fold = int(values["eval_fold"])
    emb_task = np.asarray(values["emb_task"])
    bc_task = np.asarray(values["bc_task"], dtype=object)
    emb_pool = np.asarray(values["emb_pool"])
    bc_pool = np.asarray(values["bc_pool"], dtype=object)
    fold_of = {str(key): int(value) for key, value in values["fold_of"].items()}
    if len(emb_pool) or len(bc_pool):
        raise ValueError(
            "nested predictions require an empty task_only auxiliary pool"
        )
    observed_folds = {fold_of[str(patient)] for patient in bc_task}
    if observed_folds != set(range(5)):
        raise ValueError(
            f"nested predictions require folds 0..4, got {sorted(observed_folds)}"
        )

    # The core call trains on all folds except k and evaluates only k.
    outer_result = _ORIGINAL_TRAIN_AND_EVAL(*args, **kwargs)
    outer_records = outer_result.get("predictions", [])
    eligible_patients = {str(patient) for patient in bc_task}
    expected_outer = {
        patient
        for patient in eligible_patients
        if fold_of[patient] == outer_fold
    }
    actual_outer = {str(record["patient_id"]) for record in outer_records}
    if actual_outer != expected_outer or len(outer_records) != len(expected_outer):
        raise ValueError(
            f"outer fold {outer_fold}: prediction patients do not match eligible set"
        )
    for record in outer_records:
        record["prediction_role"] = "outer_test"
        record["outer_fold"] = outer_fold
        record["original_fold"] = outer_fold

    outer_train = {
        patient
        for patient in eligible_patients
        if fold_of[patient] != outer_fold
    }
    audit: dict[str, Any] = {
        "calibration_outer_fold": outer_fold,
        "outer_test": {
            "excluded_folds": [outer_fold],
            "training_patient_count": len(outer_train),
            "training_patient_ids_sha256": _patient_set_hash(outer_train),
            "evaluation_patient_count": len(expected_outer),
            "evaluation_patient_ids_sha256": _patient_set_hash(expected_outer),
        },
        "inner_fits": [],
    }
    combined = list(outer_records)
    keep_outer_out = np.asarray(
        [fold_of[str(patient)] != outer_fold for patient in bc_task], dtype=bool
    )
    retained_patients = {
        str(patient) for patient in bc_task[keep_outer_out]
    }
    for inner_fold in sorted(observed_folds - {outer_fold}):
        # Removing k from emb_task/bc_task means the core ~is_eval training
        # mask now excludes both k (absent) and j (the requested eval fold).
        inner_values = dict(values)
        inner_values["emb_task"] = emb_task[keep_outer_out]
        inner_values["bc_task"] = bc_task[keep_outer_out]
        inner_values["label_of"] = {
            str(patient): value
            for patient, value in values["label_of"].items()
            if str(patient) in retained_patients
        }
        inner_values["fold_of"] = {
            patient: fold_of[patient] for patient in retained_patients
        }
        inner_values["sens"] = {
            str(patient): value
            for patient, value in values["sens"].items()
            if str(patient) in retained_patients
        }
        inner_values["eval_fold"] = inner_fold
        inner_values["dump_records"] = True
        implementation.SEED = _HEAD_SEED
        inner_result = _ORIGINAL_TRAIN_AND_EVAL(**inner_values)
        inner_records = inner_result.get("predictions", [])
        expected_inner = {
            patient
            for patient in eligible_patients
            if fold_of[patient] == inner_fold
        }
        actual_inner = {str(record["patient_id"]) for record in inner_records}
        if (
            actual_inner != expected_inner
            or len(inner_records) != len(expected_inner)
        ):
            raise ValueError(
                f"nested k={outer_fold}, j={inner_fold}: predictions do not "
                "match eligible calibration set"
            )
        for record in inner_records:
            record["prediction_role"] = "inner_calibration"
            record["calibration_outer_fold"] = outer_fold
            record["inner_fold"] = inner_fold
            record["original_fold"] = inner_fold
        combined.extend(inner_records)
        training_patients = {
            patient
            for patient in eligible_patients
            if fold_of[patient] not in {outer_fold, inner_fold}
        }
        audit["inner_fits"].append(
            {
                "inner_fold": inner_fold,
                "excluded_folds": sorted((outer_fold, inner_fold)),
                "training_patient_count": len(training_patients),
                "training_patient_ids_sha256": _patient_set_hash(
                    training_patients
                ),
                "evaluation_patient_count": len(expected_inner),
                "evaluation_patient_ids_sha256": _patient_set_hash(
                    expected_inner
                ),
            }
        )
    outer_result["predictions"] = combined
    _NESTED_TRAINING_AUDIT.append(audit)
    return outer_result


def _patched_train_and_eval(*args: Any, **kwargs: Any) -> Any:
    global _OUTER_FOLDS
    # Legacy positional contract: fold_of is argument 5 (zero based).
    if len(args) >= 6:
        fold_of = args[5]
        _OUTER_FOLDS = {str(key): int(value) for key, value in fold_of.items()}
    elif "fold_of" in kwargs:
        _OUTER_FOLDS = {
            str(key): int(value) for key, value in kwargs["fold_of"].items()
        }
    implementation.SEED = _HEAD_SEED
    if _NESTED_PREDICTIONS and bool(kwargs.get("dump_records", False)):
        return _nested_dump_records(*args, **kwargs)
    return _ORIGINAL_TRAIN_AND_EVAL(*args, **kwargs)


def install_runtime_patches(
    split_seed: int, head_seed: int, nested_predictions: bool = False
) -> None:
    """Install idempotent patches; exposed for synthetic contract tests."""
    global _SPLIT_SEED, _HEAD_SEED, _CACHE_EVENTS, _CACHE_CONTRACT, _OUTER_FOLDS
    global _NESTED_PREDICTIONS, _NESTED_TRAINING_AUDIT
    _SPLIT_SEED = int(split_seed)
    _HEAD_SEED = int(head_seed)
    _CACHE_EVENTS = []
    _CACHE_CONTRACT = {}
    _OUTER_FOLDS = {}
    _NESTED_PREDICTIONS = bool(nested_predictions)
    _NESTED_TRAINING_AUDIT = []
    implementation.SEED = _SPLIT_SEED
    implementation.checkpoint_cache_identity = checkpoint_content_identity
    implementation.cached_embed = validated_cached_embed
    implementation.build_head = _patched_build_head
    implementation.train_and_eval = _patched_train_and_eval


def configure_cache_contract(
    *,
    study_task: str,
    core_task: str,
    hospital_fold: str | None,
    hospital_folds_csv: str | None,
    split_seed: int,
    head_seed: int,
    runtime_contract: str | os.PathLike[str],
    demographics_csv: str | os.PathLike[str],
    tiles_dir: str | os.PathLike[str],
    checkpoint: str | os.PathLike[str],
    execution_contract: dict[str, Any],
) -> None:
    """Bind the logical study/fold contract into every embedding cache key."""
    global _CACHE_CONTRACT
    folds_identity: dict[str, Any] | None = None
    if hospital_folds_csv:
        folds_path = Path(hospital_folds_csv).expanduser().resolve()
        if not folds_path.is_file() or folds_path.is_symlink():
            raise FileNotFoundError(
                "hospital-fold contract must be a regular non-symlink file: "
                f"{folds_path}"
            )
        folds_identity = {
            "sha256": sha256_file(folds_path),
            "bytes": folds_path.stat().st_size,
        }
    runtime_path = Path(runtime_contract).expanduser().resolve()
    if not runtime_path.is_file() or runtime_path.is_symlink():
        raise FileNotFoundError(
            "runtime contract must be a regular non-symlink file: "
            f"{runtime_path}"
        )
    runtime_identity = {
        "sha256": sha256_file(runtime_path),
        "bytes": runtime_path.stat().st_size,
    }
    try:
        runtime_declaration = json.loads(runtime_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid runtime contract {runtime_path}: {error}") from error
    if not isinstance(runtime_declaration, dict) or not runtime_declaration:
        raise ValueError("runtime contract must be a nonempty JSON object")
    required_fields = {
        "study", "fold_seed", "head_architecture", "planned_head_seeds",
        "primary_estimand", "utility_noninferiority_margin", "sensitive",
        "method", "condition_col", "temperature", "posthoc_lambda",
    }
    missing_fields = sorted(required_fields - set(runtime_declaration))
    if missing_fields:
        raise ValueError(f"runtime contract is missing fields: {missing_fields}")
    if int(runtime_declaration["fold_seed"]) != int(split_seed):
        raise ValueError("runtime contract fold_seed does not match the runner")
    planned_seeds = [int(seed) for seed in runtime_declaration["planned_head_seeds"]]
    if (not planned_seeds or len(planned_seeds) > 5
            or len(set(planned_seeds)) != len(planned_seeds)):
        raise ValueError("planned_head_seeds must contain one to five unique seeds")
    if int(head_seed) not in planned_seeds:
        raise ValueError(f"head seed {head_seed} was not prospectively declared")
    if runtime_declaration["head_architecture"] != "linear-relu-dropout-linear":
        raise ValueError("unsupported head_architecture in runtime contract")
    if runtime_declaration["primary_estimand"] != \
            "posthoc_auc_gap_minus_pretraining_auc_gap":
        raise ValueError("runtime contract primary_estimand does not match analyzer")
    if float(runtime_declaration["utility_noninferiority_margin"]) < 0:
        raise ValueError("utility_noninferiority_margin must be non-negative")
    for field in ("sensitive", "method", "condition_col"):
        if str(runtime_declaration[field]) != str(execution_contract[field]):
            raise ValueError(f"runtime contract {field} does not match command")
    if not math.isclose(
        float(runtime_declaration["temperature"]),
        float(execution_contract["temperature"]), rel_tol=0, abs_tol=1e-12,
    ):
        raise ValueError("runtime contract temperature does not match command")
    allowed_lambdas = (0.0, float(runtime_declaration["posthoc_lambda"]))
    if not any(math.isclose(float(execution_contract["lambda_adv"]), value,
                            rel_tol=0, abs_tol=1e-12)
               for value in allowed_lambdas):
        raise ValueError("command lambda is not declared by the runtime contract")

    demographics_path = Path(demographics_csv).expanduser().resolve()
    if not demographics_path.is_file() or demographics_path.is_symlink():
        raise FileNotFoundError(
            f"demographics CSV must be a regular non-symlink file: {demographics_path}"
        )
    demographics_identity = {
        "sha256": sha256_file(demographics_path),
        "bytes": demographics_path.stat().st_size,
    }
    metadata_receipt_identity = None
    downstream_dataset_identity = None
    checkpoint_training_identity = None
    cohort_receipt_identity = None
    if core_task in {"brca", "nsclc", "glioma"}:
        downstream_dataset_identity = validate_dataset_receipt(
            Path(tiles_dir), "downstream"
        )
        receipt_path = demographics_path.parent / "METADATA_RECEIPT.json"
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise FileNotFoundError(
                f"TCGA workflows require the generated metadata receipt: {receipt_path}"
            )
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("schema") != RECEIPT_SCHEMA
                or int(receipt.get("fold_seed", -1)) != int(split_seed)):
            raise ValueError(
                "metadata receipt schema or fold seed does not match the runner"
            )
        if receipt.get("holdout_task") != core_task:
            raise ValueError(
                f"metadata holdout_task={receipt.get('holdout_task')} does not "
                f"match downstream task {core_task}"
            )
        expected_digest = (
            (receipt.get("outputs") or {}).get("demographics_csv") or {}
        ).get("sha256")
        if expected_digest != demographics_identity["sha256"]:
            raise ValueError(
                "demographics CSV digest does not match METADATA_RECEIPT.json"
            )
        metadata_receipt_identity = {
            "sha256": sha256_file(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "schema": receipt.get("schema"),
            "fold_seed": int(receipt["fold_seed"]),
        }
        cohort_receipt_path = demographics_path.parent / "COHORT_RECEIPT.json"
        if not cohort_receipt_path.is_file() or cohort_receipt_path.is_symlink():
            raise FileNotFoundError(
                "TCGA workflows require COHORT_RECEIPT.json; run "
                "`python scripts/prepare_data.py crosswalk`"
            )
        cohort_receipt = json.loads(cohort_receipt_path.read_text())
        cohort_inputs = cohort_receipt.get("inputs") or {}
        task_coverage = (cohort_receipt.get("tasks") or {}).get(core_task) or {}
        labeled_patients = int(task_coverage.get("labeled_patients", -1))
        patients_with_tiles = int(task_coverage.get("patients_with_tiles", -1))
        missing_patients = int(task_coverage.get("missing_patients", -1))
        if (cohort_receipt.get("schema") != "pathology-fairness-cohort/v1"
                or cohort_inputs.get("downstream_dataset_receipt_sha256")
                != downstream_dataset_identity["receipt_sha256"]
                or cohort_inputs.get("demographics_sha256")
                != demographics_identity["sha256"]
                or cohort_inputs.get("metadata_receipt_sha256")
                != metadata_receipt_identity["sha256"]
                or labeled_patients != int(
                    (receipt.get("task_patients") or {}).get(core_task, -2)
                )
                or patients_with_tiles <= 0
                or patients_with_tiles + missing_patients != labeled_patients):
            raise ValueError("cohort coverage receipt does not match study inputs")
        cohort_receipt_identity = {
            "sha256": sha256_file(cohort_receipt_path),
            "bytes": cohort_receipt_path.stat().st_size,
            "task": core_task,
            "coverage": task_coverage,
        }
        import torch

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise FileNotFoundError(
                f"checkpoint must be a regular file: {checkpoint_path}"
            )
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        input_identity = checkpoint_payload.get("input_identity") or {}
        if input_identity.get("holdout_task") != core_task:
            raise ValueError(
                "checkpoint does not prove exclusion of the downstream task"
            )
        if input_identity.get("metadata_receipt_sha256") != \
                metadata_receipt_identity["sha256"]:
            raise ValueError(
                "checkpoint metadata receipt does not match evaluation metadata"
            )
        checkpoint_training_identity = {
            "holdout_task": input_identity["holdout_task"],
            "metadata_receipt_sha256": input_identity["metadata_receipt_sha256"],
            "pretraining_revision": input_identity.get("pretraining_revision"),
            "pretraining_inventory_sha256": input_identity.get(
                "pretraining_inventory_sha256"
            ),
            "pretraining_lfs_manifest_sha256": input_identity.get(
                "pretraining_lfs_manifest_sha256"
            ),
        }
    _CACHE_CONTRACT = {
        "study_task": str(study_task),
        "core_task": str(core_task),
        "hospital_fold": (
            None if hospital_fold in (None, "") else str(hospital_fold)
        ),
        "hospital_folds_csv": folds_identity,
        "split_seed": int(split_seed),
        "runtime_contract": runtime_identity,
        "runtime_declaration": runtime_declaration,
        "demographics_csv": demographics_identity,
        "metadata_receipt": metadata_receipt_identity,
        "downstream_dataset": downstream_dataset_identity,
        "checkpoint_training": checkpoint_training_identity,
        "cohort_receipt": cohort_receipt_identity,
    }


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return None


def _replace_option(arguments: list[str], option: str, value: str) -> list[str]:
    output: list[str] = []
    replaced = False
    index = 0
    while index < len(arguments):
        current = arguments[index]
        if current == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} is missing its value")
            output.extend([option, value])
            replaced = True
            index += 2
        elif current.startswith(option + "="):
            output.append(option + "=" + value)
            replaced = True
            index += 1
        else:
            output.append(current)
            index += 1
    if not replaced:
        output.extend([option, value])
    return output


def resolve_task_mapping(
    core_arguments: Sequence[str],
    study_task: str | None,
    core_task: str | None,
) -> tuple[list[str], str, str]:
    """Resolve an explicit external adapter without silently remapping tasks."""
    arguments = list(core_arguments)
    requested_task = _option_value(arguments, "--task")
    if requested_task is None:
        raise ValueError("core --task is required")
    if core_task is not None:
        if core_task not in GENERIC_EXTERNAL_CORE_TASKS:
            raise ValueError(
                "--core-task may only select an audited generic-local adapter: "
                + ", ".join(sorted(GENERIC_EXTERNAL_CORE_TASKS))
            )
        if not _option_value(arguments, "--label-col"):
            raise ValueError("--core-task generic mapping requires --label-col")
        resolved_study = study_task or requested_task
        arguments = _replace_option(arguments, "--task", core_task)
        return arguments, resolved_study, core_task
    return arguments, study_task or requested_task, requested_task


def _metadata_site_map(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        return {}
    output: dict[str, dict[str, str]] = {}
    with resolved.open(newline="") as handle:
        for row in csv.DictReader(handle):
            lowered = {str(key).lower(): value for key, value in row.items()}
            patient = (
                lowered.get("patient_barcode")
                or lowered.get("case_id")
                or lowered.get("patient_id")
            )
            if not patient:
                continue
            values: dict[str, str] = {}
            for destination, candidates in {
                "tss": ("tss", "tissue_source_site"),
                "site": ("site", "hospital", "center", "institution"),
            }.items():
                value = next(
                    (
                        lowered[name]
                        for name in candidates
                        if lowered.get(name) not in (None, "")
                    ),
                    None,
                )
                if value is not None:
                    values[destination] = str(value)
            # TCGA's patient identifier embeds its tissue-source-site code.
            pieces = str(patient).split("-")
            if "tss" not in values and len(pieces) >= 2 and pieces[0] == "TCGA":
                values["tss"] = pieces[1]
            output[str(patient)] = values
    return output


def _combined_site_map(paths: Iterable[str | None]) -> dict[str, dict[str, str]]:
    combined: dict[str, dict[str, str]] = {}
    for path in paths:
        for patient, fields in _metadata_site_map(path).items():
            combined.setdefault(patient, {}).update(fields)
    return combined


def annotate_predictions(
    path: Path,
    fold_map: dict[str, int],
    site_map: dict[str, dict[str, str]],
    required_folds: set[int] | None = None,
    nested: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            patient = str(record["patient_id"])
            if not nested and patient in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate OOF patient {patient!r}"
                )
            seen.add(patient)
            if patient not in fold_map:
                raise ValueError(
                    f"{path}:{line_number}: patient {patient!r} has no outer fold"
                )
            original_fold = int(fold_map[patient])
            if nested:
                if int(record.get("original_fold", -1)) != original_fold:
                    raise ValueError(
                        f"{path}:{line_number}: original_fold does not match "
                        f"the frozen fold for {patient!r}"
                    )
            else:
                record["outer_fold"] = original_fold
            record.update(site_map.get(patient, {}))
            records.append(record)
    if nested:
        summary = _validate_nested_prediction_records(records)
        observed_folds = set(summary["required_outer_folds"])
    else:
        observed_folds = {int(record["outer_fold"]) for record in records}
        summary = {
            "mode": "oof",
            "schema": "oof/v1",
            "record_count": len(records),
            "unique_patient_count": len(seen),
            "role_counts": None,
            "row_multiplier": 1,
            "required_outer_folds": sorted(observed_folds),
        }
    if required_folds is not None and observed_folds != required_folds:
        raise ValueError(
            f"{path}: prediction outer folds {sorted(observed_folds)} != "
            f"{sorted(required_folds)}"
        )
    temporary = path.with_suffix(path.suffix + ".reliable.tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return summary


def _validate_nested_prediction_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    common = {
        "patient_id", "y_true", "y_score", "race", "sex", "age", "tss",
        "prediction_role", "original_fold",
    }
    optional = {"site"}
    outer_extra = {"outer_fold"}
    inner_extra = {"calibration_outer_fold", "inner_fold"}
    outer_by_patient: dict[str, dict[str, Any]] = {}
    inner_by_k: dict[int, dict[str, dict[str, Any]]] = {
        fold: {} for fold in range(5)
    }
    for index, record in enumerate(records, 1):
        role = record.get("prediction_role")
        expected_keys = common | optional | (
            outer_extra if role == "outer_test" else inner_extra
        )
        required_keys = common | (
            outer_extra if role == "outer_test" else inner_extra
        )
        if role not in {"outer_test", "inner_calibration"}:
            raise ValueError(f"nested record {index}: invalid prediction_role")
        if (
            "tss" not in record
            or not isinstance(record["tss"], str)
            or not record["tss"].strip()
        ):
            raise ValueError(f"nested record {index}: nonempty tss is required")
        if not required_keys.issubset(record) or not set(record).issubset(
            expected_keys
        ):
            raise ValueError(
                f"nested record {index}: invalid keys for role {role}"
            )
        patient = str(record["patient_id"])
        original_fold = record["original_fold"]
        if (
            isinstance(original_fold, bool)
            or not isinstance(original_fold, int)
            or original_fold not in range(5)
        ):
            raise ValueError(f"nested record {index}: invalid original_fold")
        if role == "outer_test":
            if record["outer_fold"] != original_fold:
                raise ValueError(
                    f"nested outer record {index}: outer_fold != original_fold"
                )
            if patient in outer_by_patient:
                raise ValueError(f"duplicate outer-test patient {patient!r}")
            outer_by_patient[patient] = record
        else:
            calibration_outer = record["calibration_outer_fold"]
            inner_fold = record["inner_fold"]
            if (
                isinstance(calibration_outer, bool)
                or not isinstance(calibration_outer, int)
                or calibration_outer not in range(5)
                or inner_fold != original_fold
                or calibration_outer == original_fold
            ):
                raise ValueError(
                    f"nested inner record {index}: invalid calibration/fold fields"
                )
            if patient in inner_by_k[calibration_outer]:
                raise ValueError(
                    f"duplicate calibration patient {patient!r} for "
                    f"outer fold {calibration_outer}"
                )
            inner_by_k[calibration_outer][patient] = record

    patients = set(outer_by_patient)
    if not patients:
        raise ValueError("nested predictions contain no outer-test patients")
    patient_fold = {
        patient: int(record["original_fold"])
        for patient, record in outer_by_patient.items()
    }
    if set(patient_fold.values()) != set(range(5)):
        raise ValueError("nested outer-test predictions do not cover folds 0..4")
    for calibration_outer in range(5):
        expected = {
            patient
            for patient in patients
            if patient_fold[patient] != calibration_outer
        }
        actual = set(inner_by_k[calibration_outer])
        if actual != expected:
            raise ValueError(
                f"calibration outer fold {calibration_outer}: patient set mismatch"
            )
        for patient, record in inner_by_k[calibration_outer].items():
            if int(record["original_fold"]) != patient_fold[patient]:
                raise ValueError(
                    f"calibration outer fold {calibration_outer}: original fold "
                    f"mismatch for {patient!r}"
                )
    outer_count = len(outer_by_patient)
    inner_count = sum(len(group) for group in inner_by_k.values())
    if len(records) != 5 * outer_count or inner_count != 4 * outer_count:
        raise ValueError(
            f"nested prediction count {len(records)} != 5N ({5 * outer_count})"
        )
    return {
        "mode": "nested_crossfit",
        "schema": "nested-crossfit-predictions/v1",
        "record_count": len(records),
        "unique_patient_count": outer_count,
        "role_counts": {
            "outer_test": outer_count,
            "inner_calibration": inner_count,
        },
        "row_multiplier": 5,
        "required_outer_folds": list(range(5)),
        "tss_required": True,
    }


def _validate_nested_training_audit(
    audit_rows: Sequence[dict[str, Any]],
) -> None:
    if len(audit_rows) != 5:
        raise ValueError(f"nested training audit requires 5 outer fits")
    by_outer = {
        int(row["calibration_outer_fold"]): row for row in audit_rows
    }
    if set(by_outer) != set(range(5)) or len(by_outer) != len(audit_rows):
        raise ValueError("nested training audit outer folds are incomplete")
    for outer_fold, row in by_outer.items():
        if row["outer_test"]["excluded_folds"] != [outer_fold]:
            raise ValueError("nested outer training exclusion audit mismatch")
        inner_fits = row["inner_fits"]
        if len(inner_fits) != 4:
            raise ValueError(
                f"nested outer fold {outer_fold}: expected four inner fits"
            )
        by_inner = {int(fit["inner_fold"]): fit for fit in inner_fits}
        if set(by_inner) != set(range(5)) - {outer_fold}:
            raise ValueError(
                f"nested outer fold {outer_fold}: inner folds are incomplete"
            )
        for inner_fold, fit in by_inner.items():
            if fit["excluded_folds"] != sorted((outer_fold, inner_fold)):
                raise ValueError(
                    f"nested k={outer_fold},j={inner_fold}: exclusion audit mismatch"
                )


def _atomic_json_update(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".reliable.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _private_staging_path(target: Path, role: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.{role}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=True,
        description="Reproducible wrapper around post_hoc_debias.py. Unrecognized "
                    "arguments are forwarded to the underlying runner.",
    )
    parser.add_argument("--split-seed", type=int, default=FROZEN_SPLIT_SEED)
    parser.add_argument("--head-seed", type=int, required=True)
    parser.add_argument("--study-task")
    parser.add_argument("--core-task")
    parser.add_argument("--site-metadata-csv")
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument("--nested-predictions", action="store_true")
    own, core_arguments = parser.parse_known_args()
    if own.split_seed != FROZEN_SPLIT_SEED:
        parser.error(
            f"--split-seed is frozen at {FROZEN_SPLIT_SEED} for this runner"
        )
    try:
        core_arguments, study_task, core_task = resolve_task_mapping(
            core_arguments, own.study_task, own.core_task
        )
    except ValueError as error:
        parser.error(str(error))

    requested_out_value = _option_value(core_arguments, "--out")
    if requested_out_value is None:
        parser.error("core --out is required")
    requested_predictions_value = _option_value(core_arguments, "--dump-predictions")
    if not requested_predictions_value:
        parser.error(
            "core --dump-predictions is required for paired research analysis"
        )
    adversary_data_value = _option_value(core_arguments, "--adversary-data")
    if own.nested_predictions:
        if not requested_predictions_value:
            parser.error("--nested-predictions requires --dump-predictions")
        if adversary_data_value != "task_only":
            parser.error(
                "--nested-predictions requires --adversary-data task_only"
            )
    demographics_value = _option_value(core_arguments, "--demographics-csv")
    if not demographics_value:
        parser.error("core --demographics-csv is required")
    tiles_value = _option_value(core_arguments, "--tiles-dir")
    if not tiles_value:
        parser.error("core --tiles-dir is required")
    checkpoint_value = _option_value(core_arguments, "--checkpoint")
    if not checkpoint_value:
        parser.error(
            "the reliable research runner requires --checkpoint; use the core "
            "runner directly for random-initialization plumbing smokes"
        )
    hospital_metadata_value = _option_value(core_arguments, "--hospital-folds-csv")
    hospital_fold_value = _option_value(core_arguments, "--hospital-fold")
    if hospital_fold_value and not hospital_metadata_value:
        parser.error("--hospital-fold requires --hospital-folds-csv")

    output_path = Path(requested_out_value).expanduser().resolve()
    working_output_path = _private_staging_path(output_path, "core-output")
    core_arguments = _replace_option(
        core_arguments, "--out", str(working_output_path)
    )
    prediction_path: Path | None = None
    working_prediction_path: Path | None = None
    if requested_predictions_value:
        prediction_path = Path(
            requested_predictions_value
        ).expanduser().resolve()
        working_prediction_path = _private_staging_path(
            prediction_path, "core-predictions"
        )
        core_arguments = _replace_option(
            core_arguments, "--dump-predictions", str(working_prediction_path)
        )

    install_runtime_patches(
        own.split_seed, own.head_seed, nested_predictions=own.nested_predictions
    )
    configure_cache_contract(
        study_task=study_task,
        core_task=core_task,
        hospital_fold=hospital_fold_value,
        hospital_folds_csv=hospital_metadata_value,
        split_seed=own.split_seed,
        head_seed=own.head_seed,
        runtime_contract=own.runtime_contract,
        demographics_csv=demographics_value,
        tiles_dir=tiles_value,
        checkpoint=checkpoint_value,
        execution_contract={
            "sensitive": _option_value(core_arguments, "--sensitive"),
            "method": _option_value(core_arguments, "--method") or "dann",
            "condition_col": _option_value(core_arguments, "--condition-col"),
            "temperature": float(
                _option_value(core_arguments, "--proto-temp") or 0.1
            ),
            "lambda_adv": float(
                _option_value(core_arguments, "--lambda-adv") or 1.0
            ),
        },
    )
    print(
        f"[reliable-fairness] split_seed={own.split_seed} "
        f"head_seed={own.head_seed} study_task={study_task} core_task={core_task}",
        flush=True,
    )
    sys.argv = [sys.argv[0], *core_arguments]
    implementation.main()

    prediction_artifact = None
    if prediction_path is not None and working_prediction_path is not None:
        prediction_summary = annotate_predictions(
            working_prediction_path,
            _OUTER_FOLDS,
            _combined_site_map(
                (demographics_value, hospital_metadata_value, own.site_metadata_csv)
            ),
            required_folds=set(range(5)),
            nested=own.nested_predictions,
        )
        if own.nested_predictions:
            _validate_nested_training_audit(_NESTED_TRAINING_AUDIT)
        os.replace(working_prediction_path, prediction_path)
        _fsync_directory(prediction_path.parent)
        with prediction_path.open() as prediction_handle:
            prediction_count = sum(1 for line in prediction_handle if line.strip())
        prediction_artifact = {
            "sha256": sha256_file(prediction_path),
            "bytes": prediction_path.stat().st_size,
            "record_count": prediction_count,
            "required_outer_folds": list(range(5)),
            **prediction_summary,
        }
        cohort_contract = _CACHE_CONTRACT.get("cohort_receipt")
        if cohort_contract is not None:
            expected_patients = int(
                cohort_contract["coverage"]["patients_with_tiles"]
            )
            if prediction_count != expected_patients:
                raise ValueError(
                    "prediction coverage does not match COHORT_RECEIPT.json: "
                    f"observed={prediction_count} expected={expected_patients}"
                )

    result = json.loads(working_output_path.read_text())
    runner_path = Path(__file__).resolve()
    core_path = Path(implementation.__file__).resolve()
    result["reliable_fairness"] = {
        "schema": RUNNER_SCHEMA,
        "study_task": study_task,
        "core_task": core_task,
        "split_seed": own.split_seed,
        "head_seed": own.head_seed,
        "runner": {
            "sha256": sha256_file(runner_path),
            "bytes": runner_path.stat().st_size,
        },
        "core": {
            "sha256": sha256_file(core_path),
            "bytes": core_path.stat().st_size,
        },
        "checkpoint_identity": dict(_CHECKPOINT_IDENTITY),
        "study_cache_contract": dict(_CACHE_CONTRACT),
        "ordered_cache_identities": list(_CACHE_EVENTS),
        "outer_fold_count": len(set(_OUTER_FOLDS.values())),
        "prediction_artifact": prediction_artifact,
        "nested_training_audit": (
            list(_NESTED_TRAINING_AUDIT) if own.nested_predictions else None
        ),
    }
    _atomic_json_update(working_output_path, result)
    os.replace(working_output_path, output_path)
    _fsync_directory(output_path.parent)


if __name__ == "__main__":
    main()
