"""Content-addressed cache for normalized 128-D adapter representations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Sequence

import numpy as np

from tools import reliable_fairness_head as reliable


CACHE_SCHEMA = "matched-cancer-adapter-cache/v1"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    )


def _validate_normalized(embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2 or embeddings.shape[1] != 128:
        raise reliable.CacheIntegrityError(
            f"adapter cache must have shape [N,128], got {embeddings.shape}"
        )
    if embeddings.shape[0]:
        norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=2e-5):
            raise reliable.CacheIntegrityError(
                "adapter cache rows are not L2-normalized"
            )


def cached_adapter_embeddings(
    *,
    tag: str,
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
    embed_fn: Callable[
        [Sequence[tuple[Any, bytes | bytearray | memoryview]]],
        tuple[np.ndarray, np.ndarray],
    ],
    cache_dir: str | os.PathLike[str],
    source_identity: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Path, str]:
    """Return a validated cache entry bound to E, A, source, and ordered tiles."""
    if not tiles:
        raise ValueError("diagnostic cache requires at least one tile")
    evidence = reliable._ordered_tile_evidence(tiles)
    source = {
        **source_identity,
        "ordered_tiles_sha256": evidence["ordered_tiles_sha256"],
        "tile_count": len(tiles),
        "tag_sha256": hashlib.sha256(tag.encode("utf-8")).hexdigest(),
        "normalization": "per_tile_l2",
        "embedding_dim": 128,
    }
    key = hashlib.sha256(
        (CACHE_SCHEMA + "\0" + _canonical(source)).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": CACHE_SCHEMA,
        "cache_key": key,
        "source_identity": source,
    }
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:64]
    path = directory / f"adapter_{safe}_{key}.npz"
    expected_barcodes = np.asarray(
        [str(barcode) for barcode, _ in tiles], dtype=np.str_
    )

    with reliable._exclusive_cache_key(directory, key):
        if path.exists():
            embeddings, barcodes, entry_sha = reliable._read_validated_cache(
                path, metadata, expected_barcodes, evidence
            )
        else:
            embeddings_all, keep = embed_fn(tiles)
            embeddings_all = np.asarray(embeddings_all, dtype=np.float32)
            keep = np.asarray(keep, dtype=np.bool_)
            if keep.shape != (len(tiles),):
                raise reliable.CacheIntegrityError("invalid embed keep mask")
            embeddings = embeddings_all[keep]
            barcodes = expected_barcodes[keep]
            _validate_normalized(embeddings)
            entry_sha = reliable._entry_sha256(
                metadata,
                embeddings,
                barcodes,
                keep,
                evidence["input_barcodes"],
                evidence["payload_sha256"],
                evidence["payload_bytes"],
            )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=directory, prefix=f".{path.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w+b") as handle:
                    np.savez(
                        handle,
                        emb=embeddings,
                        barcodes=barcodes,
                        keep_mask=keep,
                        input_barcodes=evidence["input_barcodes"],
                        payload_sha256=evidence["payload_sha256"],
                        payload_bytes=evidence["payload_bytes"],
                        metadata_json=np.asarray(_canonical(metadata)),
                        entry_sha256=np.asarray(entry_sha),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, path)
                reliable._fsync_directory(directory)
            finally:
                temporary.unlink(missing_ok=True)
    _validate_normalized(embeddings)
    return embeddings, barcodes, path, entry_sha
