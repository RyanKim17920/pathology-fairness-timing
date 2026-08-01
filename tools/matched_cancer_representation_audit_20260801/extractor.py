#!/usr/bin/env python3
"""Receipt-bound representation extraction for the fixed-five audit.

The extractor has two jobs: reduce the already-validated B/P/H caches to the
frozen tile views, and run the missing plain/fair slot-1 checkpoints once to
emit E and temporary-A representations together.  It never accepts an outcome
or diagnosis field.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from tools.matched_cancer_representation_audit_20260801 import contract
from tools.matched_cancer_stage_20260730.receipts import file_identity, verify_receipt
from tools import reliable_fairness_head as reliable


FINAL_CACHE_SCHEMA = "matched-cancer-adapter-cache/v1"
COMPACT_CACHE_SCHEMA = "matched-cancer-fixed5-representation-cache/v1"
TILE_BUNDLE_SCHEMA = "matched-cancer-fixed5-tile-bundle/v1"
TILE_BUNDLE_RECEIPT_SCHEMA = "matched-cancer-fixed5-tile-bundle-receipt/v1"
DIAGNOSTIC_NAMESPACE = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/diagnostic"
)
TILE_VIEW_RECEIPT = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_stage_20260730/diagnostic/"
    "tile_views_seed32001/TILE_VIEW_RECEIPT.json"
)
COMPACT_ROW_FIELDS = (
    "patient_id",
    "cancer",
    "race",
    "tss",
    "view",
    "view_rank",
    "occurrence_index",
    "global_index",
    "payload_sha256",
)


class ExtractionError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FullCache:
    path: Path
    metadata: dict[str, Any]
    embeddings: np.ndarray
    barcodes: np.ndarray
    keep_mask: np.ndarray
    input_barcodes: np.ndarray
    payload_sha256: np.ndarray
    payload_bytes: np.ndarray
    entry_sha256: str


def diagnostic_attempt_root(seed: int) -> Path:
    if seed not in contract.FM_SEEDS:
        raise ValueError("unexpected FM seed")
    return (
        DIAGNOSTIC_NAMESPACE
        / f"seed_{seed}"
        / contract.ACCEPTED_ATTEMPTS[seed]
        / "run"
    )


def discover_final_cache(seed: int, cancer: str, layer: str) -> Path:
    """Find exactly one production B/P/H cache without accepting a caller path."""
    if cancer not in contract.CANCERS or layer not in {"B", "P", "H"}:
        raise ValueError("final cache requires frozen cancer and B/P/H layer")
    pattern = f"adapter_{cancer.lower()}_tp53-{layer}_*.npz"
    matches = sorted((diagnostic_attempt_root(seed) / "cache").glob(pattern))
    if len(matches) != 1:
        raise ExtractionError(
            f"expected exactly one final cache for {seed}/{cancer}/{layer}; "
            f"found {len(matches)}"
        )
    return matches[0]


def read_full_cache(path: str | os.PathLike[str]) -> FullCache:
    """Read a final cache and independently reproduce its content digest."""
    cache_path = Path(path).resolve()
    required = {
        "emb",
        "barcodes",
        "keep_mask",
        "input_barcodes",
        "payload_sha256",
        "payload_bytes",
        "metadata_json",
        "entry_sha256",
    }
    try:
        with np.load(cache_path, allow_pickle=False) as stored:
            if set(stored.files) != required:
                raise ExtractionError("final cache member set drift")
            embeddings = np.asarray(stored["emb"])
            barcodes = np.asarray(stored["barcodes"], dtype=np.str_)
            keep_mask = np.asarray(stored["keep_mask"])
            input_barcodes = np.asarray(stored["input_barcodes"], dtype=np.str_)
            payload_sha256 = np.asarray(stored["payload_sha256"], dtype=np.str_)
            payload_bytes = np.asarray(stored["payload_bytes"])
            metadata = json.loads(str(stored["metadata_json"].item()))
            claimed = str(stored["entry_sha256"].item())
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(f"unreadable final cache {cache_path}: {error}") from error

    if metadata.get("schema") != FINAL_CACHE_SCHEMA:
        raise ExtractionError("final cache schema drift")
    count = len(input_barcodes)
    if keep_mask.dtype != np.bool_ or keep_mask.shape != (count,):
        raise ExtractionError("invalid final-cache keep_mask")
    if payload_sha256.shape != (count,) or payload_bytes.shape != (count,):
        raise ExtractionError("invalid final-cache payload evidence")
    if payload_bytes.dtype != np.int64:
        raise ExtractionError("payload byte counts must be int64")
    if embeddings.ndim != 2 or embeddings.shape[0] != int(keep_mask.sum()):
        raise ExtractionError("invalid final-cache embedding topology")
    if barcodes.shape != (int(keep_mask.sum()),):
        raise ExtractionError("invalid final-cache barcode topology")
    if not np.array_equal(barcodes, input_barcodes[keep_mask]):
        raise ExtractionError("final-cache barcode/keep alignment drift")
    if not np.issubdtype(embeddings.dtype, np.floating) or not np.isfinite(embeddings).all():
        raise ExtractionError("final-cache embeddings must be finite floats")
    if any(len(str(value)) != 64 for value in payload_sha256):
        raise ExtractionError("invalid payload SHA-256 evidence")
    actual = reliable._entry_sha256(
        metadata,
        embeddings,
        barcodes,
        keep_mask,
        input_barcodes,
        payload_sha256,
        payload_bytes,
    )
    if claimed != actual:
        raise ExtractionError("final-cache entry digest mismatch")
    return FullCache(
        path=cache_path,
        metadata=metadata,
        embeddings=embeddings,
        barcodes=barcodes,
        keep_mask=keep_mask,
        input_barcodes=input_barcodes,
        payload_sha256=payload_sha256,
        payload_bytes=payload_bytes,
        entry_sha256=actual,
    )


def validate_final_cache_provenance(
    cache: FullCache, *, seed: int, cancer: str, layer: str
) -> None:
    if cache.path != discover_final_cache(seed, cancer, layer).resolve():
        raise ExtractionError("final-cache canonical path drift")
    source = cache.metadata.get("source_identity")
    if not isinstance(source, Mapping):
        raise ExtractionError("final-cache source identity missing")
    expected = contract.production_paths(seed)
    checkpoint = expected["checkpoints"][layer]
    receipt = expected["completion_receipts"][layer]
    if source.get("checkpoint") != file_identity(checkpoint):
        raise ExtractionError("final-cache checkpoint identity drift")
    if source.get("completion_receipt") != file_identity(receipt):
        raise ExtractionError("final-cache completion receipt identity drift")
    if source.get("embedding_dim") != contract.LAYER_DIMENSIONS[layer]:
        raise ExtractionError("final-cache dimension drift")
    contract.validate_representation_normalization(layer, source.get("normalization"))
    if cache.embeddings.shape[1] != contract.LAYER_DIMENSIONS[layer]:
        raise ExtractionError("final-cache embedding width drift")
    norms = np.linalg.norm(cache.embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=0.0):
        raise ExtractionError("final-cache rows are not per-tile L2 normalized")


def seed_state_hashes(seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Verify completion receipts and return frozen encoder/adapter hashes."""
    paths = contract.production_paths(seed)
    receipts: dict[str, Mapping[str, Any]] = {}
    for run, path in paths["completion_receipts"].items():
        receipts[run] = verify_receipt(
            path, expected_schema="matched-cancer-stage-completion/v1"
        )
        identity = receipts[run].get("identities", {}).get("latest_checkpoint")
        if identity != file_identity(paths["root"] / run / "latest.pt"):
            raise ExtractionError(f"completion/checkpoint identity drift for {seed}/{run}")
    encoder = {
        "E_plain": str(receipts["slot1_plain"]["encoder_post_sha256"]),
        "E_fair": str(receipts["slot1_fair"]["encoder_post_sha256"]),
        "B": str(receipts["B"]["encoder_post_sha256"]),
        "P": str(receipts["P"]["encoder_post_sha256"]),
        "H": str(receipts["H"]["encoder_post_sha256"]),
    }
    contract.validate_encoder_state_sharing(encoder)
    adapter = {
        "A_temp_plain": str(receipts["slot1_plain"]["adapter_post_sha256"]),
        "A_temp_fair": str(receipts["slot1_fair"]["adapter_post_sha256"]),
        "B": str(receipts["B"]["adapter_post_sha256"]),
        "P": str(receipts["P"]["adapter_post_sha256"]),
        "H": str(receipts["H"]["adapter_post_sha256"]),
    }
    return encoder, adapter


def assert_same_tile_evidence(reference: FullCache, candidate: FullCache) -> None:
    for name in ("input_barcodes", "payload_sha256", "payload_bytes", "keep_mask"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise ExtractionError(f"full tile evidence differs at {name}")


def build_selection_rows(
    reference: FullCache,
    population: Sequence[Mapping[str, str]],
    *,
    cancer: str,
) -> tuple[dict[str, Any], ...]:
    """Resolve the frozen 32 valid occurrences for every population patient."""
    contract.validate_metadata_records(population)
    patient_rows = [row for row in population if row["cancer"] == cancer]
    if not patient_rows:
        raise ExtractionError(f"no population rows for {cancer}")
    expected_patients = {row["patient_id"] for row in patient_rows}
    observed_patients = set(reference.input_barcodes.tolist())
    if observed_patients != expected_patients:
        raise ExtractionError("final-cache patient membership differs from frozen cohort")

    global_by_identity: dict[tuple[str, int], int] = {}
    occurrences: dict[str, list[dict[str, Any]]] = {patient: [] for patient in expected_patients}
    within_patient = {patient: 0 for patient in expected_patients}
    for global_index, (patient, payload, keep) in enumerate(
        zip(
            reference.input_barcodes.tolist(),
            reference.payload_sha256.tolist(),
            reference.keep_mask.tolist(),
            strict=True,
        )
    ):
        occurrence_index = within_patient[patient]
        within_patient[patient] += 1
        occurrences[patient].append(
            {
                "payload_sha256": payload,
                "occurrence_index": occurrence_index,
                "keep_mask": bool(keep),
            }
        )
        global_by_identity[(patient, occurrence_index)] = global_index

    result: list[dict[str, Any]] = []
    for metadata in patient_rows:
        patient = metadata["patient_id"]
        selected = contract.select_tile_views(patient, occurrences[patient])
        for view in contract.TILE_VIEWS:
            for view_rank, item in enumerate(selected[view]):
                result.append(
                    {
                        **dict(metadata),
                        "view": view,
                        "view_rank": view_rank,
                        "occurrence_index": int(item["occurrence_index"]),
                        "global_index": global_by_identity[
                            (patient, int(item["occurrence_index"]))
                        ],
                        "payload_sha256": str(item["payload_sha256"]),
                    }
                )
    expected_count = len(patient_rows) * contract.TILES_PER_PATIENT
    if len(result) != expected_count:
        raise AssertionError("selection cardinality drift")
    return tuple(result)


def subset_final_embeddings(
    cache: FullCache, selection: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Subset valid-row embeddings using immutable full-input indices."""
    valid_position = np.full(len(cache.keep_mask), -1, dtype=np.int64)
    valid_position[np.flatnonzero(cache.keep_mask)] = np.arange(int(cache.keep_mask.sum()))
    global_indices = np.asarray([int(row["global_index"]) for row in selection])
    if np.any(global_indices < 0) or np.any(global_indices >= len(cache.keep_mask)):
        raise ExtractionError("selected global index out of range")
    positions = valid_position[global_indices]
    if np.any(positions < 0):
        raise ExtractionError("selected occurrence is invalid in candidate cache")
    expected_payloads = np.asarray([row["payload_sha256"] for row in selection], dtype=np.str_)
    if not np.array_equal(cache.payload_sha256[global_indices], expected_payloads):
        raise ExtractionError("selected payload identity differs in candidate cache")
    return np.asarray(cache.embeddings[positions], dtype=np.float32)


def _compact_digest(
    metadata: Mapping[str, Any], embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]]
) -> str:
    digest = hashlib.sha256(b"matched-cancer-fixed5-compact-cache-v1\0")
    digest.update(_canonical_json(dict(metadata)).encode("utf-8"))
    reliable._array_hash_update(digest, np.asarray(embeddings, dtype=np.float32))
    for field in COMPACT_ROW_FIELDS:
        dtype = np.int64 if field in {"view_rank", "occurrence_index", "global_index"} else np.str_
        reliable._array_hash_update(digest, np.asarray([row[field] for row in rows], dtype=dtype))
    return digest.hexdigest()


def write_compact_cache(
    path: str | os.PathLike[str],
    *,
    seed: int,
    layer: str,
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(path)
    if seed not in contract.FM_SEEDS or layer not in contract.LAYER_DIMENSIONS:
        raise ValueError("compact cache seed/layer drift")
    if any(set(row) != set(COMPACT_ROW_FIELDS) for row in rows):
        raise ExtractionError("compact row schema drift")
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (len(rows), contract.LAYER_DIMENSIONS[layer]):
        raise ExtractionError("compact embedding shape drift")
    if not np.isfinite(values).all():
        raise ExtractionError("compact embeddings contain non-finite values")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=0.0):
        raise ExtractionError("compact embeddings are not per-tile L2 normalized")
    metadata = {
        "schema": COMPACT_CACHE_SCHEMA,
        "study_id": contract.STUDY_ID,
        "seed": seed,
        "layer": layer,
        "dimension": contract.LAYER_DIMENSIONS[layer],
        "normalization": contract.REPRESENTATION_NORMALIZATION,
        "row_count": len(rows),
        "source_identity": dict(source_identity),
    }
    entry = _compact_digest(metadata, values, rows)
    arrays: dict[str, Any] = {
        "emb": values,
        "metadata_json": np.asarray(_canonical_json(metadata)),
        "entry_sha256": np.asarray(entry),
    }
    for field in COMPACT_ROW_FIELDS:
        dtype = np.int64 if field in {"view_rank", "occurrence_index", "global_index"} else np.str_
        arrays[field] = np.asarray([row[field] for row in rows], dtype=dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary = handle.name
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(output.resolve()), "sha256": _sha256_file(output), "entry_sha256": entry}


def read_compact_cache(path: str | os.PathLike[str]) -> tuple[dict[str, Any], np.ndarray, tuple[dict[str, Any], ...]]:
    cache_path = Path(path)
    expected = {"emb", "metadata_json", "entry_sha256", *COMPACT_ROW_FIELDS}
    try:
        with np.load(cache_path, allow_pickle=False) as stored:
            if set(stored.files) != expected:
                raise ExtractionError("compact cache member set drift")
            metadata = json.loads(str(stored["metadata_json"].item()))
            embeddings = np.asarray(stored["emb"], dtype=np.float32)
            claimed = str(stored["entry_sha256"].item())
            count = embeddings.shape[0]
            rows = tuple(
                {
                    field: (
                        int(stored[field][index])
                        if field in {"view_rank", "occurrence_index", "global_index"}
                        else str(stored[field][index])
                    )
                    for field in COMPACT_ROW_FIELDS
                }
                for index in range(count)
            )
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(f"unreadable compact cache: {error}") from error
    if metadata.get("schema") != COMPACT_CACHE_SCHEMA or metadata.get("row_count") != len(rows):
        raise ExtractionError("compact cache metadata drift")
    if metadata.get("dimension") != embeddings.shape[1]:
        raise ExtractionError("compact cache dimension drift")
    actual = _compact_digest(metadata, embeddings, rows)
    if claimed != actual:
        raise ExtractionError("compact cache entry digest mismatch")
    return metadata, embeddings, rows


def embed_encoder_and_adapter(
    representation: Any,
    tiles: Sequence[tuple[str, bytes | bytearray | memoryview]],
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit per-tile L2-normalized E and A(E) in one frozen forward pass."""
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    encoder_dim = contract.LAYER_DIMENSIONS["E_plain"]
    adapter_dim = contract.LAYER_DIMENSIONS["A_temp_plain"]
    encoded_output = np.zeros((len(tiles), encoder_dim), dtype=np.float32)
    adapted_output = np.zeros((len(tiles), adapter_dim), dtype=np.float32)
    keep = np.zeros(len(tiles), dtype=np.bool_)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if representation.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            images: list[Any] = []
            indices: list[int] = []
            for offset, (_, payload) in enumerate(tiles[start : start + batch_size]):
                try:
                    image = Image.open(io.BytesIO(bytes(payload))).convert("RGB")
                    images.append(transform(image))
                    indices.append(start + offset)
                except Exception:
                    continue
            if not images:
                continue
            x = torch.stack(images).to(representation.device)
            with autocast:
                encoded = representation.encoder.probe_features(
                    (x - representation.mean) / representation.std
                ).float()
                adapted = representation.adapter(encoded).float()
                encoded = functional.normalize(encoded, dim=1)
                adapted = functional.normalize(adapted, dim=1)
            encoded_output[indices] = encoded.cpu().numpy()
            adapted_output[indices] = adapted.cpu().numpy()
            keep[indices] = True
    representation.assert_unchanged()
    if not keep.all():
        raise ExtractionError("a selected cache-valid tile failed frozen inference decoding")
    for label, values, dimension in (
        ("encoder", encoded_output, encoder_dim),
        ("adapter", adapted_output, adapter_dim),
    ):
        if values.shape != (len(tiles), dimension) or not np.isfinite(values).all():
            raise ExtractionError(f"invalid {label} frozen-inference output")
        norms = np.linalg.norm(values.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4, rtol=0.0):
            raise ExtractionError(f"{label} output is not per-tile L2 normalized")
    return encoded_output, adapted_output


def materialize_tile_bundle(
    output_directory: str | os.PathLike[str],
    *,
    references: Mapping[str, FullCache],
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    tile_view_receipt: Path = TILE_VIEW_RECEIPT,
) -> Path:
    """Stream source parquets once and retain only the frozen selected payloads."""
    import pyarrow.parquet as parquet
    from tools import fairness_eval

    if set(references) != set(contract.CANCERS) or set(selections) != set(contract.CANCERS):
        raise ExtractionError("tile bundle requires exactly BRCA and LUAD")
    receipt_value = json.loads(tile_view_receipt.read_text())
    if receipt_value.get("schema") != "matched-cancer-diagnostic-tile-view/v1":
        raise ExtractionError("tile-view receipt schema drift")
    source_root = Path(receipt_value["destination_root"])
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "tiles.bin"
    manifest_path = output / "tiles.jsonl"
    receipt_path = output / "TILE_BUNDLE_RECEIPT.json"
    selected_offsets: dict[tuple[str, int], tuple[int, int]] = {}
    data_tmp = output / ".tiles.bin.tmp"
    manifest_tmp = output / ".tiles.jsonl.tmp"
    offset = 0
    with data_tmp.open("wb") as binary:
        for cancer in contract.CANCERS:
            reference = references[cancer]
            wanted = {int(row["global_index"]): row for row in selections[cancer]}
            global_index = 0
            for parquet_path in sorted((source_root / cancer).glob("*.parquet")):
                file = parquet.ParquetFile(parquet_path)
                columns = file.schema_arrow.names
                image_column = fairness_eval._detect_image_col(columns)
                if image_column is None or "slide_path" not in columns:
                    continue
                slide_path = file.read_row_group(0, columns=["slide_path"]).column("slide_path")[0].as_py()
                patient = fairness_eval._tcga_barcode_from_svs(slide_path)
                if patient is None:
                    continue
                payloads = file.read_row_group(0, columns=[image_column]).column(image_column).to_pylist()
                for payload in payloads:
                    if payload is None:
                        continue
                    if global_index >= len(reference.input_barcodes):
                        raise ExtractionError("source parquet stream exceeds reference cache")
                    if patient != str(reference.input_barcodes[global_index]):
                        raise ExtractionError("source parquet/cache barcode order drift")
                    row = wanted.get(global_index)
                    if row is not None:
                        payload_value = bytes(payload)
                        payload_hash = hashlib.sha256(payload_value).hexdigest()
                        if payload_hash != row["payload_sha256"]:
                            raise ExtractionError("selected source payload hash drift")
                        if len(payload_value) != int(reference.payload_bytes[global_index]):
                            raise ExtractionError("selected source payload byte-count drift")
                        binary.write(payload_value)
                        selected_offsets[(cancer, global_index)] = (offset, len(payload_value))
                        offset += len(payload_value)
                    global_index += 1
            if global_index != len(reference.input_barcodes):
                raise ExtractionError("source parquet/cache tile count drift")
            if set(wanted) != {index for observed_cancer, index in selected_offsets if observed_cancer == cancer}:
                raise ExtractionError("not every selected tile was materialized")
        binary.flush()
        os.fsync(binary.fileno())

    with manifest_tmp.open("w", encoding="utf-8") as manifest:
        for cancer in contract.CANCERS:
            for row in selections[cancer]:
                start, size = selected_offsets[(cancer, int(row["global_index"]))]
                value = {**dict(row), "offset": start, "payload_bytes": size}
                manifest.write(_canonical_json(value) + "\n")
        manifest.flush()
        os.fsync(manifest.fileno())
    os.replace(data_tmp, data_path)
    os.replace(manifest_tmp, manifest_path)
    receipt = {
        "schema": TILE_BUNDLE_RECEIPT_SCHEMA,
        "study_id": contract.STUDY_ID,
        "row_count": sum(len(value) for value in selections.values()),
        "payload_bytes": data_path.stat().st_size,
        "identities": {
            "data": file_identity(data_path),
            "manifest": file_identity(manifest_path),
            "tile_view_receipt": file_identity(tile_view_receipt),
        },
    }
    receipt_path.write_text(_canonical_json(receipt) + "\n")
    return receipt_path


def load_tile_bundle(receipt_path: str | os.PathLike[str]) -> tuple[tuple[dict[str, Any], ...], list[tuple[str, bytes]]]:
    receipt_file = Path(receipt_path)
    value = json.loads(receipt_file.read_text())
    if value.get("schema") != TILE_BUNDLE_RECEIPT_SCHEMA:
        raise ExtractionError("tile-bundle receipt schema drift")
    identities = value.get("identities", {})
    for name in ("data", "manifest", "tile_view_receipt"):
        identity = identities.get(name)
        if not isinstance(identity, Mapping) or file_identity(Path(identity["canonical_path"])) != identity:
            raise ExtractionError(f"tile-bundle {name} identity drift")
    data_path = Path(identities["data"]["canonical_path"])
    manifest_path = Path(identities["manifest"]["canonical_path"])
    rows = tuple(json.loads(line) for line in manifest_path.read_text().splitlines() if line)
    expected_fields = {*COMPACT_ROW_FIELDS, "offset", "payload_bytes"}
    if len(rows) != value.get("row_count") or any(set(row) != expected_fields for row in rows):
        raise ExtractionError("tile-bundle manifest topology drift")
    tiles: list[tuple[str, bytes]] = []
    with data_path.open("rb") as source:
        for row in rows:
            source.seek(int(row["offset"]))
            payload = source.read(int(row["payload_bytes"]))
            if len(payload) != int(row["payload_bytes"]):
                raise ExtractionError("truncated tile-bundle payload")
            if hashlib.sha256(payload).hexdigest() != row["payload_sha256"]:
                raise ExtractionError("tile-bundle payload digest mismatch")
            tiles.append((str(row["patient_id"]), payload))
    compact_rows = tuple({field: row[field] for field in COMPACT_ROW_FIELDS} for row in rows)
    return compact_rows, tiles
