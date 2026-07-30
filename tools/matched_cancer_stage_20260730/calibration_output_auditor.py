#!/usr/bin/env python3
"""Independent, read-only audit of the seed-32001 calibration artifacts.

This module deliberately does not import the calibration driver, receipt
implementation, replay implementation, trainer, or config builder.  It treats
their persisted outputs as untrusted inputs and recomputes the relevant
contracts from standard serialization and hashing primitives.
"""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml


STUDY_ID = "matched_cancer_stage_20260730"
SCENARIO = "brca_luad_black_white_calibration_seed32001"
RUNS = ("slot1_plain", "slot1_fair", "B", "H", "P")
MODES = {
    "slot1_plain": "joint",
    "slot1_fair": "joint",
    "B": "adapter_only",
    "H": "adapter_only",
    "P": "adapter_only",
}
FAIR_WEIGHTS = {
    "slot1_plain": 0.0,
    "slot1_fair": 0.1,
    "B": 0.0,
    "H": 0.1,
    "P": 0.0,
}
PARENTS = {
    "B": "slot1_plain",
    "H": "slot1_plain",
    "P": "slot1_fair",
}
STEPS = 781
BATCH_SIZE = 128
PRESENTATIONS = STEPS * BATCH_SIZE
REPRESENTATION_SEED = 32001
REPLAY_SEED = 52001
DATA_ORDER_SEED = 62001
ADAPTER_INIT_SEED = 72001
ADAPTER_LR = 0.001
ADAPTER_WEIGHT_DECAY = 0.0001
CANCER_IDS = (2, 15)
RACE_IDS = (2, 4)
STRATA = tuple((cancer, race) for cancer in CANCER_IDS for race in RACE_IDS)
PER_STRATUM = BATCH_SIZE // len(STRATA)
HEX = frozenset("0123456789abcdef")
IDENTITY_KEYS = frozenset({"canonical_path", "bytes", "sha256"})
FORBIDDEN_OUTCOME_TOKENS = (
    "tp53",
    "molecular_labels",
    "condition_on_label",
    "downstream_diagnosis",
    "diagnosis_label",
)


class AuditError(ValueError):
    """A persisted calibration artifact violates the frozen contract."""


class Audit:
    """Small assertion collector used to make the final audit count explicit."""

    def __init__(self) -> None:
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            raise AuditError(message)

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AuditError(f"value is not canonical-JSON serializable: {error}") from error


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_canonical_json(path: Path) -> Any:
    path = required_file(path)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_duplicate_rejector
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON in {path}: {error}") from error
    if raw != canonical_json_bytes(value) + b"\n":
        raise AuditError(f"JSON is not canonically encoded: {path}")
    return value


def required_file(path: Path | str) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise AuditError(f"symlink is not allowed: {requested}")
    try:
        canonical = requested.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"required file is unavailable: {requested}: {error}") from error
    if not canonical.is_file() or canonical.stat().st_size <= 0:
        raise AuditError(f"required nonempty regular file is unavailable: {requested}")
    return canonical


def sha256_file(path: Path | str) -> str:
    canonical = required_file(path)
    before = canonical.stat()
    digest = hashlib.sha256()
    with canonical.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = canonical.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise AuditError(f"file changed while being hashed: {canonical}")
    return digest.hexdigest()


def file_identity(path: Path | str) -> dict[str, Any]:
    canonical = required_file(path)
    return {
        "canonical_path": str(canonical),
        "bytes": canonical.stat().st_size,
        "sha256": sha256_file(canonical),
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX)
    )


def _identity_records(
    identities: Mapping[str, Any], prefix: str = ""
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in identities.items():
        if not isinstance(name, str) or not name:
            raise AuditError("receipt identity role must be a nonempty string")
        role = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping) and set(value) == IDENTITY_KEYS:
            records.append((role, value))
        elif isinstance(value, Mapping):
            records.extend(_identity_records(value, role))
        else:
            raise AuditError(f"identity role {role!r} is not a file identity")
    return records


def topology_sha256(identities: Mapping[str, Any]) -> str:
    roles = []
    for role, identity in _identity_records(identities):
        digest = identity.get("sha256")
        if not _valid_sha256(digest):
            raise AuditError(f"invalid identity SHA-256 for {role!r}")
        roles.append({"role": role, "file_sha256": digest})
    if not roles:
        raise AuditError("receipt has no file identities")
    payload = {
        "schema": "matched-cancer-stage-topology/v1",
        "roles": sorted(roles, key=lambda row: row["role"]),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_receipt(
    path: Path,
    audit: Audit,
    *,
    schema: str,
    expected_identity_roles: set[str] | None = None,
) -> dict[str, Any]:
    receipt = load_canonical_json(path)
    audit.require(isinstance(receipt, dict), f"receipt root is not an object: {path}")
    audit.equal(receipt.get("schema"), schema, f"receipt schema drift in {path}")
    audit.equal(receipt.get("study_id"), STUDY_ID, f"study ID drift in {path}")
    audit.equal(receipt.get("scenario"), SCENARIO, f"scenario drift in {path}")
    identities = receipt.get("identities")
    audit.require(isinstance(identities, dict), f"missing identities in {path}")
    if expected_identity_roles is not None:
        audit.equal(
            set(identities),
            expected_identity_roles,
            f"identity-role topology drift in {path}",
        )
    audit.equal(
        receipt.get("topology_sha256"),
        topology_sha256(identities),
        f"topology digest drift in {path}",
    )
    records = _identity_records(identities)
    audit.require(bool(records), f"receipt binds no files: {path}")
    for role, recorded in records:
        audit.equal(
            set(recorded), set(IDENTITY_KEYS), f"identity keys drift for {role}"
        )
        audit.require(
            type(recorded.get("bytes")) is int and recorded["bytes"] > 0,
            f"invalid byte count for {role}",
        )
        audit.require(
            _valid_sha256(recorded.get("sha256")),
            f"invalid SHA-256 for {role}",
        )
        audit.equal(
            file_identity(recorded["canonical_path"]),
            dict(recorded),
            f"bound file changed for {role}",
        )
    audit.equal(
        _structured_forbidden_hits(receipt),
        [],
        f"receipt contains downstream-outcome tokens: {path}",
    )
    return receipt


def _sequence_sha256(values: Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def patient_from_tile_path(tile_path: str) -> str:
    return "-".join(tile_path.split("/", 1)[0].split("-")[:3])


def patient_key(patient: str) -> int:
    raw = hashlib.blake2b(patient.encode(), digest_size=8).digest()
    return int.from_bytes(raw, "big") & 0x7FFFFFFFFFFFFFFF


def runtime_batch_trace(values: Sequence[int], batch_size: int) -> str:
    if len(values) % batch_size:
        raise AuditError("runtime trace length is not divisible by batch size")
    digest = hashlib.sha256()
    for offset in range(0, len(values), batch_size):
        batch = values[offset : offset + batch_size]
        digest.update(len(batch).to_bytes(8, "little", signed=False))
        digest.update(struct.pack(f"<{len(batch)}q", *batch))
    return digest.hexdigest()


def inspect_manifest(
    path: Path,
    audit: Audit,
    *,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
    verify_source_rows: bool = False,
    effective_config: Mapping[str, Any] | None = None,
    fino_meta_path: Path | None = None,
    exclusions_path: Path | None = None,
) -> dict[str, Any]:
    manifest = load_canonical_json(path)
    audit.require(isinstance(manifest, dict), "replay manifest root is not an object")
    audit.equal(
        set(manifest),
        {
            "schema",
            "contract",
            "occurrences",
            "traces",
            "manifest_payload_sha256",
        },
        "replay manifest top-level keys drift",
    )
    audit.equal(
        manifest["schema"],
        "matched-cancer-replay-manifest/v1",
        "replay manifest schema drift",
    )
    contract = manifest["contract"]
    expected_contract = {
        "cancer_ids": list(CANCER_IDS),
        "race_ids": list(RACE_IDS),
        "batch_size": batch_size,
        "steps": steps,
        "seed": REPLAY_SEED,
    }
    audit.equal(contract, expected_contract, "replay contract drift")
    occurrences = manifest["occurrences"]
    audit.require(isinstance(occurrences, list), "occurrences is not a list")
    audit.equal(
        len(occurrences),
        steps * batch_size,
        "replay exposure count drift",
    )
    body = {
        "schema": manifest["schema"],
        "contract": contract,
        "occurrences": occurrences,
        "traces": manifest["traces"],
    }
    payload_sha = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    audit.equal(
        manifest["manifest_payload_sha256"],
        payload_sha,
        "replay payload digest drift",
    )

    occurrence_keys = {
        "batch",
        "position",
        "dataset_index",
        "shard_path",
        "shard_sha256",
        "row",
        "tile_path",
        "tile_jpeg_sha256",
        "patient",
        "cancer",
        "race",
        "augmentation_seed",
    }
    patient_values: list[str] = []
    tile_values: list[dict[str, Any]] = []
    seeds: list[int] = []
    dataset_indices: list[int] = []
    patient_keys: list[int] = []
    identity_by_dataset_index: dict[int, dict[str, Any]] = {}
    per_batch = Counter()
    for index, raw in enumerate(occurrences):
        audit.require(isinstance(raw, dict), f"occurrence {index} is not an object")
        audit.equal(set(raw), occurrence_keys, f"occurrence {index} keys drift")
        batch, position = divmod(index, batch_size)
        audit.equal(raw["batch"], batch, f"occurrence {index} batch drift")
        audit.equal(raw["position"], position, f"occurrence {index} position drift")
        for key in (
            "batch",
            "position",
            "dataset_index",
            "row",
            "cancer",
            "race",
            "augmentation_seed",
        ):
            audit.require(type(raw[key]) is int, f"occurrence {index} {key} is not int")
        audit.require(raw["dataset_index"] >= 0, f"negative dataset index at {index}")
        audit.require(raw["row"] >= 0, f"negative Parquet row at {index}")
        audit.require(
            0 <= raw["augmentation_seed"] < 2**63 - 1,
            f"augmentation seed out of range at {index}",
        )
        for key in ("shard_path", "tile_path", "patient"):
            audit.require(
                isinstance(raw[key], str) and bool(raw[key]),
                f"occurrence {index} {key} is empty",
            )
        for key in ("shard_sha256", "tile_jpeg_sha256"):
            audit.require(
                _valid_sha256(raw[key]), f"occurrence {index} {key} invalid"
            )
        stratum = (raw["cancer"], raw["race"])
        audit.require(stratum in STRATA, f"out-of-contract stratum at {index}")
        per_batch[(batch, *stratum)] += 1
        audit.equal(
            raw["patient"],
            patient_from_tile_path(raw["tile_path"]),
            f"tile/patient mismatch at occurrence {index}",
        )
        identity = {
            key: raw[key]
            for key in occurrence_keys
            if key not in {"batch", "position", "augmentation_seed"}
        }
        previous = identity_by_dataset_index.setdefault(raw["dataset_index"], identity)
        audit.equal(
            identity,
            previous,
            f"dataset index has multiple identities at occurrence {index}",
        )
        patient_values.append(raw["patient"])
        tile_values.append(
            {
                "shard_sha256": raw["shard_sha256"],
                "row": raw["row"],
                "tile_path": raw["tile_path"],
                "tile_jpeg_sha256": raw["tile_jpeg_sha256"],
            }
        )
        seeds.append(raw["augmentation_seed"])
        dataset_indices.append(raw["dataset_index"])
        patient_keys.append(patient_key(raw["patient"]))

    for batch in range(steps):
        for cancer, race in STRATA:
            audit.equal(
                per_batch[(batch, cancer, race)],
                batch_size // len(STRATA),
                f"batch {batch} stratum {(cancer, race)} is not balanced",
            )
    expected_traces = {
        "patient_sha256": _sequence_sha256(patient_values),
        "tile_sha256": _sequence_sha256(tile_values),
        "augmentation_seed_sha256": _sequence_sha256(seeds),
    }
    audit.equal(manifest["traces"], expected_traces, "manifest trace hash drift")
    trace_values = {
        "payload_sha256": payload_sha,
        "manifest_file_sha256": sha256_file(path),
        **expected_traces,
        "sample_batch_trace_sha256": runtime_batch_trace(
            dataset_indices, batch_size
        ),
        "patient_batch_trace_sha256": runtime_batch_trace(
            patient_keys, batch_size
        ),
        "augmentation_seed_batch_trace_sha256": runtime_batch_trace(
            seeds, batch_size
        ),
        "occurrences": occurrences,
        "unique_identities": identity_by_dataset_index,
    }
    if verify_source_rows:
        if effective_config is None or fino_meta_path is None or exclusions_path is None:
            raise AuditError("source-row verification requires config, metadata, exclusions")
        verify_occurrence_sources(
            identity_by_dataset_index,
            effective_config,
            fino_meta_path,
            exclusions_path,
            audit,
        )
    return trace_values


def _patient_in_val(patient: str, seed: int, fraction: float) -> bool:
    key = f"{seed}:{patient}".encode()
    value = int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(), "big"
    ) / 2**64
    return value < fraction


def verify_occurrence_sources(
    identities: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    fino_meta_path: Path,
    exclusions_path: Path,
    audit: Audit,
) -> None:
    """Rebuild dataset indexes and verify every unique referenced JPEG row."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise AuditError("pyarrow is required for replay source verification") from error

    data = config["data"]
    dataset_dir = Path(data["dataset_dir"]).resolve(strict=True)
    shards = sorted(dataset_dir.glob("shard-*.parquet"))
    audit.require(bool(shards), f"no source shards under {dataset_dir}")
    shard_set = {str(path.resolve(strict=True)) for path in shards}
    metadata = json.loads(required_file(fino_meta_path).read_text())
    exclusions = {
        line.strip()
        for line in required_file(exclusions_path).read_text().splitlines()
        if line.strip()
    }
    include = {
        str(factor): {int(value) for value in values}
        for factor, values in data["include_discrete"].items()
    }
    audit.equal(include, {"cancer": {2, 15}, "race": {2, 4}}, "filter drift")
    targets = set(identities)
    rebuilt: dict[int, tuple[str, int, str]] = {}
    retained_index = 0
    split_seed = int(data["split_seed"])
    val_fraction = float(data["val_fraction"])
    for shard in shards:
        canonical_shard = str(shard.resolve(strict=True))
        table = pq.read_table(str(shard), columns=["path"], memory_map=True)
        for row_index, tile_path in enumerate(table["path"].to_pylist()):
            patient = patient_from_tile_path(tile_path)
            if patient in exclusions:
                continue
            if any(
                int(metadata["discrete"][factor].get(patient, -1)) not in allowed
                for factor, allowed in include.items()
            ):
                continue
            if _patient_in_val(patient, split_seed, val_fraction):
                continue
            if retained_index in targets:
                rebuilt[retained_index] = (
                    canonical_shard,
                    row_index,
                    tile_path,
                )
            retained_index += 1
    audit.equal(set(rebuilt), targets, "could not rebuild every dataset index")

    by_shard: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for dataset_index, identity in identities.items():
        shard_path = identity["shard_path"]
        audit.require(shard_path in shard_set, f"unknown shard: {shard_path}")
        audit.equal(
            rebuilt[dataset_index],
            (shard_path, identity["row"], identity["tile_path"]),
            f"dataset index mapping drift for {dataset_index}",
        )
        by_shard[shard_path][int(identity["row"])] = identity

    for shard_path, rows in by_shard.items():
        expected_shard_sha = {identity["shard_sha256"] for identity in rows.values()}
        audit.equal(
            len(expected_shard_sha), 1, f"multiple hashes claimed for {shard_path}"
        )
        audit.equal(
            sha256_file(shard_path),
            next(iter(expected_shard_sha)),
            f"source shard bytes drifted: {shard_path}",
        )
        reader = pq.ParquetFile(shard_path, memory_map=True)
        ends: list[int] = []
        running = 0
        for group in range(reader.metadata.num_row_groups):
            running += reader.metadata.row_group(group).num_rows
            ends.append(running)
        grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for row, identity in rows.items():
            group = bisect.bisect_right(ends, row)
            audit.require(group < len(ends), f"row outside shard: {shard_path}:{row}")
            start = 0 if group == 0 else ends[group - 1]
            grouped[group].append((row - start, identity))
        for group, requested in grouped.items():
            table = reader.read_row_group(group, columns=["path", "jpeg"])
            for row_in_group, identity in requested:
                tile_path = table["path"][row_in_group].as_py()
                jpeg = table["jpeg"][row_in_group].as_py()
                audit.equal(
                    tile_path,
                    identity["tile_path"],
                    f"Parquet tile path drift at {shard_path}:{identity['row']}",
                )
                audit.equal(
                    hashlib.sha256(jpeg).hexdigest(),
                    identity["tile_jpeg_sha256"],
                    f"JPEG bytes drift at {shard_path}:{identity['row']}",
                )
                patient = patient_from_tile_path(tile_path)
                audit.equal(
                    patient,
                    identity["patient"],
                    f"Parquet patient drift at {shard_path}:{identity['row']}",
                )
                audit.equal(
                    int(metadata["discrete"]["cancer"].get(patient, -1)),
                    identity["cancer"],
                    f"cancer metadata drift for {patient}",
                )
                audit.equal(
                    int(metadata["discrete"]["race"].get(patient, -1)),
                    identity["race"],
                    f"race metadata drift for {patient}",
                )


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise AuditError("checkpoint state_dict is not tensor-only")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _structured_forbidden_hits(value: Any, location: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            for token in FORBIDDEN_OUTCOME_TOKENS:
                if token in key_text:
                    hits.append(f"{location}.{key}")
            hits.extend(_structured_forbidden_hits(nested, f"{location}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            hits.extend(_structured_forbidden_hits(nested, f"{location}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for token in FORBIDDEN_OUTCOME_TOKENS:
            if token in lowered:
                hits.append(location)
    return hits


def load_metrics(
    path: Path,
    audit: Audit,
    *,
    run: str,
    steps: int = STEPS,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[dict[str, Any]], str, str]:
    train_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(required_file(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_duplicate_rejector)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid metrics JSON at {path}:{line_number}") from error
        audit.require(isinstance(row, dict), f"metrics row is not object at line {line_number}")
        audit.equal(
            _structured_forbidden_hits(row),
            [],
            f"{run} metrics contain downstream-outcome tokens at line {line_number}",
        )
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                audit.require(
                    math.isfinite(float(value)),
                    f"non-finite metric {key} in {run} line {line_number}",
                )
        if "matched_stage_mode" in row:
            train_rows.append(row)
    audit.equal(len(train_rows), steps, f"{run} train metric row count drift")
    for expected_step, row in enumerate(train_rows, 1):
        audit.equal(row.get("step"), expected_step, f"{run} step sequence drift")
        audit.equal(
            row.get("matched_stage_mode"), MODES[run], f"{run} metric mode drift"
        )
        audit.equal(row.get("batch_size"), batch_size, f"{run} metric batch drift")
        audit.equal(
            row.get("examples_seen"),
            expected_step * batch_size,
            f"{run} exposure schedule drift",
        )
        audit.require(float(row.get("cancer", 0.0)) > 0, f"{run} cancer loss is not positive")
        audit.require(float(row.get("race_fair", 0.0)) > 0, f"{run} fair loss is not positive")
        audit.require(float(row.get("total", math.nan)) > 0, f"{run} total loss invalid")
        audit.require(
            float(row.get("h_dose_main_grad_norm", 0.0)) > 0,
            f"{run} main loss lacks adapter reachability at step {expected_step}",
        )
        fair_dose = float(row.get("h_dose_fair_grad_norm", math.nan))
        if FAIR_WEIGHTS[run] > 0:
            audit.require(fair_dose > 0, f"{run} fair loss lacks reachability")
        else:
            audit.equal(fair_dose, 0.0, f"{run} zero-weight fair gradient drift")
        audit.require(
            float(row.get("stage_adapter_grad_norm", 0.0)) > 0,
            f"{run} adapter update gradient is not positive",
        )
        audit.require(
            math.isclose(
                float(row.get("race_fair_weighted")),
                FAIR_WEIGHTS[run] * float(row["race_fair"]),
                rel_tol=1e-6,
                abs_tol=1e-7,
            ),
            f"{run} weighted fair metric drift",
        )
        audit.equal(
            float(row.get("adapter_lr")), ADAPTER_LR, f"{run} adapter LR drift"
        )
        audit.equal(
            float(row.get("adapter_weight_decay")),
            ADAPTER_WEIGHT_DECAY,
            f"{run} adapter weight decay drift",
        )
        audit.require(
            math.isclose(
                float(row.get("sample_fraction")),
                expected_step / steps,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{run} sample-fraction schedule drift",
        )
    shared_schedule_fields = (
        "step",
        "batch_size",
        "examples_seen",
        "sample_fraction",
        "adapter_lr",
        "adapter_weight_decay",
    )
    mode_schedule_fields = (
        *shared_schedule_fields,
        "lr",
        "wd",
        "teacher_temp",
        "teacher_momentum",
        "kde_scale",
        "matched_stage_mode",
    )
    shared_schedule = [
        {key: row[key] for key in shared_schedule_fields} for row in train_rows
    ]
    mode_schedule = [
        {key: row[key] for key in mode_schedule_fields} for row in train_rows
    ]
    return (
        train_rows,
        hashlib.sha256(canonical_json_bytes(shared_schedule)).hexdigest(),
        hashlib.sha256(canonical_json_bytes(mode_schedule)).hexdigest(),
    )


def _identity_at(receipt: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    value: Any = receipt["identities"]
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise AuditError(f"missing receipt identity path: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, Mapping) or set(value) != IDENTITY_KEYS:
        raise AuditError(f"not a file identity: {'.'.join(path)}")
    return value


def _assert_identity(
    audit: Audit,
    recorded: Mapping[str, Any],
    expected_path: Path,
    message: str,
) -> None:
    audit.equal(dict(recorded), file_identity(expected_path), message)


def validate_config(
    run: str,
    config_path: Path,
    config: Mapping[str, Any],
    root: Path,
    audit: Audit,
) -> None:
    audit.require(isinstance(config, dict), f"{run} config root is not mapping")
    project = config["project"]
    train = config["train"]
    data = config["data"]
    probe = config["probe"]
    fino = config["fino"]
    stage = config["matched_stage"]
    audit.equal(project["name"], f"matched-cancer-calibration-seed32001-{run}", f"{run} name drift")
    audit.equal(
        Path(project["output_dir"]).resolve(),
        (root / run).resolve(),
        f"{run} output directory drift",
    )
    audit.equal(train["seed"], REPRESENTATION_SEED, f"{run} seed drift")
    expected_train = {
        "batch_size": BATCH_SIZE,
        "max_train_samples": PRESENTATIONS,
        "log_every": 1,
        "save_every": STEPS,
        "eval_every": STEPS,
        "val_batches": 1,
        "num_workers": 2,
        "persistent_workers": True,
        "resume": None,
    }
    for key, expected in expected_train.items():
        audit.equal(train.get(key), expected, f"{run} train.{key} drift")
    audit.equal(data.get("tissue_thresh"), 0.0, f"{run} tissue fallback enabled")
    audit.equal(
        data.get("include_discrete"),
        {"cancer": list(CANCER_IDS), "race": list(RACE_IDS)},
        f"{run} population filter drift",
    )
    audit.equal(probe.get("enabled"), False, f"{run} probe is enabled")
    for key in (
        "datasets",
        "segmentation_datasets",
        "slide_datasets",
        "auc_datasets",
        "survival_datasets",
        "robustness_datasets",
    ):
        audit.equal(probe.get(key), [], f"{run} probe.{key} is not empty")
    expected_fino = {
        "enabled": True,
        "objective": "contrastive-two-condition",
        "method": "contrastive",
        "gamma_max": 0.7,
        "contrastive_temp": 0.2,
        "contrastive_weight": 0.1,
        "contrastive_condition_on": "cancer",
        "dose_logging": True,
        "race_weight": "none",
        "race_resample": False,
        "discrete": [["cancer", 1], ["race", -1]],
        "continuous": [],
    }
    audit.equal(fino, expected_fino, f"{run} metadata plumbing drift")
    expected_stage = {
        "enabled": True,
        "mode": MODES[run],
        "study_id": STUDY_ID,
        "scenario": SCENARIO,
        "contract_receipt": str((root / "CALIBRATION_CONTRACT_RECEIPT.json").resolve()),
        "effective_config_receipt": str(
            (root / "configs" / f"{run}.yaml.receipt.json").resolve()
        ),
        "replay_manifest": str((root / "CALIBRATION_REPLAY_MANIFEST.json").resolve()),
        "adapter_init_seed": ADAPTER_INIT_SEED,
        "fair_weight": FAIR_WEIGHTS[run],
        "adapter_lr": ADAPTER_LR,
        "adapter_weight_decay": ADAPTER_WEIGHT_DECAY,
        "data_order_seed": DATA_ORDER_SEED,
        "replay": {
            "cancer_ids": list(CANCER_IDS),
            "race_ids": list(RACE_IDS),
            "steps": STEPS,
            "seed": REPLAY_SEED,
        },
    }
    for key, expected in expected_stage.items():
        audit.equal(stage.get(key), expected, f"{run} matched_stage.{key} drift")
    if run in PARENTS:
        parent = PARENTS[run]
        checkpoint = (root / parent / "latest.pt").resolve()
        parent_receipt = (root / parent / "COMPLETION_RECEIPT.json").resolve()
        audit.equal(
            Path(stage["encoder_checkpoint"]).resolve(),
            checkpoint,
            f"{run} encoder ancestry path drift",
        )
        audit.equal(
            stage["encoder_checkpoint_sha256"],
            sha256_file(checkpoint),
            f"{run} encoder checkpoint hash drift",
        )
        audit.equal(
            Path(stage["parent_completion_receipt"]).resolve(),
            parent_receipt,
            f"{run} parent receipt path drift",
        )
    else:
        for key in (
            "encoder_checkpoint",
            "encoder_checkpoint_sha256",
            "expected_encoder_state_sha256",
            "parent_completion_receipt",
        ):
            audit.equal(stage.get(key), None, f"{run} unexpected {key}")
    hits = _structured_forbidden_hits(config)
    audit.equal(hits, [], f"{run} contains downstream-outcome tokens")
    audit.equal(
        Path(config_path).resolve(),
        (root / "configs" / f"{run}.yaml").resolve(),
        f"{run} config location drift",
    )


def audit_job_root(job_root: Path) -> dict[str, Any]:
    audit = Audit()
    root = job_root.resolve(strict=True)
    root_receipt_path = root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
    # This first check is intentional: do not partially inspect an unfinished job.
    if not root_receipt_path.is_file():
        raise AuditError(
            "job is incomplete: ROOT_CALIBRATION_COMPLETION_RECEIPT.json is absent"
        )

    root_receipt = verify_receipt(
        root_receipt_path,
        audit,
        schema="matched-cancer-stage-calibration-root-completion/v1",
        expected_identity_roles={"contract_receipt", "replay_manifest", "runs"},
    )
    audit.equal(
        root_receipt.get("status"),
        "matched_cancer_two_slot_calibration_seed32001_valid",
        "root completion status drift",
    )
    audit.equal(root_receipt.get("representation_seed"), REPRESENTATION_SEED, "root seed drift")
    audit.equal(root_receipt.get("steps_per_run"), STEPS, "root step count drift")
    audit.equal(
        root_receipt.get("presentations_per_run"),
        PRESENTATIONS,
        "root exposure count drift",
    )
    audit.equal(root_receipt.get("arms"), list(RUNS), "root run topology drift")
    audit.equal(set(root_receipt["identities"]["runs"]), set(RUNS), "root run identities drift")
    _assert_identity(
        audit,
        _identity_at(root_receipt, "contract_receipt"),
        root / "CALIBRATION_CONTRACT_RECEIPT.json",
        "root contract receipt identity drift",
    )
    _assert_identity(
        audit,
        _identity_at(root_receipt, "replay_manifest"),
        root / "CALIBRATION_REPLAY_MANIFEST.json",
        "root replay identity drift",
    )
    for run in RUNS:
        _assert_identity(
            audit,
            _identity_at(root_receipt, "runs", run),
            root / run / "COMPLETION_RECEIPT.json",
            f"root {run} completion identity drift",
        )
    semantic_run_dirs = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and (child / "summary.json").exists()
    }
    audit.equal(semantic_run_dirs, set(RUNS), "job has extra or missing semantic runs")

    contract_receipt_path = root / "CALIBRATION_CONTRACT_RECEIPT.json"
    contract_receipt = verify_receipt(
        contract_receipt_path,
        audit,
        schema="matched-cancer-stage-provenance/v1",
        expected_identity_roles={
            "calibration_lock",
            "contract_config",
            "base_config",
            "exclusions",
            "cancer_race_population",
            "fino_meta",
            "runtime_sources",
        },
    )
    audit.equal(contract_receipt.get("status"), "valid", "contract status drift")
    audit.equal(
        contract_receipt.get("representation_seed"),
        REPRESENTATION_SEED,
        "contract representation seed drift",
    )
    audit.equal(
        contract_receipt.get("replay_presentations"),
        PRESENTATIONS,
        "contract exposure count drift",
    )
    audit.equal(contract_receipt.get("eligible_patients"), 699, "population size drift")
    audit.equal(
        contract_receipt.get("strata"),
        {
            "BRCA/black or african american": 56,
            "BRCA/white": 521,
            "LUAD/black or african american": 11,
            "LUAD/white": 111,
        },
        "population strata drift",
    )
    contract_yaml_path = Path(
        _identity_at(contract_receipt, "contract_config")["canonical_path"]
    )
    contract_yaml = yaml.safe_load(required_file(contract_yaml_path).read_text())
    expected_contract_yaml = {
        "schema": "matched-cancer-stage-calibration/v1",
        "study_id": STUDY_ID,
        "scenario": SCENARIO,
        "representation_seed": REPRESENTATION_SEED,
        "base_config": "configs_vendor/matched_cancer_stage_20260730/calibration_seed32001_base.yaml",
        "exclude_barcodes_file": "configs_vendor/matched_stage_union_20260730/exclude_union_target_hospitals.txt",
        "population": {"cancer_ids": list(CANCER_IDS), "race_ids": list(RACE_IDS)},
        "replay": {"steps": STEPS, "batch_size": BATCH_SIZE, "seed": REPLAY_SEED},
        "data_order_seed": DATA_ORDER_SEED,
        "adapter": {
            "init_seed": ADAPTER_INIT_SEED,
            "lr": ADAPTER_LR,
            "weight_decay": ADAPTER_WEIGHT_DECAY,
        },
        "fair_weight": 0.1,
        "arms": {
            "B": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.0},
            "P": {"slot1_fair_weight": 0.1, "slot2_fair_weight": 0.0},
            "H": {"slot1_fair_weight": 0.0, "slot2_fair_weight": 0.1},
        },
    }
    audit.equal(contract_yaml, expected_contract_yaml, "frozen contract YAML drift")
    audit.equal(
        _structured_forbidden_hits(contract_yaml),
        [],
        "contract YAML contains downstream-outcome tokens",
    )

    manifest_path = root / "CALIBRATION_REPLAY_MANIFEST.json"
    configs: dict[str, Mapping[str, Any]] = {}
    for run in RUNS:
        config_path = root / "configs" / f"{run}.yaml"
        config = yaml.safe_load(required_file(config_path).read_text())
        validate_config(run, config_path, config, root, audit)
        configs[run] = config
    first_config = configs["slot1_plain"]
    manifest = inspect_manifest(
        manifest_path,
        audit,
        verify_source_rows=True,
        effective_config=first_config,
        fino_meta_path=Path(_identity_at(contract_receipt, "fino_meta")["canonical_path"]),
        exclusions_path=Path(_identity_at(contract_receipt, "exclusions")["canonical_path"]),
    )
    audit.equal(
        root_receipt.get("replay_manifest_payload_sha256"),
        manifest["payload_sha256"],
        "root replay payload binding drift",
    )

    summaries: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    shared_schedule_hashes: dict[str, str] = {}
    mode_schedule_hashes: dict[str, str] = {}
    for run in RUNS:
        config_path = root / "configs" / f"{run}.yaml"
        config_receipt_path = config_path.with_suffix(".yaml.receipt.json")
        expected_config_roles = {
            "effective_config",
            "contract_receipt",
            "replay_manifest",
        }
        if run in PARENTS:
            expected_config_roles |= {
                "parent_completion_receipt",
                "encoder_checkpoint",
            }
        config_receipt = verify_receipt(
            config_receipt_path,
            audit,
            schema="matched-cancer-stage-effective-config/v1",
            expected_identity_roles=expected_config_roles,
        )
        audit.equal(config_receipt.get("mode"), MODES[run], f"{run} config receipt mode drift")
        audit.equal(
            config_receipt.get("fair_weight"),
            FAIR_WEIGHTS[run],
            f"{run} config receipt fair weight drift",
        )
        audit.equal(
            config_receipt.get("contract_topology_sha256"),
            contract_receipt["topology_sha256"],
            f"{run} contract topology ancestry drift",
        )
        _assert_identity(
            audit,
            _identity_at(config_receipt, "effective_config"),
            config_path,
            f"{run} effective config identity drift",
        )
        _assert_identity(
            audit,
            _identity_at(config_receipt, "contract_receipt"),
            contract_receipt_path,
            f"{run} contract receipt identity drift",
        )
        _assert_identity(
            audit,
            _identity_at(config_receipt, "replay_manifest"),
            manifest_path,
            f"{run} replay manifest identity drift",
        )
        if run in PARENTS:
            parent = PARENTS[run]
            _assert_identity(
                audit,
                _identity_at(config_receipt, "parent_completion_receipt"),
                root / parent / "COMPLETION_RECEIPT.json",
                f"{run} parent receipt identity drift",
            )
            _assert_identity(
                audit,
                _identity_at(config_receipt, "encoder_checkpoint"),
                root / parent / "latest.pt",
                f"{run} parent checkpoint identity drift",
            )

        completion_path = root / run / "COMPLETION_RECEIPT.json"
        completion = verify_receipt(
            completion_path,
            audit,
            schema="matched-cancer-stage-completion/v1",
            expected_identity_roles={
                "effective_config_receipt",
                "effective_config",
                "replay_manifest",
                "latest_checkpoint",
                "metrics",
                "summary",
            },
        )
        completions[run] = completion
        audit.equal(completion.get("status"), "complete", f"{run} completion status drift")
        audit.equal(completion.get("mode"), MODES[run], f"{run} completion mode drift")
        audit.equal(
            completion.get("fair_weight"), FAIR_WEIGHTS[run], f"{run} completion weight drift"
        )
        audit.equal(completion.get("steps_completed"), STEPS, f"{run} completion steps drift")
        audit.equal(
            completion.get("tile_presentations"),
            PRESENTATIONS,
            f"{run} completion exposure drift",
        )
        for role, expected_path in (
            ("effective_config_receipt", config_receipt_path),
            ("effective_config", config_path),
            ("replay_manifest", manifest_path),
            ("latest_checkpoint", root / run / "latest.pt"),
            ("metrics", root / run / "metrics.jsonl"),
            ("summary", root / run / "summary.json"),
        ):
            _assert_identity(
                audit,
                _identity_at(completion, role),
                expected_path,
                f"{run} completion {role} identity drift",
            )

        summary_path = root / run / "summary.json"
        summary = json.loads(required_file(summary_path).read_text())
        audit.require(isinstance(summary, dict), f"{run} summary is not object")
        summaries[run] = summary
        stage = summary["matched_stage"]
        audit.equal(summary.get("project"), f"matched-cancer-calibration-seed32001-{run}", f"{run} summary name drift")
        audit.equal(Path(summary["config_path"]).resolve(), config_path.resolve(), f"{run} summary config path drift")
        audit.equal(summary.get("batch_size"), BATCH_SIZE, f"{run} summary batch drift")
        audit.equal(summary.get("max_train_samples"), PRESENTATIONS, f"{run} sample budget drift")
        audit.equal(summary.get("stop_reason"), "max_train_samples", f"{run} stop reason drift")
        audit.equal(summary.get("steps_completed"), STEPS, f"{run} summary steps drift")
        audit.equal(summary.get("tile_presentations"), PRESENTATIONS, f"{run} summary exposure drift")
        audit.equal(summary.get("sample_fraction"), 1.0, f"{run} sample fraction drift")
        audit.equal(summary.get("probe_target_samples"), [], f"{run} probe targets present")
        audit.equal(summary.get("probe_target_fractions"), [], f"{run} probe target fractions present")
        audit.equal(stage.get("mode"), MODES[run], f"{run} summary mode drift")
        audit.equal(stage.get("fair_weight"), FAIR_WEIGHTS[run], f"{run} summary weight drift")
        audit.equal(stage.get("adapter_init_seed"), ADAPTER_INIT_SEED, f"{run} adapter seed drift")
        audit.equal(stage.get("adapter_lr"), ADAPTER_LR, f"{run} adapter LR drift")
        audit.equal(stage.get("adapter_weight_decay"), ADAPTER_WEIGHT_DECAY, f"{run} adapter WD drift")
        audit.equal(stage.get("data_order_seed"), DATA_ORDER_SEED, f"{run} data seed drift")
        audit.equal(stage.get("replay"), configs[run]["matched_stage"]["replay"], f"{run} replay config drift")
        expected_traces = {
            "replay_sampler_sha256": manifest["payload_sha256"],
            "replay_manifest_file_sha256": manifest["manifest_file_sha256"],
            "replay_patient_sha256": manifest["patient_sha256"],
            "replay_tile_sha256": manifest["tile_sha256"],
            "replay_augmentation_seed_sha256": manifest["augmentation_seed_sha256"],
            "augmentation_seed_manifest_trace_sha256": manifest["augmentation_seed_sha256"],
            "sample_batch_trace_sha256": manifest["sample_batch_trace_sha256"],
            "patient_batch_trace_sha256": manifest["patient_batch_trace_sha256"],
            "augmentation_seed_batch_trace_sha256": manifest[
                "augmentation_seed_batch_trace_sha256"
            ],
        }
        for key, expected in expected_traces.items():
            audit.equal(stage.get(key), expected, f"{run} {key} drift")
        audit.equal(
            Path(stage["replay_manifest_path"]).resolve(),
            manifest_path.resolve(),
            f"{run} replay path drift",
        )
        audit.equal(
            stage.get("contract_receipt_identity"),
            file_identity(contract_receipt_path),
            f"{run} runtime contract identity drift",
        )
        audit.equal(
            stage.get("effective_config_receipt_identity"),
            file_identity(config_receipt_path),
            f"{run} runtime config receipt identity drift",
        )
        audit.require(
            _valid_sha256(stage.get("adapter_pre_sha256"))
            and _valid_sha256(stage.get("adapter_post_sha256")),
            f"{run} adapter state hash invalid",
        )
        audit.require(
            stage["adapter_pre_sha256"] != stage["adapter_post_sha256"],
            f"{run} adapter did not change",
        )
        if MODES[run] == "joint":
            audit.require(not stage.get("encoder_unchanged"), f"{run} encoder did not change")
            audit.require(
                stage["encoder_pre_sha256"] != stage["encoder_post_sha256"],
                f"{run} encoder state is unchanged",
            )
            reach = stage.get("encoder_reachability")
            update = stage.get("encoder_first_update")
            audit.require(isinstance(reach, dict), f"{run} lacks reachability evidence")
            audit.require(isinstance(update, dict), f"{run} lacks first-update evidence")
            for key in ("encoder_cancer_grad_norm", "encoder_fair_raw_grad_norm"):
                audit.require(
                    math.isfinite(float(reach[key])) and float(reach[key]) > 0,
                    f"{run} invalid {key}",
                )
            audit.equal(reach.get("encoder_stage_grad_finite"), True, f"{run} non-finite reachability")
            expected_weighted = FAIR_WEIGHTS[run] * float(
                reach["encoder_fair_raw_grad_norm"]
            )
            audit.equal(
                float(reach["encoder_fair_weighted_grad_norm"]),
                expected_weighted,
                f"{run} weighted encoder fair reachability drift",
            )
            audit.require(
                int(reach["encoder_probe_parameter_count"]) > 0,
                f"{run} empty encoder reachability probe",
            )
            audit.require(
                _valid_sha256(reach["encoder_probe_parameter_names_sha256"]),
                f"{run} invalid encoder probe-name hash",
            )
            audit.require(
                float(update["encoder_first_positive_lr_update_norm"]) > 0,
                f"{run} first positive-LR update is zero",
            )
            audit.equal(
                int(update["encoder_first_positive_lr_changed_tensors"]),
                int(reach["encoder_probe_parameter_count"]),
                f"{run} not every probed encoder tensor changed",
            )
        else:
            audit.equal(stage.get("encoder_unchanged"), True, f"{run} encoder mutated")
            audit.equal(stage["encoder_pre_sha256"], stage["encoder_post_sha256"], f"{run} frozen encoder hash drift")
            audit.equal(stage.get("encoder_reachability"), None, f"{run} unexpected reachability")
            audit.equal(stage.get("encoder_first_update"), None, f"{run} unexpected encoder update")
        audit.equal(_structured_forbidden_hits(summary), [], f"{run} summary contains downstream tokens")

        checkpoint = torch.load(
            root / run / "latest.pt", map_location="cpu", weights_only=True
        )
        audit.require(isinstance(checkpoint, dict), f"{run} checkpoint is not mapping")
        audit.require("model" in checkpoint and "stage_adapter" in checkpoint, f"{run} checkpoint lacks stage states")
        encoder_state_sha = state_dict_sha256(checkpoint["model"])
        adapter_state_sha = state_dict_sha256(checkpoint["stage_adapter"])
        audit.equal(encoder_state_sha, stage["encoder_post_sha256"], f"{run} checkpoint encoder state drift")
        audit.equal(adapter_state_sha, stage["adapter_post_sha256"], f"{run} checkpoint adapter state drift")
        audit.equal(completion["encoder_pre_sha256"], stage["encoder_pre_sha256"], f"{run} completion encoder pre drift")
        audit.equal(completion["encoder_post_sha256"], encoder_state_sha, f"{run} completion encoder post drift")
        audit.equal(completion["adapter_pre_sha256"], stage["adapter_pre_sha256"], f"{run} completion adapter pre drift")
        audit.equal(completion["adapter_post_sha256"], adapter_state_sha, f"{run} completion adapter post drift")
        audit.equal(completion["replay_manifest_payload_sha256"], manifest["payload_sha256"], f"{run} completion replay payload drift")
        audit.equal(completion["replay_patient_sha256"], manifest["patient_sha256"], f"{run} completion patient trace drift")
        audit.equal(completion["replay_tile_sha256"], manifest["tile_sha256"], f"{run} completion tile trace drift")
        audit.equal(completion["replay_augmentation_seed_sha256"], manifest["augmentation_seed_sha256"], f"{run} completion augmentation trace drift")
        audit.equal(_structured_forbidden_hits(checkpoint.get("config", {})), [], f"{run} checkpoint config contains downstream tokens")

        _, shared_hash, mode_hash = load_metrics(
            root / run / "metrics.jsonl", audit, run=run
        )
        shared_schedule_hashes[run] = shared_hash
        mode_schedule_hashes[run] = mode_hash

    stages = {run: summaries[run]["matched_stage"] for run in RUNS}
    audit.equal(
        len({stages[run]["adapter_pre_sha256"] for run in RUNS}),
        1,
        "adapter initialization hash is not shared",
    )
    audit.equal(
        stages["slot1_plain"]["encoder_pre_sha256"],
        stages["slot1_fair"]["encoder_pre_sha256"],
        "Slot-1 initial encoder hash drift",
    )
    for run, parent in PARENTS.items():
        audit.equal(
            stages[run]["encoder_pre_sha256"],
            stages[parent]["encoder_post_sha256"],
            f"{run} encoder state ancestry drift",
        )
        audit.equal(
            stages[run]["expected_encoder_state_sha256"],
            stages[parent]["encoder_post_sha256"],
            f"{run} expected encoder state ancestry drift",
        )
        audit.equal(
            stages[run]["encoder_checkpoint_identity"],
            file_identity(root / parent / "latest.pt"),
            f"{run} checkpoint identity ancestry drift",
        )
        audit.equal(
            stages[run]["parent_completion_receipt_identity"],
            file_identity(root / parent / "COMPLETION_RECEIPT.json"),
            f"{run} parent receipt runtime ancestry drift",
        )
    audit.equal(
        stages["B"]["encoder_post_sha256"],
        stages["H"]["encoder_post_sha256"],
        "B/H do not share the frozen plain encoder",
    )
    audit.require(
        stages["P"]["encoder_post_sha256"] != stages["B"]["encoder_post_sha256"],
        "P fair Slot-1 ancestry collapsed to the plain encoder",
    )
    audit.require(
        stages["H"]["adapter_post_sha256"] != stages["B"]["adapter_post_sha256"],
        "H post-hoc fairness did not alter the adapter relative to B",
    )
    audit.equal(
        len(set(shared_schedule_hashes.values())),
        1,
        "shared optimization schedule drift across runs",
    )
    audit.equal(
        mode_schedule_hashes["slot1_plain"],
        mode_schedule_hashes["slot1_fair"],
        "Slot-1 optimizer schedule drift",
    )
    audit.equal(
        len({mode_schedule_hashes[run] for run in ("B", "H", "P")}),
        1,
        "Slot-2 optimizer schedule drift",
    )

    return {
        "status": "independent_calibration_audit_pass",
        "job_root": str(root),
        "study_id": STUDY_ID,
        "scenario": SCENARIO,
        "runs": list(RUNS),
        "steps_per_run": STEPS,
        "presentations_per_run": PRESENTATIONS,
        "total_presentations": PRESENTATIONS * len(RUNS),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "unique_dataset_indices": len(manifest["unique_identities"]),
        "receipt_and_artifact_checks": audit.checks,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--job-root", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = audit_job_root(args.job_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, KeyError, IndexError, OSError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "independent_calibration_audit_fail",
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
