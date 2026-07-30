#!/usr/bin/env python3
"""Independent, read-only semantic audit for one completed fixed48 calibration."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Mapping, Sequence

import yaml


STUDY_ID = "matched_cancer_fixed48_20260730"
REPO = Path("/admin/home/ryan.kim/nt")
OUTPUT_NAMESPACE = Path(
    "/data/ryan.kim/nanopath/reruns/"
    "matched_cancer_fixed48_20260730/calibration"
)
ROOT_SCHEMA = "matched-cancer-fixed48-calibration-root-completion/v1"
AUDIT_SCHEMA = "matched-cancer-fixed48-calibration-independent-audit/v1"
BASE_TEMPLATE_SEMANTIC_SHA256 = (
    "389ef0f62dbd09bb4c431712374bccffb8acecf9f25493cb3c8b7d6e0ac04e3e"
)
FROZEN_PLAN_PATHS = {
    "fixed_plan": (
        REPO
        / "configs_vendor/matched_cancer_fixed48_20260730/"
        "calibration_contract.yaml"
    ),
    "seed_plan": (
        REPO
        / "configs_vendor/matched_cancer_fixed48_20260730/"
        "calibration_seed_plan.yaml"
    ),
    "base_template": (
        REPO
        / "configs_vendor/matched_cancer_fixed48_20260730/"
        "calibration_base_template.yaml"
    ),
    "exclusions": (
        REPO
        / "configs_vendor/matched_stage_union_20260730/"
        "exclude_union_target_hospitals.txt"
    ),
    "cancer_race_population": (
        REPO
        / "configs_vendor/matched_cancer_stage_20260730/"
        "population_cancer_race.csv"
    ),
    "fino_meta": Path(
        "/data/ryan.kim/nanopath_parquet_fairness/fino_meta.json"
    ),
    "pretrained_checkpoint": Path(
        "/data/ryan.kim/nanopath/reruns/"
        "matched_cancer_fixed48_20260730/control/torch_home/"
        "hub/checkpoints/"
        "dinov2_vits14_reg4_pretrain.pth"
    ),
}
PRETRAINED_IDENTITY = {
    "canonical_path": str(FROZEN_PLAN_PATHS["pretrained_checkpoint"]),
    "bytes": 88_291_785,
    "sha256": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
}
PRETRAINED_ENCODER_STATE_SHA256 = (
    "ba9418ed2138e42250085b04e0502d621b072c4bb60240f2845a27fbf3184bd6"
)
RUNTIME_SOURCE_PATHS = {
    "package_init": (
        REPO / "tools/matched_cancer_fixed48_20260730/__init__.py"
    ),
    "contract": (
        REPO / "tools/matched_cancer_fixed48_20260730/contract.py"
    ),
    "runner": REPO / "tools/matched_cancer_fixed48_20260730/runner.py",
    "independent_auditor": (
        REPO / "tools/matched_cancer_fixed48_20260730/auditor.py"
    ),
    "calibration_driver": (
        REPO
        / "tools/matched_cancer_fixed48_20260730/"
        "calibration_one_seed.sbatch"
    ),
    "legacy_config_builder": (
        REPO / "tools/matched_cancer_stage_20260730/config_builder.py"
    ),
    "legacy_manifest_builder": (
        REPO / "tools/matched_cancer_stage_20260730/manifest_builder.py"
    ),
    "legacy_completion_receipt": (
        REPO / "tools/matched_cancer_stage_20260730/completion_receipt.py"
    ),
    "legacy_replay": (
        REPO / "tools/matched_cancer_stage_20260730/replay.py"
    ),
    "legacy_objectives": (
        REPO / "tools/matched_cancer_stage_20260730/objectives.py"
    ),
    "legacy_receipts": (
        REPO / "tools/matched_cancer_stage_20260730/receipts.py"
    ),
    "shared_fair_supcon": (
        REPO / "tools/matched_stage_union_20260730/objectives.py"
    ),
    "instrumentation": (
        REPO / "tools/matched_stage_union_20260730/instrumentation.py"
    ),
    "train": REPO / "vendor/matched_stage_train_20260730/train.py",
    "dataloader": REPO / "vendor/matched_stage_train_20260730/dataloader.py",
    "model": REPO / "vendor/matched_stage_train_20260730/model.py",
    "probe": REPO / "vendor/matched_stage_train_20260730/probe.py",
}
RUNS = ("slot1_plain", "slot1_fair", "B", "H", "P")
MODES = {
    "slot1_plain": "joint",
    "slot1_fair": "joint",
    "B": "adapter_only",
    "H": "adapter_only",
    "P": "adapter_only",
}
WEIGHTS = {
    "slot1_plain": 0.0,
    "slot1_fair": 0.1,
    "B": 0.0,
    "H": 0.1,
    "P": 0.0,
}
FROZEN_EFFECTIVE_FINO = {
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
IDENTITY_KEYS = frozenset({"canonical_path", "bytes", "sha256"})
HEX = frozenset("0123456789abcdef")


class AuditError(ValueError):
    """A completed calibration violates its frozen fixed48 contract."""


class Checks:
    def __init__(self) -> None:
        self.count = 0

    def require(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise AuditError(message)

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        self.require(actual == expected, f"{message}: {actual!r} != {expected!r}")


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise AuditError(f"noncanonical JSON value: {error}") from error


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def required_file(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise AuditError(f"symlink is forbidden: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise AuditError(f"required file unavailable: {candidate}") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise AuditError(f"required nonempty file unavailable: {candidate}")
    return resolved


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    source = required_file(path)
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path | str) -> dict[str, Any]:
    source = required_file(path)
    return {
        "canonical_path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_canonical(path: Path | str) -> dict[str, Any]:
    source = required_file(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_duplicate_rejector)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AuditError(f"invalid JSON: {source}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root is not an object: {source}")
    if raw != canonical_json(value) + b"\n":
        raise AuditError(f"JSON is not canonical: {source}")
    return value


def _identity_records(
    value: Mapping[str, Any], prefix: str = ""
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    for name, nested in value.items():
        role = f"{prefix}.{name}" if prefix else name
        if isinstance(nested, Mapping) and set(nested) == IDENTITY_KEYS:
            records.append((role, nested))
        elif isinstance(nested, Mapping):
            records.extend(_identity_records(nested, role))
        else:
            raise AuditError(f"invalid identity role: {role}")
    return records


def topology(identities: Mapping[str, Any]) -> str:
    roles = []
    for role, item in _identity_records(identities):
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest).issubset(HEX)
        ):
            raise AuditError(f"invalid digest for identity role {role}")
        roles.append({"role": role, "file_sha256": digest})
    if not roles:
        raise AuditError("receipt binds no files")
    payload = {
        "schema": "matched-cancer-stage-topology/v1",
        "roles": sorted(roles, key=lambda row: row["role"]),
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def verify_receipt(
    path: Path,
    checks: Checks,
    *,
    schema: str,
    scenario: str,
) -> dict[str, Any]:
    receipt = load_canonical(path)
    checks.equal(receipt.get("schema"), schema, f"{path} schema drift")
    checks.equal(receipt.get("study_id"), STUDY_ID, f"{path} study drift")
    checks.equal(receipt.get("scenario"), scenario, f"{path} scenario drift")
    identities = receipt.get("identities")
    checks.require(isinstance(identities, dict), f"{path} lacks identities")
    checks.equal(
        receipt.get("topology_sha256"),
        topology(identities),
        f"{path} topology drift",
    )
    for role, recorded in _identity_records(identities):
        checks.equal(
            identity(recorded["canonical_path"]),
            dict(recorded),
            f"{path} bound file changed for {role}",
        )
    return receipt


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit receipt: {path}")
    payload = canonical_json(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _patient_key(patient: str) -> int:
    raw = hashlib.blake2b(patient.encode(), digest_size=8).digest()
    return int.from_bytes(raw, "big") & 0x7FFFFFFFFFFFFFFF


def _runtime_batch_trace(values: Sequence[int], batch_size: int = 128) -> str:
    if len(values) % batch_size:
        raise AuditError("runtime trace length is not batch-divisible")
    digest = hashlib.sha256()
    for offset in range(0, len(values), batch_size):
        batch = values[offset : offset + batch_size]
        digest.update(len(batch).to_bytes(8, "little", signed=False))
        digest.update(struct.pack(f"<{len(batch)}q", *batch))
    return digest.hexdigest()


def _audit_manifest(
    path: Path, checks: Checks, *, seed: int
) -> tuple[
    dict[str, Any],
    str,
    dict[int, dict[str, Any]],
    dict[str, str],
]:
    manifest = load_canonical(path)
    checks.equal(
        set(manifest),
        {
            "schema",
            "contract",
            "occurrences",
            "traces",
            "manifest_payload_sha256",
        },
        "manifest keys drift",
    )
    checks.equal(
        manifest["schema"],
        "matched-cancer-replay-manifest/v1",
        "manifest schema drift",
    )
    expected_contract = {
        "cancer_ids": [2, 15],
        "race_ids": [2, 4],
        "batch_size": 128,
        "steps": 781,
        "seed": seed + 20_000,
    }
    checks.equal(manifest["contract"], expected_contract, "replay contract drift")
    occurrences = manifest["occurrences"]
    checks.require(isinstance(occurrences, list), "occurrences must be a list")
    checks.equal(len(occurrences), 99_968, "replay exposure drift")
    body = {
        "schema": manifest["schema"],
        "contract": manifest["contract"],
        "occurrences": occurrences,
        "traces": manifest["traces"],
    }
    payload_hash = hashlib.sha256(canonical_json(body)).hexdigest()
    checks.equal(
        manifest["manifest_payload_sha256"],
        payload_hash,
        "manifest payload hash drift",
    )
    counts: Counter[tuple[int, int, int]] = Counter()
    patient_values = []
    tile_values = []
    augmentation_values = []
    dataset_index_values = []
    patient_key_values = []
    identity_by_index: dict[int, dict[str, Any]] = {}
    expected_occurrence_keys = {
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
    for index, row in enumerate(occurrences):
        checks.require(isinstance(row, dict), f"occurrence {index} is invalid")
        checks.equal(
            set(row), expected_occurrence_keys, f"occurrence {index} keys drift"
        )
        batch, position = divmod(index, 128)
        checks.equal(row.get("batch"), batch, f"occurrence {index} batch drift")
        checks.equal(
            row.get("position"), position, f"occurrence {index} position drift"
        )
        stratum = (row.get("cancer"), row.get("race"))
        checks.require(
            stratum in {(2, 2), (2, 4), (15, 2), (15, 4)},
            f"occurrence {index} stratum drift",
        )
        for key in (
            "batch",
            "position",
            "dataset_index",
            "row",
            "cancer",
            "race",
            "augmentation_seed",
        ):
            checks.require(
                type(row[key]) is int, f"occurrence {index} {key} is not int"
            )
        checks.require(row["dataset_index"] >= 0, "negative dataset index")
        checks.require(row["row"] >= 0, "negative source row")
        checks.require(
            0 <= row["augmentation_seed"] < 2**63 - 1,
            "augmentation seed outside int64 range",
        )
        for key in ("shard_path", "tile_path", "patient"):
            checks.require(
                isinstance(row[key], str) and bool(row[key]),
                f"occurrence {index} empty {key}",
            )
        for key in ("shard_sha256", "tile_jpeg_sha256"):
            checks.require(
                isinstance(row[key], str)
                and len(row[key]) == 64
                and set(row[key]).issubset(HEX),
                f"occurrence {index} invalid {key}",
            )
        expected_patient = "-".join(
            row["tile_path"].split("/", 1)[0].split("-")[:3]
        )
        checks.equal(
            row["patient"], expected_patient, f"occurrence {index} patient drift"
        )
        source_identity = {
            key: row[key]
            for key in expected_occurrence_keys
            if key not in {"batch", "position", "augmentation_seed"}
        }
        previous = identity_by_index.setdefault(
            row["dataset_index"], source_identity
        )
        checks.equal(
            source_identity,
            previous,
            f"dataset index {row['dataset_index']} has multiple identities",
        )
        counts[(batch, *stratum)] += 1
        patient_values.append(row.get("patient"))
        tile_values.append(
            {
                "shard_sha256": row.get("shard_sha256"),
                "row": row.get("row"),
                "tile_path": row.get("tile_path"),
                "tile_jpeg_sha256": row.get("tile_jpeg_sha256"),
            }
        )
        augmentation_values.append(row.get("augmentation_seed"))
        dataset_index_values.append(row["dataset_index"])
        patient_key_values.append(_patient_key(row["patient"]))
    checks.equal(set(counts.values()), {32}, "per-batch stratum balance drift")
    expected_traces = {
        "patient_sha256": hashlib.sha256(
            canonical_json(patient_values)
        ).hexdigest(),
        "tile_sha256": hashlib.sha256(canonical_json(tile_values)).hexdigest(),
        "augmentation_seed_sha256": hashlib.sha256(
            canonical_json(augmentation_values)
        ).hexdigest(),
    }
    checks.equal(manifest["traces"], expected_traces, "manifest trace drift")
    runtime_traces = {
        "sample_batch_trace_sha256": _runtime_batch_trace(
            dataset_index_values
        ),
        "patient_batch_trace_sha256": _runtime_batch_trace(
            patient_key_values
        ),
        "augmentation_seed_batch_trace_sha256": _runtime_batch_trace(
            augmentation_values
        ),
        "augmentation_seed_manifest_trace_sha256": expected_traces[
            "augmentation_seed_sha256"
        ],
    }
    return manifest, payload_hash, identity_by_index, runtime_traces


def _patient_in_val(patient: str, seed: int, fraction: float) -> bool:
    key = f"{seed}:{patient}".encode()
    value = int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(), "big"
    ) / 2**64
    return value < fraction


def _verify_occurrence_sources(
    identities: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    fino_meta_path: Path,
    exclusions_path: Path,
    checks: Checks,
) -> None:
    """Rebuild dataset indices and verify every unique Parquet/JPEG identity."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise AuditError("pyarrow is required for source-row audit") from error
    data = config["data"]
    dataset_dir = Path(data["dataset_dir"]).resolve(strict=True)
    shards = sorted(dataset_dir.glob("shard-*.parquet"))
    checks.require(bool(shards), "no source shards found")
    shard_set = {str(path.resolve(strict=True)) for path in shards}
    metadata = json.loads(required_file(fino_meta_path).read_text())
    exclusions = {
        line.strip()
        for line in required_file(exclusions_path).read_text().splitlines()
        if line.strip()
    }
    include = {
        str(factor): {int(item) for item in values}
        for factor, values in data["include_discrete"].items()
    }
    checks.equal(
        include, {"cancer": {2, 15}, "race": {2, 4}}, "source filter drift"
    )
    targets = set(identities)
    rebuilt: dict[int, tuple[str, int, str]] = {}
    retained_index = 0
    split_seed = int(data["split_seed"])
    val_fraction = float(data["val_fraction"])
    for shard in shards:
        canonical_shard = str(shard.resolve(strict=True))
        table = pq.read_table(str(shard), columns=["path"], memory_map=True)
        for row_index, tile_path in enumerate(table["path"].to_pylist()):
            patient = "-".join(tile_path.split("/", 1)[0].split("-")[:3])
            if patient in exclusions:
                continue
            if any(
                int(metadata["discrete"][factor].get(patient, -1))
                not in allowed
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
    checks.equal(set(rebuilt), targets, "dataset-index rebuild is incomplete")
    by_shard: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for dataset_index, item in identities.items():
        shard_path = item["shard_path"]
        checks.require(shard_path in shard_set, f"unknown shard: {shard_path}")
        checks.equal(
            rebuilt[dataset_index],
            (shard_path, item["row"], item["tile_path"]),
            f"dataset index mapping drift: {dataset_index}",
        )
        by_shard[shard_path][int(item["row"])] = item
    for shard_path, rows in by_shard.items():
        claimed_hashes = {item["shard_sha256"] for item in rows.values()}
        checks.equal(len(claimed_hashes), 1, f"multiple hashes for {shard_path}")
        checks.equal(
            sha256_file(shard_path),
            next(iter(claimed_hashes)),
            f"source shard bytes drift: {shard_path}",
        )
        reader = pq.ParquetFile(shard_path, memory_map=True)
        ends: list[int] = []
        running = 0
        for group in range(reader.metadata.num_row_groups):
            running += reader.metadata.row_group(group).num_rows
            ends.append(running)
        grouped: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for row, item in rows.items():
            group = bisect.bisect_right(ends, row)
            checks.require(group < len(ends), f"row outside shard: {row}")
            start = 0 if group == 0 else ends[group - 1]
            grouped[group].append((row - start, item))
        for group, requested in grouped.items():
            table = reader.read_row_group(group, columns=["path", "jpeg"])
            for row_in_group, item in requested:
                tile_path = table["path"][row_in_group].as_py()
                jpeg = table["jpeg"][row_in_group].as_py()
                checks.equal(tile_path, item["tile_path"], "Parquet tile path drift")
                checks.equal(
                    hashlib.sha256(jpeg).hexdigest(),
                    item["tile_jpeg_sha256"],
                    "JPEG bytes drift",
                )
                patient = "-".join(tile_path.split("/", 1)[0].split("-")[:3])
                checks.equal(patient, item["patient"], "Parquet patient drift")
                checks.equal(
                    int(metadata["discrete"]["cancer"].get(patient, -1)),
                    item["cancer"],
                    "cancer metadata drift",
                )
                checks.equal(
                    int(metadata["discrete"]["race"].get(patient, -1)),
                    item["race"],
                    "race metadata drift",
                )


def audit(root: Path | str, *, seed: int) -> Path:
    """Audit a sealed root without importing any production calibration code."""
    if type(seed) is not int or seed not in range(32001, 32049):
        raise AuditError("seed must be in 32001..32048")
    attempt_root = Path(root)
    if attempt_root.is_symlink():
        raise AuditError("attempt root may not be a symlink")
    attempt_root = attempt_root.resolve(strict=True)
    if attempt_root.parent.parent != OUTPUT_NAMESPACE.resolve():
        raise AuditError(
            "attempt root is outside the exact fixed48 calibration namespace"
        )
    if attempt_root.parent.name != f"seed_{seed}":
        raise AuditError("attempt root seed directory mismatch")
    if re.fullmatch(r"attempt_[0-9]{2,}", attempt_root.name) is None:
        raise AuditError("attempt root must be named attempt_NN")
    destination = attempt_root / "INDEPENDENT_CALIBRATION_AUDIT_RECEIPT.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit receipt: {destination}")
    scenario = f"brca_luad_black_white_calibration_seed{seed}"
    checks = Checks()
    root_receipt = verify_receipt(
        attempt_root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json",
        checks,
        schema=ROOT_SCHEMA,
        scenario=scenario,
    )
    checks.equal(
        set(root_receipt["identities"]),
        {"contract_receipt", "replay_manifest", "runs"},
        "root identity roles drift",
    )
    checks.equal(
        set(root_receipt["identities"]["runs"]),
        set(RUNS),
        "root run identity roles drift",
    )
    checks.equal(
        root_receipt["identities"]["contract_receipt"],
        identity(attempt_root / "CALIBRATION_CONTRACT_RECEIPT.json"),
        "root contract identity redirect",
    )
    checks.equal(
        root_receipt["identities"]["replay_manifest"],
        identity(attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"),
        "root replay identity redirect",
    )
    for run in RUNS:
        checks.equal(
            root_receipt["identities"]["runs"][run],
            identity(attempt_root / run / "COMPLETION_RECEIPT.json"),
            f"root {run} identity redirect",
        )
    checks.equal(
        root_receipt.get("status"),
        "fixed48_two_slot_calibration_complete",
        "root status drift",
    )
    checks.equal(root_receipt.get("representation_seed"), seed, "root seed drift")
    checks.equal(root_receipt.get("steps_per_run"), 781, "root steps drift")
    checks.equal(
        root_receipt.get("presentations_per_run"), 99_968, "root exposure drift"
    )
    checks.equal(
        root_receipt.get("total_presentations"), 499_840, "root total drift"
    )
    checks.equal(root_receipt.get("run_names"), list(RUNS), "run order drift")
    checks.equal(
        root_receipt.get("legacy_seed32001_disposition"),
        {
            "disposition": "systems_only_excluded_from_inference",
            "reusable": False,
            "rerun_in_fixed48_namespace": True,
        },
        "legacy disposition drift",
    )
    contract_receipt = verify_receipt(
        attempt_root / "CALIBRATION_CONTRACT_RECEIPT.json",
        checks,
        schema="matched-cancer-stage-provenance/v1",
        scenario=scenario,
    )
    checks.equal(
        set(contract_receipt["identities"]),
        {
            "fixed_plan",
            "seed_plan",
            "base_template",
            "materialized_base",
            "materialized_contract",
            "exclusions",
            "cancer_race_population",
            "fino_meta",
            "pretrained_checkpoint",
            "runtime_sources",
        },
        "contract identity roles drift",
    )
    checks.equal(
        contract_receipt["identities"]["materialized_base"],
        identity(attempt_root / "configs" / "calibration_base.yaml"),
        "materialized base identity redirect",
    )
    checks.equal(
        contract_receipt["identities"]["materialized_contract"],
        identity(attempt_root / "configs" / "calibration_contract.yaml"),
        "materialized contract identity redirect",
    )
    for role, expected_path in FROZEN_PLAN_PATHS.items():
        checks.equal(
            contract_receipt["identities"][role],
            identity(expected_path),
            f"frozen provenance path redirect: {role}",
        )
    checks.equal(
        contract_receipt["identities"]["pretrained_checkpoint"],
        PRETRAINED_IDENTITY,
        "frozen DINOv2 pretrained checkpoint identity drift",
    )
    runtime_sources = contract_receipt["identities"]["runtime_sources"]
    checks.equal(
        set(runtime_sources),
        set(RUNTIME_SOURCE_PATHS),
        "runtime source role topology drift",
    )
    for role, expected_path in RUNTIME_SOURCE_PATHS.items():
        checks.equal(
            runtime_sources[role],
            identity(expected_path),
            f"runtime source canonical path redirect: {role}",
        )
    for key, expected in {
        "representation_seed": seed,
        "replay_seed": seed + 20_000,
        "data_order_seed": seed + 30_000,
        "adapter_init_seed": seed + 40_000,
        "replay_presentations": 99_968,
        "eligible_patients": 699,
    }.items():
        checks.equal(contract_receipt.get(key), expected, f"contract {key} drift")
    manifest, payload_hash, source_identities, runtime_traces = _audit_manifest(
        attempt_root / "CALIBRATION_REPLAY_MANIFEST.json",
        checks,
        seed=seed,
    )
    checks.equal(
        root_receipt.get("replay_manifest_payload_sha256"),
        payload_hash,
        "root replay payload drift",
    )

    stages: dict[str, dict[str, Any]] = {}
    first_config: dict[str, Any] | None = None
    materialized_base = yaml.safe_load(
        required_file(
            attempt_root / "configs" / "calibration_base.yaml"
        ).read_text()
    )
    checks.require(
        isinstance(materialized_base, dict), "materialized base is invalid"
    )
    expected_base = dict(materialized_base)
    expected_base["train"] = dict(materialized_base["train"])
    expected_base["train"]["seed"] = None
    expected_base["project"] = dict(materialized_base["project"])
    expected_base["project"]["name"] = "matched-cancer-fixed48-unmaterialized"
    expected_base["project"]["output_dir"] = str(OUTPUT_NAMESPACE)
    checks.equal(
        _semantic_sha256(expected_base),
        BASE_TEMPLATE_SEMANTIC_SHA256,
        "materialized base differs from frozen full semantic template",
    )
    expected_data = dict(materialized_base["data"])
    expected_data["include_discrete"] = {
        "cancer": [2, 15],
        "race": [2, 4],
    }
    expected_stage_keys = {
        "enabled",
        "mode",
        "study_id",
        "scenario",
        "contract_receipt",
        "effective_config_receipt",
        "replay_manifest",
        "adapter_init_seed",
        "fair_weight",
        "adapter_lr",
        "adapter_weight_decay",
        "data_order_seed",
        "encoder_checkpoint",
        "encoder_checkpoint_sha256",
        "expected_encoder_state_sha256",
        "parent_completion_receipt",
        "replay",
    }
    for run in RUNS:
        config_path = required_file(attempt_root / "configs" / f"{run}.yaml")
        config = yaml.safe_load(config_path.read_text())
        checks.require(isinstance(config, dict), f"{run} config is invalid")
        if first_config is None:
            first_config = config
        checks.equal(
            set(config),
            set(materialized_base) | {"matched_stage"},
            f"{run} top-level representation schema drift",
        )
        checks.equal(
            config["data"],
            expected_data,
            f"{run} representation data must be exactly pixels/cancer/race",
        )
        checks.equal(
            config["model"], materialized_base["model"], f"{run} model drift"
        )
        checks.equal(
            config["dino"], materialized_base["dino"], f"{run} DINO drift"
        )
        checks.equal(
            config["probe"], materialized_base["probe"], f"{run} probe drift"
        )
        checks.equal(
            config["fino"],
            FROZEN_EFFECTIVE_FINO,
            f"{run} fairness metadata must be exactly cancer/race",
        )
        checks.equal(config["train"]["seed"], seed, f"{run} seed drift")
        checks.equal(
            config["train"]["max_train_samples"],
            99_968,
            f"{run} exposure drift",
        )
        checks.equal(config["probe"]["enabled"], False, f"{run} probe enabled")
        stage_config = config["matched_stage"]
        checks.equal(
            set(stage_config),
            expected_stage_keys,
            f"{run} matched_stage key topology drift",
        )
        checks.equal(
            set(stage_config["replay"]),
            {"cancer_ids", "race_ids", "steps", "seed"},
            f"{run} replay key topology drift",
        )
        checks.equal(
            stage_config["replay"]["cancer_ids"],
            [2, 15],
            f"{run} cancer IDs drift",
        )
        checks.equal(
            stage_config["replay"]["race_ids"],
            [2, 4],
            f"{run} race IDs drift",
        )
        checks.equal(stage_config["mode"], MODES[run], f"{run} mode drift")
        checks.equal(
            stage_config["fair_weight"], WEIGHTS[run], f"{run} weight drift"
        )
        checks.equal(
            stage_config["adapter_init_seed"],
            seed + 40_000,
            f"{run} adapter seed drift",
        )
        checks.equal(
            stage_config["data_order_seed"],
            seed + 30_000,
            f"{run} data-order seed drift",
        )
        verify_receipt(
            attempt_root / "configs" / f"{run}.yaml.receipt.json",
            checks,
            schema="matched-cancer-stage-effective-config/v1",
            scenario=scenario,
        )
        completion = verify_receipt(
            attempt_root / run / "COMPLETION_RECEIPT.json",
            checks,
            schema="matched-cancer-stage-completion/v1",
            scenario=scenario,
        )
        checks.equal(
            set(completion["identities"]),
            {
                "effective_config_receipt",
                "effective_config",
                "replay_manifest",
                "latest_checkpoint",
                "metrics",
                "summary",
            },
            f"{run} completion identity roles drift",
        )
        for role, local_path in {
            "effective_config_receipt": (
                attempt_root / "configs" / f"{run}.yaml.receipt.json"
            ),
            "effective_config": attempt_root / "configs" / f"{run}.yaml",
            "replay_manifest": (
                attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"
            ),
            "latest_checkpoint": attempt_root / run / "latest.pt",
            "metrics": attempt_root / run / "metrics.jsonl",
            "summary": attempt_root / run / "summary.json",
        }.items():
            checks.equal(
                completion["identities"][role],
                identity(local_path),
                f"{run} completion {role} redirect",
            )
        checks.equal(
            completion.get("steps_completed"), 781, f"{run} completion steps drift"
        )
        checks.equal(
            completion.get("tile_presentations"),
            99_968,
            f"{run} completion exposure drift",
        )
        summary = json.loads(
            required_file(attempt_root / run / "summary.json").read_text()
        )
        checks.equal(summary.get("steps_completed"), 781, f"{run} steps drift")
        checks.equal(
            summary.get("tile_presentations"), 99_968, f"{run} exposure drift"
        )
        checks.equal(
            summary.get("stop_reason"),
            "max_train_samples",
            f"{run} stop drift",
        )
        stages[run] = summary["matched_stage"]
        checks.equal(stages[run]["mode"], MODES[run], f"{run} summary mode drift")
        checks.equal(
            stages[run]["fair_weight"],
            WEIGHTS[run],
            f"{run} summary weight drift",
        )
        rows = [
            json.loads(line)
            for line in required_file(
                attempt_root / run / "metrics.jsonl"
            ).read_text().splitlines()
            if line.strip() and '"total"' in line
        ]
        checks.equal(len(rows), 781, f"{run} metric count drift")
        for row in rows:
            for key in (
                "cancer",
                "race_fair",
                "total",
                "h_dose_main_grad_norm",
                "h_dose_fair_grad_norm",
            ):
                checks.require(
                    math.isfinite(float(row[key])),
                    f"{run} non-finite metric {key}",
                )
            checks.require(float(row["cancer"]) > 0, f"{run} cancer inactive")
            fair_grad = float(row["h_dose_fair_grad_norm"])
            if WEIGHTS[run] > 0:
                checks.require(fair_grad > 0, f"{run} fairness inactive")
            else:
                checks.equal(fair_grad, 0.0, f"{run} fairness should be off")

    for key in (
        "adapter_pre_sha256",
        "replay_sampler_sha256",
        "replay_patient_sha256",
        "replay_tile_sha256",
        "replay_augmentation_seed_sha256",
        "replay_manifest_file_sha256",
    ):
        checks.equal(
            len({stages[run][key] for run in RUNS}),
            1,
            f"five-run {key} mismatch",
        )
    manifest_file_sha256 = sha256_file(
        attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"
    )
    for run in RUNS:
        checks.equal(
            stages[run]["replay_sampler_sha256"],
            payload_hash,
            f"{run} replay payload trace drift",
        )
        checks.equal(
            stages[run]["replay_patient_sha256"],
            manifest["traces"]["patient_sha256"],
            f"{run} patient replay trace drift",
        )
        checks.equal(
            stages[run]["replay_tile_sha256"],
            manifest["traces"]["tile_sha256"],
            f"{run} tile replay trace drift",
        )
        checks.equal(
            stages[run]["replay_augmentation_seed_sha256"],
            manifest["traces"]["augmentation_seed_sha256"],
            f"{run} augmentation replay trace drift",
        )
        checks.equal(
            stages[run]["replay_manifest_file_sha256"],
            manifest_file_sha256,
            f"{run} replay file trace drift",
        )
        for key, expected in runtime_traces.items():
            checks.equal(
                stages[run][key],
                expected,
                f"{run} independently computed runtime trace drift: {key}",
            )
        checks.require(
            stages[run]["adapter_post_sha256"]
            != stages[run]["adapter_pre_sha256"],
            f"{run} trained adapter did not change",
        )
    checks.equal(
        stages["slot1_plain"]["encoder_pre_sha256"],
        stages["slot1_fair"]["encoder_pre_sha256"],
        "Slot1 initialization mismatch",
    )
    checks.equal(
        stages["slot1_plain"]["encoder_pre_sha256"],
        PRETRAINED_ENCODER_STATE_SHA256,
        "Slot1 pretrained encoder ancestor drift",
    )
    checks.equal(
        stages["B"]["encoder_pre_sha256"],
        stages["slot1_plain"]["encoder_post_sha256"],
        "B ancestry drift",
    )
    checks.equal(
        stages["H"]["encoder_pre_sha256"],
        stages["slot1_plain"]["encoder_post_sha256"],
        "H ancestry drift",
    )
    checks.equal(
        stages["P"]["encoder_pre_sha256"],
        stages["slot1_fair"]["encoder_post_sha256"],
        "P ancestry drift",
    )
    for run in ("B", "H", "P"):
        checks.equal(stages[run]["encoder_unchanged"], True, f"{run} not frozen")
    if first_config is None:
        raise AuditError("no effective configuration was audited")
    _verify_occurrence_sources(
        source_identities,
        first_config,
        fino_meta_path=Path(
            contract_receipt["identities"]["fino_meta"]["canonical_path"]
        ),
        exclusions_path=Path(
            contract_receipt["identities"]["exclusions"]["canonical_path"]
        ),
        checks=checks,
    )

    audit_identities = {
        "auditor_source": identity(Path(__file__)),
        "root_completion_receipt": identity(
            attempt_root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
        ),
        "contract_receipt": identity(
            attempt_root / "CALIBRATION_CONTRACT_RECEIPT.json"
        ),
        "replay_manifest": identity(
            attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"
        ),
        "runs": {
            run: {
                "effective_config": identity(
                    attempt_root / "configs" / f"{run}.yaml"
                ),
                "effective_config_receipt": identity(
                    attempt_root / "configs" / f"{run}.yaml.receipt.json"
                ),
                "completion_receipt": identity(
                    attempt_root / run / "COMPLETION_RECEIPT.json"
                ),
                "checkpoint": identity(attempt_root / run / "latest.pt"),
                "metrics": identity(attempt_root / run / "metrics.jsonl"),
                "summary": identity(attempt_root / run / "summary.json"),
            }
            for run in RUNS
        },
    }
    receipt = {
        "schema": AUDIT_SCHEMA,
        "study_id": STUDY_ID,
        "scenario": scenario,
        "identities": audit_identities,
        "topology_sha256": topology(audit_identities),
        "status": "fixed48_calibration_independent_audit_pass",
        "representation_seed": seed,
        "checks": checks.count,
        "replay_occurrences": len(manifest["occurrences"]),
        "run_names": list(RUNS),
        "values_or_outcomes_accessed": False,
    }
    _write_exclusive(destination, receipt)
    # Re-read and verify the persisted receipt without trusting the writer.
    verify_receipt(
        destination, Checks(), schema=AUDIT_SCHEMA, scenario=scenario
    )
    return destination


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    destination = audit(args.root, seed=args.seed)
    result = load_canonical(destination)
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": result["representation_seed"],
                "checks": result["checks"],
                "receipt": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
