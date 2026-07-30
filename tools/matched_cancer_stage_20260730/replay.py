"""Deterministic, identity-bound replay for cancer-stage interventions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA = "matched-cancer-replay-manifest/v1"
_CONTRACT_KEYS = {"cancer_ids", "race_ids", "batch_size", "steps", "seed"}
_IDENTITY_KEYS = {
    "dataset_index",
    "shard_path",
    "shard_sha256",
    "row",
    "tile_path",
    "tile_jpeg_sha256",
    "patient",
    "cancer",
    "race",
}
_OCCURRENCE_KEYS = {
    "batch",
    "position",
    *_IDENTITY_KEYS,
    "augmentation_seed",
}
_TOP_LEVEL_KEYS = {
    "schema",
    "contract",
    "occurrences",
    "traces",
    "manifest_payload_sha256",
}
_TRACE_KEYS = {
    "patient_sha256",
    "tile_sha256",
    "augmentation_seed_sha256",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sequence_sha256(values: Sequence[Any]) -> str:
    return _sha256_bytes(_canonical_json(list(values)))


def _trace_hashes(occurrences: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        "patient_sha256": _sequence_sha256(
            [row["patient"] for row in occurrences]
        ),
        "tile_sha256": _sequence_sha256(
            [
                {
                    "shard_sha256": row["shard_sha256"],
                    "row": row["row"],
                    "tile_path": row["tile_path"],
                    "tile_jpeg_sha256": row["tile_jpeg_sha256"],
                }
                for row in occurrences
            ]
        ),
        "augmentation_seed_sha256": _sequence_sha256(
            [row["augmentation_seed"] for row in occurrences]
        ),
    }


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_int(
    value: Any,
    where: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{where} must be <= {maximum}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in replay manifest: {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class ReplayContract:
    cancer_ids: tuple[int, ...]
    race_ids: tuple[int, ...]
    batch_size: int
    steps: int
    seed: int

    @property
    def strata(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (cancer, race)
            for cancer in self.cancer_ids
            for race in self.race_ids
        )

    @property
    def per_stratum(self) -> int:
        count = len(self.strata)
        if count == 0 or self.batch_size % count:
            raise ValueError("batch_size must be divisible by cancer/race strata")
        return self.batch_size // count

    def as_dict(self) -> dict[str, Any]:
        return {
            "cancer_ids": list(self.cancer_ids),
            "race_ids": list(self.race_ids),
            "batch_size": self.batch_size,
            "steps": self.steps,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayContract":
        if not isinstance(value, Mapping):
            raise ValueError("manifest contract must be a mapping")
        _exact_keys(value, _CONTRACT_KEYS, "manifest contract")
        cancer_ids = value["cancer_ids"]
        race_ids = value["race_ids"]
        if not isinstance(cancer_ids, list) or not cancer_ids:
            raise ValueError("manifest contract cancer_ids must be a nonempty list")
        if not isinstance(race_ids, list) or not race_ids:
            raise ValueError("manifest contract race_ids must be a nonempty list")
        contract = cls(
            cancer_ids=tuple(
                _require_int(item, "manifest cancer ID", 0) for item in cancer_ids
            ),
            race_ids=tuple(
                _require_int(item, "manifest race ID", 0) for item in race_ids
            ),
            batch_size=_require_int(
                value["batch_size"], "manifest batch_size", 1
            ),
            steps=_require_int(value["steps"], "manifest steps", 1),
            seed=_require_int(value["seed"], "manifest seed", 0),
        )
        if len(set(contract.cancer_ids)) != len(contract.cancer_ids):
            raise ValueError("manifest cancer_ids must be unique")
        if len(set(contract.race_ids)) != len(contract.race_ids):
            raise ValueError("manifest race_ids must be unique")
        contract.per_stratum
        return contract


class BalancedReplayBatchSampler:
    """Replay exact, validated dataset-index/augmentation-seed pairs by stratum."""

    def __init__(self, dataset, contract: ReplayContract) -> None:
        self._require_dataset(dataset)
        if contract.steps <= 0:
            raise ValueError("replay steps must be positive")
        contract.per_stratum
        cancer_map = dataset.meta_disc["cancer"]
        race_map = dataset.meta_disc["race"]
        pools: dict[tuple[int, int], list[int]] = {
            stratum: [] for stratum in contract.strata
        }
        for index, barcode in enumerate(dataset._tile_barcodes):
            stratum = (
                int(cancer_map.get(barcode, -1)),
                int(race_map.get(barcode, -1)),
            )
            if stratum in pools:
                pools[stratum].append(index)
        empty = [stratum for stratum, indices in pools.items() if not indices]
        if empty:
            raise ValueError(f"empty replay strata: {empty}")

        rng = np.random.default_rng(contract.seed)
        batches: list[tuple[tuple[int, int], ...]] = []
        for _ in range(contract.steps):
            rows: list[tuple[int, int]] = []
            for stratum in contract.strata:
                chosen = rng.choice(
                    pools[stratum],
                    size=contract.per_stratum,
                    replace=True,
                )
                seeds = rng.integers(
                    0, np.iinfo(np.int64).max, size=contract.per_stratum
                )
                rows.extend(
                    (int(index), int(seed))
                    for index, seed in zip(chosen, seeds, strict=True)
                )
            order = rng.permutation(len(rows))
            batches.append(tuple(rows[int(position)] for position in order))

        identity_cache: dict[int, dict[str, Any]] = {}
        occurrences: list[dict[str, Any]] = []
        try:
            self._clear_dataset_identity_cache(dataset)
            for batch_index, batch in enumerate(batches):
                for position, (dataset_index, augmentation_seed) in enumerate(
                    batch
                ):
                    if dataset_index not in identity_cache:
                        identity_cache[dataset_index] = self._runtime_identity(
                            dataset, dataset_index
                        )
                    occurrences.append(
                        {
                            "batch": batch_index,
                            "position": position,
                            **identity_cache[dataset_index],
                            "augmentation_seed": augmentation_seed,
                        }
                    )
        finally:
            self._clear_dataset_identity_cache(dataset)

        body = {
            "schema": MANIFEST_SCHEMA,
            "contract": contract.as_dict(),
            "occurrences": occurrences,
            "traces": _trace_hashes(occurrences),
        }
        self._initialize(contract, batches, body)

    @staticmethod
    def _require_dataset(dataset) -> None:
        if getattr(dataset, "_tile_barcodes", None) is None:
            raise ValueError("dataset must retain matched-stage barcodes")
        if not callable(getattr(dataset, "replay_identity", None)):
            raise ValueError(
                "dataset must provide replay_identity(dataset_index)"
            )
        if float(getattr(dataset, "tissue_thresh", 0.0)) != 0.0:
            raise ValueError(
                "identity-bound replay requires data.tissue_thresh == 0 "
                "because tissue fallback can substitute another tile"
            )

    @staticmethod
    def _clear_dataset_identity_cache(dataset) -> None:
        clear = getattr(dataset, "clear_replay_identity_cache", None)
        if callable(clear):
            clear()

    @staticmethod
    def _runtime_identity(dataset, dataset_index: int) -> dict[str, Any]:
        raw = dataset.replay_identity(dataset_index)
        if not isinstance(raw, Mapping):
            raise ValueError("dataset replay_identity must return a mapping")
        _exact_keys(raw, _IDENTITY_KEYS, "dataset replay identity")
        identity = dict(raw)
        for key in ("dataset_index", "row", "cancer", "race"):
            identity[key] = _require_int(
                identity[key], f"dataset replay identity {key}", 0
            )
        if identity["dataset_index"] != dataset_index:
            raise ValueError(
                "dataset replay identity returned a different dataset_index"
            )
        for key in (
            "shard_path",
            "shard_sha256",
            "tile_path",
            "tile_jpeg_sha256",
            "patient",
        ):
            if not isinstance(identity[key], str) or not identity[key]:
                raise ValueError(
                    f"dataset replay identity {key} must be a nonempty string"
                )
        for key in ("shard_sha256", "tile_jpeg_sha256"):
            value = identity[key]
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"dataset replay identity {key} must be lowercase SHA-256"
                )
        return identity

    def _initialize(
        self,
        contract: ReplayContract,
        batches: Sequence[Sequence[tuple[int, int]]],
        body: Mapping[str, Any],
        *,
        manifest_path: str | None = None,
        manifest_file_sha256: str | None = None,
    ) -> None:
        traces = body.get("traces")
        if not isinstance(traces, Mapping):
            raise ValueError("replay manifest traces must be a mapping")
        _exact_keys(traces, _TRACE_KEYS, "replay manifest traces")
        calculated_traces = _trace_hashes(body["occurrences"])
        if dict(traces) != calculated_traces:
            raise ValueError(
                "replay manifest trace hashes do not match occurrences"
            )
        payload_sha256 = _sha256_bytes(_canonical_json(body))
        manifest = {**body, "manifest_payload_sha256": payload_sha256}
        self.contract = contract
        self._batches = tuple(tuple(batch) for batch in batches)
        self._manifest = manifest
        # Backwards-compatible sampler hash now binds the complete tile identity.
        self.sha256 = payload_sha256
        self.patient_sha256 = calculated_traces["patient_sha256"]
        self.tile_sha256 = calculated_traces["tile_sha256"]
        self.augmentation_seed_sha256 = calculated_traces[
            "augmentation_seed_sha256"
        ]
        self.manifest_path = manifest_path
        self.manifest_file_sha256 = manifest_file_sha256

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a defensive JSON round-trip copy of the canonical manifest."""
        return json.loads(_canonical_json(self._manifest))

    def write_manifest(self, destination: str | Path) -> Path:
        """Create one immutable canonical manifest; never overwrite an existing file."""
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        canonical = _canonical_json(self._manifest) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(canonical)
                handle.flush()
                os.fsync(handle.fileno())
            # A hard link atomically publishes without replacing an existing lock.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.manifest_path = str(destination)
        self.manifest_file_sha256 = _sha256_bytes(canonical)
        return destination

    @classmethod
    def from_manifest(
        cls,
        dataset,
        manifest_path: str | Path,
        *,
        expected_contract: ReplayContract | None = None,
    ) -> "BalancedReplayBatchSampler":
        """Load the recorded batches and fail if any runtime tile identity differs."""
        cls._require_dataset(dataset)
        requested_manifest_path = Path(manifest_path)
        if requested_manifest_path.is_symlink():
            raise ValueError(
                f"replay manifest must not be a symlink: "
                f"{requested_manifest_path}"
            )
        manifest_path = requested_manifest_path.resolve()
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or manifest_path.stat().st_size == 0
        ):
            raise ValueError(
                f"replay manifest must be a nonempty regular nonsymlink file: "
                f"{manifest_path}"
            )
        raw = manifest_path.read_bytes()
        try:
            manifest = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid replay manifest JSON: {error}") from error
        if not isinstance(manifest, Mapping):
            raise ValueError("replay manifest root must be a mapping")
        _exact_keys(manifest, _TOP_LEVEL_KEYS, "replay manifest")
        if manifest["schema"] != MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported replay manifest schema: {manifest['schema']!r}"
            )
        if raw != _canonical_json(manifest) + b"\n":
            raise ValueError("replay manifest is not canonical JSON")
        claimed_digest = manifest["manifest_payload_sha256"]
        if not isinstance(claimed_digest, str):
            raise ValueError("manifest_payload_sha256 must be a string")
        body = {
            "schema": manifest["schema"],
            "contract": manifest["contract"],
            "occurrences": manifest["occurrences"],
            "traces": manifest["traces"],
        }
        actual_digest = _sha256_bytes(_canonical_json(body))
        if claimed_digest != actual_digest:
            raise ValueError(
                "replay manifest payload SHA-256 mismatch: "
                f"claimed={claimed_digest}, actual={actual_digest}"
            )

        contract = ReplayContract.from_dict(manifest["contract"])
        if expected_contract is not None and contract != expected_contract:
            raise ValueError(
                f"replay manifest contract mismatch: {contract!r} != "
                f"{expected_contract!r}"
            )
        occurrences = manifest["occurrences"]
        if not isinstance(occurrences, list):
            raise ValueError("replay manifest occurrences must be a list")
        expected_count = contract.steps * contract.batch_size
        if len(occurrences) != expected_count:
            raise ValueError(
                f"replay manifest has {len(occurrences)} occurrences, "
                f"expected {expected_count}"
            )

        batches: list[list[tuple[int, int]]] = [
            [] for _ in range(contract.steps)
        ]
        runtime_cache: dict[int, dict[str, Any]] = {}
        try:
            cls._clear_dataset_identity_cache(dataset)
            for occurrence_index, raw_row in enumerate(occurrences):
                if not isinstance(raw_row, Mapping):
                    raise ValueError(
                        f"replay occurrence {occurrence_index} must be a mapping"
                    )
                _exact_keys(
                    raw_row,
                    _OCCURRENCE_KEYS,
                    f"replay occurrence {occurrence_index}",
                )
                row = dict(raw_row)
                for key in (
                    "batch",
                    "position",
                    "dataset_index",
                    "row",
                    "cancer",
                    "race",
                ):
                    row[key] = _require_int(
                        row[key],
                        f"replay occurrence {occurrence_index} {key}",
                        0,
                    )
                row["augmentation_seed"] = _require_int(
                    row["augmentation_seed"],
                    f"replay occurrence {occurrence_index} augmentation_seed",
                    0,
                    int(np.iinfo(np.int64).max) - 1,
                )
                expected_batch, expected_position = divmod(
                    occurrence_index, contract.batch_size
                )
                if (
                    row["batch"] != expected_batch
                    or row["position"] != expected_position
                ):
                    raise ValueError(
                        f"replay occurrence {occurrence_index} has noncanonical "
                        "batch/position"
                    )
                dataset_index = row["dataset_index"]
                if dataset_index not in runtime_cache:
                    runtime_cache[dataset_index] = cls._runtime_identity(
                        dataset, dataset_index
                    )
                recorded_identity = {
                    key: row[key] for key in _IDENTITY_KEYS
                }
                if recorded_identity != runtime_cache[dataset_index]:
                    differing = sorted(
                        key
                        for key in _IDENTITY_KEYS
                        if recorded_identity.get(key)
                        != runtime_cache[dataset_index].get(key)
                    )
                    raise ValueError(
                        f"replay occurrence {occurrence_index} runtime tile "
                        f"identity mismatch in {differing}"
                    )
                batches[expected_batch].append(
                    (dataset_index, row["augmentation_seed"])
                )
        finally:
            cls._clear_dataset_identity_cache(dataset)

        for batch_index, batch in enumerate(batches):
            counts = {stratum: 0 for stratum in contract.strata}
            offset = batch_index * contract.batch_size
            for position in range(contract.batch_size):
                row = occurrences[offset + position]
                stratum = (row["cancer"], row["race"])
                if stratum not in counts:
                    raise ValueError(
                        f"replay batch {batch_index} contains out-of-contract "
                        f"stratum {stratum}"
                    )
                counts[stratum] += 1
            if any(value != contract.per_stratum for value in counts.values()):
                raise ValueError(
                    f"replay batch {batch_index} is not exactly balanced: "
                    f"{counts}"
                )

        sampler = cls.__new__(cls)
        sampler._initialize(
            contract,
            batches,
            body,
            manifest_path=str(manifest_path),
            manifest_file_sha256=_sha256_bytes(raw),
        )
        return sampler

    def __iter__(self) -> Iterator[Sequence[tuple[int, int]]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)
