#!/usr/bin/env python3
"""Production orchestration for the frozen fixed-five representation audit.

The two CLI stages are deliberately separate. ``preflight`` verifies and binds
all immutable inputs without producing representations. ``run`` revalidates that
receipt, extracts the 35 compact caches serially, computes the diagnosis-free
metric artifact, publishes the exact semantic report/receipt, and invokes the
independent verifier. No training operation is available in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tools.matched_cancer_diagnostic_20260730 import runner as diagnostic_runner
from tools.matched_cancer_representation_audit_20260801 import (
    analyzer,
    contract,
    extractor,
    verifier,
)
from tools.matched_cancer_stage_20260730.receipts import file_identity


PREFLIGHT_SCHEMA = "matched-cancer-fixed5-representation-preflight/v1"
METRIC_INPUT_SCHEMA = "matched-cancer-fixed5-representation-metric-input/v1"
RUN_RECEIPT_SCHEMA = verifier.ANALYSIS_RECEIPT_SCHEMA
DEFAULT_OUTPUT_ROOT = Path(
    "/data/ryan.kim/nanopath/reruns/"
    "matched_cancer_representation_audit_20260801/attempt_01"
)
EXPECTED_COMPACT_ROWS = contract.EXPECTED_UNION_PATIENTS * contract.TILES_PER_PATIENT
EXPECTED_FULL_CACHES = len(contract.FM_SEEDS) * len(contract.CANCERS) * 3
EXPECTED_COMPACT_CACHES = len(contract.FM_SEEDS) * len(contract.LAYER_DIMENSIONS)
EXPECTED_PREFLIGHT_TOPOLOGY = {
    "population_patients": contract.EXPECTED_UNION_PATIENTS,
    "union_tss": contract.EXPECTED_UNION_TSS,
    "full_cache_count": EXPECTED_FULL_CACHES,
    "compact_cache_count": EXPECTED_COMPACT_CACHES,
    "rows_per_compact_cache": EXPECTED_COMPACT_ROWS,
    "tile_bundle_count": 1,
    "training_stream_count": len(contract.FM_SEEDS) * len(contract.RUNS),
    "steps_per_training_stream": 781,
}
TRAINING_LOG_FIELDS = (
    "step",
    "dino",
    "jepa",
    "kde",
    "main",
    "fair",
    "total",
    "cancer",
    "race_fair",
    "race_fair_weighted",
    "stage_total",
    "lr",
    "wd",
    "adapter_lr",
    "adapter_weight_decay",
    "grad_norm",
    "stage_adapter_grad_norm",
    "h_dose_main_grad_norm",
    "h_dose_fair_grad_norm",
    "h_dose_fair_main_grad_ratio",
    "h_dose_grad_cosine",
    "h_dose_grad_conflict",
)
ENCODER_REACHABILITY_FIELDS = (
    "encoder_cancer_grad_norm",
    "encoder_fair_raw_grad_norm",
    "encoder_fair_weighted_grad_norm",
    "encoder_stage_grad_finite",
    "encoder_probe_parameter_names_sha256",
    "encoder_probe_parameter_count",
)
VALIDATION_ROW_FIELDS = {
    "step", "val_dino", "val_jepa", "val_kde", "val_cancer", "val_fair",
    "val_total",
}


class PipelineError(RuntimeError):
    """A fail-closed production topology, identity, or output error."""


@dataclass
class ValidatedInputs:
    population: tuple[dict[str, str], ...]
    references: dict[str, extractor.FullCache]
    selections: dict[str, tuple[dict[str, Any], ...]]
    full_caches: dict[tuple[int, str, str], extractor.FullCache]
    source_identities: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PipelineError(f"value is not canonical JSON: {error}") from error


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise PipelineError(f"refusing to overwrite {destination}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(_canonical_bytes(value))
        output.flush()
        os.fsync(output.fileno())
    return destination.resolve()


def _strict_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PipelineError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def nonfinite(value: str) -> None:
        raise PipelineError(f"{path}: non-finite JSON constant {value!r}")

    try:
        return json.loads(
            Path(path).read_text(),
            object_pairs_hook=unique,
            parse_constant=nonfinite,
        )
    except PipelineError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read strict JSON {path}: {error}") from error


def _unique_pairs(pairs: list[tuple[str, Any]], context: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PipelineError(f"{context}: duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _implementation_identities() -> dict[str, Any]:
    repo = contract.REPO
    package = Path(__file__).parent
    import scipy
    import sklearn
    import torch

    return {
        "contract": file_identity(Path(contract.__file__)),
        "extractor": file_identity(Path(extractor.__file__)),
        "analyzer": file_identity(Path(analyzer.__file__)),
        "verifier": file_identity(Path(verifier.__file__)),
        "pipeline": file_identity(Path(__file__)),
        "serial_driver": file_identity(package / "serial_audit.sbatch"),
        "safe_submit": file_identity(package / "safe_submit.sh"),
        "diagnostic_runner": file_identity(Path(diagnostic_runner.__file__)),
        "diagnostic_cache": file_identity(
            repo / "tools/matched_cancer_diagnostic_20260730/cache.py"
        ),
        "completion_receipt": file_identity(
            repo / "tools/matched_cancer_stage_20260730/completion_receipt.py"
        ),
        "receipts": file_identity(
            repo / "tools/matched_cancer_stage_20260730/receipts.py"
        ),
        "reliable_fairness_head": file_identity(
            repo / "tools/reliable_fairness_head.py"
        ),
        "fairness_eval": file_identity(repo / "tools/fairness_eval.py"),
        "encoder_model": file_identity(
            repo / "vendor/matched_stage_train_20260730/model.py"
        ),
        "stage_objectives": file_identity(
            repo / "tools/matched_cancer_stage_20260730/objectives.py"
        ),
        "python_target": file_identity(Path(sys.executable).resolve()),
        "python_venv_config": file_identity(
            Path("/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg")
        ),
        "runtime_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }


def _selection_rows(inputs: ValidatedInputs) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for cancer in contract.CANCERS:
        rows.extend(inputs.selections[cancer])
    if len(rows) != EXPECTED_COMPACT_ROWS:
        raise PipelineError("union selection must contain exactly 19,488 rows")
    return tuple(rows)


def _validate_production_inputs() -> ValidatedInputs:
    population = tuple(dict(row) for row in contract.load_sanitized_population())
    contract.validate_frozen_population(population)

    full: dict[tuple[int, str, str], extractor.FullCache] = {}
    references: dict[str, extractor.FullCache] = {}
    seed_states: dict[str, Any] = {}
    final_identities: dict[str, Any] = {}
    for seed in contract.FM_SEEDS:
        encoder_hashes, adapter_hashes = extractor.seed_state_hashes(seed)
        seed_states[str(seed)] = {
            "encoders": encoder_hashes,
            "adapters": adapter_hashes,
        }
        for cancer in contract.CANCERS:
            for layer in ("B", "P", "H"):
                cache = extractor.read_full_cache(
                    extractor.discover_final_cache(seed, cancer, layer)
                )
                extractor.validate_final_cache_provenance(
                    cache, seed=seed, cancer=cancer, layer=layer
                )
                reference = references.setdefault(cancer, cache)
                extractor.assert_same_tile_evidence(reference, cache)
                full[(seed, cancer, layer)] = cache
                final_identities[f"{seed}|{cancer}|{layer}"] = file_identity(
                    cache.path
                )
    if len(full) != EXPECTED_FULL_CACHES:
        raise PipelineError("full-cache topology must contain exactly 30 caches")

    selections = {
        cancer: extractor.build_selection_rows(
            references[cancer], population, cancer=cancer
        )
        for cancer in contract.CANCERS
    }
    combined = tuple(
        row for cancer in contract.CANCERS for row in selections[cancer]
    )
    if len(combined) != EXPECTED_COMPACT_ROWS:
        raise PipelineError("selected tile topology differs from 609 x 32")
    shared_views = {
        view: tuple(row for row in combined if row["view"] == view)
        for view in contract.TILE_VIEWS
    }
    contract.validate_shared_tile_views(
        {
            (seed, layer): shared_views
            for seed in contract.FM_SEEDS
            for layer in contract.LAYER_DIMENSIONS
        }
    )

    calibration: dict[str, Any] = {}
    for seed in contract.FM_SEEDS:
        paths = contract.production_paths(seed)
        calibration[str(seed)] = {
            "replay_manifest": file_identity(paths["replay_manifest"]),
            "checkpoints": {
                layer: file_identity(path)
                for layer, path in paths["checkpoints"].items()
            },
            "completion_receipts": {
                run: file_identity(path)
                for run, path in paths["completion_receipts"].items()
            },
            "metrics": {
                run: file_identity(paths["root"] / run / "metrics.jsonl")
                for run in contract.RUNS
            },
            "summaries": {
                run: file_identity(paths["root"] / run / "summary.json")
                for run in contract.RUNS
            },
        }
    sources = {
        "lock": file_identity(contract.LOCK_PATH),
        "numeric_amendment": file_identity(contract.NUMERIC_AMENDMENT_PATH),
        "metadata": {
            cancer: file_identity(path)
            for cancer, path in contract.METADATA_PATHS.items()
        },
        "representation_exclusion": file_identity(
            contract.REPRESENTATION_EXCLUSION_PATH
        ),
        "tile_view_receipt": file_identity(extractor.TILE_VIEW_RECEIPT),
        "calibration": calibration,
        "final_full_caches": final_identities,
        "implementation": _implementation_identities(),
        "state_hashes": seed_states,
    }
    return ValidatedInputs(
        population=population,
        references=references,
        selections=selections,
        full_caches=full,
        source_identities=sources,
    )


def _identity_nodes(value: Any, path: str = "sources") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping) and set(value) == {
        "canonical_path", "bytes", "sha256"
    }:
        yield path, value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _identity_nodes(nested, f"{path}.{key}")


def _revalidate_identity_tree(sources: Mapping[str, Any]) -> None:
    found = 0
    for role, identity in _identity_nodes(sources):
        found += 1
        path = Path(str(identity.get("canonical_path", "")))
        if file_identity(path) != dict(identity):
            raise PipelineError(f"bound source identity changed: {role}")
    if found == 0:
        raise PipelineError("preflight receipt binds no source files")


def preflight(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, launch_nonce: str
) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", launch_nonce) is None:
        raise PipelineError("launch nonce must contain exactly 128 lowercase-hex bits")
    root = Path(output_root)
    receipt_path = root / "PREFLIGHT_RECEIPT.json"
    inputs = _validate_production_inputs()
    selection = _selection_rows(inputs)
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "study_id": contract.STUDY_ID,
        "status": "pass",
        "diagnosis_free": True,
        "launch_nonce": launch_nonce,
        "expected": EXPECTED_PREFLIGHT_TOPOLOGY,
        "population_sha256": _sha256_json(inputs.population),
        "selection_sha256": _sha256_json(selection),
        "sources": inputs.source_identities,
    }
    return _exclusive_json(receipt_path, receipt)


def _load_preflight(output_root: Path) -> dict[str, Any]:
    path = Path(output_root) / "PREFLIGHT_RECEIPT.json"
    value = _strict_json(path)
    required = {
        "schema", "study_id", "status", "diagnosis_free", "expected",
        "launch_nonce", "population_sha256", "selection_sha256", "sources",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PipelineError("preflight receipt schema keys differ")
    if (
        value["schema"] != PREFLIGHT_SCHEMA
        or value["study_id"] != contract.STUDY_ID
        or value["status"] != "pass"
        or value["diagnosis_free"] is not True
    ):
        raise PipelineError("preflight receipt header drift")
    if re.fullmatch(r"[0-9a-f]{32}", str(value["launch_nonce"])) is None:
        raise PipelineError("preflight launch nonce drift")
    expected = value["expected"]
    if expected != EXPECTED_PREFLIGHT_TOPOLOGY:
        raise PipelineError("preflight expected topology drift")
    _revalidate_identity_tree(value["sources"])
    return dict(value)


def _require_new(path: Path) -> None:
    if Path(path).exists() or Path(path).is_symlink():
        raise PipelineError(f"refusing to overwrite existing artifact {path}")


def _compact_path(root: Path, seed: int, layer: str) -> Path:
    return root / "compact" / f"seed_{seed}" / f"{layer}.npz"


def _write_and_validate_compact(
    path: Path,
    *,
    seed: int,
    layer: str,
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
    population: Sequence[Mapping[str, str]],
) -> None:
    _require_new(path)
    extractor.write_compact_cache(
        path,
        seed=seed,
        layer=layer,
        embeddings=embeddings,
        rows=rows,
        source_identity=source_identity,
    )
    metadata, observed, observed_rows = extractor.read_compact_cache(path)
    extractor.validate_compact_topology(
        metadata, observed, observed_rows, population
    )


def _release_representation(representation: Any) -> None:
    del representation
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _extract_all(
    root: Path,
    inputs: ValidatedInputs,
    *,
    device: str,
    batch_size: int,
) -> dict[tuple[int, str], Path]:
    rows = _selection_rows(inputs)
    bundle_dir = root / "tile_bundle"
    _require_new(bundle_dir)
    bundle_receipt = extractor.materialize_tile_bundle(
        bundle_dir,
        references=inputs.references,
        selections=inputs.selections,
    )
    bundle_rows, tiles = extractor.load_tile_bundle(bundle_receipt)
    if tuple(bundle_rows) != rows or len(tiles) != EXPECTED_COMPACT_ROWS:
        raise PipelineError("materialized tile bundle differs from frozen selection")

    catalog: dict[tuple[int, str], Path] = {}
    # Reuse and reduce all 30 validated B/P/H caches into 15 union caches.
    for seed in contract.FM_SEEDS:
        paths = contract.production_paths(seed)
        for layer in ("B", "P", "H"):
            values = np.concatenate(
                [
                    extractor.subset_final_embeddings(
                        inputs.full_caches[(seed, cancer, layer)],
                        inputs.selections[cancer],
                    )
                    for cancer in contract.CANCERS
                ],
                axis=0,
            )
            output = _compact_path(root, seed, layer)
            source = {
                "completion_receipt": file_identity(
                    paths["completion_receipts"][layer]
                ),
                "checkpoint": file_identity(paths["checkpoints"][layer]),
                "full_caches": {
                    cancer: file_identity(
                        inputs.full_caches[(seed, cancer, layer)].path
                    )
                    for cancer in contract.CANCERS
                },
                "tile_bundle_receipt": file_identity(bundle_receipt),
            }
            _write_and_validate_compact(
                output,
                seed=seed,
                layer=layer,
                embeddings=values,
                rows=rows,
                source_identity=source,
                population=inputs.population,
            )
            catalog[(seed, layer)] = output

        # Each slot-1 checkpoint is loaded once and emits E plus its temporary A.
        for run_name, encoder_layer, adapter_layer in (
            ("slot1_plain", "E_plain", "A_temp_plain"),
            ("slot1_fair", "E_fair", "A_temp_fair"),
        ):
            receipt_path = paths["completion_receipts"][run_name]
            representation = diagnostic_runner.load_frozen_representation(
                receipt_path, device=device
            )
            encoded, adapted = extractor.embed_encoder_and_adapter(
                representation, tiles, batch_size=batch_size
            )
            representation.assert_unchanged()
            source = {
                "completion_receipt": file_identity(receipt_path),
                "checkpoint": file_identity(paths["checkpoints"][encoder_layer]),
                "tile_bundle_receipt": file_identity(bundle_receipt),
                "diagnostic_runner": file_identity(
                    Path(diagnostic_runner.__file__)
                ),
            }
            for layer, values in (
                (encoder_layer, encoded),
                (adapter_layer, adapted),
            ):
                output = _compact_path(root, seed, layer)
                _write_and_validate_compact(
                    output,
                    seed=seed,
                    layer=layer,
                    embeddings=values,
                    rows=rows,
                    source_identity=source,
                    population=inputs.population,
                )
                catalog[(seed, layer)] = output
            representation.assert_unchanged()
            representation = None
            _release_representation(representation)
    if set(catalog) != {
        (seed, layer)
        for seed in contract.FM_SEEDS
        for layer in contract.LAYER_DIMENSIONS
    }:
        raise PipelineError("compact cache catalog differs from exact 5 x 7 topology")
    return catalog


def _records_for(
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
    cancer: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row["view"] != view or (cancer is not None and row["cancer"] != cancer):
            continue
        records.append(
            {
                "metadata": {
                    field: str(row[field])
                    for field in contract.METADATA_ALLOWLIST
                },
                "tile_id": (
                    f"{row['payload_sha256']}:{int(row['occurrence_index'])}"
                ),
                "embedding": embeddings[index],
            }
        )
    expected_patients = (
        contract.EXPECTED_POPULATION[cancer]["patients"]
        if cancer is not None
        else contract.EXPECTED_UNION_PATIENTS
    )
    if len(records) != expected_patients * contract.TILES_PER_VIEW:
        raise PipelineError("analyzer record cardinality drift")
    return records


def _faircon_batch_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise PipelineError("FairCon support reconstruction requires one batch")
    counts: dict[tuple[int, int], int] = {}
    for row in rows:
        cancer = int(row["cancer"])
        race = int(row["race"])
        counts[(cancer, race)] = counts.get((cancer, race), 0) + 1
    # Training repeats metadata for two global views before FairCon.
    doubled = {key: 2 * count for key, count in counts.items()}
    anchors = 2 * len(rows)
    eligible = 0
    positives = 0
    for (cancer, race), count in doubled.items():
        opposite = sum(
            value
            for (other_cancer, other_race), value in doubled.items()
            if other_cancer == cancer and other_race != race
        )
        if opposite > 0:
            eligible += count
            positives += count * opposite
    return {
        "global_view_repetitions": 2,
        "anchor_count": anchors,
        "eligible_anchor_count": eligible,
        "omitted_anchor_without_positive_count": anchors - eligible,
        "denominator_candidates_per_anchor": anchors - 1,
        "denominator_ordered_pair_count": anchors * (anchors - 1),
        "positive_ordered_pair_count": positives,
        "batch_counts_by_cancer_race": {
            f"{cancer}|{race}": count
            for (cancer, race), count in sorted(counts.items())
        },
    }


def _replay_support(path: Path) -> list[dict[str, Any]]:
    manifest = _strict_json(path)
    occurrences = manifest.get("occurrences") if isinstance(manifest, Mapping) else None
    if not isinstance(occurrences, list) or len(occurrences) != 781 * 128:
        raise PipelineError("replay manifest must contain exactly 781 x 128 occurrences")
    grouped: list[list[Mapping[str, Any]]] = [[] for _ in range(781)]
    for row in occurrences:
        if not isinstance(row, Mapping):
            raise PipelineError("replay occurrence must be an object")
        batch = row.get("batch")
        if isinstance(batch, bool) or not isinstance(batch, int) or not 0 <= batch < 781:
            raise PipelineError("replay occurrence batch index drift")
        grouped[batch].append(row)
    if any(len(batch) != 128 for batch in grouped):
        raise PipelineError("replay batch cardinality drift")
    return [
        {"step": index + 1, **_faircon_batch_support(batch)}
        for index, batch in enumerate(grouped)
    ]


def _training_streams() -> list[dict[str, Any]]:
    streams: list[dict[str, Any]] = []
    for seed in contract.FM_SEEDS:
        paths = contract.production_paths(seed)
        support = _replay_support(paths["replay_manifest"])
        for run in contract.RUNS:
            metrics_path = paths["root"] / run / "metrics.jsonl"
            raw_lines = [line for line in metrics_path.read_text().splitlines() if line]
            parsed: list[Mapping[str, Any]] = []
            for line in raw_lines:
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=lambda pairs: _unique_pairs(
                            pairs, f"{seed}/{run} metrics row"
                        ),
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            PipelineError(
                                f"{seed}/{run}: non-finite JSON constant {value!r}"
                            )
                        ),
                    )
                except json.JSONDecodeError as error:
                    raise PipelineError(f"{seed}/{run}: malformed metrics JSON") from error
                if not isinstance(value, Mapping):
                    raise PipelineError(f"{seed}/{run}: metrics row must be an object")
                parsed.append(value)
            training_rows = [row for row in parsed if "main" in row]
            validation_rows = [row for row in parsed if "main" not in row]
            if len(training_rows) != 781 or len(validation_rows) != 1:
                raise PipelineError(
                    f"{seed}/{run}: metrics require 781 training rows and one validation row"
                )
            validation = validation_rows[0]
            if set(validation) != VALIDATION_ROW_FIELDS or validation["step"] != 781:
                raise PipelineError(f"{seed}/{run}: validation-row topology drift")
            for field in VALIDATION_ROW_FIELDS - {"step"}:
                value = validation[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise PipelineError(f"{seed}/{run}/{field}: invalid validation value")
            selected: list[dict[str, Any]] = []
            for expected_step, row in enumerate(training_rows, start=1):
                if set(TRAINING_LOG_FIELDS) - set(row):
                    raise PipelineError(f"{seed}/{run}: selected training fields missing")
                item = {field: row[field] for field in TRAINING_LOG_FIELDS}
                if item["step"] != expected_step or type(item["h_dose_grad_conflict"]) is not bool:
                    raise PipelineError(f"{seed}/{run}: training step/type drift")
                for field in TRAINING_LOG_FIELDS:
                    if field in {"step", "h_dose_grad_conflict"}:
                        continue
                    value = item[field]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise PipelineError(f"{seed}/{run}/{field}: nonfinite training value")
                item["faircon_denominator_support"] = support[expected_step - 1]
                selected.append(item)
            reachability: dict[str, Any] | None = None
            if run in {"slot1_plain", "slot1_fair"}:
                first = training_rows[0]
                if set(ENCODER_REACHABILITY_FIELDS) - set(first):
                    raise PipelineError(f"{seed}/{run}: encoder reachability fields missing")
                reachability = {
                    field: first[field] for field in ENCODER_REACHABILITY_FIELDS
                }
                if type(reachability["encoder_stage_grad_finite"]) is not bool or reachability["encoder_stage_grad_finite"] is not True:
                    raise PipelineError(f"{seed}/{run}: encoder reachability is not finite")
                if not isinstance(reachability["encoder_probe_parameter_count"], int) or reachability["encoder_probe_parameter_count"] <= 0:
                    raise PipelineError(f"{seed}/{run}: encoder probe parameter count drift")
                names_hash = reachability["encoder_probe_parameter_names_sha256"]
                if not isinstance(names_hash, str) or re.fullmatch(r"[0-9a-f]{64}", names_hash) is None:
                    raise PipelineError(f"{seed}/{run}: encoder probe-name hash drift")
                for field in ENCODER_REACHABILITY_FIELDS[:3]:
                    value = reachability[field]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        raise PipelineError(f"{seed}/{run}/{field}: invalid reachability value")
            streams.append(
                {
                    "fm_seed": seed,
                    "run": run,
                    "metrics_identity": file_identity(metrics_path),
                    "replay_manifest_identity": file_identity(paths["replay_manifest"]),
                    "validation_row": dict(validation),
                    "encoder_reachability_first_batch_only": reachability,
                    "encoder_reachability_is_not_cumulative_dose": True,
                    "steps": selected,
                }
            )
    expected_streams = len(contract.FM_SEEDS) * len(contract.RUNS)
    if len(streams) != expected_streams:
        raise PipelineError(
            f"training evidence must contain exactly {expected_streams} streams"
        )
    return streams


def _load_seed_layer_states(seed: int) -> dict[str, dict[str, Any]]:
    import torch

    paths = contract.production_paths(seed)
    layers_by_run = {
        "slot1_plain": ("E_plain", "A_temp_plain"),
        "slot1_fair": ("E_fair", "A_temp_fair"),
        "B": ("B",),
        "P": ("P",),
        "H": ("H",),
    }
    result: dict[str, dict[str, Any]] = {}
    for run, layers in layers_by_run.items():
        checkpoint = torch.load(
            paths["root"] / run / "latest.pt",
            map_location="cpu",
            weights_only=True,
        )
        for layer in layers:
            member = "model" if layer.startswith("E_") else "stage_adapter"
            state = checkpoint.get(member)
            if not isinstance(state, Mapping):
                raise PipelineError(f"{seed}/{layer}: checkpoint lacks {member}")
            result[layer] = {
                name: value.detach().cpu() for name, value in state.items()
            }
        del checkpoint
    if set(result) != set(contract.LAYER_DIMENSIONS):
        raise PipelineError(f"{seed}: parameter-state layer topology drift")
    return result


def _parameter_displacements() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contrasts = (*contract.GATE_ELIGIBLE_CONTRASTS, *contract.DESCRIPTIVE_CONTRASTS)
    for seed in contract.FM_SEEDS:
        state_cache = _load_seed_layer_states(seed)
        for candidate, baseline in contrasts:
            contract.validate_equal_dimension(candidate, baseline)
            metric = analyzer.parameter_displacement(
                state_cache[baseline],
                state_cache[candidate],
            )
            rows.append(
                {
                    "fm_seed": seed,
                    "candidate": candidate,
                    "baseline": baseline,
                    "comparison": "candidate_vs_matched_baseline_pairwise",
                    "gate_eligible": (candidate, baseline) in contract.GATE_ELIGIBLE_CONTRASTS,
                    "result": metric,
                }
            )
        del state_cache
    return rows


def _validate_cuda_allocation(preflight_value: Mapping[str, Any]) -> None:
    if os.environ.get("SLURM_JOB_NAME") != contract.CONTINUATION_LIMITS["job_name"]:
        raise PipelineError("CUDA run must execute in Slurm job main_1gpu")
    if not os.environ.get("SLURM_JOB_ID"):
        raise PipelineError("CUDA run requires a Slurm job id")
    if os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_ARRAY_TASK_ID"):
        raise PipelineError("job arrays are forbidden")
    if os.environ.get("SLURM_NTASKS", "1") != "1":
        raise PipelineError("audit allocation must have exactly one task")
    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise PipelineError("audit allocation must expose exactly one CUDA device")
    if os.environ.get("REP_AUDIT_LAUNCH_NONCE") != preflight_value["launch_nonce"]:
        raise PipelineError("runtime launch nonce differs from preflight")
    if os.environ.get("REP_AUDIT_SLURM_COMMENT") != "matched_cancer_rep_audit_20260801":
        raise PipelineError("runtime Slurm comment drift")
    import torch

    if torch.cuda.device_count() != 1:
        raise PipelineError("runtime must observe exactly one CUDA device")


def _metric_input(
    catalog: Mapping[tuple[int, str], Path],
    population: Sequence[Mapping[str, str]],
    preflight_identity: Mapping[str, Any],
) -> dict[str, Any]:
    race_probes: list[dict[str, Any]] = []
    cancer_probes: list[dict[str, Any]] = []
    energy: list[dict[str, Any]] = []
    mixing: list[dict[str, Any]] = []
    cache_data: dict[tuple[int, str], tuple[np.ndarray, tuple[dict[str, Any], ...]]] = {}
    for seed in contract.FM_SEEDS:
        for layer in contract.LAYER_DIMENSIONS:
            metadata, embeddings, rows = extractor.read_compact_cache(catalog[(seed, layer)])
            extractor.validate_compact_topology(metadata, embeddings, rows, population)
            cache_data[(seed, layer)] = (embeddings, rows)
            for view in contract.TILE_VIEWS:
                union_records = _records_for(embeddings, rows, view=view)
                cancer_result = analyzer.pooled_cancer_probe(union_records)
                cancer_probes.append(
                    {"fm_seed": seed, "layer": layer, "view": view, "result": cancer_result}
                )
                energy.append(
                    {
                        "fm_seed": seed,
                        "layer": layer,
                        "view": view,
                        "result": analyzer.cancer_conditioned_energy_distance(union_records),
                    }
                )
                for cancer in contract.CANCERS:
                    records = _records_for(embeddings, rows, view=view, cancer=cancer)
                    for level in contract.PROBE_LEVELS:
                        result = analyzer.nested_race_probe(records, level=level)
                        race_probes.append(
                            {
                                "fm_seed": seed,
                                "layer": layer,
                                "cancer": cancer,
                                "view": view,
                                "probe_level": level,
                                "result": result,
                            }
                        )
                        mixing.append(
                            {
                                "fm_seed": seed,
                                "layer": layer,
                                "cancer": cancer,
                                "view": view,
                                "probe_level": level,
                                "result": analyzer.cosine_knn_cross_race_mixing(
                                    records, level=level
                                ),
                            }
                        )

    displacements: list[dict[str, Any]] = []
    contrasts = (*contract.GATE_ELIGIBLE_CONTRASTS, *contract.DESCRIPTIVE_CONTRASTS)
    for seed in contract.FM_SEEDS:
        for candidate, baseline in contrasts:
            contract.validate_equal_dimension(candidate, baseline)
            base_emb, base_rows = cache_data[(seed, baseline)]
            candidate_emb, candidate_rows = cache_data[(seed, candidate)]
            if base_rows != candidate_rows:
                raise PipelineError("aligned compact row topology differs")
            for cancer in contract.CANCERS:
                for view in contract.TILE_VIEWS:
                    base_records = _records_for(base_emb, base_rows, view=view, cancer=cancer)
                    candidate_records = _records_for(candidate_emb, candidate_rows, view=view, cancer=cancer)
                    for level in contract.PROBE_LEVELS:
                        displacements.append(
                            {
                                "fm_seed": seed,
                                "candidate": candidate,
                                "baseline": baseline,
                                "gate_eligible": (candidate, baseline) in contract.GATE_ELIGIBLE_CONTRASTS,
                                "cancer": cancer,
                                "view": view,
                                "level": level,
                                "result": analyzer.aligned_representation_displacement(
                                    base_records, candidate_records, level=level
                                ),
                            }
                        )

    value = {
        "schema": METRIC_INPUT_SCHEMA,
        "study_id": contract.STUDY_ID,
        "status": "complete",
        "diagnosis_free": True,
        "fm_seeds": list(contract.FM_SEEDS),
        "layers": list(contract.LAYER_DIMENSIONS),
        "views": list(contract.TILE_VIEWS),
        "probe_levels": list(contract.PROBE_LEVELS),
        "compact_caches": {
            f"{seed}|{layer}": file_identity(path)
            for (seed, layer), path in sorted(catalog.items())
        },
        "preflight_receipt": dict(preflight_identity),
        "race_probes": race_probes,
        "cancer_probes": cancer_probes,
        "secondary_geometry": {
            "energy_distance": energy,
            "cosine_knn_mixing": mixing,
            "aligned_representation_displacement": displacements,
            "pairwise_parameter_displacement": _parameter_displacements(),
        },
        "training_evidence": _training_streams(),
    }
    analyzer.reject_diagnosis_fields(value)
    return value


def _semantic_contrast_inputs(metric_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    race_index: dict[tuple[str, int, str, str, str], float] = {}
    for row in metric_input["race_probes"]:
        key = (
            row["layer"], row["fm_seed"], row["cancer"],
            row["view"], row["probe_level"],
        )
        if key in race_index:
            raise PipelineError(f"duplicate race-probe cell {key!r}")
        race_index[key] = float(row["result"]["oriented_leakage"])
    cancer_index: dict[tuple[str, int, str], float] = {}
    for row in metric_input["cancer_probes"]:
        key = (row["layer"], row["fm_seed"], row["view"])
        if key in cancer_index:
            raise PipelineError(f"duplicate cancer-probe cell {key!r}")
        cancer_index[key] = float(row["result"]["pooled_heldout_patient_auroc"])
    expected_race = len(contract.LAYER_DIMENSIONS) * 40
    expected_cancer = len(contract.LAYER_DIMENSIONS) * 10
    if len(race_index) != expected_race or len(cancer_index) != expected_cancer:
        raise PipelineError("probe metric topology is incomplete")

    contrasts: list[dict[str, Any]] = []
    for candidate, baseline in contract.GATE_ELIGIBLE_CONTRASTS:
        race_cells = [
            {
                "fm_seed": seed,
                "cancer": cancer,
                "view": view,
                "probe_level": level,
                "baseline_oriented_leakage": race_index[(baseline, seed, cancer, view, level)],
                "candidate_oriented_leakage": race_index[(candidate, seed, cancer, view, level)],
            }
            for seed in contract.FM_SEEDS
            for cancer in contract.CANCERS
            for view in contract.TILE_VIEWS
            for level in contract.PROBE_LEVELS
        ]
        cancer_cells = [
            {
                "fm_seed": seed,
                "view": view,
                "baseline_auroc": cancer_index[(baseline, seed, view)],
                "candidate_auroc": cancer_index[(candidate, seed, view)],
            }
            for seed in contract.FM_SEEDS
            for view in contract.TILE_VIEWS
        ]
        contrasts.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "race_cells": race_cells,
                "cancer_cells": cancer_cells,
            }
        )
    return contrasts


def run(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    device: str = "cuda",
    batch_size: int = 128,
) -> dict[str, Path]:
    root = Path(output_root)
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise PipelineError("batch size must be positive")
    preflight_value = _load_preflight(root)
    if str(device).startswith("cuda"):
        _validate_cuda_allocation(preflight_value)
    preflight_path = root / "PREFLIGHT_RECEIPT.json"
    inputs = _validate_production_inputs()
    if inputs.source_identities != preflight_value["sources"]:
        raise PipelineError("current validated sources differ from preflight")
    if _sha256_json(inputs.population) != preflight_value["population_sha256"]:
        raise PipelineError("population differs from preflight")
    if _sha256_json(_selection_rows(inputs)) != preflight_value["selection_sha256"]:
        raise PipelineError("selected tile identities differ from preflight")

    final_paths = {
        "metric_input": root / "metric_input.json",
        "analysis_report": root / "analysis.json",
        "analysis_receipt": root / "analysis.receipt.json",
        "verification": root / "independent_verification.json",
    }
    for path in final_paths.values():
        _require_new(path)

    catalog = _extract_all(
        root, inputs, device=device, batch_size=int(batch_size)
    )
    population = inputs.population
    # Full production caches are intentionally retained only through extraction.
    # Keeping all 30 arrays resident during the multi-hour probe analysis would
    # consume more than 12 GB for evidence that has already been compacted.
    del inputs
    gc.collect()
    metric_value = _metric_input(
        catalog, population, file_identity(preflight_path)
    )
    metric_path = _exclusive_json(final_paths["metric_input"], metric_value)
    identities = {
        "metric_input": file_identity(metric_path),
        "lock": file_identity(contract.LOCK_PATH),
        "numeric_amendment": file_identity(contract.NUMERIC_AMENDMENT_PATH),
        "analyzer": file_identity(Path(analyzer.__file__)),
    }
    report = analyzer.semantic_report(
        contrasts=_semantic_contrast_inputs(metric_value),
        identities=identities,
    )
    report_path = _exclusive_json(final_paths["analysis_report"], report)
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "study_id": contract.STUDY_ID,
        "status": "complete",
        "analysis_report": file_identity(report_path),
        "identities": identities,
    }
    receipt_path = _exclusive_json(final_paths["analysis_receipt"], receipt)

    # Revalidate the preflight receipt immediately before independent verification.
    if _load_preflight(root) != preflight_value:
        raise PipelineError("preflight receipt changed during the run")
    verification = verifier.verify_analysis_files(
        report_path,
        receipt_path,
        metric_path,
        lock=contract.LOCK_PATH,
        numeric_amendment=contract.NUMERIC_AMENDMENT_PATH,
        analyzer_source=Path(analyzer.__file__),
    )
    verifier.write_json_exclusive(final_paths["verification"], verification)
    return {
        "preflight": preflight_path.resolve(),
        "metric_input": metric_path,
        "analysis_report": report_path,
        "analysis_receipt": receipt_path,
        "verification": final_paths["verification"].resolve(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    preflight_parser.add_argument("--launch-nonce", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--batch-size", type=int, default=128)
    arguments = parser.parse_args(argv)
    try:
        if arguments.stage == "preflight":
            paths = {
                "preflight": preflight(
                    arguments.output_root, launch_nonce=arguments.launch_nonce
                )
            }
        else:
            paths = run(
                arguments.output_root,
                device=arguments.device,
                batch_size=arguments.batch_size,
            )
    except (
        PipelineError,
        extractor.ExtractionError,
        analyzer.AnalysisError,
        verifier.VerificationError,
        OSError,
        ValueError,
    ) as error:
        sys.stderr.write(f"representation audit {arguments.stage} failed: {error}\n")
        return 1
    sys.stdout.write(
        json.dumps(
            {
                "study_id": contract.STUDY_ID,
                "stage": arguments.stage,
                "status": "pass",
                "paths": {key: str(path) for key, path in paths.items()},
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
