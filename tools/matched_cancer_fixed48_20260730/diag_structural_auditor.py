"""Systems-only fixed48 diagnostic auditor.

This module validates topology, ancestry, cache structure, and finite row types.
It deliberately contains no metric, threshold, analyzer, or inference code.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools import reliable_fairness_head as reliable
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_contract import (
    ARMS,
    AUDIT_SCHEMA,
    CANCERS,
    COHORT_SIZES,
    HEAD_SEEDS,
    STUDY_ID,
    TASK_IDS,
    validate_seed,
)
from .diag_deployment import RUNTIME_SOURCE_PATHS, verify_gate
from .diag_exporter import verify_collection
from .diag_loader import verify_loader_result


CELL_SCHEMA = "matched-cancer-adapter-diagnostic/v1"
ROOT_SCHEMA = "matched-cancer-adapter-diagnostic-root/v1"
CACHE_SCHEMA = "matched-cancer-adapter-cache/v1"
COMPLETION_SCHEMA = "matched-cancer-stage-completion/v1"
FORBIDDEN_RUNTIME_MODULE_FRAGMENTS = ("analyzer", "verifier")
FORBIDDEN_REPRESENTATION_KEYS = {
    "y_true", "outcome", "diagnosis", "tp53", "molecular", "label",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _assert_no_analysis_imports() -> int:
    checked = 0
    for name in (
        "diag_deployment", "diag_loader", "diag_exporter",
        "diag_structural_auditor", "diag_worker",
    ):
        path = Path(__file__).with_name(f"{name}.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if any(
                    fragment in module
                    for fragment in FORBIDDEN_RUNTIME_MODULE_FRAGMENTS
                ):
                    raise ValueError(
                        f"production source imports analysis module: {module}"
                    )
        checked += 1
    return checked


def _verify_cache(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    if file_identity(path) != dict(identity):
        raise ValueError("adapter cache identity drift")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "emb", "barcodes", "keep_mask", "input_barcodes",
            "payload_sha256", "payload_bytes", "metadata_json",
            "entry_sha256",
        }:
            raise ValueError("adapter cache archive topology differs")
        embeddings = np.asarray(archive["emb"])
        barcodes = np.asarray(archive["barcodes"])
        keep_mask = np.asarray(archive["keep_mask"])
        input_barcodes = np.asarray(archive["input_barcodes"])
        payload_sha256 = np.asarray(archive["payload_sha256"])
        payload_bytes = np.asarray(archive["payload_bytes"])
        metadata_text = str(np.asarray(archive["metadata_json"]).item())
        entry_sha = str(np.asarray(archive["entry_sha256"]).item())
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != 128
        or embeddings.shape[0] != len(barcodes)
        or not np.issubdtype(embeddings.dtype, np.floating)
        or not np.isfinite(embeddings).all()
        or len(entry_sha) != 64
    ):
        raise ValueError("adapter cache shape/type/hash differs")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=2e-5):
        raise ValueError("adapter cache is not per-tile L2-normalized")
    metadata = json.loads(metadata_text)
    if set(metadata) != {"schema", "cache_key", "source_identity"} or (
        metadata.get("schema") != CACHE_SCHEMA
    ):
        raise ValueError("adapter cache schema differs")
    source = metadata.get("source_identity")
    if isinstance(source, Mapping):
        key_tokens = {str(key).lower() for key in _walk_keys(source)}
        if key_tokens & FORBIDDEN_REPRESENTATION_KEYS:
            raise ValueError(
                "downstream outcome key entered representation cache"
            )
    expected_source_keys = {
        "completion_receipt", "checkpoint", "encoder_state_sha256",
        "adapter_state_sha256", "sources", "ordered_tiles_sha256",
        "tile_count", "tag_sha256", "normalization", "embedding_dim",
    }
    if (
        not isinstance(source, Mapping)
        or set(source) != expected_source_keys
        or source.get("normalization") != "per_tile_l2"
        or source.get("embedding_dim") != 128
        or isinstance(source.get("tile_count"), bool)
        or not isinstance(source.get("tile_count"), int)
        or source["tile_count"] <= 0
        or not isinstance(source.get("sources"), Mapping)
    ):
        raise ValueError("adapter cache representation contract differs")
    tile_count = source["tile_count"]
    if (
        keep_mask.shape != (tile_count,)
        or keep_mask.dtype != np.bool_
        or input_barcodes.shape != (tile_count,)
        or payload_sha256.shape != (tile_count,)
        or payload_bytes.shape != (tile_count,)
        or int(keep_mask.sum()) != embeddings.shape[0]
        or not np.array_equal(barcodes, input_barcodes[keep_mask])
        or any(
            len(str(value)) != 64
            or any(character not in "0123456789abcdef" for character in str(value))
            for value in payload_sha256
        )
        or payload_bytes.dtype != np.int64
        or np.any(payload_bytes < 0)
    ):
        raise ValueError("adapter cache ordered evidence differs")
    reproduced_ordered = reliable._ordered_digest_from_evidence(
        np.asarray([str(value) for value in input_barcodes], dtype=np.str_),
        np.asarray([str(value) for value in payload_sha256], dtype=np.str_),
        payload_bytes,
    )
    if reproduced_ordered != source["ordered_tiles_sha256"]:
        raise ValueError("adapter cache ordered tile digest differs")
    computed_cache_key = hashlib.sha256(
        (
            CACHE_SCHEMA + "\0"
            + json.dumps(
                source,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ).encode()
    ).hexdigest()
    computed_entry_sha = reliable._entry_sha256(
        metadata,
        embeddings,
        barcodes,
        keep_mask,
        input_barcodes,
        payload_sha256,
        payload_bytes,
    )
    if (
        metadata.get("cache_key") != computed_cache_key
        or entry_sha != computed_entry_sha
    ):
        raise ValueError("adapter cache content-addressed hash differs")
    return {
        "rows": int(embeddings.shape[0]),
        "columns": int(embeddings.shape[1]),
        "entry_sha256": entry_sha,
        "ordered_tiles_sha256": source["ordered_tiles_sha256"],
        "input_tile_count": source["tile_count"],
        "kept_barcodes_sha256": hashlib.sha256(
            json.dumps(
                [str(value) for value in barcodes.tolist()],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "keep_mask_sha256": hashlib.sha256(
            keep_mask.astype(np.uint8, copy=False).tobytes()
        ).hexdigest(),
        "completion_receipt": source["completion_receipt"],
        "checkpoint": source["checkpoint"],
        "encoder_state_sha256": source["encoder_state_sha256"],
        "adapter_state_sha256": source["adapter_state_sha256"],
        "sources": source["sources"],
    }


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield key
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def _verify_matched_cache_summaries(
    cache_summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(cache_summaries) != {
        f"{cancer}|{arm}" for cancer in CANCERS for arm in ARMS
    }:
        raise ValueError("cache summary coordinate topology differs")
    for cancer in CANCERS:
        matched = [
            cache_summaries[f"{cancer}|{arm}"] for arm in ARMS
        ]
        if (
            len({row["ordered_tiles_sha256"] for row in matched}) != 1
            or len({row["input_tile_count"] for row in matched}) != 1
            or len({row["rows"] for row in matched}) != 1
            or len({row["kept_barcodes_sha256"] for row in matched}) != 1
            or len({row["keep_mask_sha256"] for row in matched}) != 1
        ):
            raise ValueError(
                f"{cancer} B/P/H representation inputs or keep masks differ"
            )


def _verify_training_audit(
    rows: list[dict[str, Any]],
    *,
    fold_counts: Mapping[str, Any],
) -> None:
    reliable._validate_nested_training_audit(rows)
    counts = {fold: int(fold_counts[str(fold)]) for fold in range(5)}
    total = sum(counts.values())
    if set(fold_counts) != {str(fold) for fold in range(5)}:
        raise ValueError("cohort fold-count topology differs")
    for row in rows:
        if set(row) != {
            "calibration_outer_fold", "outer_test", "inner_fits"
        }:
            raise ValueError("training-audit outer field topology differs")
        outer = int(row["calibration_outer_fold"])
        outer_test = row["outer_test"]
        if set(outer_test) != {
            "excluded_folds", "train_count", "eval_count"
        } or (
            outer_test["train_count"] != total - counts[outer]
            or outer_test["eval_count"] != counts[outer]
        ):
            raise ValueError("training-audit outer counts differ")
        for inner_fit in row["inner_fits"]:
            if set(inner_fit) != {
                "inner_fold", "excluded_folds", "train_count", "eval_count"
            }:
                raise ValueError("training-audit inner field topology differs")
            inner = int(inner_fit["inner_fold"])
            if (
                inner_fit["train_count"]
                != total - counts[outer] - counts[inner]
                or inner_fit["eval_count"] != counts[inner]
            ):
                raise ValueError("training-audit inner counts differ")


def audit(
    *,
    output_root: str | Path,
    deployment_gate_receipt: str | Path,
    loader_root_receipt: str | Path,
    collection: str | Path,
    expected_fm_seed: int,
    destination: str | Path,
) -> Path:
    seed = validate_seed(expected_fm_seed)
    root = Path(output_root).resolve()
    gate_path = Path(deployment_gate_receipt).resolve()
    loader_path = Path(loader_root_receipt).resolve()
    gate = verify_gate(gate_path)
    if file_identity(Path(reliable.__file__ or "").resolve()) != gate[
        "identities"
    ]["sources"]["reliable_fairness_head"]:
        raise ValueError("structural auditor runtime import was redirected")
    loader = verify_loader_result(loader_path, gate_path)
    collection_path = Path(collection).resolve()
    collection_receipt = verify_collection(
        collection_path,
        expected_fm_seed=seed,
        deployment_gate_receipt=gate_path,
        loader_root_receipt=loader_path,
    )
    if gate["representation_seed"] != seed:
        raise ValueError("structural audit seed differs from gate")
    if loader_path.parent != root:
        raise ValueError("loader root is outside requested output root")

    cells: dict[str, Any] = {}
    cache_identities: dict[str, Mapping[str, Any]] = {}
    cache_expectations: dict[str, dict[str, Any]] = {}
    completion_receipts = {
        arm: verify_receipt(
            gate["identities"]["completion_receipts"][arm]["canonical_path"],
            expected_schema=COMPLETION_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        for arm in ARMS
    }
    expected_cell_sources = {
        "runner": gate["identities"]["sources"]["legacy_runner"],
        "cache": gate["identities"]["sources"]["legacy_cache"],
        "stage_objectives": gate["identities"]["sources"]["stage_objectives"],
        "receipts": gate["identities"]["sources"]["receipts"],
        "completion_receipt": gate["identities"]["sources"][
            "completion_receipt"
        ],
        "reliable_cache_rows": gate["identities"]["sources"][
            "reliable_fairness_head"
        ],
        "encoder_model": gate["identities"]["sources"]["encoder_model"],
    }
    for cancer in CANCERS:
        cohort = verify_receipt(
            loader["identities"]["cohorts"][cancer]["canonical_path"],
            expected_schema=(
                "matched-cancer-fixed48-diagnostic-cohort/v1"
            ),
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        fold_counts = cohort["fold_counts"]
        diagnostic = verify_receipt(
            loader["identities"]["diagnostics"][cancer]["canonical_path"],
            expected_schema=ROOT_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        nested = diagnostic["identities"]["cells"]
        if set(nested) != set(ARMS) or any(
            set(nested[arm]) != {str(head) for head in HEAD_SEEDS}
            for arm in ARMS
        ):
            raise ValueError(f"{cancer} cell matrix differs")
        n = COHORT_SIZES[cancer]
        for arm in ARMS:
            arm_cache = None
            for head in HEAD_SEEDS:
                coordinate = f"{seed}|{arm}|{cancer}|{head}"
                cell_path = Path(
                    nested[arm][str(head)]["canonical_path"]
                ).resolve()
                cell = verify_receipt(
                    cell_path,
                    expected_schema=CELL_SCHEMA,
                    expected_study_id=STUDY_ID,
                    expected_scenario=gate["scenario"],
                )
                ids = cell.get("identities", {})
                completion = completion_receipts[arm]
                if (
                    cell.get("status") != "complete"
                    or cell.get("arm") != arm
                    or cell.get("head_seed") != head
                    or cell.get("task_id") != TASK_IDS[cancer]
                    or cell.get("prediction_rows") != 5 * n
                    or cell.get("outer_rows") != n
                    or cell.get("inner_rows") != 4 * n
                    or cell.get("fit_topology") != {"outer": 5, "inner": 20}
                    or cell.get("optimizer_objective")
                    != "BCEWithLogits_task_only"
                    or cell.get("encoder_pre_sha256")
                    != cell.get("encoder_post_sha256")
                    or cell.get("adapter_pre_sha256")
                    != cell.get("adapter_post_sha256")
                    or ids.get("completion_receipt")
                    != gate["identities"]["completion_receipts"][arm]
                    or set(ids) != {
                        "completion_receipt", "checkpoint", "adapter_cache",
                        "predictions", "training_audit", "cohort_source",
                        "sources",
                    }
                    or ids.get("checkpoint")
                    != completion.get("identities", {}).get(
                        "latest_checkpoint"
                    )
                    or cell.get("encoder_pre_sha256")
                    != completion.get("encoder_post_sha256")
                    or cell.get("adapter_pre_sha256")
                    != completion.get("adapter_post_sha256")
                    or ids.get("sources") != expected_cell_sources
                ):
                    raise ValueError(f"{coordinate} cell contract differs")
                audit_path = Path(ids["training_audit"]["canonical_path"])
                if file_identity(audit_path) != ids["training_audit"]:
                    raise ValueError(f"{coordinate} training audit drift")
                training_audit = _jsonl(audit_path)
                _verify_training_audit(
                    training_audit, fold_counts=fold_counts
                )
                cache_identity = ids["adapter_cache"]
                expectation = {
                    "completion_receipt": ids["completion_receipt"],
                    "checkpoint": ids["checkpoint"],
                    "encoder_state_sha256": cell["encoder_pre_sha256"],
                    "adapter_state_sha256": cell["adapter_pre_sha256"],
                    "sources": ids["sources"],
                    "entry_sha256": cell["cache_entry_sha256"],
                }
                if arm_cache is None:
                    arm_cache = cache_identity
                elif arm_cache != cache_identity:
                    raise ValueError(
                        f"{cancer}/{arm} heads use different caches"
                    )
                cache_identities[f"{cancer}|{arm}"] = cache_identity
                existing_expectation = cache_expectations.setdefault(
                    f"{cancer}|{arm}", expectation
                )
                if existing_expectation != expectation:
                    raise ValueError(
                        f"{cancer}/{arm} cache ancestry drifts across heads"
                    )
                export_identity = collection_receipt["identities"][
                    "exports"
                ].get(coordinate)
                if not isinstance(export_identity, Mapping):
                    raise ValueError(f"{coordinate} export is missing")
                export_receipt = verify_receipt(
                    export_identity["receipt"]["canonical_path"],
                    expected_schema=(
                        "matched-cancer-fixed48-diagnostic-export/v1"
                    ),
                    expected_study_id=STUDY_ID,
                    expected_scenario=gate["scenario"],
                )
                if export_receipt["identities"].get(
                    "diagnostic_receipt"
                ) != file_identity(cell_path):
                    raise ValueError(f"{coordinate} export ancestry differs")
                cells[coordinate] = file_identity(cell_path)
    if len(cells) != 24 or len(cache_identities) != 6:
        raise ValueError("diagnostic requires exactly 24 cells and 6 caches")
    if len({
        (
            identity["canonical_path"], identity["bytes"], identity["sha256"]
        )
        for identity in cache_identities.values()
    }) != 6:
        raise ValueError("diagnostic caches must be six distinct identities")
    cache_summaries = {
        coordinate: _verify_cache(
            Path(identity["canonical_path"]), identity
        )
        for coordinate, identity in cache_identities.items()
    }
    for coordinate, expected in cache_expectations.items():
        actual = cache_summaries[coordinate]
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(
                    f"{coordinate} cache source ancestry {key} differs"
                )
    _verify_matched_cache_summaries(cache_summaries)
    source_count = _assert_no_analysis_imports()
    receipt = build_receipt(
        schema=AUDIT_SCHEMA,
        study_id=STUDY_ID,
        scenario=gate["scenario"],
        identities={
            "deployment_gate": file_identity(gate_path),
            "loader_root": file_identity(loader_path),
            "collection": file_identity(collection_path),
            "collection_receipt": file_identity(
                collection_path.with_suffix(
                    collection_path.suffix + ".receipt.json"
                )
            ),
            "cells": cells,
            "caches": cache_identities,
            "auditor": file_identity(Path(__file__)),
            "runtime_sources": {
                name: file_identity(path)
                for name, path in RUNTIME_SOURCE_PATHS.items()
            },
        },
        fields={
            "status": "pass",
            "representation_seed": seed,
            "cell_count": 24,
            "cache_count": 6,
            "export_count": 24,
            "collection_row_count": 36_540,
            "cache_summaries": cache_summaries,
            "outcome_used_in_representation": False,
            "analysis_imported_or_invoked": False,
            "production_sources_ast_checked": source_count,
        },
    )
    output = atomic_write_receipt(destination, receipt)
    verify_audit(output, expected_fm_seed=seed)
    return output


def verify_audit(
    path: str | Path, *, expected_fm_seed: int
) -> dict[str, Any]:
    seed = validate_seed(expected_fm_seed)
    receipt = verify_receipt(
        path,
        expected_schema=AUDIT_SCHEMA,
        expected_study_id=STUDY_ID,
    )
    expected = {
        "status": "pass",
        "representation_seed": seed,
        "cell_count": 24,
        "cache_count": 6,
        "export_count": 24,
        "collection_row_count": 36_540,
        "outcome_used_in_representation": False,
        "analysis_imported_or_invoked": False,
        "production_sources_ast_checked": 5,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"structural audit {key} differs")
    identities = receipt.get("identities", {})
    if set(identities) != {
        "deployment_gate", "loader_root", "collection",
        "collection_receipt", "cells", "caches", "auditor",
        "runtime_sources",
    } or (
        len(identities.get("cells", {})) != 24
        or len(identities.get("caches", {})) != 6
    ):
        raise ValueError("structural audit identity counts differ")
    gate = verify_gate(identities["deployment_gate"]["canonical_path"])
    loader = verify_loader_result(
        identities["loader_root"]["canonical_path"],
        identities["deployment_gate"]["canonical_path"],
    )
    if gate["representation_seed"] != seed or gate["scenario"] != receipt[
        "scenario"
    ]:
        raise ValueError("structural audit gate seed/scenario differs")
    if identities["auditor"] != file_identity(Path(__file__)):
        raise ValueError("structural audit source identity differs")
    if set(identities["runtime_sources"]) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("structural audit runtime topology differs")
    for name, source in RUNTIME_SOURCE_PATHS.items():
        if identities["runtime_sources"][name] != file_identity(source):
            raise ValueError(
                f"structural audit runtime source {name} differs"
            )
    cache_summaries = {
        coordinate: _verify_cache(
            Path(identity["canonical_path"]), identity
        )
        for coordinate, identity in identities["caches"].items()
    }
    if cache_summaries != receipt.get("cache_summaries"):
        raise ValueError("structural audit cache summaries differ")
    _verify_matched_cache_summaries(cache_summaries)
    completion_receipts = {
        arm: verify_receipt(
            gate["identities"]["completion_receipts"][arm]["canonical_path"],
            expected_schema=COMPLETION_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        for arm in ARMS
    }
    expected_coordinates = {
        f"{seed}|{arm}|{cancer}|{head}"
        for arm in ARMS for cancer in CANCERS for head in HEAD_SEEDS
    }
    if set(identities["cells"]) != expected_coordinates:
        raise ValueError("structural audit cell coordinates differ")
    for coordinate, cell_identity in identities["cells"].items():
        _, arm, cancer, head_text = coordinate.split("|")
        head = int(head_text)
        if cell_identity != file_identity(cell_identity["canonical_path"]):
            raise ValueError(f"{coordinate} cell identity drift")
        cell = verify_receipt(
            cell_identity["canonical_path"],
            expected_schema=CELL_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        completion = completion_receipts[arm]
        ids = cell.get("identities", {})
        for role in (
            "checkpoint", "adapter_cache", "predictions",
            "training_audit", "cohort_source",
        ):
            if ids.get(role) != file_identity(
                ids[role]["canonical_path"]
            ):
                raise ValueError(
                    f"{coordinate} stored {role} identity drift"
                )
        latest_checkpoint = completion.get("identities", {}).get(
            "latest_checkpoint"
        )
        if latest_checkpoint != file_identity(
            latest_checkpoint["canonical_path"]
        ):
            raise ValueError(
                f"{coordinate} completion checkpoint identity drift"
            )
        if (
            cell.get("arm") != arm
            or cell.get("head_seed") != head
            or cell.get("task_id") != TASK_IDS[cancer]
            or ids.get("completion_receipt")
            != gate["identities"]["completion_receipts"][arm]
            or ids.get("checkpoint") != latest_checkpoint
            or cell.get("encoder_pre_sha256")
            != completion.get("encoder_post_sha256")
            or cell.get("adapter_pre_sha256")
            != completion.get("adapter_post_sha256")
            or ids.get("adapter_cache")
            != identities["caches"][f"{cancer}|{arm}"]
        ):
            raise ValueError(f"{coordinate} resume ancestry differs")
        cohort = verify_receipt(
            loader["identities"]["cohorts"][cancer]["canonical_path"],
            expected_schema=(
                "matched-cancer-fixed48-diagnostic-cohort/v1"
            ),
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        _verify_training_audit(
            _jsonl(Path(ids["training_audit"]["canonical_path"])),
            fold_counts=cohort["fold_counts"],
        )
    return receipt
