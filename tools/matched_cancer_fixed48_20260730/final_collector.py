#!/usr/bin/env python3
"""Cross-gate fixed-final collector for the 48 sealed per-seed collections.

This module validates lineage and row topology only.  It deliberately does not
import or invoke the statistical analyzer or verifier.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730.exporter import (
    validate_prediction_row,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_contract import (
    CANCERS,
    COHORT_SIZES,
    ROW_SCHEMA,
    SEEDS,
    STUDY_ID,
)
from .diag_exporter import verify_collection
from .serial_controller import SUCCESS_NAME, verify_seed_success
from .source_manifest import REPO, verify_manifest


SCHEMA = "matched-cancer-fixed48-final-collection/v1"
SCENARIO = "brca_luad_black_white_fixed48_final"
EXPECTED_ROWS = 1_753_920
EXPECTED_COMBINATIONS = 1_152
PRETRAINED_ENCODER_STATE_SHA256 = (
    "ba9418ed2138e42250085b04e0502d621b072c4bb60240f2845a27fbf3184bd6"
)
PROTOCOL = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/"
    "FIXED48_EXECUTION_PROTOCOL.md"
)
FINAL_LOCK = (
    REPO
    / "results/matched_cancer_stage_20260730/"
    "DIAGNOSTIC_FIXED_FINAL_LOCK.md"
)
AMENDMENT = (
    REPO
    / "results/matched_cancer_stage_20260730/"
    "DIAGNOSTIC_FIXED_FINAL_AMENDMENT_01.md"
)
FULL_CARDINALITY_PREFLIGHT = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/"
    "FULL_CARDINALITY_SYNTHETIC_PREFLIGHT_RECEIPT.json"
)
ANALYZER = (
    REPO / "tools/matched_cancer_diagnostic_20260730/analyzer.py"
)
INDEPENDENT_VERIFIER = (
    REPO / "tools/matched_cancer_diagnostic_20260730/verifier.py"
)


def _success_path(production_root: Path, seed: int) -> Path:
    parent = production_root / "diagnostic" / f"seed_{seed}"
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"missing diagnostic seed root for {seed}")
    candidates = []
    for attempt in parent.iterdir():
        if not attempt.is_dir() or attempt.is_symlink():
            raise ValueError(f"invalid diagnostic attempt entry: {attempt}")
        candidate = attempt / SUCCESS_NAME
        if candidate.exists() or candidate.is_symlink():
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"seed {seed} must have exactly one success receipt, "
            f"observed {len(candidates)}"
        )
    return candidates[0]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_seed_rows(path: Path, *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            value = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            validate_prediction_row(value)
            if value["fm_seed"] != seed:
                raise ValueError(f"{path}:{line_number}: seed drift")
            rows.append(value)
    if len(rows) != 36_540:
        raise ValueError(f"seed {seed} row count differs")
    return rows


def _publish_jsonl(
    destination: Path,
    seed_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"final collection exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for rows in seed_rows:
                for row in rows:
                    stream.write(json.dumps(
                        row,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ))
                    stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def collect(
    *,
    production_root: str | Path,
    source_manifest: str | Path,
    destination: str | Path,
) -> Path:
    """Verify 48 independent gates and publish the exact final matrix."""
    root = Path(production_root)
    if root.is_symlink():
        raise ValueError("production root may not be a symlink")
    root = root.resolve(strict=True)
    manifest_path = Path(source_manifest).resolve(strict=True)
    manifest_identity = file_identity(manifest_path)
    verify_manifest(manifest_path)

    reference_metadata: dict[tuple[str, str], tuple[Any, ...]] | None = None
    reference_patients: dict[str, set[str]] | None = None
    all_seed_rows: list[list[dict[str, Any]]] = []
    seed_identities: dict[str, Any] = {}

    for seed in SEEDS:
        if file_identity(manifest_path) != manifest_identity:
            raise ValueError("source manifest drifted during collection")
        verify_manifest(manifest_path)
        success_path = _success_path(root, seed)
        success = verify_seed_success(
            success_path,
            seed=seed,
            source_manifest=manifest_path,
            production_root=root,
        )
        calibration_root = verify_receipt(
            success["identities"]["calibration_root"]["canonical_path"],
            expected_schema=(
                "matched-cancer-fixed48-calibration-root-completion/v1"
            ),
            expected_study_id=STUDY_ID,
        )
        slot1_plain = verify_receipt(
            calibration_root["identities"]["runs"]["slot1_plain"][
                "canonical_path"
            ],
            expected_schema="matched-cancer-stage-completion/v1",
            expected_study_id=STUDY_ID,
        )
        if (
            slot1_plain.get("encoder_pre_sha256")
            != PRETRAINED_ENCODER_STATE_SHA256
        ):
            raise ValueError(
                f"seed {seed} pretrained encoder ancestor differs"
            )
        collection_identity = success["identities"]["per_seed_collection"]
        receipt_identity = success["identities"][
            "per_seed_collection_receipt"
        ]
        collection_path = Path(
            collection_identity["canonical_path"]
        ).resolve(strict=True)
        collection_receipt_path = Path(
            receipt_identity["canonical_path"]
        ).resolve(strict=True)
        if (
            file_identity(collection_path) != collection_identity
            or file_identity(collection_receipt_path) != receipt_identity
            or collection_receipt_path
            != collection_path.with_suffix(
                collection_path.suffix + ".receipt.json"
            )
        ):
            raise ValueError(f"seed {seed} collection identity differs")
        collection_receipt = verify_collection(
            collection_path, expected_fm_seed=seed
        )
        if file_identity(collection_receipt_path) != receipt_identity:
            raise ValueError(f"seed {seed} collection receipt drifted")

        rows = _load_seed_rows(collection_path, seed=seed)
        metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
        patient_sets: dict[str, set[str]] = {
            cancer: set() for cancer in CANCERS
        }
        for row in rows:
            key = (row["cancer"], row["patient_id"])
            value = (row["y_true"], row["race"], row["fold"])
            if key in metadata and metadata[key] != value:
                raise ValueError(f"seed {seed} metadata drifts within seed")
            metadata[key] = value
            patient_sets[row["cancer"]].add(row["patient_id"])
        if {
            cancer: len(patients)
            for cancer, patients in patient_sets.items()
        } != COHORT_SIZES:
            raise ValueError(f"seed {seed} patient counts differ")
        if reference_metadata is None:
            reference_metadata = metadata
            reference_patients = patient_sets
        elif (
            metadata != reference_metadata
            or patient_sets != reference_patients
        ):
            raise ValueError(f"seed {seed} cohort metadata differs")

        seed_identities[str(seed)] = {
            "success_receipt": file_identity(success_path),
            "collection": collection_identity,
            "collection_receipt": receipt_identity,
            "deployment_gate": dict(
                collection_receipt["identities"][
                    "deployment_gate_receipt"
                ]
            ),
            "loader_root": dict(
                collection_receipt["identities"]["loader_root_receipt"]
            ),
        }
        all_seed_rows.append(rows)

    if sum(map(len, all_seed_rows)) != EXPECTED_ROWS:
        raise ValueError("fixed-final row count differs")
    output = _publish_jsonl(Path(destination).resolve(), all_seed_rows)
    if file_identity(manifest_path) != manifest_identity:
        raise ValueError("source manifest drifted before final seal")
    verify_manifest(manifest_path)
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities={
            "source_manifest": manifest_identity,
            "execution_protocol": file_identity(PROTOCOL),
            "final_lock": file_identity(FINAL_LOCK),
            "amendment_01": file_identity(AMENDMENT),
            "full_cardinality_preflight": file_identity(
                FULL_CARDINALITY_PREFLIGHT
            ),
            "analyzer": file_identity(ANALYZER),
            "independent_verifier": file_identity(INDEPENDENT_VERIFIER),
            "seeds": seed_identities,
            "collected_predictions": file_identity(output),
            "collector": file_identity(Path(__file__)),
        },
        fields={
            "status": "complete",
            "fm_seeds": list(SEEDS),
            "row_schema": ROW_SCHEMA,
            "row_count": EXPECTED_ROWS,
            "combination_count": EXPECTED_COMBINATIONS,
            "cohort_sizes": dict(COHORT_SIZES),
            "analysis_performed": False,
            "pretrained_encoder_state_sha256": (
                PRETRAINED_ENCODER_STATE_SHA256
            ),
        },
    )
    receipt_path = atomic_write_receipt(
        output.with_suffix(output.suffix + ".receipt.json"), receipt
    )
    verify_final_collection(
        output,
        receipt_path=receipt_path,
        source_manifest=manifest_path,
        verify_rows=False,
    )
    return output


def verify_final_collection(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    source_manifest: str | Path,
    verify_rows: bool = True,
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    receipt_source = Path(
        receipt_path
        or source.with_suffix(source.suffix + ".receipt.json")
    ).resolve(strict=True)
    receipt = verify_receipt(
        receipt_source,
        expected_schema=SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=SCENARIO,
    )
    expected_fields = {
        "status": "complete",
        "fm_seeds": list(SEEDS),
        "row_schema": ROW_SCHEMA,
        "row_count": EXPECTED_ROWS,
        "combination_count": EXPECTED_COMBINATIONS,
        "cohort_sizes": dict(COHORT_SIZES),
        "analysis_performed": False,
        "pretrained_encoder_state_sha256": (
            PRETRAINED_ENCODER_STATE_SHA256
        ),
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"final collection {key} differs")
    identities = receipt.get("identities", {})
    if set(identities) != {
        "source_manifest",
        "execution_protocol",
        "final_lock",
        "amendment_01",
        "full_cardinality_preflight",
        "analyzer",
        "independent_verifier",
        "seeds",
        "collected_predictions",
        "collector",
    } or set(identities["seeds"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("final collection identity topology differs")
    manifest_path = Path(source_manifest).resolve(strict=True)
    verify_manifest(manifest_path)
    expected_static = {
        "source_manifest": file_identity(manifest_path),
        "execution_protocol": file_identity(PROTOCOL),
        "final_lock": file_identity(FINAL_LOCK),
        "amendment_01": file_identity(AMENDMENT),
        "full_cardinality_preflight": file_identity(
            FULL_CARDINALITY_PREFLIGHT
        ),
        "analyzer": file_identity(ANALYZER),
        "independent_verifier": file_identity(INDEPENDENT_VERIFIER),
        "collected_predictions": file_identity(source),
        "collector": file_identity(Path(__file__)),
    }
    for role, identity in expected_static.items():
        if identities[role] != identity:
            raise ValueError(f"final collection {role} identity differs")
    if verify_rows:
        count = 0
        observed_seeds: set[int] = set()
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise ValueError(
                        f"final collection line {line_number} is blank"
                    )
                row = json.loads(raw, object_pairs_hook=_unique_object)
                validate_prediction_row(row)
                observed_seeds.add(row["fm_seed"])
                count += 1
        if count != EXPECTED_ROWS or observed_seeds != set(SEEDS):
            raise ValueError("final collection row/seed matrix differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--production-root", type=Path, required=True)
    collect_parser.add_argument("--source-manifest", type=Path, required=True)
    collect_parser.add_argument("--destination", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--predictions", type=Path, required=True)
    verify_parser.add_argument("--receipt", type=Path)
    verify_parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(
            production_root=args.production_root,
            source_manifest=args.source_manifest,
            destination=args.destination,
        )
    else:
        verify_final_collection(
            args.predictions,
            receipt_path=args.receipt,
            source_manifest=args.source_manifest,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
