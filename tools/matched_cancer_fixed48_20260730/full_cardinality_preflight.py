#!/usr/bin/env python3
"""Exact-cardinality synthetic preflight for the frozen analyzer/verifier.

This module never discovers or opens a real-data path.  It materializes the
full 48 x 3 x 2 x 4 x 5 prediction matrix using deterministic synthetic
patients, runs the preregistered analyzer, and then invokes the independent
verifier against the analyzer report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730 import analyzer, verifier


SCHEMA = "matched-cancer-fixed48-full-cardinality-preflight/v1"
EXPECTED_ROWS = 1_753_920
RACE_COUNTS = {
    "BRCA": {"Black": 118, "White": 210},
    "LUAD": {"Black": 40, "White": 241},
}
HEAD_OFFSETS = {
    42001: -0.015,
    42002: -0.005,
    42003: 0.005,
    42004: 0.015,
}


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _patients(cancer: str) -> Iterator[tuple[str, int, str, int]]:
    """Yield exact synthetic cohort sizes with every race/outcome/fold cell."""
    for race in ("Black", "White"):
        count = RACE_COUNTS[cancer][race]
        for index in range(count):
            fold = index % 5
            outcome = (index // 5) % 2
            yield (
                f"SYN-{cancer}-{race.upper()}-{index:04d}",
                outcome,
                race,
                fold,
            )


def _probability(
    *, seed: int, head: int, outcome: int
) -> float:
    # All arms are deliberately identical, so the synthetic decision must be
    # equivalence.  Seed/head jitter exercises averaging without changing that
    # estimand.
    seed_offset = ((seed - 32001) % 7 - 3) * 0.0002
    value = 0.15 + 0.70 * outcome + seed_offset + HEAD_OFFSETS[head]
    if not 0.0 < value < 1.0:
        raise AssertionError("synthetic probability escaped the unit interval")
    return value


def write_predictions(destination: Path) -> tuple[int, str]:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"synthetic destination exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for seed in analyzer.FM_SEEDS:
                for cancer in analyzer.CANCERS:
                    patients = tuple(_patients(cancer))
                    if len(patients) != analyzer.COHORT_SIZES[cancer]:
                        raise AssertionError(
                            f"{cancer} synthetic cohort size drift"
                        )
                    for arm in analyzer.ARMS:
                        for head in analyzer.HEADS:
                            for patient, outcome, race, fold in patients:
                                common = {
                                    "schema": analyzer.ROW_SCHEMA,
                                    "fm_seed": seed,
                                    "arm": arm,
                                    "cancer": cancer,
                                    "head_seed": head,
                                    "patient_id": patient,
                                    "y_true": outcome,
                                    "race": race,
                                    "fold": fold,
                                    "probability": _probability(
                                        seed=seed,
                                        head=head,
                                        outcome=outcome,
                                    ),
                                }
                                rows = [{
                                    **common,
                                    "role": "outer_test",
                                    "outer_fold": fold,
                                    "inner_fold": None,
                                }]
                                rows.extend({
                                    **common,
                                    "role": "inner_calibration",
                                    "outer_fold": outer,
                                    "inner_fold": fold,
                                } for outer in analyzer.FOLDS if outer != fold)
                                for row in rows:
                                    payload = (_canonical(row) + "\n").encode()
                                    stream.write(payload)
                                    digest.update(payload)
                                    count += 1
            stream.flush()
            os.fsync(stream.fileno())
        if count != EXPECTED_ROWS:
            raise AssertionError(
                f"synthetic row count {count} != {EXPECTED_ROWS}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count, digest.hexdigest()


def _atomic_json(destination: Path, value: Mapping[str, Any]) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"report destination exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            ))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(output_root: Path) -> Path:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"preflight output root exists: {output_root}")
    output_root.mkdir(parents=True)
    predictions = output_root / "synthetic_fixed_final.jsonl"
    analysis_path = output_root / "synthetic_analysis.json"
    verification_path = output_root / "synthetic_verification.json"
    summary_path = output_root / "PREFLIGHT_RECEIPT.json"

    row_count, prediction_sha = write_predictions(predictions)
    analysis = analyzer.run(predictions)
    _atomic_json(analysis_path, analysis)
    verification = verifier.verify(predictions, analysis_path)
    _atomic_json(verification_path, verification)

    analysis_semantic = analysis.get("semantic_report", {})
    verification_semantic = verification.get("semantic_report", {})
    if analysis_semantic != verification_semantic:
        raise AssertionError("analyzer/verifier semantic reports differ")
    if not verification.get("analyzer_comparison", {}).get("match"):
        raise AssertionError("independent verifier rejected analyzer report")
    if (
        analysis_semantic.get("row_count") != EXPECTED_ROWS
        or analysis_semantic.get("counts", {}).get("combination_count") != 1152
        or analysis_semantic.get("decision", {}).get("classification")
        != "equivalent"
    ):
        raise AssertionError("full-cardinality synthetic semantics differ")
    if len(analysis_semantic.get("nested_audit", ())) != 21_600:
        raise AssertionError("nested audit cardinality differs")

    receipt = {
        "schema": SCHEMA,
        "status": "pass",
        "synthetic_only": True,
        "row_count": row_count,
        "combination_count": 1152,
        "nested_audit_count": 21_600,
        "expected_decision": "equivalent",
        "prediction_sha256": prediction_sha,
        "analysis_input_sha256": analysis_semantic.get("input_sha256"),
        "verification_input_sha256": verification_semantic.get("input_sha256"),
    }
    if len({
        receipt["prediction_sha256"],
        receipt["analysis_input_sha256"],
        receipt["verification_input_sha256"],
    }) != 1:
        raise AssertionError("synthetic input identities differ")
    _atomic_json(summary_path, receipt)
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.output_root.resolve())
    print(json.dumps({"status": "pass", "receipt": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
