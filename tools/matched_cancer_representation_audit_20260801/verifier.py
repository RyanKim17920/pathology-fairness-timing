#!/usr/bin/env python3
"""Independent semantic verifier for the fixed-five representation audit.

The verifier intentionally does not import ``analyzer.py``.  It validates the
analysis artifact and its receipt, enforces the frozen four-contrast topology,
and recomputes every primary gate from the raw per-seed cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from . import contract


ANALYSIS_SCHEMA = contract.ANALYSIS_SCHEMA
ANALYSIS_RECEIPT_SCHEMA = (
    "matched-cancer-fixed5-representation-analysis-receipt/v1"
)
VERIFICATION_SCHEMA = (
    "matched-cancer-fixed5-representation-independent-verification/v1"
)
GATE_SCHEMA = "matched-cancer-representation-primary-gate/v1"
STUDY_ID = contract.STUDY_ID
FM_SEEDS = tuple(contract.FM_SEEDS)
CANCERS = tuple(contract.CANCERS)
VIEWS = tuple(contract.TILE_VIEWS)
PROBE_LEVELS = tuple(contract.PROBE_LEVELS)
CONTRASTS = tuple(contract.GATE_ELIGIBLE_CONTRASTS)
MINIMUM_LEAKAGE_REDUCTION = 0.05
MAXIMUM_CANCER_LOSS = 0.02
MINIMUM_PASSING_SEEDS = 4
ABS_TOLERANCE = 1e-12

REPO = Path("/admin/home/ryan.kim/nt")
LOCK_PATH = contract.LOCK_PATH
NUMERIC_AMENDMENT_PATH = contract.NUMERIC_AMENDMENT_PATH
ANALYZER_PATH = Path(__file__).with_name("analyzer.py")

TOP_LEVEL_KEYS = {
    "schema",
    "study_id",
    "status",
    "diagnosis_free",
    "inference_unit",
    "fm_seeds",
    "contrasts",
    "identities",
}
CONTRAST_KEYS = {
    "candidate",
    "baseline",
    "race_cells",
    "cancer_cells",
    "reported_gate",
}
RACE_CELL_KEYS = {
    "fm_seed",
    "cancer",
    "view",
    "probe_level",
    "baseline_oriented_leakage",
    "candidate_oriented_leakage",
}
CANCER_CELL_KEYS = {
    "fm_seed",
    "view",
    "baseline_auroc",
    "candidate_auroc",
}
IDENTITY_ROLES = {"metric_input", "lock", "numeric_amendment", "analyzer"}
IDENTITY_KEYS = {"canonical_path", "bytes", "sha256"}
RECEIPT_KEYS = {
    "schema",
    "study_id",
    "status",
    "analysis_report",
    "identities",
}


class VerificationError(RuntimeError):
    """A fail-closed artifact, topology, numeric, or semantic mismatch."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON constant {value!r}")


def load_json(path: Path) -> Any:
    """Load strict JSON, rejecting duplicate keys and NaN/Infinity."""
    try:
        return json.loads(
            Path(path).read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except VerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot load strict JSON {path}: {error}") from error


def file_identity(path: Path) -> dict[str, Any]:
    """Independently compute the canonical path, byte size, and SHA-256."""
    candidate = Path(path)
    if candidate.is_symlink():
        raise VerificationError(f"identity source may not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise VerificationError(f"identity source is unavailable: {candidate}") from error
    if not resolved.is_file():
        raise VerificationError(f"identity source is not a regular file: {resolved}")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {
        "canonical_path": str(resolved),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _require_exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{context} must be an object")
    if set(value) != expected:
        raise VerificationError(
            f"{context} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, context: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < low or result > high:
        raise VerificationError(
            f"{context} must be finite and in [{low}, {high}]"
        )
    return result


def _canonical_field(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def reject_diagnosis_fields(value: Any, *, context: str = "analysis") -> None:
    """Reject diagnosis/outcome-bearing keys recursively."""
    forbidden = tuple(contract.DIAGNOSIS_FIELD_DENYLIST)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _canonical_field(key)
            if (
                (normalized in forbidden and normalized != "diagnosis_free")
                or "tp53" in normalized
                or ("diagnosis" in normalized and normalized != "diagnosis_free")
                or "outcome" in normalized
                or normalized.endswith("y_true")
            ):
                raise VerificationError(
                    f"{context} contains forbidden diagnosis field {key!r}"
                )
            reject_diagnosis_fields(nested, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_diagnosis_fields(nested, context=f"{context}[{index}]")


def _ge(left: float, right: float) -> bool:
    return left >= right or math.isclose(
        left, right, rel_tol=0.0, abs_tol=ABS_TOLERANCE
    )


def _le(left: float, right: float) -> bool:
    return left <= right or math.isclose(
        left, right, rel_tol=0.0, abs_tol=ABS_TOLERANCE
    )


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(FM_SEEDS) or not all(math.isfinite(value) for value in values):
        raise VerificationError("gate summary requires exactly five finite values")
    return {
        "count": len(values),
        "mean": math.fsum(values) / len(values),
        "median": float(statistics.median(values)),
        "minimum": min(values),
        "maximum": max(values),
        "values": list(values),
    }


def _parse_race_cells(rows: Any) -> dict[tuple[int, str, str, str], tuple[float, float]]:
    if not isinstance(rows, list):
        raise VerificationError("race_cells must be an array")
    expected = {
        (seed, cancer, view, level)
        for seed in FM_SEEDS
        for cancer in CANCERS
        for view in VIEWS
        for level in PROBE_LEVELS
    }
    observed: dict[tuple[int, str, str, str], tuple[float, float]] = {}
    for index, raw in enumerate(rows):
        row = _require_exact_keys(raw, RACE_CELL_KEYS, f"race_cells[{index}]")
        seed = row["fm_seed"]
        if not _is_int(seed) or seed not in FM_SEEDS:
            raise VerificationError(f"race_cells[{index}] has invalid FM seed")
        cancer = row["cancer"]
        view = row["view"]
        level = row["probe_level"]
        if cancer not in CANCERS or view not in VIEWS or level not in PROBE_LEVELS:
            raise VerificationError(f"race_cells[{index}] topology drift")
        key = (seed, cancer, view, level)
        if key in observed:
            raise VerificationError(f"duplicate race gate cell {key!r}")
        baseline = _finite_number(
            row["baseline_oriented_leakage"],
            f"race_cells[{index}].baseline_oriented_leakage",
            low=0.0,
            high=0.5,
        )
        candidate = _finite_number(
            row["candidate_oriented_leakage"],
            f"race_cells[{index}].candidate_oriented_leakage",
            low=0.0,
            high=0.5,
        )
        observed[key] = (baseline, candidate)
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise VerificationError(
            f"race gate does not contain exactly 40 cells; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    return observed


def _parse_cancer_cells(rows: Any) -> dict[tuple[int, str], tuple[float, float]]:
    if not isinstance(rows, list):
        raise VerificationError("cancer_cells must be an array")
    expected = {(seed, view) for seed in FM_SEEDS for view in VIEWS}
    observed: dict[tuple[int, str], tuple[float, float]] = {}
    for index, raw in enumerate(rows):
        row = _require_exact_keys(raw, CANCER_CELL_KEYS, f"cancer_cells[{index}]")
        seed = row["fm_seed"]
        view = row["view"]
        if not _is_int(seed) or seed not in FM_SEEDS or view not in VIEWS:
            raise VerificationError(f"cancer_cells[{index}] topology drift")
        key = (seed, view)
        if key in observed:
            raise VerificationError(f"duplicate cancer gate cell {key!r}")
        baseline = _finite_number(
            row["baseline_auroc"],
            f"cancer_cells[{index}].baseline_auroc",
            low=0.0,
            high=1.0,
        )
        candidate = _finite_number(
            row["candidate_auroc"],
            f"cancer_cells[{index}].candidate_auroc",
            low=0.0,
            high=1.0,
        )
        observed[key] = (baseline, candidate)
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise VerificationError(
            f"cancer gate does not contain exactly 10 cells; missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    return observed


def recompute_gate(race_rows: Any, cancer_rows: Any) -> dict[str, Any]:
    """Independently recompute the exact 4/5-plus-median primary gate."""
    race = _parse_race_cells(race_rows)
    cancer = _parse_cancer_cells(cancer_rows)

    race_strata: dict[str, dict[str, Any]] = {}
    all_race = True
    for cancer_name in CANCERS:
        for view in VIEWS:
            for level in PROBE_LEVELS:
                reductions = [
                    race[(seed, cancer_name, view, level)][0]
                    - race[(seed, cancer_name, view, level)][1]
                    for seed in FM_SEEDS
                ]
                summary = _summary(reductions)
                passing = sum(
                    _ge(value, MINIMUM_LEAKAGE_REDUCTION) for value in reductions
                )
                median_pass = _ge(
                    summary["median"], MINIMUM_LEAKAGE_REDUCTION
                )
                stratum_pass = passing >= MINIMUM_PASSING_SEEDS and median_pass
                all_race = all_race and stratum_pass
                race_strata[f"{cancer_name}|{view}|{level}"] = {
                    "reduction_summary": summary,
                    "seeds_at_or_above_threshold": passing,
                    "four_of_five_pass": passing >= MINIMUM_PASSING_SEEDS,
                    "median_at_or_above_threshold": median_pass,
                    "pass": stratum_pass,
                }

    cancer_views: dict[str, dict[str, Any]] = {}
    all_cancer = True
    for view in VIEWS:
        losses = [
            cancer[(seed, view)][0] - cancer[(seed, view)][1]
            for seed in FM_SEEDS
        ]
        summary = _summary(losses)
        passing = sum(_le(value, MAXIMUM_CANCER_LOSS) for value in losses)
        median_pass = _le(summary["median"], MAXIMUM_CANCER_LOSS)
        view_pass = passing >= MINIMUM_PASSING_SEEDS and median_pass
        all_cancer = all_cancer and view_pass
        cancer_views[view] = {
            "loss_summary": summary,
            "seeds_at_or_below_maximum_loss": passing,
            "four_of_five_pass": passing >= MINIMUM_PASSING_SEEDS,
            "median_at_or_below_maximum_loss": median_pass,
            "pass": view_pass,
        }

    passed = bool(all_race and all_cancer)
    return {
        "schema": GATE_SCHEMA,
        "audit_contract_schema": contract.SCHEMA,
        "semantics": {
            "leakage_reduction": (
                "matched_oriented_leakage_minus_fair_oriented_leakage"
            ),
            "cancer_probe_loss": "matched_auroc_minus_fair_auroc",
            "seed_rule": (
                "each stratum/view requires threshold in >=4/5 seeds and "
                "threshold-consistent median"
            ),
            "missing_or_duplicate_cells": "fail_closed_with_AnalysisError",
        },
        "thresholds": {
            "minimum_oriented_leakage_reduction": MINIMUM_LEAKAGE_REDUCTION,
            "maximum_cancer_probe_auroc_loss": MAXIMUM_CANCER_LOSS,
        },
        "exact_dimensions": {
            "fm_seeds": list(FM_SEEDS),
            "cancers": list(CANCERS),
            "views": list(VIEWS),
            "levels": list(PROBE_LEVELS),
            "race_cell_count": 40,
            "cancer_cell_count": 10,
        },
        "race_leakage_strata": race_strata,
        "cancer_probe_views": cancer_views,
        "all_race_leakage_strata_pass": bool(all_race),
        "all_cancer_probe_views_pass": bool(all_cancer),
        "pass": passed,
        "classification": "active" if passed else "inactive",
    }


def _semantic_equal(left: Any, right: Any, context: str = "reported_gate") -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not bool or type(right) is not bool or left != right:
            raise VerificationError(f"{context} boolean mismatch")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            raise VerificationError(f"{context} contains non-finite numeric value")
        if not math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise VerificationError(
                f"{context} numeric mismatch: {left!r} != {right!r}"
            )
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise VerificationError(f"{context} object-key mismatch")
        for key in left:
            _semantic_equal(left[key], right[key], f"{context}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise VerificationError(f"{context} array-length mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _semantic_equal(left_item, right_item, f"{context}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise VerificationError(f"{context} mismatch: {left!r} != {right!r}")


def _validate_identity(value: Any, expected: Mapping[str, Any], context: str) -> None:
    identity = _require_exact_keys(value, IDENTITY_KEYS, context)
    if dict(identity) != dict(expected):
        raise VerificationError(f"{context} identity mismatch")


def validate_analysis_value(
    value: Any,
    *,
    expected_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a loaded report and return independently computed gates."""
    report = _require_exact_keys(value, TOP_LEVEL_KEYS, "analysis report")
    reject_diagnosis_fields(report)
    if report["schema"] != ANALYSIS_SCHEMA:
        raise VerificationError("analysis report schema drift")
    if report["study_id"] != STUDY_ID or report["status"] != "complete":
        raise VerificationError("analysis report study/status drift")
    if report["diagnosis_free"] is not True:
        raise VerificationError("analysis report is not diagnosis-free")
    if report["inference_unit"] != "FM seed":
        raise VerificationError("analysis inference unit must be FM seed")
    if report["fm_seeds"] != list(FM_SEEDS):
        raise VerificationError("analysis FM-seed topology drift")

    identities = _require_exact_keys(
        report["identities"], IDENTITY_ROLES, "analysis identities"
    )
    if set(expected_identities) != IDENTITY_ROLES:
        raise VerificationError("internal expected-identity topology drift")
    for role in sorted(IDENTITY_ROLES):
        _validate_identity(
            identities[role], expected_identities[role], f"identities.{role}"
        )

    contrasts = report["contrasts"]
    if not isinstance(contrasts, list) or len(contrasts) != len(CONTRASTS):
        raise VerificationError("analysis must contain exactly four contrasts")
    computed: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(contrasts):
        item = _require_exact_keys(raw, CONTRAST_KEYS, f"contrasts[{index}]")
        pair = (item["candidate"], item["baseline"])
        if pair != CONTRASTS[index] or pair in seen:
            raise VerificationError(f"invalid or duplicate gate contrast {pair!r}")
        seen.add(pair)
        gate = recompute_gate(item["race_cells"], item["cancer_cells"])
        _semantic_equal(item["reported_gate"], gate)
        computed[f"{pair[0]}|{pair[1]}"] = gate
    if seen != set(CONTRASTS):
        raise VerificationError("gate-eligible contrast topology is incomplete")
    return computed


def verify_analysis_files(
    analysis_report: Path,
    analysis_receipt: Path,
    metric_input: Path,
    *,
    lock: Path = LOCK_PATH,
    numeric_amendment: Path = NUMERIC_AMENDMENT_PATH,
    analyzer_source: Path = ANALYZER_PATH,
) -> dict[str, Any]:
    """Verify identities, receipt binding, topology, and gate semantics."""
    report_path = Path(analysis_report)
    receipt_path = Path(analysis_receipt)
    expected_identities = {
        "metric_input": file_identity(metric_input),
        "lock": file_identity(lock),
        "numeric_amendment": file_identity(numeric_amendment),
        "analyzer": file_identity(analyzer_source),
    }
    report = load_json(report_path)
    gates = validate_analysis_value(
        report, expected_identities=expected_identities
    )

    receipt = _require_exact_keys(
        load_json(receipt_path), RECEIPT_KEYS, "analysis receipt"
    )
    reject_diagnosis_fields(receipt, context="analysis receipt")
    if receipt["schema"] != ANALYSIS_RECEIPT_SCHEMA:
        raise VerificationError("analysis receipt schema drift")
    if receipt["study_id"] != STUDY_ID or receipt["status"] != "complete":
        raise VerificationError("analysis receipt study/status drift")
    _validate_identity(
        receipt["analysis_report"], file_identity(report_path), "analysis_report"
    )
    bound_identities = _require_exact_keys(
        receipt["identities"], IDENTITY_ROLES, "receipt identities"
    )
    if dict(bound_identities) != dict(report["identities"]):
        raise VerificationError("analysis receipt does not bind report identities")

    source_paths = {
        "metric_input": Path(metric_input),
        "lock": Path(lock),
        "numeric_amendment": Path(numeric_amendment),
        "analyzer": Path(analyzer_source),
    }
    for role, path in source_paths.items():
        if file_identity(path) != expected_identities[role]:
            raise VerificationError(f"{role} changed during verification")
    if file_identity(report_path) != receipt["analysis_report"]:
        raise VerificationError("analysis report changed during verification")

    return {
        "schema": VERIFICATION_SCHEMA,
        "study_id": STUDY_ID,
        "status": "pass",
        "diagnosis_free": True,
        "scientific_gate_passes": {
            key: bool(gate["pass"]) for key, gate in gates.items()
        },
        "semantic_report": {"contrasts": gates},
        "verification_provenance": {
            "analysis_report": file_identity(report_path),
            "analysis_receipt": file_identity(receipt_path),
            **expected_identities,
            "independent_verifier": file_identity(Path(__file__)),
        },
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one verification artifact without overwriting prior state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise VerificationError(
            f"verification output already exists: {destination}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        # Preserve a partial artifact for fail-closed incident inspection.
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_report", type=Path)
    parser.add_argument("analysis_receipt", type=Path)
    parser.add_argument("metric_input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument(
        "--numeric-amendment", type=Path, default=NUMERIC_AMENDMENT_PATH
    )
    parser.add_argument("--analyzer-source", type=Path, default=ANALYZER_PATH)
    arguments = parser.parse_args(argv)
    try:
        report = verify_analysis_files(
            arguments.analysis_report,
            arguments.analysis_receipt,
            arguments.metric_input,
            lock=arguments.lock,
            numeric_amendment=arguments.numeric_amendment,
            analyzer_source=arguments.analyzer_source,
        )
        write_json_exclusive(arguments.output, report)
    except VerificationError as error:
        sys.stderr.write(f"representation audit verification failed: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
