#!/usr/bin/env python3
"""Independent fixed-five verifier for the matched-cancer diagnostic.

Only low-level prediction parsing, ensembling, and endpoint metric routines
are reused from the pre-existing independent fixed-final verifier.  This
module independently defines the five-seed matrix contract, inference,
decision gates, and analyzer-report comparison.  It never imports or
executes the fixed-five analyzer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from scipy import stats

from tools.matched_cancer_diagnostic_20260730 import verifier as low_level
from tools.matched_cancer_stage_20260730.receipts import file_identity


REPORT_SCHEMA = "matched-cancer-fixed5-diagnostic-verification/v1"
ANALYZER_REPORT_SCHEMA = "matched-cancer-fixed5-diagnostic-analysis/v1"
FM_SEEDS = (32001, 32002, 32003, 32004, 32005)
ARMS = ("B", "P", "H")
CANCERS = ("BRCA", "LUAD")
COHORT_SIZES = {"BRCA": 328, "LUAD": 281}
HEADS = (42001, 42002, 42003, 42004)
HEAD_HALVES = ((42001, 42002), (42003, 42004))
FOLDS = tuple(range(5))
SPECIFICITIES = low_level.SPECIFICITIES
EQUIVALENCE_MARGIN = 0.03
MATERIALITY = 0.02
ALPHA = 0.05
GATE_ABS_TOL = 1e-12
GATE_REL_TOL = 0.0
SEMANTIC_ABS_TOL = 1e-12
SEMANTIC_REL_TOL = 1e-12
EXPECTED_ROWS = 182_700
EXPECTED_COMBINATIONS = 120
EXPECTED_NESTED_AUDITS = 2_250


class VerificationError(RuntimeError):
    """A fail-closed fixed-five contract or semantic mismatch."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise VerificationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=GATE_REL_TOL,
        abs_tol=GATE_ABS_TOL,
    )


def _inclusive_ge(left: float, right: float) -> bool:
    return left > right or _close(left, right)


def _inclusive_le(left: float, right: float) -> bool:
    return left < right or _close(left, right)


def _strict_gt(left: float, right: float) -> bool:
    return left > right and not _close(left, right)


def _strict_lt(left: float, right: float) -> bool:
    return left < right and not _close(left, right)


def _sign(value: float) -> int:
    if _close(value, 0.0):
        return 0
    return 1 if value > 0.0 else -1


def _mean(values: Iterable[float]) -> float:
    observed = list(values)
    if not observed:
        raise VerificationError("cannot average an empty collection")
    if not all(math.isfinite(float(value)) for value in observed):
        raise VerificationError("cannot average non-finite values")
    return math.fsum(observed) / len(observed)


def load_predictions(
    source: Path, database: Path,
) -> tuple[sqlite3.Connection, str, int]:
    """Load rows with the older independent parser, then enforce fixed five."""
    try:
        return low_level.load_predictions(source, database)
    except low_level.VerificationError as error:
        raise VerificationError(str(error)) from error


def validate_complete(connection: sqlite3.Connection) -> dict[str, Any]:
    """Validate the exact fixed-five matrix with independent SQL.

    The cross-seed clauses are deliberately literal: a cancer/patient appears
    in exactly five FM seeds and 300 rows (5 seeds x 3 arms x 4 heads x
    5 nested roles).  No repeated measurement becomes an inferential unit.
    """
    actual_combinations = {
        (int(seed), arm, cancer, int(head))
        for seed, arm, cancer, head in connection.execute("""
            SELECT DISTINCT fm_seed, arm, cancer, head_seed
            FROM predictions
        """)
    }
    expected_combinations = {
        (seed, arm, cancer, head)
        for seed in FM_SEEDS
        for arm in ARMS
        for cancer in CANCERS
        for head in HEADS
    }
    if actual_combinations != expected_combinations:
        raise VerificationError(
            "input does not contain exactly 120 fixed-five combinations"
        )

    malformed = connection.execute("""
        SELECT fm_seed, arm, cancer, head_seed, patient_id
        FROM predictions
        GROUP BY fm_seed, arm, cancer, head_seed, patient_id
        HAVING COUNT(*) != 5
            OR SUM(role = 'outer_test') != 1
            OR SUM(role = 'inner_calibration') != 4
            OR COUNT(DISTINCT outer_fold) != 5
        LIMIT 1
    """).fetchone()
    if malformed is not None:
        raise VerificationError(
            "a patient lacks the exact one-outer/four-inner structure: "
            + repr(malformed)
        )

    within_seed_drift = connection.execute("""
        SELECT fm_seed, cancer, patient_id
        FROM predictions
        GROUP BY fm_seed, cancer, patient_id
        HAVING COUNT(*) != 60
            OR COUNT(DISTINCT y_true) != 1
            OR COUNT(DISTINCT race) != 1
            OR COUNT(DISTINCT fold) != 1
        LIMIT 1
    """).fetchone()
    if within_seed_drift is not None:
        raise VerificationError(
            "patient set or metadata differs across arms/heads: "
            + repr(within_seed_drift)
        )

    cross_seed_drift = connection.execute("""
        SELECT cancer, patient_id
        FROM predictions
        GROUP BY cancer, patient_id
        HAVING COUNT(DISTINCT fm_seed) != 5
            OR COUNT(*) != 300
            OR COUNT(DISTINCT y_true) != 1
            OR COUNT(DISTINCT race) != 1
            OR COUNT(DISTINCT fold) != 1
        LIMIT 1
    """).fetchone()
    if cross_seed_drift is not None:
        raise VerificationError(
            "patient set or metadata differs across five FM seeds: "
            + repr(cross_seed_drift)
        )

    patient_counts = {
        f"{int(seed)}:{cancer}": int(count)
        for seed, cancer, count in connection.execute("""
            SELECT fm_seed, cancer, COUNT(DISTINCT patient_id)
            FROM predictions
            GROUP BY fm_seed, cancer
            ORDER BY fm_seed, cancer
        """)
    }
    expected_counts = {
        f"{seed}:{cancer}": COHORT_SIZES[cancer]
        for seed in FM_SEEDS
        for cancer in CANCERS
    }
    if set(patient_counts) != set(expected_counts):
        raise VerificationError("cohort count keys differ from fixed five")
    for key, expected in expected_counts.items():
        if patient_counts[key] != expected:
            raise VerificationError(
                f"fixed cohort size differs at {key}: "
                f"observed {patient_counts[key]}, expected {expected}"
            )

    row_count = int(
        connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    )
    if row_count != EXPECTED_ROWS:
        raise VerificationError(
            f"fixed matrix has {row_count} rows, expected {EXPECTED_ROWS}"
        )
    return {
        "combination_count": len(actual_combinations),
        "fm_pair_count": len(FM_SEEDS),
        "patient_counts_by_seed_cancer": patient_counts,
    }


def ensemble_rows(
    connection: sqlite3.Connection,
    seed: int,
    arm: str,
    cancer: str,
    heads: Sequence[int],
) -> list[dict[str, Any]]:
    try:
        return low_level.ensemble_rows(connection, seed, arm, cancer, heads)
    except low_level.VerificationError as error:
        raise VerificationError(str(error)) from error


def nested_ieo_with_audit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    try:
        return low_level.nested_ieo_with_audit(rows)
    except low_level.VerificationError as error:
        raise VerificationError(str(error)) from error


def nested_ieo(rows: Sequence[Mapping[str, Any]]) -> float:
    value, _ = nested_ieo_with_audit(rows)
    return value


def utility_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    try:
        return low_level.utility_metrics(rows)
    except low_level.VerificationError as error:
        raise VerificationError(str(error)) from error


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise VerificationError(
            "paired inference requires exactly five independent FM seeds"
        )
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise VerificationError("paired effects must all be finite")
    mean = _mean(vector)
    df = 4
    sd = math.sqrt(
        math.fsum((value - mean) ** 2 for value in vector) / df
    )
    se = sd / math.sqrt(5)
    if se == 0.0:
        statistic = 0.0 if mean == 0.0 else None
        pvalue = 1.0 if mean == 0.0 else 0.0
    else:
        statistic = mean / se
        pvalue = float(2.0 * stats.t.sf(abs(statistic), df))
    critical90 = float(stats.t.ppf(0.95, df))
    critical95 = float(stats.t.ppf(0.975, df))
    nonzero = [value for value in vector if _sign(value) != 0]
    positive = sum(value > 0.0 for value in nonzero)
    exact_p = (
        float(stats.binomtest(positive, len(nonzero), 0.5).pvalue)
        if nonzero
        else 1.0
    )
    return {
        "per_seed_theta": vector,
        "mean": mean,
        "median": float(statistics.median(vector)),
        "sd": sd,
        "se": se,
        "minimum": min(vector),
        "maximum": max(vector),
        "t_statistic": statistic,
        "df": df,
        "two_sided_t_p": pvalue,
        "ci90": [
            mean - critical90 * se,
            mean + critical90 * se,
        ],
        "ci95": [
            mean - critical95 * se,
            mean + critical95 * se,
        ],
        "leave_one_seed_out_means": [
            _mean(
                value
                for index, value in enumerate(vector)
                if index != heldout
            )
            for heldout in range(5)
        ],
        "exact_sign_test": {
            "nonzero_pairs": len(nonzero),
            "positive_pairs": positive,
            "negative_pairs": len(nonzero) - positive,
            "zero_pairs": 5 - len(nonzero),
            "two_sided_p": exact_p,
        },
    }


def _utility_report(
    values: Mapping[tuple[int, str, str], Mapping[str, float]],
) -> dict[str, Any]:
    metrics = ("overall_auroc", "black_auprc", "black_ece")
    report: dict[str, Any] = {}
    for arm in ARMS:
        cancers = {
            cancer: {
                metric: _mean(
                    values[(seed, arm, cancer)][metric]
                    for seed in FM_SEEDS
                )
                for metric in metrics
            }
            for cancer in CANCERS
        }
        report[arm] = {
            "overall": {
                metric: _mean(
                    cancers[cancer][metric] for cancer in CANCERS
                )
                for metric in metrics
            },
            "cancers": cancers,
        }
    for arm in ("P", "H"):
        report[arm]["deltas_vs_B"] = {
            "overall": {
                metric: _mean(
                    report[arm]["cancers"][cancer][metric]
                    - report["B"]["cancers"][cancer][metric]
                    for cancer in CANCERS
                )
                for metric in metrics
            },
            "cancers": {
                cancer: {
                    metric: (
                        report[arm]["cancers"][cancer][metric]
                        - report["B"]["cancers"][cancer][metric]
                    )
                    for metric in metrics
                }
                for cancer in CANCERS
            },
        }
    return report


def utility_gate(utility: Mapping[str, Any], arm: str) -> dict[str, Any]:
    overall = utility[arm]["overall"]
    delta = utility[arm]["deltas_vs_B"]["overall"]
    mean_bounds = {
        "mean_overall_auroc_gt_0.60": _strict_gt(
            overall["overall_auroc"], 0.60
        ),
        "mean_auroc_delta_ge_-0.02": _inclusive_ge(
            delta["overall_auroc"], -0.02
        ),
        "mean_black_auprc_delta_ge_-0.02": _inclusive_ge(
            delta["black_auprc"], -0.02
        ),
        "mean_black_ece_delta_le_+0.02": _inclusive_le(
            delta["black_ece"], 0.02
        ),
    }
    cancer_bounds: dict[str, dict[str, bool]] = {}
    for cancer in CANCERS:
        local = utility[arm]["cancers"][cancer]
        change = utility[arm]["deltas_vs_B"]["cancers"][cancer]
        cancer_bounds[cancer] = {
            "auroc_gt_0.57": _strict_gt(local["overall_auroc"], 0.57),
            "auroc_delta_ge_-0.05": _inclusive_ge(
                change["overall_auroc"], -0.05
            ),
            "black_auprc_delta_ge_-0.05": _inclusive_ge(
                change["black_auprc"], -0.05
            ),
            "black_ece_delta_le_+0.05": _inclusive_le(
                change["black_ece"], 0.05
            ),
        }
    return {
        "pass": all(mean_bounds.values())
        and all(all(bounds.values()) for bounds in cancer_bounds.values()),
        "mean_bounds": mean_bounds,
        "cancer_bounds": cancer_bounds,
    }


def harm_gate(
    mean_ieo_delta_vs_b_by_cancer: Mapping[str, float],
    favored_arm: str | None,
) -> dict[str, Any]:
    values = dict(mean_ieo_delta_vs_b_by_cancer)
    passed = favored_arm is not None and set(values) == set(CANCERS) and all(
        _inclusive_le(float(values[cancer]), 0.03) for cancer in CANCERS
    )
    return {
        "favored_arm": favored_arm,
        "mean_ieo_delta_vs_B_by_cancer": values,
        "threshold_le": 0.03,
        "pass": passed,
    }


def screen_decision(
    full: Mapping[str, Any],
    head_halves: Sequence[Mapping[str, Any]],
    cancers: Mapping[str, Mapping[str, Any]],
    *,
    harm_pass: bool,
    utility_pass: bool,
) -> dict[str, Any]:
    """Apply every practical and secondary rule to five seed-level effects."""
    sign = _sign(float(full["mean"]))
    favored = "H" if sign > 0 else "P" if sign < 0 else None
    head_direction = sign != 0 and all(
        _sign(float(half["mean"])) == sign for half in head_halves
    )
    cancer_direction = sign != 0 and all(
        _sign(float(cancers[cancer]["mean"])) == sign
        for cancer in CANCERS
    )
    equivalence = (
        _strict_gt(float(full["ci90"][0]), -EQUIVALENCE_MARGIN)
        and _strict_lt(float(full["ci90"][1]), EQUIVALENCE_MARGIN)
    )
    superiority = {
        "paired_t_p_lt_0.05": _strict_lt(
            float(full["two_sided_t_p"]), ALPHA
        ),
        "absolute_mean_ge_0.02": _inclusive_ge(
            abs(float(full["mean"])), MATERIALITY
        ),
        "both_head_halves_same_strict_direction": head_direction,
        "both_cancers_same_strict_direction": cancer_direction,
        "favored_arm_harm_gate": harm_pass,
        "favored_arm_utility_gate": utility_pass,
    }
    matching_sign_count = (
        sum(
            _sign(float(value)) == sign
            for value in full["per_seed_theta"]
        )
        if sign
        else 0
    )
    large = {
        "absolute_mean_ge_0.02": _inclusive_ge(
            abs(float(full["mean"])), MATERIALITY
        ),
        "median_same_strict_direction": (
            sign != 0 and _sign(float(full["median"])) == sign
        ),
        "absolute_median_ge_0.02": _inclusive_ge(
            abs(float(full["median"])), MATERIALITY
        ),
        "at_least_four_of_five_strict_seed_signs_match": (
            matching_sign_count >= 4
        ),
        "all_leave_one_out_means_same_strict_direction": (
            sign != 0
            and all(
                _sign(float(value)) == sign
                for value in full["leave_one_seed_out_means"]
            )
        ),
        "both_head_halves_same_strict_direction": head_direction,
        "both_cancers_same_strict_direction": cancer_direction,
        "favored_arm_harm_gate": harm_pass,
        "favored_arm_utility_gate": utility_pass,
    }
    small = {
        "absolute_mean_lt_0.02": _strict_lt(
            abs(float(full["mean"])), MATERIALITY
        ),
        "every_absolute_seed_effect_lt_0.03": all(
            _strict_lt(abs(float(value)), EQUIVALENCE_MARGIN)
            for value in full["per_seed_theta"]
        ),
        "ci90_strictly_inside_+/-0.03": equivalence,
    }
    large_stable = favored is not None and all(large.values())
    small_across_five = all(small.values())
    classification = (
        f"large_stable_practical_effect_favoring_{favored}"
        if large_stable
        else (
            "small_across_five_tested_seeds"
            if small_across_five
            else "unstable_insufficient"
        )
    )
    secondary_qualified = all(superiority.values())
    secondary_classification = (
        "equivalent"
        if equivalence
        else (
            f"{favored}_superior"
            if secondary_qualified and favored is not None
            else "inconclusive"
        )
    )
    return {
        "gates": {
            "fixed5_practical_screen": {
                "large_stable": large,
                "small_across_five": small,
            },
            "secondary_original_rules": {
                "equivalence_ci90_strictly_inside_+/-0.03": equivalence,
                "superiority": superiority,
            },
        },
        "decision": {
            "classification": classification,
            "favored_arm_by_mean": favored,
            "positive_theta_favors": "H",
            "theta_definition": "iEO(P)-iEO(H)",
            "matching_strict_seed_sign_count": matching_sign_count,
            "secondary_original_classification": secondary_classification,
            "population_claim_authorized": False,
        },
    }


def analyze(connection: sqlite3.Connection) -> dict[str, Any]:
    ieo: dict[tuple[int, str, str], float] = {}
    utility_values: dict[tuple[int, str, str], Mapping[str, float]] = {}
    nested_audit: list[dict[str, Any]] = []
    for seed in FM_SEEDS:
        for arm in ARMS:
            for cancer in CANCERS:
                rows = ensemble_rows(connection, seed, arm, cancer, HEADS)
                endpoint, audit = nested_ieo_with_audit(rows)
                ieo[(seed, arm, cancer)] = endpoint
                nested_audit.extend(
                    {
                        "fm_seed": seed,
                        "arm": arm,
                        "cancer": cancer,
                        **record,
                    }
                    for record in audit
                )
                utility_values[(seed, arm, cancer)] = utility_metrics(rows)
    if len(nested_audit) != EXPECTED_NESTED_AUDITS:
        raise VerificationError(
            "nested-threshold audit does not contain exactly 2250 records"
        )

    cancer_vectors = {
        cancer: [
            ieo[(seed, "P", cancer)] - ieo[(seed, "H", cancer)]
            for seed in FM_SEEDS
        ]
        for cancer in CANCERS
    }
    full_vector = [
        _mean(cancer_vectors[cancer][index] for cancer in CANCERS)
        for index in range(5)
    ]
    full = paired_summary(full_vector)
    cancer_report = {
        cancer: {
            "per_seed_theta": cancer_vectors[cancer],
            "mean": _mean(cancer_vectors[cancer]),
        }
        for cancer in CANCERS
    }

    half_report: list[dict[str, Any]] = []
    for heads in HEAD_HALVES:
        vector = [
            _mean(
                nested_ieo(
                    ensemble_rows(connection, seed, "P", cancer, heads)
                )
                - nested_ieo(
                    ensemble_rows(connection, seed, "H", cancer, heads)
                )
                for cancer in CANCERS
            )
            for seed in FM_SEEDS
        ]
        half_report.append({
            "head_seeds": list(heads),
            "per_seed_theta": vector,
            "mean": _mean(vector),
        })

    sign = _sign(full["mean"])
    favored = "H" if sign > 0 else "P" if sign < 0 else None
    harm_values = (
        {
            cancer: _mean(
                ieo[(seed, favored, cancer)] - ieo[(seed, "B", cancer)]
                for seed in FM_SEEDS
            )
            for cancer in CANCERS
        }
        if favored is not None
        else {}
    )
    harm = harm_gate(harm_values, favored)
    harm_pass = bool(harm["pass"])

    utilities = _utility_report(utility_values)
    favored_utility = (
        utility_gate(utilities, favored)
        if favored is not None
        else {"pass": False, "mean_bounds": {}, "cancer_bounds": {}}
    )
    screened = screen_decision(
        full,
        half_report,
        cancer_report,
        harm_pass=harm_pass,
        utility_pass=bool(favored_utility["pass"]),
    )
    return {
        "nested_audit": nested_audit,
        "full": full,
        "head_halves": half_report,
        "cancers": cancer_report,
        "harm_gate": harm,
        "utility": {
            "metrics": utilities,
            "favored_arm_gate": favored_utility,
        },
        **screened,
    }


def contract_report() -> dict[str, Any]:
    return {
        "analysis": "fixed_final_5_practical_screen_no_optional_stopping",
        "fm_seeds": list(FM_SEEDS),
        "independent_fm_seed_units": 5,
        "arms": list(ARMS),
        "cancers": list(CANCERS),
        "cohort_sizes": dict(COHORT_SIZES),
        "expected_row_count": EXPECTED_ROWS,
        "expected_combination_count": EXPECTED_COMBINATIONS,
        "expected_nested_audit_count": EXPECTED_NESTED_AUDITS,
        "head_seeds": list(HEADS),
        "head_halves": [list(half) for half in HEAD_HALVES],
        "folds": list(FOLDS),
        "specificity_targets": list(SPECIFICITIES),
        "probability_ensemble_before_thresholds": True,
        "theta": "iEO(P)-iEO(H); positive favors H",
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "materiality": MATERIALITY,
        "alpha": ALPHA,
        "numeric_comparison": {
            "absolute_tolerance": GATE_ABS_TOL,
            "relative_tolerance": GATE_REL_TOL,
            "raw_values_unrounded_for_reporting": True,
        },
        "heads_cancers_folds_targets_patients_are_repeated": True,
    }


def semantic_report(
    input_sha256: str,
    row_count: int,
    counts: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "input_sha256": input_sha256,
        "row_count": row_count,
        "contract": contract_report(),
        "counts": dict(counts),
        **dict(analysis),
    }


def semantic_comparison_contract() -> dict[str, Any]:
    return {
        "scope": "analyzer_semantic_report",
        "absolute_tolerance": SEMANTIC_ABS_TOL,
        "relative_tolerance": SEMANTIC_REL_TOL,
    }


def _semantic_equal(expected: Any, observed: Any, path: str = "$") -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(observed) is not type(expected) or observed != expected:
            raise VerificationError(f"analyzer mismatch at {path}")
        return
    if isinstance(expected, int):
        if not _is_int(observed) or observed != expected:
            raise VerificationError(f"analyzer mismatch at {path}")
        return
    if isinstance(expected, float):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                expected,
                float(observed),
                rel_tol=SEMANTIC_REL_TOL,
                abs_tol=SEMANTIC_ABS_TOL,
            )
        ):
            raise VerificationError(f"analyzer mismatch at {path}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise VerificationError(f"analyzer mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, observed)):
            _semantic_equal(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            raise VerificationError(f"analyzer key mismatch at {path}")
        for key, value in expected.items():
            _semantic_equal(value, observed[key], f"{path}.{key}")
        return
    raise VerificationError(f"unsupported analyzer value at {path}")


def compare_analyzer(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=_unique_object)
    except low_level.VerificationError as error:
        raise VerificationError(str(error)) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read analyzer report: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "semantic_report"
    }:
        raise VerificationError("analyzer report envelope differs")
    if payload["schema"] != ANALYZER_REPORT_SCHEMA:
        raise VerificationError("analyzer report schema differs")
    _semantic_equal(dict(expected), payload["semantic_report"])


def verify(
    predictions: Path,
    analyzer_report: Path | None = None,
    *,
    analyzer_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not predictions.is_file() or predictions.is_symlink():
        raise VerificationError(
            f"prediction JSONL must be a regular non-symlink file: {predictions}"
        )
    with tempfile.TemporaryDirectory(
        prefix="matched-cancer-fixed5-verifier-"
    ) as temporary:
        connection, digest, row_count = load_predictions(
            predictions, Path(temporary) / "predictions.sqlite3"
        )
        try:
            counts = validate_complete(connection)
            if row_count != EXPECTED_ROWS:
                raise VerificationError(
                    f"fixed matrix has {row_count} rows, "
                    f"expected {EXPECTED_ROWS}"
                )
            analysis = analyze(connection)
        finally:
            connection.close()
    semantics = semantic_report(digest, row_count, counts, analysis)
    if analyzer_provenance is not None:
        semantics["provenance"] = dict(analyzer_provenance)
    comparison = {
        "requested": analyzer_report is not None,
        "match": None,
        "numeric_comparison": semantic_comparison_contract(),
    }
    if analyzer_report is not None:
        compare_analyzer(analyzer_report, semantics)
        comparison["match"] = True
    return {
        "schema": REPORT_SCHEMA,
        "semantic_report": semantics,
        "analyzer_comparison": comparison,
    }


def run_sealed(
    predictions: Path,
    *,
    analyzer_report: Path,
    collection_receipt: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    """Verify production artifacts; unfinished dependencies stay lazy."""
    from .final_collector import verify_final_collection
    from .source_manifest import verify_manifest

    verify_manifest(source_manifest)
    verify_final_collection(
        predictions,
        receipt_path=collection_receipt,
        source_manifest=source_manifest,
    )
    analyzer_source = Path(__file__).with_name("analyzer.py")
    provenance = {
        "source_manifest": file_identity(source_manifest),
        "collection_receipt": file_identity(collection_receipt),
        "collected_predictions": file_identity(predictions),
        "analyzer": file_identity(analyzer_source),
    }
    report = verify(
        predictions,
        analyzer_report,
        analyzer_provenance=provenance,
    )
    report["verification_provenance"] = {
        "source_manifest": file_identity(source_manifest),
        "collection_receipt": file_identity(collection_receipt),
        "collected_predictions": file_identity(predictions),
        "analyzer_report": file_identity(analyzer_report),
        "independent_verifier": file_identity(Path(__file__)),
    }
    verify_manifest(source_manifest)
    return report


def validate_output_path(output: Path, sealed_inputs: Sequence[Path]) -> None:
    if os.path.lexists(output):
        raise VerificationError(
            "output must be a new path and may not replace a file/link"
        )
    output_canonical = output.resolve(strict=False)
    for sealed in sealed_inputs:
        if output_canonical == sealed.resolve(strict=False):
            raise VerificationError(
                f"output path collides with sealed input: {sealed}"
            )


def write_output_exclusively(
    output: Path,
    rendered: str,
    sealed_inputs: Sequence[Path],
) -> None:
    validate_output_path(output, sealed_inputs)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o664)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--analyzer-report", type=Path, required=True)
    parser.add_argument("--collection-receipt", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sealed_inputs = [
        arguments.predictions,
        arguments.analyzer_report,
        arguments.collection_receipt,
        arguments.source_manifest,
    ]
    try:
        validate_output_path(arguments.output, sealed_inputs)
        report = run_sealed(
            arguments.predictions,
            analyzer_report=arguments.analyzer_report,
            collection_receipt=arguments.collection_receipt,
            source_manifest=arguments.source_manifest,
        )
        rendered = json.dumps(
            report, sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
        write_output_exclusively(arguments.output, rendered, sealed_inputs)
    except (VerificationError, OSError, ValueError) as error:
        sys.stderr.write(f"verification failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
