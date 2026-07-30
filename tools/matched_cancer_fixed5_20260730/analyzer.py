#!/usr/bin/env python3
"""Fixed-five practical-effect analyzer over one sealed prediction matrix."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats

from tools.matched_cancer_diagnostic_20260730 import analyzer as base
from tools.matched_cancer_stage_20260730.receipts import file_identity


REPORT_SCHEMA = "matched-cancer-fixed5-diagnostic-analysis/v1"
FM_SEEDS = (32001, 32002, 32003, 32004, 32005)
ARMS = base.ARMS
CANCERS = base.CANCERS
COHORT_SIZES = base.COHORT_SIZES
HEADS = base.HEADS
HEAD_HALVES = base.HEAD_HALVES
FOLDS = base.FOLDS
SPECIFICITIES = base.SPECIFICITIES
EQUIVALENCE_MARGIN = 0.03
MATERIALITY = 0.02
ALPHA = 0.05
ABS_TOL = 1e-12
EXPECTED_ROWS = 182_700
EXPECTED_COMBINATIONS = 120
EXPECTED_NESTED_AUDITS = 2_250
AnalysisError = base.AnalysisError


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=ABS_TOL)


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
    return 1 if value > 0 else -1


@contextmanager
def _fixed5_base_context() -> Iterable[None]:
    """Limit the reused row-contract implementation to the five locked seeds."""
    original = base.FM_SEEDS
    base.FM_SEEDS = FM_SEEDS
    try:
        yield
    finally:
        base.FM_SEEDS = original


def _mean(values: Iterable[float]) -> float:
    observed = list(values)
    if not observed:
        raise AnalysisError("cannot average empty values")
    return math.fsum(observed) / len(observed)


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(FM_SEEDS):
        raise AnalysisError("paired inference requires exactly five FM seeds")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise AnalysisError("paired effects must all be finite")
    count = len(vector)
    df = count - 1
    mean = _mean(vector)
    sd = math.sqrt(
        math.fsum((value - mean) ** 2 for value in vector) / df
    )
    se = sd / math.sqrt(count)
    if se == 0:
        statistic = 0.0 if mean == 0 else None
        pvalue = 1.0 if mean == 0 else 0.0
    else:
        statistic = mean / se
        pvalue = float(2 * stats.t.sf(abs(statistic), df))
    ci90_critical = float(stats.t.ppf(0.95, df))
    ci95_critical = float(stats.t.ppf(0.975, df))
    nonzero = [value for value in vector if _sign(value) != 0]
    positive = sum(value > 0 for value in nonzero)
    exact_sign_p = (
        float(stats.binomtest(positive, len(nonzero), 0.5).pvalue)
        if nonzero
        else 1.0
    )
    return {
        "per_seed_theta": vector,
        "mean": mean,
        "median": float(np.median(np.asarray(vector, dtype=float))),
        "sd": sd,
        "se": se,
        "minimum": min(vector),
        "maximum": max(vector),
        "t_statistic": statistic,
        "df": df,
        "two_sided_t_p": pvalue,
        "ci90": [
            mean - ci90_critical * se,
            mean + ci90_critical * se,
        ],
        "ci95": [
            mean - ci95_critical * se,
            mean + ci95_critical * se,
        ],
        "leave_one_seed_out_means": [
            _mean(value for index, value in enumerate(vector) if index != heldout)
            for heldout in range(count)
        ],
        "exact_sign_test": {
            "nonzero_pairs": len(nonzero),
            "positive_pairs": positive,
            "negative_pairs": len(nonzero) - positive,
            "zero_pairs": count - len(nonzero),
            "two_sided_p": exact_sign_p,
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
                    values[(seed, arm, cancer)][metric] for seed in FM_SEEDS
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
                    metric: report[arm]["cancers"][cancer][metric]
                    - report["B"]["cancers"][cancer][metric]
                    for metric in metrics
                }
                for cancer in CANCERS
            },
        }
    return report


def _utility_gate(utility: Mapping[str, Any], arm: str) -> dict[str, Any]:
    overall = utility[arm]["overall"]
    delta = utility[arm]["deltas_vs_B"]["overall"]
    main = {
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
    cancer_bounds = {}
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
    passed = all(main.values()) and all(
        all(values.values()) for values in cancer_bounds.values()
    )
    return {
        "pass": passed,
        "mean_bounds": main,
        "cancer_bounds": cancer_bounds,
    }


def analyze(
    cells: Mapping[tuple[int, str, str, int], Mapping[str, dict]],
) -> dict[str, Any]:
    ieo: dict[tuple[int, str, str], float] = {}
    utility_values: dict[tuple[int, str, str], Mapping[str, float]] = {}
    nested_audit: list[dict[str, Any]] = []
    for seed in FM_SEEDS:
        for arm in ARMS:
            for cancer in CANCERS:
                rows = base.ensemble_rows(cells, seed, arm, cancer, HEADS)
                endpoint, audit = base.nested_endpoint(rows)
                ieo[(seed, arm, cancer)] = endpoint
                nested_audit.extend(
                    {
                        "fm_seed": seed,
                        "arm": arm,
                        "cancer": cancer,
                        **entry,
                    }
                    for entry in audit
                )
                utility_values[(seed, arm, cancer)] = base.utility_metrics(rows)
    if len(nested_audit) != EXPECTED_NESTED_AUDITS:
        raise AnalysisError("nested-threshold audit cardinality differs")

    cancer_vectors = {
        cancer: [
            ieo[(seed, "P", cancer)] - ieo[(seed, "H", cancer)]
            for seed in FM_SEEDS
        ]
        for cancer in CANCERS
    }
    full_vector = [
        _mean(cancer_vectors[cancer][index] for cancer in CANCERS)
        for index in range(len(FM_SEEDS))
    ]
    full = paired_summary(full_vector)
    cancers = {
        cancer: {
            "per_seed_theta": cancer_vectors[cancer],
            "mean": _mean(cancer_vectors[cancer]),
        }
        for cancer in CANCERS
    }
    halves = []
    for heads in HEAD_HALVES:
        vector = [
            _mean(
                base.nested_ieo(
                    base.ensemble_rows(cells, seed, "P", cancer, heads)
                )
                - base.nested_ieo(
                    base.ensemble_rows(cells, seed, "H", cancer, heads)
                )
                for cancer in CANCERS
            )
            for seed in FM_SEEDS
        ]
        halves.append(
            {
                "head_seeds": list(heads),
                "per_seed_theta": vector,
                "mean": _mean(vector),
            }
        )

    sign = _sign(full["mean"])
    favored = "H" if sign > 0 else "P" if sign < 0 else None
    head_direction = sign != 0 and all(
        _sign(half["mean"]) == sign for half in halves
    )
    cancer_direction = sign != 0 and all(
        _sign(cancers[cancer]["mean"]) == sign for cancer in CANCERS
    )
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
    harm_pass = bool(favored) and all(
        _inclusive_le(value, 0.03) for value in harm_values.values()
    )
    harm_gate = {
        "favored_arm": favored,
        "mean_ieo_delta_vs_B_by_cancer": harm_values,
        "threshold_le": 0.03,
        "pass": harm_pass,
    }
    utilities = _utility_report(utility_values)
    selected_utility = (
        _utility_gate(utilities, favored)
        if favored is not None
        else {"pass": False, "mean_bounds": {}, "cancer_bounds": {}}
    )

    equivalence = (
        _strict_gt(full["ci90"][0], -EQUIVALENCE_MARGIN)
        and _strict_lt(full["ci90"][1], EQUIVALENCE_MARGIN)
    )
    secondary_superiority = {
        "paired_t_p_lt_0.05": _strict_lt(
            full["two_sided_t_p"], ALPHA
        ),
        "absolute_mean_ge_0.02": _inclusive_ge(
            abs(full["mean"]), MATERIALITY
        ),
        "both_head_halves_same_strict_direction": head_direction,
        "both_cancers_same_strict_direction": cancer_direction,
        "favored_arm_harm_gate": harm_pass,
        "favored_arm_utility_gate": bool(selected_utility["pass"]),
    }
    secondary_qualified = all(secondary_superiority.values())
    secondary_classification = (
        "equivalent"
        if equivalence
        else (
            f"{favored}_superior"
            if secondary_qualified and favored is not None
            else "inconclusive"
        )
    )

    matching_sign_count = (
        sum(_sign(value) == sign for value in full_vector) if sign else 0
    )
    median_same_strict_direction = (
        sign != 0 and _sign(full["median"]) == sign
    )
    leave_one_out_direction = sign != 0 and all(
        _sign(value) == sign for value in full["leave_one_seed_out_means"]
    )
    large_gates = {
        "absolute_mean_ge_0.02": _inclusive_ge(
            abs(full["mean"]), MATERIALITY
        ),
        "median_same_strict_direction": median_same_strict_direction,
        "absolute_median_ge_0.02": _inclusive_ge(
            abs(full["median"]), MATERIALITY
        ),
        "at_least_four_of_five_strict_seed_signs_match": (
            matching_sign_count >= 4
        ),
        "all_leave_one_out_means_same_strict_direction": (
            leave_one_out_direction
        ),
        "both_head_halves_same_strict_direction": head_direction,
        "both_cancers_same_strict_direction": cancer_direction,
        "favored_arm_harm_gate": harm_pass,
        "favored_arm_utility_gate": bool(selected_utility["pass"]),
    }
    small_gates = {
        "absolute_mean_lt_0.02": _strict_lt(
            abs(full["mean"]), MATERIALITY
        ),
        "every_absolute_seed_effect_lt_0.03": all(
            _strict_lt(abs(value), EQUIVALENCE_MARGIN)
            for value in full_vector
        ),
        "ci90_strictly_inside_+/-0.03": equivalence,
    }
    large_stable = favored is not None and all(large_gates.values())
    small_across_five = all(small_gates.values())
    classification = (
        f"large_stable_practical_effect_favoring_{favored}"
        if large_stable
        else (
            "small_across_five_tested_seeds"
            if small_across_five
            else "unstable_insufficient"
        )
    )
    return {
        "nested_audit": nested_audit,
        "full": full,
        "head_halves": halves,
        "cancers": cancers,
        "harm_gate": harm_gate,
        "utility": {
            "metrics": utilities,
            "favored_arm_gate": selected_utility,
        },
        "gates": {
            "fixed5_practical_screen": {
                "large_stable": large_gates,
                "small_across_five": small_gates,
            },
            "secondary_original_rules": {
                "equivalence_ci90_strictly_inside_+/-0.03": equivalence,
                "superiority": secondary_superiority,
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


def contract_report() -> dict[str, Any]:
    return {
        "analysis": "fixed_final_5_practical_screen_no_optional_stopping",
        "fm_seeds": list(FM_SEEDS),
        "independent_fm_seed_units": len(FM_SEEDS),
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
            "absolute_tolerance": ABS_TOL,
            "relative_tolerance": 0.0,
            "raw_values_unrounded_for_reporting": True,
        },
        "heads_cancers_folds_targets_patients_are_repeated": True,
    }


def analyze_predictions(source: Path) -> dict[str, Any]:
    """Analyze a synthetic/preflight matrix; production uses run_sealed."""
    with _fixed5_base_context():
        cells, digest, row_count = base.load_predictions(source)
        counts = base.validate_complete(cells)
        if row_count != EXPECTED_ROWS:
            raise AnalysisError(
                f"fixed matrix has {row_count} rows, expected {EXPECTED_ROWS}"
            )
        semantic = {
            "input_sha256": digest,
            "row_count": row_count,
            "contract": contract_report(),
            "counts": counts,
            **analyze(cells),
        }
    return {"schema": REPORT_SCHEMA, "semantic_report": semantic}


def run_sealed(
    source: Path,
    *,
    collection_receipt: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    from .final_collector import verify_final_collection
    from .source_manifest import verify_manifest

    verify_final_collection(
        source,
        receipt_path=collection_receipt,
        source_manifest=source_manifest,
    )
    report = analyze_predictions(source)
    report["semantic_report"]["provenance"] = {
        "source_manifest": file_identity(source_manifest),
        "collection_receipt": file_identity(collection_receipt),
        "collected_predictions": file_identity(source),
        "analyzer": file_identity(Path(__file__)),
    }
    verify_manifest(source_manifest)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("predictions", type=Path)
    result.add_argument("--collection-receipt", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run_sealed(
            args.predictions,
            collection_receipt=args.collection_receipt,
            source_manifest=args.source_manifest,
        )
        payload = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if os.path.lexists(args.output):
            raise AnalysisError(
                "output must be a new path and may not replace a file/link"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(args.output, flags, 0o664)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except (AnalysisError, OSError, ValueError) as error:
        sys.stderr.write(f"analysis failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
