#!/usr/bin/env python3
"""Independent fixed-final verifier for the matched-cancer diagnostic.

The verifier intentionally consumes only the frozen prediction JSONL contract.
It does not import, execute, or trust the study analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


ROW_SCHEMA = "matched-cancer-diagnostic-prediction/v1"
REPORT_SCHEMA = "matched-cancer-diagnostic-verification/v1"
FIELDS = frozenset({
    "schema", "fm_seed", "arm", "cancer", "head_seed", "patient_id",
    "y_true", "race", "fold", "role", "outer_fold", "inner_fold",
    "probability",
})
FM_SEEDS = tuple(range(32001, 32049))
ARMS = ("B", "P", "H")
CANCERS = ("BRCA", "LUAD")
COHORT_SIZES = {"BRCA": 328, "LUAD": 281}
HEADS = (42001, 42002, 42003, 42004)
HEAD_HALVES = ((42001, 42002), (42003, 42004))
FOLDS = tuple(range(5))
SPECIFICITIES = tuple(float(x) for x in np.linspace(0.60, 0.95, 15))
EQUIVALENCE_MARGIN = 0.03
MATERIALITY = 0.02
ALPHA = 0.05


class VerificationError(RuntimeError):
    """A fail-closed contract or semantic verification failure."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise VerificationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _validate_row(row: Any, line_no: int) -> tuple[Any, ...]:
    where = f"line {line_no}"
    if not isinstance(row, dict) or set(row) != FIELDS:
        missing = sorted(FIELDS - set(row)) if isinstance(row, dict) else []
        extra = sorted(set(row) - FIELDS) if isinstance(row, dict) else []
        raise VerificationError(
            f"{where}: row fields differ (missing={missing}, extra={extra})"
        )
    if row["schema"] != ROW_SCHEMA:
        raise VerificationError(f"{where}: invalid schema")
    if not _is_int(row["fm_seed"]) or row["fm_seed"] not in FM_SEEDS:
        raise VerificationError(f"{where}: invalid fm_seed")
    if row["arm"] not in ARMS or row["cancer"] not in CANCERS:
        raise VerificationError(f"{where}: invalid arm/cancer")
    if not _is_int(row["head_seed"]) or row["head_seed"] not in HEADS:
        raise VerificationError(f"{where}: invalid head_seed")
    if not isinstance(row["patient_id"], str) or not row["patient_id"]:
        raise VerificationError(f"{where}: invalid patient_id")
    if not _is_int(row["y_true"]) or row["y_true"] not in (0, 1):
        raise VerificationError(f"{where}: invalid y_true")
    if row["race"] not in ("Black", "White"):
        raise VerificationError(f"{where}: invalid race")
    if not _is_int(row["fold"]) or row["fold"] not in FOLDS:
        raise VerificationError(f"{where}: invalid fold")
    if not _is_int(row["outer_fold"]) or row["outer_fold"] not in FOLDS:
        raise VerificationError(f"{where}: invalid outer_fold")
    if row["role"] == "outer_test":
        if row["outer_fold"] != row["fold"] or row["inner_fold"] is not None:
            raise VerificationError(f"{where}: invalid outer-test fold fields")
    elif row["role"] == "inner_calibration":
        if (
            row["outer_fold"] == row["fold"]
            or not _is_int(row["inner_fold"])
            or row["inner_fold"] != row["fold"]
        ):
            raise VerificationError(
                f"{where}: invalid inner-calibration fold fields"
            )
    else:
        raise VerificationError(f"{where}: invalid role")
    probability = row["probability"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise VerificationError(f"{where}: invalid probability")
    return (
        row["fm_seed"], row["arm"], row["cancer"], row["head_seed"],
        row["patient_id"], row["y_true"], row["race"], row["fold"],
        row["role"], row["outer_fold"], row["inner_fold"], float(probability),
    )


def _create_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("""
        CREATE TABLE predictions (
            fm_seed INTEGER NOT NULL,
            arm TEXT NOT NULL,
            cancer TEXT NOT NULL,
            head_seed INTEGER NOT NULL,
            patient_id TEXT NOT NULL,
            y_true INTEGER NOT NULL,
            race TEXT NOT NULL,
            fold INTEGER NOT NULL,
            role TEXT NOT NULL,
            outer_fold INTEGER NOT NULL,
            inner_fold INTEGER,
            probability REAL NOT NULL,
            PRIMARY KEY (
                fm_seed, arm, cancer, head_seed, patient_id, role, outer_fold
            )
        ) WITHOUT ROWID
    """)
    return connection


def load_predictions(
    source: Path, database: Path,
) -> tuple[sqlite3.Connection, str, int]:
    connection = _create_db(database)
    digest = hashlib.sha256()
    count = 0
    insert = """
        INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with source.open("rb") as stream:
            for line_no, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    raise VerificationError(f"line {line_no}: blank row")
                try:
                    row = json.loads(raw, object_pairs_hook=_unique_object)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise VerificationError(
                        f"line {line_no}: invalid JSON: {error}"
                    ) from error
                try:
                    connection.execute(insert, _validate_row(row, line_no))
                except sqlite3.IntegrityError as error:
                    raise VerificationError(
                        f"line {line_no}: duplicate semantic row"
                    ) from error
                count += 1
                if count % 10000 == 0:
                    connection.commit()
        connection.commit()
    except Exception:
        connection.close()
        raise
    if count == 0:
        connection.close()
        raise VerificationError("prediction input is empty")
    return connection, digest.hexdigest(), count


def validate_complete(connection: sqlite3.Connection) -> dict[str, Any]:
    actual_combinations = {
        (int(seed), arm, cancer, int(head))
        for seed, arm, cancer, head in connection.execute("""
            SELECT DISTINCT fm_seed, arm, cancer, head_seed FROM predictions
        """)
    }
    expected_combinations = {
        (seed, arm, cancer, head)
        for seed in FM_SEEDS for arm in ARMS
        for cancer in CANCERS for head in HEADS
    }
    if actual_combinations != expected_combinations:
        raise VerificationError(
            "input does not contain exactly the fixed-final combination matrix"
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

    drift = connection.execute("""
        SELECT fm_seed, cancer, patient_id
        FROM predictions
        GROUP BY fm_seed, cancer, patient_id
        HAVING COUNT(*) != 60
            OR COUNT(DISTINCT y_true) != 1
            OR COUNT(DISTINCT race) != 1
            OR COUNT(DISTINCT fold) != 1
        LIMIT 1
    """).fetchone()
    if drift is not None:
        raise VerificationError(
            "patient set or metadata differs across arms/heads: " + repr(drift)
        )

    cross_seed_drift = connection.execute("""
        SELECT cancer, patient_id
        FROM predictions
        GROUP BY cancer, patient_id
        HAVING COUNT(DISTINCT fm_seed) != 48
            OR COUNT(*) != 2880
            OR COUNT(DISTINCT y_true) != 1
            OR COUNT(DISTINCT race) != 1
            OR COUNT(DISTINCT fold) != 1
        LIMIT 1
    """).fetchone()
    if cross_seed_drift is not None:
        raise VerificationError(
            "patient set or metadata differs across FM seeds: "
            + repr(cross_seed_drift)
        )

    empty_combo = connection.execute("""
        SELECT fm_seed, arm, cancer, head_seed
        FROM predictions
        GROUP BY fm_seed, arm, cancer, head_seed
        HAVING COUNT(*) = 0
        LIMIT 1
    """).fetchone()
    if empty_combo is not None:
        raise VerificationError("empty prediction combination")

    patient_counts = {
        f"{seed}:{cancer}": int(count)
        for seed, cancer, count in connection.execute("""
            SELECT fm_seed, cancer, COUNT(DISTINCT patient_id)
            FROM predictions
            GROUP BY fm_seed, cancer
            ORDER BY fm_seed, cancer
        """)
    }
    if any(count <= 0 for count in patient_counts.values()):
        raise VerificationError("empty patient cohort")
    _validate_cohort_sizes(patient_counts)
    return {
        "combination_count": len(actual_combinations),
        "fm_pair_count": len(FM_SEEDS),
        "patient_counts_by_seed_cancer": patient_counts,
    }


def _validate_cohort_sizes(patient_counts: Mapping[str, int]) -> None:
    expected = {
        f"{seed}:{cancer}": COHORT_SIZES[cancer]
        for seed in FM_SEEDS for cancer in CANCERS
    }
    if set(patient_counts) != set(expected):
        raise VerificationError("cohort count keys differ from fixed final")
    wrong = {
        key: {"observed": patient_counts[key], "expected": expected[key]}
        for key in expected if patient_counts[key] != expected[key]
    }
    if wrong:
        first = next(iter(wrong))
        detail = wrong[first]
        raise VerificationError(
            "fixed cohort size differs at "
            f"{first}: observed {detail['observed']}, "
            f"expected {detail['expected']}"
        )


def ensemble_rows(
    connection: sqlite3.Connection,
    seed: int,
    arm: str,
    cancer: str,
    heads: Sequence[int],
) -> list[dict[str, Any]]:
    marks = ",".join("?" for _ in heads)
    query = f"""
        SELECT patient_id, y_true, race, fold, role, outer_fold,
               head_seed, probability
        FROM predictions
        WHERE fm_seed = ? AND arm = ? AND cancer = ?
          AND head_seed IN ({marks})
        ORDER BY patient_id, role, outer_fold, head_seed
    """
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for patient, y, race, fold, role, outer, _, probability in (
        connection.execute(query, (seed, arm, cancer, *heads))
    ):
        groups[(patient, int(y), race, int(fold), role, int(outer))].append(
            float(probability)
        )
    if not groups or any(len(values) != len(heads) for values in groups.values()):
        raise VerificationError("head ensemble is incomplete")
    return [
        {
            "patient_id": key[0],
            "y_true": key[1],
            "race": key[2],
            "fold": key[3],
            "role": key[4],
            "outer_fold": key[5],
            "probability": math.fsum(values) / len(values),
        }
        for key, values in groups.items()
    ]


def nested_ieo_with_audit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    outer = [row for row in rows if row["role"] == "outer_test"]
    inner = [row for row in rows if row["role"] == "inner_calibration"]
    scores = np.asarray([row["probability"] for row in outer], dtype=float)
    labels = np.asarray([row["y_true"] for row in outer], dtype=int)
    races = np.asarray([row["race"] for row in outer], dtype=object)
    folds = np.asarray([row["fold"] for row in outer], dtype=int)
    if not outer or not inner:
        raise VerificationError("missing outer or inner predictions")

    disparities = []
    audit: list[dict[str, Any]] = []
    for specificity in SPECIFICITIES:
        positive = np.zeros(len(outer), dtype=bool)
        for heldout in FOLDS:
            calibration = np.asarray([
                row["probability"] for row in inner
                if row["outer_fold"] == heldout
                and row["race"] == "White"
                and row["y_true"] == 0
            ], dtype=float)
            if calibration.size == 0:
                raise VerificationError("empty White-negative calibration set")
            threshold = float(np.quantile(
                calibration, specificity, method="linear"
            ))
            mask = folds == heldout
            if not np.any(mask):
                raise VerificationError("empty heldout fold")
            positive[mask] = scores[mask] >= threshold
            heldout_white_negative = (
                mask & (races == "White") & (labels == 0)
            )
            heldout_count = int(np.sum(heldout_white_negative))
            if heldout_count == 0:
                raise VerificationError(
                    "empty heldout White-negative denominator"
                )
            audit.append({
                "specificity_target": specificity,
                "outer_fold": heldout,
                "threshold": threshold,
                "calibration_white_negative_count": int(calibration.size),
                "heldout_white_negative_count": heldout_count,
                "achieved_heldout_white_specificity": float(
                    1.0 - np.mean(positive[heldout_white_negative])
                ),
            })
        differences = []
        for outcome in (0, 1):
            rates = []
            for race in ("White", "Black"):
                mask = (labels == outcome) & (races == race)
                if not np.any(mask):
                    raise VerificationError("empty equalized-odds denominator")
                rates.append(float(np.mean(positive[mask])))
            differences.append(abs(rates[1] - rates[0]))
        disparities.append(max(differences))
    return math.fsum(disparities) / len(disparities), audit


def nested_ieo(rows: Sequence[Mapping[str, Any]]) -> float:
    value, _ = nested_ieo_with_audit(rows)
    return value


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        raise VerificationError("degenerate AUROC")
    total = 0.0
    for value in positive:
        total += float(np.sum(value > negative))
        total += 0.5 * float(np.sum(value == negative))
    return total / (positive.size * negative.size)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or int(np.sum(labels)) == 0:
        raise VerificationError("degenerate Black AUPRC")
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    cumulative_positive = np.cumsum(labels)[ends]
    cumulative_count = ends + 1
    increments = np.diff(np.r_[0, cumulative_positive])
    return float(np.sum(
        cumulative_positive / cumulative_count * increments
    ) / cumulative_positive[-1])


def _ece(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        raise VerificationError("degenerate Black ECE")
    total = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (
            (scores >= low) & (scores <= high)
            if index == 9 else (scores >= low) & (scores < high)
        )
        if np.any(mask):
            total += abs(float(np.sum(labels[mask] - scores[mask])))
    return total / labels.size


def utility_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    outer = [row for row in rows if row["role"] == "outer_test"]
    labels = np.asarray([row["y_true"] for row in outer], dtype=int)
    scores = np.asarray([row["probability"] for row in outer], dtype=float)
    races = np.asarray([row["race"] for row in outer], dtype=object)
    black = races == "Black"
    return {
        "overall_auroc": _auroc(labels, scores),
        "black_auprc": _auprc(labels[black], scores[black]),
        "black_ece": _ece(labels[black], scores[black]),
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise VerificationError("cannot average an empty collection")
    return math.fsum(materialized) / len(materialized)


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 48:
        raise VerificationError("paired inference requires exactly 48 FM seeds")
    mean = _mean(values)
    centered = [value - mean for value in values]
    sd = math.sqrt(math.fsum(value * value for value in centered) / 47)
    se = sd / math.sqrt(48)
    critical = float(stats.t.ppf(0.95, 47))
    if se == 0.0:
        # JSON has no standards-compliant representation for an infinite
        # statistic. Null records the degenerate nonzero case without losing
        # the exact p-value or point interval.
        statistic = 0.0 if mean == 0.0 else None
        pvalue = 1.0 if mean == 0.0 else 0.0
    else:
        statistic = mean / se
        pvalue = float(2 * stats.t.sf(abs(statistic), 47))
    return {
        "per_seed_theta": [float(value) for value in values],
        "mean": mean,
        "sd": sd,
        "se": se,
        "t_statistic": statistic,
        "df": 47,
        "two_sided_p": pvalue,
        "ci90": [mean - critical * se, mean + critical * se],
    }


def _utility_report(
    values: Mapping[tuple[int, str, str], Mapping[str, float]],
) -> dict[str, Any]:
    metrics = ("overall_auroc", "black_auprc", "black_ece")
    by_arm: dict[str, Any] = {}
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
        overall = {
            metric: _mean(cancers[cancer][metric] for cancer in CANCERS)
            for metric in metrics
        }
        by_arm[arm] = {"overall": overall, "cancers": cancers}

    for arm in ("P", "H"):
        by_arm[arm]["deltas_vs_B"] = {
            "overall": {
                metric: _mean(
                    by_arm[arm]["cancers"][cancer][metric]
                    - by_arm["B"]["cancers"][cancer][metric]
                    for cancer in CANCERS
                )
                for metric in metrics
            },
            "cancers": {
                cancer: {
                    metric: (
                        by_arm[arm]["cancers"][cancer][metric]
                        - by_arm["B"]["cancers"][cancer][metric]
                    )
                    for metric in metrics
                }
                for cancer in CANCERS
            },
        }
    return by_arm


def utility_gate(utility: Mapping[str, Any], arm: str) -> dict[str, Any]:
    arm_values = utility[arm]
    overall = arm_values["overall"]
    delta = arm_values["deltas_vs_B"]["overall"]
    main = {
        "mean_overall_auroc_gt_0.60": overall["overall_auroc"] > 0.60,
        "mean_auroc_delta_ge_-0.02": delta["overall_auroc"] >= -0.02,
        "mean_black_auprc_delta_ge_-0.02": delta["black_auprc"] >= -0.02,
        "mean_black_ece_delta_le_+0.02": delta["black_ece"] <= 0.02,
    }
    cancer_detail = {}
    for cancer in CANCERS:
        local = arm_values["cancers"][cancer]
        local_delta = arm_values["deltas_vs_B"]["cancers"][cancer]
        cancer_detail[cancer] = {
            "auroc_gt_0.57": local["overall_auroc"] > 0.57,
            "auroc_delta_ge_-0.05": local_delta["overall_auroc"] >= -0.05,
            "black_auprc_delta_ge_-0.05":
                local_delta["black_auprc"] >= -0.05,
            "black_ece_delta_le_+0.05": local_delta["black_ece"] <= 0.05,
        }
    passed = all(main.values()) and all(
        all(detail.values()) for detail in cancer_detail.values()
    )
    return {"pass": passed, "mean_bounds": main, "cancer_bounds": cancer_detail}


def analyze(connection: sqlite3.Connection) -> dict[str, Any]:
    ieo: dict[tuple[int, str, str], float] = {}
    utilities: dict[tuple[int, str, str], dict[str, float]] = {}
    nested_audit: list[dict[str, Any]] = []
    for seed in FM_SEEDS:
        for arm in ARMS:
            for cancer in CANCERS:
                rows = ensemble_rows(connection, seed, arm, cancer, HEADS)
                value, cell_audit = nested_ieo_with_audit(rows)
                ieo[(seed, arm, cancer)] = value
                nested_audit.extend({
                    "fm_seed": seed,
                    "arm": arm,
                    "cancer": cancer,
                    **record,
                } for record in cell_audit)
                utilities[(seed, arm, cancer)] = utility_metrics(rows)

    cancer_vectors = {
        cancer: [
            ieo[(seed, "P", cancer)] - ieo[(seed, "H", cancer)]
            for seed in FM_SEEDS
        ]
        for cancer in CANCERS
    }
    full_values = [
        _mean(cancer_vectors[cancer][index] for cancer in CANCERS)
        for index in range(len(FM_SEEDS))
    ]
    full = paired_summary(full_values)
    cancer_report = {
        cancer: {
            "per_seed_theta": cancer_vectors[cancer],
            "mean": _mean(cancer_vectors[cancer]),
        }
        for cancer in CANCERS
    }

    half_report = []
    for heads in HEAD_HALVES:
        per_seed = []
        for seed in FM_SEEDS:
            effects = []
            for cancer in CANCERS:
                p = nested_ieo(ensemble_rows(
                    connection, seed, "P", cancer, heads
                ))
                h = nested_ieo(ensemble_rows(
                    connection, seed, "H", cancer, heads
                ))
                effects.append(p - h)
            per_seed.append(_mean(effects))
        half_report.append({
            "head_seeds": list(heads),
            "per_seed_theta": per_seed,
            "mean": _mean(per_seed),
        })

    favored = "H" if full["mean"] > 0 else "P" if full["mean"] < 0 else None
    sign = 1 if favored == "H" else -1 if favored == "P" else 0
    head_direction = (
        sign != 0 and all(sign * half["mean"] > 0 for half in half_report)
    )
    cancer_direction = (
        sign != 0 and all(
            sign * cancer_report[cancer]["mean"] > 0 for cancer in CANCERS
        )
    )

    harm_detail: dict[str, float] = {}
    harm_pass = False
    if favored is not None:
        harm_detail = {
            cancer: _mean(
                ieo[(seed, favored, cancer)] - ieo[(seed, "B", cancer)]
                for seed in FM_SEEDS
            )
            for cancer in CANCERS
        }
        harm_pass = all(value <= 0.03 for value in harm_detail.values())
    harm_gate = {
        "favored_arm": favored,
        "mean_ieo_delta_vs_B_by_cancer": harm_detail,
        "threshold_le": 0.03,
        "pass": harm_pass,
    }

    utility = _utility_report(utilities)
    selected_utility = (
        utility_gate(utility, favored) if favored is not None
        else {"pass": False, "mean_bounds": {}, "cancer_bounds": {}}
    )
    ci_low, ci_high = full["ci90"]
    equivalence = (
        ci_low > -EQUIVALENCE_MARGIN and ci_high < EQUIVALENCE_MARGIN
    )
    superiority_gates = {
        "paired_p_lt_0.05": full["two_sided_p"] < ALPHA,
        "absolute_mean_ge_0.02": abs(full["mean"]) >= MATERIALITY,
        "both_head_halves_same_strict_direction": head_direction,
        "both_cancers_same_strict_direction": cancer_direction,
        "favored_arm_harm_gate": harm_pass,
        "favored_arm_utility_gate": bool(selected_utility["pass"]),
    }
    superiority = all(superiority_gates.values())
    if equivalence:
        classification = "equivalent"
    elif superiority and favored is not None:
        classification = f"{favored}_superior"
    else:
        classification = "inconclusive"
    decision = {
        "classification": classification,
        "favored_arm": favored,
        "equivalence_precedence": equivalence,
        "positive_theta_favors": "H",
        "theta_definition": "iEO(P)-iEO(H)",
    }
    return {
        "nested_audit": nested_audit,
        "full": full,
        "head_halves": half_report,
        "cancers": cancer_report,
        "harm_gate": harm_gate,
        "utility": {
            "metrics": utility,
            "favored_arm_gate": selected_utility,
        },
        "gates": {
            "equivalence_ci90_strictly_inside_+/-0.03": equivalence,
            "superiority": superiority_gates,
        },
        "decision": decision,
    }


def contract_report() -> dict[str, Any]:
    expected_row_count = (
        len(FM_SEEDS) * len(ARMS) * len(HEADS) * 5
        * sum(COHORT_SIZES.values())
    )
    return {
        "analysis": "fixed_final_48_no_optional_stopping",
        "fm_seeds": list(FM_SEEDS),
        "arms": list(ARMS),
        "cancers": list(CANCERS),
        "cohort_sizes": dict(COHORT_SIZES),
        "expected_row_count": expected_row_count,
        "head_seeds": list(HEADS),
        "head_halves": [list(group) for group in HEAD_HALVES],
        "folds": list(FOLDS),
        "specificity_targets": list(SPECIFICITIES),
        "probability_ensemble_before_thresholds": True,
        "theta": "iEO(P)-iEO(H); positive favors H",
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "materiality": MATERIALITY,
        "alpha": ALPHA,
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
            or not math.isclose(
                expected, float(observed), rel_tol=1e-12, abs_tol=1e-12
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
        for key in expected:
            _semantic_equal(expected[key], observed[key], f"{path}.{key}")
        return
    raise VerificationError(f"unsupported comparison value at {path}")


def compare_analyzer(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read analyzer report: {error}") from error
    candidate = (
        payload.get("semantic_report")
        if isinstance(payload, dict) and "semantic_report" in payload
        else payload
    )
    _semantic_equal(dict(expected), candidate)


def validate_output_path(output: Path, sealed_inputs: Sequence[Path]) -> None:
    if output.is_symlink():
        raise VerificationError("output path must not be a symlink")
    output_canonical = output.resolve(strict=False)
    for sealed in sealed_inputs:
        sealed_canonical = sealed.resolve(strict=False)
        if output_canonical == sealed_canonical:
            raise VerificationError(
                f"output path collides with sealed input: {sealed}"
            )
        if output.exists() and sealed.exists():
            try:
                if os.path.samefile(output, sealed):
                    raise VerificationError(
                        f"output path aliases sealed input: {sealed}"
                    )
            except OSError as error:
                raise VerificationError(
                    f"cannot compare output with sealed input: {error}"
                ) from error


def write_output_safely(
    output: Path, rendered: str, sealed_inputs: Sequence[Path],
) -> None:
    # Repeat immediately before opening to narrow the check/write interval.
    validate_output_path(output, sealed_inputs)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o666)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(rendered)


def verify(predictions: Path, analyzer_report: Path | None = None) -> dict[str, Any]:
    if not predictions.is_file():
        raise VerificationError(f"prediction JSONL is not a file: {predictions}")
    with tempfile.TemporaryDirectory(prefix="matched-cancer-verifier-") as temp:
        connection, digest, row_count = load_predictions(
            predictions, Path(temp) / "predictions.sqlite3"
        )
        try:
            counts = validate_complete(connection)
            expected_row_count = (
                len(FM_SEEDS) * len(ARMS) * len(HEADS) * 5
                * sum(COHORT_SIZES.values())
            )
            if row_count != expected_row_count:
                raise VerificationError(
                    f"fixed matrix has {row_count} rows, "
                    f"expected {expected_row_count}"
                )
            analysis = analyze(connection)
        finally:
            connection.close()
    semantics = semantic_report(digest, row_count, counts, analysis)
    comparison = {"requested": analyzer_report is not None, "match": None}
    if analyzer_report is not None:
        compare_analyzer(analyzer_report, semantics)
        comparison["match"] = True
    return {
        "schema": REPORT_SCHEMA,
        "semantic_report": semantics,
        "analyzer_comparison": comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="strict prediction JSONL")
    parser.add_argument(
        "--analyzer-report", type=Path,
        help="optional analyzer semantic JSON to compare completely",
    )
    parser.add_argument(
        "--output", type=Path,
        help="write verifier JSON here instead of stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        sealed_inputs = [arguments.predictions]
        if arguments.analyzer_report is not None:
            sealed_inputs.append(arguments.analyzer_report)
        if arguments.output is not None:
            validate_output_path(arguments.output, sealed_inputs)
        report = verify(arguments.predictions, arguments.analyzer_report)
        rendered = json.dumps(
            report, sort_keys=True, indent=2, allow_nan=False
        ) + "\n"
        if arguments.output is None:
            sys.stdout.write(rendered)
        else:
            write_output_safely(
                arguments.output, rendered, sealed_inputs
            )
    except (VerificationError, OSError, ValueError) as error:
        sys.stderr.write(f"verification failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
