#!/usr/bin/env python3
"""Preregistered fixed-final paired analyzer for the matched-cancer diagnostic.

There is deliberately no default prediction location. The analyzer opens only
the explicit synthetic or future sealed JSONL supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


ROW_SCHEMA = "matched-cancer-diagnostic-prediction/v1"
REPORT_SCHEMA = "matched-cancer-diagnostic-analysis/v1"
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
SPECIFICITIES = tuple(float(value) for value in np.linspace(0.60, 0.95, 15))
EQUIVALENCE_MARGIN = 0.03
MATERIALITY = 0.02
ALPHA = 0.05


class AnalysisError(RuntimeError):
    """The prediction contract or a fixed analysis invariant failed."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_row(row: Any, line: int) -> tuple[Any, ...]:
    where = f"line {line}"
    if not isinstance(row, dict) or set(row) != FIELDS:
        missing = sorted(FIELDS - set(row)) if isinstance(row, dict) else []
        extra = sorted(set(row) - FIELDS) if isinstance(row, dict) else []
        raise AnalysisError(
            f"{where}: fields differ (missing={missing}, extra={extra})"
        )
    if row["schema"] != ROW_SCHEMA:
        raise AnalysisError(f"{where}: wrong row schema")
    if not _is_int(row["fm_seed"]) or row["fm_seed"] not in FM_SEEDS:
        raise AnalysisError(f"{where}: invalid FM seed")
    if row["arm"] not in ARMS or row["cancer"] not in CANCERS:
        raise AnalysisError(f"{where}: invalid arm/cancer")
    if not _is_int(row["head_seed"]) or row["head_seed"] not in HEADS:
        raise AnalysisError(f"{where}: invalid head seed")
    if not isinstance(row["patient_id"], str) or not row["patient_id"]:
        raise AnalysisError(f"{where}: invalid patient ID")
    if not _is_int(row["y_true"]) or row["y_true"] not in (0, 1):
        raise AnalysisError(f"{where}: invalid binary outcome")
    if row["race"] not in ("Black", "White"):
        raise AnalysisError(f"{where}: race must be Black or White")
    if not _is_int(row["fold"]) or row["fold"] not in FOLDS:
        raise AnalysisError(f"{where}: invalid patient fold")
    if not _is_int(row["outer_fold"]) or row["outer_fold"] not in FOLDS:
        raise AnalysisError(f"{where}: invalid outer fold")
    if row["role"] == "outer_test":
        if row["outer_fold"] != row["fold"] or row["inner_fold"] is not None:
            raise AnalysisError(f"{where}: malformed outer-test row")
    elif row["role"] == "inner_calibration":
        if (
            row["outer_fold"] == row["fold"]
            or not _is_int(row["inner_fold"])
            or row["inner_fold"] != row["fold"]
        ):
            raise AnalysisError(f"{where}: malformed inner-calibration row")
    else:
        raise AnalysisError(f"{where}: invalid prediction role")
    probability = row["probability"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= float(probability) <= 1.0
    ):
        raise AnalysisError(f"{where}: invalid probability")
    return (
        row["fm_seed"], row["arm"], row["cancer"], row["head_seed"],
        row["patient_id"], row["y_true"], row["race"], row["fold"],
        row["role"], row["outer_fold"], float(probability),
    )


def load_predictions(
    source: Path,
) -> tuple[dict[tuple[int, str, str, int], dict[str, dict]], str, int]:
    """Load a strict JSONL into an in-memory cell/patient index."""
    if not source.is_file() or source.is_symlink() or source.stat().st_size == 0:
        raise AnalysisError(f"input must be a nonempty nonsymlink file: {source}")
    cells: dict[tuple[int, str, str, int], dict[str, dict]] = defaultdict(dict)
    digest = hashlib.sha256()
    row_count = 0
    with source.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                raise AnalysisError(f"line {line}: blank JSONL row")
            try:
                value = json.loads(raw, object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AnalysisError(f"line {line}: invalid JSON: {error}") from error
            seed, arm, cancer, head, patient, y, race, fold, role, outer, p = (
                _validate_row(value, line)
            )
            cell = cells[(seed, arm, cancer, head)]
            patient_rows = cell.setdefault(
                patient,
                {"y_true": y, "race": race, "fold": fold, "scores": {}},
            )
            if (
                patient_rows["y_true"], patient_rows["race"],
                patient_rows["fold"],
            ) != (y, race, fold):
                raise AnalysisError(f"line {line}: within-cell metadata drift")
            score_key = (role, outer)
            if score_key in patient_rows["scores"]:
                raise AnalysisError(f"line {line}: duplicate semantic row")
            patient_rows["scores"][score_key] = p
            row_count += 1
    return dict(cells), digest.hexdigest(), row_count


def validate_complete(
    cells: Mapping[tuple[int, str, str, int], Mapping[str, dict]],
) -> dict[str, Any]:
    expected = {
        (seed, arm, cancer, head)
        for seed in FM_SEEDS for arm in ARMS
        for cancer in CANCERS for head in HEADS
    }
    if set(cells) != expected:
        raise AnalysisError("input is not the exact fixed-final combination matrix")
    references: dict[str, dict[str, tuple[int, str, int]]] = {}
    counts: dict[str, int] = {}
    for seed in FM_SEEDS:
        for cancer in CANCERS:
            cell_keys = [
                (seed, arm, cancer, head) for arm in ARMS for head in HEADS
            ]
            patient_sets = [set(cells[key]) for key in cell_keys]
            if not patient_sets[0] or any(
                patients != patient_sets[0] for patients in patient_sets[1:]
            ):
                raise AnalysisError("patient cohort drift across arms/heads")
            patients = patient_sets[0]
            if len(patients) != COHORT_SIZES[cancer]:
                raise AnalysisError(
                    f"{cancer} cohort has {len(patients)} patients, "
                    f"expected {COHORT_SIZES[cancer]}"
                )
            current: dict[str, tuple[int, str, int]] = {}
            for patient in patients:
                reference = cells[cell_keys[0]][patient]
                metadata = (
                    reference["y_true"], reference["race"], reference["fold"]
                )
                current[patient] = metadata
                expected_scores = {
                    ("outer_test", metadata[2]),
                    *(("inner_calibration", outer) for outer in FOLDS
                      if outer != metadata[2]),
                }
                for key in cell_keys:
                    row = cells[key][patient]
                    if (
                        row["y_true"], row["race"], row["fold"]
                    ) != metadata:
                        raise AnalysisError("metadata drift across arms/heads")
                    if set(row["scores"]) != expected_scores:
                        raise AnalysisError(
                            "patient lacks exact one-outer/four-inner structure"
                        )
            if cancer in references and current != references[cancer]:
                raise AnalysisError("patient cohort or metadata drift across FM seeds")
            references.setdefault(cancer, current)
            counts[f"{seed}:{cancer}"] = len(patients)
    return {
        "combination_count": len(expected),
        "fm_pair_count": len(FM_SEEDS),
        "patient_counts_by_seed_cancer": counts,
    }


def ensemble_rows(
    cells: Mapping[tuple[int, str, str, int], Mapping[str, dict]],
    seed: int,
    arm: str,
    cancer: str,
    heads: Sequence[int],
) -> list[dict[str, Any]]:
    patients = cells[(seed, arm, cancer, heads[0])]
    result = []
    for patient in sorted(patients):
        base = patients[patient]
        for role, outer in sorted(base["scores"]):
            scores = [
                cells[(seed, arm, cancer, head)][patient]["scores"][(role, outer)]
                for head in heads
            ]
            result.append({
                "patient_id": patient,
                "y_true": base["y_true"],
                "race": base["race"],
                "fold": base["fold"],
                "role": role,
                "outer_fold": outer,
                "probability": math.fsum(scores) / len(scores),
            })
    return result


def nested_endpoint(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    outer = [row for row in rows if row["role"] == "outer_test"]
    inner = [row for row in rows if row["role"] == "inner_calibration"]
    scores = np.asarray([row["probability"] for row in outer], dtype=float)
    labels = np.asarray([row["y_true"] for row in outer], dtype=int)
    races = np.asarray([row["race"] for row in outer], dtype=object)
    folds = np.asarray([row["fold"] for row in outer], dtype=int)
    disparities = []
    audit: list[dict[str, Any]] = []
    for specificity in SPECIFICITIES:
        predicted = np.zeros(len(outer), dtype=bool)
        for heldout in FOLDS:
            calibration = np.asarray([
                row["probability"] for row in inner
                if row["outer_fold"] == heldout
                and row["race"] == "White" and row["y_true"] == 0
            ])
            if calibration.size == 0:
                raise AnalysisError("empty White-negative calibration set")
            threshold = float(np.quantile(
                calibration, specificity, method="linear"
            ))
            mask = folds == heldout
            if not np.any(mask):
                raise AnalysisError("empty heldout fold")
            predicted[mask] = scores[mask] >= threshold
            heldout_white_negative = (
                mask & (races == "White") & (labels == 0)
            )
            heldout_count = int(np.sum(heldout_white_negative))
            if heldout_count == 0:
                raise AnalysisError("empty heldout White-negative denominator")
            audit.append({
                "specificity_target": specificity,
                "outer_fold": heldout,
                "threshold": threshold,
                "calibration_white_negative_count": int(calibration.size),
                "heldout_white_negative_count": heldout_count,
                "achieved_heldout_white_specificity": float(
                    1.0 - np.mean(
                        scores[heldout_white_negative] >= threshold
                    )
                ),
            })
        differences = []
        for outcome in (0, 1):
            rates = []
            for race in ("White", "Black"):
                mask = (labels == outcome) & (races == race)
                if not np.any(mask):
                    raise AnalysisError("empty equalized-odds denominator")
                rates.append(float(np.mean(predicted[mask])))
            differences.append(abs(rates[1] - rates[0]))
        disparities.append(max(differences))
    return math.fsum(disparities) / len(disparities), audit


def nested_ieo(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return iEO alone; the fixed-final run also persists nested_endpoint audit."""
    return nested_endpoint(rows)[0]


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive, negative = scores[labels == 1], scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        raise AnalysisError("degenerate AUROC")
    comparisons = [
        float(np.sum(value > negative)) + 0.5 * float(np.sum(value == negative))
        for value in positive
    ]
    return math.fsum(comparisons) / (positive.size * negative.size)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or int(np.sum(labels)) == 0:
        raise AnalysisError("degenerate Black AUPRC")
    order = np.argsort(-scores, kind="mergesort")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    positives = np.cumsum(labels)[ends]
    increments = np.diff(np.r_[0, positives])
    return float(np.sum(positives / (ends + 1) * increments) / positives[-1])


def _ece(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        raise AnalysisError("degenerate Black ECE")
    error = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (
            (scores >= low) & (scores <= high)
            if index == 9 else (scores >= low) & (scores < high)
        )
        if np.any(mask):
            error += abs(float(np.sum(labels[mask] - scores[mask])))
    return error / labels.size


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
    values = list(values)
    if not values:
        raise AnalysisError("cannot average empty values")
    return math.fsum(values) / len(values)


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 48:
        raise AnalysisError("paired inference requires exactly 48 FM seeds")
    mean = _mean(values)
    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / 47)
    se = sd / math.sqrt(48)
    critical = float(stats.t.ppf(0.95, 47))
    if se == 0:
        statistic = 0.0 if mean == 0 else None
        pvalue = 1.0 if mean == 0 else 0.0
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
    report = {}
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
                metric: _mean(cancers[cancer][metric] for cancer in CANCERS)
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


def utility_gate(utility: Mapping[str, Any], arm: str) -> dict[str, Any]:
    overall = utility[arm]["overall"]
    delta = utility[arm]["deltas_vs_B"]["overall"]
    main = {
        "mean_overall_auroc_gt_0.60": overall["overall_auroc"] > 0.60,
        "mean_auroc_delta_ge_-0.02": delta["overall_auroc"] >= -0.02,
        "mean_black_auprc_delta_ge_-0.02": delta["black_auprc"] >= -0.02,
        "mean_black_ece_delta_le_+0.02": delta["black_ece"] <= 0.02,
    }
    cancer_bounds = {}
    for cancer in CANCERS:
        local = utility[arm]["cancers"][cancer]
        change = utility[arm]["deltas_vs_B"]["cancers"][cancer]
        cancer_bounds[cancer] = {
            "auroc_gt_0.57": local["overall_auroc"] > 0.57,
            "auroc_delta_ge_-0.05": change["overall_auroc"] >= -0.05,
            "black_auprc_delta_ge_-0.05": change["black_auprc"] >= -0.05,
            "black_ece_delta_le_+0.05": change["black_ece"] <= 0.05,
        }
    passed = all(main.values()) and all(
        all(values.values()) for values in cancer_bounds.values()
    )
    return {"pass": passed, "mean_bounds": main, "cancer_bounds": cancer_bounds}


def analyze(
    cells: Mapping[tuple[int, str, str, int], Mapping[str, dict]],
) -> dict[str, Any]:
    ieo, utility_values = {}, {}
    nested_audit: list[dict[str, Any]] = []
    for seed in FM_SEEDS:
        for arm in ARMS:
            for cancer in CANCERS:
                rows = ensemble_rows(cells, seed, arm, cancer, HEADS)
                endpoint, audit = nested_endpoint(rows)
                ieo[(seed, arm, cancer)] = endpoint
                nested_audit.extend({
                    "fm_seed": seed,
                    "arm": arm,
                    "cancer": cancer,
                    **entry,
                } for entry in audit)
                utility_values[(seed, arm, cancer)] = utility_metrics(rows)
    cancer_vectors = {
        cancer: [
            ieo[(seed, "P", cancer)] - ieo[(seed, "H", cancer)]
            for seed in FM_SEEDS
        ]
        for cancer in CANCERS
    }
    full_vector = [
        _mean(cancer_vectors[cancer][index] for cancer in CANCERS)
        for index in range(48)
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
        vector = []
        for seed in FM_SEEDS:
            vector.append(_mean(
                nested_ieo(ensemble_rows(cells, seed, "P", cancer, heads))
                - nested_ieo(ensemble_rows(cells, seed, "H", cancer, heads))
                for cancer in CANCERS
            ))
        halves.append({
            "head_seeds": list(heads),
            "per_seed_theta": vector,
            "mean": _mean(vector),
        })
    favored = "H" if full["mean"] > 0 else "P" if full["mean"] < 0 else None
    sign = 1 if favored == "H" else -1 if favored == "P" else 0
    head_direction = sign != 0 and all(
        sign * half["mean"] > 0 for half in halves
    )
    cancer_direction = sign != 0 and all(
        sign * cancers[cancer]["mean"] > 0 for cancer in CANCERS
    )
    harm_values = (
        {
            cancer: _mean(
                ieo[(seed, favored, cancer)] - ieo[(seed, "B", cancer)]
                for seed in FM_SEEDS
            )
            for cancer in CANCERS
        }
        if favored is not None else {}
    )
    harm_pass = bool(favored) and all(value <= 0.03 for value in harm_values.values())
    harm_gate = {
        "favored_arm": favored,
        "mean_ieo_delta_vs_B_by_cancer": harm_values,
        "threshold_le": 0.03,
        "pass": harm_pass,
    }
    utilities = _utility_report(utility_values)
    selected_utility = (
        utility_gate(utilities, favored) if favored is not None
        else {"pass": False, "mean_bounds": {}, "cancer_bounds": {}}
    )
    equivalence = (
        full["ci90"][0] > -EQUIVALENCE_MARGIN
        and full["ci90"][1] < EQUIVALENCE_MARGIN
    )
    superiority = {
        "paired_p_lt_0.05": full["two_sided_p"] < ALPHA,
        "absolute_mean_ge_0.02": abs(full["mean"]) >= MATERIALITY,
        "both_head_halves_same_strict_direction": head_direction,
        "both_cancers_same_strict_direction": cancer_direction,
        "favored_arm_harm_gate": harm_pass,
        "favored_arm_utility_gate": bool(selected_utility["pass"]),
    }
    qualified = all(superiority.values())
    classification = (
        "equivalent" if equivalence
        else f"{favored}_superior" if qualified and favored is not None
        else "inconclusive"
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
            "equivalence_ci90_strictly_inside_+/-0.03": equivalence,
            "superiority": superiority,
        },
        "decision": {
            "classification": classification,
            "favored_arm": favored,
            "equivalence_precedence": equivalence,
            "positive_theta_favors": "H",
            "theta_definition": "iEO(P)-iEO(H)",
        },
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
        "head_halves": [list(half) for half in HEAD_HALVES],
        "folds": list(FOLDS),
        "specificity_targets": list(SPECIFICITIES),
        "probability_ensemble_before_thresholds": True,
        "theta": "iEO(P)-iEO(H); positive favors H",
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "materiality": MATERIALITY,
        "alpha": ALPHA,
    }


def run(source: Path) -> dict[str, Any]:
    cells, digest, row_count = load_predictions(source)
    counts = validate_complete(cells)
    expected_row_count = (
        len(FM_SEEDS) * len(ARMS) * len(HEADS) * 5
        * sum(COHORT_SIZES.values())
    )
    if row_count != expected_row_count:
        raise AnalysisError(
            f"fixed matrix has {row_count} rows, expected {expected_row_count}"
        )
    semantic = {
        "input_sha256": digest,
        "row_count": row_count,
        "contract": contract_report(),
        "counts": counts,
        **analyze(cells),
    }
    return {"schema": REPORT_SCHEMA, "semantic_report": semantic}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("predictions", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args.predictions)
        payload = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
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
