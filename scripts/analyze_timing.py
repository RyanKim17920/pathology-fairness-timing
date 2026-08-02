#!/usr/bin/env python3
"""Compare head-matched control, pretraining, and post-hoc prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


SCHEMA = "pathology-fairness-timing-analysis/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_predictions(path: Path) -> dict[str, dict]:
    records = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            patient = str(record.get("patient_id", "")).strip()
            if not patient:
                raise ValueError(f"{path}:{line_number}: missing patient_id")
            if patient in records:
                raise ValueError(f"{path}:{line_number}: duplicate patient {patient}")
            label = int(record["y_true"])
            score = float(record["y_score"])
            if label not in (0, 1) or not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: invalid label or score")
            records[patient] = {**record, "y_true": label, "y_score": score}
    if not records:
        raise ValueError(f"{path}: no prediction records")
    return records


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = y == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(score)
    statistic = ranks[positives].sum() - n_pos * (n_pos + 1) / 2
    return float(statistic / (n_pos * n_neg))


def eligible_groups(y: np.ndarray, groups: np.ndarray, min_group_n: int,
                    min_class_n: int) -> list[str]:
    eligible = []
    for group in sorted({str(value) for value in groups if value is not None}):
        selected = groups == group
        yy = y[selected]
        if (len(yy) >= min_group_n and int((yy == 0).sum()) >= min_class_n
                and int((yy == 1).sum()) >= min_class_n):
            eligible.append(group)
    return eligible


def metrics(y: np.ndarray, scores: np.ndarray, groups: np.ndarray,
            included_groups: list[str]) -> dict[str, float]:
    overall = auc(y, scores)
    subgroup = {
        group: auc(y[groups == group], scores[groups == group])
        for group in included_groups
    }
    if overall is None or any(value is None for value in subgroup.values()):
        raise ValueError("a resample has insufficient class support for AUROC")
    values = list(subgroup.values())
    return {
        "overall_auc": overall,
        "auc_gap": float(max(values) - min(values)),
        "subgroup_auc": subgroup,
    }


def _group_value(record: dict, sensitive: str, age_cutoff: float) -> str | None:
    if sensitive != "age":
        value = record.get(sensitive)
        return None if value in (None, "") else str(value)
    value = record.get("age")
    if value in (None, ""):
        return None
    age = float(value)
    return f"age<{age_cutoff:g}" if age < age_cutoff else f"age>={age_cutoff:g}"


def load_arms(control_paths: list[Path], pretraining_paths: list[Path],
              posthoc_paths: list[Path], sensitive: str,
              age_cutoff: float) -> dict:
    if not (len(control_paths) == len(pretraining_paths) == len(posthoc_paths)):
        raise ValueError("control, pretraining, and posthoc must have equal run counts")
    if not control_paths:
        raise ValueError("at least one run per arm is required")
    path_sets = {
        "control": control_paths,
        "pretraining": pretraining_paths,
        "posthoc": posthoc_paths,
    }
    loaded = {arm: [read_predictions(path) for path in paths]
              for arm, paths in path_sets.items()}
    patients = sorted(loaded["control"][0])
    expected = set(patients)
    reference = loaded["control"][0]
    for arm, runs in loaded.items():
        for run_index, run in enumerate(runs):
            if set(run) != expected:
                raise ValueError(f"{arm} run {run_index}: patient set does not match")
            for patient in patients:
                if run[patient]["y_true"] != reference[patient]["y_true"]:
                    raise ValueError(f"{arm} run {run_index}: label mismatch for {patient}")
                if (_group_value(run[patient], sensitive, age_cutoff)
                        != _group_value(reference[patient], sensitive, age_cutoff)):
                    raise ValueError(f"{arm} run {run_index}: subgroup mismatch for {patient}")
    return {
        "patients": patients,
        "y": np.asarray([reference[patient]["y_true"] for patient in patients]),
        "groups": np.asarray([
            _group_value(reference[patient], sensitive, age_cutoff)
            for patient in patients
        ], dtype=object),
        "scores": {
            arm: np.asarray([
                [run[patient]["y_score"] for patient in patients]
                for run in runs
            ], dtype=float)
            for arm, runs in loaded.items()
        },
        "inputs": {
            arm: [
                {"file": path.name, "sha256": sha256_file(path),
                 "bytes": path.stat().st_size}
                for path in paths
            ]
            for arm, paths in path_sets.items()
        },
    }


def contrasts(arm_metrics: dict[str, dict[str, float]]) -> dict:
    control = arm_metrics["control"]
    pretraining = arm_metrics["pretraining"]
    posthoc = arm_metrics["posthoc"]
    return {
        "pretraining_vs_control": {
            "fairness_improvement": control["auc_gap"] - pretraining["auc_gap"],
            "utility_delta": pretraining["overall_auc"] - control["overall_auc"],
        },
        "posthoc_vs_control": {
            "fairness_improvement": control["auc_gap"] - posthoc["auc_gap"],
            "utility_delta": posthoc["overall_auc"] - control["overall_auc"],
        },
        "pretraining_vs_posthoc": {
            "fairness_advantage": posthoc["auc_gap"] - pretraining["auc_gap"],
            "utility_delta": pretraining["overall_auc"] - posthoc["overall_auc"],
        },
    }


def _run_summary(y, groups, scores, included_groups):
    per_run = []
    for run_index in range(scores["control"].shape[0]):
        arm_metrics = {
            arm: metrics(y, values[run_index], groups, included_groups)
            for arm, values in scores.items()
        }
        per_run.append({"arms": arm_metrics, "contrasts": contrasts(arm_metrics)})
    return per_run


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for label in (0, 1):
        members = np.flatnonzero(y == label)
        selected.extend(rng.choice(members, size=len(members), replace=True))
    return np.asarray(selected, dtype=int)


def bootstrap_timing(y, groups, scores, included_groups, iterations, seed):
    rng = np.random.default_rng(seed)
    samples = []
    run_count = scores["control"].shape[0]
    for _ in range(iterations):
        patient_indices = _stratified_indices(y, rng)
        run_indices = rng.choice(run_count, size=run_count, replace=True)
        run_contrasts = []
        try:
            for run_index in run_indices:
                arm_metrics = {
                    arm: metrics(
                        y[patient_indices], values[run_index, patient_indices],
                        groups[patient_indices], included_groups,
                    )
                    for arm, values in scores.items()
                }
                run_contrasts.append(contrasts(arm_metrics)["pretraining_vs_posthoc"])
        except ValueError:
            continue
        samples.append({
            key: float(np.median([row[key] for row in run_contrasts]))
            for key in ("fairness_advantage", "utility_delta")
        })
    if len(samples) < max(100, iterations // 2):
        raise ValueError("too few valid bootstrap resamples")
    output = {"valid_iterations": len(samples), "requested_iterations": iterations}
    for key in ("fairness_advantage", "utility_delta"):
        values = np.asarray([sample[key] for sample in samples])
        output[key] = {
            "ci_95": [float(np.quantile(values, 0.025)),
                      float(np.quantile(values, 0.975))],
            "probability_le_zero": float(np.mean(values <= 0)),
        }
    return output


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_runtime_contract(path: Path, sensitive: str, run_count: int) -> dict:
    declaration = json.loads(path.read_text())
    if declaration.get("primary_estimand") != \
            "posthoc_auc_gap_minus_pretraining_auc_gap":
        raise ValueError("runtime contract primary estimand does not match analyzer")
    if declaration.get("sensitive") != sensitive:
        raise ValueError("runtime contract sensitive attribute does not match command")
    planned_seeds = declaration.get("planned_head_seeds")
    if not isinstance(planned_seeds, list) or len(planned_seeds) != run_count:
        raise ValueError("prediction run count does not match planned_head_seeds")
    return declaration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, nargs="+", required=True)
    parser.add_argument("--pretraining", type=Path, nargs="+", required=True)
    parser.add_argument("--posthoc", type=Path, nargs="+", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--sensitive", choices=["race", "sex", "age"], default="race")
    parser.add_argument("--age-cutoff", type=float, default=65.0)
    parser.add_argument("--min-group-n", type=int, default=15)
    parser.add_argument("--min-class-n", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    data = load_arms(
        args.control, args.pretraining, args.posthoc,
        args.sensitive, args.age_cutoff,
    )
    runtime_contract = load_runtime_contract(
        args.runtime_contract, args.sensitive, len(args.control)
    )
    min_fairness_advantage = float(
        runtime_contract.get("min_fairness_advantage", 0.0)
    )
    max_utility_loss = float(runtime_contract["utility_noninferiority_margin"])
    if min_fairness_advantage < 0 or max_utility_loss < 0:
        raise ValueError("runtime contract decision thresholds must be non-negative")
    groups = eligible_groups(
        data["y"], data["groups"], args.min_group_n, args.min_class_n
    )
    if len(groups) < 2:
        raise ValueError("fewer than two subgroups meet the declared support thresholds")
    per_run = _run_summary(data["y"], data["groups"], data["scores"], groups)
    point = {
        key: float(np.median([
            run["contrasts"]["pretraining_vs_posthoc"][key] for run in per_run
        ]))
        for key in ("fairness_advantage", "utility_delta")
    }
    bootstrap = bootstrap_timing(
        data["y"], data["groups"], data["scores"], groups,
        args.bootstrap, args.seed,
    )
    fairness_lower = bootstrap["fairness_advantage"]["ci_95"][0]
    utility_lower = bootstrap["utility_delta"]["ci_95"][0]
    result = {
        "schema": SCHEMA,
        "estimand": {
            "primary": "posthoc_auc_gap_minus_pretraining_auc_gap",
            "interpretation": "positive values favor pretraining",
            "utility": "pretraining_overall_auc_minus_posthoc_overall_auc",
            "multiplicity": "single_primary_estimand_no_adjustment",
        },
        "design": {
            "paired_patients": True,
            "run_count": len(args.control),
            "patient_count": len(data["patients"]),
            "sensitive": args.sensitive,
            "eligible_groups": groups,
            "min_group_n": args.min_group_n,
            "min_class_n": args.min_class_n,
            "age_cutoff": args.age_cutoff if args.sensitive == "age" else None,
            "bootstrap_seed": args.seed,
        },
        "inputs": data["inputs"],
        "runtime_contract": {
            "file": args.runtime_contract.name,
            "sha256": sha256_file(args.runtime_contract),
            "declaration": runtime_contract,
        },
        "per_run": per_run,
        "timing_point_estimate": point,
        "bootstrap": bootstrap,
        "decision_thresholds": {
            "min_fairness_advantage": min_fairness_advantage,
            "max_utility_loss": max_utility_loss,
        },
        "decision": {
            "fairness_superiority": fairness_lower > min_fairness_advantage,
            "utility_noninferiority": utility_lower >= -max_utility_loss,
            "pretraining_preferred": (
                fairness_lower > min_fairness_advantage
                and utility_lower >= -max_utility_loss
            ),
        },
    }
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
