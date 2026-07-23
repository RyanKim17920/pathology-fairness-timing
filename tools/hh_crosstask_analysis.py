#!/usr/bin/env python3
"""Aggregate M5/M6 hospital-holdout fairness results across BRCA and LUAD.

The analysis uses ``hh_metrics.py`` as its metric engine.  BRCA's three
hospital folds are concatenated into a temporary JSONL before evaluation;
LUAD uses its single target JSONL.  Missing prediction sets are recorded as
pending rather than treated as fatal.

Default outputs:
  * results/hh_crosstask_results.json
  * results/hh_crosstask_table.md

Example:
    python tools/hh_crosstask_analysis.py --boot-n 10000
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = str(Path(__file__).resolve().parent)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
DEFAULT_PREDS_DIR = Path("/data/ryan.kim/nanopath/results/preds")
DEFAULT_JSON_OUT = REPO_ROOT / "results/hh_crosstask_results.json"
DEFAULT_MARKDOWN_OUT = REPO_ROOT / "results/hh_crosstask_table.md"
DEFAULT_BOOT_N = 10_000
DEFAULT_SEED = 20_260_716

BLACK_EVENT_FLOOR = 15
BLACK_AUPRC_TOLERANCE = 0.02
BLACK_ECE_TOLERANCE = 0.02
OVERALL_AUROC_FLOOR = 0.60

TASKS = {
    "brca": {
        "label": "BRCA-TP53",
        "pred_task": "brca_tp53",
        "suffixes": ("F1", "F2", "F3"),
        "folds_csv": REPO_ROOT / "data/metadata/brca_hospital_folds.csv",
    },
    "luad": {
        "label": "LUAD-TP53",
        "pred_task": "luad_tp53",
        "suffixes": ("target",),
        "folds_csv": REPO_ROOT / "data/metadata/luad_hospital_folds.csv",
    },
}

BAKE_IN_FMS = (
    "m3_contrastive_cancer",
    "m3_contrastive_demo_race",
    "m3_contrastive_demo_racesexage",
    "m3_twocond_race",
    "m3_twocond_racesexage",
    "m3_fino_race",
    "m3_fino_racesexage",
    "m3_dann_race",
    "m3_dann_racesexage",
    "m3_pcgrad_race",
    "m3_pcgrad_racesexage",
)

POSTHOC_ARMS = (
    "contrastive_marginal",
    "contrastive_labelcond",
    "dann_marginal",
    "dann_labelcond",
    "fino_marginal",
    "fino_labelcond",
    "pcgrad_marginal",
    "pcgrad_labelcond",
)

FAMILY_BAKE_IN = {
    "contrastive": (
        "m3_contrastive_cancer",
        "m3_contrastive_demo_race",
        "m3_contrastive_demo_racesexage",
        "m3_twocond_race",
        "m3_twocond_racesexage",
    ),
    "dann": ("m3_dann_race", "m3_dann_racesexage"),
    "fino": ("m3_fino_race", "m3_fino_racesexage"),
    "pcgrad": ("m3_pcgrad_race", "m3_pcgrad_racesexage"),
}


def _prediction_sets() -> list[dict[str, Any]]:
    sets = [{
        "id": "baseline",
        "stage": "baseline",
        "family": "plain",
        "variant": "plain FM",
        "file_stem": "hh_m2_baseline",
    }]
    for fm in BAKE_IN_FMS:
        family = next(k for k, values in FAMILY_BAKE_IN.items() if fm in values)
        sets.append({
            "id": fm,
            "stage": "bake_in",
            "family": family,
            "variant": fm.removeprefix("m3_"),
            "file_stem": f"hh_m5_{fm}",
        })
    for arm in POSTHOC_ARMS:
        family, variant = arm.rsplit("_", 1)
        sets.append({
            "id": arm,
            "stage": "post_hoc",
            "family": family,
            "variant": variant,
            "file_stem": f"hh_m4_{arm}",
        })
    return sets


PREDICTION_SETS = _prediction_sets()
PREDICTION_SET_BY_ID = {item["id"]: item for item in PREDICTION_SETS}


def expected_prediction_paths(
    pred_set: dict[str, Any], task: str, preds_dir: Path
) -> list[Path]:
    """Return the exact expected JSONL paths for one prediction set/task."""
    cfg = TASKS[task]
    return [
        preds_dir / (
            f"{pred_set['file_stem']}__{cfg['pred_task']}__{suffix}.jsonl"
        )
        for suffix in cfg["suffixes"]
    ]


def inventory_prediction_sets(preds_dir: Path) -> list[dict[str, Any]]:
    """Inventory all 20 rows by task, distinguishing missing from incomplete."""
    inventory = []
    for pred_set in PREDICTION_SETS:
        tasks: dict[str, Any] = {}
        for task in TASKS:
            expected = expected_prediction_paths(pred_set, task, preds_dir)
            present = [path for path in expected if path.is_file()]
            missing = [path for path in expected if not path.is_file()]
            status = (
                "present" if not missing
                else "incomplete" if present
                else "missing"
            )
            tasks[task] = {
                "status": status,
                "expected_files": [str(path) for path in expected],
                "present_files": [str(path) for path in present],
                "missing_files": [str(path) for path in missing],
            }
        inventory.append({
            "id": pred_set["id"],
            "stage": pred_set["stage"],
            "family": pred_set["family"],
            "tasks": tasks,
        })
    return inventory


@contextmanager
def _evaluation_prediction_path(
    pred_paths: Sequence[Path], task: str
) -> Iterator[Path]:
    """Yield one engine input path, pooling BRCA folds in a temporary JSONL."""
    if task != "brca":
        yield pred_paths[0]
        return

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix="hh_brca_pooled_", suffix=".jsonl", delete=False
        ) as pooled:
            tmp_path = Path(pooled.name)
            for source in pred_paths:
                payload = source.read_bytes()
                pooled.write(payload)
                if payload and not payload.endswith(b"\n"):
                    pooled.write(b"\n")
        yield tmp_path
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _finite_or_none(value: Any) -> Any:
    """Recursively convert NumPy scalars/non-finite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(k): _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Dependency-free AUROC via average ranks, with correct tie handling."""
    n = len(labels)
    n_pos = sum(int(y) for y in labels)
    n_neg = n - n_pos
    if not n or not n_pos or not n_neg:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    rank_sum_pos = 0.0
    start = 0
    while start < n:
        end = start + 1
        while end < n and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum_pos += average_rank * sum(labels[order[i]] for i in range(start, end))
        start = end
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _overall_auroc(pred_paths: Sequence[Path]) -> tuple[float | None, int]:
    """Compute overall task AUROC from all unique prediction rows/all races."""
    seen: set[str] = set()
    labels: list[int] = []
    scores: list[float] = []
    for path in pred_paths:
        for raw_line in path.read_text().splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            patient_id = str(
                row.get("patient_id") or row.get("patient_barcode") or ""
            )
            if not patient_id or patient_id in seen:
                continue
            seen.add(patient_id)
            label = row["y_true"] if "y_true" in row else row["label"]
            score = row["y_score"] if "y_score" in row else row["score"]
            labels.append(int(float(label)))
            scores.append(float(score))
    return _binary_auroc(labels, scores), len(labels)


def _ci_block(metric_module: Any, ci: dict[str, Any], key: str) -> dict[str, Any]:
    row = ci[key]
    return {
        "point": row["point"],
        "percentile_ci": [row["pct_lo"], row["pct_hi"]],
        "bca_ci": [row["bca_lo"], row["bca_hi"]],
        "centering_percentile": metric_module.centering_fraction(
            row["point"], row["pct_lo"], row["pct_hi"]
        ),
        "centering_bca": metric_module.centering_fraction(
            row["point"], row["bca_lo"], row["bca_hi"]
        ),
        "bootstrap_n_used": row["n_used"],
    }


def _run_direct(
    metric_module: Any,
    engine_pred_path: Path,
    source_pred_paths: Sequence[Path],
    task: str,
    boot_n: int,
    seed: int,
) -> dict[str, Any]:
    cfg = TASKS[task]
    patients, join_stats = metric_module.load_and_join_sensitive(
        [str(engine_pred_path)], str(cfg["folds_csv"]), "race"
    )
    if not patients:
        raise ValueError("hh_metrics kept zero White/Black patients")
    point = metric_module.compute_metrics(patients, "White", "Black")
    # Calibration is a point-only guardrail here, not a bootstrap CI target.
    # Avoid thousands of irrelevant logistic fits while retaining hh_metrics'
    # exact statistic, resampling, jackknife, and BCa implementations.
    original_calibration = metric_module._calibration
    metric_module._calibration = lambda _y, _s: (
        metric_module.NAN, metric_module.NAN, metric_module.NAN
    )
    try:
        ci = metric_module.bootstrap_ci(
            patients,
            metric_keys=("fpr_disparity", "tpr_disparity", "eo", "auroc_gap"),
            n_boot=boot_n,
            seed=seed,
            reference_group="White",
            minority_group="Black",
        )
    finally:
        metric_module._calibration = original_calibration
    overall_auroc, overall_n = _overall_auroc(source_pred_paths)
    result = {
        "status": "complete",
        "engine": "hh_metrics_direct_import",
        "source_files": [str(path) for path in source_pred_paths],
        "brca_folds_pooled_via_tempfile": task == "brca",
        "folds_csv": str(cfg["folds_csv"]),
        "bootstrap": {
            "requested": boot_n,
            "seed": seed,
            "effective_draws": ci["_n_boot_effective"],
            "tss_clusters": ci["_n_clusters"],
            "method": "TSS-cluster bootstrap; percentile and BCa 95% CIs",
            "point_only_metrics_excluded_from_bootstrap": [
                "calibration slope",
                "calibration intercept",
                "ECE",
            ],
        },
        "join_stats": join_stats,
        "threshold": {
            "target_white_specificity": metric_module.TARGET_SPEC,
            "tau": point["tau"],
        },
        "metrics": {
            "signed_fpr_disparity": _ci_block(
                metric_module, ci, "fpr_disparity"
            ),
            "signed_tpr_disparity": _ci_block(
                metric_module, ci, "tpr_disparity"
            ),
            "eo_max": _ci_block(metric_module, ci, "eo"),
            "auroc_gap_white_minus_black": _ci_block(
                metric_module, ci, "auroc_gap"
            ),
            "overall_auroc_all_races": overall_auroc,
            "overall_n_all_races": overall_n,
        },
        "groups": {
            "White": {
                "n": point["n_white"],
                "events": point["white_events"],
                "auroc": point["white_auroc"],
                "auprc": point["white_auprc"],
                "ppv": point["white_ppv"],
                "ece": point["white_cal_ece"],
                "calibration_slope": point["white_cal_slope"],
                "calibration_intercept": point["white_cal_intercept"],
                "fpr": point["white_fpr"],
                "tpr": point["white_tpr"],
            },
            "Black": {
                "n": point["n_black"],
                "events": point["black_events"],
                "auroc": point["black_auroc"],
                "auprc": point["black_auprc"],
                "ppv": point["black_ppv"],
                "ece": point["black_cal_ece"],
                "calibration_slope": point["black_cal_slope"],
                "calibration_intercept": point["black_cal_intercept"],
                "fpr": point["black_fpr"],
                "tpr": point["black_tpr"],
            },
        },
        "underpowered_black_events": point["black_events"] < BLACK_EVENT_FLOOR,
    }
    return _finite_or_none(result)


def _parse_cli_number(token: str) -> float | None:
    token = token.strip()
    if token.lower() in {"n/a", "nan", "none"}:
        return None
    return float(token)


_METRIC_LINE = re.compile(
    r"^(signed FPR-disp|signed TPR-disp|EO \(max\)|AUROC-gap)\s+"
    r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)|n/a)\s+"
    r"\[\s*([^,\]]+),\s*([^\]]+)\]\s+"
    r"\[\s*([^,\]]+),\s*([^\]]+)\]\s+"
    r"(\S+)\s+(\S+)\s+(\d+)\s*$"
)
_GROUP_LINE = re.compile(
    r"^(White|Black)\s+(\d+)\s+(\d+)\s+"
    r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)


def _parse_hh_metrics_stdout(
    stdout: str,
    source_pred_paths: Sequence[Path],
    task: str,
    boot_n: int,
    seed: int,
) -> dict[str, Any]:
    """Robust fallback parser for hh_metrics.py's labeled CLI tables."""
    metric_names = {
        "signed FPR-disp": "signed_fpr_disparity",
        "signed TPR-disp": "signed_tpr_disparity",
        "EO (max)": "eo_max",
        "AUROC-gap": "auroc_gap_white_minus_black",
    }
    metrics: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        metric_match = _METRIC_LINE.match(stripped)
        if metric_match:
            (
                label, point, pct_lo, pct_hi, bca_lo, bca_hi,
                center_pct, center_bca, n_used,
            ) = metric_match.groups()
            metrics[metric_names[label]] = {
                "point": _parse_cli_number(point),
                "percentile_ci": [
                    _parse_cli_number(pct_lo), _parse_cli_number(pct_hi)
                ],
                "bca_ci": [
                    _parse_cli_number(bca_lo), _parse_cli_number(bca_hi)
                ],
                "centering_percentile": _parse_cli_number(center_pct),
                "centering_bca": _parse_cli_number(center_bca),
                "bootstrap_n_used": int(n_used),
            }
            continue
        group_match = _GROUP_LINE.match(stripped)
        if group_match:
            (
                group, n, events, auroc, auprc, ppv, slope, intercept, ece
            ) = group_match.groups()
            groups[group] = {
                "n": int(n),
                "events": int(events),
                "auroc": _parse_cli_number(auroc),
                "auprc": _parse_cli_number(auprc),
                "ppv": _parse_cli_number(ppv),
                "ece": _parse_cli_number(ece),
                "calibration_slope": _parse_cli_number(slope),
                "calibration_intercept": _parse_cli_number(intercept),
            }
    missing_metrics = sorted(set(metric_names.values()) - set(metrics))
    missing_groups = sorted({"White", "Black"} - set(groups))
    if missing_metrics or missing_groups:
        raise ValueError(
            "could not parse hh_metrics stdout: "
            f"missing metrics={missing_metrics}, groups={missing_groups}"
        )
    overall_auroc, overall_n = _overall_auroc(source_pred_paths)
    metrics["overall_auroc_all_races"] = overall_auroc
    metrics["overall_n_all_races"] = overall_n
    return {
        "status": "complete",
        "engine": "hh_metrics_subprocess_stdout",
        "source_files": [str(path) for path in source_pred_paths],
        "brca_folds_pooled_via_tempfile": task == "brca",
        "folds_csv": str(TASKS[task]["folds_csv"]),
        "bootstrap": {
            "requested": boot_n,
            "seed": seed,
            "method": "TSS-cluster bootstrap; parsed CLI output",
        },
        "metrics": metrics,
        "groups": groups,
        "underpowered_black_events": groups["Black"]["events"] < BLACK_EVENT_FLOOR,
    }


def _run_subprocess(
    engine_pred_path: Path,
    source_pred_paths: Sequence[Path],
    task: str,
    boot_n: int,
    seed: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/hh_metrics.py"),
        "--preds", str(engine_pred_path),
        "--folds-csv", str(TASKS[task]["folds_csv"]),
        "--sensitive-axis", "race",
        "--boot-n", str(boot_n),
        "--seed", str(seed),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return _parse_hh_metrics_stdout(
        completed.stdout, source_pred_paths, task, boot_n, seed
    )


def run_hh_metrics(
    pred_paths: Sequence[str | Path],
    task: str,
    boot_n: int = DEFAULT_BOOT_N,
    seed: int = DEFAULT_SEED,
    force_subprocess: bool = False,
) -> dict[str, Any]:
    """Run/parse hh_metrics for one complete prediction set and task.

    Direct import is preferred.  If hh_metrics cannot be imported, its CLI is
    invoked and the labeled stdout tables are parsed.  BRCA inputs must contain
    exactly F1/F2/F3 and are pooled through a temporary file.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {tuple(TASKS)}")
    paths = [Path(path) for path in pred_paths]
    expected_count = len(TASKS[task]["suffixes"])
    if len(paths) != expected_count:
        raise ValueError(
            f"{task} requires {expected_count} prediction file(s), got {len(paths)}"
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prediction files: {missing}")
    if boot_n < 1:
        raise ValueError("boot_n must be >= 1")

    metric_module = None
    if not force_subprocess:
        try:
            metric_module = importlib.import_module("hh_metrics")
        except ImportError:
            metric_module = None

    with _evaluation_prediction_path(paths, task) as engine_path:
        if metric_module is not None:
            return _run_direct(
                metric_module, engine_path, paths, task, boot_n, seed
            )
        return _run_subprocess(engine_path, paths, task, boot_n, seed)


def _complete(task_result: dict[str, Any] | None) -> bool:
    return bool(task_result and task_result.get("status") == "complete")


def _point(task_result: dict[str, Any], metric: str) -> float | None:
    return task_result["metrics"][metric]["point"]


def _signed_delta_and_reduction(
    arm_result: dict[str, Any], baseline_result: dict[str, Any]
) -> tuple[float | None, float | None]:
    arm = _point(arm_result, "signed_fpr_disparity")
    baseline = _point(baseline_result, "signed_fpr_disparity")
    if arm is None or baseline is None:
        return None, None
    return arm - baseline, abs(baseline) - abs(arm)


def add_baseline_comparisons(results: dict[str, Any]) -> None:
    """Attach disparity deltas and the frozen guardrails to complete arms."""
    for task in TASKS:
        baseline = results["baseline"]["tasks"][task]
        if not _complete(baseline):
            continue
        baseline["baseline_comparison"] = {
            "is_baseline": True,
            "delta_signed_fpr_disparity": 0.0,
            "absolute_fpr_disparity_reduction": 0.0,
        }
        baseline["guardrails"] = {
            "status": "reference",
            "failed": False,
            "reasons": [],
        }
        for pred_set in PREDICTION_SETS[1:]:
            arm = results[pred_set["id"]]["tasks"][task]
            if not _complete(arm):
                continue
            signed_delta, reduction = _signed_delta_and_reduction(arm, baseline)
            black_auprc_delta = (
                arm["groups"]["Black"]["auprc"]
                - baseline["groups"]["Black"]["auprc"]
                if arm["groups"]["Black"]["auprc"] is not None
                and baseline["groups"]["Black"]["auprc"] is not None
                else None
            )
            black_ece_delta = (
                arm["groups"]["Black"]["ece"]
                - baseline["groups"]["Black"]["ece"]
                if arm["groups"]["Black"]["ece"] is not None
                and baseline["groups"]["Black"]["ece"] is not None
                else None
            )
            overall_delta = (
                arm["metrics"]["overall_auroc_all_races"]
                - baseline["metrics"]["overall_auroc_all_races"]
                if arm["metrics"]["overall_auroc_all_races"] is not None
                and baseline["metrics"]["overall_auroc_all_races"] is not None
                else None
            )
            reasons = []
            if (
                black_auprc_delta is not None
                and black_auprc_delta < -BLACK_AUPRC_TOLERANCE
            ):
                reasons.append(
                    "Black AUPRC fell >0.02 absolute from baseline"
                )
            if (
                black_ece_delta is not None
                and black_ece_delta > BLACK_ECE_TOLERANCE
            ):
                reasons.append(
                    "Black ECE worsened >0.02 absolute from baseline"
                )
            overall_auroc = arm["metrics"]["overall_auroc_all_races"]
            if overall_auroc is not None and overall_auroc <= OVERALL_AUROC_FLOOR:
                reasons.append("overall task AUROC <=0.60")
            arm["baseline_comparison"] = {
                "is_baseline": False,
                "delta_signed_fpr_disparity": signed_delta,
                "absolute_fpr_disparity_reduction": reduction,
                "black_auprc_delta": black_auprc_delta,
                "black_ece_delta": black_ece_delta,
                "overall_auroc_delta": overall_delta,
            }
            arm["guardrails"] = {
                "status": "fail" if reasons else "pass",
                "failed": bool(reasons),
                "reasons": reasons,
            }


def build_cross_task_generalization(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for fm in BAKE_IN_FMS:
        tasks: dict[str, Any] = {}
        improvements: dict[str, bool] = {}
        for task in TASKS:
            arm = results[fm]["tasks"][task]
            baseline = results["baseline"]["tasks"][task]
            if not (_complete(arm) and _complete(baseline)):
                tasks[task] = {"status": "pending"}
                continue
            signed_delta, reduction = _signed_delta_and_reduction(arm, baseline)
            tasks[task] = {
                "status": "complete",
                "fpr_disparity": _point(arm, "signed_fpr_disparity"),
                "baseline_fpr_disparity": _point(
                    baseline, "signed_fpr_disparity"
                ),
                "delta_signed_from_baseline": signed_delta,
                "absolute_disparity_reduction": reduction,
            }
            improvements[task] = bool(reduction is not None and reduction > 0)
        if len(improvements) < len(TASKS):
            verdict = "pending"
        elif all(improvements.values()):
            verdict = "transfers_to_both_tasks"
        elif improvements.get("brca") and not improvements.get("luad"):
            verdict = "BRCA_only"
        elif improvements.get("luad") and not improvements.get("brca"):
            verdict = "LUAD_only"
        else:
            verdict = "neither_task"
        rows.append({"fm": fm, "tasks": tasks, "verdict": verdict})
    return rows


def build_head_to_head(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        baseline = results["baseline"]["tasks"][task]
        for family, bake_ids in FAMILY_BAKE_IN.items():
            bake_in = []
            for arm_id in bake_ids:
                arm = results[arm_id]["tasks"][task]
                if _complete(arm) and _complete(baseline):
                    delta, reduction = _signed_delta_and_reduction(arm, baseline)
                    bake_in.append({
                        "arm": arm_id,
                        "delta_signed_from_baseline": delta,
                        "absolute_disparity_reduction": reduction,
                    })
                else:
                    bake_in.append({"arm": arm_id, "status": "pending"})
            posthoc: dict[str, Any] = {}
            for variant in ("marginal", "labelcond"):
                arm_id = f"{family}_{variant}"
                arm = results[arm_id]["tasks"][task]
                if _complete(arm) and _complete(baseline):
                    delta, reduction = _signed_delta_and_reduction(arm, baseline)
                    posthoc[variant] = {
                        "arm": arm_id,
                        "delta_signed_from_baseline": delta,
                        "absolute_disparity_reduction": reduction,
                    }
                else:
                    posthoc[variant] = {
                        "arm": arm_id,
                        "status": "pending",
                    }
            marginal_reduction = posthoc["marginal"].get(
                "absolute_disparity_reduction"
            )
            labelcond_reduction = posthoc["labelcond"].get(
                "absolute_disparity_reduction"
            )
            if marginal_reduction is None or labelcond_reduction is None:
                check: bool | None = None
            else:
                check = labelcond_reduction > marginal_reduction
            rows.append({
                "task": task,
                "family": family,
                "bake_in": bake_in,
                "post_hoc": posthoc,
                "labelconditional_beats_marginal": check,
            })
    return rows


def collect_guardrails(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pred_set in PREDICTION_SETS[1:]:
        for task in TASKS:
            arm = results[pred_set["id"]]["tasks"][task]
            if not _complete(arm):
                continue
            rows.append({
                "arm": pred_set["id"],
                "stage": pred_set["stage"],
                "task": task,
                "underpowered_black_events": arm["underpowered_black_events"],
                "guardrails": arm["guardrails"],
                "comparison": arm["baseline_comparison"],
            })
    return rows


def _fmt(value: float | None, signed: bool = False, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _fmt_ci(task_result: dict[str, Any], metric: str, signed: bool) -> str:
    if not _complete(task_result):
        return "pending"
    block = task_result["metrics"][metric]
    lo, hi = block["bca_ci"]
    return (
        f"{_fmt(block['point'], signed)} "
        f"[{_fmt(lo, signed)}, {_fmt(hi, signed)}]"
    )


def _fmt_group_n(task_result: dict[str, Any]) -> str:
    if not _complete(task_result):
        return "pending"
    black = task_result["groups"]["Black"]
    marker = " ⚠" if task_result["underpowered_black_events"] else ""
    return f"{black['n']}/{black['events']}{marker}"


def _fmt_delta(entry: dict[str, Any]) -> str:
    value = entry.get("delta_signed_from_baseline")
    reduction = entry.get("absolute_disparity_reduction")
    if value is None or reduction is None:
        return "pending"
    return f"Δ={_fmt(value, True)}; |disp|↓={_fmt(reduction, True)}"


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    lines = [
        "# M5/M6 Cross-task Fairness Results",
        "",
        (
            f"Generated {payload['generated_at']} using `hh_metrics.py`, "
            f"{payload['config']['boot_n']:,} TSS-cluster bootstrap draws per "
            "available prediction set/task. Signed disparity is "
            "FPR_Black − FPR_White; AUROC gap is AUROC_White − AUROC_Black."
        ),
        "",
        (
            "Black N/events marked ⚠ are underpowered (<15 Black events). "
            "`pending` means one or more expected prediction JSONLs were absent."
        ),
        "",
        "## Primary table",
        "",
        (
            "| Stage | Arm | BRCA signed FPR-disp [BCa] | BRCA AUROC-gap "
            "[BCa] | BRCA Black AUROC | BRCA Black N/events | LUAD signed "
            "FPR-disp [BCa] | LUAD AUROC-gap [BCa] | LUAD Black AUROC | "
            "LUAD Black N/events |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pred_set in PREDICTION_SETS:
        arm = results[pred_set["id"]]
        brca = arm["tasks"]["brca"]
        luad = arm["tasks"]["luad"]
        lines.append(
            f"| {pred_set['stage']} | `{pred_set['id']}` | "
            f"{_fmt_ci(brca, 'signed_fpr_disparity', True)} | "
            f"{_fmt_ci(brca, 'auroc_gap_white_minus_black', True)} | "
            f"{_fmt(brca['groups']['Black']['auroc']) if _complete(brca) else 'pending'} | "
            f"{_fmt_group_n(brca)} | "
            f"{_fmt_ci(luad, 'signed_fpr_disparity', True)} | "
            f"{_fmt_ci(luad, 'auroc_gap_white_minus_black', True)} | "
            f"{_fmt(luad['groups']['Black']['auroc']) if _complete(luad) else 'pending'} | "
            f"{_fmt_group_n(luad)} |"
        )

    lines.extend([
        "",
        "## Cross-task generalization (bake-in FMs)",
        "",
        (
            "Positive `|disp| reduction` means the absolute signed FPR disparity "
            "is smaller than baseline. The transfer verdict is point-estimate "
            "based; the BCa intervals remain in the primary table."
        ),
        "",
        "| Bake-in FM | BRCA FPR / delta | LUAD FPR / delta | Verdict |",
        "|---|---:|---:|---|",
    ])
    for row in payload["cross_task_generalization"]:
        cells = []
        for task in ("brca", "luad"):
            entry = row["tasks"][task]
            if entry.get("status") != "complete":
                cells.append("pending")
            else:
                cells.append(
                    f"{_fmt(entry['fpr_disparity'], True)}; "
                    f"{_fmt_delta(entry)}"
                )
        lines.append(
            f"| `{row['fm']}` | {cells[0]} | {cells[1]} | "
            f"{row['verdict']} |"
        )

    lines.extend([
        "",
        "## Bake-in vs post-hoc, by method family",
        "",
        (
            "Comparisons use change in signed FPR disparity from baseline and "
            "absolute-disparity reduction (positive is better)."
        ),
        "",
        "| Task | Family | Bake-in variants | Post-hoc marginal | "
        "Post-hoc label-conditional | Label-conditional beats marginal? |",
        "|---|---|---|---:|---:|---|",
    ])
    for row in payload["head_to_head"]:
        bake_text = "<br>".join(
            f"`{entry['arm']}`: {_fmt_delta(entry)}"
            for entry in row["bake_in"]
        )
        label_check = row["labelconditional_beats_marginal"]
        check_text = "pending" if label_check is None else ("yes" if label_check else "no")
        lines.append(
            f"| {TASKS[row['task']]['label']} | {row['family']} | "
            f"{bake_text} | {_fmt_delta(row['post_hoc']['marginal'])} | "
            f"{_fmt_delta(row['post_hoc']['labelcond'])} | {check_text} |"
        )

    lines.extend([
        "",
        "## Guardrails",
        "",
        (
            "An evaluated arm fails if Black AUPRC drops by >0.02 absolute, "
            "Black ECE rises by >0.02 absolute, or overall task AUROC is ≤0.60. "
            "These are the study's prior guardrails; underpowering is reported "
            "separately."
        ),
        "",
    ])
    guardrails = payload["guardrails"]
    if not guardrails:
        lines.append(
            "No non-baseline arm is currently evaluable; guardrail comparisons "
            "are pending."
        )
    else:
        lines.extend([
            "| Task | Arm | Black AUPRC Δ | Black ECE Δ | Overall AUROC Δ | "
            "Status | Notes |",
            "|---|---|---:|---:|---:|---|---|",
        ])
        for row in guardrails:
            comparison = row["comparison"]
            guardrail = row["guardrails"]
            notes = "; ".join(guardrail["reasons"]) or "passes"
            if row["underpowered_black_events"]:
                notes += "; underpowered Black events"
            lines.append(
                f"| {TASKS[row['task']]['label']} | `{row['arm']}` | "
                f"{_fmt(comparison['black_auprc_delta'], True)} | "
                f"{_fmt(comparison['black_ece_delta'], True)} | "
                f"{_fmt(comparison['overall_auroc_delta'], True)} | "
                f"{guardrail['status']} | {notes} |"
            )

    present = []
    pending = []
    for item in payload["inventory"]:
        for task, task_inventory in item["tasks"].items():
            label = f"{item['id']} / {TASKS[task]['label']}"
            if task_inventory["status"] == "present":
                present.append(label)
            else:
                pending.append(f"{label} ({task_inventory['status']})")
    lines.extend([
        "",
        "## Prediction inventory",
        "",
        f"Present ({len(present)} set/task cells): "
        + (", ".join(f"`{x}`" for x in present) if present else "none"),
        "",
        f"Pending ({len(pending)} set/task cells): "
        + (", ".join(f"`{x}`" for x in pending) if pending else "none"),
        "",
    ])
    return "\n".join(lines)


def run_analysis(
    preds_dir: Path,
    boot_n: int,
    seed: int,
    force_subprocess: bool = False,
) -> dict[str, Any]:
    inventory = inventory_prediction_sets(preds_dir)
    inventory_by_id = {item["id"]: item for item in inventory}
    results: dict[str, Any] = {}
    for pred_set in PREDICTION_SETS:
        arm_result = {
            key: pred_set[key]
            for key in ("id", "stage", "family", "variant", "file_stem")
        }
        arm_result["tasks"] = {}
        for task in TASKS:
            task_inventory = inventory_by_id[pred_set["id"]]["tasks"][task]
            if task_inventory["status"] != "present":
                arm_result["tasks"][task] = {
                    "status": "pending",
                    "inventory_status": task_inventory["status"],
                    "missing_files": task_inventory["missing_files"],
                    "present_files": task_inventory["present_files"],
                    "note": "skipped because the complete prediction set is absent",
                }
                continue
            paths = [Path(path) for path in task_inventory["expected_files"]]
            print(
                f"[run] {pred_set['id']} / {TASKS[task]['label']}: "
                f"{boot_n:,} bootstrap draws",
                flush=True,
            )
            try:
                arm_result["tasks"][task] = run_hh_metrics(
                    paths,
                    task,
                    boot_n=boot_n,
                    seed=seed,
                    force_subprocess=force_subprocess,
                )
            except Exception as exc:
                arm_result["tasks"][task] = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "source_files": [str(path) for path in paths],
                }
                print(
                    f"[error] {pred_set['id']} / {TASKS[task]['label']}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        results[pred_set["id"]] = arm_result

    add_baseline_comparisons(results)
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "preds_dir": str(preds_dir),
            "boot_n": boot_n,
            "seed": seed,
            "sensitive_axis": "race",
            "reference_group": "White",
            "minority_group": "Black",
            "signed_fpr_definition": "FPR_Black - FPR_White",
            "signed_tpr_definition": "TPR_Black - TPR_White",
            "auroc_gap_definition": "AUROC_White - AUROC_Black",
            "black_event_floor": BLACK_EVENT_FLOOR,
            "guardrail_thresholds": {
                "black_auprc_max_absolute_drop": BLACK_AUPRC_TOLERANCE,
                "black_ece_max_absolute_increase": BLACK_ECE_TOLERANCE,
                "overall_auroc_floor": OVERALL_AUROC_FLOOR,
            },
        },
        "inventory": inventory,
        "results": results,
    }
    payload["cross_task_generalization"] = build_cross_task_generalization(results)
    payload["head_to_head"] = build_head_to_head(results)
    payload["guardrails"] = collect_guardrails(results)
    return _finite_or_none(payload)


def _print_inventory_summary(payload: dict[str, Any]) -> None:
    present = []
    pending = []
    for item in payload["inventory"]:
        for task, task_inventory in item["tasks"].items():
            cell = f"{item['id']}:{task}"
            if task_inventory["status"] == "present":
                present.append(cell)
            else:
                pending.append(f"{cell}({task_inventory['status']})")
    print(f"[inventory] present ({len(present)}): {', '.join(present) or 'none'}")
    print(f"[inventory] pending ({len(pending)}): {', '.join(pending) or 'none'}")


def _print_baseline_sample(payload: dict[str, Any]) -> None:
    sample: dict[str, Any] = {}
    for task in TASKS:
        result = payload["results"]["baseline"]["tasks"][task]
        if not _complete(result):
            sample[task] = {"status": result["status"]}
            continue
        sample[task] = {
            "signed_fpr_disparity": result["metrics"]["signed_fpr_disparity"],
            "signed_tpr_disparity": result["metrics"]["signed_tpr_disparity"],
            "auroc_gap": result["metrics"]["auroc_gap_white_minus_black"],
            "eo_max": result["metrics"]["eo_max"],
            "Black": result["groups"]["Black"],
            "White": result["groups"]["White"],
            "overall_auroc_all_races": result["metrics"][
                "overall_auroc_all_races"
            ],
            "underpowered_black_events": result["underpowered_black_events"],
        }
    print("[baseline sample]")
    print(json.dumps(sample, indent=2, sort_keys=True, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    parser.add_argument("--boot-n", type=int, default=DEFAULT_BOOT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT
    )
    parser.add_argument(
        "--force-subprocess",
        action="store_true",
        help="exercise the hh_metrics.py stdout-parser fallback",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.boot_n < 1:
        raise SystemExit("--boot-n must be >= 1")
    payload = run_analysis(
        preds_dir=args.preds_dir,
        boot_n=args.boot_n,
        seed=args.seed,
        force_subprocess=args.force_subprocess,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.markdown_out.write_text(render_markdown(payload))
    _print_inventory_summary(payload)
    _print_baseline_sample(payload)
    print(f"[output] JSON: {args.json_out}")
    print(f"[output] Markdown: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
