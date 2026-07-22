#!/usr/bin/env python3
"""
hh_analysis.py -- Hospital-holdout fairness analysis for the nanopath-JEPA
histopathology fairness study.

Evaluates arms (baseline + fair-encoder + fair-head variants) on a
hospital-held-out BRCA TP53 target set, with per-arm threshold refit at
80% White specificity to avoid score-compression artifacts.

Usage:
    python hh_analysis.py [--preds-dir DIR] [--out-dir DIR] [--boot-n N]

INPUTS
  - OOF prediction JSONLs at <preds-dir>/hh_<arm>__brca_tp53__<fold>.jsonl
    where <fold> in {F1, F2, F3} and <arm> in the ARM list below.
    Each JSONL row: {patient_barcode, score, label} or {patient_id, y_score, y_true}.
  - Hospital folds metadata: data/metadata/brca_hospital_folds.csv
    (patient_barcode, tss, race, tp53_status, fold)
  - Demographics: data/metadata/tcga12k_demographics.csv

OUTPUTS
  A. Per-arm table -> <out-dir>/hh_arms_table.json
  B. Primary confirmatory contrast (paired center-clustered bootstrap)
  C. Guardrail checks (minority AUPRC retention, calibration)
  D. Marginal-vs-conditional ablation
  E. Per-fold EO corroboration
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reused helpers from meta_analysis.py and fairness_eval.py
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta_analysis import (
    _auc as bootstrap_auc,
    threshold_at_spec,
    tpr_fpr_at,
    norm_cdf,
    two_sided_p,
)
from fairness_eval import ece_score as ece_score_fn

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ARM_LIST = [
    "baseline",
    "A_marginal", "A_cancercond",
    "B_contrastive_marginal", "B_contrastive_labelcond",
    "B_dann_marginal", "B_fino_marginal", "B_pcgrad_marginal",
]
FOLDS = ["F1", "F2", "F3"]
RACE_MAP = {"white": "White", "black or african american": "Black",
            "asian": "Asian"}
TARGET_SPEC = 0.80
BOOT_N_DEFAULT = 10000
BOOT_SEED = 20260716
AUROC_OK_THRESH = 0.6
MINORITY_AUPRC_TOLERANCE = 0.02
ECE_GUARDRAIL_TOLERANCE = 0.02
ECE_BINS = 10
METADATA_CSV = Path("/admin/home/ryan.kim/nt/data/metadata/brca_hospital_folds.csv")
DEMO_CSV = Path("/admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning list of dicts."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_pred_row(row: dict) -> dict:
    """Normalize prediction row to common field names."""
    return {
        "patient_barcode": row.get("patient_barcode") or row.get("patient_id", ""),
        "label": int(float(row.get("label") or row.get("y_true", 0))),
        "score": float(row.get("score") or row.get("y_score", 0.0)),
    }


def load_arm_folds(preds_dir: Path, arm: str) -> dict[str, list[dict]]:
    """Load all folds for one arm. Returns {fold: [rows]} for folds that exist."""
    folds = {}
    for fold in FOLDS:
        p = preds_dir / f"hh_{arm}__brca_tp53__{fold}.jsonl"
        if p.exists():
            raw = load_jsonl(p)
            folds[fold] = [normalize_pred_row(r) for r in raw]
    return folds


def load_metadata(csv_path: Path) -> dict[str, dict]:
    """Load brca_hospital_folds.csv keyed by patient_barcode."""
    import csv
    rows = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            bc = r["patient_barcode"].strip()
            rows[bc] = {
                "tss": r.get("tss", "").strip(),
                "race": RACE_MAP.get(r.get("race", "").strip().lower(), "Unknown"),
                "tp53_status": int(float(r.get("tp53_status", 0))),
                "fold": r.get("fold", "").strip(),
            }
    return rows


def load_demographics(csv_path: Path) -> dict[str, dict]:
    """Load tcga12k_demographics.csv keyed by patient_barcode."""
    import csv
    rows = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            bc = r["patient_barcode"].strip()
            rows[bc] = {
                "race": RACE_MAP.get(r.get("race", "").strip().lower(), "Unknown"),
                "gender": r.get("gender", "").strip(),
                "age_years": r.get("age_years", ""),
            }
    return rows


# --------------------------------------------------------------------------- #
# Pooling + joining
# --------------------------------------------------------------------------- #
def pool_and_join(
    fold_data: dict[str, list[dict]],
    metadata: dict[str, dict],
    demographics: dict[str, dict],
) -> list[dict] | None:
    """Pool 3 folds, join with metadata + demographics, return patient-level list.

    Returns None if no folds were loaded.
    Each returned dict: {patient_barcode, label, score, race, tss, fold}
    """
    if not fold_data:
        return None
    all_rows = []
    for fold, rows in fold_data.items():
        for r in rows:
            all_rows.append({**r, "_fold": fold})
    # Deduplicate by patient_barcode (keep first occurrence)
    seen = set()
    deduped = []
    for r in all_rows:
        bc = r["patient_barcode"]
        if bc not in seen:
            seen.add(bc)
            deduped.append(r)
    # Join with metadata for race and tss
    result = []
    for r in deduped:
        bc = r["patient_barcode"]
        meta = metadata.get(bc, {})
        demo = demographics.get(bc, {})
        race = meta.get("race") or demo.get("race") or "Unknown"
        result.append({
            "patient_barcode": bc,
            "label": r["label"],
            "score": r["score"],
            "race": race,
            "tss": meta.get("tss", ""),
            "fold": r.get("_fold", ""),
        })
    return result


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_auprc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Compute AUPRC (average precision). Returns None for degenerate inputs."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / n_pos
    precision = tp / (tp + fp)
    recall_full = np.concatenate([[0.0], recall])
    precision_full = np.concatenate([[1.0], precision])
    # AP = sum((R_k - R_{k-1}) * P_k)  (descending-score step function)
    return float(np.sum(np.diff(recall_full) * precision_full[1:]))


def group_metrics(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    """Compute FPR, TPR, PPV at threshold tau for one group."""
    pos = s[y == 1]
    neg = s[y == 0]
    tpr = float(np.mean(pos >= tau)) if len(pos) > 0 else None
    fpr = float(np.mean(neg >= tau)) if len(neg) > 0 else None
    preds = (s >= tau).astype(int)
    pred_pos = preds == 1
    ppv = float(y[pred_pos].mean()) if pred_pos.sum() > 0 else None
    return {"fpr": fpr, "tpr": tpr, "ppv": ppv}


def calibration_logistic(s: np.ndarray, y: np.ndarray) -> dict:
    """Fit logistic recalibration: logit(E[y|s]) = intercept + slope * s."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return {"slope": None, "intercept": None}
    slope = 0.0
    ymean = y.mean()
    ymean = max(min(ymean, 1 - 1e-7), 1e-7)
    intercept = float(np.log(ymean / (1 - ymean)))
    lr = 0.01
    for _ in range(500):
        z = intercept + slope * s
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        diff = y - p
        grad_slope = np.dot(s, diff) / n
        grad_intercept = np.mean(diff)
        slope += lr * grad_slope
        intercept += lr * grad_intercept
    return {"slope": float(slope), "intercept": float(intercept)}


def calibration_metrics(s: np.ndarray, y: np.ndarray) -> dict:
    """Compute calibration slope, intercept, and ECE for one group."""
    cal = calibration_logistic(s, y)
    ece = ece_score_fn(y, s, n_bins=ECE_BINS)
    cal["ece"] = ece
    return cal


def analyze_arm(
    patients: list[dict], arm_name: str
) -> dict | None:
    """Full per-arm analysis. Returns a dict of all metrics, or None if degenerate."""
    if not patients:
        return None

    # Split by race
    black = [p for p in patients if p["race"] == "Black"]
    white = [p for p in patients if p["race"] == "White"]
    if not black or not white:
        return None

    all_y = np.array([p["label"] for p in patients], dtype=int)
    all_s = np.array([p["score"] for p in patients], dtype=float)
    bl_y = np.array([p["label"] for p in black], dtype=int)
    bl_s = np.array([p["score"] for p in black], dtype=float)
    wh_y = np.array([p["label"] for p in white], dtype=int)
    wh_s = np.array([p["score"] for p in white], dtype=float)

    # Per-arm threshold refit: tau so WHITE hits 80% specificity
    tau = threshold_at_spec(wh_y, wh_s, target_spec=TARGET_SPEC)
    if tau is None:
        return None

    # Threshold-free metrics
    overall_auroc = bootstrap_auc(all_y, all_s)
    overall_auprc = compute_auprc(all_y, all_s)
    bl_auroc = bootstrap_auc(bl_y, bl_s)
    bl_auprc = compute_auprc(bl_y, bl_s)
    wh_auroc = bootstrap_auc(wh_y, wh_s)
    wh_auprc = compute_auprc(wh_y, wh_s)

    # At-threshold metrics
    bl_at = group_metrics(bl_y, bl_s, tau)
    wh_at = group_metrics(wh_y, wh_s, tau)

    # Disparities
    fpr_disp = (bl_at["fpr"] or 0) - (wh_at["fpr"] or 0)
    tpr_disp = (bl_at["tpr"] or 0) - (wh_at["tpr"] or 0)
    eo = max(abs(fpr_disp), abs(tpr_disp))

    # Calibration per group
    bl_cal = calibration_metrics(bl_s, bl_y)
    wh_cal = calibration_metrics(wh_s, wh_y)

    auroc_ok = bool(overall_auroc is not None and overall_auroc > AUROC_OK_THRESH)

    return {
        "arm": arm_name,
        "n_total": len(patients),
        "n_black": len(black),
        "n_white": len(white),
        "black_events": int(bl_y.sum()),
        "white_events": int(wh_y.sum()),
        "tau": float(tau),
        # Overall
        "overall_auroc": overall_auroc,
        "overall_auprc": overall_auprc,
        "auroc_ok": auroc_ok,
        # Per-group threshold-free
        "black_auroc": bl_auroc,
        "black_auprc": bl_auprc,
        "white_auroc": wh_auroc,
        "white_auprc": wh_auprc,
        # At-threshold
        "black_fpr": bl_at["fpr"],
        "black_tpr": bl_at["tpr"],
        "black_ppv": bl_at["ppv"],
        "white_fpr": wh_at["fpr"],
        "white_tpr": wh_at["tpr"],
        "white_ppv": wh_at["ppv"],
        # Disparities
        "fpr_disparity": fpr_disp,
        "tpr_disparity": tpr_disp,
        "eo": eo,
        # Calibration
        "black_cal_slope": bl_cal["slope"],
        "black_cal_intercept": bl_cal["intercept"],
        "black_cal_ece": bl_cal["ece"],
        "white_cal_slope": wh_cal["slope"],
        "white_cal_intercept": wh_cal["intercept"],
        "white_cal_ece": wh_cal["ece"],
    }


# --------------------------------------------------------------------------- #
# Per-fold EO (section E)
# --------------------------------------------------------------------------- #
def per_fold_eo(
    fold_data: dict[str, list[dict]],
    metadata: dict[str, dict],
    demographics: dict[str, dict],
    arm_name: str,
) -> list[dict]:
    """Compute EO per fold for corroboration."""
    results = []
    for fold in FOLDS:
        if fold not in fold_data:
            continue
        patients = pool_and_join({fold: fold_data[fold]}, metadata, demographics)
        if not patients:
            continue
        black = [p for p in patients if p["race"] == "Black"]
        white = [p for p in patients if p["race"] == "White"]
        if not black or not white:
            results.append({
                "arm": arm_name, "fold": fold,
                "n_black": len(black), "n_white": len(white),
                "eo": None, "fpr_disparity": None, "tpr_disparity": None,
                "black_events": 0,
            })
            continue
        bl_y = np.array([p["label"] for p in black], dtype=int)
        bl_s = np.array([p["score"] for p in black], dtype=float)
        wh_y = np.array([p["label"] for p in white], dtype=int)
        wh_s = np.array([p["score"] for p in white], dtype=float)
        tau = threshold_at_spec(wh_y, wh_s, target_spec=TARGET_SPEC)
        if tau is None:
            results.append({
                "arm": arm_name, "fold": fold,
                "n_black": len(black), "n_white": len(white),
                "eo": None, "fpr_disparity": None, "tpr_disparity": None,
                "black_events": int(bl_y.sum()),
            })
            continue
        bl_at = group_metrics(bl_y, bl_s, tau)
        wh_at = group_metrics(wh_y, wh_s, tau)
        fpr_disp = (bl_at["fpr"] or 0) - (wh_at["fpr"] or 0)
        tpr_disp = (bl_at["tpr"] or 0) - (wh_at["tpr"] or 0)
        eo = max(abs(fpr_disp), abs(tpr_disp))
        results.append({
            "arm": arm_name, "fold": fold,
            "n_black": len(black), "n_white": len(white),
            "black_events": int(bl_y.sum()),
            "eo": eo,
            "fpr_disparity": fpr_disp,
            "tpr_disparity": tpr_disp,
            "tau": float(tau),
        })
    return results


# --------------------------------------------------------------------------- #
# Paired center-clustered bootstrap (section B)
# --------------------------------------------------------------------------- #
def paired_bootstrap_eo_diff(
    arm_a_patients: list[dict],
    arm_b_patients: list[dict],
    n_boot: int = BOOT_N_DEFAULT,
    seed: int = BOOT_SEED,
) -> dict:
    """Paired center-clustered bootstrap of EO(A) - EO(B).

    Patients are joined by barcode. TSS clusters are resampled with replacement,
    then both arms' EO are recomputed on the SAME resampled patients.
    tau is refit per arm within each bootstrap draw.
    """
    rng = np.random.default_rng(seed)

    idx_a = {p["patient_barcode"]: p for p in arm_a_patients}
    idx_b = {p["patient_barcode"]: p for p in arm_b_patients}

    shared_bcs = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    if not shared_bcs:
        return {"error": "no shared patients between arms"}

    tss_groups = defaultdict(list)
    for bc in shared_bcs:
        tss = idx_a[bc].get("tss", bc[:7])
        if not tss:
            tss = bc[:7]
        tss_groups[tss].append(bc)

    tss_clusters = list(tss_groups.keys())
    if not tss_clusters:
        return {"error": "no TSS clusters found"}

    def compute_eo(sample_bcs: list[str], idx_map: dict) -> float | None:
        """Compute EO for one arm on the given patient sample."""
        black_p = [idx_map[bc] for bc in sample_bcs
                    if idx_map[bc]["race"] == "Black"]
        white_p = [idx_map[bc] for bc in sample_bcs
                    if idx_map[bc]["race"] == "White"]
        if not black_p or not white_p:
            return None
        bl_y = np.array([p["label"] for p in black_p], dtype=int)
        bl_s = np.array([p["score"] for p in black_p], dtype=float)
        wh_y = np.array([p["label"] for p in white_p], dtype=int)
        wh_s = np.array([p["score"] for p in white_p], dtype=float)
        tau = threshold_at_spec(wh_y, wh_s, target_spec=TARGET_SPEC)
        if tau is None:
            return None
        bl_at = group_metrics(bl_y, bl_s, tau)
        wh_at = group_metrics(wh_y, wh_s, tau)
        fpr_disp = (bl_at["fpr"] or 0) - (wh_at["fpr"] or 0)
        tpr_disp = (bl_at["tpr"] or 0) - (wh_at["tpr"] or 0)
        return max(abs(fpr_disp), abs(tpr_disp))

    # Full-sample EO difference
    eo_a_full = compute_eo(shared_bcs, idx_a)
    eo_b_full = compute_eo(shared_bcs, idx_b)
    if eo_a_full is None or eo_b_full is None:
        return {"error": "cannot compute EO on full sample"}
    full_diff = eo_a_full - eo_b_full

    # Bootstrap: resample TSS clusters with replacement
    draws = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(tss_clusters, size=len(tss_clusters), replace=True)
        sample_bcs = []
        for cl in sampled_clusters:
            sample_bcs.extend(tss_groups[cl])
        eo_a = compute_eo(sample_bcs, idx_a)
        eo_b = compute_eo(sample_bcs, idx_b)
        if eo_a is not None and eo_b is not None:
            draws.append(eo_a - eo_b)

    draws = np.array(draws)
    if len(draws) == 0:
        return {"error": "all bootstrap draws failed"}

    ci_lo = float(np.percentile(draws, 2.5))
    ci_hi = float(np.percentile(draws, 97.5))
    excludes_zero = not (ci_lo <= 0 <= ci_hi)

    if full_diff > 0 and excludes_zero:
        verdict = "A_cancercond has HIGHER EO (worse fairness)"
    elif full_diff < 0 and excludes_zero:
        verdict = "B_contrastive_labelcond has HIGHER EO (worse fairness)"
    else:
        verdict = "indistinguishable"

    return {
        "eo_diff": float(full_diff),
        "eo_a_full": eo_a_full,
        "eo_b_full": eo_b_full,
        "n_bootstrap_draws": len(draws),
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "ci_excludes_zero": excludes_zero,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def f3(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.3f}"


def fs(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.3f}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Hospital-holdout fairness analysis for nanopath-JEPA"
    )
    ap.add_argument(
        "--preds-dir",
        default="/data/ryan.kim/nanopath/results/preds",
        help="Directory containing hh_<arm>__brca_tp53__<fold>.jsonl files",
    )
    ap.add_argument(
        "--out-dir",
        default="/admin/home/ryan.kim/nt/results",
        help="Output directory for results (default: results/)",
    )
    ap.add_argument(
        "--boot-n", type=int, default=BOOT_N_DEFAULT,
        help=f"Number of bootstrap resamples (default: {BOOT_N_DEFAULT})",
    )
    ap.add_argument(
        "--seed", type=int, default=BOOT_SEED,
        help=f"Random seed (default: {BOOT_SEED})",
    )
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    log = lambda *a: print(*a, flush=True)

    log("=" * 72)
    log("  HOSPITAL-HOLDBACK FAIRNESS ANALYSIS")
    log("=" * 72)
    log(f"  preds-dir : {preds_dir}")
    log(f"  out-dir   : {out_dir}")
    log(f"  boot-n    : {args.boot_n}")
    log(f"  seed      : {args.seed}")
    log("")

    # Load reference data
    log("[1] Loading metadata + demographics ...")
    metadata = load_metadata(METADATA_CSV)
    demographics = load_demographics(DEMO_CSV)
    log(f"  metadata : {len(metadata)} patients")
    log(f"  demo     : {len(demographics)} patients")
    log("")

    # Discover arms
    log("[2] Scanning for arm files ...")
    found_arms = []
    missing_arms = []
    for arm in ARM_LIST:
        fold_data = load_arm_folds(preds_dir, arm)
        if fold_data:
            found_arms.append(arm)
            log(f"  FOUND  {arm}  ({len(fold_data)} folds: {sorted(fold_data.keys())})")
        else:
            missing_arms.append(arm)
            log(f"  MISSING {arm}")
    log("")

    if not found_arms:
        log("[RESULT] No arm files found. Nothing to analyze.")
        log("  Expected pattern: hh_<arm>__brca_tp53__<fold>.jsonl")
        log(f"  in {preds_dir}")
        placeholder = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "arms_found": [],
            "arms_missing": missing_arms,
            "message": "No hospital-holdout prediction files found. Re-run when preds are ready.",
        }
        out_path = out_dir / "hh_arms_table.json"
        out_path.write_text(json.dumps(placeholder, indent=2))
        log(f"  Wrote placeholder -> {out_path}")
        elapsed = time.monotonic() - t0
        log(f"\n[DONE] {elapsed:.1f}s (no arms found)")
        return

    # -------------------------------------------------------------------------
    # A. Per-arm analysis
    # -------------------------------------------------------------------------
    log("[3] Per-arm analysis ...")
    arm_results = {}
    arm_patients = {}
    for arm in found_arms:
        fold_data = load_arm_folds(preds_dir, arm)
        patients = pool_and_join(fold_data, metadata, demographics)
        if patients is None:
            log(f"  SKIP   {arm} (no pooled data)")
            continue
        arm_patients[arm] = patients
        result = analyze_arm(patients, arm)
        if result is None:
            n_bl = sum(1 for p in patients if p["race"] == "Black")
            n_wh = sum(1 for p in patients if p["race"] == "White")
            log(f"  SKIP   {arm} (degenerate: {len(patients)} patients, "
                f"Black={n_bl}, White={n_wh})")
            continue
        arm_results[arm] = result
        log(f"  OK     {arm:<35s}  AUROC={f3(result['overall_auroc'])}  "
            f"EO={f3(result['eo'])}  FPRdisp={fs(result['fpr_disparity'])}  "
            f"Black={result['n_black']}  White={result['n_white']}  "
            f"auroc_ok={result['auroc_ok']}")
    log("")

    # Print full per-arm table
    log("=" * 72)
    log("  A. PER-ARM TABLE")
    log("=" * 72)
    if arm_results:
        headers = (
            f"{'arm':<32s} "
            f"{'AUROC':>6s} {'AUPRC':>6s} "
            f"{'Bl_AUC':>8s} {'Wh_AUC':>8s} "
            f"{'Bl_AUPRC':>8s} {'Wh_AUPRC':>8s} "
            f"{'Bl_FPR':>6s} {'Wh_FPR':>6s} "
            f"{'Bl_TPR':>6s} {'Wh_TPR':>6s} "
            f"{'Bl_PPV':>6s} {'Wh_PPV':>6s} "
            f"{'FPRdisp':>8s} {'TPRdisp':>8s} {'EO':>6s} "
            f"{'Bl_slope':>8s} {'Wh_slope':>8s} "
            f"{'Bl_ECE':>6s} {'Wh_ECE':>6s} "
            f"{'auroc_ok':>8s}"
        )
        log(headers)
        log("-" * len(headers))
        for arm in ARM_LIST:
            if arm not in arm_results:
                continue
            r = arm_results[arm]
            auroc_ok_str = "YES" if r["auroc_ok"] else "NO"
            log(
                f"{arm:<32s} "
                f"{f3(r['overall_auroc']):>6s} {f3(r['overall_auprc']):>6s} "
                f"{f3(r['black_auroc']):>8s} {f3(r['white_auroc']):>8s} "
                f"{f3(r['black_auprc']):>8s} {f3(r['white_auprc']):>8s} "
                f"{f3(r['black_fpr']):>6s} {f3(r['white_fpr']):>6s} "
                f"{f3(r['black_tpr']):>6s} {f3(r['white_tpr']):>6s} "
                f"{f3(r['black_ppv']):>6s} {f3(r['white_ppv']):>6s} "
                f"{fs(r['fpr_disparity']):>8s} {fs(r['tpr_disparity']):>8s} {f3(r['eo']):>6s} "
                f"{f3(r['black_cal_slope']):>8s} {f3(r['white_cal_slope']):>8s} "
                f"{f3(r['black_cal_ece']):>6s} {f3(r['white_cal_ece']):>6s} "
                f"{auroc_ok_str:>8s}"
            )
    log("")

    # -------------------------------------------------------------------------
    # B. Primary confirmatory contrast
    # -------------------------------------------------------------------------
    log("=" * 72)
    log("  B. PRIMARY CONFIRMATORY CONTRAST")
    log("     EO(A_cancercond) - EO(B_contrastive_labelcond)")
    log("     (positive => A has HIGHER EO / worse fairness)")
    log("=" * 72)
    confirmatory = {}
    a_arm = "A_cancercond"
    b_arm = "B_contrastive_labelcond"
    if a_arm in arm_patients and b_arm in arm_patients:
        log(f"  Running paired center-clustered bootstrap (n={args.boot_n}) ...")
        confirmatory = paired_bootstrap_eo_diff(
            arm_patients[a_arm], arm_patients[b_arm],
            n_boot=args.boot_n, seed=args.seed,
        )
        if "error" in confirmatory:
            log(f"  ERROR: {confirmatory['error']}")
        else:
            log(f"  EO(A) - EO(B) = {fs(confirmatory['eo_diff'])}")
            log(f"  EO(A) = {f3(confirmatory['eo_a_full'])}, "
                f"EO(B) = {f3(confirmatory['eo_b_full'])}")
            log(f"  95% CI = [{f3(confirmatory['ci_95_lo'])}, "
                f"{f3(confirmatory['ci_95_hi'])}]")
            log(f"  CI excludes 0: {confirmatory['ci_excludes_zero']}")
            log(f"  Verdict: {confirmatory['verdict']}")
            log(f"  Bootstrap draws: {confirmatory['n_bootstrap_draws']}")
    else:
        missing = []
        if a_arm not in arm_patients:
            missing.append(a_arm)
        if b_arm not in arm_patients:
            missing.append(b_arm)
        log(f"  SKIP: missing arms {missing}")
    log("")

    # -------------------------------------------------------------------------
    # C. Guardrail checks
    # -------------------------------------------------------------------------
    log("=" * 72)
    log("  C. GUARDRAIL CHECKS")
    log("     Minority (Black) AUPRC >= baseline_Black_AUPRC - 0.02")
    log("     AND calibration not worse (ECE <= baseline_Black_ECE + 0.02)")
    log("=" * 72)
    guardrail_results = {}
    baseline_result = arm_results.get("baseline")
    if baseline_result:
        bl_auprc_ref = baseline_result["black_auprc"]
        bl_ece_ref = baseline_result["black_cal_ece"]
        if bl_auprc_ref is None:
            log("  SKIP: baseline Black AUPRC is None (insufficient data)")
        else:
            bl_ece_ref = bl_ece_ref or 0
            log(f"  Baseline Black AUPRC = {f3(bl_auprc_ref)}, "
                f"ECE = {f3(bl_ece_ref)}")
            log(f"  AUPRC floor = {f3(bl_auprc_ref - MINORITY_AUPRC_TOLERANCE)}, "
                f"ECE ceiling = {f3(bl_ece_ref + ECE_GUARDRAIL_TOLERANCE)}")
            log("")
            for arm in ARM_LIST:
                if arm == "baseline" or arm not in arm_results:
                    continue
                r = arm_results[arm]
                arm_auprc = r["black_auprc"]
                arm_ece = r["black_cal_ece"]
                auprc_pass = (arm_auprc is not None
                             and arm_auprc >= (bl_auprc_ref - MINORITY_AUPRC_TOLERANCE))
                ece_pass = (arm_ece is not None
                           and arm_ece <= (bl_ece_ref + ECE_GUARDRAIL_TOLERANCE))
                overall_pass = auprc_pass and ece_pass
                guardrail_results[arm] = {
                    "black_auprc": arm_auprc,
                    "auprc_pass": auprc_pass,
                    "black_ece": arm_ece,
                    "ece_pass": ece_pass,
                    "pass": overall_pass,
                }
                status = "PASS" if overall_pass else "FAIL"
                auprc_flag = "OK" if auprc_pass else "LOW"
                ece_flag = "OK" if ece_pass else "HIGH"
                log(f"  {status}  {arm:<35s}  BlAUPRC={f3(arm_auprc)} ({auprc_flag}), "
                    f"BlECE={f3(arm_ece)} ({ece_flag})")
    else:
        log("  SKIP: baseline arm not analyzed")
    log("")

    # -------------------------------------------------------------------------
    # D. Marginal-vs-conditional ablation
    # -------------------------------------------------------------------------
    log("=" * 72)
    log("  D. MARGINAL-VS-CONDITIONAL ABLATION")
    log("=" * 72)
    ablation_pairs = [
        ("A_marginal", "A_cancercond", "Fair FM (bake-in)"),
        ("B_contrastive_marginal", "B_contrastive_labelcond",
         "Fair head (patch-local)"),
    ]
    for marg, cond, label in ablation_pairs:
        log(f"  {label}: {marg} vs {cond}")
        if marg in arm_results and cond in arm_results:
            mr = arm_results[marg]
            cr = arm_results[cond]
            eo_change = cr["eo"] - mr["eo"]
            auprc_change = (cr["black_auprc"] or 0) - (mr["black_auprc"] or 0)
            log(f"    EO(marginal)      = {f3(mr['eo'])}")
            log(f"    EO(conditional)   = {f3(cr['eo'])}")
            log(f"    EO change (cond-marg) = {fs(eo_change)}")
            log(f"    Black AUPRC(marginal)   = {f3(mr['black_auprc'])}")
            log(f"    Black AUPRC(conditional)= {f3(cr['black_auprc'])}")
            log(f"    Black AUPRC change = {fs(auprc_change)}")
            if eo_change < 0:
                log(f"    => Conditional REDUCES disparity by "
                    f"{f3(abs(eo_change))}")
            elif eo_change > 0:
                log(f"    => Conditional INCREASES disparity by "
                    f"{f3(eo_change)}")
            else:
                log(f"    => Conditional has SAME EO as marginal")
            if auprc_change >= 0:
                log(f"    => Conditional PRESERVES/improves minority AUPRC")
            else:
                log(f"    => Conditional LOSES {f3(abs(auprc_change))} "
                    f"minority AUPRC")
        else:
            missing_c = []
            if marg not in arm_results:
                missing_c.append(marg)
            if cond not in arm_results:
                missing_c.append(cond)
            log(f"    SKIP: missing {missing_c}")
        log("")

    # -------------------------------------------------------------------------
    # E. Per-fold EO
    # -------------------------------------------------------------------------
    log("=" * 72)
    log("  E. PER-FOLD EO CORROBORATION")
    log("=" * 72)
    per_fold_results = {}
    for arm in found_arms:
        fold_data = load_arm_folds(preds_dir, arm)
        pf = per_fold_eo(fold_data, metadata, demographics, arm)
        per_fold_results[arm] = pf
        for entry in pf:
            eo_str = f3(entry["eo"]) if entry["eo"] is not None else "n/a"
            log(f"  {arm:<35s}  {entry['fold']}: EO={eo_str}  "
                f"Black={entry['n_black']} (events={entry['black_events']})  "
                f"White={entry['n_white']}")
    log("")

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "preds_dir": str(preds_dir),
        "boot_n": args.boot_n,
        "seed": args.seed,
        "arms_found": found_arms,
        "arms_missing": missing_arms,
        "arms_table": arm_results,
        "confirmatory_contrast": confirmatory,
        "guardrail_checks": guardrail_results,
        "per_fold_eo": per_fold_results,
    }

    out_path = out_dir / "hh_arms_table.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    log(f"[OUTPUT] {out_path}")

    elapsed = time.monotonic() - t0
    log(f"\n[DONE] {elapsed:.1f}s")


if __name__ == "__main__":
    main()