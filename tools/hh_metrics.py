#!/usr/bin/env python3
"""
hh_metrics.py -- reusable fairness-metrics module for the hospital-holdout BRCA
TP53 race analysis, with a patient- (TSS-cluster-) clustered bootstrap that
reports BOTH percentile and BCa 95% CIs.

Milestone M0. The point of this module is to demonstrate a fix to a known
CI-skew defect: the prior primary metric EO = max(|dFPR|, |dTPR|) has an
UPWARD-BIASED percentile bootstrap CI (max() is upward-biased under resampling),
so the point estimate sits near the BOTTOM of its CI. Reporting the SIGNED
disparities (FPR_Black - FPR_White and TPR_Black - TPR_White) instead yields
CENTERED CIs. This module computes all four and reports centering fractions so
the fix can be verified.

Operating point: tau is REFIT to the WHITE group's 80% specificity on the WHITE
NEGATIVES' scores (tau = threshold_at_spec(white_y, white_s, 0.80)); a patient is
classified positive iff score >= tau. By construction FPR_White ~= 0.20. tau is
re-fit within EVERY bootstrap draw / jackknife subset so threshold uncertainty
propagates into the CIs.

Reuses (does not modify):
  * meta_analysis.threshold_at_spec  -- tau = 80th pct of the arm's negatives
  * fairness_eval.ece_score          -- 10-bin equal-width ECE
  * fairness_eval.RACE_MAP           -- lowercase race string -> canonical label
AUROC / AUPRC use sklearn (roc_auc_score / average_precision_score) directly.

CPU-only, read-only. Importable (all top-level exec guarded under __main__).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# --- reuse existing helpers (add tools/ to sys.path so imports resolve) ------
_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from meta_analysis import threshold_at_spec          # noqa: E402
from fairness_eval import ece_score, RACE_MAP         # noqa: E402

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
TARGET_SPEC = 0.80
BOOT_N_DEFAULT = 10000
BOOT_SEED = 20260716
EVENT_FLOOR = 15                 # < this many minority events => underpowered
CAL_EPS = 1e-6                   # score clip for logit() in calibration slope
DEFAULT_FOLDS_CSV = "/admin/home/ryan.kim/nt/data/metadata/brca_hospital_folds.csv"
DEFAULT_DEMOGRAPHICS_CSV = (
    "/admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv"
)
DEFAULT_PREDS = [
    f"/data/ryan.kim/nanopath/results/preds/hh_baseline__brca_tp53__F{f}.jsonl"
    for f in (1, 2, 3)
]

# The numeric metrics carried through the bootstrap / jackknife (the CI targets).
METRIC_KEYS = [
    "fpr_disparity", "tpr_disparity", "eo", "auroc_gap",
    "white_auroc", "black_auroc", "white_auprc", "black_auprc",
    "white_ppv", "black_ppv",
    "white_cal_slope", "white_cal_intercept", "white_cal_ece",
    "black_cal_slope", "black_cal_intercept", "black_cal_ece",
    "white_fpr", "black_fpr", "white_tpr", "black_tpr", "tau",
]
NAN = float("nan")


# --------------------------------------------------------------------------- #
# single-group primitives
# --------------------------------------------------------------------------- #
def _auroc(y, s):
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return NAN
    return float(roc_auc_score(y, s))


def _auprc(y, s):
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2:
        return NAN
    return float(average_precision_score(y, s))


def _ppv_at(y, s, tau):
    """Precision among predicted-positives (score >= tau). NaN if none predicted."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pred_pos = s >= tau
    if pred_pos.sum() == 0:
        return NAN
    return float(y[pred_pos].mean())


def _tpr_fpr(y, s, tau):
    """(TPR, FPR) at tau for one group; component NaN if that class is empty."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    tpr = float(np.mean(pos >= tau)) if len(pos) else NAN
    fpr = float(np.mean(neg >= tau)) if len(neg) else NAN
    return tpr, fpr


def _calibration(y, s):
    """Cox calibration slope + intercept: logistic regression of y on
    logit(clip(score)). ECE via the reused equal-width ece_score on the RAW
    probabilities. Returns (slope, intercept, ece); slope/intercept NaN if a
    group is single-class or the fit fails."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    ece = float(ece_score(y, s)) if len(y) else NAN
    if len(set(y.tolist())) < 2:
        return NAN, NAN, ece
    sc = np.clip(s, CAL_EPS, 1.0 - CAL_EPS)
    logit = np.log(sc / (1.0 - sc)).reshape(-1, 1)
    try:
        clf = LogisticRegression(solver="lbfgs", max_iter=1000).fit(logit, y)
        return float(clf.coef_[0][0]), float(clf.intercept_[0]), ece
    except Exception:
        return NAN, NAN, ece


# --------------------------------------------------------------------------- #
# point-metric computation (the statistic; also used per bootstrap/jackknife)
# --------------------------------------------------------------------------- #
def compute_metrics(patients, reference_group="White", minority_group="Black"):
    """Compute every fairness metric on a pooled patient list.

    `patients`: iterable of dicts with keys 'race' ('White'/'Black'), 'y_true'
    (0/1), 'y_score' (float). tau is refit to the WHITE arm's 80% specificity on
    its own negatives; positive iff score >= tau. Any metric whose inputs are
    degenerate in this sample is returned as NaN (so a bootstrap/jackknife pass
    can drop it for that draw without discarding the others).

    Returns a dict over METRIC_KEYS plus per-group counts (n_*, *_events).
    """
    white = [p for p in patients
             if p.get("sensitive_group", p["race"]) == reference_group]
    black = [p for p in patients
             if p.get("sensitive_group", p["race"]) == minority_group]
    out = {k: NAN for k in METRIC_KEYS}
    wy = np.asarray([p["y_true"] for p in white], dtype=int)
    ws = np.asarray([p["y_score"] for p in white], dtype=float)
    by = np.asarray([p["y_true"] for p in black], dtype=int)
    bs = np.asarray([p["y_score"] for p in black], dtype=float)
    out.update({
        "n_white": len(white), "n_black": len(black),
        "white_events": int(wy.sum()) if len(wy) else 0,
        "black_events": int(by.sum()) if len(by) else 0,
    })

    # threshold-free per-group
    out["white_auroc"] = _auroc(wy, ws) if len(wy) else NAN
    out["black_auroc"] = _auroc(by, bs) if len(by) else NAN
    out["white_auprc"] = _auprc(wy, ws) if len(wy) else NAN
    out["black_auprc"] = _auprc(by, bs) if len(by) else NAN
    if not (np.isnan(out["white_auroc"]) or np.isnan(out["black_auroc"])):
        out["auroc_gap"] = out["white_auroc"] - out["black_auroc"]

    # calibration per group
    w_slope, w_int, w_ece = _calibration(wy, ws) if len(wy) else (NAN, NAN, NAN)
    b_slope, b_int, b_ece = _calibration(by, bs) if len(by) else (NAN, NAN, NAN)
    out.update({
        "white_cal_slope": w_slope, "white_cal_intercept": w_int, "white_cal_ece": w_ece,
        "black_cal_slope": b_slope, "black_cal_intercept": b_int, "black_cal_ece": b_ece,
    })

    # operating point: tau from the WHITE negatives -> FPR_White ~= 0.20
    tau = threshold_at_spec(wy, ws, target_spec=TARGET_SPEC) if len(wy) else None
    if tau is None:
        return out
    out["tau"] = float(tau)
    w_tpr, w_fpr = _tpr_fpr(wy, ws, tau)
    b_tpr, b_fpr = _tpr_fpr(by, bs, tau)
    out.update({"white_tpr": w_tpr, "white_fpr": w_fpr,
                "black_tpr": b_tpr, "black_fpr": b_fpr})
    out["white_ppv"] = _ppv_at(wy, ws, tau)
    out["black_ppv"] = _ppv_at(by, bs, tau)

    if not (np.isnan(b_fpr) or np.isnan(w_fpr)):
        out["fpr_disparity"] = b_fpr - w_fpr
    if not (np.isnan(b_tpr) or np.isnan(w_tpr)):
        out["tpr_disparity"] = b_tpr - w_tpr
    if not (np.isnan(out["fpr_disparity"]) or np.isnan(out["tpr_disparity"])):
        out["eo"] = max(abs(out["fpr_disparity"]), abs(out["tpr_disparity"]))
    return out


# --------------------------------------------------------------------------- #
# clustered bootstrap + percentile & BCa CIs
# --------------------------------------------------------------------------- #
def _tss_clusters(patients):
    """dict tss -> list of patient dicts (the resampling units)."""
    groups = defaultdict(list)
    for p in patients:
        groups[p["tss"]].append(p)
    return groups


def bootstrap_ci(patients, metric_keys=METRIC_KEYS, n_boot=BOOT_N_DEFAULT,
                 seed=BOOT_SEED, reference_group="White",
                 minority_group="Black"):
    """Patient- (TSS-cluster-) clustered bootstrap with percentile AND BCa 95%
    CIs for every metric in `metric_keys`.

    Bootstrap: resample TSS clusters WITH REPLACEMENT (each drawn cluster
    contributes ALL its patients); refit tau on that draw's White negatives;
    recompute every metric. NaN metrics are dropped per-metric.

    BCa (clustered):
      z0 = Phi^{-1}( #{draws < point} / n_used )
      a  = leave-one-TSS-cluster-out jackknife acceleration:
           theta_(i) = statistic on all patients EXCEPT cluster i,
           m = mean_i theta_(i),
           a = sum (m - theta_(i))^3 / (6 * (sum (m - theta_(i))^2)^1.5)
      alpha_lo/hi = Phi( z0 + (z0 +/- z_.025) / (1 - a (z0 +/- z_.025)) )
      BCa CI = percentiles(draws, [100*alpha_lo, 100*alpha_hi]).

    Returns {metric: {point, pct_lo, pct_hi, bca_lo, bca_hi, n_used, z0, a}}.
    """
    point = compute_metrics(patients, reference_group, minority_group)
    clusters = list(_tss_clusters(patients).values())
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)

    # --- bootstrap draws -----------------------------------------------------
    draws = {k: [] for k in metric_keys}
    n_effective = 0
    for _ in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        sample = [p for ci in pick for p in clusters[ci]]
        m = compute_metrics(sample, reference_group, minority_group)
        n_effective += 1
        for k in metric_keys:
            v = m[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                draws[k].append(v)
    draws = {k: np.asarray(v, dtype=float) for k, v in draws.items()}

    # --- leave-one-cluster-out jackknife (for acceleration) ------------------
    jack = {k: [] for k in metric_keys}
    for i in range(n_clusters):
        sub = [p for j, cl in enumerate(clusters) if j != i for p in cl]
        m = compute_metrics(sub, reference_group, minority_group)
        for k in metric_keys:
            v = m[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                jack[k].append(v)
    jack = {k: np.asarray(v, dtype=float) for k, v in jack.items()}

    z_lo, z_hi = norm.ppf(0.025), norm.ppf(0.975)
    results = {}
    for k in metric_keys:
        arr = draws[k]
        pt = point[k]
        res = {"point": pt, "n_used": int(len(arr)),
               "pct_lo": NAN, "pct_hi": NAN, "bca_lo": NAN, "bca_hi": NAN,
               "z0": NAN, "a": NAN}
        if len(arr) == 0 or pt is None or (isinstance(pt, float) and np.isnan(pt)):
            results[k] = res
            continue
        res["pct_lo"] = float(np.percentile(arr, 2.5))
        res["pct_hi"] = float(np.percentile(arr, 97.5))

        # bias correction z0
        frac = float(np.mean(arr < pt))
        frac = min(max(frac, 1.0 / (len(arr) + 1)), len(arr) / (len(arr) + 1.0))
        z0 = float(norm.ppf(frac))
        res["z0"] = z0

        # acceleration a from jackknife
        jv = jack[k]
        a = 0.0
        if len(jv) >= 2:
            m = jv.mean()
            d = m - jv
            denom = (d ** 2).sum()
            if denom > 0:
                a = float((d ** 3).sum() / (6.0 * denom ** 1.5))
        if np.isnan(a):
            a = 0.0
        res["a"] = a

        def _alpha(z):
            num = z0 + z
            adj = z0 + num / (1.0 - a * num)
            return float(norm.cdf(adj))

        alo, ahi = _alpha(z_lo), _alpha(z_hi)
        alo = min(max(alo, 0.0), 1.0)
        ahi = min(max(ahi, 0.0), 1.0)
        res["bca_lo"] = float(np.percentile(arr, 100.0 * alo))
        res["bca_hi"] = float(np.percentile(arr, 100.0 * ahi))
        results[k] = res

    results["_n_boot_effective"] = n_effective
    results["_n_clusters"] = n_clusters
    return results


def centering_fraction(point, lo, hi):
    """(point - lo) / (hi - lo). ~0.5 = centered; near 0 = point at bottom."""
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (point, lo, hi)):
        return NAN
    if hi == lo:
        return NAN
    return (point - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# data loading / joining (CLI)
# --------------------------------------------------------------------------- #
def load_folds_meta(csv_path):
    """patient_barcode -> {tss, race(canonical)} from brca_hospital_folds.csv."""
    meta = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            bc = r["patient_barcode"].strip()
            meta[bc] = {
                "tss": r.get("tss", "").strip(),
                "race": RACE_MAP.get(r.get("race", "").strip().lower()),
            }
    return meta


def load_and_join(preds_paths, folds_csv):
    """Pool preds JSONLs, join race+TSS from folds CSV, keep White/Black with a
    valid TSS. Returns (patients, stats)."""
    meta = load_folds_meta(folds_csv)
    patients, seen = [], set()
    n_rows = n_nojoin = n_notwb = n_notss = 0
    for path in preds_paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            pid = row["patient_id"]
            if pid in seen:
                continue
            m = meta.get(pid)
            if m is None:
                n_nojoin += 1
                continue
            if m["race"] not in ("White", "Black"):
                n_notwb += 1
                continue
            if not m["tss"]:
                n_notss += 1
                continue
            seen.add(pid)
            patients.append({
                "patient_id": pid,
                "y_true": int(row["y_true"]),
                "y_score": float(row["y_score"]),
                "race": m["race"],
                "tss": m["tss"],
            })
    stats = {"n_rows": n_rows, "n_kept": len(patients), "n_nojoin": n_nojoin,
             "n_not_white_black": n_notwb, "n_no_tss": n_notss}
    return patients, stats


def load_demographics(csv_path=DEFAULT_DEMOGRAPHICS_CSV):
    """patient_barcode -> normalized sex and numeric age."""
    meta = {}
    sex_map = {"female": "Female", "f": "Female",
               "male": "Male", "m": "Male"}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            bc = r["patient_barcode"].strip()
            sex_raw = r.get("sex", r.get("gender", "")).strip().lower()
            age_raw = r.get("age", r.get("age_years", "")).strip()
            try:
                age = float(age_raw)
            except (TypeError, ValueError):
                age = None
            meta[bc] = {"sex": sex_map.get(sex_raw), "age": age}
    return meta


def load_and_join_sensitive(preds_paths, folds_csv, sensitive_axis):
    """Join sex/age via demographics while retaining hospital-fold TSS."""
    if sensitive_axis == "race":
        patients, stats = load_and_join(preds_paths, folds_csv)
        stats["age_cutoff"] = None
        return patients, stats

    folds = load_folds_meta(folds_csv)
    demographics = load_demographics()
    candidates, seen = [], set()
    n_rows = n_nojoin = n_notss = 0
    for path in preds_paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            pid = row["patient_id"]
            if pid in seen:
                continue
            fold = folds.get(pid)
            demo = demographics.get(pid)
            if fold is None or demo is None:
                n_nojoin += 1
                continue
            if not fold["tss"]:
                n_notss += 1
                continue
            seen.add(pid)
            candidates.append({
                "patient_id": pid,
                "y_true": int(row["y_true"]),
                "y_score": float(row["y_score"]),
                "race": fold["race"],
                "tss": fold["tss"],
                "sex": demo["sex"],
                "age": demo["age"],
            })

    age_cutoff = None
    if sensitive_axis == "age":
        ages = [p["age"] for p in candidates if p["age"] is not None]
        age_cutoff = float(np.median(ages)) if ages else None
        for p in candidates:
            if p["age"] is not None and age_cutoff is not None:
                p["sensitive_group"] = (
                    "Younger" if p["age"] < age_cutoff else "Older"
                )
    else:
        for p in candidates:
            p["sensitive_group"] = p["sex"]

    groups = (("Female", "Male") if sensitive_axis == "sex"
              else ("Younger", "Older"))
    patients = [p for p in candidates if p.get("sensitive_group") in groups]
    stats = {
        "n_rows": n_rows,
        "n_kept": len(patients),
        "n_nojoin": n_nojoin,
        "n_not_sensitive_groups": len(candidates) - len(patients),
        "n_no_tss": n_notss,
        "age_cutoff": age_cutoff,
    }
    return patients, stats


# --------------------------------------------------------------------------- #
# CLI runner
# --------------------------------------------------------------------------- #
def _fmt(x, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", nargs="+", default=DEFAULT_PREDS)
    ap.add_argument("--folds-csv", default=DEFAULT_FOLDS_CSV)
    ap.add_argument("--boot-n", type=int, default=BOOT_N_DEFAULT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--sensitive-axis", choices=("race", "sex", "age"),
                    default="race")
    args = ap.parse_args()

    group_labels = {
        "race": ("White", "Black"),
        "sex": ("Female", "Male"),
        "age": ("Younger", "Older"),
    }
    reference_group, minority_group = group_labels[args.sensitive_axis]
    patients, st = load_and_join_sensitive(
        args.preds, args.folds_csv, args.sensitive_axis
    )
    pt = compute_metrics(patients, reference_group, minority_group)
    n_tss = len(_tss_clusters(patients))
    dropped_group = (
        f"non-WB race={st['n_not_white_black']}"
        if args.sensitive_axis == "race"
        else f"outside {args.sensitive_axis} groups={st['n_not_sensitive_groups']}"
    )
    print(f"[hh_metrics] pooled rows={st['n_rows']} kept={st['n_kept']} "
          f"(dropped: no-join={st['n_nojoin']}, {dropped_group}, "
          f"no-tss={st['n_no_tss']}); TSS clusters={n_tss}")
    if args.sensitive_axis == "age":
        print(f"[hh_metrics] age median cutoff (evaluated cohort) = "
              f"{st['age_cutoff']:.1f} years; Younger < cutoff, Older >= cutoff")
    print(f"[hh_metrics] {reference_group}: N={pt['n_white']} "
          f"events(pos)={pt['white_events']} | "
          f"{minority_group}: N={pt['n_black']} events(pos)={pt['black_events']}"
          + (f"  *** UNDERPOWERED ({minority_group} events < %d) ***" % EVENT_FLOOR
             if pt["black_events"] < EVENT_FLOOR else ""))
    tau_text = ("n/a" if pt["tau"] is None or np.isnan(pt["tau"])
                else f"{pt['tau']:.4f}")
    print(f"[hh_metrics] tau ({reference_group} 80%-spec) = {tau_text}  "
          f"=> FPR_{reference_group}={_fmt(pt['white_fpr'])} (target ~0.200)")

    ci = bootstrap_ci(
        patients, n_boot=args.boot_n, seed=args.seed,
        reference_group=reference_group, minority_group=minority_group
    )
    print(f"[hh_metrics] bootstrap draws effectively used (per metric shown "
          f"below); n_boot requested={args.boot_n}")

    # ---- Output 1: disparity table -----------------------------------------
    rows = [("signed FPR-disp", "fpr_disparity"),
            ("signed TPR-disp", "tpr_disparity"),
            ("EO (max)", "eo"),
            ("AUROC-gap", "auroc_gap")]
    print("\n=== TABLE 1: disparities (point | percentile CI | BCa CI | "
          "centering-frac[pct]) ===")
    hdr = f"{'metric':<16} {'point':>8} {'pct 95% CI':>20} {'BCa 95% CI':>20} {'cf(pct)':>8} {'cf(BCa)':>8} {'nboot':>7}"
    print(hdr)
    for label, k in rows:
        r = ci[k]
        cf_p = centering_fraction(r["point"], r["pct_lo"], r["pct_hi"])
        cf_b = centering_fraction(r["point"], r["bca_lo"], r["bca_hi"])
        pct = f"[{_fmt(r['pct_lo'],1)},{_fmt(r['pct_hi'],1)}]"
        bca = f"[{_fmt(r['bca_lo'],1)},{_fmt(r['bca_hi'],1)}]"
        print(f"{label:<16} {_fmt(r['point'],1):>8} {pct:>20} {bca:>20} "
              f"{_fmt(cf_p):>8} {_fmt(cf_b):>8} {r['n_used']:>7}")

    # ---- Output 2: per-group compact ---------------------------------------
    print("\n=== TABLE 2: per-group (point estimates) ===")
    print(f"{'group':<7} {'N':>5} {'events':>7} {'AUROC':>7} {'AUPRC':>7} "
          f"{'PPV':>7} {'cal.slope':>10} {'cal.int':>9} {'ECE':>7}")
    for g, label in (("white", reference_group), ("black", minority_group)):
        print(f"{label:<7} {pt['n_'+g]:>5} {pt[g+'_events']:>7} "
              f"{_fmt(pt[g+'_auroc']):>7} {_fmt(pt[g+'_auprc']):>7} "
              f"{_fmt(pt[g+'_ppv']):>7} {_fmt(pt[g+'_cal_slope'],1):>10} "
              f"{_fmt(pt[g+'_cal_intercept'],1):>9} {_fmt(pt[g+'_cal_ece']):>7}")
    print(f"(FPR/TPR at tau -> {reference_group} FPR={_fmt(pt['white_fpr'])} "
          f"TPR={_fmt(pt['white_tpr'])} | {minority_group} "
          f"FPR={_fmt(pt['black_fpr'])} TPR={_fmt(pt['black_tpr'])})")

    # ---- Output 3: GATE ----------------------------------------------------
    cf_fpr = centering_fraction(ci["fpr_disparity"]["point"],
                                ci["fpr_disparity"]["pct_lo"],
                                ci["fpr_disparity"]["pct_hi"])
    cf_tpr = centering_fraction(ci["tpr_disparity"]["point"],
                                ci["tpr_disparity"]["pct_lo"],
                                ci["tpr_disparity"]["pct_hi"])
    cf_eo = centering_fraction(ci["eo"]["point"], ci["eo"]["pct_lo"],
                               ci["eo"]["pct_hi"])
    signed_ok = (0.35 <= cf_fpr <= 0.65) and (0.35 <= cf_tpr <= 0.65)
    eo_skewed = cf_eo < 0.35
    verdict = "YES" if (signed_ok and eo_skewed) else "NO"
    print("\n=== GATE ===")
    print(f"signed FPR-disp centering(pct)={_fmt(cf_fpr)}, "
          f"signed TPR-disp centering(pct)={_fmt(cf_tpr)}, "
          f"max-EO centering(pct)={_fmt(cf_eo)}")
    print(f"GATE VERDICT: {verdict} -- signed disparities centered "
          f"(0.35-0.65) while max-EO is bottom-skewed (<0.35)? {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
