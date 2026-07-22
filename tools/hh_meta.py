#!/usr/bin/env python
"""hh_meta.py -- Cross-cohort meta-analysis over hh_arms_table.json files.

Reads one or more hh_arms_table.json (produced by hh_analysis.py), extracts
per-arm EO values, and performs a DerSimonian-Laird random-effects meta-analysis
across cohorts for three effect sizes:

  (a)  bake-in reduction    : EO(baseline) - EO(A_cancercond)
  (b)  post-hoc reduction   : EO(baseline) - EO(B_contrastive_labelcond)
  (c)  head-to-head gap     : EO(A_cancercond) - EO(B_contrastive_labelcond)

Reports pooled estimate, 95% CI, I^2 heterogeneity, and sign-consistency counts.
Cohorts whose baseline EO 95% CI excludes 0 (or whose point EO exceeds a stated
threshold) enter the confirmatory pool; others are reported as specificity controls.

When the source JSON lacks baseline EO CIs (current hh_arms_table.json schema),
baseline point EO is used and a flag is raised that CIs are needed.

Usage:
    python hh_meta.py --tables brca=results/hh_arms_table.json ucec=results/hh_arms_table_ucec.json --out results/hh_meta.json
"""
import argparse
import json
import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis
# ---------------------------------------------------------------------------

def _dl_meta(estimates, variances):
    """DerSimonian-Laird random-effects meta-analysis.

    Args:
        estimates: list of point estimates (float).
        variances: list of sampling variances (float).

    Returns dict with pooled estimate, CI, I^2, tau^2, weights, k.
    """
    k = len(estimates)
    if k == 0:
        return None

    if k == 1:
        # Single study: no between-study variance estimation
        se = math.sqrt(variances[0]) if variances[0] > 0 else 0.0
        return {
            "pooled": estimates[0],
            "se": se,
            "ci_95_lo": estimates[0] - 1.96 * se,
            "ci_95_hi": estimates[0] + 1.96 * se,
            "tau2": 0.0,
            "i2": 0.0,
            "weights": [1.0],
            "k": 1,
        }

    # Fixed-effect weights
    w_fe = [1.0 / v if v > 0 else 0.0 for v in variances]
    w_sum = sum(w_fe)
    if w_sum == 0:
        # All variances missing/infinite -> equal weights
        w_fe = [1.0] * k
        w_sum = k

    # Fixed-effect pooled
    mu_fe = sum(w * e for w, e in zip(w_fe, estimates)) / w_sum

    # Cochran's Q
    Q = sum(w * (e - mu_fe) ** 2 for w, e in zip(w_fe, estimates))
    df = k - 1

    # tau^2 (DL estimator)
    C = sum(w_fe) - sum(w ** 2 for w in w_fe) / w_sum if w_sum else 0.0
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0

    # Random-effects weights
    w_re = [1.0 / (v + tau2) for v in variances]
    w_re_sum = sum(w_re)

    # Pooled random-effects estimate
    pooled = sum(w * e for w, e in zip(w_re, estimates)) / w_re_sum if w_re_sum else 0.0
    se = math.sqrt(1.0 / w_re_sum) if w_re_sum else 0.0

    # Heterogeneity stats
    i2 = max(0.0, min(100.0, (Q - df) / Q * 100)) if Q > 0 else 0.0

    return {
        "pooled": pooled,
        "se": se,
        "ci_95_lo": pooled - 1.96 * se,
        "ci_95_hi": pooled + 1.96 * se,
        "tau2": tau2,
        "i2": i2,
        "weights": [w / w_re_sum if w_re_sum else 0.0 for w in w_re],
        "k": k,
    }


# ---------------------------------------------------------------------------
# Schema extraction from a single hh_arms_table.json
# ---------------------------------------------------------------------------

def _extract(table_path, cohort_name, eo_gate_threshold=0.10):
    """Extract EO metrics from one hh_arms_table.json.

    Returns a dict with:
        - cohort: name
        - baseline_eo: float
        - baseline_eo_ci_available: bool (True only if the JSON provided CI)
        - eo_by_arm: {arm: eo}
        - eo_reduction_by_arm: {arm: baseline_eo - eo(arm)}
        - confirmatory_contrast: eo_diff, CI, ci_excludes_zero (if present)
        - guardrails: {arm: pass/fail}
        - passes_baseline_gate: bool (True if baseline CI excludes 0 or EO > threshold)
    """
    with open(table_path) as f:
        data = json.load(f)

    arms_table = data.get("arms_table", {})
    baseline = arms_table.get("baseline", {})
    baseline_eo = baseline.get("eo", None)

    # EO per arm
    eo_by_arm = {}
    for arm_name, arm_data in arms_table.items():
        eo_by_arm[arm_name] = arm_data.get("eo", None)

    # EO reduction vs baseline per arm
    eo_reduction_by_arm = {}
    for arm_name, eo in eo_by_arm.items():
        if baseline_eo is not None and eo is not None:
            eo_reduction_by_arm[arm_name] = baseline_eo - eo
        else:
            eo_reduction_by_arm[arm_name] = None

    # Confirmatory contrast (if present)
    confirm = data.get("confirmatory_contrast", {})
    confirmatory = {
        "eo_diff": confirm.get("eo_diff"),
        "eo_a_full": confirm.get("eo_a_full"),
        "eo_b_full": confirm.get("eo_b_full"),
        "ci_95_lo": confirm.get("ci_95_lo"),
        "ci_95_hi": confirm.get("ci_95_hi"),
        "ci_excludes_zero": confirm.get("ci_excludes_zero"),
        "verdict": confirm.get("verdict"),
    }

    # Guardrails
    guardrails = {}
    for arm_name, gr_data in data.get("guardrail_checks", {}).items():
        guardrails[arm_name] = gr_data.get("pass", None)

    # Baseline-gap gate
    # The current hh_arms_table.json schema does NOT include baseline EO CI,
    # so we fall back to point estimate and flag that CIs are needed.
    baseline_ci_available = ("ci_95_lo" in baseline and "ci_95_hi" in baseline)
    if baseline_ci_available:
        passes_gate = (baseline.get("ci_95_lo", 0) > 0
                       or baseline.get("ci_95_hi", 0) < 0)
    else:
        # Fallback: flag baseline EO above threshold as confirmatory
        passes_gate = (baseline_eo is not None
                       and baseline_eo > eo_gate_threshold)

    # Per-effect REAL bootstrap variances (SE^2 from each contrast's 95% CI).
    #   head_to_head       : confirmatory_contrast (EO(A) - EO(B))       -- always in table
    #   bake_in_reduction  : bakein_reduction_ci   (EO(base) - EO(A))    -- re-run bootstrap CI
    #   posthoc_reduction  : posthoc_reduction_ci  (EO(base) - EO(B))    -- re-run bootstrap CI
    bakein_ci = data.get("bakein_reduction_ci", {})
    posthoc_ci = data.get("posthoc_reduction_ci", {})
    effect_var = {
        "head_to_head": _ci_to_variance(confirm.get("ci_95_lo"), confirm.get("ci_95_hi")),
        "bake_in_reduction": _ci_to_variance(bakein_ci.get("ci_95_lo"), bakein_ci.get("ci_95_hi")),
        "posthoc_reduction": _ci_to_variance(posthoc_ci.get("ci_95_lo"), posthoc_ci.get("ci_95_hi")),
    }

    return {
        "cohort": cohort_name,
        "table_path": str(table_path),
        "baseline_eo": baseline_eo,
        "baseline_eo_ci_available": baseline_ci_available,
        "eo_by_arm": eo_by_arm,
        "eo_reduction_by_arm": eo_reduction_by_arm,
        "confirmatory_contrast": confirmatory,
        "effect_var": effect_var,
        "guardrails": guardrails,
        "passes_baseline_gate": passes_gate,
    }


# ---------------------------------------------------------------------------
# Sign-consistency helper
# ---------------------------------------------------------------------------

def _ci_to_variance(ci_lo, ci_hi):
    """Sampling variance from a bootstrap 95% CI: SE=(hi-lo)/(2*1.96), var=SE^2.

    Returns None if either bound is missing (caller must not fall back to a
    placeholder -- a missing CI means the cohort bootstrap must be re-run).
    """
    if ci_lo is None or ci_hi is None:
        return None
    se = (ci_hi - ci_lo) / (2.0 * 1.96)
    return se * se


def _sign_consistency(values):
    """Count how many estimates are positive, negative, or zero."""
    pos = sum(1 for v in values if v is not None and v > 0)
    neg = sum(1 for v in values if v is not None and v < 0)
    zero = sum(1 for v in values if v is not None and v == 0)
    missing = sum(1 for v in values if v is None)
    return {"positive": pos, "negative": neg, "zero": zero, "missing": missing}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cross-cohort meta-analysis over hh_arms_table.json files")
    ap.add_argument("--tables", nargs="+", required=True,
                    help="cohort_name=path pairs, e.g. "
                         "brca=results/hh_arms_table.json ucec=results/hh_arms_table_ucec.json")
    ap.add_argument("--out", required=True,
                    help="Output JSON path for meta-analysis results")
    ap.add_argument("--eo-gate-threshold", type=float, default=0.10,
                    help="Baseline EO threshold for confirmatory gate when CI unavailable "
                         "(default 0.10 = 10 percentage-point FPR disparity)")
    args = ap.parse_args()

    # Parse table paths
    tables = {}
    for token in args.tables:
        if "=" not in token:
            ap.error(f"--tables entry must be name=path, got: {token}")
        name, path = token.split("=", 1)
        tables[name] = path

    # Extract per-cohort metrics
    cohorts = {}
    for name, path in tables.items():
        try:
            cohorts[name] = _extract(path, name, args.eo_gate_threshold)
        except FileNotFoundError:
            print(f"[hh_meta] ERROR: table not found for {name}: {path}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"[hh_meta] ERROR: invalid JSON for {name}: {e}", file=sys.stderr)
            sys.exit(1)

    # Partition into confirmatory pool vs specificity controls
    confirmatory = {n: c for n, c in cohorts.items() if c["passes_baseline_gate"]}
    specificity = {n: c for n, c in cohorts.items() if not c["passes_baseline_gate"]}

    # ---- Derive three effect sizes per cohort ----
    # (a) bake-in reduction: baseline_eo - eo(A_cancercond)
    # (b) post-hoc reduction: baseline_eo - eo(B_contrastive_labelcond)
    # (c) head-to-head: eo(A_cancercond) - eo(B_contrastive_labelcond)

    effect_sizes = {
        "bake_in_reduction": {},
        "posthoc_reduction": {},
        "head_to_head": {},
    }

    for name, c in cohorts.items():
        eo_base = c["baseline_eo"]
        eo_a = c["eo_by_arm"].get("A_cancercond")
        eo_b = c["eo_by_arm"].get("B_contrastive_labelcond")

        if eo_base is not None and eo_a is not None:
            effect_sizes["bake_in_reduction"][name] = eo_base - eo_a
        if eo_base is not None and eo_b is not None:
            effect_sizes["posthoc_reduction"][name] = eo_base - eo_b
        if eo_a is not None and eo_b is not None:
            effect_sizes["head_to_head"][name] = eo_a - eo_b

    # ---- Meta-analysis over confirmatory cohorts ----
    k_confirmatory = len(confirmatory)
    meta_results = {}

    for effect_name in ("bake_in_reduction", "posthoc_reduction", "head_to_head"):
        # Use each cohort's REAL bootstrap variance (SE^2 from its 95% CI) as the
        # per-study sampling variance in DL pooling. No placeholder: a cohort with
        # a missing CI is a hard error (its bootstrap must be re-run upstream).
        ests, vars_, se_used = [], [], {}
        for n in confirmatory:
            e = effect_sizes[effect_name].get(n)
            if e is None:
                continue
            var = cohorts[n]["effect_var"].get(effect_name)
            if var is None or var <= 0:
                sys.exit(f"[hh_meta] ERROR: missing/degenerate bootstrap CI for "
                         f"{n}/{effect_name}; re-run that cohort's paired "
                         f"TSS-cluster bootstrap to obtain a real SE.")
            ests.append(e)
            vars_.append(var)
            se_used[n] = math.sqrt(var)
        meta = _dl_meta(ests, vars_)

        # Sign consistency across ALL cohorts (not just confirmatory)
        all_ests = [effect_sizes[effect_name].get(n) for n in cohorts]
        sc = _sign_consistency(all_ests)

        meta_results[effect_name] = {
            "meta": meta,
            "sign_consistency": sc,
            "per_cohort": {n: effect_sizes[effect_name].get(n)
                           for n in cohorts},
            "per_cohort_se": se_used,
            "k_confirmatory": len(ests),
        }

    # ---- Build output ----
    output = {
        "meta_analysis_version": "1.0",
        "eo_gate_threshold": args.eo_gate_threshold,
        "k_total": len(cohorts),
        "k_confirmatory": k_confirmatory,
        "k_specificity_controls": len(specificity),
        "confirmatory_cohorts": list(confirmatory.keys()),
        "specificity_control_cohorts": list(specificity.keys()),
        "per_cohort": {},
        "meta_results": meta_results,
    }

    for name, c in cohorts.items():
        output["per_cohort"][name] = {
            "baseline_eo": c["baseline_eo"],
            "baseline_eo_ci_available": c["baseline_eo_ci_available"],
            "passes_baseline_gate": c["passes_baseline_gate"],
            "eo_by_arm": c["eo_by_arm"],
            "eo_reduction_by_arm": c["eo_reduction_by_arm"],
            "confirmatory_contrast": c["confirmatory_contrast"],
            "guardrails": c["guardrails"],
        }

    # ---- Print summary table ----
    print("=" * 80)
    print("HH META-ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    # Per-cohort table
    print(f"{'Cohort':<12} {'Baseline EO':>12} {'EO(A_cancer)':>14} "
          f"{'EO(B_label)':>13} {'Bake-in Red.':>14} {'Posthoc Red.':>14} "
          f"{'H2H A-B':>10} {'Gate':>6}")
    print("-" * 108)
    for name in cohorts:
        c = cohorts[name]
        eo_base = c["baseline_eo"]
        eo_a = c["eo_by_arm"].get("A_cancercond")
        eo_b = c["eo_by_arm"].get("B_contrastive_labelcond")
        bi = c["eo_reduction_by_arm"].get("A_cancercond")
        ph = c["eo_reduction_by_arm"].get("B_contrastive_labelcond")
        h2h = eo_a - eo_b if eo_a is not None and eo_b is not None else None
        gate = "Y" if c["passes_baseline_gate"] else "N"
        print(f"{name:<12} "
              f"{eo_base:>12.4f} "
              f"{eo_a:>14.4f} "
              f"{eo_b:>13.4f} "
              f"{bi:>14.4f} "
              f"{ph:>14.4f} "
              f"{h2h:>10.4f} "
              f"{gate:>6}")

    print()

    # Meta-analysis results
    if k_confirmatory < 3:
        print(f"[hh_meta] WARNING: k={k_confirmatory} confirmatory cohort(s); "
              f"meta-analysis underpowered at k<3. Reporting per-cohort only.")
        print()

    for effect_name, display_name in [
        ("bake_in_reduction", "Bake-in Reduction (baseline - A_cancercond)"),
        ("posthoc_reduction", "Post-hoc Reduction (baseline - B_contrastive_labelcond)"),
        ("head_to_head", "Head-to-Head (A_cancercond - B_contrastive_labelcond)"),
    ]:
        mr = meta_results[effect_name]
        meta = mr["meta"]
        sc = mr["sign_consistency"]
        print(f"--- {display_name} ---")
        if meta and meta["k"] > 0:
            print(f"  k={meta['k']}  pooled={meta['pooled']:.4f} "
                  f"[{meta['ci_95_lo']:.4f}, {meta['ci_95_hi']:.4f}]  "
                  f"I^2={meta['i2']:.1f}%  tau^2={meta['tau2']:.6f}")
        else:
            print("  (no estimates)")
        print(f"  sign consistency: {sc['positive']}+, {sc['negative']}-, "
              f"{sc['zero']}=, {sc['missing']} missing")
        print()

    # Baseline CI warning
    any_missing_ci = any(not c["baseline_eo_ci_available"] for c in cohorts.values())
    if any_missing_ci:
        print("[hh_meta] NOTE: baseline EO CIs are not available in the source JSON.")
        print("          Gate decisions used point EO > threshold heuristic.")
        print("          Recommended: add baseline EO bootstrap CIs to hh_arms_table.json.")
        print()

    # Specificity controls
    if specificity:
        print("Specificity controls (baseline gate NOT passed):")
        for name, c in specificity.items():
            print(f"  {name}: baseline_eo={c['baseline_eo']}")
        print()

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[hh_meta] wrote {args.out}")


if __name__ == "__main__":
    main()