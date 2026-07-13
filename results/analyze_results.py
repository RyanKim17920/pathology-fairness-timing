#!/usr/bin/env python3
"""
Finalized fairness analysis: pretraining-time vs post-hoc debiasing.

Task = CPTAC-NSCLC subtype (LUAD=0 / LSCC=1), single 206-patient external cohort.
25 per-patient prediction files in /data/ryan.kim/nanopath/results/preds/*.jsonl.

Produces:
  - results/stats.csv        (tidy long-format: point estimates + 95% CI + difference tests + BH-FDR)
  - results/figures/*.png     (colorblind-safe figures)
  - results/_summary.json     (headline numbers for the report)

Statistics:
  - 2000 patient-level bootstrap resamples, FIXED seed, shared index matrix (paired).
  - Percentile 95% CIs for overall AUC, AUCdelta (race/sex/age), ES-AUC, ECEdelta.
  - Paired bootstrap difference tests vs baseline and pretraining-vs-posthoc:
      two-sided bootstrap p = 2*min(frac<0, frac>0), 95% CI of the difference.
  - Benjamini-Hochberg FDR across all difference tests.

Reproducible: `python analyze_results.py`
"""
import json, glob, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- config
PREDS_DIR = "/data/ryan.kim/nanopath/results/preds"
OUT_DIR   = "/admin/home/ryan.kim/nt/results"
FIG_DIR   = os.path.join(OUT_DIR, "figures")
SEED      = 20240710
N_BOOT    = 2000
MIN_N     = 15          # adequate-power threshold for delta / ES / ECE gaps
ECE_BINS  = 10
METHODS   = ["fino", "dann", "contrastive", "pcgrad"]
RACEMODES = ["none", "weight", "resample"]
os.makedirs(FIG_DIR, exist_ok=True)

# Okabe-Ito colorblind-safe palette
OI = {"black":"#000000","orange":"#E69F00","skyblue":"#56B4E9","green":"#009E73",
      "yellow":"#F0E442","blue":"#0072B2","vermillion":"#D55E00","purple":"#CC79A7"}
ARM_COLOR = {"pretraining": OI["blue"], "post-hoc": OI["vermillion"], "baseline": OI["black"]}
METHOD_MARKER = {"fino":"o","dann":"s","contrastive":"^","pcgrad":"D","baseline":"*"}

# ----------------------------------------------------------------------------- load
def load(fn):
    rows = [json.loads(l) for l in open(fn)]
    return rows

base_rows = load(os.path.join(PREDS_DIR, "eval_fair-baseline.jsonl"))
PIDS = [r["patient_id"] for r in base_rows]
PID_IDX = {p:i for i,p in enumerate(PIDS)}
N = len(PIDS)

# demographic vectors (from baseline; verified identical across all files)
def _norm(v):  # normalise missing
    return v if v not in (None, "", "NA", "nan") else None
race = np.array([_norm(r["race"]) for r in base_rows], dtype=object)
sex  = np.array([_norm(r["sex"])  for r in base_rows], dtype=object)
age  = np.array([r["age"] if r["age"] is not None else np.nan for r in base_rows], dtype=float)
y    = np.array([int(r["y_true"]) for r in base_rows], dtype=int)

age_med = np.nanmedian(age)                     # 66.0
age_grp = np.where(np.isnan(age), None, np.where(age < age_med, "age<med", "age>=med")).astype(object)

# subgroup masks (boolean over N). Adequate = original-cohort n >= MIN_N.
GROUPS = {
    "race": {"White": race=="White", "Asian": race=="Asian", "Black": race=="Black"},
    "sex":  {"Male":  sex=="M",       "Female": sex=="F"},
    "age":  {"age<med": age_grp=="age<med", "age>=med": age_grp=="age>=med"},
}
ADEQUATE = {attr:{g:(m.sum()>=MIN_N) for g,m in gs.items()} for attr,gs in GROUPS.items()}
GROUP_N  = {attr:{g:int(m.sum()) for g,m in gs.items()} for attr,gs in GROUPS.items()}

# ----------------------------------------------------------------------------- runs
def run_list():
    runs = []  # (run_id, arm, method, racemode, filepath)
    runs.append(("baseline", "baseline", "baseline", "none",
                 os.path.join(PREDS_DIR, "eval_fair-baseline.jsonl")))
    for m in METHODS:
        for rm in RACEMODES:
            runs.append((f"eval_fair-{m}-{rm}", "pretraining", m, rm,
                         os.path.join(PREDS_DIR, f"eval_fair-{m}-{rm}.jsonl")))
    for m in METHODS:
        for rm in RACEMODES:
            runs.append((f"posthoc_{m}_{rm}", "post-hoc", m, rm,
                         os.path.join(PREDS_DIR, f"posthoc_{m}_{rm}.jsonl")))
    return runs

RUNS = run_list()

# score matrix aligned to PIDS order
SCORES = {}
for rid, arm, m, rm, fp in RUNS:
    s = np.full(N, np.nan)
    for r in load(fp):
        s[PID_IDX[r["patient_id"]]] = r["y_score"]
    assert not np.isnan(s).any(), f"missing scores in {rid}"
    SCORES[rid] = s

# ----------------------------------------------------------------------------- metrics
def auc(labels, scores):
    """Mann-Whitney AUROC with tie handling. NaN if a class is absent."""
    labels = np.asarray(labels); scores = np.asarray(scores)
    npos = int(labels.sum()); nneg = int(len(labels) - npos)
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    # average ranks for ties
    ranks[order] = np.arange(1, len(scores)+1, dtype=float)
    # adjust ties to mean rank
    # find groups of equal scores
    uniq, inv, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    if len(uniq) != len(scores):
        cum = np.cumsum(counts)
        start = cum - counts
        mean_rank = (start + cum + 1) / 2.0   # 1-based mean rank per unique value
        ranks[order] = mean_rank[inv]
    r_pos = ranks[labels == 1].sum()
    return (r_pos - npos*(npos+1)/2.0) / (npos*nneg)

def ece(labels, scores, bins=ECE_BINS):
    """Expected calibration error, equal-width bins on [0,1]."""
    labels = np.asarray(labels, float); scores = np.asarray(scores, float)
    if len(labels) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, bins+1)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins-1)
    tot = 0.0
    n = len(labels)
    for b in range(bins):
        m = idx == b
        if m.any():
            conf = scores[m].mean()
            acc = labels[m].mean()
            tot += (m.sum()/n) * abs(conf - acc)
    return tot

def compute_point(scores):
    """All point metrics for one run on the full cohort."""
    out = {}
    out["auc_overall"] = auc(y, scores)
    for attr, gs in GROUPS.items():
        adeq_aucs = []
        adeq_eces = []
        for g, mask in gs.items():
            a = auc(y[mask], scores[mask])
            e = ece(y[mask], scores[mask])
            out[f"auc_{attr}_{g}"] = a
            out[f"ece_{attr}_{g}"] = e
            if ADEQUATE[attr][g] and not np.isnan(a):
                adeq_aucs.append(a)
            if ADEQUATE[attr][g] and not np.isnan(e):
                adeq_eces.append(e)
        aucd = (max(adeq_aucs)-min(adeq_aucs)) if len(adeq_aucs) >= 2 else np.nan
        eced = (max(adeq_eces)-min(adeq_eces)) if len(adeq_eces) >= 2 else np.nan
        out[f"aucdelta_{attr}"] = aucd
        out[f"ecedelta_{attr}"] = eced
        # ES-AUC over adequate subgroups
        ov = out["auc_overall"]
        denom = 1.0 + sum(abs(ov - a) for a in adeq_aucs)
        out[f"esauc_{attr}"] = ov/denom if len(adeq_aucs) >= 2 else np.nan
    return out

POINT = {rid: compute_point(SCORES[rid]) for rid,_,_,_,_ in RUNS}

# ----------------------------------------------------------------------------- bootstrap
rng = np.random.default_rng(SEED)
BOOT_IDX = rng.integers(0, N, size=(N_BOOT, N))   # shared, paired across all runs

# metrics we bootstrap (scalar per run per resample)
BOOT_METRICS = (["auc_overall"]
                + [f"aucdelta_{a}" for a in GROUPS]
                + [f"ecedelta_{a}" for a in GROUPS]
                + [f"esauc_{a}" for a in GROUPS])

def boot_run(scores):
    """Return dict metric -> array(N_BOOT) of bootstrap values for one run."""
    res = {mname: np.full(N_BOOT, np.nan) for mname in BOOT_METRICS}
    yb_all = y[BOOT_IDX]                # (B,N)
    sb_all = scores[BOOT_IDX]           # (B,N)
    # precompute resampled group masks per attr
    gmask = {}
    for attr, gs in GROUPS.items():
        gmask[attr] = {g: mask[BOOT_IDX] for g,mask in gs.items()}
    for b in range(N_BOOT):
        yb = yb_all[b]; sb = sb_all[b]
        res["auc_overall"][b] = auc(yb, sb)
        ov = res["auc_overall"][b]
        for attr, gs in GROUPS.items():
            adeq_aucs=[]; adeq_eces=[]
            for g in gs:
                mm = gmask[attr][g][b]
                if ADEQUATE[attr][g] and mm.sum() > 0:
                    a = auc(yb[mm], sb[mm]); e = ece(yb[mm], sb[mm])
                    if not np.isnan(a): adeq_aucs.append(a)
                    if not np.isnan(e): adeq_eces.append(e)
            res[f"aucdelta_{attr}"][b] = (max(adeq_aucs)-min(adeq_aucs)) if len(adeq_aucs)>=2 else np.nan
            res[f"ecedelta_{attr}"][b] = (max(adeq_eces)-min(adeq_eces)) if len(adeq_eces)>=2 else np.nan
            if len(adeq_aucs)>=2 and not np.isnan(ov):
                res[f"esauc_{attr}"][b] = ov/(1.0+sum(abs(ov-a) for a in adeq_aucs))
    return res

print("Bootstrapping %d runs x %d resamples..." % (len(RUNS), N_BOOT))
BOOT = {}
for rid,_,_,_,_ in RUNS:
    BOOT[rid] = boot_run(SCORES[rid])
print("done bootstrap")

def ci(arr):
    a = arr[~np.isnan(arr)]
    if len(a) == 0: return (np.nan, np.nan)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

def paired_diff_test(arr_a, arr_b):
    """Diff = a - b, paired bootstrap. Returns (point_est, lo, hi, p)."""
    d = arr_a - arr_b
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    frac_lt = np.mean(d < 0); frac_gt = np.mean(d > 0)
    p = 2.0 * min(frac_lt, frac_gt)
    p = min(1.0, p)
    return (float(np.median(d)), float(lo), float(hi), float(p))

# ----------------------------------------------------------------------------- difference tests
# Diff metrics of interest: aucdelta per attr (fairness gap) + auc_overall (accuracy cost)
DIFF_METRICS = [f"aucdelta_{a}" for a in GROUPS] + ["auc_overall"]

diff_rows = []  # collected for stats.csv & FDR
# (1) vs baseline: each debiased run - baseline
for rid, arm, m, rm, fp in RUNS:
    if rid == "baseline": continue
    for met in DIFF_METRICS:
        # point diff from real data
        pt_diff = POINT[rid][met] - POINT["baseline"][met]
        med, lo, hi, p = paired_diff_test(BOOT[rid][met], BOOT["baseline"][met])
        diff_rows.append(dict(comparison="vs_baseline", run=rid, arm=arm, method=m, racemode=rm,
                              metric=met, point=pt_diff, boot_med=med, ci_lo=lo, ci_hi=hi, p_raw=p))
# (2) pretraining vs post-hoc, per method x racemode: eval_fair - posthoc
for m in METHODS:
    for rm in RACEMODES:
        pre = f"eval_fair-{m}-{rm}"; post = f"posthoc_{m}_{rm}"
        for met in DIFF_METRICS:
            pt_diff = POINT[pre][met] - POINT[post][met]
            med, lo, hi, p = paired_diff_test(BOOT[pre][met], BOOT[post][met])
            diff_rows.append(dict(comparison="vs_timing", run=f"{m}_{rm}", arm="pretraining-minus-posthoc",
                                  method=m, racemode=rm, metric=met,
                                  point=pt_diff, boot_med=med, ci_lo=lo, ci_hi=hi, p_raw=p))

# Benjamini-Hochberg FDR across all difference tests
pvals = np.array([r["p_raw"] for r in diff_rows], float)
valid = ~np.isnan(pvals)
order = np.argsort(np.where(valid, pvals, np.inf))
mtests = valid.sum()
bh_sig = np.zeros(len(diff_rows), bool)
p_bh = np.full(len(diff_rows), np.nan)
ranked = [i for i in order if valid[i]]
# BH adjusted p-values
prev = 1.0
for k in range(len(ranked)-1, -1, -1):
    i = ranked[k]
    adj = pvals[i] * mtests / (k+1)
    prev = min(prev, adj)
    p_bh[i] = min(prev, 1.0)
for i in range(len(diff_rows)):
    if valid[i]:
        bh_sig[i] = p_bh[i] < 0.05
for i,r in enumerate(diff_rows):
    r["p_bh"] = float(p_bh[i]) if not np.isnan(p_bh[i]) else np.nan
    r["fdr_sig"] = bool(bh_sig[i])

# ----------------------------------------------------------------------------- write stats.csv (tidy long)
import csv
csv_path = os.path.join(OUT_DIR, "stats.csv")
POINT_METRICS = (["auc_overall"]
                 + [f"auc_{a}_{g}" for a in GROUPS for g in GROUPS[a]]
                 + [f"aucdelta_{a}" for a in GROUPS]
                 + [f"esauc_{a}" for a in GROUPS]
                 + [f"ecedelta_{a}" for a in GROUPS]
                 + [f"ece_{a}_{g}" for a in GROUPS for g in GROUPS[a]])
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["comparison","run","arm","method","racemode","metric",
                "point_estimate","boot_median","ci_lo","ci_hi","p_raw","p_bh","fdr_sig"])
    # point metrics (with CI where bootstrapped)
    for rid, arm, m, rm, fp in RUNS:
        for met in POINT_METRICS:
            pe = POINT[rid].get(met, np.nan)
            if met in BOOT[rid]:
                lo, hi = ci(BOOT[rid][met])
            else:
                lo, hi = (np.nan, np.nan)
            w.writerow(["point", rid, arm, m, rm, met,
                        f"{pe:.6f}" if not (pe is None or np.isnan(pe)) else "",
                        "", f"{lo:.6f}" if not np.isnan(lo) else "",
                        f"{hi:.6f}" if not np.isnan(hi) else "", "", "", ""])
    # difference tests
    for r in diff_rows:
        w.writerow([r["comparison"], r["run"], r["arm"], r["method"], r["racemode"], r["metric"],
                    f"{r['point']:.6f}" if not np.isnan(r["point"]) else "",
                    f"{r['boot_med']:.6f}" if not np.isnan(r["boot_med"]) else "",
                    f"{r['ci_lo']:.6f}" if not np.isnan(r["ci_lo"]) else "",
                    f"{r['ci_hi']:.6f}" if not np.isnan(r["ci_hi"]) else "",
                    f"{r['p_raw']:.5f}" if not np.isnan(r["p_raw"]) else "",
                    f"{r['p_bh']:.5f}" if not np.isnan(r["p_bh"]) else "",
                    int(r["fdr_sig"])])
print("wrote", csv_path)

# ----------------------------------------------------------------------------- helpers for figures
def err(pe, lo, hi):
    """Non-negative errorbar deltas (point can fall outside percentile CI)."""
    return [[max(0.0, pe-lo)], [max(0.0, hi-pe)]]

def run_meta(rid):
    for r in RUNS:
        if r[0]==rid: return r
    return None

def label_short(m, rm):
    return f"{m}/{rm}"

# ============================ FIGURE 1 & 4a: accuracy-fairness scatter (race/sex/age)
def scatter_acc_fair(attr, fname, title):
    fig, ax = plt.subplots(figsize=(8.5,6.2))
    for rid, arm, m, rm, fp in RUNS:
        x = POINT[rid]["auc_overall"]; ydv = POINT[rid][f"aucdelta_{attr}"]
        if np.isnan(ydv): continue
        xlo,xhi = ci(BOOT[rid]["auc_overall"])
        ylo,yhi = ci(BOOT[rid][f"aucdelta_{attr}"])
        if arm=="baseline":
            color=ARM_COLOR["baseline"]; mk="*"; ms=320; z=5; ec="k"
        else:
            color=ARM_COLOR[arm]; mk=METHOD_MARKER[m]; ms=95; z=3; ec="k"
        ax.errorbar(x, ydv, xerr=err(x,xlo,xhi), yerr=err(ydv,ylo,yhi),
                    fmt="none", ecolor=color, elinewidth=0.8, alpha=0.35, zorder=z-1)
        ax.scatter(x, ydv, marker=mk, s=ms, c=color, edgecolors=ec, linewidths=0.7, zorder=z,
                   label=None)
    bx = POINT["baseline"]["auc_overall"]; by = POINT["baseline"][f"aucdelta_{attr}"]
    ax.axhline(by, ls=":", c="grey", lw=1, zorder=0)
    ax.axvline(bx, ls=":", c="grey", lw=1, zorder=0)
    ax.set_xlabel("Overall AUROC (higher = more accurate)")
    ax.set_ylabel(f"{attr} AUCΔ (max−min subgroup AUROC; lower = fairer)")
    ax.set_title(title)
    handles = [Line2D([0],[0],marker="*",color="w",markerfacecolor=OI["black"],markersize=16,label="baseline"),
               Line2D([0],[0],marker="s",color="w",markerfacecolor=ARM_COLOR["pretraining"],markersize=11,label="pretraining arm"),
               Line2D([0],[0],marker="s",color="w",markerfacecolor=ARM_COLOR["post-hoc"],markersize=11,label="post-hoc arm")]
    for m in METHODS:
        handles.append(Line2D([0],[0],marker=METHOD_MARKER[m],color="w",markerfacecolor="grey",
                              markeredgecolor="k",markersize=10,label=m))
    ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,fname), dpi=170); plt.close(fig)

scatter_acc_fair("race","fig1_scatter_race.png","Accuracy–Fairness trade-off (RACE, White-vs-Asian)")
scatter_acc_fair("sex","fig4a_scatter_sex.png","Accuracy–Fairness trade-off (SEX)")
scatter_acc_fair("age","fig4b_scatter_age.png","Accuracy–Fairness trade-off (AGE, median-split)")

# ============================ FIGURE 2 & 4c/d: forest plot AUCdelta with CI
def forest(attr, fname, title):
    # order: baseline, then pretraining (method x rm), then posthoc
    ordered = [r for r in RUNS if r[1]=="baseline"] \
            + [r for r in RUNS if r[1]=="pretraining"] \
            + [r for r in RUNS if r[1]=="post-hoc"]
    ys=[]; labels=[]; colors=[]
    for i,(rid,arm,m,rm,fp) in enumerate(ordered):
        ys.append(i); labels.append("baseline" if arm=="baseline" else f"[{arm[:4]}] {m}/{rm}")
        colors.append(ARM_COLOR[arm])
    fig, ax = plt.subplots(figsize=(8.2, 9))
    baseval = POINT["baseline"][f"aucdelta_{attr}"]
    ax.axvline(baseval, ls="--", c=OI["black"], lw=1.2, label="baseline gap")
    for i,(rid,arm,m,rm,fp) in enumerate(ordered):
        pe = POINT[rid][f"aucdelta_{attr}"]
        lo,hi = ci(BOOT[rid][f"aucdelta_{attr}"])
        ax.errorbar(pe, i, xerr=err(pe,lo,hi), fmt=METHOD_MARKER.get(m,"o") if arm!="baseline" else "*",
                    color=colors[i], ecolor=colors[i], ms=11 if arm!="baseline" else 17,
                    capsize=2.5, elinewidth=1.2, markeredgecolor="k", markeredgewidth=0.6)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"{attr} AUCΔ (95% CI); lower = fairer")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    # arm separators
    n_pre = sum(1 for r in RUNS if r[1]=="pretraining")
    ax.axhline(0.5, color="grey", lw=0.6); ax.axhline(0.5+n_pre, color="grey", lw=0.6)
    handles=[Line2D([0],[0],color=ARM_COLOR["pretraining"],lw=6,label="pretraining"),
             Line2D([0],[0],color=ARM_COLOR["post-hoc"],lw=6,label="post-hoc"),
             Line2D([0],[0],ls="--",color="k",label="baseline gap")]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,fname), dpi=170); plt.close(fig)

forest("race","fig2_forest_race.png","Forest: RACE AUCΔ by run (White-vs-Asian)")
forest("sex","fig4c_forest_sex.png","Forest: SEX AUCΔ by run")
forest("age","fig4d_forest_age.png","Forest: AGE AUCΔ by run")

# ============================ FIGURE 3: pretraining-vs-posthoc paired, per method x racemode
def paired_fig(attr, fname, title):
    cells = [(m,rm) for m in METHODS for rm in RACEMODES]
    fig, ax = plt.subplots(figsize=(11,5.6))
    xpos = np.arange(len(cells)); wdt=0.36
    for j,(m,rm) in enumerate(cells):
        pre=f"eval_fair-{m}-{rm}"; post=f"posthoc_{m}_{rm}"
        for off,rid,arm in [(-wdt/2,pre,"pretraining"),(wdt/2,post,"post-hoc")]:
            pe=POINT[rid][f"aucdelta_{attr}"]; lo,hi=ci(BOOT[rid][f"aucdelta_{attr}"])
            ax.bar(j+off, pe, wdt, color=ARM_COLOR[arm], alpha=0.85,
                   yerr=err(pe,lo,hi), capsize=2.5, ecolor="k",
                   error_kw=dict(elinewidth=0.9))
    baseval=POINT["baseline"][f"aucdelta_{attr}"]
    ax.axhline(baseval, ls="--", c="k", lw=1.1, label="baseline gap")
    ax.set_xticks(xpos); ax.set_xticklabels([f"{m}\n{rm}" for m,rm in cells], fontsize=7.5)
    ax.set_ylabel(f"{attr} AUCΔ (95% CI)")
    ax.set_title(title)
    handles=[Line2D([0],[0],color=ARM_COLOR["pretraining"],lw=8,label="pretraining"),
             Line2D([0],[0],color=ARM_COLOR["post-hoc"],lw=8,label="post-hoc"),
             Line2D([0],[0],ls="--",color="k",label="baseline gap")]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,fname), dpi=170); plt.close(fig)

paired_fig("race","fig3_paired_race.png","Pretraining vs Post-hoc paired (RACE AUCΔ)")
paired_fig("sex","fig3b_paired_sex.png","Pretraining vs Post-hoc paired (SEX AUCΔ)")
paired_fig("age","fig3c_paired_age.png","Pretraining vs Post-hoc paired (AGE AUCΔ)")

# ============================ FIGURE 5: ES-AUC (race) and overall AUC bars
def bars_esauc_auc(fname):
    ordered=[r for r in RUNS]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,7.5))
    labels=["baseline" if r[1]=="baseline" else f"{r[2]}/{r[3]}" for r in ordered]
    colors=[ARM_COLOR[r[1]] for r in ordered]
    xs=np.arange(len(ordered))
    for ax,met,ttl,lohi in [(ax1,"esauc_race","ES-AUC (RACE, higher = fairer+accurate)",True),
                            (ax2,"auc_overall","Overall AUROC (higher = more accurate)",True)]:
        vals=[POINT[r[0]][met] for r in ordered]
        los=[ci(BOOT[r[0]][met])[0] for r in ordered]
        his=[ci(BOOT[r[0]][met])[1] for r in ordered]
        yerr=[[max(0.0,v-l) for v,l in zip(vals,los)],[max(0.0,h-v) for v,h in zip(vals,his)]]
        ax.bar(xs,vals,color=colors,alpha=0.85,yerr=yerr,capsize=2,ecolor="k",error_kw=dict(elinewidth=0.8))
        ax.axhline(POINT["baseline"][met], ls="--", c="k", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(labels,rotation=90,fontsize=7.5)
        ax.set_title(ttl); ax.grid(axis="y",alpha=0.25)
        lo=min(los); ax.set_ylim(max(0.0,lo-0.03), 1.0)
    handles=[Line2D([0],[0],color=ARM_COLOR["pretraining"],lw=8,label="pretraining"),
             Line2D([0],[0],color=ARM_COLOR["post-hoc"],lw=8,label="post-hoc"),
             Line2D([0],[0],color=ARM_COLOR["baseline"],lw=8,label="baseline"),
             Line2D([0],[0],ls="--",color="k",label="baseline level")]
    ax1.legend(handles=handles,fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,fname),dpi=170); plt.close(fig)

bars_esauc_auc("fig5_esauc_auc_bars.png")
print("wrote figures to", FIG_DIR)

# ----------------------------------------------------------------------------- summary json for report
summary = {
    "config": dict(seed=SEED, n_boot=N_BOOT, n_patients=int(N), age_median=float(age_med),
                   min_n=MIN_N, ece_bins=ECE_BINS),
    "group_n": GROUP_N,
    "adequate": {a:{g:bool(v) for g,v in gs.items()} for a,gs in ADEQUATE.items()},
    "point": {rid: {k:(None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(float(v),4))
                    for k,v in POINT[rid].items()} for rid,_,_,_,_ in RUNS},
    "point_ci": {rid: {met:[round(ci(BOOT[rid][met])[0],4), round(ci(BOOT[rid][met])[1],4)]
                       for met in BOOT_METRICS} for rid,_,_,_,_ in RUNS},
    "diff_tests": [dict(comparison=r["comparison"], run=r["run"], arm=r["arm"], method=r["method"],
                        racemode=r["racemode"], metric=r["metric"],
                        point=round(r["point"],4) if not np.isnan(r["point"]) else None,
                        ci_lo=round(r["ci_lo"],4) if not np.isnan(r["ci_lo"]) else None,
                        ci_hi=round(r["ci_hi"],4) if not np.isnan(r["ci_hi"]) else None,
                        p_raw=round(r["p_raw"],5) if not np.isnan(r["p_raw"]) else None,
                        p_bh=round(r["p_bh"],5) if not np.isnan(r["p_bh"]) else None,
                        fdr_sig=bool(r["fdr_sig"])) for r in diff_rows],
    "n_diff_tests": int(mtests),
}
with open(os.path.join(OUT_DIR,"_summary.json"),"w") as f:
    json.dump(summary, f, indent=1)
print("wrote _summary.json")

# ----------------------------------------------------------------------------- console headline
def show(rid):
    p=POINT[rid]
    print(f"\n{rid}: AUC={p['auc_overall']:.3f} raceAUCd={p['aucdelta_race']:.3f} "
          f"sexAUCd={p['aucdelta_sex']:.3f} ageAUCd={p['aucdelta_age']:.3f} ESauc_race={p['esauc_race']:.3f}")
print("\n=== POINT ESTIMATES ===")
for rid,_,_,_,_ in RUNS: show(rid)
print("\n=== SIGNIFICANT (FDR<0.05) vs-baseline AUCdelta reductions ===")
for r in diff_rows:
    if r["comparison"]=="vs_baseline" and r["metric"].startswith("aucdelta") and r["fdr_sig"] and r["point"]<0:
        print(f"  {r['run']:28s} {r['metric']:16s} d={r['point']:+.3f} CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_raw']:.4f} pBH={r['p_bh']:.4f}")
print("\n=== SIGNIFICANT (FDR<0.05) pretraining-vs-posthoc diffs ===")
for r in diff_rows:
    if r["comparison"]=="vs_timing" and r["fdr_sig"]:
        print(f"  {r['run']:22s} {r['metric']:16s} d(pre-post)={r['point']:+.3f} CI[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_raw']:.4f} pBH={r['p_bh']:.4f}")
print("\nDONE.")
