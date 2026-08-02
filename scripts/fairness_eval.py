#!/usr/bin/env python
"""fairness_eval.py -- standalone fairness evaluation harness for pathology
foundation-model (JEPA / DINOv2) checkpoints.

Given a trained backbone checkpoint and a binary task, this tool:
  1. embeds tiles with the *same* code path as probe.py (DinoV2ViT.probe_features
     on ((x - mean) / std) CLS-token features),
  2. mean-pools tile embeddings to the PATIENT level,
  3. joins patient demographics (race / sex / age),
  4. runs cross-validated (or external-test) logistic-regression probing,
  5. reports OVERALL and PER-SUBGROUP fairness metrics.

It is the eval side of a pretraining-vs-post-hoc fairness study and works for
in-distribution TCGA tasks (nsclc / glioma / brca) and the external CPTAC task
(cptac_nsclc), in either internal-CV or external-test mode.

--------------------------------------------------------------------------------
The backbone and pooling path mirrors the bundled pretraining probe:
  * Backbone build + load  (probe.py:975-982, run_probe_job):
        model = DinoV2ViT(variant=cfg["model"]["type"]).eval()
        state_key = {"ema": "model_ema", "model": "model"}[cfg["probe"]["model_weights"]]
        model.load_state_dict(checkpoint[state_key], strict=True)
    with mean/std from cfg["data"] reshaped (1,3,1,1)  (probe.py:983-984).
  * Tile -> CLS embedding + per-slide mean pool  (probe.py:271-309, embed_slide_dataset
    and 683-688, inline_surgen_ras_auc):
        e = model.probe_features((x - mean) / std)          # -> x_norm_clstoken
        sums.index_add_(0, slide_idx, e); X = sums / counts  # mean pool
    probe_features is model.py:182-183 -> self(x)["x_norm_clstoken"].
  * LR probing (probe.py:690-696, inline_surgen_ras_auc):
        StratifiedKFold(n_splits, shuffle=True, random_state=1337)
        LogisticRegression(C, class_weight="balanced", max_iter=5000,
                           random_state=0, solver="liblinear",
                           dual=X.shape[0] < X.shape[1])
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    C grid = (0.001,0.01,0.1,0.5,1.0,10.0,100.0); best C = argmax mean fold AUC.
  * Transform = model.probe_transforms(): Resize((224,224), antialias) + ToTensor
    (model.py:26-29); normalization done in-loop as (x-mean)/std, NOT in transform.

--------------------------------------------------------------------------------
Fairness metric definitions
  * AUC-delta (AUCd)     = max_g AUC_g - min_g AUC_g over eligible subgroups.
  * Equity-Scaled AUC    (FairPath / Harvard-Ophthalmology, Luo et al. 2023,
    "Harvard Glaucoma Fairness"; Tian et al. FairCLIP / FairVision):
        ES-AUC = AUC_overall / (1 + sum_g |AUC_overall - AUC_g|)
    summed over the sensitive subgroups g of one attribute. Perfectly equal
    subgroups -> ES-AUC == AUC_overall; disparity shrinks it.
  * ECE (10-bin, equal-width): ECE = sum_b (n_b/N) * |acc_b - conf_b|.
    ECE-delta (ECEd) = max_g ECE_g - min_g ECE_g over eligible subgroups.
  * Subgroups with too few total patients OR too few patients in either outcome
    class are reported as having insufficient support and excluded from gap
    metrics. These thresholds are heuristics, not a formal power analysis.
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from pathology_fairness.data_contracts import sha256_file, validate_dataset_receipt

# ------------------------------------------------------------------ constants
LR_CS = (0.001, 0.01, 0.1, 0.5, 1.0, 10.0, 100.0)  # probe.py SURGEN_LR_CS
LR_MAX_ITER = 5000
LR_SOLVER = "liblinear"
FOLD_SEED = 1337
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_VARIANT = "dinov2_vits14_reg"
DEFAULT_AGE_CUTOFF = 65.0
IMAGE_COL_CANDIDATES = ("jpeg", "image", "image_bytes")
RACE_MAP = {"white": "White", "black": "Black",
            "black or african american": "Black", "asian": "Asian"}
SEX_MAP = {"male": "M", "female": "F"}
WORKTREE = str((Path(__file__).resolve().parents[1] / "pretraining").resolve())

# --task -> HuggingFace cohort folder (consumed by hf_tiles.pull, which fetches
# <repo>/<cohort>/* via snapshot_download). In-distribution tasks use the local
# directory supplied by the caller and therefore map to None.
TASK_TO_COHORT = {
    "nsclc": None,
    "glioma": None,
    "brca": None,
    "cptac_nsclc": "cptac_lung",  # external CPTAC LUAD-vs-LSCC cohort on HF
}


# ============================================================== backbone (probe.py path)
def build_backbone(checkpoint_path, device, variant_override=None,
                   model_dir=WORKTREE, allow_random_init=False):
    """Replicates probe.py run_probe_job backbone load. Returns
    (model, mean(1,3,1,1), std(1,3,1,1), info).

    Random initialization is available only through an explicit smoke-test
    opt-in. A missing requested checkpoint always fails before model creation.
    """
    if checkpoint_path is None:
        if not allow_random_init:
            raise ValueError(
                "a checkpoint is required; pass --allow-random-init only for "
                "a non-scientific plumbing smoke test"
            )
    else:
        checkpoint_path = str(Path(checkpoint_path).expanduser())
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    import torch

    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from model import DinoV2ViT  # noqa: E402  (probe.py:951)

    random_init = checkpoint_path is None
    variant = variant_override or DEFAULT_VARIANT
    mean_v, std_v, weights = IMAGENET_MEAN, IMAGENET_STD, "random"

    if random_init:
        model = DinoV2ViT(variant=variant).to(device).eval()
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {})
        variant = variant_override or cfg.get("model", {}).get("type", DEFAULT_VARIANT)
        data_cfg = cfg.get("data", {})
        mean_v = data_cfg.get("mean", IMAGENET_MEAN)
        std_v = data_cfg.get("std", IMAGENET_STD)
        # probe.py:977 -> {"ema":"model_ema","model":"model"}[cfg["probe"]["model_weights"]]
        wsel = str(cfg.get("probe", {}).get("model_weights", "ema"))
        state_key = {"ema": "model_ema", "model": "model"}.get(wsel, "model_ema")
        if state_key not in ckpt:  # fall back if that key isn't present
            state_key = "model_ema" if "model_ema" in ckpt else "model"
        weights = state_key
        model = DinoV2ViT(variant=variant).to(device).eval()
        model.load_state_dict(ckpt[state_key], strict=True)  # probe.py:978
        del ckpt

    for p in model.parameters():
        p.requires_grad = False
    mean = torch.tensor(mean_v, device=device).view(1, 3, 1, 1)  # probe.py:983
    std = torch.tensor(std_v, device=device).view(1, 3, 1, 1)  # probe.py:984
    info = {"variant": variant, "weights": weights, "random_init": random_init,
            "mean": mean_v, "std": std_v, "embed_dim": int(model.embed_dim)}
    return model, mean, std, info


def _transform():
    from torchvision import transforms  # probe.py model.py:26-29
    return transforms.Compose([transforms.Resize((224, 224), antialias=True),
                               transforms.ToTensor()])


def embed_patients(model, mean, std, device, tiles, patient_ids, batch_size=128, log=print):
    """tiles: list of (patient_index, jpeg_bytes). Mean-pools CLS features per
    patient exactly like probe.embed_slide_dataset (probe.py:299-309)."""
    import torch
    from PIL import Image

    tf = _transform()
    n = len(patient_ids)
    embed_dim = int(model.embed_dim)
    sums = np.zeros((n, embed_dim), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())
    done = 0
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start:start + batch_size]
            xs, idxs = [], []
            for pidx, jpg in chunk:
                try:
                    img = Image.open(io.BytesIO(jpg)).convert("RGB")
                except Exception:
                    continue
                xs.append(tf(img))
                idxs.append(pidx)
            if not xs:
                continue
            x = torch.stack(xs).to(device)
            with autocast:
                e = model.probe_features((x - mean) / std).float().cpu().numpy()  # probe.py:304/683
            for j, pidx in enumerate(idxs):
                sums[pidx] += e[j].astype(np.float64)
                counts[pidx] += 1
            done += len(idxs)
            if done % (batch_size * 20) < batch_size:
                log(f"    embedded {done}/{len(tiles)} tiles")
    keep = counts > 0
    X = np.zeros((n, embed_dim), dtype=np.float32)
    X[keep] = (sums[keep] / counts[keep, None]).astype(np.float32)
    return X, counts


# ============================================================== tile indexing
def _detect_image_col(cols):
    for c in IMAGE_COL_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(f"no image column among {IMAGE_COL_CANDIDATES} in {cols}")


def _tcga_barcode_from_svs(name):
    base = os.path.basename(name)
    parts = base.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else None


def _patient_id_for_file(pf, cols, stem):
    """Cheaply derive the patient barcode/case_id for a parquet file by reading
    one row of the id-bearing column (TCGA: slide_path/path svs stem; CPTAC:
    case_id column; else the filename stem)."""
    m = re.match(r"(TCGA-\w\w-\w{4})", stem)
    if m:
        return m.group(1)
    if "case_id" in cols:
        return pf.read_row_group(0, columns=["case_id"]).column("case_id")[0].as_py()
    for pc in ("slide_path", "path"):
        if pc in cols:
            val = pf.read_row_group(0, columns=[pc]).column(pc)[0].as_py()
            bc = _tcga_barcode_from_svs(val)
            if bc:
                return bc
    return stem  # last resort


def build_tile_index(tiles_dir, allowed_patients, max_slides, max_tiles_per_slide, log=print):
    """Scan parquet files in tiles_dir, keep tiles for patients in
    `allowed_patients` (None = keep all). Returns (patient_ids, tiles) where
    tiles is a list of (patient_index, jpeg_bytes)."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(tiles_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {tiles_dir}")
    log(f"  scanning {len(files)} parquet file(s) in {tiles_dir}")

    pid_to_idx = {}
    patient_ids = []
    tiles = []
    slides_used = 0
    for f in files:
        stem = Path(f).stem
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        img_col = _detect_image_col(cols)
        pid = _patient_id_for_file(pf, cols, stem)
        if allowed_patients is not None and pid not in allowed_patients:
            continue
        if pid not in pid_to_idx:
            pid_to_idx[pid] = len(patient_ids)
            patient_ids.append(pid)
        pidx = pid_to_idx[pid]
        # collect up to max_tiles_per_slide tiles, evenly across row groups
        collected = []
        for rg in range(pf.num_row_groups):
            if max_tiles_per_slide and len(collected) >= max_tiles_per_slide:
                break
            col = pf.read_row_group(rg, columns=[img_col]).column(img_col).to_pylist()
            for b in col:
                collected.append(b)
                if max_tiles_per_slide and len(collected) >= max_tiles_per_slide:
                    break
        for b in collected:
            tiles.append((pidx, b))
        slides_used += 1
        if max_slides and slides_used >= max_slides:
            log(f"  reached max_slides={max_slides}")
            break
    log(f"  indexed {len(patient_ids)} patient(s), {len(tiles)} tile(s) from {slides_used} slide(s)")
    return patient_ids, tiles


# ============================================================== metadata / labels
def load_demographics(csv_path, key_col):
    import csv
    rows = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r[key_col]] = r
    return rows


def tcga_labels_folds(demo, task):
    """Return {barcode: (label, fold)} for TCGA task using label_<task>/fold_<task>."""
    lcol, fcol = f"label_{task}", f"fold_{task}"
    out = {}
    for bc, r in demo.items():
        lv, fv = r.get(lcol, ""), r.get(fcol, "")
        if lv not in ("", None) and fv not in ("", None):
            try:
                out[bc] = (int(float(lv)), int(float(fv)))
            except ValueError:
                continue
    return out


def cptac_labels(labels_tsv):
    """case_id -> subtype (LUAD=0 / LSCC=1) from labels.tsv."""
    import csv
    out = {}
    with open(labels_tsv, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["case_id"]] = int(float(r["subtype"]))
    return out


# ============================================================== probing
def _fit_predict_proba(Xtr, ytr, Xte, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=LR_MAX_ITER,
                             random_state=0, solver=LR_SOLVER,
                             dual=Xtr.shape[0] < Xtr.shape[1]).fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def probe_internal_cv(X, y, folds, n_splits=5, log=print):
    """Out-of-fold predictions. `folds` = per-patient fold ids (TCGA leak-free)
    or None to build StratifiedKFold(seed 1337). Picks C maximizing overall OOF
    AUC (mirrors probe's per-C-then-best-C selection). Returns (oof_p, best_C)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n = len(y)
    if folds is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=FOLD_SEED)
        fold_of = np.empty(n, dtype=np.int64)
        for fi, (_, va) in enumerate(skf.split(X, y)):
            fold_of[va] = fi
        folds = fold_of
    unique_folds = sorted(set(int(f) for f in folds))

    best_C, best_auc, best_oof = None, -1.0, None
    for C in LR_CS:
        oof = np.full(n, np.nan)
        for f in unique_folds:
            va = folds == f
            tr = ~va
            if len(set(y[tr])) < 2 or va.sum() == 0:
                continue
            oof[va] = _fit_predict_proba(X[tr], y[tr], X[va], C)
        valid = ~np.isnan(oof)
        if len(set(y[valid])) < 2:
            continue
        auc = roc_auc_score(y[valid], oof[valid])
        log(f"    C={C:<7} OOF AUC={auc:.4f}")
        if auc > best_auc:
            best_C, best_auc, best_oof = C, auc, oof
    return best_oof, best_C


def probe_external(Xtr, ytr, Xte, n_splits=5, log=print):
    """Pick C by StratifiedKFold CV on TRAIN (TCGA), fit on all train, predict on
    TEST (CPTAC). Returns (test_p, best_C)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=min(n_splits, np.bincount(ytr).min()),
                          shuffle=True, random_state=FOLD_SEED)
    best_C, best_auc = LR_CS[0], -1.0
    for C in LR_CS:
        aucs = []
        for tr, va in skf.split(Xtr, ytr):
            if len(set(ytr[tr])) < 2:
                continue
            p = _fit_predict_proba(Xtr[tr], ytr[tr], Xtr[va], C)
            aucs.append(roc_auc_score(ytr[va], p))
        if aucs and np.mean(aucs) > best_auc:
            best_C, best_auc = C, float(np.mean(aucs))
    log(f"    external: best C={best_C} (train CV AUC={best_auc:.4f})")
    return _fit_predict_proba(Xtr, ytr, Xte, best_C), best_C


# ============================================================== fairness metrics
def ece_score(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    e = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


def _safe_auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def subgroup_report(y, p, group_of, min_n, min_class_n=5):
    """group_of: array of subgroup labels (str) aligned to y/p; None = exclude
    patient from this attribute. Returns dict with per-subgroup + fairness gaps."""
    y, p = np.asarray(y), np.asarray(p)
    groups = defaultdict(list)
    for i, g in enumerate(group_of):
        if g is not None:
            groups[g].append(i)

    subgroups = {}
    eligible = {}  # name -> auc for subgroups meeting the support heuristic
    eligible_ece = {}
    for g, idx in sorted(groups.items()):
        idx = np.asarray(idx)
        yy, pp = y[idx], p[idx]
        n = int(len(idx))
        auc = _safe_auc(yy, pp)
        ece = ece_score(yy, pp) if n > 0 else None
        n_pos = int(yy.sum())
        n_neg = int((1 - yy).sum())
        insufficient_support = n < min_n or min(n_pos, n_neg) < min_class_n
        subgroups[g] = {
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "auc": auc, "ece": ece,
            "insufficient_support": insufficient_support,
            "eligible_for_gap": not insufficient_support and auc is not None,
        }
        if not insufficient_support and auc is not None:
            eligible[g] = auc
            eligible_ece[g] = ece

    overall_auc = _safe_auc(y[[g is not None for g in group_of]] if False else y, p)
    # overall over patients that belong to *some* group of this attribute
    in_attr = np.asarray([g is not None for g in group_of])
    attr_overall = _safe_auc(y[in_attr], p[in_attr]) if in_attr.sum() else None

    auc_delta = es_auc = ece_delta = None
    if len(eligible) >= 2 and attr_overall is not None:
        vals = list(eligible.values())
        auc_delta = float(max(vals) - min(vals))
        denom = 1.0 + sum(abs(attr_overall - a) for a in vals)
        es_auc = float(attr_overall / denom)
        ev = list(eligible_ece.values())
        ece_delta = float(max(ev) - min(ev))
    return {
        "attr_overall_auc": attr_overall,
        "subgroups": subgroups,
        "auc_delta": auc_delta,
        "es_auc": es_auc,
        "ece_delta": ece_delta,
        "n_eligible_subgroups": len(eligible),
        "min_n": min_n, "min_class_n": min_class_n,
    }


def build_group_arrays(patient_ids, demo, key_lookup, age_cutoff=None):
    """Return dict attr -> list of subgroup labels (or None) aligned to patient_ids,
    plus join mask. key_lookup(pid) -> demographics row or None."""
    race, sex, age_vals = [], [], []
    joined = []
    for pid in patient_ids:
        r = key_lookup(pid)
        joined.append(r is not None)
        if r is None:
            race.append(None); sex.append(None); age_vals.append(np.nan); continue
        race.append(RACE_MAP.get(str(r.get("race", "")).strip().lower()))
        sex.append(SEX_MAP.get(str(r.get("gender", "")).strip().lower()))
        try:
            age_vals.append(float(r.get("age_years", "")))
        except (ValueError, TypeError):
            age_vals.append(np.nan)
    age_vals = np.asarray(age_vals)
    cutoff = DEFAULT_AGE_CUTOFF if age_cutoff is None else float(age_cutoff)
    age = []
    for v in age_vals:
        if not np.isfinite(v):
            age.append(None)
        else:
            age.append(f"age<{cutoff:g}" if v < cutoff else f"age>={cutoff:g}")
    return {"race": race, "sex": sex, "age": age}, np.asarray(joined), cutoff


# ============================================================== markdown
def to_markdown(res):
    L = []
    b = res["backbone"]
    L.append(f"## Fairness eval -- task={res['task']} mode={res['mode']}")
    L.append(f"backbone: variant={b['variant']} weights={b['weights']} "
             f"random_init={b['random_init']} embed_dim={b['embed_dim']}")
    L.append(f"patients evaluated: {res['n_patients']}  |  join_rate: {res['join_rate']:.3f}  "
             f"|  tiles: {res['tiles_embedded']}  |  overall AUROC: "
             f"{res['overall_auc']:.4f}" if res['overall_auc'] is not None else "overall AUROC: n/a")
    L.append("")
    for attr in ("race", "sex", "age"):
        a = res["attributes"][attr]
        L.append(f"### {attr}")
        L.append(f"attr AUROC={_f(a['attr_overall_auc'])}  AUCd={_f(a['auc_delta'])}  "
                 f"ES-AUC={_f(a['es_auc'])}  ECEd={_f(a['ece_delta'])}  "
                 f"(eligible subgroups: {a['n_eligible_subgroups']}, "
                 f"min_n={a['min_n']}, min_class_n={a['min_class_n']})")
        L.append("| subgroup | n | pos | neg | AUROC | ECE | flag |")
        L.append("|---|---|---|---|---|---|---|")
        for g, s in a["subgroups"].items():
            flag = "INSUFFICIENT-SUPPORT" if s["insufficient_support"] else ""
            L.append(f"| {g} | {s['n']} | {s['n_pos']} | {s['n_neg']} | "
                     f"{_f(s['auc'])} | {_f(s['ece'])} | {flag} |")
        L.append("")
    return "\n".join(L)


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


# ============================================================== HF tile pull
def _dir_has_parquet(d):
    """True if `d` exists and contains at least one .parquet file (recursively)."""
    return bool(d) and os.path.isdir(d) and bool(
        glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))


def maybe_pull_hf_tiles(task, tiles_dir, hf_repo, hf_revision, out_path, log=print):
    """Auto-pull a tile cohort from HuggingFace when needed.

    Returns (effective_tiles_dir, scratch_to_clean). When --hf-repo is unset OR
    the local `tiles_dir` already holds parquet tiles, the ORIGINAL tiles_dir is
    returned unchanged and scratch_to_clean is None. Otherwise the cohort for
    `task` (TASK_TO_COHORT) is pulled
    via hf_tiles.pull into a scratch dir beside --out, and that dir is returned.
    """
    if not hf_repo or _dir_has_parquet(tiles_dir):
        return tiles_dir, None            # default path: pull SKIPPED
    if not hf_revision:
        raise SystemExit(
            "[fairness_eval] --hf-revision is required with --hf-repo"
        )
    cohort = TASK_TO_COHORT.get(task)
    if cohort is None:
        raise SystemExit(
            f"[fairness_eval] --hf-repo set but task '{task}' has no HF cohort "
            "(in-distribution tasks require a populated local --tiles-dir); "
            f"provide a populated --tiles-dir instead")
    import hf_tiles                        # same tools/ dir; reuse pull() verbatim
    dest = os.path.join(str(Path(out_path).parent), "_hf_tiles", cohort)
    os.makedirs(dest, exist_ok=True)
    log(f"[fairness_eval] local tiles-dir '{tiles_dir}' missing/empty; pulling "
        f"cohort '{cohort}' from HF repo '{hf_repo}' at {hf_revision} -> {dest}")
    hf_tiles.pull(cohort, dest, repo=hf_repo, revision=hf_revision)
    return dest, dest


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(description="Fairness eval harness for pathology FM checkpoints")
    checkpoint_group = ap.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", help="trained checkpoint .pt")
    checkpoint_group.add_argument(
        "--allow-random-init", action="store_true",
        help="use random weights for a non-scientific plumbing smoke test",
    )
    ap.add_argument("--task", required=True, choices=["nsclc", "glioma", "brca", "cptac_nsclc", "dcis_duke", "cptac_gbm"])
    ap.add_argument("--tiles-dir", required=True, help="dir of tile parquet files (eval/test set)")
    ap.add_argument("--demographics-csv", required=True)
    ap.add_argument("--labels-tsv", default=None, help="CPTAC labels.tsv (cptac_nsclc)")
    ap.add_argument("--molecular-csv", default=None,
                    help="TCGA molecular-labels CSV (patient_barcode key). When set "
                         "together with --label-col, the TCGA label (and fold, via "
                         "--fold-col) are read from HERE instead of the default "
                         "label_<task>/fold_<task> columns in --demographics-csv -- "
                         "e.g. BRCA TP53: --label-col tp53_status --fold-col "
                         "fold_tp53_brca. Default None = unchanged.")
    ap.add_argument("--label-col", default=None,
                    help="override TCGA label column, read from --molecular-csv.")
    ap.add_argument("--fold-col", default=None,
                    help="override TCGA fold column, read from --molecular-csv.")
    ap.add_argument("--test-fold", type=int, default=None,
                    help="leak-free designated TEST fold for TCGA tasks: fit the LR "
                         "probe on ALL OTHER folds and evaluate/dump ONLY the "
                         "patients in this fold (the encoder never saw them). "
                         "Default None = internal out-of-fold CV (unchanged).")
    ap.add_argument("--external-test", action="store_true",
                    help="train LR on --train-tiles-dir (TCGA-NSCLC), test on --tiles-dir")
    ap.add_argument("--train-tiles-dir", default=None, help="external-test: TCGA-NSCLC tiles")
    ap.add_argument("--train-demographics-csv", default=None, help="external-test: TCGA demographics")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--min-class-n", type=int, default=5,
                    help="minimum positives and negatives required per subgroup")
    ap.add_argument("--age-cutoff", type=float, default=DEFAULT_AGE_CUTOFF,
                    help="fixed age threshold used for every fold (default: 65)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-slides", type=int, default=0, help="cap slides (0=all; for CPU smoke)")
    ap.add_argument("--max-tiles-per-slide", type=int, default=0, help="cap tiles/slide (0=all)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--variant", default=None, help="override backbone variant")
    ap.add_argument("--model-dir", default=WORKTREE,
                    help="directory containing model.py (default: bundled pretraining code)")
    ap.add_argument("--hf-repo", default=None,
                    help="Hugging Face dataset repo id (for example, org/fairness-tiles); "
                         "when set AND --tiles-dir is missing/empty, auto-pull the cohort "
                         "for --task (see TASK_TO_COHORT) before eval")
    ap.add_argument("--hf-revision", default=None,
                    help="immutable dataset commit SHA; required with --hf-repo")
    ap.add_argument("--hf-clean", action="store_true",
                    help="remove the HF-pulled scratch tiles after eval")
    ap.add_argument("--dump-predictions", default=None,
                    help="if set, write a per-patient JSONL (one line each: "
                         "patient_id, y_true, y_score, race, sex, age) to this "
                         "path -- the same per-patient pooled scores/labels/demo "
                         "that feed subgroup_report.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log = lambda *a: print(*a, flush=True)
    t0 = time.monotonic()
    log(f"[fairness_eval] task={args.task} device={device} external_test={args.external_test}")

    is_cptac = args.task == "cptac_nsclc"

    # --- labels + allowed patients for the EVAL set -------------------------
    demo = load_demographics(args.demographics_csv,
                             "case_id" if is_cptac else "patient_barcode")
    if is_cptac:
        if not args.labels_tsv:
            ap.error("--labels-tsv required for cptac_nsclc")
        case_label = cptac_labels(args.labels_tsv)
        allowed = set(case_label)          # patients with a known label
        label_of = case_label
        fold_of_pid = None                 # internal: StratifiedKFold
    elif args.molecular_csv and args.label_col:
        # TCGA label/fold from an explicit molecular CSV (e.g. BRCA TP53). Mirrors
        # post_hoc_debias.build_task_cohort: a populated --fold-col naturally
        # restricts the cohort to the task's patients (only they carry that fold).
        mol = load_demographics(args.molecular_csv, "patient_barcode")
        allowed, label_of, fold_of_pid = set(), {}, {}
        for bc, m in mol.items():
            lv = m.get(args.label_col, "")
            if lv in ("", None):
                continue
            fv = m.get(args.fold_col, "") if args.fold_col else None
            if args.fold_col and fv in ("", None):
                continue
            try:
                label_of[bc] = int(float(lv))
            except (ValueError, TypeError):
                continue
            if args.fold_col:
                try:
                    fold_of_pid[bc] = int(float(fv))
                except (ValueError, TypeError):
                    label_of.pop(bc, None)
                    continue
            allowed.add(bc)
        if not args.fold_col:
            fold_of_pid = None
        log(f"[fairness_eval] TCGA labels from {args.molecular_csv} "
            f"label_col={args.label_col} fold_col={args.fold_col}: "
            f"{len(allowed)} labeled patients")
    else:
        lf = tcga_labels_folds(demo, args.task)
        allowed = set(lf)
        label_of = {bc: v[0] for bc, v in lf.items()}
        fold_of_pid = {bc: v[1] for bc, v in lf.items()}

    # --- backbone (probe.py path) ------------------------------------------
    log("[fairness_eval] building backbone")
    model, mean, std, binfo = build_backbone(
        args.checkpoint, device, args.variant, args.model_dir,
        allow_random_init=args.allow_random_init,
    )
    log(f"  backbone: {binfo['variant']} weights={binfo['weights']} "
        f"random_init={binfo['random_init']} dim={binfo['embed_dim']}")

    # --- index + embed EVAL tiles ------------------------------------------
    # Optional HF auto-pull: no-op (returns args.tiles_dir unchanged) unless
    # --hf-repo is set and the local tiles-dir is missing/empty.
    tiles_dir, hf_scratch = maybe_pull_hf_tiles(
        args.task, args.tiles_dir, args.hf_repo, args.hf_revision, args.out, log
    )
    dataset_identity = None
    if args.task in {"brca", "nsclc", "glioma"}:
        dataset_identity = validate_dataset_receipt(tiles_dir, "downstream")
    log("[fairness_eval] indexing eval tiles")
    patient_ids, tiles = build_tile_index(tiles_dir, allowed, args.max_slides,
                                          args.max_tiles_per_slide, log)
    if not patient_ids:
        ap.error("no patients matched the task label set in --tiles-dir")
    log("[fairness_eval] embedding eval tiles")
    X, counts = embed_patients(model, mean, std, device, tiles, patient_ids, args.batch_size, log)
    keep = counts > 0
    patient_ids = [p for p, k in zip(patient_ids, keep) if k]
    X = X[keep]
    y = np.asarray([label_of[p] for p in patient_ids], dtype=np.int64)
    log(f"  embedded {int(counts.sum())} tiles -> {len(patient_ids)} patients "
        f"(pos={int(y.sum())}, neg={int((1 - y).sum())})")

    # --- probing ------------------------------------------------------------
    train_dataset_identity = None
    if args.external_test:
        if not (args.train_tiles_dir and args.train_demographics_csv):
            ap.error("--external-test requires --train-tiles-dir and --train-demographics-csv")
        log("[fairness_eval] external-test: indexing + embedding TCGA train tiles")
        train_dataset_identity = validate_dataset_receipt(
            args.train_tiles_dir, "downstream"
        )
        tdemo = load_demographics(args.train_demographics_csv, "patient_barcode")
        tlf = tcga_labels_folds(tdemo, "nsclc")
        tr_ids, tr_tiles = build_tile_index(args.train_tiles_dir, set(tlf),
                                            args.max_slides, args.max_tiles_per_slide, log)
        Xtr, tcounts = embed_patients(model, mean, std, device, tr_tiles, tr_ids,
                                      args.batch_size, log)
        tk = tcounts > 0
        tr_ids = [p for p, k in zip(tr_ids, tk) if k]
        Xtr = Xtr[tk]
        ytr = np.asarray([tlf[p][0] for p in tr_ids], dtype=np.int64)
        log(f"  train: {len(tr_ids)} patients (pos={int(ytr.sum())}, neg={int((1-ytr).sum())})")
        p_hat, best_C = probe_external(Xtr, ytr, X, args.n_splits, log)
        mode = "external_test"
    elif args.test_fold is not None:
        if fold_of_pid is None:
            ap.error("--test-fold requires a fold column (set --fold-col, or use a "
                     "TCGA task that carries per-patient folds)")
        folds = np.asarray([fold_of_pid[p] for p in patient_ids])
        te = folds == args.test_fold
        tr = ~te
        if te.sum() == 0:
            ap.error(f"--test-fold {args.test_fold}: no eval patients in that fold")
        if len(set(y[tr])) < 2:
            ap.error("--test-fold: the training folds are single-class")
        log(f"[fairness_eval] leak-free test-fold={args.test_fold}: fit probe on "
            f"train={int(tr.sum())} patients (folds!={args.test_fold}), evaluate "
            f"test={int(te.sum())} held-out patients")
        p_te, best_C = probe_external(X[tr], y[tr], X[te], args.n_splits, log)
        # collapse the cohort to the held-out test fold; only it is scored/dumped
        patient_ids = [p for p, t in zip(patient_ids, te) if t]
        X, y, p_hat = X[te], y[te], p_te
        mode = f"test_fold_{args.test_fold}"
    else:
        folds = (np.asarray([fold_of_pid[p] for p in patient_ids])
                 if fold_of_pid is not None else None)
        p_hat, best_C = probe_internal_cv(X, y, folds, args.n_splits, log)
        mode = "internal_cv"
    if p_hat is None:
        log("  [warn] probing produced no valid predictions (degenerate cohort, "
            "e.g. a single class present) -- filling 0.5; AUROC will be n/a")
        p_hat = np.full(len(y), 0.5, dtype=float)
    p_hat = np.where(np.isnan(p_hat), 0.5, p_hat)  # patients in single-class train folds
    overall_auc = _safe_auc(y, p_hat)

    # --- demographics join + subgroup metrics ------------------------------
    key_lookup = (lambda pid: demo.get(pid))  # eval demographics is the target cohort
    groups, joined, age_cutoff = build_group_arrays(
        patient_ids, demo, key_lookup, age_cutoff=args.age_cutoff
    )
    join_rate = float(joined.mean()) if len(joined) else 0.0
    log(f"[fairness_eval] demographics join_rate={join_rate:.3f} "
        f"({int(joined.sum())}/{len(joined)}), age cutoff={age_cutoff}")
    assert join_rate > 0.0, "demographics join failed for ALL patients -- check key column"

    attributes = {attr: subgroup_report(
        y, p_hat, groups[attr], args.min_n, args.min_class_n
    )
                  for attr in ("race", "sex", "age")}

    # optional per-patient prediction dump (raw material for bootstrap CIs /
    # paired p-values). Uses the SAME arrays that feed subgroup_report:
    # y (labels), p_hat (pooled scores), and the joined race/sex subgroup labels;
    # age is the raw age_years from the joined demographics row.
    if args.dump_predictions:
        dpath = Path(args.dump_predictions)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        with open(dpath, "w") as fh:
            for i, pid in enumerate(patient_ids):
                row = demo.get(pid) or {}
                try:
                    age_val = float(row.get("age_years", ""))
                except (ValueError, TypeError):
                    age_val = None
                fh.write(json.dumps({
                    "patient_id": pid,
                    "y_true": int(y[i]),
                    "y_score": float(p_hat[i]),
                    "race": groups["race"][i],
                    "sex": groups["sex"][i],
                    "age": age_val,
                }) + "\n")
        log(f"[fairness_eval] dumped {len(patient_ids)} per-patient predictions "
            f"-> {dpath}")

    insufficient_support_flags = [
        f"{attr}:{group}" for attr in attributes
        for group, summary in attributes[attr]["subgroups"].items()
        if summary["insufficient_support"]
    ]

    res = {
        "task": args.task, "mode": mode, "external_test": args.external_test,
        "checkpoint": (
            {"kind": "random-init", "sha256": None, "bytes": 0}
            if args.checkpoint is None else {
                "kind": "checkpoint",
                "sha256": sha256_file(Path(args.checkpoint)),
                "bytes": Path(args.checkpoint).stat().st_size,
            }
        ),
        "backbone": binfo,
        "dataset_identity": dataset_identity,
        "train_dataset_identity": train_dataset_identity,
        "n_patients": len(patient_ids), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "tiles_embedded": int(counts.sum()), "join_rate": join_rate,
        "age_cutoff": age_cutoff, "best_C": best_C, "min_n": args.min_n,
        "min_class_n": args.min_class_n,
        "overall_auc": overall_auc, "attributes": attributes,
        "insufficient_support_flags": insufficient_support_flags,
        "elapsed_sec": round(time.monotonic() - t0, 1),
        "patients": patient_ids,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n" + to_markdown(res))
    print(f"\n[fairness_eval] wrote {args.out}  ({res['elapsed_sec']}s)")
    if hf_scratch and args.hf_clean:
        import shutil
        shutil.rmtree(hf_scratch, ignore_errors=True)
        print(f"[fairness_eval] cleaned HF scratch {hf_scratch}")
    if insufficient_support_flags:
        print(f"[fairness_eval] INSUFFICIENT-SUPPORT subgroups (n<{args.min_n} "
              f"or either class n<{args.min_class_n}; excluded from gaps): "
              f"{', '.join(insufficient_support_flags)}")


if __name__ == "__main__":
    main()
