#!/usr/bin/env python
"""post_hoc_debias.py -- POST-HOC fairness debiasing head for pathology FMs.

A DANN-style gradient-reversal adversary trained on a FROZEN encoder:

    frozen encoder -> tile embedding
        Linear1 -> h (hidden) -> Linear2 -> task logit          (task branch)
        GRL(h)  -> MLP        -> demographic logits (race/sex)   (adversary branch)

The Gradient Reversal Layer (GRL) multiplies the gradient flowing back into `h`
by ``-lambda``, so `h` is driven to be task-PREDICTIVE but demographic-INVARIANT.
This is *post-hoc*: the encoder is frozen; only this small head trains on the
downstream task's tiles. Baseline == lambda=0 (a plain head, ~ the LR probe).

Training happens at TILE level (each tile inherits its patient's task label +
demographics -> thousands of examples), prediction is per-tile, mean-aggregated
to the patient for evaluation.

Two adversary-data regimes (the ablation):
  * task_only    : adversary trains only on the task cohort's tiles.
  * matched_pool : adversary ALSO trains on a large POOLED demographic tile set
                   (many patients across cancers with demographics) so it is not
                   data-starved -- isolating intervention TIMING from data SIZE.
                   The TASK head still trains only on task tiles; only the
                   adversary's demographic-supervision pool grows.

Selectable debiasing --method (all on the SAME frozen-encoder head; baseline is
always lambda=0, a plain task head):
  * dann         : learned CE adversary off GRL(h)                  (original).
  * fino         : EMA prototype-bank comparison off GRL(h)         (original).
  * contrastive  : a fair supervised-contrastive term on the hidden h using the
                   DEMOGRAPHIC label with DIFFERENT-demographic tiles as
                   positives, so h is pulled to be demographic-INVARIANT
                   (loss = BCE(task) + lambda * SupCon_fair(h, demo)). No
                   adversary; demographic separability is read out by a post-hoc
                   logistic-regression probe on h -- the demo-AUC diagnostic,
                   which should fall toward 0.5 as lambda engages. Reuses
                   --proto-temp as the SupCon temperature.
  * pcgrad       : gradient-projection at the head level. An aux CE demographic
                   head (NO gradient reversal) gives g_demo; the task-head
                   gradient g_task (both w.r.t. the shared trunk Linear1) is
                   projected onto the orthogonal complement of g_demo:
                     g_task' = g_task - lambda * (<g_task,g_demo>/||g_demo||^2) g_demo
                   removing the trunk-update component that would improve
                   demographic predictability, before the optimizer step. The
                   pre/post projection cosine is printed as evidence and the same
                   post-hoc LR demo-AUC probe is reported.

Reuses fairness_eval.py verbatim for: encoder load (probe.py path), tile->CLS
embedding, and the fairness metric schema (overall + per-subgroup AUROC / AUCd /
ES-AUC / ECEd). fairness_eval's CLI is left intact -- we only IMPORT from it.

Emits the same metric schema as fairness_eval for the TASK head, plus a
diagnostic adversary demographic-prediction AUC (should DROP toward 0.5 as the
debiasing knob engages). Baseline (lambda=0) and debiased (lambda>0) are trained
in the same run and compared in the output.

Author-note: no locked files are edited (probe.py, benchmarking/,
labless/submit_to_labless.py); no git push.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- reuse fairness_eval (same directory) -- import only, CLI left intact -----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fairness_eval as fe  # noqa: E402

RACE_CLASSES = ["White", "Black", "Asian"]   # fe.RACE_MAP targets
SEX_CLASSES = ["M", "F"]                       # fe.SEX_MAP targets
ATTR_CLASSES = {"race": RACE_CLASSES, "sex": SEX_CLASSES}
SEED = 1337

# --- Feature 2: --hf-repo auto-pull -- task -> HF cohort DIRECTORY name --------
# The fairness-tiles repo (hf_tiles.py) stores each cohort under "<cohort>/"; the
# repo currently holds the CPTAC cohorts {cptac_lung, cptac_gbm, cptac_ccrcc}.
# Each --task maps to its tumor-type cohort dir -- the dir name intentionally
# differs from the task name (a task is label+cohort; the cohort dir is just the
# tumor type), exactly like the doc example  cptac_nsclc -> cptac_lung.
#   * nsclc/glioma resolve to cohorts that EXIST in the repo today and pull live.
#   * the breast/liver/uterine tasks have no CPTAC cohort in the repo yet, so
#     their dirs must be pushed (hf_tiles.py push) before a live pull can find
#     them; the mapping/skip-logic below is otherwise identical for every task.
TASK_COHORT = {
    "brca_tp53":  "cptac_brca",   # TP53 status  -> breast   (not yet in repo)
    "ucec_tp53":  "cptac_ucec",   # TP53 status  -> uterine  (not yet in repo)
    "lihc_grade": "cptac_hcc",    # tumor grade  -> liver/HCC(not yet in repo)
    "nsclc":      "cptac_lung",   # NSCLC subtype-> lung      (in repo)
    "glioma":     "cptac_gbm",    # glioma       -> GBM       (in repo)
}


def _dir_absent_or_empty(d):
    """True if ``d`` is missing or holds no parquet tiles (checked recursively).
    Used to decide whether the --hf-repo auto-pull should trigger."""
    if not d or not os.path.isdir(d):
        return True
    if glob.glob(os.path.join(d, "*.parquet")):
        return False
    return not glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)


# ============================================================== GRL (few lines)
def _torch():
    import torch  # local import so --help works without torch
    return torch


class GradReverse:
    """Placeholder; the real autograd.Function is built lazily (needs torch)."""


def _build_grl():
    torch = _torch()

    class _GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = float(lambd)
            return x.view_as(x)                     # identity forward

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output.neg() * ctx.lambd, None   # -lambda * grad backward

    return _GradReverse


# ============================================================== head + adversary
def build_head(in_dim, hidden, sensitive, dropout=0.1, method="dann",
               proto_temp=0.1, proto_ema=0.9):
    """Shared task branch (Linear1 -> h -> Linear2 -> task logit) with a
    selectable demographic branch off ``GRL(h)``:
      * method="dann" : a learned CE MLP adversary (original behavior).
      * method="fino" : EMA *prototype banks* (one normalized vector per
                        demographic-class value per sensitive attr). The demo
                        logits are prototype comparisons
                        ``logits = GRL(h_norm) @ proto_norm.T / temp`` -- no
                        learned linear head; the GRL drives h to be NOT
                        clusterable by the demographic prototypes."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    grl = _build_grl()
    attrs = {"race": ["race"], "sex": ["sex"], "both": ["race", "sex"]}[sensitive]

    class DebiasHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, hidden)        # encoder -> h
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.task_head = nn.Linear(hidden, 1)         # Linear2 -> task logit
            self.attrs = attrs
            self.method = method
            self.proto_temp = float(proto_temp)
            self.proto_ema = float(proto_ema)
            if method == "dann":
                self.adv = nn.ModuleDict({
                    a: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, len(ATTR_CLASSES[a])))
                    for a in attrs
                })
            else:  # fino: prototype banks are BUFFERS (EMA-updated, no grad)
                for a in attrs:
                    n = len(ATTR_CLASSES[a])
                    self.register_buffer(f"proto_{a}", torch.zeros(n, hidden))
                    self.register_buffer(f"proto_init_{a}",
                                         torch.zeros(n, dtype=torch.bool))
            if method == "pcgrad":   # aux CE demographic head (NO gradient reversal)
                self.demo = nn.ModuleDict({
                    a: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, len(ATTR_CLASSES[a])))
                    for a in attrs
                })

        def features(self, x):
            return self.drop(self.act(self.lin1(x)))

        @torch.no_grad()
        def update_prototypes(self, h, labels):
            """EMA-update each demographic prototype toward the mean (normalized)
            hidden vector of tiles carrying that class label. Detached: the bank
            is a target, not a trained parameter."""
            hd = h.detach()
            for a in self.attrs:
                proto = getattr(self, f"proto_{a}")       # [C, hidden]
                init = getattr(self, f"proto_init_{a}")   # [C] bool
                lab = labels[a]
                for c in range(proto.shape[0]):
                    m = lab == c
                    if bool(m.any()):
                        mu = F.normalize(hd[m].mean(0), dim=0, eps=1e-6)
                        if bool(init[c]):
                            proto[c].mul_(self.proto_ema).add_(
                                mu, alpha=1.0 - self.proto_ema)
                        else:
                            proto[c].copy_(mu); init[c] = True

        def forward(self, x, lambd, labels=None):
            h = self.features(x)
            task_logit = self.task_head(h).squeeze(-1)
            if self.method in ("contrastive", "pcgrad"):
                # trunk-level methods: no GRL/prototype branch. pcgrad exposes an
                # aux (non-reversed) demo head; contrastive has no demo module.
                if self.method == "pcgrad":
                    return task_logit, {a: self.demo[a](h) for a in self.attrs}
                return task_logit, {}
            if self.method == "fino" and self.training and labels is not None:
                self.update_prototypes(h, labels)         # refresh bank from THIS h
            r = grl.apply(h, lambd)                        # gradient-reversed h
            if self.method == "dann":
                adv_logits = {a: self.adv[a](r) for a in self.attrs}
            else:                                          # prototype comparison
                rn = F.normalize(r, dim=1, eps=1e-6)
                adv_logits = {}
                for a in self.attrs:
                    proto = F.normalize(getattr(self, f"proto_{a}"),
                                        dim=1, eps=1e-6)
                    adv_logits[a] = (rn @ proto.t()) / self.proto_temp
            return task_logit, adv_logits

    return DebiasHead()


# ============================================================== task cohorts
def _clean(v):
    return v not in ("", None)


def build_task_cohort(task, demo, mol, fold_col_arg, log=print):
    """Returns (label_of, fold_of, cohort_barcodes).
    label_of[bc] in {0,1}; fold_of[bc] = int fold (KFold-generated if absent)."""
    label_of, fold_of = {}, {}

    # (label_source, label_col, default_fold_col, cohort_rule)
    if task in ("brca_tp53", "ucec_tp53"):
        fold_col = fold_col_arg or ("fold_tp53_brca" if task == "brca_tp53"
                                    else "fold_tp53_ucec")
        for bc, m in mol.items():
            if _clean(m.get(fold_col)) and _clean(m.get("tp53_status")):
                label_of[bc] = int(float(m["tp53_status"]))
                fold_of[bc] = int(float(m[fold_col]))
    elif task == "lihc_grade":
        # no dedicated grade-fold column -> KFold below; cohort = LIHC & graded
        for bc, m in mol.items():
            d = demo.get(bc, {})
            if (_clean(m.get("tumor_grade_bin"))
                    and str(d.get("cancer_type", "")).upper() == "LIHC"):
                label_of[bc] = int(float(m["tumor_grade_bin"]))
        fold_col = fold_col_arg  # usually None -> KFold
        if fold_col:
            for bc in list(label_of):
                if _clean(mol.get(bc, {}).get(fold_col)):
                    fold_of[bc] = int(float(mol[bc][fold_col]))
    elif task in ("nsclc", "glioma"):
        label_col, fold_col = f"label_{task}", (fold_col_arg or f"fold_{task}")
        for bc, r in demo.items():
            if _clean(r.get(label_col)) and _clean(r.get(fold_col)):
                label_of[bc] = int(float(r[label_col]))
                fold_of[bc] = int(float(r[fold_col]))
    else:
        raise ValueError(f"unknown task {task}")

    cohort = sorted(label_of)
    # any barcode missing a fold -> generate deterministic StratifiedKFold for ALL
    if len(fold_of) < len(cohort):
        from sklearn.model_selection import StratifiedKFold
        y = np.array([label_of[b] for b in cohort])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_of = {}
        for k, (_, va) in enumerate(skf.split(cohort, y)):
            for i in va:
                fold_of[cohort[i]] = k
        log(f"  [{task}] no fold column -> generated 5-fold StratifiedKFold")
    return label_of, fold_of, set(cohort)


def sensitive_of(demo):
    """barcode -> {'race': str|None, 'sex': str|None} using fe maps."""
    out = {}
    for bc, r in demo.items():
        out[bc] = {
            "race": fe.RACE_MAP.get(str(r.get("race", "")).strip().lower()),
            "sex": fe.SEX_MAP.get(str(r.get("gender", "")).strip().lower()),
        }
    return out


# ============================================================== tile collection
def _needed_attrs(sensitive):
    return {"race": ["race"], "sex": ["sex"], "both": ["race", "sex"]}[sensitive]


def collect_tiles(tiles_dir, task_cohort, sens, sensitive, adversary_data,
                  max_task_slides, max_pool_slides, max_tiles_per_slide, log=print):
    """Single pass over parquet slides. Returns (task_tiles, pool_tiles) as lists
    of (barcode, jpeg_bytes). task_tiles: slides whose patient is in the cohort.
    pool_tiles: (matched_pool only) slides of NON-cohort patients that carry the
    needed sensitive label(s)."""
    import pyarrow.parquet as pq

    need = _needed_attrs(sensitive)
    files = sorted(glob.glob(os.path.join(tiles_dir, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(tiles_dir, "**", "*.parquet"),
                                 recursive=True))
    log(f"  scanning up to {len(files)} parquet slide(s) in {tiles_dir}")

    task_tiles, pool_tiles = [], []
    task_slides = pool_slides = 0
    task_pat, pool_pat = set(), set()

    def has_sens(bc):
        s = sens.get(bc)
        return s is not None and all(s.get(a) is not None for a in need)

    for f in files:
        done_task = task_slides >= max_task_slides
        done_pool = (adversary_data != "matched_pool"
                     or pool_slides >= max_pool_slides)
        if done_task and done_pool:
            break
        pf = pq.ParquetFile(f)
        cols = pf.schema_arrow.names
        img_col = fe._detect_image_col(cols)
        if img_col is None or "slide_path" not in cols:
            continue
        sp = pf.read_row_group(0, columns=["slide_path"]).column("slide_path")[0].as_py()
        bc = fe._tcga_barcode_from_svs(sp)
        if bc is None:
            continue

        is_task = bc in task_cohort
        is_pool = (not is_task) and (adversary_data == "matched_pool") and has_sens(bc)
        if is_task and done_task:
            continue
        if is_pool and done_pool:
            continue
        if not (is_task or is_pool):
            continue

        imgs = pf.read_row_group(0, columns=[img_col]).column(img_col).to_pylist()
        if max_tiles_per_slide:
            imgs = imgs[:max_tiles_per_slide]
        rows = [(bc, b) for b in imgs if b is not None]
        if not rows:
            continue
        if is_task:
            task_tiles.extend(rows); task_slides += 1; task_pat.add(bc)
        else:
            pool_tiles.extend(rows); pool_slides += 1; pool_pat.add(bc)

    log(f"  task tiles: {len(task_tiles)} from {len(task_pat)} patients "
        f"({task_slides} slides)")
    log(f"  pool tiles: {len(pool_tiles)} from {len(pool_pat)} patients "
        f"({pool_slides} slides)")
    return task_tiles, pool_tiles


# ============================================================== per-tile embed (cached)
def embed_tiles(model, mean, std, device, tiles, batch_size, log=print):
    """tiles: list of (barcode, jpeg). Returns (emb[N,D] float32, keep_mask).
    Uses probe.py's exact CLS path via model.probe_features (no pooling)."""
    torch = _torch()
    from PIL import Image
    from contextlib import nullcontext

    tf = fe._transform()
    D = int(model.embed_dim)
    embs = np.zeros((len(tiles), D), dtype=np.float32)
    ok = np.zeros(len(tiles), dtype=bool)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start:start + batch_size]
            xs, idxs = [], []
            for j, (_, jpg) in enumerate(chunk):
                try:
                    img = Image.open(io.BytesIO(jpg)).convert("RGB")
                except Exception:
                    continue
                xs.append(tf(img)); idxs.append(start + j)
            if not xs:
                continue
            x = torch.stack(xs).to(device)
            with autocast:
                e = model.probe_features((x - mean) / std).float().cpu().numpy()
            embs[idxs] = e.astype(np.float32)
            for k in idxs:
                ok[k] = True
            if (start // batch_size) % 10 == 0:
                log(f"    embedded {start + len(chunk)}/{len(tiles)} tiles")
    return embs, ok


def cached_embed(tag, tiles, embed_fn, cache_dir, log=print):
    """Cache per-tile embeddings to an .npz keyed by `tag` (config hash + set)."""
    if not tiles:
        return np.zeros((0, 0), dtype=np.float32), np.asarray([], dtype=object)
    key = hashlib.md5((tag + f"|n={len(tiles)}").encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"emb_{tag.split('|')[0]}_{key}.npz")
    barcodes = np.asarray([bc for bc, _ in tiles], dtype=object)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if d["emb"].shape[0] == len(tiles):
            log(f"  [cache hit] {path}")
            return d["emb"], d["barcodes"]
    emb, ok = embed_fn(tiles)
    emb, barcodes = emb[ok], barcodes[ok]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=emb, barcodes=barcodes)
    log(f"  [cache write] {path}  ({emb.shape[0]} tiles)")
    return emb, barcodes


# ============================================================== train + eval
def _attr_idx(barcodes, sens, attr):
    """int class index per tile for `attr`, -1 if unknown."""
    cls = {c: i for i, c in enumerate(ATTR_CLASSES[attr])}
    return np.asarray([cls.get(sens.get(bc, {}).get(attr), -1) for bc in barcodes],
                      dtype=np.int64)


def _inverse_freq_weights(aidx_train, attrs, device, torch, log=print):
    """Feature 1: per-attr class-weight tensors for the DEMOGRAPHIC loss.

    Weights come from the TRAINING-FOLD demographic label distribution (the task
    training-fold tiles, i.e. every tile the head trains on excluding the eval
    fold). Inverse-frequency: each class weight is 1/count, so the minority
    race/sex is UPWEIGHTED; the vector is normalized to mean 1 over the classes
    PRESENT in the training fold (absent classes -> weight 1, harmless since the
    loss ignores index -1 / never sees them). Returns {attr: FloatTensor[C]}."""
    weights = {}
    for a in attrs:
        C = len(ATTR_CLASSES[a])
        y = np.asarray(aidx_train[a])
        y = y[y >= 0]
        counts = np.bincount(y, minlength=C).astype(np.float64)[:C]
        w = np.ones(C, dtype=np.float64)
        present = counts > 0
        if present.any():
            inv = 1.0 / counts[present]
            inv = inv / inv.mean()                     # normalize to mean 1
            w[present] = inv
        weights[a] = torch.tensor(w, dtype=torch.float32, device=device)
        pretty = ", ".join(f"{ATTR_CLASSES[a][c]}={w[c]:.3f}(n={int(counts[c])})"
                           for c in range(C))
        log(f"    [race-weight inverse_freq] {a}: {pretty}")
    return weights


# =============================================== contrastive + pcgrad (trunk-level)
def _fair_supcon(h, labels_dict, attrs, temp, torch, F, weights=None):
    """Fair supervised-contrastive term on the hidden ``h`` using the DEMOGRAPHIC
    label. It is the standard SupCon objective with the positive set INVERTED:
    for each anchor, the positives are the tiles of a DIFFERENT demographic value.
    Minimizing it pulls different-demographic tiles together (and pushes
    same-demographic tiles apart), so ``h`` becomes NOT separable by the sensitive
    attribute. Summed (mean) across sensitive attrs; tiles with a missing label
    (idx < 0) are excluded from both anchors and positives. temperature-scaled.

    Feature 1: when ``weights`` is given ({attr: FloatTensor[C]}), each ANCHOR is
    reweighted by the inverse-frequency weight of its own demographic class, so
    minority anchors dominate the term (weighted mean). ``weights=None`` keeps
    the original unweighted mean byte-for-byte."""
    z = F.normalize(h, dim=1, eps=1e-6)
    sim = (z @ z.t()) / float(temp)                    # [B, B] cosine / temp
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    logits = sim.masked_fill(eye, float("-inf"))       # drop self-similarity
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    total = z.new_tensor(0.0)
    nterms = 0
    for a in attrs:
        lab = labels_dict[a]                           # [B] long, -1 = missing
        valid = lab >= 0
        if int(valid.sum()) < 2:
            continue
        vv = valid.unsqueeze(0) & valid.unsqueeze(1)
        pos = (lab.unsqueeze(0) != lab.unsqueeze(1)) & vv & ~eye   # DIFFERENT demo
        pos_cnt = pos.sum(1)
        rows = valid & (pos_cnt > 0)
        if int(rows.sum()) == 0:
            continue
        mean_logp_pos = (logp * pos).sum(1)[rows] / pos_cnt[rows].clamp(min=1)
        per_anchor = -mean_logp_pos
        if weights is not None:
            wrow = weights[a][lab[rows]]               # inv-freq weight per anchor
            total = total + (per_anchor * wrow).sum() / wrow.sum().clamp(min=1e-6)
        else:
            total = total + per_anchor.mean()
        nterms += 1
    return total / nterms if nterms else z.new_tensor(0.0)


def _flat_grad(grads, params, torch):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


def _assign_flat(params, vec):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = vec[i:i + n].view_as(p).detach().clone()
        i += n


def _set_grads(params, grads, torch):
    for p, g in zip(params, grads):
        p.grad = (g.detach().clone() if g is not None else torch.zeros_like(p))


def _demo_probe_auc(h_tr, aidx_tr, h_ev, aidx_ev, attrs):
    """Post-hoc demographic-AUC diagnostic: fit a logistic-regression probe on the
    TRAIN hidden vectors -> demographic label, score it on the EVAL hidden
    vectors. Falls toward 0.5 as the debiasing knob makes h demographic-invariant.
    Independent of any adversary/aux head (works for contrastive AND pcgrad)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    out = {}
    for a in attrs:
        ytr, yev = aidx_tr[a], aidx_ev[a]
        mtr, mev = ytr >= 0, yev >= 0
        if mtr.sum() < 10 or mev.sum() < 10 or len(set(ytr[mtr])) < 2 \
                or len(set(yev[mev])) < 2:
            out[a] = None
            continue
        try:
            sc = StandardScaler().fit(h_tr[mtr])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(sc.transform(h_tr[mtr]), ytr[mtr])
            prob = clf.predict_proba(sc.transform(h_ev[mev]))
            yy, classes = yev[mev], clf.classes_
            if len(classes) == 2:
                out[a] = float(roc_auc_score((yy == classes[1]).astype(int),
                                             prob[:, 1]))
            else:
                aucs = []
                for ci, c in enumerate(classes):
                    yc = (yy == c).astype(int)
                    if 0 < yc.sum() < len(yc):
                        aucs.append(roc_auc_score(yc, prob[:, ci]))
                out[a] = float(np.mean(aucs)) if aucs else None
        except Exception:
            out[a] = None
    return out


def _train_eval_projfree(emb_task, bc_task, emb_pool, bc_pool, label_of, fold_of,
                         sens, sensitive, eval_fold, lambd, hidden, lr, epochs,
                         batch_size, device, method, contrast_temp=0.1,
                         race_weight="none", log=print):
    """Trunk-level debiasing loop for method in {contrastive, pcgrad}. The TASK
    head (Linear1 -> h -> Linear2) trains on the task label as usual; the
    debiasing loss operates on h with weight/knob ``lambd`` (lambd=0 == baseline,
    a plain task head -- no contrastive term / no projection). Returns the SAME
    result schema as train_and_eval, with an extra 'pcgrad_cosine' for pcgrad."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(SEED); np.random.seed(SEED)
    attrs = _needed_attrs(sensitive)
    D = emb_task.shape[1]

    y_task = np.asarray([label_of[bc] for bc in bc_task], dtype=np.float32)
    fold = np.asarray([fold_of[bc] for bc in bc_task], dtype=np.int64)
    is_eval = fold == eval_fold
    is_train_task = ~is_eval

    def attr_arr(barcodes):
        return {a: _attr_idx(barcodes, sens, a) for a in attrs}
    aidx_task = attr_arr(bc_task)
    aidx_pool = (attr_arr(bc_pool) if len(emb_pool)
                 else {a: np.zeros(0, np.int64) for a in attrs})

    Xtr = torch.tensor(emb_task[is_train_task], dtype=torch.float32).to(device)
    ytr = torch.tensor(y_task[is_train_task], dtype=torch.float32).to(device)
    aidx_tr_np = {a: aidx_task[a][is_train_task] for a in attrs}
    aidx_tr = {a: torch.tensor(aidx_tr_np[a], device=device) for a in attrs}

    has_pool = len(emb_pool) > 0
    Xpool = (torch.tensor(emb_pool, dtype=torch.float32).to(device)
             if has_pool else None)
    aidx_pool_t = ({a: torch.tensor(aidx_pool[a], device=device) for a in attrs}
                   if has_pool else None)

    n_task = Xtr.shape[0]
    n_adv = n_task + (Xpool.shape[0] if has_pool else 0)

    model = build_head(D, hidden, sensitive, method=method).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    # Feature 1: inverse-frequency demographic weights (contrastive only here;
    # pcgrad keeps unweighted CE). None -> byte-identical original behavior.
    cw = None
    if race_weight == "inverse_freq" and method == "contrastive":
        cw = _inverse_freq_weights(aidx_tr_np, attrs, device, torch, log)

    trunk_params = list(model.lin1.parameters())        # shared: encoder -> h
    task_params = list(model.task_head.parameters())
    demo_params = list(model.demo.parameters()) if method == "pcgrad" else []
    cos_pre, cos_post = [], []

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_task, device=device)
        for s in range(0, n_task, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            labb = {a: aidx_tr[a][idx] for a in attrs}
            h = model.features(xb)
            task_logit = model.task_head(h).squeeze(-1)
            task_loss = bce(task_logit, yb)

            if method == "contrastive":
                loss = task_loss
                if lambd != 0:
                    loss = loss + lambd * _fair_supcon(h, labb, attrs,
                                                       contrast_temp, torch, F,
                                                       weights=cw)
                opt.zero_grad(); loss.backward(); opt.step()

            elif lambd == 0:                             # pcgrad baseline: task only
                opt.zero_grad(); task_loss.backward(); opt.step()

            else:                                        # pcgrad projection step
                demo_logits = {a: model.demo[a](h) for a in attrs}
                demo_loss = sum(ce(demo_logits[a], labb[a]) for a in attrs)
                if not (torch.isfinite(demo_loss) and float(demo_loss) > 0):
                    opt.zero_grad(); task_loss.backward(); opt.step(); continue
                # grads w.r.t. head params via autograd.grad (retain shared graph)
                g_task_tr = torch.autograd.grad(task_loss, trunk_params,
                                                retain_graph=True)
                g_task_hd = torch.autograd.grad(task_loss, task_params,
                                                retain_graph=True)
                g_demo_tr = torch.autograd.grad(demo_loss, trunk_params,
                                                retain_graph=True)
                g_demo_hd = torch.autograd.grad(demo_loss, demo_params)
                gt = _flat_grad(g_task_tr, trunk_params, torch)
                gd = _flat_grad(g_demo_tr, trunk_params, torch)
                denom = (gd * gd).sum().clamp(min=1e-12)
                dot = (gt * gd).sum()
                # project OUT the component along the demographic gradient
                gproj = gt - lambd * (dot / denom) * gd
                cos_pre.append(float(dot / (gt.norm() * gd.norm() + 1e-12)))
                cos_post.append(float((gproj * gd).sum()
                                      / (gproj.norm() * gd.norm() + 1e-12)))
                opt.zero_grad()
                _assign_flat(trunk_params, gproj)        # trunk: projected task grad
                _set_grads(task_params, g_task_hd, torch)
                _set_grads(demo_params, g_demo_hd, torch)
                opt.step()

        # extra demographic supervision from the pool (debiasing signal only)
        if has_pool and lambd != 0:
            npool = Xpool.shape[0]
            perm2 = torch.randperm(npool, device=device)
            for s in range(0, npool, batch_size):
                idx = perm2[s:s + batch_size]
                labb = {a: aidx_pool_t[a][idx] for a in attrs}
                if method == "contrastive":
                    h = model.features(Xpool[idx])
                    loss = lambd * _fair_supcon(h, labb, attrs, contrast_temp,
                                                torch, F, weights=cw)
                else:                                    # pcgrad: train demo probe
                    h = model.features(Xpool[idx]).detach()   # trunk frozen here
                    demo_logits = {a: model.demo[a](h) for a in attrs}
                    loss = sum(ce(demo_logits[a], labb[a]) for a in attrs)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(); loss.backward(); opt.step()

    # -- eval (held-out fold task tiles) -----------------------------------
    model.eval()
    Xev = torch.tensor(emb_task[is_eval], dtype=torch.float32).to(device)
    bc_ev = bc_task[is_eval]
    with torch.no_grad():
        h_ev = model.features(Xev)
        p_tile = torch.sigmoid(model.task_head(h_ev).squeeze(-1)).cpu().numpy()
        h_ev_np = h_ev.cpu().numpy()
        h_tr_np = model.features(Xtr).cpu().numpy()

    from collections import defaultdict
    agg = defaultdict(list)
    for bc, p in zip(bc_ev, p_tile):
        agg[bc].append(p)
    patient_ids = sorted(agg)
    p_hat = np.array([np.mean(agg[b]) for b in patient_ids])
    y_pat = np.array([label_of[b] for b in patient_ids], dtype=np.int64)
    overall_auc = fe._safe_auc(y_pat, p_hat)

    groups, joined, age_med = fe.build_group_arrays(patient_ids, None, _DEMO_ROW)
    attributes = {a: fe.subgroup_report(y_pat, p_hat, groups[a], MIN_N)
                  for a in ("race", "sex", "age")}

    # demographic-AUC diagnostic: post-hoc LR probe on h (should fall toward 0.5)
    aidx_ev_np = {a: _attr_idx(bc_ev, sens, a) for a in attrs}
    adv_auc = _demo_probe_auc(h_tr_np, aidx_tr_np, h_ev_np, aidx_ev_np, attrs)

    res = {
        "lambda": lambd,
        "overall_auc": overall_auc,
        "n_eval_patients": len(patient_ids),
        "n_eval_tiles": int(is_eval.sum()),
        "n_train_task_tiles": int(n_task),
        "n_adversary_tiles": int(n_adv),
        "attributes": attributes,
        "adversary_demo_auc": adv_auc,
    }
    if method == "pcgrad":
        res["pcgrad_cosine"] = {
            "pre_mean": float(np.mean(cos_pre)) if cos_pre else None,
            "post_mean": float(np.mean(cos_post)) if cos_post else None,
            "n_batches": len(cos_pre),
        }
        log(f"    [pcgrad] grad-cosine(task,demo) pre="
            f"{res['pcgrad_cosine']['pre_mean']} -> post="
            f"{res['pcgrad_cosine']['post_mean']}  ({len(cos_pre)} proj batches)")
    return res


def train_and_eval(emb_task, bc_task, emb_pool, bc_pool, label_of, fold_of,
                   sens, sensitive, eval_fold, lambd, hidden, lr, epochs,
                   batch_size, device, method="dann", proto_temp=0.1,
                   proto_ema=0.9, race_weight="none", log=print):
    torch = _torch()
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    if method in ("contrastive", "pcgrad"):        # trunk-level methods (no GRL)
        return _train_eval_projfree(
            emb_task, bc_task, emb_pool, bc_pool, label_of, fold_of, sens,
            sensitive, eval_fold, lambd, hidden, lr, epochs, batch_size, device,
            method, contrast_temp=proto_temp, race_weight=race_weight, log=log)

    torch.manual_seed(SEED); np.random.seed(SEED)
    attrs = _needed_attrs(sensitive)
    D = emb_task.shape[1]

    y_task = np.asarray([label_of[bc] for bc in bc_task], dtype=np.float32)
    fold = np.asarray([fold_of[bc] for bc in bc_task], dtype=np.int64)
    is_eval = fold == eval_fold
    is_train_task = ~is_eval

    # sensitive indices per tile (task + pool)
    def attr_arr(barcodes):
        return {a: _attr_idx(barcodes, sens, a) for a in attrs}
    aidx_task = attr_arr(bc_task)
    aidx_pool = attr_arr(bc_pool) if len(emb_pool) else {a: np.zeros(0, np.int64) for a in attrs}

    # training pools: task-train tiles (task + adv loss) + pool tiles (adv only)
    Xtr_task = torch.tensor(emb_task[is_train_task], dtype=torch.float32)
    ytr_task = torch.tensor(y_task[is_train_task], dtype=torch.float32)
    aidx_tr_task = {a: aidx_task[a][is_train_task] for a in attrs}

    Xtr_pool = (torch.tensor(emb_pool, dtype=torch.float32) if len(emb_pool)
                else torch.zeros((0, D)))

    # concatenated adversary-training set (task-train tiles + pool tiles)
    X_adv = torch.cat([Xtr_task, Xtr_pool], 0)
    aidx_adv = {a: np.concatenate([aidx_tr_task[a], aidx_pool[a]]) for a in attrs}
    adv_tiles = X_adv.shape[0]

    model = build_head(D, hidden, sensitive, method=method,
                       proto_temp=proto_temp, proto_ema=proto_ema).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    # Feature 1: inverse-frequency demographic weighting for the adversary CE
    # (dann + fino). Weights from the task training-fold label distribution.
    # None -> identical to original (falls through to the unweighted `ce`).
    ce_by_attr = None
    if race_weight == "inverse_freq":
        _w = _inverse_freq_weights(aidx_tr_task, attrs, device, torch, log)
        ce_by_attr = {a: nn.CrossEntropyLoss(ignore_index=-1, weight=_w[a])
                      for a in attrs}

    def ce_a(a, logits, target):
        return (ce_by_attr[a] if ce_by_attr is not None else ce)(logits, target)

    n_task = Xtr_task.shape[0]
    Xtr_task = Xtr_task.to(device); ytr_task = ytr_task.to(device)
    X_adv = X_adv.to(device)
    aidx_tr_task_t = {a: torch.tensor(aidx_tr_task[a], device=device) for a in attrs}
    aidx_adv_t = {a: torch.tensor(aidx_adv[a], device=device) for a in attrs}

    model.train()
    for ep in range(epochs):
        # -- task branch: minibatch over task-train tiles
        perm = torch.randperm(n_task, device=device)
        for s in range(0, n_task, batch_size):
            idx = perm[s:s + batch_size]
            lab = {a: aidx_tr_task_t[a][idx] for a in attrs}  # fino proto update
            logit, adv_logits = model(Xtr_task[idx], lambd, labels=lab)
            loss = bce(logit, ytr_task[idx])
            for a in attrs:
                loss = loss + ce_a(a, adv_logits[a], aidx_tr_task_t[a][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        # -- extra adversary supervision from the pool (adv loss only; GRL-reversed)
        if adv_tiles > n_task:
            perm2 = torch.randperm(adv_tiles, device=device)
            for s in range(0, adv_tiles, batch_size):
                idx = perm2[s:s + batch_size]
                lab = {a: aidx_adv_t[a][idx] for a in attrs}  # fino proto update
                _, adv_logits = model(X_adv[idx], lambd, labels=lab)
                aloss = sum(ce_a(a, adv_logits[a], aidx_adv_t[a][idx]) for a in attrs)
                if not torch.isfinite(aloss):
                    continue
                opt.zero_grad(); aloss.backward(); opt.step()

    # -- eval (held-out fold task tiles) -----------------------------------
    model.eval()
    Xev = torch.tensor(emb_task[is_eval], dtype=torch.float32).to(device)
    bc_ev = bc_task[is_eval]
    with torch.no_grad():
        logit, adv_logits = model(Xev, lambd)
        p_tile = torch.sigmoid(logit).cpu().numpy()
        adv_prob = {a: torch.softmax(adv_logits[a], -1).cpu().numpy() for a in attrs}

    # aggregate task prob to patient (mean over tiles)
    from collections import defaultdict
    agg = defaultdict(list)
    for bc, p in zip(bc_ev, p_tile):
        agg[bc].append(p)
    patient_ids = sorted(agg)
    p_hat = np.array([np.mean(agg[b]) for b in patient_ids])
    y_pat = np.array([label_of[b] for b in patient_ids], dtype=np.int64)
    overall_auc = fe._safe_auc(y_pat, p_hat)

    # fairness metrics reuse fairness_eval schema (group arrays via demo lookup)
    groups, joined, age_med = fe.build_group_arrays(patient_ids, None, _DEMO_ROW)
    attributes = {a: fe.subgroup_report(y_pat, p_hat, groups[a], MIN_N)
                  for a in ("race", "sex", "age")}

    # adversary diagnostic AUC (tile-level, on eval tiles)
    adv_auc = {}
    for a in attrs:
        yi = _attr_idx(bc_ev, sens, a)
        m = yi >= 0
        prob = adv_prob[a][m]
        yy = yi[m]
        if len(set(yy)) < 2:
            adv_auc[a] = None; continue
        try:
            if prob.shape[1] == 2:
                adv_auc[a] = float(roc_auc_score(yy, prob[:, 1]))
            else:
                # macro one-vs-rest over classes actually present in this fold
                aucs = []
                for c in range(prob.shape[1]):
                    yc = (yy == c).astype(int)
                    if 0 < yc.sum() < len(yc):
                        aucs.append(roc_auc_score(yc, prob[:, c]))
                adv_auc[a] = float(np.mean(aucs)) if aucs else None
        except Exception:
            adv_auc[a] = None

    return {
        "lambda": lambd,
        "overall_auc": overall_auc,
        "n_eval_patients": len(patient_ids),
        "n_eval_tiles": int(is_eval.sum()),
        "n_train_task_tiles": int(n_task),
        "n_adversary_tiles": int(adv_tiles),
        "attributes": attributes,
        "adversary_demo_auc": adv_auc,
    }


# module-level demographics row lookup for build_group_arrays -------------------
_DEMO_MAP = {}
def _DEMO_ROW(pid):
    return _DEMO_MAP.get(pid)
def sens_row_lookup(pid):
    return _DEMO_MAP.get(pid)
MIN_N = 15


# ============================================================== main
def main():
    ap = argparse.ArgumentParser(description="Post-hoc DANN debiasing head (frozen encoder)")
    ap.add_argument("--checkpoint", default=None,
                    help="DinoV2ViT .pt (omit/missing -> random backbone, proves plumbing)")
    ap.add_argument("--task", required=True,
                    choices=["brca_tp53", "lihc_grade", "ucec_tp53", "nsclc", "glioma"])
    ap.add_argument("--sensitive", required=True, choices=["race", "sex", "both"])
    ap.add_argument("--adversary-data", required=True, choices=["task_only", "matched_pool"])
    ap.add_argument("--method", choices=["dann", "fino", "contrastive", "pcgrad"],
                    default="dann",
                    help="dann = learned CE adversary + GRL (default, unchanged); "
                         "fino = EMA prototype-bank + GRL (post-hoc FINO); "
                         "contrastive = fair SupCon on h (demo-invariant, uses "
                         "--proto-temp as temperature); "
                         "pcgrad = project task grad off the demographic grad")
    ap.add_argument("--proto-temp", type=float, default=0.1,
                    help="fino: temperature for prototype-comparison logits")
    ap.add_argument("--proto-ema", type=float, default=0.9,
                    help="fino: EMA decay for prototype-bank updates")
    ap.add_argument("--lambda-adv", type=float, default=1.0)
    ap.add_argument("--race-weight", choices=["none", "inverse_freq"], default="none",
                    help="Feature 1: 'inverse_freq' weights the DEMOGRAPHIC loss "
                         "by inverse class frequency (minority race/sex upweighted), "
                         "computed from the training-fold label distribution and "
                         "normalized to mean 1. Applies to dann/fino (adversary CE "
                         "weight) and contrastive (per-anchor weight). 'none' "
                         "(default) is byte-identical to the original behavior.")
    ap.add_argument("--fold-col", default=None)
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--embed-batch-size", type=int, default=128)
    ap.add_argument("--max-task-slides", type=int, default=10**9)
    ap.add_argument("--max-pool-slides", type=int, default=400)
    ap.add_argument("--max-tiles-per-slide", type=int, default=0, help="0 = all")
    ap.add_argument("--tiles-dir", default="/data/TCGA-12K-parquet")
    ap.add_argument("--hf-repo", default=None,
                    help="Feature 2: HF dataset repo id (e.g. "
                         "ryankim17920/nanopath-fairness-tiles). When set AND the "
                         "local --tiles-dir is absent/empty, the task's cohort is "
                         "pulled from HF (via hf_tiles.pull) before embedding; "
                         "otherwise the existing local tiles are used and the pull "
                         "is SKIPPED.")
    ap.add_argument("--hf-scratch", default=None,
                    help="Feature 2: dir to pull HF tiles into "
                         "(default <cache-dir>/hf_tiles).")
    ap.add_argument("--hf-clean", action="store_true",
                    help="Feature 2: delete the pulled HF tiles after the run.")
    ap.add_argument("--demographics-csv",
                    default="/admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv")
    ap.add_argument("--molecular-csv",
                    default="/admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv")
    ap.add_argument("--cache-dir", default="/admin/home/ryan.kim/nt/tools/.debias_cache")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch = _torch()
    log = print
    t0 = time.monotonic()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"[debias] device={device}  task={args.task}  sensitive={args.sensitive}  "
        f"adversary_data={args.adversary_data}  lambda={args.lambda_adv}")

    # metadata ------------------------------------------------------------
    demo = fe.load_demographics(args.demographics_csv, "patient_barcode")
    mol = fe.load_demographics(args.molecular_csv, "patient_barcode")
    label_of, fold_of, cohort = build_task_cohort(args.task, demo, mol, args.fold_col, log)
    sens = sensitive_of(demo)
    global _DEMO_MAP
    _DEMO_MAP = demo
    log(f"[debias] cohort={len(cohort)} patients; folds present for "
        f"{len(fold_of)}; eval_fold={args.eval_fold}")

    # backbone (probe.py path) -------------------------------------------
    log("[debias] building backbone")
    model, mean, std, binfo = fe.build_backbone(args.checkpoint, device, args.variant)
    log(f"  backbone: {binfo['variant']} weights={binfo['weights']} "
        f"random_init={binfo['random_init']} dim={binfo['embed_dim']}")

    # Feature 2: --hf-repo auto-pull of the task's tile cohort -------------
    tiles_dir = args.tiles_dir
    pulled_dir = None
    if args.hf_repo:
        cohort_dir = TASK_COHORT.get(args.task)
        if cohort_dir is None:
            ap.error(f"--hf-repo set but no cohort mapping for task '{args.task}'")
        if _dir_absent_or_empty(tiles_dir):
            import hf_tiles                       # reuse the existing pull logic
            hf_tiles.REPO = args.hf_repo          # honor the requested repo id
            scratch = args.hf_scratch or os.path.join(args.cache_dir, "hf_tiles")
            dest = os.path.join(scratch, cohort_dir)
            log(f"[debias] local tiles-dir '{tiles_dir}' absent/empty -> pulling "
                f"cohort '{cohort_dir}' from HF repo {args.hf_repo} into {dest}")
            hf_tiles.pull(cohort_dir, dest)       # snapshot_download nests under dest/
            tiles_dir = dest                      # recursive parquet glob finds them
            pulled_dir = dest
        else:
            log(f"[debias] --hf-repo set but local tiles-dir '{tiles_dir}' exists "
                f"and holds tiles -> SKIP HF pull (cohort would be '{cohort_dir}')")

    # collect + embed tiles (cached) -------------------------------------
    log("[debias] collecting tiles")
    task_tiles, pool_tiles = collect_tiles(
        tiles_dir, cohort, sens, args.sensitive, args.adversary_data,
        args.max_task_slides, args.max_pool_slides, args.max_tiles_per_slide, log)
    if not task_tiles:
        ap.error("no task tiles collected -- check tiles-dir / cohort")

    cfg_tag = (f"{args.task}-{Path(args.checkpoint).name if args.checkpoint else 'random'}"
               f"-{binfo['weights']}-mt{args.max_tiles_per_slide}")
    embed_fn = lambda tl: embed_tiles(model, mean, std, device, tl, args.embed_batch_size, log)  # noqa: E731
    log("[debias] embedding task tiles")
    emb_task, bc_task = cached_embed(f"task-{cfg_tag}", task_tiles, embed_fn, args.cache_dir, log)
    emb_pool = np.zeros((0, emb_task.shape[1]), np.float32)
    bc_pool = np.asarray([], dtype=object)
    if args.adversary_data == "matched_pool" and pool_tiles:
        log("[debias] embedding matched-pool tiles")
        emb_pool, bc_pool = cached_embed(f"pool-{args.sensitive}-{cfg_tag}",
                                         pool_tiles, embed_fn, args.cache_dir, log)

    # keep only task tiles whose patient has a fold (eval needs it) --------
    keep = np.asarray([bc in fold_of for bc in bc_task])
    emb_task, bc_task = emb_task[keep], bc_task[keep]

    # train baseline (lambda=0) and debiased (lambda-adv) -----------------
    lambdas = [0.0]
    if abs(args.lambda_adv) > 1e-12:
        lambdas.append(args.lambda_adv)
    runs = {}
    for lam in lambdas:
        tag = "baseline" if lam == 0.0 else "debiased"
        log(f"[debias] training {tag} (lambda={lam}) ...")
        runs[tag] = train_and_eval(
            emb_task, bc_task, emb_pool, bc_pool, label_of, fold_of, sens,
            args.sensitive, args.eval_fold, lam, args.hidden, args.lr,
            args.epochs, args.batch_size, device, method=args.method,
            proto_temp=args.proto_temp, proto_ema=args.proto_ema,
            race_weight=args.race_weight, log=log)

    res = {
        "task": args.task, "sensitive": args.sensitive,
        "method": args.method, "proto_temp": args.proto_temp,
        "proto_ema": args.proto_ema,
        "adversary_data": args.adversary_data, "fold_col": args.fold_col,
        "race_weight": args.race_weight,
        "hf_repo": args.hf_repo, "tiles_dir": tiles_dir,
        "eval_fold": args.eval_fold, "lambda_adv": args.lambda_adv,
        "checkpoint": args.checkpoint, "backbone": binfo,
        "n_cohort_patients": len(cohort),
        "n_task_tiles": int(emb_task.shape[0]),
        "n_pool_tiles": int(emb_pool.shape[0]),
        "min_n": MIN_N, "epochs": args.epochs, "hidden": args.hidden,
        "runs": runs,
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\n" + to_markdown(res))
    print(f"\n[debias] wrote {args.out} ({res['elapsed_sec']}s)")

    # Feature 2: optionally free the pulled HF tiles ----------------------
    if args.hf_clean and pulled_dir and os.path.isdir(pulled_dir):
        import shutil
        shutil.rmtree(pulled_dir, ignore_errors=True)
        print(f"[debias] --hf-clean removed pulled tiles {pulled_dir}")


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


def to_markdown(res):
    L = [f"# post-hoc debias: {res['task']}  (method={res.get('method', 'dann')}, "
         f"sensitive={res['sensitive']}, adversary_data={res['adversary_data']})",
         f"backbone={res['backbone']['variant']} weights={res['backbone']['weights']} "
         f"random_init={res['backbone']['random_init']}",
         f"task tiles={res['n_task_tiles']}  pool tiles={res['n_pool_tiles']}  "
         f"eval_fold={res['eval_fold']}", ""]
    L.append("| run | lambda | task AUROC | adv demo-AUC | #adv tiles |")
    L.append("|---|---|---|---|---|")
    for tag, r in res["runs"].items():
        adv = ", ".join(f"{a}={_f(v)}" for a, v in r["adversary_demo_auc"].items())
        L.append(f"| {tag} | {r['lambda']} | {_f(r['overall_auc'])} | {adv} | "
                 f"{r['n_adversary_tiles']} |")
    L.append("")
    for tag, r in res["runs"].items():
        if "pcgrad_cosine" in r:
            pc = r["pcgrad_cosine"]
            L.append(f"pcgrad projection evidence [{tag}]: grad-cosine(task,demo) "
                     f"pre={_f(pc['pre_mean'])} -> post={_f(pc['post_mean'])} "
                     f"({pc['n_batches']} projected batches)")
    L.append("")
    for tag, r in res["runs"].items():
        L.append(f"## {tag} (lambda={r['lambda']})  overall task AUROC={_f(r['overall_auc'])}")
        for attr, a in r["attributes"].items():
            L.append(f"### {attr}: AUCd={_f(a['auc_delta'])} ES-AUC={_f(a['es_auc'])} "
                     f"ECEd={_f(a['ece_delta'])} (powered={a['n_powered_subgroups']})")
            L.append("| subgroup | n | pos | neg | AUROC | flag |")
            L.append("|---|---|---|---|---|---|")
            for g, s in a["subgroups"].items():
                flag = "LOW-POWER" if s["low_power"] else ""
                L.append(f"| {g} | {s['n']} | {s['n_pos']} | {s['n_neg']} | "
                         f"{_f(s['auc'])} | {flag} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
