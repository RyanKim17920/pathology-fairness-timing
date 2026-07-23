# Continual DINOv2 pretraining on TCGA tiles (single-GPU). Three loss terms:
# DINO CLS self-distillation (Sinkhorn-Knopp centred teacher targets),
# I-JEPA patch-feature regression, and a KDE uniformity term on the
# L2-normalised CLS tokens. YAML drives the tunable knobs (backbone variant,
# LR + LR scheduler, drop path, layerwise decay, KDE weight + concentration,
# FLOP/sample budgets, batch size); other DINOv2 hyperparameters are hardcoded
# inline at their use sites.

import atexit
import contextlib
import fnmatch
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.flop_counter import FlopCounterMode

from dataloader import TCGATileDataset, TILE_SIZE
from model import DINOHead, DinoV2ViT, GradScale, JEPAPredictor, load_dinov2_pretrained
from probe import (
    completed_probe_summary,
    collect_probe_results,
    prepare_probe_state,
    probe_enabled,
    queue_probe_job,
)


# Prefix every console line with wall time and job/process id so SLURM logs are easy to scan.
def console_prefix(): return f"{time.strftime('%H:%M:%S')} {os.environ.get('SLURM_JOB_ID', str(os.getpid()))}"


M3_OBJECTIVES = {
    "contrastive-cancer",
    "contrastive-demographics",
    "contrastive-two-condition",
    "fino",
    "dann",
    "pcgrad",
}
M3_DEMOGRAPHIC_FACTORS = {"race", "sex", "age"}


def resolve_fino_objective(fino_cfg):
    objective = fino_cfg.get("objective")
    if objective:
        return objective
    method = fino_cfg.get("method", "fino")
    if method == "contrastive":
        return "contrastive-two-condition" if fino_cfg.get("contrastive_condition_on") else "contrastive-demographics"
    return method


def resolve_fino_meta_path(cfg):
    data = cfg["data"]
    if data.get("fino_meta_file"):
        return Path(os.path.expandvars(data["fino_meta_file"]))
    dataset_dir = Path(data["dataset_dir"])
    nested = dataset_dir / "metadata" / "fino_meta.json"
    return nested if nested.exists() else dataset_dir / "fino_meta.json"


def validate_fino_config(cfg):
    fino_cfg = cfg.get("fino") or {}
    if not fino_cfg.get("enabled"):
        return
    explicit_objective = bool(fino_cfg.get("objective"))
    objective = resolve_fino_objective(fino_cfg)
    if objective not in M3_OBJECTIVES:
        raise ValueError(f"fino.objective must be one of {sorted(M3_OBJECTIVES)}, got {objective!r}")
    fino_cfg["objective"] = objective
    expected_method = "contrastive" if objective.startswith("contrastive-") else objective
    configured_method = fino_cfg.get("method", expected_method)
    if configured_method != expected_method:
        raise ValueError(
            f"fino.objective={objective!r} requires fino.method={expected_method!r}, "
            f"got {configured_method!r}"
        )
    fino_cfg["method"] = expected_method

    discrete = fino_cfg.get("discrete", [])
    continuous = fino_cfg.get("continuous", [])
    selected = [f for f, _ in discrete + continuous]
    duplicates = sorted({f for f in selected if selected.count(f) > 1})
    if duplicates:
        raise ValueError(f"metadata factors may be selected only once: {duplicates}")

    meta_path = resolve_fino_meta_path(cfg)
    if not meta_path.is_file():
        raise FileNotFoundError(f"FINO metadata file does not exist: {meta_path}")
    cfg["data"]["fino_meta_file"] = str(meta_path.resolve())
    meta = json.loads(meta_path.read_text())
    missing_discrete = sorted(
        f for f, _ in discrete
        if f not in meta.get("discrete", {}) or f not in meta.get("n", {})
    )
    missing_continuous = sorted(
        f for f, _ in continuous
        if f not in meta.get("continuous", {}) or f not in meta.get("cont_dim", {})
    )
    if missing_discrete or missing_continuous:
        raise ValueError(
            "configured metadata factors are missing from "
            f"{meta_path}: discrete={missing_discrete}, continuous={missing_continuous}"
        )
    for f, _ in discrete:
        nclasses = int(meta["n"][f])
        labels = list(meta["discrete"][f].values())
        if nclasses < 2 or any(not isinstance(v, int) or v < 0 or v >= nclasses for v in labels):
            raise ValueError(f"invalid discrete metadata factor {f!r} in {meta_path}")
    for f, _ in continuous:
        if int(meta["cont_dim"][f]) < 1:
            raise ValueError(f"invalid continuous metadata factor {f!r} in {meta_path}")

    # Explicit M3 recipes get strict objective contracts; inferred legacy recipes
    # retain their existing factor selections.
    if explicit_objective:
        signs = {f: float(sign) for f, sign in discrete + continuous}
        negative = {f for f, sign in signs.items() if sign < 0}
        factors = set(signs)
        demographic = factors & M3_DEMOGRAPHIC_FACTORS
        cancer_positive = signs.get("cancer", 0.0) > 0
        if objective == "contrastive-cancer" and (factors != {"cancer"} or not cancer_positive):
            raise ValueError("contrastive-cancer requires +cancer and no demographic-removal factors")
        if objective in {"contrastive-demographics", "pcgrad"} and (
            not demographic or factors != demographic or negative != demographic
        ):
            raise ValueError(f"{objective} requires demographic-removal factors only")
        if objective in {"contrastive-two-condition", "fino", "dann"} and (
            not cancer_positive
            or not demographic
            or factors != demographic | {"cancer"}
            or negative != demographic
        ):
            raise ValueError(f"{objective} requires +cancer and at least one -demographic factor")
        if objective == "contrastive-two-condition" and fino_cfg.get("contrastive_condition_on") != "cancer":
            raise ValueError("contrastive-two-condition requires fino.contrastive_condition_on: cancer")

    exclude_path = cfg["data"].get("exclude_barcodes_file")
    excluded = (
        {line.strip() for line in Path(exclude_path).read_text().splitlines() if line.strip()}
        if exclude_path else set()
    )
    coverage = []
    for f, _ in discrete:
        keys = set(meta["discrete"][f])
        coverage.append(f"{f}={len(keys - excluded):,}/{len(keys):,}")
    for f, _ in continuous:
        keys = set(meta["continuous"][f])
        coverage.append(f"{f}={len(keys - excluded):,}/{len(keys):,}")
    print(
        f"{console_prefix()} Metadata  objective={objective} file={meta_path} "
        f"coverage_after_exclude/total: {', '.join(coverage)}",
        flush=True,
    )


# Read the YAML recipe and fail before any GPU work if the parquet tile dataset is absent.
# expandvars is necessary to resolve `$USER` for checked-in configs.
def load_config():
    if len(sys.argv) < 2:
        raise ValueError("usage: python train.py <config.yaml> [output_dir=<path>]")
    cfg = yaml.safe_load(os.path.expandvars(Path(sys.argv[1]).read_text()))
    cfg["config_path"] = str(Path(sys.argv[1]).resolve())
    # Optional `key=value` overrides after the config; only output_dir is supported,
    # since it's the run identifier and routinely set per-submission from the CLI.
    for arg in sys.argv[2:]:
        key, _, value = arg.partition("=")
        if key != "output_dir":
            raise ValueError(f"unsupported override {arg!r}; only output_dir=<path> is supported")
        cfg["project"]["output_dir"] = os.path.expandvars(value)
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    if not any(dataset_dir.glob("shard-*.parquet")):
        raise FileNotFoundError(
            f"No parquet shards (shard-*.parquet) under {dataset_dir}. Pull the 4M-tile "
            f"parquet dataset from medarc/nanopath on HF by running "
            f"`python prepare.py {cfg['config_path']} download=True`. Follow the data setup in "
            f"README.md before launching train.py."
        )
    validate_fino_config(cfg)
    return cfg


# Arm Labless before any GPU work so direct `python train.py ...` gets the same
# no-scope GitHub device login path as the SLURM launcher. Noninteractive runs
# train locally unless the launcher passed a preauthorized token file.
def maybe_arm_labless_autosubmit(cfg, repo_dir):
    token_path = os.environ.get("LABLESS_AUTOSUBMIT_FILE", "")
    eligible = (
        bool(cfg["probe"]["enabled"])
        and int(cfg["probe"]["count"]) > 0
        and int(cfg["train"]["max_train_samples"]) == 1_000_000
        and int(cfg["train"]["max_train_flops"]) == 1_000_000_000_000_000_000
    )
    if token_path:
        atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
        return token_path
    if not eligible:
        return ""
    if not sys.stdin.isatty():
        if not os.environ.get("SLURM_JOB_ID"):
            print(f"{console_prefix()} Labless  no interactive stdin; training will run without auto-submit.", flush=True)
        return ""
    print("This looks like a full Labless-eligible run. Leave either prompt blank to train without auto-submit.", flush=True)
    run_name = input("Labless run name (<=20 chars): ").strip()
    notes = input("Labless notes: ").strip()
    if not run_name or not notes or len(run_name) > 20:
        print("Labless auto-submit skipped; run name and notes are required, and run name must be <=20 chars.", flush=True)
        return ""
    token_path = str(Path(str(Path(cfg["project"]["output_dir"]).expanduser().resolve()) + ".labless_autosubmit.json"))
    status = subprocess.run(
        [sys.executable, str(repo_dir / "labless" / "submit_to_labless.py"), "login_only=true", f"token_output={token_path}", f"run_name={run_name}", f"notes={notes}"],
        cwd=repo_dir,
    ).returncode
    if status != 0:
        print("Labless login did not complete; training will run without auto-submit.", flush=True)
        Path(token_path).unlink(missing_ok=True)
        return ""
    os.environ["LABLESS_AUTOSUBMIT_FILE"] = token_path
    atexit.register(lambda p=Path(token_path): p.unlink(missing_ok=True))
    return token_path


def finish_labless_autosubmit(token_path, output_dir, repo_dir):
    token_file = Path(token_path) if token_path else None
    if token_file is None or not token_file.exists():
        return
    token = json.loads(token_file.read_text())
    status = subprocess.run(
        [
            sys.executable,
            str(repo_dir / "labless" / "submit_to_labless.py"),
            f"output_dir={output_dir.resolve()}",
            f"run_name={token['run_name']}",
            f"notes={token['notes']}",
            f"github_token_file={token_file}",
        ],
        cwd=repo_dir,
    ).returncode
    token_file.unlink(missing_ok=True)
    if status == 2:
        print(f"{console_prefix()} Labless  auto-submit skipped because the completed run did not satisfy submission restrictions.", flush=True)
    elif status != 0:
        raise SystemExit(status)


# Cosine schedule from `start` to `end` over fractional progress in [0, 1].
def cosine_schedule(start, end, frac):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(1.0, max(0.0, frac))))


# Sinkhorn-Knopp centring across this batch, used for DINO teacher targets.
def sinkhorn(x, temp):
    q = torch.exp(x.float() / temp).t()
    b = q.shape[1]
    k = q.shape[0]
    q /= q.sum()
    for _ in range(3):
        q /= q.sum(1, keepdim=True) * k
        q /= q.sum(0, keepdim=True) * b
    return (q * b).t()


# Cross-entropy between teacher distribution and softmax(student / 0.1).
def dino_ce(student, teacher):
    return -(teacher * F.log_softmax(student / 0.1, dim=-1)).sum(-1).mean()


# KDE uniformity loss on L2-normalised CLS tokens.
def kde_loss(x, concentration):
    x = F.normalize(x, p=2, dim=-1)
    sim = concentration * (x @ x.T)
    sim.fill_diagonal_(-float("inf"))
    return torch.logsumexp(sim, dim=1).mean() - math.log(max(1, sim.shape[1] - 1))


# I-JEPA target mask: contiguous square blocks so the predictor must infer missing tissue context.
def make_block_mask(batch, grid, device, n_blocks=4, block_scale=0.10):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    for i in range(batch):
        for _ in range(n_blocks):
            top = random.randint(0, grid - side)
            left = random.randint(0, grid - side)
            masks[i, top : top + side, left : left + side] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


# AdamW parameter groups with layer-wise LR decay on the backbone:
# block i gets lr * layerwise_decay^(depth - 1 - i); patch_embed gets the deepest decay
# multiplied by patch_embed_lr_mult; biases and norms get no weight decay; the head's
# DINO final weight-norm last_layer parameters get an LR-freeze for the first dino.freeze_last_layer_fraction.
def build_param_groups(student_backbone, student_dino_head, student_predictor, layerwise_decay, patch_embed_lr_mult):
    depth = len(student_backbone.blocks)
    # Coalesce params that share (lr_mult, wd_mult, last_layer) into a single group each (~30 groups
    # instead of one-per-param), so AdamW's foreach path fuses the step across many tensors rather than
    # launching per-parameter kernels. Per-param lr/wd are unchanged, so the optimization is numerically identical.
    coalesced = {}
    modules = ((student_backbone, "backbone"), (student_dino_head, "dino_head"), (student_predictor, "jepa_predictor"))
    for module, kind in modules:
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            lr_mult = 1.0
            if kind == "backbone" and name.startswith("blocks."):
                lr_mult = layerwise_decay ** (depth - 1 - int(name.split(".")[1]))
            elif kind == "backbone" and name.startswith("patch_embed."):
                lr_mult = (layerwise_decay ** depth) * patch_embed_lr_mult
            wd_mult = 0.0 if name.endswith("bias") or "norm" in name or p.ndim < 2 else 1.0
            key = (lr_mult, wd_mult, "last_layer" in name)
            coalesced.setdefault(key, {"params": [], "lr_mult": lr_mult, "wd_mult": wd_mult, "last_layer": key[2]})["params"].append(p)
    return list(coalesced.values())


# EMA-update teacher modules from student modules with a single multiplicative decay.
# Params are fused into two _foreach kernels (mul then add) instead of a Python per-tensor loop;
# numerically identical (pt = pt*m + ps*(1-m) per tensor). Called under torch.no_grad() by the caller.
def update_ema(student_module, teacher_module, momentum):
    teacher_params, student_params = list(teacher_module.parameters()), list(student_module.parameters())
    torch._foreach_mul_(teacher_params, momentum)
    torch._foreach_add_(teacher_params, student_params, alpha=1 - momentum)
    for bs, bt in zip(student_module.buffers(), teacher_module.buffers()):
        bt.copy_(bs)


def fair_supcon(z, y, temp, w=None, cond=None):
    # Fairness (cross-demographic) supervised contrastive loss. z: (N, D) L2-normalized CLS features; y: (N,) demographic
    # labels. Positives for anchor i are DIFFERENT-demographic tiles (y_j != y_i); the SupCon objective pulls them
    # together against all other tiles, so minimizing it destroys demographic cluster structure (the sensitive attribute
    # becomes non-separable in the representation). Anchors with no cross-demographic partner contribute nothing.
    # cond (optional): conditioning labels, e.g. cancer type. When set, positives are SAME-COND AND DIFFERENT-demographic
    # tiles. The denominator remains all non-self tiles, so different-cond/same-demo pairs stay in the denominator.
    # w (optional): per-class weight vector; when given, each valid anchor's contribution is scaled by w[y_i] so
    # minority-demographic anchors dominate the reduction (weighted mean). w=None reduces to the plain mean below.
    n = z.shape[0]
    if n < 2:
        return z.new_zeros(())
    sim = (z @ z.t()) / temp
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    logdenom = torch.logsumexp(sim, dim=1, keepdim=True)
    logprob = sim - logdenom
    if cond is not None:
        pos = (cond[:, None] == cond[None, :]) & (y[:, None] != y[None, :]) & ~self_mask
    else:
        pos = (y[:, None] != y[None, :]) & ~self_mask
    pos_count = pos.sum(1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    per_anchor = -(logprob.masked_fill(~pos, 0.0)).sum(1) / pos_count.clamp(min=1)
    if w is None:
        return per_anchor[valid].mean()
    wa = w[y][valid]  # inverse-frequency weight of each valid anchor's own class
    return (per_anchor[valid] * wa).sum() / wa.sum().clamp(min=1e-12)


def cancer_supcon(z, cancer, temp):
    # Biology-only supervised contrastive control: same-cancer tiles are positives,
    # all non-self tiles remain in the denominator, and no demographic labels enter.
    n = z.shape[0]
    if n < 2:
        return z.new_zeros(())
    sim = (z @ z.t()) / temp
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (cancer[:, None] == cancer[None, :]) & ~self_mask
    pos_count = pos.sum(1)
    valid = pos_count > 0
    if not bool(valid.any()):
        return z.new_zeros(())
    per_anchor = -(logprob.masked_fill(~pos, 0.0)).sum(1) / pos_count.clamp(min=1)
    return per_anchor[valid].mean()


def pcgrad_project(g_main, g_dem, params):
    # Fairness gradient projection (PCGrad variant). g_main/g_dem: per-parameter gradient lists (entries may be None)
    # of the main SSL loss and the demographic-CE loss over `params`. If <g_main,g_dem> > 0 (descending g_main would
    # also descend the demographic CE, i.e. IMPROVE demographic predictability), remove that component:
    #   g_proj = g_main - (<g_main,g_dem> / ||g_dem||^2) g_dem   =>   <g_proj,g_dem> = 0 (orthogonal complement).
    # Otherwise g_proj = g_main. Returns (projected_grad_list, info dict with cosines / norms / projected flag).
    fm = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g_main, params)])
    fd = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(g_dem, params)])
    dot = torch.dot(fm, fd); dem_sq = torch.dot(fd, fd)
    cos_before = (dot / (fm.norm() * fd.norm() + 1e-12)).item()
    projected = bool((dem_sq > 1e-12).item() and (dot > 0).item())
    if projected:
        coef = dot / dem_sq
        new = [(gm if gm is not None else torch.zeros_like(p)) - coef * (gd if gd is not None else torch.zeros_like(p))
               for gm, gd, p in zip(g_main, g_dem, params)]
    else:
        new = [(gm if gm is not None else torch.zeros_like(p)) for gm, p in zip(g_main, params)]
    fn = torch.cat([g.reshape(-1) for g in new])
    cos_after = (torch.dot(fn, fd) / (fn.norm() * fd.norm() + 1e-12)).item()
    return new, {"cos_before": cos_before, "cos_after": cos_after, "dot": dot.item(),
                 "norm_before": fm.norm().item(), "norm_after": fn.norm().item(), "projected": projected}


# Orchestrates one pretraining run: setup, train+probe loop, checkpoint, summary.
def main():
    cfg = load_config()
    repo_dir = Path(__file__).resolve().parent
    labless_autosubmit_file = maybe_arm_labless_autosubmit(cfg, repo_dir)
    train_cfg = cfg["train"]
    dino_cfg = cfg["dino"]
    # FINO metadata-guidance: select factors + signs (float; + encourage M+ / - suppress M-). fino_meta (built or
    # copied beside the dataset by prepare.py) holds per-factor barcode maps + cardinalities (n) / vector dims.
    fino_cfg = cfg["fino"] if (cfg.get("fino") or {}).get("enabled") else None
    fino_disc = [(f, float(s)) for f, s in fino_cfg.get("discrete", [])] if fino_cfg else []
    fino_cont = [(f, float(s)) for f, s in fino_cfg.get("continuous", [])] if fino_cfg else []
    fino_meta = json.loads(Path(cfg["data"]["fino_meta_file"]).read_text()) if fino_cfg else {"n": {}, "cont_dim": {}}
    # Fairness method selector (reuses the fino: block so all metadata plumbing — dataloader labels, gamma ramp,
    # meta tuple, optimizer/checkpoint wiring — is shared). "fino" (default): EMA prototype-bank cross-entropy.
    # "dann": a learned linear adversary per discrete factor + gradient reversal; NO prototype bank. Continuous
    # factors (age) use the same GRL+MLP regression branch for every non-PCGrad
    # demographic-removing objective, and a plain regression gradient for PCGrad.
    fairness_objective = resolve_fino_objective(fino_cfg) if fino_cfg else None
    fairness_method = (fino_cfg.get("method", "fino") if fino_cfg else "fino")
    # Contrastive-fairness hyperparams (method=contrastive): temperature + weight for the cross-demographic
    # SupCon term added to meta_loss. Unused by fino/dann/pcgrad.
    contrastive_temp = float(fino_cfg.get("contrastive_temp", 0.2)) if fino_cfg else 0.2
    contrastive_weight = float(fino_cfg.get("contrastive_weight", 0.1)) if fino_cfg else 0.1
    # Contrastive cancer conditioning: when set, the named discrete factor is used as a
    # condition (same-cancer positives) for the demographic contrastive loss. The conditioner
    # factor itself is skipped as a contrastive target.
    contrastive_condition_on = fino_cfg.get("contrastive_condition_on") if fino_cfg else None
    cond_factor_col = [f for f, _ in fino_disc].index(contrastive_condition_on) if contrastive_condition_on else None
    # Inverse-frequency demographic reweighting (ablation axis; default "none" == current behavior, byte-identical).
    # "inverse_freq" upweights minority demographic classes in the discrete-factor loss (see race_weights below).
    race_weight_mode = (fino_cfg.get("race_weight", "none") if fino_cfg else "none")
    if race_weight_mode not in ("none", "inverse_freq"):
        raise ValueError(f"fino.race_weight must be 'none' or 'inverse_freq', got {race_weight_mode!r}")
    # Race-balanced RESAMPLING (3rd race-handling mode, orthogonal to race_weight's loss reweighting).
    # When True the training DataLoader draws tiles via a WeightedRandomSampler with per-tile weight =
    # inverse race-class frequency (dataset.race_sample_weights), so minority races are oversampled to
    # parity in expectation. Default False -> sampler=None, shuffle=True (byte-identical to prior behavior).
    # Single-GPU assumption: these fairness runs are single-process (no DDP/DistributedSampler in this
    # train.py), so the weighted sampler simply replaces shuffle without needing a distributed-aware wrapper.
    race_resample = bool(fino_cfg.get("race_resample", False)) if fino_cfg else False
    # FINO two-phase: freeze the backbone (except patch_embed) for the first this-fraction of the run so the DINO/JEPA
    # heads + metadata prototypes/predictors converge against a fixed target before they steer the encoder. 0 = off.
    freeze_backbone_frac = float(dino_cfg.get("freeze_backbone_fraction", 0.0))
    # JEPA-T: optionally condition the JEPA predictor on a discrete factor (must be in fino.discrete so its per-tile
    # label rides in the batch). cond_col indexes that factor's column in batch["meta_disc"].
    jepa_cond = fino_cfg.get("jepa_cond") if fino_cfg else None
    cond_col = [f for f, _ in fino_disc].index(jepa_cond) if jepa_cond else None
    save_every = train_cfg["save_every"]
    save_checkpoints = save_every is not None
    device = torch.device("cuda")
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    variant = cfg["model"]["type"]
    student_backbone = load_dinov2_pretrained(DinoV2ViT(variant=variant, drop_path_rate=dino_cfg["drop_path_rate"])).to(device)
    teacher_backbone = deepcopy(student_backbone)
    teacher_backbone.train(False)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    student_dino_head = DINOHead(student_backbone.embed_dim, 131072, dino_cfg["head_hidden_dim"], dino_cfg["head_bottleneck_dim"], 3).to(device)
    teacher_dino_head = deepcopy(student_dino_head)
    student_predictor = JEPAPredictor(student_backbone.embed_dim, depth=int(dino_cfg["jepa_pred_depth"]), width=int(dino_cfg["jepa_pred_width"]), n_cond=(fino_meta["n"][jepa_cond] if jepa_cond else 0)).to(device)
    for p in teacher_dino_head.parameters():
        p.requires_grad = False
    backbone_activated_params = sum(p.numel() for p in student_backbone.parameters() if p.requires_grad)
    # FINO continuous-factor predictors (phi -> vector regressors); their params join the optimizer.
    predictors = {f: nn.Sequential(nn.Linear(student_backbone.embed_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, fino_meta.get("cont_dim", {}).get(f, 1))).to(device) for f, _ in fino_cont}
    # DANN adversaries (method=dann only): one learned linear classifier per discrete factor predicting the
    # attribute. A gradient reversal (model.GradScale with negative scale = -gamma) sits between the encoder CLS
    # and each head in compute_losses, so the head learns the attribute while the encoder is pushed to make CLS
    # non-predictive of it. No prototype bank. Their params join the optimizer like the continuous predictors.
    adversaries = {f: nn.Linear(student_backbone.embed_dim, fino_meta["n"][f]).to(device) for f, _ in fino_disc} if (fino_cfg and fairness_method in ("dann", "pcgrad")) else {}
    # AdamW param groups carry per-parameter LR/WD multipliers (LWD + patch_embed + biases-no-WD).
    param_groups = build_param_groups(student_backbone, student_dino_head, student_predictor, dino_cfg["layerwise_decay"], dino_cfg["patch_embed_lr_mult"])
    if predictors:
        param_groups.append({"params": [p for m in predictors.values() for p in m.parameters()], "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    if adversaries:
        param_groups.append({"params": [p for m in adversaries.values() for p in m.parameters()], "lr_mult": 1.0, "wd_mult": 1.0, "last_layer": False})
    # Per-discrete-factor inverse-frequency class weights (built only when fino.race_weight == "inverse_freq"; else {}).
    # SOURCE: fino_meta["discrete"][f] is the per-barcode {barcode: class_idx} map the labels are drawn from; we tally
    # patient-level class counts from it (no data pass needed, deterministic). NORMALIZATION: w_c = (P / sum_inv) * (1/n_c)
    # for classes with count n_c > 0 (P = #present classes), 0 for absent classes, so the mean weight over present
    # classes is exactly 1.0 (keeps the demographic-loss magnitude comparable to the unweighted default). Consumed as
    # the weight= arg of the demographic cross_entropy (dann/fino/pcgrad) and as per-anchor weights for contrastive.
    race_weights = {}
    if fino_cfg and race_weight_mode == "inverse_freq":
        for f, _ in fino_disc:
            ncls = int(fino_meta["n"][f])
            counts = torch.zeros(ncls, dtype=torch.double)
            for c in fino_meta.get("discrete", {}).get(f, {}).values():
                if isinstance(c, int) and 0 <= c < ncls:
                    counts[c] += 1.0
            inv = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
            present = (counts > 0).sum().clamp(min=1).double()
            w = inv * (present / inv.sum().clamp(min=1e-12))  # mean over present classes == 1.0
            race_weights[f] = w.float().to(device)
            print(f"{console_prefix()} [race_weight] factor={f} counts={counts.long().tolist()} "
                  f"weights={[round(x, 4) for x in w.tolist()]}", flush=True)
    opt = torch.optim.AdamW(param_groups, lr=1.0, betas=(0.9, dino_cfg["adam_beta2"]))
    # FINO prototype banks: one unit vector per discrete-factor value, EMA-updated from teacher CLS in compute_losses.
    # Not built for method=dann (that path uses the learned adversary heads above instead).
    protos = {f: F.normalize(torch.randn(fino_meta["n"][f], student_backbone.embed_dim, device=device), dim=-1) for f, _ in fino_disc} if (fino_cfg and fairness_method != "dann") else {}
    # FINO grad-equalisation EMA bank (one running grad-norm per factor); init 1.0 -> s_t~1 early. Not checkpointed
    # (mu=0.99 -> ~100-step memory, re-warms quickly on resume). Used only when fino.grad_equalize is set.
    grad_eq_ema = {f: torch.ones((), device=device) for f, _ in (fino_disc + fino_cont)} if fino_cfg else {}
    # PCGrad fairness (method=pcgrad): SSL-trained params whose gradient gets projected, vs the demographic-classifier
    # params trained normally (no reversal). pcgrad_apply computes the two gradients and projects at the step site.
    # Filter to trainable params: DINOHead's weight-norm magnitude (last_layer...original0) is a registered but frozen
    # (requires_grad=False) parameter; torch.autograd.grad rejects non-requiring inputs. Excluding it is a no-op
    # mathematically (its gradient is zero) and avoids a RuntimeError at the first pcgrad step.
    pcgrad_ssl_params = [p for p in (*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()) if p.requires_grad]
    pcgrad_clf_params = [p for p in ([p for m in adversaries.values() for p in m.parameters()] + [p for m in predictors.values() for p in m.parameters()]) if p.requires_grad]
    def pcgrad_apply(main_loss, dem_loss, verbose=False):
        g_main = torch.autograd.grad(main_loss, pcgrad_ssl_params, retain_graph=True, allow_unused=True)
        g_all = torch.autograd.grad(dem_loss, pcgrad_ssl_params + pcgrad_clf_params, allow_unused=True)
        n_ssl = len(pcgrad_ssl_params)
        new_main, info = pcgrad_project(g_main, g_all[:n_ssl], pcgrad_ssl_params)
        for p, g in zip(pcgrad_ssl_params, new_main):
            p.grad = g
        for p, g in zip(pcgrad_clf_params, g_all[n_ssl:]):
            if g is not None:
                p.grad = g
        if verbose:
            print(f"[pcgrad] cos_before={info['cos_before']:.4e} cos_after={info['cos_after']:.4e} "
                  f"dot={info['dot']:.4e} projected={info['projected']}", flush=True)
        return info
    step = 0
    batch_size = int(train_cfg["batch_size"])
    max_train_samples = int(train_cfg["max_train_samples"])
    examples_seen = 0
    visible_patch_presentations = 0
    train_flops = 0
    output_dir = Path(cfg["project"]["output_dir"])
    wandb_dir = Path(cfg["project"]["wandb_dir"])
    wandb_name = cfg["project"]["name"]
    if labless_autosubmit_file:
        wandb_name = json.loads(Path(labless_autosubmit_file).read_text()).get("run_name") or wandb_name
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    latest_checkpoint_path = output_dir / "latest.pt"
    # Fresh launches always start from scratch and wipe output_dir.
    resume_path = Path(train_cfg["resume"]) if train_cfg["resume"] else None
    if resume_path is None and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    wandb_meta = None
    if resume_path is not None:
        print(f"{console_prefix()} Resume  loading checkpoint: {resume_path}", flush=True)
        # Resume restores training progress, optimizer state, and wandb identity.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        student_backbone.load_state_dict(checkpoint["model"])
        teacher_backbone.load_state_dict(checkpoint["model_ema"])
        student_dino_head.load_state_dict(checkpoint["dino_head"])
        teacher_dino_head.load_state_dict(checkpoint["dino_head_ema"])
        student_predictor.load_state_dict(checkpoint["predictor"])
        opt.load_state_dict(checkpoint["opt"])
        if fino_cfg:
            protos = {k: v.to(device) for k, v in checkpoint["protos"].items()}
            for f, mdl in predictors.items():
                mdl.load_state_dict(checkpoint["predictors"][f])
            for f, mdl in adversaries.items():
                mdl.load_state_dict(checkpoint["adversaries"][f])
        step = int(checkpoint["step"])
        examples_seen = int(checkpoint["examples_seen"])
        visible_patch_presentations = int(checkpoint["visible_patch_presentations"])
        train_flops = int(checkpoint["train_flops"])
        wandb_meta = dict(checkpoint["wandb"])
    wandb_init = {
        "project": "nanopath",
        "name": wandb_name,
        "dir": str(wandb_dir),
        "config": cfg,
        "settings": wandb.Settings(
            console="wrap",
            x_file_stream_transmit_interval=5,
        ),
    }
    if wandb_meta is not None:
        wandb_init["id"] = wandb_meta["id"]
        wandb_init["resume"] = "must"
    wandb_run = wandb.init(**wandb_init)
    for key in ("probe/target_flops", "probe/wall_seconds"):
        wandb_run.define_metric(key, hidden=True, overwrite=True)
    print(
        f"{console_prefix()} Run  start: {wandb_name}  "
        f"config: {cfg['config_path']}  batch_size: {batch_size}  max_train_samples: {max_train_samples}  "
        f"max_train_flops: {train_cfg['max_train_flops']}  "
        f"probe_count: {cfg['probe']['count']}  warmup_fraction: {dino_cfg['warmup_fraction']}  "
        f"lr: {dino_cfg['lr']}  adam_beta2: {dino_cfg['adam_beta2']}  kde_loss_weight: {dino_cfg['kde_loss_weight']}  "
        f"kde_concentration: {dino_cfg['kde_concentration']}  drop_path: {dino_cfg['drop_path_rate']}  "
        f"layerwise_decay: {dino_cfg['layerwise_decay']}",
        flush=True,
    )
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    git_remote = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=repo_dir, text=True, capture_output=True, check=False).stdout.strip()
    source_id = f"nanopath-source-{wandb_run.id}"
    artifact_ignore = [
        line.strip() for line in (repo_dir / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ] + [".git/", "baselines/", "slurm/", "AGENTS.md", "CLAUDE.md"]
    ignored_roots = [output_dir.resolve(), wandb_dir.resolve()]

    def artifact_ignored(path):
        if any(path.resolve().is_relative_to(root) for root in ignored_roots):
            return True
        rel_path = path.relative_to(repo_dir)
        if any(part.startswith(".") for part in rel_path.parts):
            return True
        rel, name = rel_path.as_posix(), path.name
        for pat in artifact_ignore:
            pat = pat.rstrip("/") if pat.endswith("/") else pat
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) or rel == pat or rel.startswith(pat + "/"):
                return True
        return False

    source_files = []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if not artifact_ignored(Path(root) / d))
        for name in sorted(files):
            path = Path(root) / name
            if artifact_ignored(path):
                continue
            rel = path.relative_to(repo_dir)
            source_files.append((path, rel))
    source_snapshot_dir = output_dir / "labless_source"
    if source_snapshot_dir.exists():
        shutil.rmtree(source_snapshot_dir)
    for path, rel in source_files:
        target = source_snapshot_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    wandb_meta = {"entity": wandb_run.entity, "project": "nanopath", "id": wandb_run.id, "name": wandb_name, "url": wandb_run.url,
                  "mode": getattr(wandb_run.settings, "mode", ""), "source_artifact": source_id,
                  "source_dir": str(source_snapshot_dir), "git": {"commit": git_commit, "remote": git_remote}}
    train_ds = TCGATileDataset(cfg, is_train=True)
    val_ds = TCGATileDataset(cfg, is_train=False)
    probe_state = prepare_probe_state(cfg, output_dir) if probe_enabled(cfg) else None

    # Train shuffles + drops partials; the loop never starts a batch that would exceed
    # max_train_samples, so every optimizer step keeps the configured batch size.
    loader_kwargs = dict(batch_size=batch_size, drop_last=True, num_workers=train_cfg["num_workers"], pin_memory=True,
                         prefetch_factor=train_cfg["prefetch_factor"] if train_cfg["num_workers"] > 0 else None,
                         persistent_workers=train_cfg["persistent_workers"] and train_cfg["num_workers"] > 0)
    # Race-balanced resampling: build a WeightedRandomSampler over the train split (replacement=True,
    # one epoch's worth of draws) that oversamples minority races to parity. Only when fino.race_resample;
    # otherwise train_sampler stays None and the loader keeps shuffle=True (byte-identical default path).
    train_sampler = None
    if race_resample:
        race_w = train_ds.race_sample_weights("race")
        train_sampler = WeightedRandomSampler(race_w, num_samples=len(train_ds), replacement=True)
        print(f"{console_prefix()} [race_resample] weighted sampler over {len(train_ds)} tiles "
              f"(w min/mean/max={race_w.min():.4f}/{race_w.mean():.4f}/{race_w.max():.4f})", flush=True)
    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    activation_checkpointing = bool(train_cfg["activation_checkpointing"])
    global_grid = train_cfg["global_size"] // student_backbone.patch_size
    global_patches = global_grid ** 2
    local_patches = (train_cfg["local_size"] // student_backbone.patch_size) ** 2
    last_time = time.time()
    last_examples = examples_seen
    last_visible_patch_presentations = visible_patch_presentations
    last_train_flops = train_flops
    unique_tile_patch_count = (TILE_SIZE // student_backbone.patch_size) ** 2
    seen_ids = {"sample": set(), "slide": set(), "patient": set()}
    pending_ids = {key: set() for key in seen_ids}

    # cpu_state(m) materializes an on-CPU copy of a module's state_dict for torch.save.
    def cpu_state(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # Full checkpoint (latest.pt) covers explicit train.resume whereas probe checkpoint is a slim
    # weights-only ckpt, given probe.py does not need optimizer or projection heads.
    def checkpoint_payload(next_step, full):
        payload = {"model": cpu_state(student_backbone), "model_ema": cpu_state(teacher_backbone), "step": next_step, "config": cfg}
        if not full:
            return payload
        return {**payload, "dino_head": cpu_state(student_dino_head), "dino_head_ema": cpu_state(teacher_dino_head),
                "predictor": cpu_state(student_predictor), "opt": opt.state_dict(),
                "examples_seen": examples_seen, "visible_patch_presentations": visible_patch_presentations,
                "train_flops": train_flops, "wandb": wandb_meta,
                **({"protos": {k: v.cpu() for k, v in protos.items()}, "predictors": {f: cpu_state(m) for f, m in predictors.items()},
                    "adversaries": {f: cpu_state(m) for f, m in adversaries.items()}} if fino_cfg else {})}

    def save_latest_checkpoint(checkpoint_step):
        nonlocal last_saved_step
        print(f"{console_prefix()} Checkpoint  [{checkpoint_step}]  save: latest.pt", flush=True)
        tmp_path = latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint_payload(checkpoint_step, full=True), tmp_path)
        os.replace(tmp_path, latest_checkpoint_path)
        for stale_checkpoint_path in output_dir.glob("step_*.pt"):
            stale_checkpoint_path.unlink()
        last_saved_step = checkpoint_step

    # Count unique tiles/slides/patients for data-coverage diagnostics.
    def flush_unique_counts():
        for key in seen_ids:
            seen_ids[key].update(pending_ids[key])
            pending_ids[key].clear()
        unique_tiles_seen = len(seen_ids["sample"])
        return {
            "unique_slides_seen": len(seen_ids["slide"]),
            "unique_patients_seen": len(seen_ids["patient"]),
            "unique_tiles_seen": unique_tiles_seen,
            "unique_patches_seen": unique_tiles_seen * unique_tile_patch_count,
        }

    # Compute (dino_loss, jepa_loss, kde) for one batch of (gf, lf) crops with the given masks +
    # schedule values. Used by both the train step and evaluate() (no_grad).
    def compute_losses(gf, lf, b, masks, mask_idx, mask_w, t_temp, k_scale, ckpt=False, meta=None, cond=None):
        with torch.no_grad():
            t = teacher_backbone(gf)
            t_cls = teacher_dino_head(t["x_norm_clstoken"]).chunk(train_cfg["global_views"])
            t_prob = sinkhorn(torch.cat((t_cls[1], t_cls[0])), t_temp).view(2, b, -1)
        sg = student_backbone(gf, masks=masks, checkpoint=ckpt)
        sl = student_backbone(lf, checkpoint=ckpt)
        sg_cls, sl_cls = student_dino_head(sg["x_norm_clstoken"]), student_dino_head(sl["x_norm_clstoken"])
        L = train_cfg["local_views"]
        local_loss = sum(dino_ce(x, y) for x in sl_cls.chunk(L) for y in t_prob) / (2 * L + 2)
        global_loss = dino_ce(sg_cls, t_prob.flatten(0, 1)) * 2 / (2 * L + 2)
        target = F.layer_norm(t["x_norm_patchtokens"].flatten(0, 1), (student_backbone.embed_dim,))[mask_idx]
        pred = student_predictor(sg["x_norm_patchtokens"], cond).flatten(0, 1)[mask_idx]
        jepa_loss = F.smooth_l1_loss(pred, target, reduction="none").mean(-1).mul(mask_w).sum() / max(1, b * 2)
        kde = dino_cfg["kde_loss_weight"] * k_scale * sum(kde_loss(x, dino_cfg["kde_concentration"]) for x in sg["x_norm_clstoken"].chunk(train_cfg["global_views"]))
        # FINO metadata guidance on the CLS token (train-only; meta=None in eval), orthogonal to the JEPA patch
        # objective. lambda_meta=0.03/branch; GradScale gates the encoder gradient by the DANN ramp gamma with the
        # per-factor sign (+ M+ encourage / - M- suppress). fp32 island (1/tau=0.023 too sharp for bf16); missing
        # factors masked. Discrete: L2-normed student CLS vs EMA prototype bank (clone-rebind keeps the backward-saved
        # bank valid). Continuous: an MLP regresses the z-scored value.
        meta_loss = sg["x_norm_clstoken"].new_zeros(())
        if meta is not None:
            gamma, md, mc = meta  # md (B,n_disc) int64 (-1 missing); mc {factor: (B,dim) float, nan missing}
            phi_s = F.normalize(sg["x_norm_clstoken"].float(), dim=-1)
            phi_t = F.normalize(t["x_norm_clstoken"].float(), dim=-1)
            terms = []  # (factor, per-branch loss 0.03*L_t); combined below, optionally gradient-equalized
            with torch.autocast(device_type="cuda", enabled=False):
                for j, (f, sign) in enumerate(fino_disc):
                    lab = md[:, j].repeat(train_cfg["global_views"]); ok = lab >= 0  # repeat, NOT interleave
                    if ok.any():
                        if fairness_method == "dann":
                            # DANN: a learned linear adversary predicts the attribute from the RAW student CLS; the
                            # GradScale reverses the encoder gradient (sign*gamma; fairness sign=-1 -> -gamma), so the
                            # head is trained to classify the attribute while the encoder is pushed to unlearn it. Same
                            # sigmoid gamma ramp and missing-label mask (ok) as FINO; no prototype bank / no EMA update.
                            logits = adversaries[f](GradScale.apply(sg["x_norm_clstoken"].float()[ok], sign * gamma))
                            terms.append((f, 0.03 * F.cross_entropy(logits, lab[ok], weight=race_weights.get(f))))
                        elif fairness_method == "pcgrad":
                            # PCGrad: plain demographic CE from a learned classifier on the RAW student CLS, NO gradient
                            # reversal. This term is returned as meta_loss but the pcgrad step site does NOT add it to
                            # total_loss; it uses this gradient to project the SSL gradient (see pcgrad_apply).
                            logits = adversaries[f](sg["x_norm_clstoken"].float()[ok])
                            terms.append((f, F.cross_entropy(logits, lab[ok], weight=race_weights.get(f))))
                        elif fairness_method == "contrastive":
                            if fairness_objective == "contrastive-cancer":
                                terms.append((f, contrastive_weight * cancer_supcon(
                                    phi_s[ok], lab[ok], contrastive_temp
                                )))
                            elif fairness_objective == "contrastive-two-condition":
                                if j == cond_factor_col:
                                    continue
                                cond_label = md[:, cond_factor_col].repeat(train_cfg["global_views"])
                                both_known = ok & (cond_label >= 0)
                                terms.append((f, contrastive_weight * fair_supcon(
                                    phi_s[both_known],
                                    lab[both_known],
                                    contrastive_temp,
                                    w=race_weights.get(f),
                                    cond=cond_label[both_known],
                                )))
                            else:
                                terms.append((f, contrastive_weight * fair_supcon(
                                    phi_s[ok], lab[ok], contrastive_temp, w=race_weights.get(f)
                                )))
                        else:
                            logits = (GradScale.apply(phi_s[ok], sign * gamma) @ protos[f].t()) / 0.023
                            terms.append((f, 0.03 * F.cross_entropy(logits, lab[ok], weight=race_weights.get(f))))
                            with torch.no_grad():
                                pt, lt = phi_t[ok], lab[ok]
                                upd = torch.zeros_like(protos[f]).index_add_(0, lt, pt)
                                cnt = torch.zeros(protos[f].shape[0], 1, device=device).index_add_(0, lt, torch.ones_like(pt[:, :1]))
                                seen = cnt.squeeze(1) > 0; new = protos[f].clone()
                                new[seen] = F.normalize(0.99 * new[seen] + 0.01 * (upd[seen] / cnt[seen]), dim=-1); protos[f] = new
                # FINO Eq.3 regresses continuous factors from the RAW backbone CLS; phi_s is L2-normalized (needed only
                # for the cosine discrete branch and it strips the radial magnitude). raw_cls=True feeds the raw CLS.
                cls_cont = sg["x_norm_clstoken"].float() if fino_cfg.get("raw_cls") else phi_s
                for f, sign in fino_cont:
                    val = mc[f].repeat(train_cfg["global_views"], 1); ok = ~torch.isnan(val).any(dim=1)
                    if ok.any():
                        if fairness_method == "pcgrad":
                            cpred = predictors[f](cls_cont[ok])  # NO reversal: continuous demographic-prediction gradient
                            terms.append((f, F.mse_loss(cpred, val[ok])))
                        else:
                            cpred = predictors[f](GradScale.apply(cls_cont[ok], sign * gamma))
                            terms.append((f, 0.03 * F.mse_loss(cpred, val[ok])))
                # FINO Alg A.3 per-branch gradient equalisation: rescale each branch by n_bar/EMA(||dL_t/dCLS||) so the
                # discrete-CE and continuous-MSE gradients reach the encoder at matched magnitudes (detached -> reweight
                # only; geometric-mean target; no-op for <2 branches). grad_eq_ema = per-factor EMA bank (mu=0.99).
                if fino_cfg.get("grad_equalize") and len(terms) > 1:
                    g = {f: torch.autograd.grad(L, sg["x_norm_clstoken"], retain_graph=True)[0].norm() for f, L in terms}
                    for f in g: grad_eq_ema[f] = 0.99 * grad_eq_ema[f] + 0.01 * g[f].detach().float()
                    nbar = torch.exp(torch.stack([grad_eq_ema[f].log() for f, _ in terms]).mean())
                    meta_loss = sum((nbar / grad_eq_ema[f]).detach() * L for f, L in terms)
                else:
                    for _, L in terms: meta_loss = meta_loss + L
        return local_loss + global_loss, jepa_loss, kde, meta_loss

    # Held-out validation pass: same DINO + JEPA + KDE losses on `val_batches` of the val split.
    # Schedule terms (teacher_temp, kde_scale) drift over training, so read val curves as same-step
    # diagnostics. RNG is snapshotted/restored so val masks don't perturb the next training step.
    def evaluate(eval_step, eval_teacher_temp, eval_kde_scale):
        for m in (student_backbone, student_dino_head, student_predictor):
            m.eval()
        py_rng, cpu_rng, cuda_rng = random.getstate(), torch.random.get_rng_state(), torch.cuda.get_rng_state(device)
        random.seed(train_cfg["seed"] + eval_step)
        torch.manual_seed(train_cfg["seed"] + eval_step)
        sums = torch.zeros(4, device=device)
        n_batches = 0
        for vb_idx, vbatch in enumerate(val_loader):
            if vb_idx >= int(train_cfg["val_batches"]):
                break
            vg, vl = vbatch["global_views"].to(device, non_blocking=True), vbatch["local_views"].to(device, non_blocking=True)
            b = vg.shape[0]
            with torch.no_grad(), autocast:
                gf, lf = vg.transpose(0, 1).flatten(0, 1), vl.transpose(0, 1).flatten(0, 1)
                masks, mask_idx, mask_w = make_block_mask(b * train_cfg["global_views"], global_grid, device, n_blocks=int(dino_cfg["jepa_blocks"]), block_scale=float(dino_cfg["jepa_block_scale"]))
                dino_l, jepa_l, kde_v, _ = compute_losses(gf, lf, b, masks, mask_idx, mask_w, eval_teacher_temp, eval_kde_scale)
            sums += torch.tensor([float(dino_l), float(jepa_l), float(kde_v), float(dino_l + jepa_l + kde_v)], device=device)
            n_batches += 1
        random.setstate(py_rng)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return dict(zip(("dino", "jepa", "kde", "total"), (sums / max(1, n_batches)).tolist()))

    # Ingest completed probe result JSONs into metrics.jsonl and wandb.
    def log_probe_results():
        if probe_state is not None:
            collect_probe_results(probe_state, wandb_run, metrics_path)

    # Queue a probe at `checkpoint_step` for the given sample target; no-op if already done.
    def run_probe_at(checkpoint_step, target_samples):
        if probe_state is None or (probe_state["paths"]["results_dir"] / f"step_{checkpoint_step:07d}.json").exists():
            log_probe_results()
            return
        queue_probe_job(probe_state, checkpoint_payload(checkpoint_step, full=False), checkpoint_step, train_flops, min(1.0, target_samples / max_train_samples))
        log_probe_results()

    # Queue the furthest crossed sample milestone so delayed probes do not run on stale checkpoints.
    def maybe_run_probe(checkpoint_step):
        nonlocal next_probe_idx
        if probe_state is None or next_probe_idx >= len(probe_targets) or examples_seen < probe_targets[next_probe_idx]:
            return
        while next_probe_idx + 1 < len(probe_targets) and examples_seen >= probe_targets[next_probe_idx + 1]:
            next_probe_idx += 1
        run_probe_at(checkpoint_step, probe_targets[next_probe_idx])
        next_probe_idx += 1

    log_probe_results()
    max_train_flops = int(train_cfg["max_train_flops"])
    warmup_train_samples = math.ceil(max_train_samples * dino_cfg["warmup_fraction"])
    # Probe targets are sample milestones: one tile counts once even with many global/local crops.
    probe_count = int(cfg["probe"]["count"]) if probe_enabled(cfg) else 0
    probe_targets = [math.ceil(max_train_samples * (i + 1) / probe_count) for i in range(probe_count)]
    if len(set(probe_targets)) != len(probe_targets):
        raise ValueError(f"probe.count={probe_count} is too large for max_train_samples={max_train_samples}")
    next_probe_idx = 0
    if probe_state is not None:
        completed = [round(float(json.loads(p.read_text()).get("target_fraction", -1)) * max_train_samples) for p in probe_state["paths"]["results_dir"].glob("step_*.json")]
        if completed:
            next_probe_idx = sum(target <= max(completed) for target in probe_targets)
    train_loop_started_at = time.monotonic()
    last_saved_step = step
    last_console_step = step
    last_console_monotonic = time.monotonic()
    data_wait_started_at = time.monotonic()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if train_cfg["bf16"] else contextlib.nullcontext()
    # Per-step FLOPs are measured once via FlopCounterMode on the first wrapped step (forward +
    # backward + opt.step) and reused for every subsequent step since the shapes don't change.
    # Counts the EMA teacher forward + DINO/JEPA heads, not just the backbone, so the
    # 1e18 leaderboard cap reflects real GPU work.
    measured_flops_per_step = None

    while examples_seen + batch_size <= max_train_samples and train_flops < max_train_flops:
        for batch in train_loader:
            if examples_seen + batch_size > max_train_samples or train_flops >= max_train_flops:
                break
            batch_started_at = time.monotonic()
            data_seconds = batch_started_at - data_wait_started_at
            student_backbone.train()
            student_dino_head.train()
            student_predictor.train()
            completed_step = step + 1
            should_log = completed_step == 1 or completed_step % train_cfg["log_every"] == 0
            # Data identifiers stay on CPU and feed coverage metrics; image tensors move below.
            for key, batch_key in (("sample", "sample_idx"), ("slide", "slide_id"), ("patient", "patient_id")):
                pending_ids[key].update(int(x) for x in batch[batch_key].tolist())
            global_views, local_views = [batch[key].to(device, non_blocking=True) for key in ("global_views", "local_views")]
            visible_now = batch_size * (train_cfg["global_views"] * global_patches + train_cfg["local_views"] * local_patches)
            # LR warmup uses the 1M-tile sample cap; decay/WD/teacher/freeze/KDE default to the public FLOP budget.
            # But this run hits the sample cap at ~19% of the FLOP budget, so a FLOP-keyed cosine only traverses ~0.11
            # of its arc (LR never anneals, KDE peaks at 0.22, WD ~0.05). lr_key/reg_key="sample" re-key the decay/reg
            # schedules to SAMPLE progress so they complete over the actual 1M-tile run (same fix as the FINO gamma ramp).
            frac = min(1.0, train_flops / max_train_flops)
            sfrac = min(1.0, examples_seen / max_train_samples)
            lr_frac = sfrac if dino_cfg.get("lr_key") == "sample" else frac
            reg_frac = sfrac if dino_cfg.get("reg_key") == "sample" else frac
            warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
            if warmup < 1.0:
                lr = dino_cfg["lr"] * warmup
            else:
                lr = cosine_schedule(dino_cfg["lr"], dino_cfg["lr_min"], (lr_frac - dino_cfg["warmup_fraction"]) / max(1e-9, 1 - dino_cfg["warmup_fraction"]))
            wd = cosine_schedule(0.04, 0.2, reg_frac)
            teacher_temp = 0.04 + min(1.0, reg_frac / 0.2727) * (0.07 - 0.04)
            last_layer_lr = 0.0 if frac < dino_cfg["freeze_last_layer_fraction"] else lr
            for group in opt.param_groups:
                base_lr = last_layer_lr if group["last_layer"] else lr
                group["lr"] = base_lr * group["lr_mult"]
                group["weight_decay"] = wd * group["wd_mult"]
            masks, mask_idx, mask_w = make_block_mask(batch_size * train_cfg["global_views"], global_grid, device, n_blocks=int(dino_cfg["jepa_blocks"]), block_scale=float(dino_cfg["jepa_block_scale"]))
            kde_scale = min(1.0, max(0.0, (reg_frac - 0.1) / 0.4))
            # Wrap forward + backward + opt.step in FlopCounterMode on the first step only;
            # subsequent steps reuse measured_flops_per_step (fixed shapes => fixed cost).
            # FlopCounterMode installs _will_engine_execute_node autograd hooks that are incompatible
            # with torch.autograd.grad() (the method=pcgrad path -> pcgrad_apply): wrapping the backward
            # raises "A leaf node was passed to _will_engine_execute_node but we are currently running
            # autograd.grad()". So for pcgrad we meter the FORWARD only (a nested counter entered here and
            # exited before pcgrad_apply) and scale by 3 (fwd + 2*bwd, the standard single-step estimate)
            # so measured_flops_per_step stays a sane positive value; every other method wraps the whole
            # step exactly as before. (meta is not None iff fino_cfg is set -- see the pcgrad branch below.)
            measuring = measured_flops_per_step is None
            forward_only = measuring and fairness_method == "pcgrad" and bool(fino_cfg)
            flop_ctx = FlopCounterMode(display=False) if (measuring and not forward_only) else contextlib.nullcontext()
            fwd_ctx = FlopCounterMode(display=False) if forward_only else None
            if fwd_ctx is not None:
                fwd_ctx.__enter__()
            with flop_ctx:
                with autocast:
                    # Crop-major flatten: collate shape is (B, V, 3, H, W) but DINO wants per-crop chunks
                    # so [crop0_img0, crop0_img1, ..., crop1_img0, ...] for clean teacher/student alignment.
                    gf = global_views.transpose(0, 1).flatten(0, 1)
                    lf = local_views.transpose(0, 1).flatten(0, 1)
                    # FINO DANN ramp keyed to nanopath's SAMPLE budget (NOT FLOPs — sample-capped at ~19% of the FLOP
                    # cap, so a flop-keyed ramp stalls gamma at ~0.75*gamma_max). Counted from the backbone-unfreeze
                    # point: gamma=0 through the frozen Phase 1 (banks warm), then ramps to full gamma_max by the cap.
                    ramp = max(0.0, (examples_seen / max_train_samples - freeze_backbone_frac) / max(1e-6, 1.0 - freeze_backbone_frac))
                    meta = ((fino_cfg["gamma_max"] * (2.0 / (1.0 + math.exp(-10.0 * ramp)) - 1.0),
                             batch["meta_disc"].to(device, non_blocking=True),
                             {f: batch["mc_" + f].to(device, non_blocking=True) for f, _ in fino_cont}) if fino_cfg else None)
                    cond = batch["meta_disc"][:, cond_col].repeat(train_cfg["global_views"]).to(device, non_blocking=True) if jepa_cond else None
                    dino_loss_value, jepa_loss, kde, meta_loss = compute_losses(
                        gf, lf, batch_size, masks, mask_idx, mask_w, teacher_temp, kde_scale,
                        ckpt=activation_checkpointing, meta=meta, cond=cond,
                    )
                    total_loss = dino_loss_value + jepa_loss + kde + meta_loss
                if fwd_ctx is not None:
                    # Close the forward-only counter before autograd.grad() runs (pcgrad path).
                    fwd_ctx.__exit__(None, None, None)
                if fairness_method == "pcgrad" and meta is not None:
                    # PCGrad fairness: no single total_loss.backward(). Get the main SSL gradient and the demographic
                    # gradient separately, project the former onto the orthogonal complement of the latter, then apply.
                    opt.zero_grad(set_to_none=True)
                    pcgrad_apply(dino_loss_value + jepa_loss + kde, meta_loss)
                else:
                    opt.zero_grad(set_to_none=True)
                    total_loss.backward()
                if examples_seen / max_train_samples < freeze_backbone_frac:  # Phase 1: backbone frozen (patch_embed + heads + metadata still train)
                    for n, p in student_backbone.named_parameters():
                        if not n.startswith("patch_embed"): p.grad = None
                grad_norm = nn.utils.clip_grad_norm_(
                    [*student_backbone.parameters(), *student_dino_head.parameters(), *student_predictor.parameters()],
                    dino_cfg["clip_grad"],
                )
                opt.step()
            if measured_flops_per_step is None:
                measured_flops_per_step = (3 * int(fwd_ctx.get_total_flops())) if forward_only else int(flop_ctx.get_total_flops())
                print(f"{console_prefix()} measured_flops_per_step: {measured_flops_per_step:,}", flush=True)
            step_train_flops = measured_flops_per_step
            with torch.no_grad():
                m = cosine_schedule(0.994, 1.0, reg_frac)
                update_ema(student_backbone, teacher_backbone, m)
                update_ema(student_dino_head, teacher_dino_head, m)
            step_seconds = time.monotonic() - batch_started_at
            examples_seen += batch_size
            visible_patch_presentations += visible_now
            train_flops += step_train_flops
            if should_log:
                reduced = {
                    "dino": float(dino_loss_value.detach()),
                    "jepa": float(jepa_loss.detach()),
                    "kde": float(kde.detach()),
                    "total": float(total_loss.detach()),
                }
                unique_counts = flush_unique_counts()
                now = time.time()
                elapsed = max(1e-6, now - last_time)
                items_per_sec = (examples_seen - last_examples) / elapsed
                visible_patches_per_sec = (visible_patch_presentations - last_visible_patch_presentations) / elapsed
                flops_per_sec = (train_flops - last_train_flops) / elapsed
                train_loop_wall_seconds = time.monotonic() - train_loop_started_at
                last_time = now
                last_examples = examples_seen
                last_visible_patch_presentations = visible_patch_presentations
                last_train_flops = train_flops
                gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024**3)
                gpu_peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                console_now = time.monotonic()
                console_gap_ms = 1000.0 * (console_now - last_console_monotonic)
                steps_since_console = max(1, completed_step - last_console_step)
                flop_steps_remaining = math.ceil(max(0, max_train_flops - train_flops) / max(1, step_train_flops))
                sample_steps_remaining = max(0, max_train_samples - examples_seen) // batch_size
                steps_remaining = min(flop_steps_remaining, sample_steps_remaining)
                total_steps_estimate = completed_step + steps_remaining
                eta_seconds = int(max(0.0, steps_remaining * console_gap_ms / 1000.0 / steps_since_console))
                eta_string = f"{eta_seconds // 3600}:{(eta_seconds % 3600) // 60:02d}:{eta_seconds % 60:02d}"
                current_lr = opt.param_groups[0]["lr"]
                train_log = {
                    "step": completed_step,
                    **reduced,
                    "items_per_sec": items_per_sec,
                    "visible_patches_per_sec": visible_patches_per_sec,
                    "flops_per_sec": flops_per_sec,
                    "wall_seconds": train_loop_wall_seconds,
                    "step_seconds": step_seconds,
                    "data_seconds": data_seconds,
                    "console_gap_ms": console_gap_ms,
                    "eta_seconds": eta_seconds,
                    "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
                    "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
                    "lr": current_lr,
                    "wd": wd,
                    "teacher_temp": teacher_temp,
                    "teacher_momentum": m,
                    "kde_scale": kde_scale,
                    "batch_size": batch_size,
                    "examples_seen": examples_seen,
                    "visible_patch_presentations": visible_patch_presentations,
                    "train_flops": train_flops,
                    "gpu_mem_gb": gpu_mem_gb,
                    "gpu_peak_mem_gb": gpu_peak_mem_gb,
                    "grad_norm": float(grad_norm.detach()),
                }
                train_log.update(unique_counts)
                print(
                    f"{console_prefix()} Training  "
                    f"[{completed_step}/{total_steps_estimate}]  eta: {eta_string}  gap: {console_gap_ms:.2f} ms  "
                    f"lr: {current_lr:.6f}  total: {reduced['total']:.4f}  "
                    f"dino: {reduced['dino']:.4f}  jepa: {reduced['jepa']:.4f}  kde: {reduced['kde']:.4f}  "
                    f"grad_norm: {train_log['grad_norm']:.4f}  flops/s: {flops_per_sec:.3e}  "
                    f"time: {step_seconds:.6f}  data: {data_seconds:.6f}  "
                    f"max mem: {int(gpu_peak_mem_gb * 1024)}",
                    flush=True,
                )
                last_console_step = completed_step
                last_console_monotonic = console_now
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(train_log) + "\n")
                wandb_run.log(
                    {f"train/{key}": value for key, value in train_log.items() if key != "step"},
                    step=completed_step,
                )
                log_probe_results()
                torch.cuda.reset_peak_memory_stats(device)
            if save_checkpoints and completed_step % save_every == 0:
                # Atomic rename keeps the previous good latest.pt intact if a
                # kill lands mid-save.
                save_latest_checkpoint(completed_step)
            # Probe at intermediate sample milestones (probe.count > 1); the final probe
            # always runs after the loop exits, regardless of milestones.
            maybe_run_probe(completed_step)
            if completed_step % int(train_cfg["eval_every"]) == 0 or train_flops >= max_train_flops or examples_seen + batch_size > max_train_samples:
                val = evaluate(completed_step, teacher_temp, kde_scale)
                val_log = {"step": completed_step, **{f"val_{k}": v for k, v in val.items()}}
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(val_log) + "\n")
                wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=completed_step)
                print(f"{console_prefix()} Validation  [{completed_step}]  total: {val['total']:.4f}  dino: {val['dino']:.4f}  jepa: {val['jepa']:.4f}  kde: {val['kde']:.4f}", flush=True)
                # Reset rate clocks after validation so the next train log is train-rate only.
                last_console_step, last_console_monotonic = completed_step, time.monotonic()
                last_time, last_examples, last_visible_patch_presentations, last_train_flops = time.time(), examples_seen, visible_patch_presentations, train_flops
            step = completed_step
            data_wait_started_at = time.monotonic()
            if train_flops >= max_train_flops or examples_seen + batch_size > max_train_samples:
                break
    train_loop_wall_seconds = time.monotonic() - train_loop_started_at
    stop_reason = "max_train_flops" if train_flops >= max_train_flops else "max_train_samples"
    final_unique_counts = flush_unique_counts()
    if step > 0:
        # Final probes have their own readers; close pretraining workers before they compete for CPU/IO.
        if train_cfg["num_workers"] > 0:
            if train_loader._iterator is not None:
                train_loader._iterator._shutdown_workers()
                train_loader._iterator = None
        # Probes get their own short-lived checkpoint via run_probe_at; only persist latest.pt
        # at end-of-run when periodic saving is on (save_every set) so smoke runs leave nothing.
        if save_checkpoints and step != last_saved_step:
            save_latest_checkpoint(step)
        run_probe_at(step, examples_seen)
    log_probe_results()
    # Summary is the small, stable artifact downstream scripts and humans compare across runs.
    summary = {
        "project": cfg["project"]["name"],
        "family": cfg["project"]["family"],
        "recipe_id": cfg["project"]["recipe_id"],
        "config_path": cfg["config_path"],
        "wandb": wandb_meta,
        "slurm_job_id": slurm_job_id,
        "backbone_activated_params": backbone_activated_params,
        "batch_size": batch_size,
        "max_train_samples": max_train_samples,
        "max_train_flops": max_train_flops,
        "train_loop_wall_seconds": train_loop_wall_seconds,
        "stop_reason": stop_reason,
        "steps_completed": step,
        "tile_presentations": examples_seen,
        "visible_patch_presentations": visible_patch_presentations,
        **final_unique_counts,
        "train_flops": train_flops,
        "flop_fraction": min(1.0, float(train_flops) / float(max_train_flops)),
        "sample_fraction": min(1.0, float(examples_seen) / float(max_train_samples)),
        # Average throughput over the train loop; wall time is diagnostic, not an eligibility cap.
        "flops_per_sec": train_flops / max(1.0, train_loop_wall_seconds),
        "visible_patches_per_sec": visible_patch_presentations / max(1.0, train_loop_wall_seconds),
        "warmup_fraction": dino_cfg["warmup_fraction"],
        "warmup_train_samples": warmup_train_samples,
        "lr": dino_cfg["lr"],
        "adam_beta2": dino_cfg["adam_beta2"],
        "kde_loss_weight": dino_cfg["kde_loss_weight"],
        "kde_concentration": dino_cfg["kde_concentration"],
        "drop_path_rate": dino_cfg["drop_path_rate"],
        "layerwise_decay": dino_cfg["layerwise_decay"],
        "probe_target_samples": probe_targets,
        "probe_target_fractions": [None if max_train_samples == 0 else target / max_train_samples for target in probe_targets],
        **({} if probe_state is None else completed_probe_summary(output_dir)),
    }
    if probe_state is not None and "final_probe_score" not in summary:
        raise ValueError("probe.enabled is true but final_probe_score is missing; check probe.count, probe failures, and final checkpoint scheduling")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{console_prefix()} Summary  "
        f"steps: {step}  train_wall: {train_loop_wall_seconds:.2f}s  "
        f"final_probe_score: {summary.get('final_probe_score')}",
        flush=True,
    )
    for key in summary.keys():
        wandb_run.summary[key] = summary[key]
    wandb_run.finish()
    finish_labless_autosubmit(labless_autosubmit_file, output_dir, repo_dir)


if __name__ == "__main__":
    main()
