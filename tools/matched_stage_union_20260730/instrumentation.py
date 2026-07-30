"""Schedule-neutral diagnostics used by matched-stage pretraining."""

from __future__ import annotations

import contextlib
import math
from typing import Iterable

import torch
from torch.utils.flop_counter import FlopCounterMode


def uses_fino_prototypes(fairness_method: str, fino_enabled: bool) -> bool:
    """Only the FINO prototype objective may allocate a random prototype bank."""

    return fino_enabled and fairness_method == "fino"


def forward_flop_meter(enabled: bool):
    """Return a forward-only FLOP context, independent of dose logging.

    The trainer uses the same first-step metering path whether diagnostics are
    enabled or disabled.  Backward work is estimated as twice the forward
    work, avoiding autograd-hook interactions with ``autograd.grad``.
    """

    return FlopCounterMode(display=False) if enabled else contextlib.nullcontext()


def estimated_step_flops(forward_meter: FlopCounterMode) -> int:
    """Convert a completed forward-only meter into a positive step estimate."""

    value = 3 * int(forward_meter.get_total_flops())
    if value <= 0:
        raise ValueError("forward FLOP meter produced a nonpositive step estimate")
    return value


def training_schedule_state(
    *,
    train_flops: int,
    max_train_flops: int,
    examples_seen: int,
    max_train_samples: int,
    warmup_train_samples: int,
    base_lr: float,
    min_lr: float,
    warmup_fraction: float,
    freeze_last_layer_fraction: float,
    lr_key: str | None,
    reg_key: str | None,
) -> dict[str, float | bool]:
    """Compute the production schedule state from cumulative work counters."""

    def cosine(start: float, end: float, fraction: float) -> float:
        clipped = min(1.0, max(0.0, fraction))
        return end + 0.5 * (start - end) * (
            1.0 + math.cos(math.pi * clipped)
        )

    frac = min(1.0, train_flops / max_train_flops)
    sample_frac = min(1.0, examples_seen / max_train_samples)
    lr_frac = sample_frac if lr_key == "sample" else frac
    reg_frac = sample_frac if reg_key == "sample" else frac
    warmup = min(1.0, examples_seen / max(1, warmup_train_samples))
    if warmup < 1.0:
        lr = base_lr * warmup
    else:
        lr = cosine(
            base_lr,
            min_lr,
            (lr_frac - warmup_fraction) / max(1e-9, 1 - warmup_fraction),
        )
    wd = cosine(0.04, 0.2, reg_frac)
    teacher_temp = 0.04 + min(1.0, reg_frac / 0.2727) * (0.07 - 0.04)
    last_layer_frozen = frac < freeze_last_layer_fraction
    return {
        "frac": frac,
        "sample_frac": sample_frac,
        "lr_frac": lr_frac,
        "reg_frac": reg_frac,
        "warmup": warmup,
        "lr": lr,
        "wd": wd,
        "teacher_temp": teacher_temp,
        "last_layer_frozen": last_layer_frozen,
        "last_layer_lr": 0.0 if last_layer_frozen else lr,
        "kde_scale": min(1.0, max(0.0, (reg_frac - 0.1) / 0.4)),
    }


def gradient_dose_diagnostic(
    main_loss: torch.Tensor,
    fair_loss: torch.Tensor,
    params: Iterable[torch.nn.Parameter],
) -> dict[str, float | bool]:
    """Measure fairness/main gradient geometry without populating ``.grad``."""

    parameters = tuple(params)
    g_main = torch.autograd.grad(
        main_loss, parameters, retain_graph=True, allow_unused=True
    )
    g_fair = torch.autograd.grad(
        fair_loss, parameters, retain_graph=True, allow_unused=True
    )
    main_sq = main_loss.new_tensor(0.0, dtype=torch.float32)
    fair_sq = fair_loss.new_tensor(0.0, dtype=torch.float32)
    dot = main_loss.new_tensor(0.0, dtype=torch.float32)
    for gm, gf in zip(g_main, g_fair):
        if gm is not None:
            gm32 = gm.detach().float()
            main_sq = main_sq + gm32.square().sum()
        if gf is not None:
            gf32 = gf.detach().float()
            fair_sq = fair_sq + gf32.square().sum()
        if gm is not None and gf is not None:
            dot = dot + (gm.detach().float() * gf.detach().float()).sum()
    main_norm = main_sq.sqrt()
    fair_norm = fair_sq.sqrt()
    cosine = dot / (main_norm * fair_norm).clamp(min=1e-12)
    result: dict[str, float | bool] = {
        "dose_main_grad_norm": float(main_norm),
        "dose_fair_grad_norm": float(fair_norm),
        "dose_fair_main_grad_ratio": float(
            fair_norm / main_norm.clamp(min=1e-12)
        ),
        "dose_grad_cosine": float(cosine),
        "dose_grad_conflict": bool(float(cosine) < 0.0),
    }
    numeric = [value for value in result.values() if not isinstance(value, bool)]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("gradient-dose diagnostic produced a nonfinite value")
    return result
