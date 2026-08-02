"""Small, dataset-agnostic modules for post-hoc fairness interventions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class StageAdapter(nn.Module):
    """A configurable normalized adapter usable at either intervention stage."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        init_seed: int = 0,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("adapter dimensions must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.linear_in = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.GELU()
            self.linear_out = nn.Linear(hidden_dim, output_dim)
            self.normalization = nn.LayerNorm(output_dim)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        if representations.ndim != 2:
            raise ValueError("representations must be two-dimensional")
        hidden = self.activation(self.linear_in(representations))
        return F.normalize(
            self.normalization(self.linear_out(hidden)), dim=1, eps=1e-6
        )


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = float(weight)
        return value.view_as(value)

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.weight * gradient, None


def gradient_reverse(value: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Return ``value`` while reversing its backward gradient."""
    if not isinstance(weight, (int, float)):
        raise ValueError("gradient-reversal weight must be numeric")
    return _GradientReverse.apply(value, float(weight))


class PostHocOutput(NamedTuple):
    task_logits: torch.Tensor
    sensitive_logits: dict[str, torch.Tensor]


class PostHocDANNHead(nn.Module):
    """Task head with one gradient-reversal adversary per sensitive attribute."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sensitive_classes: Mapping[str, int],
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("head dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not sensitive_classes:
            raise ValueError("at least one sensitive attribute is required")
        invalid = {
            name: count
            for name, count in sensitive_classes.items()
            if not name or not isinstance(count, int) or count < 2
        }
        if invalid:
            raise ValueError(f"invalid sensitive class counts: {invalid}")

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.task_head = nn.Linear(hidden_dim, 1)
        self.adversaries = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, count),
                )
                for name, count in sensitive_classes.items()
            }
        )

    def forward(
        self, representations: torch.Tensor, *, reversal_weight: float = 1.0
    ) -> PostHocOutput:
        if representations.ndim != 2:
            raise ValueError("representations must be two-dimensional")
        hidden = self.trunk(representations)
        reversed_hidden = gradient_reverse(hidden, reversal_weight)
        return PostHocOutput(
            task_logits=self.task_head(hidden).squeeze(-1),
            sensitive_logits={
                name: adversary(reversed_hidden)
                for name, adversary in self.adversaries.items()
            },
        )


class DANNLoss(NamedTuple):
    total: torch.Tensor
    task: torch.Tensor
    sensitive: torch.Tensor


def dann_loss(
    output: PostHocOutput,
    task_targets: torch.Tensor,
    sensitive_targets: Mapping[str, torch.Tensor],
    *,
    sensitive_weight: float = 1.0,
) -> DANNLoss:
    """Compute task BCE plus sensitive-attribute cross entropy.

    Negative sensitive targets are treated as missing. Gradient reversal is
    applied by :class:`PostHocDANNHead`, not by this scalar loss.
    """
    targets = task_targets.flatten().to(output.task_logits)
    if targets.shape != output.task_logits.shape:
        raise ValueError("task targets and logits must have equal shapes")
    if not isinstance(sensitive_weight, (int, float)) or sensitive_weight < 0:
        raise ValueError("sensitive_weight must be non-negative")
    task = F.binary_cross_entropy_with_logits(output.task_logits, targets)
    sensitive = task * 0.0
    if set(sensitive_targets) != set(output.sensitive_logits):
        raise ValueError("sensitive target names must match adversary names")
    for name, logits in output.sensitive_logits.items():
        labels = sensitive_targets[name].flatten().to(device=logits.device)
        if labels.shape[0] != logits.shape[0]:
            raise ValueError(f"{name} targets and logits must have equal rows")
        known = labels >= 0
        if bool(known.any()):
            sensitive = sensitive + F.cross_entropy(
                logits[known], labels[known].long()
            )
    return DANNLoss(
        total=task + float(sensitive_weight) * sensitive,
        task=task,
        sensitive=sensitive,
    )
