"""Shared fairness objectives used at both intervention stages."""

from __future__ import annotations

import torch


def fair_supcon(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    *,
    relation: str = "different",
    condition: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervised contrastive loss with an explicit pair relation.

    The stage-only primary method uses ``relation="different"`` on race:
    different-race examples are positives and every non-self example remains
    in the denominator. Anchors without a valid positive are omitted.
    """
    if z.ndim != 2:
        raise ValueError(f"z must be two-dimensional, got {tuple(z.shape)}")
    if z.shape[0] < 2:
        return z.sum() * 0.0
    labels = labels.flatten()
    if labels.shape[0] != z.shape[0]:
        raise ValueError("labels and representations have different row counts")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = torch.nn.functional.normalize(z.float(), dim=1, eps=1e-6)
    count = z.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=z.device)
    similarity = (z @ z.t()) / float(temperature)
    similarity = similarity.masked_fill(self_mask, float("-inf"))
    log_probability = similarity - torch.logsumexp(
        similarity, dim=1, keepdim=True
    )
    if relation == "same":
        positives = labels[:, None] == labels[None, :]
    elif relation == "different":
        positives = labels[:, None] != labels[None, :]
    elif relation == "same-condition-different":
        if condition is None:
            raise ValueError("condition is required for the conditional relation")
        condition = condition.flatten()
        if condition.shape[0] != count:
            raise ValueError("condition and representations have different rows")
        positives = (
            (condition[:, None] == condition[None, :])
            & (labels[:, None] != labels[None, :])
        )
    else:
        raise ValueError(f"unknown pair relation: {relation!r}")
    positives = positives & ~self_mask
    positive_count = positives.sum(1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return z.sum() * 0.0
    per_anchor = -(
        log_probability.masked_fill(~positives, 0.0).sum(1)
        / positive_count.clamp(min=1)
    )
    if anchor_weights is None:
        return per_anchor[valid].mean()
    anchor_weights = anchor_weights.flatten().to(per_anchor)
    if anchor_weights.shape[0] != count:
        raise ValueError("anchor weights and representations have different rows")
    selected = anchor_weights[valid]
    return (per_anchor[valid] * selected).sum() / selected.sum().clamp(min=1e-12)
