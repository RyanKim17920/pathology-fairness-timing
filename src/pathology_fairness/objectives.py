"""Fairness objectives shared across pretraining and post-hoc interventions.

The functions in this module consume only representations, a preservation
condition, and a sensitive attribute. They intentionally have no downstream
diagnosis or outcome input.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
from torch.nn import functional as F


PairRelation = Literal["same", "different", "same_condition_different"]
FairnessObjective = Literal["all_pairs", "relation_consistent"]


class FairnessLoss(NamedTuple):
    """Condition-preservation and fairness loss components."""

    total: torch.Tensor
    condition: torch.Tensor
    fairness: torch.Tensor


def _validate_rows(
    representations: torch.Tensor,
    labels: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if representations.ndim != 2:
        raise ValueError("representations must be a two-dimensional tensor")
    flattened = labels.flatten()
    if flattened.shape[0] != representations.shape[0]:
        raise ValueError(f"{name} and representations must have equal row counts")
    return flattened


def _zero(representations: torch.Tensor) -> torch.Tensor:
    return representations.sum() * 0.0


def fair_supcon(
    representations: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
    *,
    relation: PairRelation = "different",
    condition: torch.Tensor | None = None,
    anchor_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervised contrastive loss with an explicit positive-pair relation.

    Every non-self row remains in the denominator. Anchors without a valid
    positive are omitted. ``same_condition_different`` treats rows with the
    same preservation condition and different sensitive labels as positives.
    """
    labels = _validate_rows(representations, labels, name="labels")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if representations.shape[0] < 2:
        return _zero(representations)

    normalized = F.normalize(representations.float(), dim=1, eps=1e-6)
    count = normalized.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=normalized.device)
    logits = (normalized @ normalized.T) / float(temperature)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    if relation == "same":
        positives = labels[:, None] == labels[None, :]
    elif relation == "different":
        positives = labels[:, None] != labels[None, :]
    elif relation == "same_condition_different":
        if condition is None:
            raise ValueError("condition is required for the conditional relation")
        condition = _validate_rows(representations, condition, name="condition")
        positives = (
            (condition[:, None] == condition[None, :])
            & (labels[:, None] != labels[None, :])
        )
    else:
        raise ValueError(f"unknown pair relation: {relation!r}")

    positives = positives & ~self_mask
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return _zero(representations)
    per_anchor = -(
        log_probability.masked_fill(~positives, 0.0).sum(dim=1)
        / positive_count.clamp(min=1)
    )
    if anchor_weights is None:
        return per_anchor[valid].mean()

    weights = _validate_rows(representations, anchor_weights, name="anchor_weights")
    weights = weights.to(device=per_anchor.device, dtype=per_anchor.dtype)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("anchor_weights must be finite and non-negative")
    selected = weights[valid]
    if not bool((selected > 0).any()):
        return _zero(representations)
    return (per_anchor[valid] * selected).sum() / selected.sum()


def relation_consistent_mask(
    representations: torch.Tensor,
    condition: torch.Tensor,
    sensitive: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Contrast matched positive and negative pair relations.

    Positives share the preservation condition and differ in the sensitive
    attribute. Negatives differ in the condition and share the sensitive
    attribute. All other relations are excluded from the denominator. Negative
    label values are treated as missing and removed before pairs are formed.
    """
    condition = _validate_rows(representations, condition, name="condition")
    sensitive = _validate_rows(representations, sensitive, name="sensitive")
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be positive")

    known = (condition >= 0) & (sensitive >= 0)
    selected = representations[known]
    condition = condition[known]
    sensitive = sensitive[known]
    if selected.shape[0] < 2:
        return _zero(representations)

    normalized = F.normalize(selected.float(), dim=1, eps=1e-6)
    count = normalized.shape[0]
    self_mask = torch.eye(count, dtype=torch.bool, device=normalized.device)
    same_condition = condition[:, None] == condition[None, :]
    same_sensitive = sensitive[:, None] == sensitive[None, :]
    positives = same_condition & ~same_sensitive & ~self_mask
    negatives = ~same_condition & same_sensitive & ~self_mask
    allowed = positives | negatives

    positive_count = positives.sum(dim=1)
    negative_count = negatives.sum(dim=1)
    valid = (positive_count > 0) & (negative_count > 0)
    if not bool(valid.any()):
        return _zero(representations)

    logits = (normalized @ normalized.T) / float(temperature)
    allowed_logits = logits.masked_fill(~allowed, float("-inf"))
    log_probability = logits - torch.logsumexp(
        allowed_logits, dim=1, keepdim=True
    )
    per_anchor = -(
        log_probability.masked_fill(~positives, 0.0).sum(dim=1)
        / positive_count.clamp(min=1)
    )
    return per_anchor[valid].mean()


def condition_preserving_fairness_loss(
    representations: torch.Tensor,
    condition: torch.Tensor,
    sensitive: torch.Tensor,
    *,
    fairness_weight: float = 0.1,
    temperature: float = 0.2,
    objective: FairnessObjective = "all_pairs",
) -> FairnessLoss:
    """Apply a matched condition-preservation and fairness objective."""
    condition = _validate_rows(representations, condition, name="condition")
    sensitive = _validate_rows(representations, sensitive, name="sensitive")
    if not isinstance(fairness_weight, (int, float)):
        raise ValueError("fairness_weight must be numeric")
    weight = float(fairness_weight)
    if not torch.isfinite(torch.tensor(weight)) or weight < 0:
        raise ValueError("fairness_weight must be finite and non-negative")

    known_condition = condition >= 0
    preservation = fair_supcon(
        representations[known_condition],
        condition[known_condition],
        temperature,
        relation="same",
    )
    if objective == "all_pairs":
        known = known_condition & (sensitive >= 0)
        fairness = fair_supcon(
            representations[known],
            sensitive[known],
            temperature,
            relation="same_condition_different",
            condition=condition[known],
        )
    elif objective == "relation_consistent":
        fairness = relation_consistent_mask(
            representations, condition, sensitive, temperature
        )
    else:
        raise ValueError(f"unknown fairness objective: {objective!r}")
    return FairnessLoss(
        total=preservation + weight * fairness,
        condition=preservation,
        fairness=fairness,
    )
