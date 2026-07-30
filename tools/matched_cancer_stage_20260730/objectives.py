"""Canonical representation adapter and cancer-only matched-stage objective.

This module deliberately has no downstream diagnosis input. Both intervention
stages receive the same representation, cancer identity, and race identity, so
neither stage can condition its fairness objective on TP53 or another diagnosis.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from tools.matched_stage_union_20260730.objectives import fair_supcon


INPUT_DIM = 384
HIDDEN_DIM = 256
OUTPUT_DIM = 128
TEMPERATURE = 0.2
BLACK_RACE_ID = 2
WHITE_RACE_ID = 4
DEFAULT_ADAPTER_INIT_SEED = 20_260_730


class StageAdapter(nn.Module):
    """The stage-shared ``384 -> 256 -> 128`` representation adapter.

    Initialization uses a fixed, adapter-specific RNG scope. It is reproducible
    for a given ``init_seed`` and does not consume or replace the caller's
    process-wide PyTorch RNG state.
    """

    def __init__(self, *, init_seed: int = DEFAULT_ADAPTER_INIT_SEED) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.lin1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
            self.act1 = nn.GELU()
            self.lin2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            self.norm2 = nn.LayerNorm(OUTPUT_DIM)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or h.shape[1] != INPUT_DIM:
            raise ValueError(
                f"h must have shape [batch, {INPUT_DIM}], got {tuple(h.shape)}"
            )
        return F.normalize(
            self.norm2(self.lin2(self.act1(self.lin1(h)))),
            dim=1,
            eps=1e-6,
        )


class CancerStageLoss(NamedTuple):
    """Loss components returned identically to both intervention stages."""

    total: torch.Tensor
    cancer: torch.Tensor
    fair: torch.Tensor


def cancer_stage_loss(
    h: torch.Tensor,
    cancer_id: torch.Tensor,
    race_id: torch.Tensor,
    *,
    fair_weight: float = 0.1,
) -> CancerStageLoss:
    """Compute cancer preservation plus cancer-conditioned race alignment.

    ``cancer`` is a same-cancer supervised contrastive loss. ``fair`` treats a
    Black/White pair as positive only when both examples have the same cancer.
    Missing cancer labels (negative IDs) are excluded from both objectives.
    Races other than the canonical Black/White IDs (2/4), including missing
    race, are excluded only from the fairness objective.
    """
    if h.ndim != 2:
        raise ValueError(f"h must be two-dimensional, got {tuple(h.shape)}")
    cancer_id = cancer_id.flatten()
    race_id = race_id.flatten()
    if cancer_id.shape[0] != h.shape[0] or race_id.shape[0] != h.shape[0]:
        raise ValueError("h, cancer_id, and race_id must have equal row counts")
    if not isinstance(fair_weight, (int, float)) or not torch.isfinite(
        torch.tensor(float(fair_weight))
    ):
        raise ValueError("fair_weight must be finite")
    if float(fair_weight) < 0:
        raise ValueError("fair_weight must be non-negative")

    known_cancer = cancer_id >= 0
    cancer = fair_supcon(
        h[known_cancer],
        cancer_id[known_cancer],
        TEMPERATURE,
        relation="same",
    )

    target_race = (race_id == BLACK_RACE_ID) | (race_id == WHITE_RACE_ID)
    fair_rows = known_cancer & target_race
    fair = fair_supcon(
        h[fair_rows],
        race_id[fair_rows],
        TEMPERATURE,
        relation="same-condition-different",
        condition=cancer_id[fair_rows],
    )
    total = cancer + float(fair_weight) * fair
    return CancerStageLoss(total=total, cancer=cancer, fair=fair)
