"""Shared cancer-only representation objective for matched-stage studies."""

from .objectives import (
    BLACK_RACE_ID,
    WHITE_RACE_ID,
    CancerStageLoss,
    StageAdapter,
    cancer_stage_loss,
)

__all__ = [
    "BLACK_RACE_ID",
    "WHITE_RACE_ID",
    "CancerStageLoss",
    "StageAdapter",
    "cancer_stage_loss",
]
