"""Frozen constants and strict contracts for the fixed-48 diagnostic layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STUDY_ID = "matched_cancer_fixed48_20260730"
LEGACY_STUDY_ID = "matched_cancer_stage_20260730"
LEGACY_SCENARIO = "brca_luad_black_white_calibration_seed32001"
SEEDS = tuple(range(32001, 32049))
ARMS = ("B", "P", "H")
CANCERS = ("BRCA", "LUAD")
HEAD_SEEDS = (42001, 42002, 42003, 42004)
COHORT_SIZES = {"BRCA": 328, "LUAD": 281}
TASK_IDS = {"BRCA": "brca_tp53", "LUAD": "luad_tp53"}
CONTRACT_SCHEMA = "matched-cancer-fixed48-diagnostic-deployment/v1"
AUTHORIZATION_SCHEMA = "matched-cancer-fixed48-diagnostic-authorization/v1"
GATE_SCHEMA = "matched-cancer-fixed48-diagnostic-gate/v1"
COHORT_SCHEMA = "matched-cancer-fixed48-diagnostic-cohort/v1"
LOADER_ROOT_SCHEMA = "matched-cancer-fixed48-diagnostic-loader-root/v1"
EXPORT_SCHEMA = "matched-cancer-fixed48-diagnostic-export/v1"
COLLECTION_SCHEMA = "matched-cancer-fixed48-diagnostic-collection/v1"
AUDIT_SCHEMA = "matched-cancer-fixed48-diagnostic-structural-audit/v1"
PHASE_SCHEMA = "matched-cancer-fixed48-diagnostic-phase/v1"
CALIBRATION_ROOT_SCHEMA = (
    "matched-cancer-fixed48-calibration-root-completion/v1"
)
COMPLETION_SCHEMA = "matched-cancer-stage-completion/v1"
ROW_SCHEMA = "matched-cancer-diagnostic-prediction/v1"


def scenario_for(seed: int) -> str:
    validate_seed(seed)
    return f"brca_luad_black_white_calibration_seed{seed}"


def validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError("representation seed must be exactly one of 32001..32048")
    return seed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(), object_pairs_hook=_unique_object
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def cohort_contracts() -> dict[str, dict[str, Any]]:
    return {
        "BRCA": {
            "logical_name": "brca_racepanel",
            "expected_target_rows": 334,
            "expected_eligible_patients": 328,
            "expected_exclusions_by_race": {
                "Asian": 5,
                "American Indian or Alaska Native": 1,
            },
            "expected_race_counts": {"Black": 118, "White": 210},
            "task": "brca_tp53",
        },
        "LUAD": {
            "logical_name": "luad_target_hospitals",
            "expected_target_rows": 281,
            "expected_eligible_patients": 281,
            "expected_exclusions_by_race": {},
            "expected_race_counts": {"Black": 40, "White": 241},
            "task": "luad_tp53",
        },
    }


def build_contract(seed: int) -> dict[str, Any]:
    seed = validate_seed(seed)
    return {
        "schema": CONTRACT_SCHEMA,
        "study_id": STUDY_ID,
        "scenario": scenario_for(seed),
        "representation_seed": seed,
        "arms": list(ARMS),
        "cancers": list(CANCERS),
        "head_seeds": list(HEAD_SEEDS),
        "folds": list(range(5)),
        "prediction_schema": ROW_SCHEMA,
        "cohorts": cohort_contracts(),
    }


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    seed = validate_seed(value.get("representation_seed"))
    expected = build_contract(seed)
    if dict(value) != expected:
        raise ValueError("fixed48 diagnostic deployment contract differs")
    return expected


def load_contract(path: str | Path) -> dict[str, Any]:
    return validate_contract(load_json_object(path))
