#!/usr/bin/env python3
"""Fail-closed contract for the frozen fixed-five representation audit.

This module contains no model or metric code.  It is the population and
topology boundary shared by the later extractor, analyzer, and verifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


REPO = Path("/admin/home/ryan.kim/nt")
LOCK_PATH = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_REPRESENTATION_AUDIT_LOCK.md"
)
NUMERIC_AMENDMENT_PATH = (
    LOCK_PATH.parent / "FIXED5_REPRESENTATION_AUDIT_NUMERIC_AMENDMENT_01.md"
)
SCHEMA = "matched-cancer-fixed5-representation-audit/v1"
ANALYSIS_SCHEMA = "matched-cancer-fixed5-representation-analysis/v1"
STUDY_ID = "matched_cancer_representation_audit_20260801"
SOLVER_SEED = 288_850_999

FM_SEEDS = (32001, 32002, 32003, 32004, 32005)
ACCEPTED_ATTEMPTS = MappingProxyType(
    {
        32001: "attempt_03",
        32002: "attempt_01",
        32003: "attempt_01",
        32004: "attempt_01",
        32005: "attempt_01",
    }
)

CALIBRATION_NAMESPACE = Path(
    "/data/ryan.kim/nanopath/reruns/"
    "matched_cancer_fixed48_20260730/calibration"
)
METADATA_PATHS = MappingProxyType(
    {
        "BRCA": REPO / "data/metadata/brca_racepanel_folds.csv",
        "LUAD": REPO / "data/metadata/luad_hospital_folds.csv",
    }
)
REPRESENTATION_EXCLUSION_PATH = (
    REPO
    / "configs_vendor/matched_stage_union_20260730/"
    "exclude_union_target_hospitals.txt"
)
EXPECTED_EXCLUSION_COUNT = 979
RUNS = ("slot1_plain", "slot1_fair", "B", "P", "H")
CHECKPOINT_RELATIVE_PATHS = MappingProxyType(
    {run: Path(run) / "latest.pt" for run in RUNS}
)
COMPLETION_RECEIPT_RELATIVE_PATHS = MappingProxyType(
    {run: Path(run) / "COMPLETION_RECEIPT.json" for run in RUNS}
)
REPLAY_MANIFEST_NAME = "CALIBRATION_REPLAY_MANIFEST.json"

# Equal-dimensional comparisons are enforced by family membership.
LAYER_DIMENSIONS = MappingProxyType(
    {
        "E_plain": 384,
        "E_fair": 384,
        "A_temp_plain": 128,
        "A_temp_fair": 128,
        "B": 128,
        "P": 128,
        "H": 128,
    }
)
LAYER_FAMILIES = MappingProxyType(
    {
        384: ("E_plain", "E_fair"),
        128: ("A_temp_plain", "A_temp_fair", "B", "P", "H"),
    }
)
REPRESENTATION_NORMALIZATION = "per_tile_l2"
PATIENT_POOLING = "arithmetic_mean_16_no_renormalization"
ZERO_OR_NONFINITE_NORM = "fail_closed"
LAYER_NORMALIZATION = MappingProxyType(
    {layer: REPRESENTATION_NORMALIZATION for layer in LAYER_DIMENSIONS}
)
LAYER_CHECKPOINT_RUN = MappingProxyType(
    {
        "E_plain": "slot1_plain",
        "A_temp_plain": "slot1_plain",
        "E_fair": "slot1_fair",
        "A_temp_fair": "slot1_fair",
        "B": "B",
        "P": "P",
        "H": "H",
    }
)
GATE_ELIGIBLE_CONTRASTS = (
    ("E_fair", "E_plain"),
    ("A_temp_fair", "A_temp_plain"),
    ("P", "B"),
    ("H", "B"),
)
DESCRIPTIVE_CONTRASTS = (("P", "A_temp_fair"),)
EXPECTED_ENCODER_PARENTS = MappingProxyType(
    {"B": "E_plain", "H": "E_plain", "P": "E_fair"}
)

CANCERS = ("BRCA", "LUAD")
RACES = ("Black", "White")
TILE_VIEWS = ("A", "B")
PROBE_LEVELS = ("patient", "tile")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TILES_PER_PATIENT = 32
TILES_PER_VIEW = 16
TILE_HASH_PREFIX = "rep-audit/v1"
CANCER_FOLD_HASH_PREFIX = "rep-audit-cancer-fold/v1"
CANCER_PROBE_OUTER_FOLDS = 5
KNN_K = 5

METADATA_ALLOWLIST = frozenset({"patient_id", "cancer", "race", "tss"})
DIAGNOSIS_FIELD_DENYLIST = frozenset(
    {"tp53", "tp53_status", "diagnosis", "outcome", "target", "y_true"}
)
# Stable analyzer-facing names for the same frozen declarations.
ALLOWED_METADATA_FIELDS = METADATA_ALLOWLIST
FORBIDDEN_DIAGNOSIS_FIELDS = DIAGNOSIS_FIELD_DENYLIST
PROBE_SEED = SOLVER_SEED
VIEWS = TILE_VIEWS
LEVELS = PROBE_LEVELS
LEAKAGE_DELTA = 0.05
CANCER_LOSS_MAX = 0.02
EXPECTED_POPULATION = MappingProxyType(
    {
        "BRCA": MappingProxyType(
            {
                "patients": 328,
                "tss": frozenset({"A2", "A7", "AC", "B6", "EW", "LL", "OL", "S3"}),
                "race": MappingProxyType({"Black": 118, "White": 210}),
            }
        ),
        "LUAD": MappingProxyType(
            {
                "patients": 281,
                "tss": frozenset({"38", "44", "49", "50", "55", "67", "69", "73", "86", "95", "NJ"}),
                "race": MappingProxyType({"Black": 40, "White": 241}),
            }
        ),
    }
)
EXPECTED_UNION_PATIENTS = 609
EXPECTED_UNION_TSS = 19

PRIMARY_GATE = MappingProxyType(
    {
        "gate_eligible_contrasts": GATE_ELIGIBLE_CONTRASTS,
        "descriptive_only_contrasts": DESCRIPTIVE_CONTRASTS,
        "leakage_reduction": "L_baseline-L_candidate",
        "race_minimum_reduction": 0.05,
        "race_seed_comparator": ">=",
        "race_minimum_passing_seeds_per_cell": 4,
        "race_total_seeds_per_cell": 5,
        "race_median_comparator": ">=",
        "race_median_minimum": 0.05,
        "race_required_cancers": CANCERS,
        "race_required_views": TILE_VIEWS,
        "race_required_probe_levels": PROBE_LEVELS,
        "cancer_information_loss": "AUROC_baseline-AUROC_candidate",
        "cancer_maximum_loss": 0.02,
        "cancer_seed_comparator": "<=",
        "cancer_minimum_passing_seeds_per_view": 4,
        "cancer_total_seeds_per_view": 5,
        "cancer_median_comparator": "<=",
        "cancer_median_maximum": 0.02,
        "cancer_required_views": TILE_VIEWS,
        "complete_gate": "race_leakage_activity_and_cancer_information_preservation",
        "missing_cells_fail": True,
        "nonfinite_values_fail": True,
        "post_value_drops_allowed": False,
    }
)
PROBE_CONTRACT = MappingProxyType(
    {
        "outer_split": "leave_one_tss_out",
        "inner_split": "leave_one_training_tss_out",
        "standardization_scope": "outer_training_tss_only",
        "c_selection_scope": "outer_training_tss_only",
        "c_tie_break": "smallest",
        "minimum_valid_inner_folds": 2,
        "exclude_inner_folds_without_both_races": True,
        "count_excluded_inner_folds": True,
        "outer_tss_evaluated_exactly_once": True,
        "score_scope": "pooled_held_out_patient_predictions",
        "tile_training_weight": 1.0 / TILES_PER_VIEW,
        "tile_evaluation_aggregation": "mean_probability_to_patient",
        "tile_outer_holdout": "entire_tss",
        "leakage_orientation": "max(auroc,1-auroc)-0.5",
        "cancer_probe_outer_folds": CANCER_PROBE_OUTER_FOLDS,
        "cancer_probe_grouping": "whole_tss_grouped_with_both_cancers",
        "cancer_probe_inner_folds": "other_four_grouped_fold_labels",
    }
)
GEOMETRY_CONTRACT = MappingProxyType(
    {
        "knn_k": KNN_K,
        "knn_distance": "cosine",
        "patient_knn_candidates": "other_patients_same_cancer",
        "tile_knn_candidates": "tiles_same_cancer_excluding_query_patient",
        "knn_aggregation": "query_mean_then_race_mean_then_equal_race_mean",
        "energy_distance": "euclidean_unbiased_u_statistic_distinct_ordered_pairs",
        "aligned_displacement": ("mean_euclidean", "mean_cosine"),
        "parameter_displacement": ("absolute_frobenius", "baseline_frobenius_ratio"),
        "parameter_displacement_gate_eligible": False,
        "equal_dimensions_only": True,
    }
)
CONTINUATION_LIMITS = MappingProxyType(
    {
        "targeted_mechanism_fm_seeds": 1,
        "unseen_scenario_fm_seeds_after_pass": 3,
        "maximum_new_fm_seeds_total": 4,
        "fifth_allowed_seed": "unused",
        "gpus_at_once": 1,
        "job_name": "main_1gpu",
    }
)


def attempt_root(seed: int) -> Path:
    """Return the one accepted production attempt for a frozen FM seed."""
    if type(seed) is not int or seed not in ACCEPTED_ATTEMPTS:
        raise ValueError("FM seed must be exactly one of 32001..32005")
    return CALIBRATION_NAMESPACE / f"seed_{seed}" / ACCEPTED_ATTEMPTS[seed]


def production_paths(seed: int) -> dict[str, Any]:
    """Materialize checkpoint, receipt, and replay paths for one seed."""
    root = attempt_root(seed)
    return {
        "root": root,
        "replay_manifest": root / REPLAY_MANIFEST_NAME,
        "checkpoints": {
            layer: root / CHECKPOINT_RELATIVE_PATHS[run]
            for layer, run in LAYER_CHECKPOINT_RUN.items()
        },
        "completion_receipts": {
            run: root / path
            for run, path in COMPLETION_RECEIPT_RELATIVE_PATHS.items()
        },
    }


def assert_diagnosis_free_fields(fields: Iterable[str]) -> None:
    """Reject diagnosis/outcome-bearing schemas at every downstream boundary."""
    normalized = {str(field).strip().lower() for field in fields}
    denied: set[str] = set()
    for field in normalized:
        tokens = {token for token in re.split(r"[^a-z0-9]+", field) if token}
        if field in DIAGNOSIS_FIELD_DENYLIST or tokens & DIAGNOSIS_FIELD_DENYLIST:
            denied.add(field)
    if denied:
        raise ValueError(f"diagnosis/outcome fields are forbidden: {sorted(denied)}")


def validate_metadata_records(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require the exact downstream metadata allowlist for every row."""
    for index, row in enumerate(rows):
        keys = set(row)
        assert_diagnosis_free_fields(keys)
        if keys != METADATA_ALLOWLIST:
            raise ValueError(
                f"sanitized row {index} keys differ: "
                f"missing={sorted(METADATA_ALLOWLIST - keys)}, "
                f"unexpected={sorted(keys - METADATA_ALLOWLIST)}"
            )


def _normalize_race(raw: str) -> str | None:
    value = raw.strip().lower()
    if value in {"black", "black or african american"}:
        return "Black"
    if value == "white":
        return "White"
    return None


def _read_target_rows(cancer: str, path: Path) -> list[dict[str, str]]:
    """Read a source CSV internally and emit no source-only fields."""
    if cancer not in CANCERS:
        raise ValueError(f"unexpected cancer {cancer!r}")
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {"patient_barcode", "tss", "race", "fold"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"metadata schema missing required fields: {path}")
        rows: list[dict[str, str]] = []
        for source_row in reader:
            if source_row["fold"].strip().lower() != "target":
                continue
            race = _normalize_race(source_row["race"])
            if race is None:
                continue
            patient_id = source_row["patient_barcode"].strip().upper()
            tss = source_row["tss"].strip().upper()
            if not patient_id or not tss:
                raise ValueError(f"blank patient_id/TSS in {path}")
            rows.append(
                {
                    "patient_id": patient_id,
                    "cancer": cancer,
                    "race": race,
                    "tss": tss,
                }
            )
    return rows


def validate_population_counts(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate exact patient, race, and non-overlapping TSS populations."""
    validate_metadata_records(rows)
    identities = [(row["cancer"], row["patient_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate cancer/patient identity in sanitized population")
    if len(rows) != EXPECTED_UNION_PATIENTS:
        raise ValueError(f"population must contain exactly {EXPECTED_UNION_PATIENTS} patients")
    tss_by_cancer: dict[str, set[str]] = {}
    for cancer in CANCERS:
        subset = [row for row in rows if row["cancer"] == cancer]
        expected = EXPECTED_POPULATION[cancer]
        if len(subset) != expected["patients"]:
            raise ValueError(f"{cancer} patient count drift")
        if Counter(row["race"] for row in subset) != Counter(expected["race"]):
            raise ValueError(f"{cancer} race counts drift")
        observed_tss = {str(row["tss"]) for row in subset}
        if observed_tss != expected["tss"]:
            raise ValueError(f"{cancer} TSS membership drift")
        tss_by_cancer[cancer] = observed_tss
    if tss_by_cancer["BRCA"] & tss_by_cancer["LUAD"]:
        raise ValueError("BRCA and LUAD TSS must not overlap")
    if len(tss_by_cancer["BRCA"] | tss_by_cancer["LUAD"]) != EXPECTED_UNION_TSS:
        raise ValueError("union must contain exactly 19 non-overlapping TSS")


def load_sanitized_population(
    metadata_paths: Mapping[str, Path] = METADATA_PATHS,
) -> tuple[dict[str, str], ...]:
    """Load the exact target Black/White cohort and erase diagnosis fields."""
    if set(metadata_paths) != set(CANCERS):
        raise ValueError("metadata paths must be exactly BRCA and LUAD")
    rows: list[dict[str, str]] = []
    for cancer in CANCERS:
        rows.extend(_read_target_rows(cancer, Path(metadata_paths[cancer])))
    rows.sort(key=lambda row: (row["cancer"], row["patient_id"]))
    validate_population_counts(rows)
    return tuple(rows)


def _read_exclusion_members(path: Path) -> frozenset[str]:
    members = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
    if len(members) != EXPECTED_EXCLUSION_COUNT or len(set(members)) != EXPECTED_EXCLUSION_COUNT:
        raise ValueError("representation exclusion list must contain 979 unique patients")
    return frozenset(members)


def validate_exclusion_membership(
    rows: Sequence[Mapping[str, Any]],
    exclusion_path: Path = REPRESENTATION_EXCLUSION_PATH,
) -> None:
    """Fail unless every audit patient belongs to the frozen exclusion list."""
    validate_metadata_records(rows)
    excluded = _read_exclusion_members(Path(exclusion_path))
    missing = sorted({str(row["patient_id"]).upper() for row in rows} - excluded)
    if missing:
        raise ValueError(f"audit patients missing from frozen exclusion list: {missing[:5]}")


def replay_manifest_paths() -> tuple[Path, ...]:
    return tuple(production_paths(seed)["replay_manifest"] for seed in FM_SEEDS)


def validate_replay_nonoverlap(
    rows: Sequence[Mapping[str, Any]],
    manifests: Sequence[Path] | None = None,
) -> None:
    """Fail if an audit patient occurs in any accepted training replay."""
    validate_metadata_records(rows)
    paths = replay_manifest_paths() if manifests is None else tuple(map(Path, manifests))
    if len(paths) != len(FM_SEEDS):
        raise ValueError("exactly five accepted replay manifests are required")
    audit_patients = {str(row["patient_id"]).upper() for row in rows}
    for seed, path in zip(FM_SEEDS, paths, strict=True):
        expected = production_paths(seed)["replay_manifest"].resolve()
        if manifests is None and path.resolve() != expected:
            raise ValueError(f"replay ancestry mismatch for seed {seed}")
        value = json.loads(path.read_text())
        if value.get("schema") != "matched-cancer-replay-manifest/v1":
            raise ValueError(f"replay schema drift for seed {seed}")
        occurrences = value.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError(f"missing replay occurrences for seed {seed}")
        replay_patients: set[str] = set()
        for item in occurrences:
            if not isinstance(item, Mapping) or "patient" not in item:
                raise ValueError(f"malformed replay occurrence for seed {seed}")
            patient = str(item["patient"]).strip().upper()
            if not patient:
                raise ValueError(f"malformed replay occurrence for seed {seed}")
            replay_patients.add(patient)
        overlap = sorted(audit_patients & replay_patients)
        if overlap:
            raise ValueError(f"audit/replay overlap for seed {seed}: {overlap[:5]}")


def validate_frozen_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclusion_path: Path = REPRESENTATION_EXCLUSION_PATH,
    manifests: Sequence[Path] | None = None,
) -> None:
    """Run all population, exclusion, and replay membership gates."""
    validate_population_counts(rows)
    validate_exclusion_membership(rows, exclusion_path)
    validate_replay_nonoverlap(rows, manifests)


def validate_encoder_state_sharing(state_hashes: Mapping[str, str]) -> None:
    """Hook for the extractor to enforce the frozen encoder ancestry graph."""
    expected_keys = {"E_plain", "E_fair", "B", "P", "H"}
    if set(state_hashes) != expected_keys:
        raise ValueError("encoder-state hash keys differ from frozen topology")
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in state_hashes.values()):
        raise ValueError("encoder-state hashes must be lowercase SHA-256 digests")
    if not (state_hashes["B"] == state_hashes["H"] == state_hashes["E_plain"]):
        raise ValueError("final B/H must share the exact plain encoder state")
    if state_hashes["P"] != state_hashes["E_fair"]:
        raise ValueError("final P must share the exact fair encoder state")


def validate_equal_dimension(left: str, right: str) -> None:
    if left not in LAYER_DIMENSIONS or right not in LAYER_DIMENSIONS:
        raise ValueError("unknown representation layer")
    if LAYER_DIMENSIONS[left] != LAYER_DIMENSIONS[right]:
        raise ValueError("only equal-dimensional layers may be contrasted")


def validate_representation_normalization(layer: str, normalization: str) -> None:
    """Require per-tile L2 normalization before any cache or metric boundary."""
    if layer not in LAYER_NORMALIZATION:
        raise ValueError("unknown representation layer")
    if normalization != LAYER_NORMALIZATION[layer]:
        raise ValueError(f"{layer} representations must use per_tile_l2 normalization")


def normalize_representation_row(
    layer: str, values: Sequence[float]
) -> tuple[float, ...]:
    """Apply the frozen per-tile L2 operation and fail on invalid norms."""
    if layer not in LAYER_DIMENSIONS:
        raise ValueError("unknown representation layer")
    if len(values) != LAYER_DIMENSIONS[layer]:
        raise ValueError(f"{layer} row dimensionality drift")
    row = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in row):
        raise ValueError("representation row contains non-finite values")
    norm = math.sqrt(math.fsum(value * value for value in row))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("representation row has zero or non-finite norm")
    normalized = tuple(value / norm for value in row)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("normalized representation row is non-finite")
    return normalized


def tile_rank_digest(patient_id: str, payload_sha256: str, occurrence_index: int) -> str:
    """Compute the exact frozen tile-ranking digest."""
    if re.fullmatch(r"[0-9a-f]{64}", payload_sha256) is None:
        raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
    if type(occurrence_index) is not int or occurrence_index < 0:
        raise ValueError("occurrence_index must be a non-negative integer")
    value = (
        f"{TILE_HASH_PREFIX}|{SOLVER_SEED}|{patient_id}|"
        f"{payload_sha256}|{occurrence_index}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cancer_fold_digest(cancer: str, tss: str) -> str:
    """Compute the deterministic grouped cancer-probe TSS rank."""
    if cancer not in CANCERS or not tss:
        raise ValueError("cancer fold requires a frozen cancer and non-empty TSS")
    value = f"{CANCER_FOLD_HASH_PREFIX}|{SOLVER_SEED}|{cancer}|{tss}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def grouped_cancer_probe_folds(
    tss_by_cancer: Mapping[str, Iterable[str]],
) -> dict[tuple[str, str], int]:
    """Rank TSS within cancer and allocate them round-robin to five folds."""
    if set(tss_by_cancer) != set(CANCERS):
        raise ValueError("grouped cancer folds require exactly BRCA and LUAD")
    result: dict[tuple[str, str], int] = {}
    for cancer in CANCERS:
        observed = {str(tss).strip().upper() for tss in tss_by_cancer[cancer]}
        if observed != EXPECTED_POPULATION[cancer]["tss"]:
            raise ValueError(f"{cancer} grouped-fold TSS membership drift")
        ranked = sorted(observed, key=lambda tss: (cancer_fold_digest(cancer, tss), tss))
        for rank, tss in enumerate(ranked):
            result[(cancer, tss)] = rank % CANCER_PROBE_OUTER_FOLDS
    for fold in range(CANCER_PROBE_OUTER_FOLDS):
        cancers = {cancer for (cancer, _), assigned in result.items() if assigned == fold}
        if cancers != set(CANCERS):
            raise ValueError(f"grouped cancer fold {fold} does not contain both cancers")
    return result


def select_tile_views(
    patient_id: str,
    occurrences: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Rank valid occurrences once and form disjoint even/odd 16-tile views."""
    ranked: list[tuple[str, str, int, dict[str, Any]]] = []
    for position, item in enumerate(occurrences):
        if set(item) != {"payload_sha256", "occurrence_index", "keep_mask"}:
            raise ValueError(
                "tile occurrence keys must be payload_sha256/occurrence_index/keep_mask"
            )
        payload = str(item["payload_sha256"])
        index = item["occurrence_index"]
        if type(index) is not int or index != position:
            raise ValueError(
                "occurrence_index must be the zero-based full cache/input position"
            )
        if type(item["keep_mask"]) is not bool:
            raise ValueError("keep_mask must be boolean")
        if not item["keep_mask"]:
            continue
        digest = tile_rank_digest(patient_id, payload, index)
        ranked.append(
            (
                digest,
                payload,
                index,
                {"payload_sha256": payload, "occurrence_index": index},
            )
        )
    if len(ranked) < TILES_PER_PATIENT:
        raise ValueError("every patient requires at least 32 valid tile occurrences")
    identities = [(payload, index) for _, payload, index, _ in ranked]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate tile occurrence identity")
    kept = sorted(ranked, key=lambda value: value[:3])[:TILES_PER_PATIENT]
    views = {
        "A": tuple(value[3] for rank, value in enumerate(kept) if rank % 2 == 0),
        "B": tuple(value[3] for rank, value in enumerate(kept) if rank % 2 == 1),
    }
    if any(len(view) != TILES_PER_VIEW for view in views.values()):
        raise AssertionError("internal tile-view cardinality error")
    return views


def validate_selected_payload_reads(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    payloads_read: Mapping[int, str],
) -> None:
    """Fail if frozen inference read a payload other than the ranked payload."""
    if set(selected) != set(TILE_VIEWS):
        raise ValueError("selected payload views must be exactly A and B")
    if any(len(selected[view]) != TILES_PER_VIEW for view in TILE_VIEWS):
        raise ValueError("selected payload views must contain exactly 16 tiles each")
    expected_indices: set[int] = set()
    for view in TILE_VIEWS:
        for item in selected[view]:
            index = item["occurrence_index"]
            expected_indices.add(index)
            if payloads_read.get(index) != item["payload_sha256"]:
                raise ValueError("selected payload differs from frozen-inference payload")
    if set(payloads_read) != expected_indices:
        raise ValueError("frozen-inference payload index set differs from selected tiles")


def validate_shared_tile_views(
    views_by_layer_seed: Mapping[tuple[int, str], Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    """Require identical tile identities/views across every layer and seed."""
    if not views_by_layer_seed:
        raise ValueError("tile-view mapping may not be empty")
    expected: dict[str, tuple[tuple[str, int], ...]] | None = None
    for (seed, layer), views in views_by_layer_seed.items():
        if seed not in FM_SEEDS or layer not in LAYER_DIMENSIONS or set(views) != set(TILE_VIEWS):
            raise ValueError("tile-view seed/layer topology drift")
        current = {
            view: tuple((str(item["payload_sha256"]), int(item["occurrence_index"])) for item in views[view])
            for view in TILE_VIEWS
        }
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError("tile identities/view assignment differ across layer or seed")
