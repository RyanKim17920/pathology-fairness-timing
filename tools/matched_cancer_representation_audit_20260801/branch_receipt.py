#!/usr/bin/env python3
"""Source-bound branch and H-mask feasibility receipt for the fixed-five audit.

This module is intentionally separate from the historical audit analyzer and
independent verifier.  It reads the diagnosis-free ``metric_input.json``
directly and applies only the value-blind branch frozen by the objective-mask
erratum at commit c94f688.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

from . import contract


SCHEMA = "matched-cancer-fixed5-representation-branch-receipt/v1"
METRIC_INPUT_SCHEMA = "matched-cancer-fixed5-representation-metric-input/v1"
STUDY_ID = contract.STUDY_ID
FM_SEEDS = tuple(contract.FM_SEEDS)
LAYERS = tuple(contract.LAYER_DIMENSIONS)
CANCERS = tuple(contract.CANCERS)
VIEWS = tuple(contract.TILE_VIEWS)
LEVELS = tuple(contract.PROBE_LEVELS)

ERRATUM_COMMIT = "c94f688"
ERRATUM_SHA256 = "3fe98b877e86d6a6bea3e747d5708c503969e36a6fedb2f771786eb2458b96e5"
NUMERIC_AMENDMENT_SHA256 = "91d3b56bbf87f97faf14bbcb31689d0cdb750befed184b92073d31c2052f7f5b"
ERRATUM_PATH = (
    contract.REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_OBJECTIVE_MASK_ERRATUM.md"
)
NUMERIC_AMENDMENT_PATH = (
    contract.REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_NUMERIC_AMENDMENT_07.md"
)

MINIMUM_REDUCTION = 0.05
MAXIMUM_CANCER_LOSS = 0.02
MINIMUM_PASSING_SEEDS = 4
MINIMUM_H_MEDIAN = 0.03
ABS_TOLERANCE = 1e-12

TOP_LEVEL_KEYS = {
    "schema", "study_id", "status", "diagnosis_free", "fm_seeds",
    "layers", "views", "probe_levels", "compact_caches",
    "preflight_receipt", "race_probes", "cancer_probes",
    "secondary_geometry", "training_evidence",
}
RACE_ROW_KEYS = {"fm_seed", "layer", "cancer", "view", "probe_level", "result"}
CANCER_ROW_KEYS = {"fm_seed", "layer", "view", "result"}
IDENTITY_KEYS = {"canonical_path", "bytes", "sha256"}
CONTRASTS = {
    "E": ("E_fair", "E_plain"),
    "A_temp": ("A_temp_fair", "A_temp_plain"),
    "P": ("P", "B"),
    "H": ("H", "B"),
}


class BranchReceiptError(RuntimeError):
    """A frozen identity, schema, topology, or numeric requirement failed."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BranchReceiptError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise BranchReceiptError(f"non-finite JSON constant {value!r}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except BranchReceiptError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BranchReceiptError(f"cannot load strict JSON {path}: {error}") from error


def file_identity(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise BranchReceiptError(f"identity source may not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise BranchReceiptError(f"identity source is unavailable: {candidate}") from error
    if not resolved.is_file():
        raise BranchReceiptError(f"identity source is not a regular file: {resolved}")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"canonical_path": str(resolved), "bytes": size, "sha256": digest.hexdigest()}


def _exact_object(value: Any, keys: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BranchReceiptError(f"{context} must be an object")
    if set(value) != keys:
        raise BranchReceiptError(
            f"{context} keys differ: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _finite(value: Any, context: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BranchReceiptError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < low or number > high:
        raise BranchReceiptError(f"{context} must be finite and in [{low}, {high}]")
    return number


def _canonical_field(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def reject_diagnosis_fields(value: Any, *, context: str = "metric_input") -> None:
    forbidden = tuple(contract.DIAGNOSIS_FIELD_DENYLIST)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _canonical_field(key)
            if (
                (normalized in forbidden and normalized != "diagnosis_free")
                or "tp53" in normalized
                or ("diagnosis" in normalized and normalized != "diagnosis_free")
                or "outcome" in normalized
                or normalized.endswith("y_true")
            ):
                raise BranchReceiptError(
                    f"{context} contains forbidden diagnosis field {key!r}"
                )
            reject_diagnosis_fields(nested, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_diagnosis_fields(nested, context=f"{context}[{index}]")


def _ge(left: float, right: float) -> bool:
    return left > right or math.isclose(left, right, rel_tol=0.0, abs_tol=ABS_TOLERANCE)


def _le(left: float, right: float) -> bool:
    return left < right or math.isclose(left, right, rel_tol=0.0, abs_tol=ABS_TOLERANCE)


def _strict_gt(left: float, right: float) -> bool:
    return left > right and not math.isclose(
        left, right, rel_tol=0.0, abs_tol=ABS_TOLERANCE
    )


def _validate_embedded_identity(value: Any, context: str) -> None:
    identity = _exact_object(value, IDENTITY_KEYS, context)
    if not isinstance(identity["canonical_path"], str) or not identity["canonical_path"]:
        raise BranchReceiptError(f"{context}.canonical_path must be nonempty")
    if (
        isinstance(identity["bytes"], bool)
        or not isinstance(identity["bytes"], int)
        or identity["bytes"] <= 0
    ):
        raise BranchReceiptError(f"{context}.bytes must be a positive integer")
    digest = identity["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise BranchReceiptError(f"{context}.sha256 must be a lowercase SHA-256")


def validate_metric_input(
    value: Any,
) -> tuple[
    dict[tuple[str, int, str, str, str], float],
    dict[tuple[str, int, str], float],
]:
    document = _exact_object(value, TOP_LEVEL_KEYS, "metric input")
    reject_diagnosis_fields(document)
    if document["schema"] != METRIC_INPUT_SCHEMA:
        raise BranchReceiptError("metric-input schema drift")
    if document["study_id"] != STUDY_ID or document["status"] != "complete":
        raise BranchReceiptError("metric-input study/status drift")
    if document["diagnosis_free"] is not True:
        raise BranchReceiptError("metric input is not diagnosis-free")
    if document["fm_seeds"] != list(FM_SEEDS):
        raise BranchReceiptError("FM-seed topology drift")
    if document["layers"] != list(LAYERS):
        raise BranchReceiptError("layer topology drift")
    if document["views"] != list(VIEWS) or document["probe_levels"] != list(LEVELS):
        raise BranchReceiptError("view/probe-level topology drift")

    caches = document["compact_caches"]
    expected_cache_keys = {f"{seed}|{layer}" for seed in FM_SEEDS for layer in LAYERS}
    if not isinstance(caches, Mapping) or set(caches) != expected_cache_keys:
        raise BranchReceiptError("compact-cache topology must be exactly 5 x 7")
    for key, identity in caches.items():
        _validate_embedded_identity(identity, f"compact_caches.{key}")
    _validate_embedded_identity(document["preflight_receipt"], "preflight_receipt")

    race_rows = document["race_probes"]
    if not isinstance(race_rows, list):
        raise BranchReceiptError("race_probes must be an array")
    race: dict[tuple[str, int, str, str, str], float] = {}
    for index, raw in enumerate(race_rows):
        row = _exact_object(raw, RACE_ROW_KEYS, f"race_probes[{index}]")
        seed, layer = row["fm_seed"], row["layer"]
        cancer, view, level = row["cancer"], row["view"], row["probe_level"]
        if (
            isinstance(seed, bool) or not isinstance(seed, int) or seed not in FM_SEEDS
            or layer not in LAYERS or cancer not in CANCERS
            or view not in VIEWS or level not in LEVELS
        ):
            raise BranchReceiptError(f"race_probes[{index}] topology drift")
        result = row["result"]
        if not isinstance(result, Mapping) or "oriented_leakage" not in result:
            raise BranchReceiptError(f"race_probes[{index}].result lacks oriented_leakage")
        key = (layer, seed, cancer, view, level)
        if key in race:
            raise BranchReceiptError(f"duplicate race-probe cell {key!r}")
        race[key] = _finite(
            result["oriented_leakage"],
            f"race_probes[{index}].result.oriented_leakage",
            low=0.0,
            high=0.5,
        )
    expected_race = {
        (layer, seed, cancer, view, level)
        for layer in LAYERS for seed in FM_SEEDS for cancer in CANCERS
        for view in VIEWS for level in LEVELS
    }
    if set(race) != expected_race:
        raise BranchReceiptError("race-probe topology must be exactly 7 x 5 x 2 x 2 x 2")

    cancer_rows = document["cancer_probes"]
    if not isinstance(cancer_rows, list):
        raise BranchReceiptError("cancer_probes must be an array")
    cancer_values: dict[tuple[str, int, str], float] = {}
    for index, raw in enumerate(cancer_rows):
        row = _exact_object(raw, CANCER_ROW_KEYS, f"cancer_probes[{index}]")
        seed, layer, view = row["fm_seed"], row["layer"], row["view"]
        if (
            isinstance(seed, bool) or not isinstance(seed, int) or seed not in FM_SEEDS
            or layer not in LAYERS or view not in VIEWS
        ):
            raise BranchReceiptError(f"cancer_probes[{index}] topology drift")
        result = row["result"]
        if not isinstance(result, Mapping) or "pooled_heldout_patient_auroc" not in result:
            raise BranchReceiptError(
                f"cancer_probes[{index}].result lacks pooled_heldout_patient_auroc"
            )
        key = (layer, seed, view)
        if key in cancer_values:
            raise BranchReceiptError(f"duplicate cancer-probe cell {key!r}")
        cancer_values[key] = _finite(
            result["pooled_heldout_patient_auroc"],
            f"cancer_probes[{index}].result.pooled_heldout_patient_auroc",
            low=0.0,
            high=1.0,
        )
    expected_cancer = {
        (layer, seed, view) for layer in LAYERS for seed in FM_SEEDS for view in VIEWS
    }
    if set(cancer_values) != expected_cancer:
        raise BranchReceiptError("cancer-probe topology must be exactly 7 x 5 x 2")
    return race, cancer_values


def _contrast_gate(
    name: str,
    race: Mapping[tuple[str, int, str, str, str], float],
    cancer: Mapping[tuple[str, int, str], float],
) -> dict[str, Any]:
    candidate, baseline = CONTRASTS[name]
    activity: dict[str, Any] = {}
    for level in LEVELS:
        strata: dict[str, Any] = {}
        level_pass = True
        for cancer_name in CANCERS:
            for view in VIEWS:
                reductions = [
                    race[(baseline, seed, cancer_name, view, level)]
                    - race[(candidate, seed, cancer_name, view, level)]
                    for seed in FM_SEEDS
                ]
                median = float(statistics.median(reductions))
                passing = sum(_ge(value, MINIMUM_REDUCTION) for value in reductions)
                passed = passing >= MINIMUM_PASSING_SEEDS and _ge(median, MINIMUM_REDUCTION)
                level_pass = level_pass and passed
                strata[f"{cancer_name}|{view}"] = {
                    "reductions_by_seed": dict(zip(map(str, FM_SEEDS), reductions, strict=True)),
                    "median_reduction": median,
                    "seeds_at_or_above_0.05": passing,
                    "pass": passed,
                }
        activity[level] = {"strata": strata, "pass": bool(level_pass)}

    preservation_views: dict[str, Any] = {}
    preservation_pass = True
    for view in VIEWS:
        losses = [
            cancer[(baseline, seed, view)] - cancer[(candidate, seed, view)]
            for seed in FM_SEEDS
        ]
        median = float(statistics.median(losses))
        passing = sum(_le(value, MAXIMUM_CANCER_LOSS) for value in losses)
        passed = passing >= MINIMUM_PASSING_SEEDS and _le(median, MAXIMUM_CANCER_LOSS)
        preservation_pass = preservation_pass and passed
        preservation_views[view] = {
            "losses_by_seed": dict(zip(map(str, FM_SEEDS), losses, strict=True)),
            "median_loss": median,
            "seeds_at_or_below_0.02": passing,
            "pass": passed,
        }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "activity": activity,
        "both_activity_subgates_pass": bool(all(activity[level]["pass"] for level in LEVELS)),
        "any_activity_subgate_pass": bool(any(activity[level]["pass"] for level in LEVELS)),
        "preservation": {"views": preservation_views, "pass": bool(preservation_pass)},
    }


def _hmask_feasibility(
    race: Mapping[tuple[str, int, str, str, str], float]
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    all_strata = True
    for cancer_name in CANCERS:
        for view in VIEWS:
            for level in LEVELS:
                b_values = [race[("B", seed, cancer_name, view, level)] for seed in FM_SEEDS]
                h_values = [race[("H", seed, cancer_name, view, level)] for seed in FM_SEEDS]
                b_median = float(statistics.median(b_values))
                h_median = float(statistics.median(h_values))
                b_count = sum(_ge(value, MINIMUM_REDUCTION) for value in b_values)
                h_count = sum(_strict_gt(value, 0.0) for value in h_values)
                passed = (
                    b_count >= MINIMUM_PASSING_SEEDS
                    and _ge(b_median, MINIMUM_REDUCTION)
                    and h_count >= MINIMUM_PASSING_SEEDS
                    and _ge(h_median, MINIMUM_H_MEDIAN)
                )
                all_strata = all_strata and passed
                strata[f"{cancer_name}|{view}|{level}"] = {
                    "B_by_seed": dict(zip(map(str, FM_SEEDS), b_values, strict=True)),
                    "H_by_seed": dict(zip(map(str, FM_SEEDS), h_values, strict=True)),
                    "B_median": b_median,
                    "H_median": h_median,
                    "B_seeds_at_or_above_0.05": b_count,
                    "H_seeds_strictly_positive": h_count,
                    "pass": passed,
                }
    target_keys = [
        (cancer_name, view, level)
        for cancer_name in CANCERS for view in VIEWS for level in LEVELS
    ]
    target_b = [race[("B", FM_SEEDS[0], *key)] for key in target_keys]
    target_h = [race[("H", FM_SEEDS[0], *key)] for key in target_keys]
    target = {
        "fm_seed": FM_SEEDS[0],
        "B_values": target_b,
        "H_values": target_h,
        "all_B_at_or_above_0.05": all(_ge(value, MINIMUM_REDUCTION) for value in target_b),
        "all_H_strictly_positive": all(_strict_gt(value, 0.0) for value in target_h),
        "H_median": float(statistics.median(target_h)),
    }
    target["H_median_at_or_above_0.03"] = _ge(target["H_median"], MINIMUM_H_MEDIAN)
    target["pass"] = bool(
        target["all_B_at_or_above_0.05"]
        and target["all_H_strictly_positive"]
        and target["H_median_at_or_above_0.03"]
    )
    passed = bool(target["pass"] and all_strata)
    return {
        "evaluated": True,
        "target_seed": target,
        "five_seed_strata": strata,
        "all_five_seed_strata_pass": bool(all_strata),
        "pass": passed,
        "classification": "adequate_headroom" if passed else "inadequate_headroom",
    }


def evaluate_decision(value: Any) -> dict[str, Any]:
    race, cancer = validate_metric_input(value)
    gates = {name: _contrast_gate(name, race, cancer) for name in CONTRASTS}
    h, p, e, a = gates["H"], gates["P"], gates["E"], gates["A_temp"]
    feasibility: dict[str, Any] = {"evaluated": False, "reason": "branch_is_not_route_7"}
    if h["both_activity_subgates_pass"] and h["preservation"]["pass"]:
        route, name, action = 1, "final_H_active_preserved", "use_H"
    elif p["both_activity_subgates_pass"] and p["preservation"]["pass"]:
        route, name, action = 2, "final_P_active_preserved", "use_P"
    elif (
        (h["any_activity_subgate_pass"] and not h["preservation"]["pass"])
        or (p["any_activity_subgate_pass"] and not p["preservation"]["pass"])
    ):
        route, name, action = 3, "final_activity_without_preservation", "stop_utility_harm"
    elif e["any_activity_subgate_pass"]:
        route, name = 4, "fair_E_activity"
        action = "run_carry_versus_fresh" if e["preservation"]["pass"] else "stop_utility_harm"
    elif a["activity"]["patient"]["pass"]:
        route, name = 5, "temporary_A_patient_activity"
        action = "run_carry_versus_fresh" if a["preservation"]["pass"] else "stop_utility_harm"
    elif a["activity"]["tile"]["pass"]:
        route, name = 6, "temporary_A_tile_only_activity"
        action = "run_patient_mean_training" if a["preservation"]["pass"] else "stop_utility_harm"
    else:
        route, name = 7, "temporary_A_inactive"
        if not a["preservation"]["pass"]:
            action = "stop_utility_harm"
            feasibility = {"evaluated": False, "reason": "A_temp_preservation_failed"}
        else:
            feasibility = _hmask_feasibility(race)
            action = (
                "train_H_mask_32001"
                if feasibility["pass"]
                else "no_training_inadequate_headroom"
            )
    return {
        "branch_precedence_version": "objective-mask-erratum-c94f688",
        "contrast_gates": gates,
        "selected_route": route,
        "selected_route_name": name,
        "hmask_feasibility": feasibility,
        "action": action,
    }


def verify_metric_input(
    metric_input: Path,
    *,
    erratum: Path = ERRATUM_PATH,
    numeric_amendment: Path = NUMERIC_AMENDMENT_PATH,
) -> dict[str, Any]:
    sources = {
        "metric_input": Path(metric_input),
        "erratum": Path(erratum),
        "numeric_amendment_07": Path(numeric_amendment),
        "contract_source": Path(contract.__file__),
        "branch_verifier_source": Path(__file__),
    }
    identities = {name: file_identity(path) for name, path in sources.items()}
    if identities["erratum"]["sha256"] != ERRATUM_SHA256:
        raise BranchReceiptError("erratum differs from frozen c94f688 content")
    if identities["numeric_amendment_07"]["sha256"] != NUMERIC_AMENDMENT_SHA256:
        raise BranchReceiptError("numeric Amendment 07 identity drift")
    decision = evaluate_decision(load_json(Path(metric_input)))
    for name, path in sources.items():
        if file_identity(path) != identities[name]:
            raise BranchReceiptError(f"{name} changed during branch verification")
    return {
        "schema": SCHEMA,
        "study_id": STUDY_ID,
        "status": "complete",
        "diagnosis_free": True,
        "frozen_specification": {
            "erratum_commit": ERRATUM_COMMIT,
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": 0.0,
            "missing_nonfinite_or_topology_drift": "fail_closed",
        },
        "identities": identities,
        "decision": decision,
    }


def write_json_atomic_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a strict JSON receipt without replacing an old one."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise BranchReceiptError(f"refusing to overwrite {destination}") from error
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric_input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--erratum", type=Path, default=ERRATUM_PATH)
    parser.add_argument("--numeric-amendment", type=Path, default=NUMERIC_AMENDMENT_PATH)
    arguments = parser.parse_args(argv)
    try:
        receipt = verify_metric_input(
            arguments.metric_input,
            erratum=arguments.erratum,
            numeric_amendment=arguments.numeric_amendment,
        )
        write_json_atomic_exclusive(arguments.output, receipt)
    except BranchReceiptError as error:
        sys.stderr.write(f"representation branch receipt failed: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
