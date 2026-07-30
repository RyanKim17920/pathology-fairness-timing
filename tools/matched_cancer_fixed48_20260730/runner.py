#!/usr/bin/env python3
"""Prepare, build, verify, and seal one fixed48 calibration attempt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml


REPO = Path("/admin/home/ryan.kim/nt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.matched_cancer_fixed48_20260730 import contract  # noqa: E402
from tools.matched_cancer_stage_20260730 import config_builder  # noqa: E402
from tools.matched_cancer_stage_20260730.receipts import (  # noqa: E402
    atomic_write_receipt,
    build_receipt,
    file_identity,
    require_regular_file,
    verify_receipt,
)


ROOT_RECEIPT_SCHEMA = (
    "matched-cancer-fixed48-calibration-root-completion/v1"
)
PRETRAINED_ENCODER_STATE_SHA256 = (
    "ba9418ed2138e42250085b04e0502d621b072c4bb60240f2845a27fbf3184bd6"
)
RUN_SPECS = {
    "slot1_plain": {
        "mode": "joint",
        "fair_weight": 0.0,
        "parent": None,
    },
    "slot1_fair": {
        "mode": "joint",
        "fair_weight": 0.1,
        "parent": None,
    },
    "B": {
        "mode": "adapter_only",
        "fair_weight": 0.0,
        "parent": "slot1_plain",
    },
    "H": {
        "mode": "adapter_only",
        "fair_weight": 0.1,
        "parent": "slot1_plain",
    },
    "P": {
        "mode": "adapter_only",
        "fair_weight": 0.0,
        "parent": "slot1_fair",
    },
}


def _paths(seed: int, root: Path | str) -> tuple[dict[str, Any], Path]:
    plan, lookup, _ = contract.load_and_validate_plan()
    if seed not in lookup:
        raise ValueError("representation seed must be in 32001..32048")
    attempt_root = contract.validate_attempt_root(
        root, seed=seed, output_namespace=plan["output_namespace"]
    )
    return plan, attempt_root


def _scenario(seed: int) -> str:
    return f"brca_luad_black_white_calibration_seed{seed}"


def _verify_contract(seed: int, root: Path) -> dict[str, Any]:
    receipt = verify_receipt(
        root / "CALIBRATION_CONTRACT_RECEIPT.json",
        expected_schema=contract.RECEIPT_SCHEMA,
        expected_study_id=contract.STUDY_ID,
        expected_scenario=_scenario(seed),
    )
    if receipt.get("representation_seed") != seed:
        raise ValueError("calibration receipt seed mismatch")
    expected = {
        "replay_seed": seed + 20_000,
        "data_order_seed": seed + 30_000,
        "adapter_init_seed": seed + 40_000,
        "replay_presentations": 99_968,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"calibration receipt {key} drift")
    return receipt


def prepare(seed: int, root: Path | str) -> Path:
    """Create an exclusive attempt and its immutable seed-specific inputs."""
    _, attempt_root = _paths(seed, root)
    if attempt_root.exists() or attempt_root.is_symlink():
        raise FileExistsError(
            f"refusing to reuse calibration attempt: {attempt_root}"
        )
    attempt_root.mkdir(parents=True, exist_ok=False)
    _, contract_path, receipt_path = contract.materialize(
        seed=seed, root=attempt_root
    )
    replay_manifest = attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"
    config_path = attempt_root / "configs" / "manifest_build.yaml"
    args = argparse.Namespace(
        contract=str(contract_path),
        contract_receipt=str(receipt_path),
        replay_manifest=str(replay_manifest),
        manifest_build_only=True,
        mode="joint",
        fair_weight=0.0,
        encoder_checkpoint=None,
        parent_completion_receipt=None,
        name=f"matched-cancer-fixed48-seed{seed}-manifest-build",
        output_dir=str(attempt_root / "manifest_build_not_run"),
        destination=str(config_path),
        receipt=None,
    )
    config_builder.build(args)
    effective = yaml.safe_load(require_regular_file(config_path).read_text())
    contract.validate_effective_representation_metadata(effective)
    return config_path


def build_run_config(seed: int, root: Path | str, run: str) -> Path:
    """Build exactly one frozen run config; no mode/weight is caller-controlled."""
    _, attempt_root = _paths(seed, root)
    if run not in RUN_SPECS:
        raise ValueError(f"unknown calibration run: {run}")
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise ValueError("calibration attempt root is missing or a symlink")
    _verify_contract(seed, attempt_root)
    replay_manifest = require_regular_file(
        attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"
    )
    contract_path = require_regular_file(
        attempt_root / "configs" / "calibration_contract.yaml"
    )
    config_path = attempt_root / "configs" / f"{run}.yaml"
    config_receipt = config_path.with_suffix(
        config_path.suffix + ".receipt.json"
    )
    output = attempt_root / run
    for path in (config_path, config_receipt, output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite run artifact: {path}")

    spec = RUN_SPECS[run]
    checkpoint = None
    parent_receipt = None
    if spec["parent"] is not None:
        parent = attempt_root / str(spec["parent"])
        checkpoint = require_regular_file(parent / "latest.pt")
        parent_receipt = require_regular_file(
            parent / "COMPLETION_RECEIPT.json"
        )
    args = argparse.Namespace(
        contract=str(contract_path),
        contract_receipt=str(
            attempt_root / "CALIBRATION_CONTRACT_RECEIPT.json"
        ),
        replay_manifest=str(replay_manifest),
        manifest_build_only=False,
        mode=spec["mode"],
        fair_weight=spec["fair_weight"],
        encoder_checkpoint=str(checkpoint) if checkpoint else None,
        parent_completion_receipt=(
            str(parent_receipt) if parent_receipt else None
        ),
        name=f"matched-cancer-fixed48-seed{seed}-{run}",
        output_dir=str(output),
        destination=str(config_path),
        receipt=None,
    )
    config_builder.build(args)
    effective = yaml.safe_load(require_regular_file(config_path).read_text())
    contract.validate_effective_representation_metadata(effective)
    stage = effective.get("matched_stage", {})
    expected_stage = {
        "mode": spec["mode"],
        "fair_weight": spec["fair_weight"],
        "adapter_init_seed": seed + 40_000,
        "data_order_seed": seed + 30_000,
    }
    for key, value in expected_stage.items():
        if stage.get(key) != value:
            raise ValueError(f"effective config {run} {key} drift")
    if effective.get("train", {}).get("seed") != seed:
        raise ValueError("effective config representation seed drift")
    if effective.get("train", {}).get("max_train_samples") != 99_968:
        raise ValueError("effective config exposure drift")
    verify_receipt(
        config_receipt,
        expected_schema="matched-cancer-stage-effective-config/v1",
        expected_study_id=contract.STUDY_ID,
        expected_scenario=_scenario(seed),
    )
    return config_path


def verify_ready_to_train(seed: int, root: Path | str, run: str) -> Path:
    """Fail closed immediately before invoking the external trainer."""
    _, attempt_root = _paths(seed, root)
    _verify_contract(seed, attempt_root)
    if run not in RUN_SPECS:
        raise ValueError(f"unknown calibration run: {run}")
    config_path = require_regular_file(
        attempt_root / "configs" / f"{run}.yaml"
    )
    effective = yaml.safe_load(config_path.read_text())
    contract.validate_effective_representation_metadata(effective)
    config_receipt = verify_receipt(
        config_path.with_suffix(config_path.suffix + ".receipt.json"),
        expected_schema="matched-cancer-stage-effective-config/v1",
        expected_study_id=contract.STUDY_ID,
        expected_scenario=_scenario(seed),
    )
    if config_receipt.get("mode") != RUN_SPECS[run]["mode"]:
        raise ValueError("effective config receipt mode drift")
    if config_receipt.get("fair_weight") != RUN_SPECS[run]["fair_weight"]:
        raise ValueError("effective config receipt fair-weight drift")
    output = attempt_root / run
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse run output: {output}")
    return config_path


def _load_stage(root: Path, run: str) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = require_regular_file(root / run / "summary.json")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or not isinstance(
        summary.get("matched_stage"), dict
    ):
        raise ValueError(f"invalid summary for {run}")
    return summary, summary["matched_stage"]


def finalize(seed: int, root: Path | str) -> Path:
    """Verify all five runs and atomically seal the immutable attempt root."""
    _, attempt_root = _paths(seed, root)
    _, _, legacy = contract.load_and_validate_plan()
    contract_receipt = _verify_contract(seed, attempt_root)
    destination = attempt_root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite root receipt: {destination}")
    summaries: dict[str, dict[str, Any]] = {}
    stages: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    for run in contract.RUN_ORDER:
        spec = RUN_SPECS[run]
        summary, stage = _load_stage(attempt_root, run)
        summaries[run] = summary
        stages[run] = stage
        if summary.get("steps_completed") != 781:
            raise ValueError(f"{run} did not complete exactly 781 steps")
        if summary.get("tile_presentations") != 99_968:
            raise ValueError(f"{run} exposure is not exactly 99,968")
        if summary.get("stop_reason") != "max_train_samples":
            raise ValueError(f"{run} did not stop at its fixed exposure")
        if stage.get("mode") != spec["mode"]:
            raise ValueError(f"{run} mode drift")
        if stage.get("fair_weight") != spec["fair_weight"]:
            raise ValueError(f"{run} fairness timing drift")
        completion = verify_receipt(
            attempt_root / run / "COMPLETION_RECEIPT.json",
            expected_schema="matched-cancer-stage-completion/v1",
            expected_study_id=contract.STUDY_ID,
            expected_scenario=_scenario(seed),
        )
        completions[run] = completion
        if completion.get("steps_completed") != 781:
            raise ValueError(f"{run} completion step drift")
        if completion.get("tile_presentations") != 99_968:
            raise ValueError(f"{run} completion exposure drift")
        metrics = require_regular_file(attempt_root / run / "metrics.jsonl")
        rows = [
            json.loads(line)
            for line in metrics.read_text().splitlines()
            if line.strip() and '"total"' in line
        ]
        if len(rows) != 781:
            raise ValueError(f"{run} metrics do not contain exactly 781 steps")
        for row in rows:
            for key in (
                "cancer",
                "race_fair",
                "total",
                "h_dose_main_grad_norm",
                "h_dose_fair_grad_norm",
            ):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"{run} has non-finite metric {key}")
            if float(row["cancer"]) <= 0:
                raise ValueError(f"{run} cancer objective is inactive")
            fair_grad = float(row["h_dose_fair_grad_norm"])
            if spec["fair_weight"] > 0 and fair_grad <= 0:
                raise ValueError(f"{run} fairness objective is inactive")
            if spec["fair_weight"] == 0 and fair_grad != 0:
                raise ValueError(f"{run} fairness objective should be disabled")

    replay_keys = (
        "adapter_pre_sha256",
        "replay_sampler_sha256",
        "replay_patient_sha256",
        "replay_tile_sha256",
        "replay_augmentation_seed_sha256",
        "replay_manifest_file_sha256",
        "sample_batch_trace_sha256",
        "patient_batch_trace_sha256",
        "augmentation_seed_batch_trace_sha256",
        "augmentation_seed_manifest_trace_sha256",
    )
    for key in replay_keys:
        if len({stages[run].get(key) for run in contract.RUN_ORDER}) != 1:
            raise ValueError(f"five-run replay/initialization mismatch: {key}")
    if (
        stages["slot1_plain"].get("encoder_pre_sha256")
        != stages["slot1_fair"].get("encoder_pre_sha256")
    ):
        raise ValueError("Slot1 encoders do not share initialization")
    if (
        stages["slot1_plain"].get("encoder_pre_sha256")
        != PRETRAINED_ENCODER_STATE_SHA256
    ):
        raise ValueError("Slot1 pretrained encoder ancestor differs")
    for run in ("B", "H", "P"):
        if not stages[run].get("encoder_unchanged"):
            raise ValueError(f"{run} Slot2 encoder was not frozen")
    if not (
        stages["B"].get("encoder_pre_sha256")
        == stages["H"].get("encoder_pre_sha256")
        == stages["slot1_plain"].get("encoder_post_sha256")
    ):
        raise ValueError("B/H do not descend from the plain Slot1 encoder")
    if (
        stages["P"].get("encoder_pre_sha256")
        != stages["slot1_fair"].get("encoder_post_sha256")
    ):
        raise ValueError("P does not descend from the fair Slot1 encoder")
    for run in ("slot1_plain", "slot1_fair"):
        reach = stages[run].get("encoder_reachability", {})
        update = stages[run].get("encoder_first_update", {})
        if not reach.get("encoder_stage_grad_finite"):
            raise ValueError(f"{run} encoder stage gradient is invalid")
        if float(reach.get("encoder_cancer_grad_norm", 0)) <= 0:
            raise ValueError(f"{run} cancer gradient did not reach encoder")
        if float(reach.get("encoder_fair_raw_grad_norm", 0)) <= 0:
            raise ValueError(f"{run} fair gradient diagnostic is inactive")
        weighted = float(reach.get("encoder_fair_weighted_grad_norm", -1))
        if RUN_SPECS[run]["fair_weight"] > 0 and weighted <= 0:
            raise ValueError(f"{run} weighted fair gradient is inactive")
        if RUN_SPECS[run]["fair_weight"] == 0 and weighted != 0:
            raise ValueError(f"{run} weighted fair gradient should be zero")
        if float(update.get("encoder_first_positive_lr_update_norm", 0)) <= 0:
            raise ValueError(f"{run} encoder was not updated")
        if int(update.get("encoder_first_positive_lr_changed_tensors", 0)) <= 0:
            raise ValueError(f"{run} encoder tensors did not change")

    replay_manifest = require_regular_file(
        attempt_root / "CALIBRATION_REPLAY_MANIFEST.json"
    )
    root_receipt = build_receipt(
        schema=ROOT_RECEIPT_SCHEMA,
        study_id=contract.STUDY_ID,
        scenario=_scenario(seed),
        identities={
            "contract_receipt": file_identity(
                attempt_root / "CALIBRATION_CONTRACT_RECEIPT.json"
            ),
            "replay_manifest": file_identity(replay_manifest),
            "runs": {
                run: file_identity(
                    attempt_root / run / "COMPLETION_RECEIPT.json"
                )
                for run in contract.RUN_ORDER
            },
        },
        fields={
            "status": "fixed48_two_slot_calibration_complete",
            "representation_seed": seed,
            "steps_per_run": 781,
            "presentations_per_run": 99_968,
            "total_presentations": 499_840,
            "run_names": list(contract.RUN_ORDER),
            "arm_timing": {
                arm: {
                    "slot1_fair_weight": values["slot1_fair_weight"],
                    "slot2_fair_weight": values["slot2_fair_weight"],
                }
                for arm, values in (
                    yaml.safe_load(
                        require_regular_file(contract.PLAN).read_text()
                    )["arms"].items()
                )
            },
            "replay_manifest_payload_sha256": stages["B"][
                "replay_sampler_sha256"
            ],
            "contract_topology_sha256": contract_receipt["topology_sha256"],
            "legacy_seed32001_disposition": legacy,
        },
    )
    atomic_write_receipt(destination, root_receipt)
    verify_receipt(
        destination,
        expected_schema=ROOT_RECEIPT_SCHEMA,
        expected_study_id=contract.STUDY_ID,
        expected_scenario=_scenario(seed),
    )
    return destination


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--seed", type=int, required=True)
        item.add_argument("--root", type=Path, required=True)
    for name in ("build", "verify-ready"):
        item = sub.add_parser(name)
        item.add_argument("--seed", type=int, required=True)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--run", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare(args.seed, args.root)
    elif args.command == "build":
        path = build_run_config(args.seed, args.root, args.run)
    elif args.command == "verify-ready":
        path = verify_ready_to_train(args.seed, args.root, args.run)
    else:
        path = finalize(args.seed, args.root)
    print(
        json.dumps(
            {
                "status": f"fixed48_calibration_{args.command}_valid",
                "seed": args.seed,
                "path": str(path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
