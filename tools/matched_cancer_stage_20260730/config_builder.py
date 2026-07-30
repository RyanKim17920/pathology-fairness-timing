#!/usr/bin/env python3
"""Build effective Slot-1/Slot-2 configs from the frozen smoke contract."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path("/admin/home/ryan.kim/nt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def build(args: argparse.Namespace) -> None:
    contract_path = Path(args.contract).resolve()
    contract = load_yaml(contract_path)
    study_id = str(contract["study_id"])
    scenario = str(contract["scenario"])
    contract_receipt_path = Path(args.contract_receipt).resolve()
    contract_receipt = verify_receipt(
        contract_receipt_path,
        expected_schema="matched-cancer-stage-provenance/v1",
        expected_study_id=study_id,
        expected_scenario=scenario,
    )
    replay_manifest_path = Path(args.replay_manifest).resolve()
    base = load_yaml(REPO / contract["base_config"])
    mode = args.mode
    if mode not in {"joint", "adapter_only"}:
        raise ValueError("mode must be joint or adapter_only")
    fair_weight = float(args.fair_weight)
    if fair_weight not in {0.0, float(contract["fair_weight"])}:
        raise ValueError("fair weight is outside the frozen smoke contract")
    if args.manifest_build_only and mode != "joint":
        raise ValueError("--manifest-build-only requires joint mode")
    if not args.manifest_build_only and not replay_manifest_path.is_file():
        raise ValueError(
            f"replay manifest is not a regular file: {replay_manifest_path}"
        )

    parent_receipt_path = None
    parent_receipt = None
    encoder_checkpoint_identity = None
    expected_encoder_state_sha256 = None
    if mode == "adapter_only":
        if not args.parent_completion_receipt:
            raise ValueError(
                "adapter_only requires --parent-completion-receipt"
            )
        parent_receipt_path = Path(
            args.parent_completion_receipt
        ).resolve()
        parent_receipt = verify_receipt(
            parent_receipt_path,
            expected_schema="matched-cancer-stage-completion/v1",
            expected_study_id=study_id,
            expected_scenario=scenario,
        )
        encoder_checkpoint_identity = parent_receipt.get(
            "identities", {}
        ).get("latest_checkpoint")
        expected_encoder_state_sha256 = parent_receipt.get(
            "encoder_post_sha256"
        )
        if (
            not isinstance(encoder_checkpoint_identity, dict)
            or not isinstance(expected_encoder_state_sha256, str)
        ):
            raise ValueError(
                "parent completion receipt lacks checkpoint/encoder identity"
            )
        if file_identity(
            encoder_checkpoint_identity["canonical_path"]
        ) != encoder_checkpoint_identity:
            raise ValueError("parent encoder checkpoint identity drift")
        if (
            args.encoder_checkpoint
            and Path(args.encoder_checkpoint).resolve()
            != Path(encoder_checkpoint_identity["canonical_path"])
        ):
            raise ValueError(
                "--encoder-checkpoint differs from parent completion receipt"
            )
    elif args.encoder_checkpoint or args.parent_completion_receipt:
        raise ValueError(
            "joint mode cannot receive encoder/parent checkpoint provenance"
        )

    base["project"]["name"] = args.name
    base["project"]["output_dir"] = str(Path(args.output_dir).resolve())
    base["data"]["exclude_barcodes_file"] = str(
        (REPO / contract["exclude_barcodes_file"]).resolve()
    )
    base["data"]["include_discrete"] = {
        "cancer": list(contract["population"]["cancer_ids"]),
        "race": list(contract["population"]["race_ids"]),
    }
    replay = contract["replay"]
    base["train"].update(
        {
            "batch_size": int(replay["batch_size"]),
            "max_train_samples": int(replay["steps"]) * int(replay["batch_size"]),
            "log_every": 1,
            "save_every": int(replay["steps"]),
            "eval_every": int(replay["steps"]),
            "val_batches": 1,
            "num_workers": 2,
            "persistent_workers": True,
            "resume": None,
        }
    )
    base["probe"]["enabled"] = False
    base["fino"] = {
        "enabled": True,
        "objective": "contrastive-two-condition",
        "method": "contrastive",
        "gamma_max": 0.7,
        "contrastive_temp": 0.2,
        "contrastive_weight": 0.1,
        "contrastive_condition_on": "cancer",
        "dose_logging": True,
        "race_weight": "none",
        "race_resample": False,
        "discrete": [["cancer", 1], ["race", -1]],
        "continuous": [],
    }
    destination = Path(args.destination).resolve()
    receipt_path = (
        Path(args.receipt).resolve()
        if args.receipt
        else destination.with_suffix(destination.suffix + ".receipt.json")
    )
    base["matched_stage"] = {
        "enabled": True,
        "mode": mode,
        "study_id": study_id,
        "scenario": scenario,
        "contract_receipt": str(contract_receipt_path),
        "effective_config_receipt": (
            str(receipt_path) if not args.manifest_build_only else ""
        ),
        "replay_manifest": str(replay_manifest_path),
        "adapter_init_seed": int(contract["adapter"]["init_seed"]),
        "fair_weight": fair_weight,
        "adapter_lr": float(contract["adapter"]["lr"]),
        "adapter_weight_decay": float(contract["adapter"]["weight_decay"]),
        "data_order_seed": int(contract["data_order_seed"]),
        "encoder_checkpoint": (
            encoder_checkpoint_identity["canonical_path"]
            if encoder_checkpoint_identity is not None
            else None
        ),
        "encoder_checkpoint_sha256": (
            encoder_checkpoint_identity["sha256"]
            if encoder_checkpoint_identity is not None
            else None
        ),
        "expected_encoder_state_sha256": expected_encoder_state_sha256,
        "parent_completion_receipt": (
            str(parent_receipt_path)
            if parent_receipt_path is not None
            else None
        ),
        "replay": {
            "cancer_ids": list(contract["population"]["cancer_ids"]),
            "race_ids": list(contract["population"]["race_ids"]),
            "steps": int(replay["steps"]),
            "seed": int(replay["seed"]),
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w") as handle:
        yaml.safe_dump(base, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    if args.manifest_build_only:
        return

    identities = {
        "effective_config": file_identity(destination),
        "contract_receipt": file_identity(contract_receipt_path),
        "replay_manifest": file_identity(replay_manifest_path),
    }
    if parent_receipt_path is not None:
        identities.update(
            {
                "parent_completion_receipt": file_identity(
                    parent_receipt_path
                ),
                "encoder_checkpoint": encoder_checkpoint_identity,
            }
        )
    effective_receipt = build_receipt(
        schema="matched-cancer-stage-effective-config/v1",
        study_id=study_id,
        scenario=scenario,
        identities=identities,
        fields={
            "contract_topology_sha256": contract_receipt[
                "topology_sha256"
            ],
            "mode": mode,
            "fair_weight": fair_weight,
            "expected_encoder_state_sha256": (
                expected_encoder_state_sha256
            ),
        },
    )
    atomic_write_receipt(receipt_path, effective_receipt)
    verify_receipt(
        receipt_path,
        expected_schema="matched-cancer-stage-effective-config/v1",
        expected_study_id=study_id,
        expected_scenario=scenario,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--contract", required=True)
    result.add_argument("--mode", required=True)
    result.add_argument("--fair-weight", type=float, required=True)
    result.add_argument("--encoder-checkpoint")
    result.add_argument("--parent-completion-receipt")
    result.add_argument("--contract-receipt", required=True)
    result.add_argument("--replay-manifest", required=True)
    result.add_argument("--receipt")
    result.add_argument("--manifest-build-only", action="store_true")
    result.add_argument("--name", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--destination", required=True)
    return result


if __name__ == "__main__":
    build(parser().parse_args())
