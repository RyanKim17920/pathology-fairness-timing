#!/usr/bin/env python3
"""Seal one completed matched-stage run into a tamper-evident receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPO = Path("/admin/home/ryan.kim/nt")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.matched_cancer_stage_20260730.receipts import (  # noqa: E402
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--effective-config-receipt", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    config_receipt_path = Path(args.effective_config_receipt).resolve()
    config_receipt = verify_receipt(
        config_receipt_path,
        expected_schema="matched-cancer-stage-effective-config/v1",
    )
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "latest.pt"
    summary = json.loads(summary_path.read_text())
    stage = summary["matched_stage"]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    encoder_sha256 = state_dict_sha256(checkpoint["model"])
    adapter_sha256 = state_dict_sha256(checkpoint["stage_adapter"])
    if encoder_sha256 != stage["encoder_post_sha256"]:
        raise ValueError("checkpoint encoder differs from summary")
    if adapter_sha256 != stage["adapter_post_sha256"]:
        raise ValueError("checkpoint adapter differs from summary")
    if summary["stop_reason"] != "max_train_samples":
        raise ValueError("run did not reach its fixed sample budget")

    identities = {
        "effective_config_receipt": file_identity(config_receipt_path),
        "effective_config": config_receipt["identities"]["effective_config"],
        "replay_manifest": config_receipt["identities"]["replay_manifest"],
        "latest_checkpoint": file_identity(checkpoint_path),
        "metrics": file_identity(metrics_path),
        "summary": file_identity(summary_path),
    }
    receipt = build_receipt(
        schema="matched-cancer-stage-completion/v1",
        study_id=config_receipt["study_id"],
        scenario=config_receipt["scenario"],
        identities=identities,
        fields={
            "status": "complete",
            "run_name": summary["project"],
            "mode": stage["mode"],
            "fair_weight": stage["fair_weight"],
            "steps_completed": summary["steps_completed"],
            "tile_presentations": summary["tile_presentations"],
            "encoder_pre_sha256": stage["encoder_pre_sha256"],
            "encoder_post_sha256": encoder_sha256,
            "adapter_pre_sha256": stage["adapter_pre_sha256"],
            "adapter_post_sha256": adapter_sha256,
            "replay_manifest_payload_sha256": stage[
                "replay_sampler_sha256"
            ],
            "replay_tile_sha256": stage["replay_tile_sha256"],
            "replay_patient_sha256": stage["replay_patient_sha256"],
            "replay_augmentation_seed_sha256": stage[
                "replay_augmentation_seed_sha256"
            ],
        },
    )
    destination = atomic_write_receipt(args.destination, receipt)
    verify_receipt(
        destination,
        expected_schema="matched-cancer-stage-completion/v1",
        expected_study_id=config_receipt["study_id"],
        expected_scenario=config_receipt["scenario"],
    )
    print(
        json.dumps(
            {
                "status": "matched_cancer_completion_receipt_valid",
                "path": str(destination),
                "encoder_post_sha256": encoder_sha256,
                "adapter_post_sha256": adapter_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
