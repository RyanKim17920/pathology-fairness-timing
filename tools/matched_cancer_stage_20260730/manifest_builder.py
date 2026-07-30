#!/usr/bin/env python3
"""Create and independently reload one immutable matched-stage replay manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


REPO = Path("/admin/home/ryan.kim/nt")
TRAIN_REPO = REPO / "vendor/matched_stage_train_20260730"
for path in (REPO, TRAIN_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataloader import TCGATileDataset  # noqa: E402
from tools.matched_cancer_stage_20260730.replay import (  # noqa: E402
    BalancedReplayBatchSampler,
    ReplayContract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("effective config root must be a mapping")
    stage = config.get("matched_stage") or {}
    replay = stage.get("replay") or {}
    contract = ReplayContract(
        cancer_ids=tuple(int(value) for value in replay["cancer_ids"]),
        race_ids=tuple(int(value) for value in replay["race_ids"]),
        batch_size=int(config["train"]["batch_size"]),
        steps=int(replay["steps"]),
        seed=int(replay["seed"]),
    )
    dataset = TCGATileDataset(config, is_train=True)
    generated = BalancedReplayBatchSampler(dataset, contract)
    destination = generated.write_manifest(Path(args.destination).resolve())
    reloaded = BalancedReplayBatchSampler.from_manifest(
        dataset,
        destination,
        expected_contract=contract,
    )
    if generated.sha256 != reloaded.sha256:
        raise RuntimeError("written replay manifest does not round-trip")
    print(
        json.dumps(
            {
                "status": "matched_cancer_replay_manifest_valid",
                "path": str(destination),
                "manifest_file_sha256": reloaded.manifest_file_sha256,
                "manifest_payload_sha256": reloaded.sha256,
                "patient_sha256": reloaded.patient_sha256,
                "tile_sha256": reloaded.tile_sha256,
                "augmentation_seed_sha256": (
                    reloaded.augmentation_seed_sha256
                ),
                "steps": len(reloaded),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
