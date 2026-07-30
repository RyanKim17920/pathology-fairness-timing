"""Fail-closed calibration-root ancestry checks for diagnostic deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.matched_cancer_stage_20260730.receipts import (
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)


ROOT_SCHEMA = "matched-cancer-stage-calibration-root-completion/v1"
COMPLETION_SCHEMA = "matched-cancer-stage-completion/v1"
EFFECTIVE_CONFIG_SCHEMA = "matched-cancer-stage-effective-config/v1"
ARMS = ("B", "P", "H")
RUNS = ("slot1_plain", "slot1_fair", "B", "H", "P")
EXPECTED_FAIR_WEIGHTS = {"B": 0.0, "P": 0.0, "H": 0.1}
EXPECTED_RUN_MODE_WEIGHT = {
    "slot1_plain": ("joint", 0.0),
    "slot1_fair": ("joint", 0.1),
    "B": ("adapter_only", 0.0),
    "H": ("adapter_only", 0.1),
    "P": ("adapter_only", 0.0),
}
EXPECTED_STATUS = "matched_cancer_two_slot_calibration_seed32001_valid"
CALIBRATION_CONTRACT_SCHEMA = "matched-cancer-stage-provenance/v1"
REPLAY_SCHEMA = "matched-cancer-replay-manifest/v1"


def _verify_replay_manifest(path: Path, expected_payload_sha256: str) -> None:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("calibration replay manifest is not canonical JSON")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema", "contract", "occurrences", "traces",
        "manifest_payload_sha256",
    }:
        raise ValueError("calibration replay manifest topology differs")
    if manifest["schema"] != REPLAY_SCHEMA:
        raise ValueError("calibration replay manifest schema differs")
    if manifest["contract"] != {
        "cancer_ids": [2, 15],
        "race_ids": [2, 4],
        "batch_size": 128,
        "steps": 781,
        "seed": 52001,
    }:
        raise ValueError("calibration replay contract differs")
    body = {
        key: manifest[key]
        for key in ("schema", "contract", "occurrences", "traces")
    }
    actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if (
        manifest["manifest_payload_sha256"] != actual
        or actual != expected_payload_sha256
    ):
        raise ValueError("calibration replay payload ancestry differs")


def verify_calibration_ancestry(
    calibration_root_receipt: str | Path,
    completion_receipts: Mapping[str, str | Path],
    *,
    expected_representation_seed: int,
) -> dict[str, Any]:
    """Verify the root, fixed seed, and exact B/P/H receipt ancestry."""
    if set(completion_receipts) != set(ARMS):
        raise ValueError("completion receipts must contain exactly B, P, and H")
    root_path = Path(calibration_root_receipt).resolve()
    root = verify_receipt(root_path, expected_schema=ROOT_SCHEMA)
    if root.get("status") != EXPECTED_STATUS:
        raise ValueError("calibration root is not a completed valid calibration")
    if root.get("representation_seed") != int(expected_representation_seed):
        raise ValueError(
            "calibration root representation_seed differs from deployment"
        )
    identities = root.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "contract_receipt", "replay_manifest", "runs"
    }:
        raise ValueError("calibration root identity topology differs")
    runs = identities.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(RUNS):
        raise ValueError("calibration root does not bind the exact five-run topology")
    if root.get("arms") != list(RUNS):
        raise ValueError("calibration root arm order differs")
    steps = root.get("steps_per_run")
    presentations = root.get("presentations_per_run")
    if steps != 781 or presentations != 99_968:
        raise ValueError("calibration root fixed budget differs")
    replay_sha = root.get("replay_manifest_payload_sha256")
    if not isinstance(replay_sha, str) or len(replay_sha) != 64:
        raise ValueError("calibration root replay identity is invalid")
    contract_receipt = verify_receipt(
        identities["contract_receipt"]["canonical_path"],
        expected_schema=CALIBRATION_CONTRACT_SCHEMA,
        expected_study_id=root["study_id"],
        expected_scenario=root["scenario"],
    )
    if (
        contract_receipt.get("status") != "valid"
        or contract_receipt.get("representation_seed")
        != expected_representation_seed
        or contract_receipt.get("replay_presentations") != presentations
    ):
        raise ValueError("calibration semantic contract differs")
    _verify_replay_manifest(
        Path(identities["replay_manifest"]["canonical_path"]), replay_sha
    )

    receipts: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for run in RUNS:
        path = Path(runs[run]["canonical_path"]).resolve()
        if run in ARMS:
            requested = Path(completion_receipts[run]).resolve()
            if file_identity(requested) != dict(runs[run]):
                raise ValueError(f"{run} completion receipt is not root-bound")
            path = requested
        receipt = verify_receipt(
            path,
            expected_schema=COMPLETION_SCHEMA,
            expected_study_id=root["study_id"],
            expected_scenario=root["scenario"],
        )
        mode, fair_weight = EXPECTED_RUN_MODE_WEIGHT[run]
        if receipt.get("status") != "complete":
            raise ValueError(f"{run} completion status differs")
        if receipt.get("mode") != mode or receipt.get("fair_weight") != fair_weight:
            raise ValueError(f"{run} completion mode/fairness differs")
        if receipt.get("steps_completed") != steps:
            raise ValueError(f"{run} step budget differs")
        if receipt.get("tile_presentations") != presentations:
            raise ValueError(f"{run} presentation budget differs")
        if receipt.get("replay_manifest_payload_sha256") != replay_sha:
            raise ValueError(f"{run} replay ancestry differs")
        expected_name = f"matched-cancer-calibration-seed32001-{run}"
        if receipt.get("run_name") != expected_name:
            raise ValueError(f"{run} completion run name differs")
        checkpoint = receipt.get("identities", {}).get("latest_checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or set(checkpoint) != {"canonical_path", "bytes", "sha256"}
            or Path(checkpoint["canonical_path"]).parent.name != run
        ):
            raise ValueError(f"{run} checkpoint path is not run-bound")
        receipts[run] = receipt
        paths[run] = path

    plain = receipts["slot1_plain"]
    fair = receipts["slot1_fair"]
    if plain.get("encoder_pre_sha256") != fair.get("encoder_pre_sha256"):
        raise ValueError("slot-1 encoders did not share initialization")
    parent_for = {"B": "slot1_plain", "H": "slot1_plain", "P": "slot1_fair"}
    verified: dict[str, Any] = {}
    for arm in ARMS:
        receipt = receipts[arm]
        parent_name = parent_for[arm]
        parent = receipts[parent_name]
        parent_checkpoint = parent["identities"]["latest_checkpoint"]
        parent_encoder = parent.get("encoder_post_sha256")
        if (
            receipt.get("encoder_pre_sha256") != parent_encoder
            or receipt.get("encoder_post_sha256") != parent_encoder
        ):
            raise ValueError(f"{arm} encoder does not descend from {parent_name}")
        effective_identity = receipt.get("identities", {}).get(
            "effective_config_receipt"
        )
        if not isinstance(effective_identity, Mapping):
            raise ValueError(f"{arm} lacks effective-config ancestry")
        effective = verify_receipt(
            effective_identity["canonical_path"],
            expected_schema=EFFECTIVE_CONFIG_SCHEMA,
            expected_study_id=root["study_id"],
            expected_scenario=root["scenario"],
        )
        effective_ids = effective.get("identities", {})
        if effective_ids.get("parent_completion_receipt") != dict(
            runs[parent_name]
        ):
            raise ValueError(f"{arm} parent completion ancestry differs")
        if effective_ids.get("encoder_checkpoint") != parent_checkpoint:
            raise ValueError(f"{arm} parent checkpoint ancestry differs")
        expected_mode, expected_weight = EXPECTED_RUN_MODE_WEIGHT[arm]
        if (
            effective.get("mode") != expected_mode
            or effective.get("fair_weight") != expected_weight
            or effective.get("expected_encoder_state_sha256") != parent_encoder
        ):
            raise ValueError(f"{arm} effective-config ancestry differs")
        verified[arm] = {
            "completion_receipt": paths[arm],
            "receipt": receipt,
            "checkpoint": Path(
                receipt["identities"]["latest_checkpoint"]["canonical_path"]
            ),
            "parent_run": parent_name,
        }
    return {"root_path": root_path, "root": root, "arms": verified}
