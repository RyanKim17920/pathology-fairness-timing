"""Synthetic, outcome-free tests for calibration ancestry verification."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.matched_cancer_diagnostic_20260730.integration import (
    ARMS,
    ROOT_SCHEMA,
    verify_calibration_ancestry,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    canonical_json_bytes,
    file_identity,
)


STUDY = "matched_cancer_stage_20260730"
SCENARIO = "brca_luad_black_white_calibration_seed32001"


def make_calibration_tree(root: Path) -> tuple[Path, dict[str, Path]]:
    """Create the smallest valid synthetic five-run receipt topology."""
    contract_input = root / "calibration_contract_source.yaml"
    contract_input.write_text("synthetic: true\n")
    contract = build_receipt(
        schema="matched-cancer-stage-provenance/v1",
        study_id=STUDY,
        scenario=SCENARIO,
        identities={"contract_source": file_identity(contract_input)},
        fields={
            "status": "valid",
            "representation_seed": 32001,
            "replay_presentations": 99_968,
        },
    )
    contract_file = atomic_write_receipt(
        root / "CALIBRATION_CONTRACT_RECEIPT.json", contract
    )
    replay_file = root / "CALIBRATION_REPLAY_MANIFEST.json"
    replay_body = {
        "schema": "matched-cancer-replay-manifest/v1",
        "contract": {
            "cancer_ids": [2, 15],
            "race_ids": [2, 4],
            "batch_size": 128,
            "steps": 781,
            "seed": 52001,
        },
        "occurrences": [],
        "traces": {
            "patient_sha256": "a" * 64,
            "tile_sha256": "b" * 64,
            "augmentation_seed_sha256": "c" * 64,
        },
    }
    replay_sha = hashlib.sha256(
        canonical_json_bytes(replay_body)
    ).hexdigest()
    replay_manifest = {
        **replay_body, "manifest_payload_sha256": replay_sha
    }
    replay_file.write_bytes(canonical_json_bytes(replay_manifest) + b"\n")
    encoder_initial = "1" * 64
    encoder_plain = "2" * 64
    encoder_fair = "3" * 64
    completions: dict[str, Path] = {}
    for run, fair_weight, encoder_post in (
        ("slot1_plain", 0.0, encoder_plain),
        ("slot1_fair", 0.1, encoder_fair),
    ):
        run_dir = root / run
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "latest.pt"
        checkpoint.write_bytes(f"synthetic checkpoint {run}".encode())
        completion = build_receipt(
            schema="matched-cancer-stage-completion/v1",
            study_id=STUDY,
            scenario=SCENARIO,
            identities={"latest_checkpoint": file_identity(checkpoint)},
            fields={
                "status": "complete",
                "run_name": f"matched-cancer-calibration-seed32001-{run}",
                "mode": "joint",
                "fair_weight": fair_weight,
                "steps_completed": 781,
                "tile_presentations": 99_968,
                "encoder_pre_sha256": encoder_initial,
                "encoder_post_sha256": encoder_post,
                "replay_manifest_payload_sha256": replay_sha,
            },
        )
        completions[run] = atomic_write_receipt(
            run_dir / "COMPLETION_RECEIPT.json", completion
        )

    parent_for = {"B": "slot1_plain", "H": "slot1_plain", "P": "slot1_fair"}
    for arm, fair_weight in {"B": 0.0, "H": 0.1, "P": 0.0}.items():
        parent_name = parent_for[arm]
        parent = json.loads(completions[parent_name].read_text())
        encoder_sha = parent["encoder_post_sha256"]
        run_dir = root / arm
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "latest.pt"
        checkpoint.write_bytes(f"synthetic checkpoint {arm}".encode())
        effective_config = run_dir / "effective.yaml"
        effective_config.write_text("synthetic: true\n")
        effective_receipt = build_receipt(
            schema="matched-cancer-stage-effective-config/v1",
            study_id=STUDY,
            scenario=SCENARIO,
            identities={
                "effective_config": file_identity(effective_config),
                "parent_completion_receipt": file_identity(
                    completions[parent_name]
                ),
                "encoder_checkpoint": parent["identities"][
                    "latest_checkpoint"
                ],
            },
            fields={
                "mode": "adapter_only",
                "fair_weight": fair_weight,
                "expected_encoder_state_sha256": encoder_sha,
            },
        )
        effective_path = atomic_write_receipt(
            run_dir / "effective.yaml.receipt.json", effective_receipt
        )
        completion = build_receipt(
            schema="matched-cancer-stage-completion/v1",
            study_id=STUDY,
            scenario=SCENARIO,
            identities={
                "effective_config_receipt": file_identity(effective_path),
                "latest_checkpoint": file_identity(checkpoint),
            },
            fields={
                "status": "complete",
                "run_name": f"matched-cancer-calibration-seed32001-{arm}",
                "mode": "adapter_only",
                "fair_weight": fair_weight,
                "steps_completed": 781,
                "tile_presentations": 99_968,
                "encoder_pre_sha256": encoder_sha,
                "encoder_post_sha256": encoder_sha,
                "replay_manifest_payload_sha256": replay_sha,
            },
        )
        completions[arm] = atomic_write_receipt(
            run_dir / "COMPLETION_RECEIPT.json", completion
        )

    identities = {
        "contract_receipt": file_identity(contract_file),
        "replay_manifest": file_identity(replay_file),
        "runs": {
            run: file_identity(completions[run])
            for run in ("slot1_plain", "slot1_fair", "B", "H", "P")
        }
    }
    receipt = build_receipt(
        schema=ROOT_SCHEMA,
        study_id=STUDY,
        scenario=SCENARIO,
        identities=identities,
        fields={
            "status": "matched_cancer_two_slot_calibration_seed32001_valid",
            "representation_seed": 32001,
            "steps_per_run": 781,
            "presentations_per_run": 99_968,
            "arms": ["slot1_plain", "slot1_fair", "B", "H", "P"],
            "replay_manifest_payload_sha256": replay_sha,
        },
    )
    root_receipt = atomic_write_receipt(
        root / "CALIBRATION_ROOT_COMPLETION_RECEIPT.json", receipt
    )
    return root_receipt, {arm: completions[arm] for arm in ARMS}


class CalibrationAncestryTests(unittest.TestCase):
    def test_exact_seed_and_bph_ancestry_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, completions = make_calibration_tree(Path(temporary))
            result = verify_calibration_ancestry(
                root, completions, expected_representation_seed=32001
            )
            self.assertEqual(set(result["arms"]), set(ARMS))
            self.assertEqual(result["root"]["representation_seed"], 32001)

    def test_wrong_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, completions = make_calibration_tree(Path(temporary))
            with self.assertRaisesRegex(ValueError, "representation_seed"):
                verify_calibration_ancestry(
                    root, completions, expected_representation_seed=32002
                )

    def test_swapped_arm_receipt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, completions = make_calibration_tree(Path(temporary))
            swapped = dict(completions)
            swapped["B"], swapped["P"] = swapped["P"], swapped["B"]
            with self.assertRaisesRegex(ValueError, "not root-bound"):
                verify_calibration_ancestry(
                    root, swapped, expected_representation_seed=32001
                )

    def test_checkpoint_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, completions = make_calibration_tree(Path(temporary))
            (Path(temporary) / "H" / "latest.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_calibration_ancestry(
                    root, completions, expected_representation_seed=32001
                )


if __name__ == "__main__":
    unittest.main()
