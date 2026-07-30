"""Synthetic and tamper tests for the independent calibration auditor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools.matched_cancer_stage_20260730 import calibration_output_auditor as auditor


def write_canonical(path: Path, value) -> None:
    path.write_bytes(auditor.canonical_json_bytes(value) + b"\n")


def synthetic_manifest(*, steps: int = 2, batch_size: int = 4) -> dict:
    occurrences = []
    strata = list(auditor.STRATA)
    for batch in range(steps):
        for position, (cancer, race) in enumerate(strata):
            dataset_index = batch * batch_size + position
            patient = f"TCGA-AA-{dataset_index:04d}"
            tile_path = f"{patient}-01Z/example-{dataset_index}.jpg"
            occurrences.append(
                {
                    "batch": batch,
                    "position": position,
                    "dataset_index": dataset_index,
                    "shard_path": f"/synthetic/shard-{batch:05d}.parquet",
                    "shard_sha256": hashlib.sha256(
                        f"shard-{batch}".encode()
                    ).hexdigest(),
                    "row": position,
                    "tile_path": tile_path,
                    "tile_jpeg_sha256": hashlib.sha256(
                        f"jpeg-{dataset_index}".encode()
                    ).hexdigest(),
                    "patient": patient,
                    "cancer": cancer,
                    "race": race,
                    "augmentation_seed": 1000 + dataset_index,
                }
            )
    traces = {
        "patient_sha256": auditor._sequence_sha256(
            [row["patient"] for row in occurrences]
        ),
        "tile_sha256": auditor._sequence_sha256(
            [
                {
                    "shard_sha256": row["shard_sha256"],
                    "row": row["row"],
                    "tile_path": row["tile_path"],
                    "tile_jpeg_sha256": row["tile_jpeg_sha256"],
                }
                for row in occurrences
            ]
        ),
        "augmentation_seed_sha256": auditor._sequence_sha256(
            [row["augmentation_seed"] for row in occurrences]
        ),
    }
    body = {
        "schema": "matched-cancer-replay-manifest/v1",
        "contract": {
            "cancer_ids": list(auditor.CANCER_IDS),
            "race_ids": list(auditor.RACE_IDS),
            "batch_size": batch_size,
            "steps": steps,
            "seed": auditor.REPLAY_SEED,
        },
        "occurrences": occurrences,
        "traces": traces,
    }
    return {
        **body,
        "manifest_payload_sha256": hashlib.sha256(
            auditor.canonical_json_bytes(body)
        ).hexdigest(),
    }


def reseal_manifest(manifest: dict) -> None:
    occurrences = manifest["occurrences"]
    manifest["traces"] = {
        "patient_sha256": auditor._sequence_sha256(
            [row["patient"] for row in occurrences]
        ),
        "tile_sha256": auditor._sequence_sha256(
            [
                {
                    "shard_sha256": row["shard_sha256"],
                    "row": row["row"],
                    "tile_path": row["tile_path"],
                    "tile_jpeg_sha256": row["tile_jpeg_sha256"],
                }
                for row in occurrences
            ]
        ),
        "augmentation_seed_sha256": auditor._sequence_sha256(
            [row["augmentation_seed"] for row in occurrences]
        ),
    }
    body = {
        key: manifest[key]
        for key in ("schema", "contract", "occurrences", "traces")
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        auditor.canonical_json_bytes(body)
    ).hexdigest()


class ReceiptTests(unittest.TestCase):
    def test_independent_receipt_verification_and_bound_file_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = root / "bound.txt"
            bound.write_text("original\n")
            identities = {"input": auditor.file_identity(bound)}
            receipt = {
                "schema": "test-schema/v1",
                "study_id": auditor.STUDY_ID,
                "scenario": auditor.SCENARIO,
                "identities": identities,
                "topology_sha256": auditor.topology_sha256(identities),
            }
            receipt_path = root / "receipt.json"
            write_canonical(receipt_path, receipt)
            audit = auditor.Audit()
            auditor.verify_receipt(
                receipt_path,
                audit,
                schema="test-schema/v1",
                expected_identity_roles={"input"},
            )
            self.assertGreater(audit.checks, 0)
            bound.write_text("tampered\n")
            with self.assertRaisesRegex(auditor.AuditError, "bound file changed"):
                auditor.verify_receipt(
                    receipt_path,
                    auditor.Audit(),
                    schema="test-schema/v1",
                )

    def test_noncanonical_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"b": 2, "a": 1}\n')
            with self.assertRaisesRegex(auditor.AuditError, "not canonically"):
                auditor.load_canonical_json(path)

    def test_duplicate_receipt_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"a":1,"a":2}\n')
            with self.assertRaisesRegex(auditor.AuditError, "duplicate JSON key"):
                auditor.load_canonical_json(path)


class ManifestTests(unittest.TestCase):
    def test_synthetic_manifest_and_runtime_traces_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = synthetic_manifest()
            write_canonical(path, manifest)
            result = auditor.inspect_manifest(
                path, auditor.Audit(), steps=2, batch_size=4
            )
            self.assertEqual(result["payload_sha256"], manifest["manifest_payload_sha256"])
            self.assertEqual(len(result["unique_identities"]), 8)
            indices = [row["dataset_index"] for row in manifest["occurrences"]]
            self.assertEqual(
                result["sample_batch_trace_sha256"],
                auditor.runtime_batch_trace(indices, 4),
            )

    def test_payload_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = synthetic_manifest()
            manifest["occurrences"][0]["augmentation_seed"] += 1
            write_canonical(path, manifest)
            with self.assertRaisesRegex(auditor.AuditError, "payload digest"):
                auditor.inspect_manifest(
                    path, auditor.Audit(), steps=2, batch_size=4
                )

    def test_resealed_unbalanced_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = synthetic_manifest()
            manifest["occurrences"][0]["cancer"] = manifest["occurrences"][1][
                "cancer"
            ]
            manifest["occurrences"][0]["race"] = manifest["occurrences"][1]["race"]
            reseal_manifest(manifest)
            write_canonical(path, manifest)
            with self.assertRaisesRegex(auditor.AuditError, "not balanced"):
                auditor.inspect_manifest(
                    path, auditor.Audit(), steps=2, batch_size=4
                )

    def test_dataset_index_identity_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = synthetic_manifest()
            manifest["occurrences"][1]["dataset_index"] = 0
            reseal_manifest(manifest)
            write_canonical(path, manifest)
            with self.assertRaisesRegex(auditor.AuditError, "multiple identities"):
                auditor.inspect_manifest(
                    path, auditor.Audit(), steps=2, batch_size=4
                )


class MetricsAndStateTests(unittest.TestCase):
    def metric_rows(self) -> list[dict]:
        rows = []
        for step in (1, 2):
            rows.append(
                {
                    "step": step,
                    "wd": 0.04,
                    "teacher_temp": 0.04,
                    "teacher_momentum": 0.994,
                    "kde_scale": 0.0,
                    "batch_size": 4,
                    "examples_seen": step * 4,
                    "sample_fraction": step / 2,
                    "lr": auditor.ADAPTER_LR,
                    "matched_stage_mode": "adapter_only",
                    "cancer": 2.0,
                    "race_fair": 3.0,
                    "race_fair_weighted": 0.0,
                    "total": 2.0,
                    "h_dose_main_grad_norm": 0.2,
                    "h_dose_fair_grad_norm": 0.0,
                    "stage_adapter_grad_norm": 0.4,
                    "adapter_lr": auditor.ADAPTER_LR,
                    "adapter_weight_decay": auditor.ADAPTER_WEIGHT_DECAY,
                }
            )
        return rows

    def test_metrics_pass_and_nonfinite_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = self.metric_rows()
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            _, shared, mode = auditor.load_metrics(
                path, auditor.Audit(), run="B", steps=2, batch_size=4
            )
            self.assertTrue(auditor._valid_sha256(shared))
            self.assertTrue(auditor._valid_sha256(mode))
            bad = copy.deepcopy(rows)
            bad[1]["cancer"] = float("nan")
            path.write_text("\n".join(json.dumps(row) for row in bad) + "\n")
            with self.assertRaisesRegex(auditor.AuditError, "non-finite"):
                auditor.load_metrics(
                    path, auditor.Audit(), run="B", steps=2, batch_size=4
                )

    def test_weighted_fair_metric_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = self.metric_rows()
            rows[0]["race_fair_weighted"] = 0.1
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(auditor.AuditError, "weighted fair metric"):
                auditor.load_metrics(
                    path, auditor.Audit(), run="B", steps=2, batch_size=4
                )

    def test_state_dict_hash_is_order_independent_and_content_bound(self) -> None:
        left = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
        right = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        self.assertEqual(
            auditor.state_dict_sha256(left), auditor.state_dict_sha256(right)
        )
        right["b"][0] = 3.0
        self.assertNotEqual(
            auditor.state_dict_sha256(left), auditor.state_dict_sha256(right)
        )


class SafetyTests(unittest.TestCase):
    def test_downstream_token_is_found_recursively(self) -> None:
        self.assertEqual(
            auditor._structured_forbidden_hits(
                {"representation": {"target": "TP53 mutation"}}
            ),
            ["$.representation.target"],
        )

    def test_incomplete_job_is_refused_before_partial_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(auditor.AuditError, "job is incomplete"):
                auditor.audit_job_root(Path(directory))


if __name__ == "__main__":
    unittest.main()
