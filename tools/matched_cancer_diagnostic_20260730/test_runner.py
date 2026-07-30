#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from tools.matched_cancer_diagnostic_20260730.cache import (
    cached_adapter_embeddings,
)
from tools.matched_cancer_diagnostic_20260730.runner import (
    HEAD_SEEDS,
    PatientRecord,
    TaskProbe,
    assert_task_only_api,
    load_frozen_representation,
    nested_crossfit_predictions,
    run_paired_diagnostic,
    train_task_probe,
)
from tools.matched_cancer_stage_20260730.objectives import StageAdapter
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
)
from tools.matched_cancer_stage_20260730.completion_receipt import (
    state_dict_sha256,
)
from tools import reliable_fairness_head as reliable


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 384)

    def forward(self, x):
        return self.linear(x)

    def probe_features(self, x):
        return self.forward(x)


class DiagnosticRunnerTest(unittest.TestCase):
    def _bundle(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(7)
        encoder = TinyEncoder()
        adapter = StageAdapter(init_seed=72001)
        checkpoint = root / "latest.pt"
        torch.save(
            {
                "model": encoder.state_dict(),
                "stage_adapter": adapter.state_dict(),
                "config": {"data": {"mean": [0, 0, 0], "std": [1, 1, 1]}},
            },
            checkpoint,
        )
        receipt = build_receipt(
            schema="matched-cancer-stage-completion/v1",
            study_id="synthetic-study",
            scenario="synthetic-scenario",
            identities={"latest_checkpoint": file_identity(checkpoint)},
            fields={
                "encoder_post_sha256": state_dict_sha256(encoder.state_dict()),
                "adapter_post_sha256": state_dict_sha256(adapter.state_dict()),
            },
        )
        receipt_path = atomic_write_receipt(root / "completion.json", receipt)
        return load_frozen_representation(
            receipt_path,
            encoder_factory=lambda _: TinyEncoder(),
            expected_study_id="synthetic-study",
            expected_scenario="synthetic-scenario",
        )

    def test_exact_probe_and_task_only_api(self) -> None:
        probe = TaskProbe()
        self.assertEqual((probe.linear1.in_features, probe.linear1.out_features),
                         (128, 64))
        self.assertIsInstance(probe.relu, nn.ReLU)
        self.assertEqual(probe.dropout.p, 0.1)
        self.assertEqual((probe.linear2.in_features, probe.linear2.out_features),
                         (64, 1))
        self.assertEqual(probe(torch.randn(3, 128)).shape, (3,))
        assert_task_only_api()
        self.assertEqual(HEAD_SEEDS, (42001, 42002, 42003, 42004))
        rng = np.random.default_rng(12)
        x = rng.normal(size=(12, 128)).astype(np.float32)
        y = np.asarray([0, 1] * 6, dtype=np.float32)
        first = train_task_probe(x, y, x[:3], seed=42001, epochs=2)
        second = train_task_probe(x, y, x[:3], seed=42001, epochs=2)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())

    def test_completion_bound_bundle_is_frozen_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._bundle(root)
            self.assertTrue(all(
                not parameter.requires_grad
                for module in (bundle.encoder, bundle.adapter)
                for parameter in module.parameters()
            ))
            bundle.assert_unchanged()
            with torch.no_grad():
                bundle.adapter.lin1.weight[0, 0] += 1
            with self.assertRaisesRegex(RuntimeError, "adapter"):
                bundle.assert_unchanged()

    def test_normalized_cache_round_trip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = {"count": 0}
            tiles = [("P1", b"a"), ("P2", b"b")]

            def embed(_):
                calls["count"] += 1
                values = np.zeros((2, 128), dtype=np.float32)
                values[:, 0] = 1
                return values, np.ones(2, dtype=bool)

            kwargs = dict(
                tag="B",
                tiles=tiles,
                embed_fn=embed,
                cache_dir=root,
                source_identity={"encoder": "a" * 64, "adapter": "b" * 64},
            )
            first = cached_adapter_embeddings(**kwargs)
            second = cached_adapter_embeddings(**kwargs)
            self.assertEqual(calls["count"], 1)
            np.testing.assert_array_equal(first[0], second[0])
            with np.load(first[2], allow_pickle=False) as stored:
                arrays = {name: np.array(stored[name]) for name in stored.files}
            arrays["emb"][0, 1] = 0.25
            np.savez(first[2], **arrays)
            with self.assertRaises(reliable.CacheIntegrityError):
                cached_adapter_embeddings(**kwargs)

    def test_nested_patient_rows_exact_topology_and_race_is_metadata_only(self):
        patients = [
            PatientRecord(
                patient_id=f"TCGA-{fold:02d}-{index:04d}",
                y_true=index % 2,
                race="Black" if index % 2 else "White",
                tss=f"{fold:02d}",
                outer_fold=fold,
            )
            for fold in range(5)
            for index in range(4)
        ]
        rng = np.random.default_rng(3)
        features = rng.normal(size=(len(patients), 128)).astype(np.float32)
        calls = []

        def fit(x_train, y_train, x_eval, *, seed, epochs):
            calls.append((len(x_train), len(x_eval), seed))
            # No sensitive value is available here.
            return 1 / (1 + np.exp(-x_eval[:, 0]))

        rows, audit = nested_crossfit_predictions(
            features, patients, head_seed=42001, epochs=1, fit_fn=fit
        )
        summary = reliable._validate_nested_prediction_records(rows)
        reliable._validate_nested_training_audit(audit)
        self.assertEqual(len(calls), 25)
        self.assertEqual(summary["record_count"], 5 * len(patients))
        self.assertEqual(summary["role_counts"], {
            "outer_test": len(patients),
            "inner_calibration": 4 * len(patients),
        })
        self.assertEqual(
            sum(row["prediction_role"] == "outer_test" for row in rows),
            len(patients),
        )
        with self.assertRaisesRegex(ValueError, "head_seed"):
            nested_crossfit_predictions(
                features, patients, head_seed=1, epochs=1, fit_fn=fit
            )

    def test_full_bph_head_seed_matrix_is_paired_and_receipt_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = {
                arm: self._bundle(root / arm)
                for arm in ("B", "P", "H")
            }
            cohort = root / "synthetic_cohort.csv"
            cohort.write_text("synthetic,task-only\n")
            patients = [
                PatientRecord(
                    patient_id=f"TCGA-{fold:02d}-{index:04d}",
                    y_true=index % 2,
                    race="Black" if index % 2 else "White",
                    tss=f"{fold:02d}",
                    outer_fold=fold,
                )
                for fold in range(5)
                for index in range(2)
            ]
            tiles = [
                (patient.patient_id, patient.patient_id.encode())
                for patient in patients
            ]

            def embed(_representation, rows):
                values = np.zeros((len(rows), 128), dtype=np.float32)
                for index in range(len(rows)):
                    values[index, index % 128] = 1
                return values, np.ones(len(rows), dtype=bool)

            def fit(_x_train, _y_train, x_eval, *, seed, epochs):
                return np.full(len(x_eval), (seed % 997) / 997)

            receipt_path = run_paired_diagnostic(
                representations=bundles,
                tiles=tiles,
                patients=patients,
                task_id="synthetic-task",
                cohort_source=cohort,
                output_root=root / "diagnostic",
                cache_dir=root / "cache",
                epochs=1,
                fit_fn=fit,
                embed_fns={arm: embed for arm in ("B", "P", "H")},
            )
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["cell_count"], 12)
            self.assertEqual(receipt["head_seeds"], list(HEAD_SEEDS))
            for arm in ("B", "P", "H"):
                for seed in HEAD_SEEDS:
                    cell = (
                        root / "diagnostic" / arm / f"head_seed_{seed}"
                    )
                    self.assertTrue((cell / "DIAGNOSTIC_RECEIPT.json").is_file())


if __name__ == "__main__":
    unittest.main()
