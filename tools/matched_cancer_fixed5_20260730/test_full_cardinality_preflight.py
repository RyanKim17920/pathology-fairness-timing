from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import finalizer
from . import full_cardinality_preflight as preflight


class FullCardinalityPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_matrix_analyzer_verifier_and_finalizer_topology(self) -> None:
        output = self.root / "run"
        durable = self.root / "durable-receipt.json"
        receipt = preflight.run(
            output,
            receipt_destination=durable,
        )
        self.assertEqual(receipt, durable.resolve())
        value = preflight.verify_preflight_receipt(receipt)
        self.assertEqual(value["row_count"], 182_700)
        self.assertEqual(value["combination_count"], 120)
        self.assertEqual(value["nested_audit_count"], 2_250)
        self.assertEqual(
            value["expected_classification"],
            "small_across_five_tested_seeds",
        )
        self.assertEqual(value["analyzer_invocation_count"], 1)
        self.assertEqual(
            value["independent_verifier_invocation_count"], 1
        )
        self.assertIs(value["independent_verifier_match"], True)
        self.assertIs(value["synthetic_only"], True)
        self.assertIs(value["scientific_values_opened"], False)
        attempt = (
            output / "synthetic_production/finalization/attempt_01"
        )
        self.assertEqual(
            {path.name for path in attempt.iterdir()},
            set(value["finalization_artifact_names"]),
        )
        analysis = json.loads(
            (attempt / finalizer.ANALYSIS_NAME).read_text()
        )
        verification = json.loads(
            (attempt / finalizer.VERIFICATION_NAME).read_text()
        )
        self.assertEqual(
            analysis["semantic_report"],
            verification["semantic_report"],
        )

    def test_api_cannot_accept_real_predictions_or_controls(self) -> None:
        parameters = set(inspect.signature(preflight.run).parameters)
        self.assertEqual(
            parameters,
            {"output_root", "receipt_destination", "finalizer_runner"},
        )
        forbidden = {
            "predictions",
            "authorization_manifest",
            "source_manifest",
            "production_root",
            "outcome",
        }
        self.assertFalse(parameters & forbidden)

    def test_existing_or_symlink_output_is_never_reused(self) -> None:
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            preflight.run(existing)
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            preflight.run(link)

    def test_synthetic_output_inside_production_root_is_rejected(self) -> None:
        production = self.root / "production"
        production.mkdir()
        with mock.patch.object(
            preflight, "REAL_PRODUCTION_ROOT", production
        ):
            with self.assertRaisesRegex(ValueError, "production root"):
                preflight._canonical_new_root(production / "synthetic")

    def test_prediction_destination_symlink_is_never_followed(self) -> None:
        target = self.root / "target.jsonl"
        target.write_text("preserve\n")
        link = self.root / "predictions.jsonl"
        link.symlink_to(target)
        with self.assertRaises(FileExistsError):
            preflight.write_predictions(link)
        self.assertEqual(target.read_text(), "preserve\n")


if __name__ == "__main__":
    unittest.main()
