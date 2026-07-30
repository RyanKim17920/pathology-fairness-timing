"""Synthetic-only tests for the fixed-final analyzer and verifier agreement."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.matched_cancer_diagnostic_20260730 import analyzer, verifier


PATIENTS = (
    ("white-neg-0", 0, "White", 0, 0.15),
    ("white-neg-1", 0, "White", 1, 0.17),
    ("white-neg-2", 0, "White", 2, 0.19),
    ("white-neg-3", 0, "White", 3, 0.21),
    ("white-neg-4", 0, "White", 4, 0.23),
    ("white-pos", 1, "White", 2, 0.75),
    ("black-neg", 0, "Black", 3, 0.18),
    ("black-pos", 1, "Black", 4, 0.78),
)


def write_fixture(path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for seed in analyzer.FM_SEEDS:
            for arm in analyzer.ARMS:
                for cancer in analyzer.CANCERS:
                    for head in analyzer.HEADS:
                        for patient, label, race, fold, probability in PATIENTS:
                            common = {
                                "schema": analyzer.ROW_SCHEMA,
                                "fm_seed": seed,
                                "arm": arm,
                                "cancer": cancer,
                                "head_seed": head,
                                "patient_id": f"{cancer}-{patient}",
                                "y_true": label,
                                "race": race,
                                "fold": fold,
                                "probability": probability,
                            }
                            stream.write(json.dumps({
                                **common, "role": "outer_test",
                                "outer_fold": fold, "inner_fold": None,
                            }, sort_keys=True) + "\n")
                            for outer in analyzer.FOLDS:
                                if outer != fold:
                                    stream.write(json.dumps({
                                        **common, "role": "inner_calibration",
                                        "outer_fold": outer, "inner_fold": fold,
                                    }, sort_keys=True) + "\n")


class ContractTests(unittest.TestCase):
    def test_frozen_contract_constants(self) -> None:
        self.assertEqual(analyzer.FM_SEEDS, tuple(range(32001, 32049)))
        self.assertEqual(analyzer.COHORT_SIZES, {"BRCA": 328, "LUAD": 281})
        self.assertEqual(
            analyzer.contract_report()["expected_row_count"], 1_753_920
        )
        self.assertEqual(analyzer.HEAD_HALVES, ((42001, 42002), (42003, 42004)))
        self.assertEqual(analyzer.SPECIFICITIES[0], 0.60)
        self.assertEqual(analyzer.SPECIFICITIES[-1], 0.95)

    def test_paired_equivalence_is_strict_and_positive_favors_h(self) -> None:
        zero = analyzer.paired_summary([0.0] * 48)
        self.assertEqual(zero["ci90"], [0.0, 0.0])
        boundary = analyzer.paired_summary([0.03] * 48)
        self.assertFalse(
            boundary["ci90"][0] > -0.03 and boundary["ci90"][1] < 0.03
        )
        positive = analyzer.paired_summary([0.02] * 48)
        self.assertGreater(positive["mean"], 0)
        self.assertEqual(positive["two_sided_p"], 0.0)

    def test_wrong_cohort_size_fails_closed(self) -> None:
        patient = {
            "synthetic": {
                "y_true": 0, "race": "White", "fold": 0,
                "scores": {
                    ("outer_test", 0): 0.2,
                    **{
                        ("inner_calibration", outer): 0.2
                        for outer in (1, 2, 3, 4)
                    },
                },
            }
        }
        cells = {
            (seed, arm, cancer, head): patient
            for seed in analyzer.FM_SEEDS for arm in analyzer.ARMS
            for cancer in analyzer.CANCERS for head in analyzer.HEADS
        }
        with self.assertRaisesRegex(analyzer.AnalysisError, "cohort has"):
            analyzer.validate_complete(cells)

    def test_cross_seed_patient_drift_fails_closed(self) -> None:
        def patient(identifier: str) -> dict:
            return {
                identifier: {
                    "y_true": 0, "race": "White", "fold": 0,
                    "scores": {
                        ("outer_test", 0): 0.2,
                        **{
                            ("inner_calibration", outer): 0.2
                            for outer in (1, 2, 3, 4)
                        },
                    },
                }
            }
        cells = {
            (seed, arm, cancer, head): patient("same")
            for seed in analyzer.FM_SEEDS for arm in analyzer.ARMS
            for cancer in analyzer.CANCERS for head in analyzer.HEADS
        }
        for arm in analyzer.ARMS:
            for head in analyzer.HEADS:
                cells[(32048, arm, "BRCA", head)] = patient("changed")
        with (
            mock.patch.object(
                analyzer, "COHORT_SIZES", {"BRCA": 1, "LUAD": 1}
            ),
            self.assertRaisesRegex(analyzer.AnalysisError, "across FM seeds"),
        ):
            analyzer.validate_complete(cells)


class EndToEndSyntheticTests(unittest.TestCase):
    def test_analyzer_and_independent_verifier_agree_completely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "synthetic.jsonl"
            analysis_path = root / "analysis.json"
            write_fixture(predictions)
            synthetic_sizes = {"BRCA": len(PATIENTS), "LUAD": len(PATIENTS)}
            with (
                mock.patch.object(analyzer, "COHORT_SIZES", synthetic_sizes),
                mock.patch.object(verifier, "COHORT_SIZES", synthetic_sizes),
            ):
                report = analyzer.run(predictions)
                analysis_path.write_text(
                    json.dumps(report, sort_keys=True), encoding="utf-8"
                )
                checked = verifier.verify(predictions, analysis_path)
        self.assertEqual(
            report["semantic_report"]["decision"]["classification"],
            "equivalent",
        )
        self.assertTrue(checked["analyzer_comparison"]["match"])
        self.assertEqual(report["semantic_report"], checked["semantic_report"])

    def test_four_heads_are_averaged_before_nested_thresholds(self) -> None:
        cells = {}
        for head, offset in zip(analyzer.HEADS, (-0.09, -0.03, 0.03, 0.09)):
            patients = {}
            for patient, label, race, fold, probability in PATIENTS:
                score = min(1.0, max(0.0, probability + offset))
                scores = {("outer_test", fold): score}
                scores.update({
                    ("inner_calibration", outer): score
                    for outer in analyzer.FOLDS if outer != fold
                })
                patients[patient] = {
                    "y_true": label, "race": race, "fold": fold,
                    "scores": scores,
                }
            cells[(32001, "P", "BRCA", head)] = patients
        rows = analyzer.ensemble_rows(
            cells, 32001, "P", "BRCA", analyzer.HEADS
        )
        row = next(
            value for value in rows
            if value["patient_id"] == "white-neg-0"
            and value["role"] == "outer_test"
        )
        self.assertAlmostEqual(row["probability"], 0.15, places=15)


if __name__ == "__main__":
    unittest.main()
