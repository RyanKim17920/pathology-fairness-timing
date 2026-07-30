"""Synthetic, outcome-blind tests for the independent verifier."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).with_name("verifier.py")
SPEC = importlib.util.spec_from_file_location("matched_cancer_verifier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def metric_rows(head_offsets: dict[int, float] | None = None) -> list[dict]:
    """Create a small complete nested cohort for metric-only tests."""
    offsets = head_offsets or {42001: 0.0}
    rows = []
    for head, offset in offsets.items():
        for fold in range(5):
            for race in ("White", "Black"):
                for label in (0, 1):
                    patient = f"{fold}-{race}-{label}"
                    base = 0.18 + 0.60 * label
                    if race == "Black":
                        base += 0.04
                    probability = min(0.99, max(0.01, base + offset))
                    rows.append({
                        "patient_id": patient,
                        "y_true": label,
                        "race": race,
                        "fold": fold,
                        "role": "outer_test",
                        "outer_fold": fold,
                        "probability": probability,
                        "head_seed": head,
                    })
                    for outer in range(5):
                        if outer != fold:
                            rows.append({
                                "patient_id": patient,
                                "y_true": label,
                                "race": race,
                                "fold": fold,
                                "role": "inner_calibration",
                                "outer_fold": outer,
                                "probability": probability,
                                "head_seed": head,
                            })
    return rows


class RowContractTests(unittest.TestCase):
    def base(self) -> dict:
        return {
            "schema": verifier.ROW_SCHEMA,
            "fm_seed": 32001,
            "arm": "B",
            "cancer": "BRCA",
            "head_seed": 42001,
            "patient_id": "synthetic-patient",
            "y_true": 0,
            "race": "White",
            "fold": 0,
            "role": "outer_test",
            "outer_fold": 0,
            "inner_fold": None,
            "probability": 0.25,
        }

    def test_exact_fields_and_nested_role_rules(self) -> None:
        verifier._validate_row(self.base(), 1)
        bad = self.base()
        bad["unexpected"] = True
        with self.assertRaisesRegex(verifier.VerificationError, "fields differ"):
            verifier._validate_row(bad, 1)
        bad = self.base()
        bad.update({
            "role": "inner_calibration", "outer_fold": 0, "inner_fold": 0,
        })
        with self.assertRaisesRegex(
            verifier.VerificationError, "inner-calibration"
        ):
            verifier._validate_row(bad, 1)

    def test_probability_must_be_finite_unit_interval(self) -> None:
        for value in (-0.01, 1.01, math.nan, math.inf, True):
            row = self.base()
            row["probability"] = value
            with self.assertRaisesRegex(
                verifier.VerificationError, "probability"
            ):
                verifier._validate_row(row, 1)


class MetricTests(unittest.TestCase):
    def test_nested_ieo_and_utility_are_bounded(self) -> None:
        rows = metric_rows()
        value, audit = verifier.nested_ieo_with_audit(rows)
        utility = verifier.utility_metrics(rows)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertEqual(len(audit), 75)
        self.assertEqual(audit[0]["specificity_target"], 0.60)
        self.assertEqual(audit[0]["outer_fold"], 0)
        self.assertEqual(
            audit[0]["calibration_white_negative_count"], 4
        )
        self.assertEqual(audit[0]["heldout_white_negative_count"], 1)
        self.assertGreaterEqual(
            audit[0]["achieved_heldout_white_specificity"], 0.0
        )
        self.assertLessEqual(
            audit[0]["achieved_heldout_white_specificity"], 1.0
        )
        self.assertEqual(set(utility), {
            "overall_auroc", "black_auprc", "black_ece",
        })
        self.assertTrue(all(0.0 <= result <= 1.0 for result in utility.values()))

    def test_head_probabilities_are_averaged_before_thresholding(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute("""
            CREATE TABLE predictions (
                fm_seed INTEGER, arm TEXT, cancer TEXT, head_seed INTEGER,
                patient_id TEXT, y_true INTEGER, race TEXT, fold INTEGER,
                role TEXT, outer_fold INTEGER, inner_fold INTEGER,
                probability REAL
            )
        """)
        raw = metric_rows({42001: -0.10, 42002: 0.10})
        connection.executemany(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    32001, "P", "BRCA", row["head_seed"], row["patient_id"],
                    row["y_true"], row["race"], row["fold"], row["role"],
                    row["outer_fold"],
                    None if row["role"] == "outer_test" else row["fold"],
                    row["probability"],
                )
                for row in raw
            ],
        )
        averaged = verifier.ensemble_rows(
            connection, 32001, "P", "BRCA", (42001, 42002)
        )
        by_key = {
            (row["patient_id"], row["role"], row["outer_fold"]): row
            for row in averaged
        }
        target = by_key[("0-White-0", "outer_test", 0)]
        self.assertAlmostEqual(target["probability"], 0.18, places=15)
        connection.close()


class InferenceTests(unittest.TestCase):
    def test_paired_t_and_strict_equivalence_boundary(self) -> None:
        summary = verifier.paired_summary([0.0] * 48)
        self.assertEqual(summary["two_sided_p"], 1.0)
        self.assertEqual(summary["ci90"], [0.0, 0.0])
        self.assertTrue(
            summary["ci90"][0] > -0.03 and summary["ci90"][1] < 0.03
        )
        boundary = verifier.paired_summary([0.03] * 48)
        self.assertFalse(
            boundary["ci90"][0] > -0.03 and boundary["ci90"][1] < 0.03
        )

    def test_positive_theta_convention(self) -> None:
        summary = verifier.paired_summary([0.025] * 48)
        favored = "H" if summary["mean"] > 0 else "P"
        self.assertEqual(favored, "H")
        self.assertLess(summary["two_sided_p"], 0.05)

    def test_utility_gate_includes_relaxed_per_cancer_bounds(self) -> None:
        utility = {
            arm: {
                "overall": {
                    "overall_auroc": 0.70,
                    "black_auprc": 0.60,
                    "black_ece": 0.10,
                },
                "cancers": {
                    cancer: {
                        "overall_auroc": 0.70,
                        "black_auprc": 0.60,
                        "black_ece": 0.10,
                    } for cancer in verifier.CANCERS
                },
            } for arm in verifier.ARMS
        }
        utility["H"]["deltas_vs_B"] = {
            "overall": {
                "overall_auroc": 0.0,
                "black_auprc": 0.0,
                "black_ece": 0.0,
            },
            "cancers": {
                cancer: {
                    "overall_auroc": 0.0,
                    "black_auprc": 0.0,
                    "black_ece": 0.0,
                } for cancer in verifier.CANCERS
            },
        }
        self.assertTrue(verifier.utility_gate(utility, "H")["pass"])
        utility["H"]["deltas_vs_B"]["cancers"]["BRCA"][
            "black_ece"
        ] = np.nextafter(0.05, math.inf)
        self.assertFalse(verifier.utility_gate(utility, "H")["pass"])


class ComparisonTests(unittest.TestCase):
    def minimal_validation_database(
        self, path: Path,
    ) -> sqlite3.Connection:
        connection = verifier._create_db(path)
        insert = "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
        rows = []
        for seed in verifier.FM_SEEDS:
            for arm in verifier.ARMS:
                for cancer in verifier.CANCERS:
                    for head in verifier.HEADS:
                        common = (
                            seed, arm, cancer, head, "shared-patient",
                            0, "White", 0,
                        )
                        rows.append((*common, "outer_test", 0, None, 0.2))
                        for outer in (1, 2, 3, 4):
                            rows.append((
                                *common, "inner_calibration", outer, 0, 0.2,
                            ))
        connection.executemany(insert, rows)
        connection.commit()
        return connection

    def test_complete_semantic_comparison_and_sha(self) -> None:
        expected = {
            "input_sha256": "abc",
            "row_count": 123,
            "nested": {"value": 0.2, "flags": [True, "H"]},
        }
        verifier._semantic_equal(expected, json.loads(json.dumps(expected)))
        altered = json.loads(json.dumps(expected))
        altered["input_sha256"] = "def"
        with self.assertRaisesRegex(verifier.VerificationError, "mismatch"):
            verifier._semantic_equal(expected, altered)
        incomplete = {"input_sha256": "abc", "row_count": 123}
        with self.assertRaisesRegex(verifier.VerificationError, "key mismatch"):
            verifier._semantic_equal(expected, incomplete)

    def test_output_guard_rejects_symlink_and_sealed_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            analyzer = root / "analyzer.json"
            predictions.write_text("sealed predictions\n", encoding="utf-8")
            analyzer.write_text("{}\n", encoding="utf-8")
            sealed = (predictions, analyzer)

            with self.assertRaisesRegex(
                verifier.VerificationError, "collides with sealed input"
            ):
                verifier.validate_output_path(predictions, sealed)
            with self.assertRaisesRegex(
                verifier.VerificationError, "collides with sealed input"
            ):
                verifier.validate_output_path(
                    root / "." / "analyzer.json", sealed
                )

            hardlink = root / "hardlink-output.json"
            os.link(predictions, hardlink)
            with self.assertRaisesRegex(
                verifier.VerificationError, "aliases sealed input"
            ):
                verifier.validate_output_path(hardlink, sealed)

            report_target = root / "report.json"
            report_target.write_text("existing report\n", encoding="utf-8")
            symlink = root / "report-link.json"
            symlink.symlink_to(report_target)
            with self.assertRaisesRegex(
                verifier.VerificationError, "must not be a symlink"
            ):
                verifier.validate_output_path(symlink, sealed)
            self.assertEqual(
                report_target.read_text(encoding="utf-8"),
                "existing report\n",
            )

    def test_cli_rejects_input_as_output_before_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            predictions.write_text("sealed\n", encoding="utf-8")
            stderr = io.StringIO()
            with (
                mock.patch.object(verifier, "verify") as verify,
                mock.patch.object(verifier.sys, "stderr", stderr),
            ):
                status = verifier.main([
                    str(predictions), "--output", str(predictions),
                ])
            self.assertEqual(status, 2)
            self.assertIn("collides with sealed input", stderr.getvalue())
            verify.assert_not_called()
            self.assertEqual(
                predictions.read_text(encoding="utf-8"), "sealed\n"
            )

    def test_duplicate_jsonl_semantic_row_fails(self) -> None:
        row = RowContractTests().base()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "rows.jsonl"
            source.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verifier.VerificationError, "duplicate semantic row"
            ):
                verifier.load_predictions(source, root / "rows.sqlite3")

    def test_cross_seed_patient_and_metadata_drift_fail(self) -> None:
        mutations = (
            "patient_id = 'seed-specific-patient'",
            "y_true = 1",
            "race = 'Black'",
            "fold = 1",
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, mutation in enumerate(mutations):
                connection = self.minimal_validation_database(
                    Path(directory) / f"drift-{index}.sqlite3"
                )
                connection.execute(
                    f"UPDATE predictions SET {mutation} "
                    "WHERE fm_seed = 32048 AND cancer = 'BRCA'"
                )
                connection.commit()
                with self.assertRaisesRegex(
                    verifier.VerificationError, "across FM seeds"
                ):
                    verifier.validate_complete(connection)
                connection.close()

    def test_exact_production_cohort_sizes_are_required(self) -> None:
        self.assertEqual(
            verifier.COHORT_SIZES, {"BRCA": 328, "LUAD": 281}
        )
        self.assertEqual(
            verifier.contract_report()["expected_row_count"], 1_753_920
        )
        counts = {
            f"{seed}:{cancer}": verifier.COHORT_SIZES[cancer]
            for seed in verifier.FM_SEEDS for cancer in verifier.CANCERS
        }
        verifier._validate_cohort_sizes(counts)
        for key, wrong_size in (
            ("32001:BRCA", 333),
            ("32048:LUAD", 282),
        ):
            altered = dict(counts)
            altered[key] = wrong_size
            with self.assertRaisesRegex(
                verifier.VerificationError, "fixed cohort size differs"
            ):
                verifier._validate_cohort_sizes(altered)

    def test_validate_complete_invokes_cohort_size_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = self.minimal_validation_database(
                Path(directory) / "wrong-size.sqlite3"
            )
            with mock.patch.object(
                verifier, "COHORT_SIZES", {"BRCA": 2, "LUAD": 1}
            ):
                with self.assertRaisesRegex(
                    verifier.VerificationError,
                    "fixed cohort size differs at 32001:BRCA",
                ):
                    verifier.validate_complete(connection)
            connection.close()

    def test_exact_fixed_final_synthetic_study(self) -> None:
        patients = tuple(
            (f"white-neg-{fold}", 0, "White", fold, 0.10 + 0.01 * fold)
            for fold in verifier.FOLDS
        ) + (
            ("white-pos", 1, "White", 2, 0.75),
            ("black-neg", 0, "Black", 3, 0.18),
            ("black-pos", 1, "Black", 4, 0.78),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixed-final.jsonl"
            with source.open("w", encoding="utf-8") as stream:
                for seed in verifier.FM_SEEDS:
                    for arm in verifier.ARMS:
                        for cancer in verifier.CANCERS:
                            for head in verifier.HEADS:
                                for patient, label, race, fold, probability in patients:
                                    common = {
                                        "schema": verifier.ROW_SCHEMA,
                                        "fm_seed": seed,
                                        "arm": arm,
                                        "cancer": cancer,
                                        "head_seed": head,
                                        "patient_id": patient,
                                        "y_true": label,
                                        "race": race,
                                        "fold": fold,
                                        "probability": probability,
                                    }
                                    stream.write(json.dumps({
                                        **common,
                                        "role": "outer_test",
                                        "outer_fold": fold,
                                        "inner_fold": None,
                                    }, sort_keys=True) + "\n")
                                    for outer in verifier.FOLDS:
                                        if outer != fold:
                                            stream.write(json.dumps({
                                                **common,
                                                "role": "inner_calibration",
                                                "outer_fold": outer,
                                                "inner_fold": fold,
                                            }, sort_keys=True) + "\n")
            # The production 328/281 cardinalities are tested independently
            # above; shrink only this end-to-end mechanics fixture.
            with mock.patch.object(
                verifier, "COHORT_SIZES", {"BRCA": 8, "LUAD": 8}
            ):
                report = verifier.verify(source)
        semantic = report["semantic_report"]
        self.assertEqual(semantic["counts"]["combination_count"], 1152)
        self.assertEqual(semantic["row_count"], 46080)
        self.assertEqual(
            semantic["decision"]["classification"], "equivalent"
        )
        self.assertEqual(semantic["full"]["per_seed_theta"], [0.0] * 48)
        # 48 seeds * 3 arms * 2 cancers * 15 targets * 5 folds.
        self.assertEqual(len(semantic["nested_audit"]), 21600)
        first = semantic["nested_audit"][0]
        self.assertEqual(set(first), {
            "fm_seed", "arm", "cancer", "specificity_target", "outer_fold",
            "threshold", "calibration_white_negative_count",
            "heldout_white_negative_count",
            "achieved_heldout_white_specificity",
        })
        self.assertEqual(
            (first["fm_seed"], first["arm"], first["cancer"]),
            (32001, "B", "BRCA"),
        )
        self.assertEqual(first["specificity_target"], 0.60)
        self.assertEqual(first["outer_fold"], 0)
        self.assertAlmostEqual(first["threshold"], 0.128, places=15)
        self.assertEqual(first["calibration_white_negative_count"], 4)
        self.assertEqual(first["heldout_white_negative_count"], 1)
        self.assertEqual(first["achieved_heldout_white_specificity"], 1.0)
        last = semantic["nested_audit"][-1]
        self.assertEqual(
            (
                last["fm_seed"], last["arm"], last["cancer"],
                last["specificity_target"], last["outer_fold"],
            ),
            (32048, "H", "LUAD", 0.95, 4),
        )
        self.assertFalse(report["analyzer_comparison"]["requested"])


if __name__ == "__main__":
    unittest.main()
