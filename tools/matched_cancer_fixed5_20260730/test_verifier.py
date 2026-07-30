from __future__ import annotations

import ast
import builtins
import copy
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from scipy import stats

from . import verifier


FIXED5_PACKAGE = "tools.matched_cancer_fixed5_20260730"
FORBIDDEN_ANALYZER = f"{FIXED5_PACKAGE}.analyzer"


def _absolute_from_target(
    module: str | None, level: int, package: str,
) -> str:
    if level:
        return importlib.util.resolve_name(
            "." * level + (module or ""), package
        )
    return module or ""


def _ast_import_targets(source: str, package: str) -> set[str]:
    targets: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_from_target(node.module, node.level, package)
            if base:
                targets.add(base)
            for alias in node.names:
                if alias.name != "*":
                    targets.add(
                        f"{base}.{alias.name}" if base else alias.name
                    )
    return targets


def _runtime_import_targets(
    name: str,
    globals_value: dict | None,
    fromlist: tuple | list,
    level: int,
) -> set[str]:
    package = (
        str(globals_value.get("__package__", ""))
        if globals_value is not None
        else ""
    )
    base = _absolute_from_target(name, level, package) if level else name
    targets = {base} if base else set()
    targets.update(
        f"{base}.{item}" if base else str(item)
        for item in fromlist
        if item != "*"
    )
    return targets


class Fixed5VerifierTests(unittest.TestCase):
    def _large_inputs(
        self,
    ) -> tuple[dict, list[dict], dict[str, dict]]:
        full = verifier.paired_summary([0.03] * 5)
        halves = [
            {"mean": 0.03, "head_seeds": list(heads)}
            for heads in verifier.HEAD_HALVES
        ]
        cancers = {
            cancer: {"mean": 0.03} for cancer in verifier.CANCERS
        }
        return full, halves, cancers

    def _small_inputs(
        self,
    ) -> tuple[dict, list[dict], dict[str, dict]]:
        full = verifier.paired_summary([0.005] * 5)
        halves = [
            {"mean": 0.005, "head_seeds": list(heads)}
            for heads in verifier.HEAD_HALVES
        ]
        cancers = {
            cancer: {"mean": 0.005} for cancer in verifier.CANCERS
        }
        return full, halves, cancers

    def _screen(
        self,
        full: dict,
        halves: list[dict],
        cancers: dict[str, dict],
        *,
        harm: bool = True,
        utility: bool = True,
    ) -> dict:
        return verifier.screen_decision(
            full,
            halves,
            cancers,
            harm_pass=harm,
            utility_pass=utility,
        )

    def _analyze_subtracted_effect(
        self, p_endpoint: float, h_endpoint: float,
    ) -> dict:
        baseline = h_endpoint if p_endpoint >= h_endpoint else p_endpoint

        def ensemble(
            connection: object,
            seed: int,
            arm: str,
            cancer: str,
            heads: tuple[int, ...],
        ) -> tuple[int, str, str, tuple[int, ...]]:
            return seed, arm, cancer, tuple(heads)

        def value(marker: tuple) -> float:
            _, arm, _, _ = marker
            if arm == "P":
                return p_endpoint
            if arm == "H":
                return h_endpoint
            return baseline

        def endpoint(marker: tuple) -> tuple[float, list[dict]]:
            return value(marker), [
                {"index": index} for index in range(75)
            ]

        with (
            mock.patch.object(verifier, "ensemble_rows", side_effect=ensemble),
            mock.patch.object(
                verifier, "nested_ieo_with_audit", side_effect=endpoint
            ),
            mock.patch.object(verifier, "nested_ieo", side_effect=value),
            mock.patch.object(
                verifier,
                "utility_metrics",
                return_value={
                    "overall_auroc": 0.80,
                    "black_auprc": 0.50,
                    "black_ece": 0.10,
                },
            ),
        ):
            return verifier.analyze(mock.sentinel.connection)

    def _utility_values(
        self,
        *,
        b_cancers: dict[str, dict[str, float]] | None = None,
        h_cancers: dict[str, dict[str, float]] | None = None,
    ) -> dict[tuple[int, str, str], dict[str, float]]:
        defaults = {
            cancer: {
                "overall_auroc": 0.80,
                "black_auprc": 0.50,
                "black_ece": 0.10,
            }
            for cancer in verifier.CANCERS
        }
        b = b_cancers or defaults
        h = h_cancers or defaults
        values = {}
        for seed in verifier.FM_SEEDS:
            for arm in verifier.ARMS:
                for cancer in verifier.CANCERS:
                    source = h if arm == "H" else b
                    values[(seed, arm, cancer)] = dict(source[cancer])
        return values

    def _tiny_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(":memory:")
        connection.execute("""
            CREATE TABLE predictions (
                fm_seed INTEGER NOT NULL,
                arm TEXT NOT NULL,
                cancer TEXT NOT NULL,
                head_seed INTEGER NOT NULL,
                patient_id TEXT NOT NULL,
                y_true INTEGER NOT NULL,
                race TEXT NOT NULL,
                fold INTEGER NOT NULL,
                role TEXT NOT NULL,
                outer_fold INTEGER NOT NULL,
                inner_fold INTEGER,
                probability REAL NOT NULL,
                PRIMARY KEY (
                    fm_seed, arm, cancer, head_seed, patient_id, role,
                    outer_fold
                )
            ) WITHOUT ROWID
        """)
        rows = []
        for seed in verifier.FM_SEEDS:
            for arm in verifier.ARMS:
                for cancer in verifier.CANCERS:
                    for head in verifier.HEADS:
                        patient = f"{cancer}-patient"
                        rows.append(
                            (
                                seed, arm, cancer, head, patient, 0, "White",
                                0, "outer_test", 0, None, 0.1,
                            )
                        )
                        for outer in range(1, 5):
                            rows.append(
                                (
                                    seed, arm, cancer, head, patient, 0,
                                    "White", 0, "inner_calibration", outer,
                                    0, 0.1,
                                )
                            )
        connection.executemany(
            "INSERT INTO predictions VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        return connection

    def test_known_five_seed_vector_has_df4_intervals_and_exact_sign(self) -> None:
        vector = [0.01, 0.02, 0.03, 0.04, 0.05]
        summary = verifier.paired_summary(vector)
        expected_mean = 0.03
        expected_sd = math.sqrt(0.001 / 4)
        expected_se = expected_sd / math.sqrt(5)
        self.assertEqual(summary["df"], 4)
        self.assertAlmostEqual(summary["mean"], expected_mean, places=15)
        self.assertAlmostEqual(summary["median"], 0.03, places=15)
        self.assertAlmostEqual(summary["sd"], expected_sd, places=15)
        self.assertAlmostEqual(summary["se"], expected_se, places=15)
        self.assertEqual(summary["minimum"], 0.01)
        self.assertEqual(summary["maximum"], 0.05)
        for observed, expected in zip(
            summary["leave_one_seed_out_means"],
            [0.035, 0.0325, 0.03, 0.0275, 0.025],
            strict=True,
        ):
            self.assertAlmostEqual(observed, expected, places=15)
        self.assertAlmostEqual(
            summary["ci90"][0],
            expected_mean - stats.t.ppf(0.95, 4) * expected_se,
            places=15,
        )
        self.assertAlmostEqual(
            summary["ci95"][1],
            expected_mean + stats.t.ppf(0.975, 4) * expected_se,
            places=15,
        )
        self.assertEqual(
            summary["exact_sign_test"],
            {
                "nonzero_pairs": 5,
                "positive_pairs": 5,
                "negative_pairs": 0,
                "zero_pairs": 0,
                "two_sided_p": 0.0625,
            },
        )

    def test_paired_summary_rejects_pseudoreplicated_cardinality(self) -> None:
        for size in (4, 6, 120, 182_700):
            with self.subTest(size=size), self.assertRaisesRegex(
                verifier.VerificationError, "exactly five independent"
            ):
                verifier.paired_summary([0.0] * size)

    def test_exact_sign_test_uses_tolerant_zero_classification(self) -> None:
        summary = verifier.paired_summary(
            [5e-13, -5e-13, 2e-12, -2e-12, 0.01]
        )
        self.assertEqual(summary["exact_sign_test"]["zero_pairs"], 2)
        self.assertEqual(summary["exact_sign_test"]["nonzero_pairs"], 3)
        self.assertEqual(summary["exact_sign_test"]["positive_pairs"], 2)
        self.assertEqual(summary["exact_sign_test"]["negative_pairs"], 1)

    def test_exact_fixed5_sql_cardinality_uses_five_and_300(self) -> None:
        connection = self._tiny_connection()
        try:
            with (
                mock.patch.object(
                    verifier, "COHORT_SIZES", {"BRCA": 1, "LUAD": 1}
                ),
                mock.patch.object(verifier, "EXPECTED_ROWS", 600),
            ):
                counts = verifier.validate_complete(connection)
                self.assertEqual(counts["combination_count"], 120)
                self.assertEqual(counts["fm_pair_count"], 5)
                self.assertEqual(
                    counts["patient_counts_by_seed_cancer"]["32001:BRCA"],
                    1,
                )
                connection.execute("""
                    UPDATE predictions SET y_true = 1
                    WHERE fm_seed = 32005 AND cancer = 'BRCA'
                """)
                with self.assertRaisesRegex(
                    verifier.VerificationError, "across five FM seeds"
                ):
                    verifier.validate_complete(connection)
        finally:
            connection.close()

    def test_contract_has_exact_matrix_and_no_pseudoreplication(self) -> None:
        contract = verifier.contract_report()
        self.assertEqual(contract["fm_seeds"], list(verifier.FM_SEEDS))
        self.assertEqual(contract["independent_fm_seed_units"], 5)
        self.assertEqual(contract["expected_row_count"], 182_700)
        self.assertEqual(contract["expected_combination_count"], 120)
        self.assertEqual(contract["expected_nested_audit_count"], 2_250)
        self.assertIs(
            contract["heads_cancers_folds_targets_patients_are_repeated"],
            True,
        )
        self.assertEqual(
            contract["numeric_comparison"],
            {
                "absolute_tolerance": verifier.GATE_ABS_TOL,
                "relative_tolerance": verifier.GATE_REL_TOL,
                "raw_values_unrounded_for_reporting": True,
            },
        )
        self.assertEqual(
            verifier.semantic_comparison_contract(),
            {
                "scope": "analyzer_semantic_report",
                "absolute_tolerance": verifier.SEMANTIC_ABS_TOL,
                "relative_tolerance": verifier.SEMANTIC_REL_TOL,
            },
        )
        self.assertEqual(verifier.GATE_REL_TOL, 0.0)
        self.assertEqual(verifier.SEMANTIC_REL_TOL, 1e-12)

    def test_large_stable_baseline_passes_every_gate(self) -> None:
        full, halves, cancers = self._large_inputs()
        result = self._screen(full, halves, cancers)
        self.assertEqual(
            result["decision"]["classification"],
            "large_stable_practical_effect_favoring_H",
        )
        self.assertTrue(all(
            result["gates"]["fixed5_practical_screen"]["large_stable"].values()
        ))

    def test_each_large_stable_gate_is_independently_required(self) -> None:
        cases = {
            "absolute_mean_ge_0.02": lambda f, h, c: f.update(mean=0.019),
            "median_same_strict_direction": (
                lambda f, h, c: f.update(median=-0.03)
            ),
            "absolute_median_ge_0.02": (
                lambda f, h, c: f.update(median=0.019)
            ),
            "at_least_four_of_five_strict_seed_signs_match": (
                lambda f, h, c: f.update(
                    per_seed_theta=[0.03, 0.03, 0.03, -0.03, 0.0]
                )
            ),
            "all_leave_one_out_means_same_strict_direction": (
                lambda f, h, c: f.update(
                    leave_one_seed_out_means=[0.03, 0.03, 0.03, 0.03, -0.01]
                )
            ),
            "both_head_halves_same_strict_direction": (
                lambda f, h, c: h[1].update(mean=-0.03)
            ),
            "both_cancers_same_strict_direction": (
                lambda f, h, c: c["LUAD"].update(mean=-0.03)
            ),
        }
        for gate, mutation in cases.items():
            with self.subTest(gate=gate):
                full, halves, cancers = self._large_inputs()
                mutation(full, halves, cancers)
                result = self._screen(full, halves, cancers)
                gates = result["gates"]["fixed5_practical_screen"][
                    "large_stable"
                ]
                self.assertFalse(gates[gate])
                self.assertEqual(
                    result["decision"]["classification"],
                    "unstable_insufficient",
                )

        for gate, keyword in (
            ("favored_arm_harm_gate", {"harm": False}),
            ("favored_arm_utility_gate", {"utility": False}),
        ):
            with self.subTest(gate=gate):
                full, halves, cancers = self._large_inputs()
                result = self._screen(full, halves, cancers, **keyword)
                self.assertFalse(
                    result["gates"]["fixed5_practical_screen"][
                        "large_stable"
                    ][gate]
                )
                self.assertEqual(
                    result["decision"]["classification"],
                    "unstable_insufficient",
                )

    def test_negative_large_stable_direction_favors_p(self) -> None:
        full = verifier.paired_summary([-0.03] * 5)
        halves = [{"mean": -0.03}, {"mean": -0.03}]
        cancers = {"BRCA": {"mean": -0.03}, "LUAD": {"mean": -0.03}}
        result = self._screen(full, halves, cancers)
        self.assertEqual(
            result["decision"]["classification"],
            "large_stable_practical_effect_favoring_P",
        )
        self.assertEqual(result["decision"]["favored_arm_by_mean"], "P")

    def test_small_screen_requires_each_of_its_three_gates(self) -> None:
        full, halves, cancers = self._small_inputs()
        result = self._screen(full, halves, cancers)
        self.assertEqual(
            result["decision"]["classification"],
            "small_across_five_tested_seeds",
        )
        cases = {
            "absolute_mean_lt_0.02": (
                lambda f: f.update(mean=0.02 - 5e-13)
            ),
            "every_absolute_seed_effect_lt_0.03": (
                lambda f: f.update(
                    per_seed_theta=[0.005, 0.005, 0.005, 0.005, 0.03 - 5e-13]
                )
            ),
            "ci90_strictly_inside_+/-0.03": (
                lambda f: f.update(ci90=[-0.03 + 5e-13, 0.02])
            ),
        }
        for gate, mutation in cases.items():
            with self.subTest(gate=gate):
                full, halves, cancers = self._small_inputs()
                mutation(full)
                result = self._screen(full, halves, cancers)
                self.assertFalse(
                    result["gates"]["fixed5_practical_screen"][
                        "small_across_five"
                    ][gate]
                )
                self.assertEqual(
                    result["decision"]["classification"],
                    "unstable_insufficient",
                )

    def test_secondary_rules_each_remain_independent_and_df4_only(self) -> None:
        full, halves, cancers = self._large_inputs()
        result = self._screen(full, halves, cancers)
        secondary = result["gates"]["secondary_original_rules"][
            "superiority"
        ]
        self.assertTrue(all(secondary.values()))
        mutations = {
            "paired_t_p_lt_0.05": lambda f, h, c: f.update(
                two_sided_t_p=0.05 - 5e-13
            ),
            "absolute_mean_ge_0.02": lambda f, h, c: f.update(mean=0.019),
            "both_head_halves_same_strict_direction": (
                lambda f, h, c: h[0].update(mean=-0.03)
            ),
            "both_cancers_same_strict_direction": (
                lambda f, h, c: c["BRCA"].update(mean=-0.03)
            ),
        }
        for gate, mutation in mutations.items():
            with self.subTest(gate=gate):
                full, halves, cancers = self._large_inputs()
                mutation(full, halves, cancers)
                result = self._screen(full, halves, cancers)
                self.assertFalse(
                    result["gates"]["secondary_original_rules"][
                        "superiority"
                    ][gate]
                )
        for gate, kwargs in (
            ("favored_arm_harm_gate", {"harm": False}),
            ("favored_arm_utility_gate", {"utility": False}),
        ):
            full, halves, cancers = self._large_inputs()
            result = self._screen(full, halves, cancers, **kwargs)
            self.assertFalse(
                result["gates"]["secondary_original_rules"][
                    "superiority"
                ][gate]
            )

    def test_numeric_comparison_boundaries_follow_amendments_07_08(self) -> None:
        self.assertTrue(verifier._inclusive_ge(0.02 - 5e-13, 0.02))
        self.assertFalse(verifier._inclusive_ge(0.02 - 2e-12, 0.02))
        self.assertTrue(verifier._inclusive_ge(-0.02 - 5e-13, -0.02))
        self.assertFalse(verifier._inclusive_ge(-0.02 - 2e-12, -0.02))
        self.assertFalse(verifier._strict_lt(0.03 - 5e-13, 0.03))
        self.assertTrue(verifier._strict_lt(0.03 - 2e-12, 0.03))
        self.assertFalse(verifier._strict_gt(-0.03 + 5e-13, -0.03))
        self.assertTrue(verifier._strict_gt(-0.03 + 2e-12, -0.03))
        self.assertEqual(verifier._sign(5e-13), 0)
        self.assertEqual(verifier._sign(-5e-13), 0)
        self.assertEqual(verifier._sign(2e-12), 1)
        self.assertEqual(verifier._sign(-2e-12), -1)
        self.assertFalse(verifier._strict_lt(0.05, 0.05))
        self.assertFalse(verifier._strict_lt(0.05 - 5e-13, 0.05))
        self.assertFalse(verifier._strict_lt(0.05 + 5e-13, 0.05))
        self.assertTrue(verifier._strict_lt(0.05 - 2e-12, 0.05))
        self.assertFalse(verifier._strict_lt(0.05 + 2e-12, 0.05))

    def test_subtraction_produces_locked_practical_boundaries_end_to_end(
        self,
    ) -> None:
        cases = (
            (
                0.22,
                0.20,
                "H",
                "absolute_mean_ge_0.02",
                "absolute_mean_lt_0.02",
            ),
            (
                0.20,
                0.22,
                "P",
                "absolute_mean_ge_0.02",
                "absolute_mean_lt_0.02",
            ),
        )
        for p_value, h_value, favored, large_gate, small_gate in cases:
            with self.subTest(p=p_value, h=h_value):
                result = self._analyze_subtracted_effect(p_value, h_value)
                self.assertEqual(
                    result["decision"]["favored_arm_by_mean"], favored
                )
                self.assertTrue(
                    result["gates"]["fixed5_practical_screen"][
                        "large_stable"
                    ][large_gate]
                )
                self.assertFalse(
                    result["gates"]["fixed5_practical_screen"][
                        "small_across_five"
                    ][small_gate]
                )

        for p_value, h_value in ((0.23, 0.20), (0.20, 0.23)):
            with self.subTest(p=p_value, h=h_value):
                result = self._analyze_subtracted_effect(p_value, h_value)
                self.assertFalse(
                    result["gates"]["fixed5_practical_screen"][
                        "small_across_five"
                    ]["every_absolute_seed_effect_lt_0.03"]
                )

    def test_subtraction_produces_all_utility_delta_boundaries(self) -> None:
        b = {
            "BRCA": {
                "overall_auroc": 0.80,
                "black_auprc": 0.50,
                "black_ece": 0.10,
            },
            "LUAD": {
                "overall_auroc": 0.80,
                "black_auprc": 0.50,
                "black_ece": 0.10,
            },
        }
        h = {
            "BRCA": {
                "overall_auroc": 0.80 - 0.05,
                "black_auprc": 0.50 - 0.05,
                "black_ece": 0.10 + 0.05,
            },
            "LUAD": {
                "overall_auroc": 0.80 + 0.01,
                "black_auprc": 0.50 + 0.01,
                "black_ece": 0.10 - 0.01,
            },
        }
        report = verifier._utility_report(
            self._utility_values(b_cancers=b, h_cancers=h)
        )
        gate = verifier.utility_gate(report, "H")
        self.assertTrue(gate["pass"])
        overall = report["H"]["deltas_vs_B"]["overall"]
        brca = report["H"]["deltas_vs_B"]["cancers"]["BRCA"]
        self.assertTrue(verifier._close(overall["overall_auroc"], -0.02))
        self.assertTrue(verifier._close(overall["black_auprc"], -0.02))
        self.assertTrue(verifier._close(overall["black_ece"], 0.02))
        self.assertTrue(verifier._close(brca["overall_auroc"], -0.05))
        self.assertTrue(verifier._close(brca["black_auprc"], -0.05))
        self.assertTrue(verifier._close(brca["black_ece"], 0.05))
        self.assertTrue(all(gate["mean_bounds"].values()))
        self.assertTrue(all(gate["cancer_bounds"]["BRCA"].values()))

    def test_averaging_produces_strict_utility_auroc_boundaries(self) -> None:
        at_overall = {
            cancer: {
                "overall_auroc": 0.60,
                "black_auprc": 0.50,
                "black_ece": 0.10,
            }
            for cancer in verifier.CANCERS
        }
        report = verifier._utility_report(
            self._utility_values(
                b_cancers=at_overall, h_cancers=at_overall
            )
        )
        gate = verifier.utility_gate(report, "H")
        self.assertFalse(
            gate["mean_bounds"]["mean_overall_auroc_gt_0.60"]
        )
        above_overall = copy.deepcopy(at_overall)
        for cancer in verifier.CANCERS:
            above_overall[cancer]["overall_auroc"] = 0.60 + 2e-12
        report = verifier._utility_report(
            self._utility_values(
                b_cancers=above_overall, h_cancers=above_overall
            )
        )
        self.assertTrue(
            verifier.utility_gate(report, "H")["mean_bounds"][
                "mean_overall_auroc_gt_0.60"
            ]
        )

        at_cancer = {
            "BRCA": {
                "overall_auroc": 0.57,
                "black_auprc": 0.50,
                "black_ece": 0.10,
            },
            "LUAD": {
                "overall_auroc": 0.65,
                "black_auprc": 0.50,
                "black_ece": 0.10,
            },
        }
        report = verifier._utility_report(
            self._utility_values(b_cancers=at_cancer, h_cancers=at_cancer)
        )
        self.assertFalse(
            verifier.utility_gate(report, "H")["cancer_bounds"]["BRCA"][
                "auroc_gt_0.57"
            ]
        )
        above_cancer = copy.deepcopy(at_cancer)
        above_cancer["BRCA"]["overall_auroc"] = 0.57 + 2e-12
        report = verifier._utility_report(
            self._utility_values(
                b_cancers=above_cancer, h_cancers=above_cancer
            )
        )
        self.assertTrue(
            verifier.utility_gate(report, "H")["cancer_bounds"]["BRCA"][
                "auroc_gt_0.57"
            ]
        )

    def test_harm_gate_boundary_is_inclusive_with_tolerance(self) -> None:
        for value, expected in (
            (0.03, True),
            (0.03 + 5e-13, True),
            (0.03 + 2e-12, False),
        ):
            with self.subTest(value=value):
                gate = verifier.harm_gate(
                    {"BRCA": value, "LUAD": value}, "H"
                )
                self.assertIs(gate["pass"], expected)
        self.assertFalse(verifier.harm_gate({}, None)["pass"])

    def test_every_utility_boundary_is_independently_applied(self) -> None:
        utility = {
            "B": {
                "overall": {
                    "overall_auroc": 0.80,
                    "black_auprc": 0.50,
                    "black_ece": 0.10,
                },
                "cancers": {
                    cancer: {
                        "overall_auroc": 0.80,
                        "black_auprc": 0.50,
                        "black_ece": 0.10,
                    }
                    for cancer in verifier.CANCERS
                },
            },
            "H": {
                "overall": {
                    "overall_auroc": 0.61,
                    "black_auprc": 0.50,
                    "black_ece": 0.10,
                },
                "cancers": {
                    cancer: {
                        "overall_auroc": 0.58,
                        "black_auprc": 0.50,
                        "black_ece": 0.10,
                    }
                    for cancer in verifier.CANCERS
                },
                "deltas_vs_B": {
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
                        }
                        for cancer in verifier.CANCERS
                    },
                },
            },
        }
        self.assertTrue(verifier.utility_gate(utility, "H")["pass"])

        cases = [
            ("mean", "overall_auroc", 0.60 + 5e-13),
            ("mean_delta", "overall_auroc", -0.02 - 2e-12),
            ("mean_delta", "black_auprc", -0.02 - 2e-12),
            ("mean_delta", "black_ece", 0.02 + 2e-12),
            ("cancer", "overall_auroc", 0.57 + 5e-13),
            ("cancer_delta", "overall_auroc", -0.05 - 2e-12),
            ("cancer_delta", "black_auprc", -0.05 - 2e-12),
            ("cancer_delta", "black_ece", 0.05 + 2e-12),
        ]
        for scope, metric, value in cases:
            with self.subTest(scope=scope, metric=metric):
                changed = copy.deepcopy(utility)
                if scope == "mean":
                    changed["H"]["overall"][metric] = value
                elif scope == "mean_delta":
                    changed["H"]["deltas_vs_B"]["overall"][metric] = value
                elif scope == "cancer":
                    changed["H"]["cancers"]["BRCA"][metric] = value
                else:
                    changed["H"]["deltas_vs_B"]["cancers"]["BRCA"][
                        metric
                    ] = value
                self.assertFalse(verifier.utility_gate(changed, "H")["pass"])

        for boundary in (0.60, 0.57):
            self.assertFalse(
                verifier._strict_gt(boundary + 5e-13, boundary)
            )
            self.assertTrue(
                verifier._strict_gt(boundary + 2e-12, boundary)
            )
        for boundary in (-0.02, -0.05):
            self.assertTrue(
                verifier._inclusive_ge(boundary - 5e-13, boundary)
            )
        for boundary in (0.02, 0.05):
            self.assertTrue(
                verifier._inclusive_le(boundary + 5e-13, boundary)
            )

    def test_analyze_recomputes_2250_audits_and_practical_screen(self) -> None:
        def ensemble(
            connection: object,
            seed: int,
            arm: str,
            cancer: str,
            heads: tuple[int, ...],
        ) -> tuple[int, str, str, tuple[int, ...]]:
            return seed, arm, cancer, tuple(heads)

        def endpoint(marker: tuple) -> tuple[float, list[dict]]:
            _, arm, _, _ = marker
            value = 0.23 if arm == "P" else 0.20
            return value, [{"index": index} for index in range(75)]

        def endpoint_only(marker: tuple) -> float:
            _, arm, _, _ = marker
            return 0.23 if arm == "P" else 0.20

        with (
            mock.patch.object(verifier, "ensemble_rows", side_effect=ensemble),
            mock.patch.object(
                verifier, "nested_ieo_with_audit", side_effect=endpoint
            ),
            mock.patch.object(
                verifier, "nested_ieo", side_effect=endpoint_only
            ),
            mock.patch.object(
                verifier,
                "utility_metrics",
                return_value={
                    "overall_auroc": 0.80,
                    "black_auprc": 0.50,
                    "black_ece": 0.10,
                },
            ),
        ):
            result = verifier.analyze(mock.sentinel.connection)
        self.assertEqual(len(result["nested_audit"]), 2_250)
        self.assertEqual(result["full"]["df"], 4)
        self.assertEqual(
            result["decision"]["classification"],
            "large_stable_practical_effect_favoring_H",
        )

    def test_semantic_analyzer_comparison_accepts_only_complete_match(self) -> None:
        expected = {
            "integer": 5,
            "float": 0.02,
            "list": [True, None, "H"],
            "nested": {"value": -0.03},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "analysis.json"

            def write(semantic: dict, *, schema: str | None = None) -> None:
                path.write_text(json.dumps({
                    "schema": schema or verifier.ANALYZER_REPORT_SCHEMA,
                    "semantic_report": semantic,
                }), encoding="utf-8")

            write(expected)
            verifier.compare_analyzer(path, expected)

            close = copy.deepcopy(expected)
            close["nested"]["value"] += 5e-13
            write(close)
            verifier.compare_analyzer(path, expected)

            far = copy.deepcopy(expected)
            far["nested"]["value"] += 2e-12
            write(far)
            with self.assertRaisesRegex(
                verifier.VerificationError, r"nested\.value"
            ):
                verifier.compare_analyzer(path, expected)

            missing = copy.deepcopy(expected)
            del missing["integer"]
            write(missing)
            with self.assertRaisesRegex(
                verifier.VerificationError, "key mismatch"
            ):
                verifier.compare_analyzer(path, expected)

            write(expected, schema="wrong/v1")
            with self.assertRaisesRegex(
                verifier.VerificationError, "schema differs"
            ):
                verifier.compare_analyzer(path, expected)

    def test_semantic_comparison_uses_legacy_relative_tolerance(self) -> None:
        expected = {"large": 1_000_000_000.0, "near_zero": 0.0}
        verifier._semantic_equal(
            expected,
            {"large": 1_000_000_000.0005, "near_zero": 5e-13},
        )
        with self.assertRaisesRegex(
            verifier.VerificationError, r"\$\.large"
        ):
            verifier._semantic_equal(
                expected,
                {"large": 1_000_000_000.002, "near_zero": 0.0},
            )
        with self.assertRaisesRegex(
            verifier.VerificationError, r"\$\.near_zero"
        ):
            verifier._semantic_equal(
                expected,
                {"large": 1_000_000_000.0, "near_zero": 2e-12},
            )

    def test_analyzer_import_firewall_holds_in_ast_and_runtime(self) -> None:
        source_path = Path(verifier.__file__)
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn(
            FORBIDDEN_ANALYZER,
            _ast_import_targets(source, FIXED5_PACKAGE),
        )
        ast_forms = (
            f"import {FORBIDDEN_ANALYZER}",
            f"from {FIXED5_PACKAGE} import analyzer",
            "from . import analyzer",
            "from .analyzer import analyze",
        )
        for form in ast_forms:
            with self.subTest(ast_form=form):
                self.assertIn(
                    FORBIDDEN_ANALYZER,
                    _ast_import_targets(form, FIXED5_PACKAGE),
                )

        observed: set[str] = set()
        original_import = builtins.__import__

        def recording_import(
            name: str,
            globals: dict | None = None,
            locals: dict | None = None,
            fromlist: tuple = (),
            level: int = 0,
        ):
            observed.update(
                _runtime_import_targets(
                    name, globals, fromlist or (), level
                )
            )
            return original_import(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location(
            "fixed5_verifier_firewall_probe", source_path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=recording_import):
            spec.loader.exec_module(module)
        self.assertNotIn(FORBIDDEN_ANALYZER, observed)
        self.assertNotIn(
            "tools.matched_cancer_fixed5_20260730.final_collector", observed
        )
        self.assertNotIn(
            "tools.matched_cancer_fixed5_20260730.source_manifest", observed
        )

        runtime_forms = (
            (FORBIDDEN_ANALYZER, {}, (), 0),
            (FIXED5_PACKAGE, {}, ("analyzer",), 0),
            ("", {"__package__": FIXED5_PACKAGE}, ("analyzer",), 1),
            (
                "analyzer",
                {"__package__": FIXED5_PACKAGE},
                ("analyze",),
                1,
            ),
        )
        for name, globals_value, fromlist, level in runtime_forms:
            with self.subTest(
                runtime_name=name, fromlist=fromlist, level=level
            ):
                self.assertIn(
                    FORBIDDEN_ANALYZER,
                    _runtime_import_targets(
                        name, globals_value, fromlist, level
                    ),
                )

    def test_output_is_exclusive_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sealed = root / "sealed.json"
            sealed.write_text("sealed", encoding="utf-8")
            output = root / "verification.json"
            verifier.write_output_exclusively(output, "first", [sealed])
            self.assertEqual(output.read_text(encoding="utf-8"), "first")
            with self.assertRaisesRegex(
                verifier.VerificationError, "new path"
            ):
                verifier.write_output_exclusively(output, "second", [sealed])
            self.assertEqual(output.read_text(encoding="utf-8"), "first")
            with self.assertRaisesRegex(
                verifier.VerificationError, "new path"
            ):
                verifier.write_output_exclusively(sealed, "changed", [sealed])
            self.assertEqual(sealed.read_text(encoding="utf-8"), "sealed")

            dangling = root / "dangling"
            dangling.symlink_to(root / "missing")
            with self.assertRaisesRegex(
                verifier.VerificationError, "new path"
            ):
                verifier.write_output_exclusively(
                    dangling, "changed", [sealed]
                )


if __name__ == "__main__":
    unittest.main()
