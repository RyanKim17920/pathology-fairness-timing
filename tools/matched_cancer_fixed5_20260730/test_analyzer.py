from __future__ import annotations

import unittest
from unittest import mock

from tools.matched_cancer_diagnostic_20260730 import analyzer as base

from . import analyzer


class Fixed5AnalyzerTests(unittest.TestCase):
    def _analyze_vector(
        self,
        vector: list[float],
        *,
        cancer_vectors: dict[str, list[float]] | None = None,
        half_multipliers: dict[tuple[int, ...], float] | None = None,
        baseline_ieo: float = 0.2,
        utility_auroc: float = 0.8,
    ) -> dict:
        by_cancer = cancer_vectors or {
            cancer: vector for cancer in analyzer.CANCERS
        }
        effects = {
            cancer: dict(zip(analyzer.FM_SEEDS, values, strict=True))
            for cancer, values in by_cancer.items()
        }
        half_multipliers = half_multipliers or {}

        def ensemble(
            cells: object,
            seed: int,
            arm: str,
            cancer: str,
            heads: tuple[int, ...],
        ) -> tuple[int, str, str, tuple[int, ...]]:
            return seed, arm, cancer, tuple(heads)

        def endpoint(
            marker: tuple[int, str, str, tuple[int, ...]],
        ) -> tuple[float, list[dict[str, int]]]:
            seed, arm, cancer, _ = marker
            if arm == "P":
                value = 0.2 + effects[cancer][seed]
            elif arm == "B":
                value = baseline_ieo
            else:
                value = 0.2
            return value, [{"audit_index": index} for index in range(75)]

        def endpoint_only(
            marker: tuple[int, str, str, tuple[int, ...]],
        ) -> float:
            seed, arm, cancer, heads = marker
            multiplier = half_multipliers.get(heads, 1.0)
            if arm == "P":
                return 0.2 + multiplier * effects[cancer][seed]
            if arm == "B":
                return baseline_ieo
            return 0.2

        utility = {
            "overall_auroc": utility_auroc,
            "black_auprc": 0.5,
            "black_ece": 0.1,
        }
        with (
            mock.patch.object(base, "ensemble_rows", side_effect=ensemble),
            mock.patch.object(base, "nested_endpoint", side_effect=endpoint),
            mock.patch.object(base, "nested_ieo", side_effect=endpoint_only),
            mock.patch.object(
                base, "utility_metrics", return_value=utility
            ),
        ):
            return analyzer.analyze({})

    def test_paired_summary_uses_five_independent_units(self) -> None:
        summary = analyzer.paired_summary([0.01] * 5)
        self.assertEqual(summary["df"], 4)
        self.assertEqual(summary["median"], 0.01)
        self.assertEqual(summary["minimum"], 0.01)
        self.assertEqual(summary["maximum"], 0.01)
        self.assertEqual(len(summary["leave_one_seed_out_means"]), 5)
        self.assertAlmostEqual(
            summary["exact_sign_test"]["two_sided_p"], 0.0625
        )
        self.assertEqual(summary["exact_sign_test"]["nonzero_pairs"], 5)

    def test_large_stable_requires_typical_material_effect(self) -> None:
        result = self._analyze_vector([0.04, 0.03, 0.025, 0.022, -0.001])
        self.assertEqual(
            result["decision"]["classification"],
            "large_stable_practical_effect_favoring_H",
        )
        gates = result["gates"]["fixed5_practical_screen"]["large_stable"]
        self.assertTrue(all(gates.values()))

    def test_outlier_cannot_drive_large_stable_label(self) -> None:
        result = self._analyze_vector([0.2, 0.001, 0.001, 0.001, -0.001])
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        gates = result["gates"]["fixed5_practical_screen"]["large_stable"]
        self.assertFalse(gates["absolute_median_ge_0.02"])

    def test_small_label_requires_every_locked_gate(self) -> None:
        result = self._analyze_vector([0.005] * 5)
        self.assertEqual(
            result["decision"]["classification"],
            "small_across_five_tested_seeds",
        )
        gates = result["gates"]["fixed5_practical_screen"][
            "small_across_five"
        ]
        self.assertTrue(all(gates.values()))

    def test_mixed_or_zero_directions_are_unstable(self) -> None:
        result = self._analyze_vector([0.04, -0.03, 0.02, -0.01, 0.0])
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        self.assertLess(
            result["decision"]["matching_strict_seed_sign_count"], 4
        )

    def test_leave_one_out_direction_is_independently_required(self) -> None:
        result = self._analyze_vector([0.2, 0.021, 0.021, 0.021, -0.1])
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        gates = result["gates"]["fixed5_practical_screen"]["large_stable"]
        self.assertTrue(gates["absolute_mean_ge_0.02"])
        self.assertTrue(gates["absolute_median_ge_0.02"])
        self.assertTrue(gates["at_least_four_of_five_strict_seed_signs_match"])
        self.assertFalse(
            gates["all_leave_one_out_means_same_strict_direction"]
        )

    def test_both_cancer_directions_are_independently_required(self) -> None:
        result = self._analyze_vector(
            [0.03] * 5,
            cancer_vectors={
                "BRCA": [0.08] * 5,
                "LUAD": [-0.02] * 5,
            },
        )
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        self.assertFalse(
            result["gates"]["fixed5_practical_screen"]["large_stable"][
                "both_cancers_same_strict_direction"
            ]
        )

    def test_both_head_half_directions_are_independently_required(self) -> None:
        result = self._analyze_vector(
            [0.03] * 5,
            half_multipliers={
                tuple(analyzer.HEAD_HALVES[0]): 1.0,
                tuple(analyzer.HEAD_HALVES[1]): -1.0,
            },
        )
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        self.assertFalse(
            result["gates"]["fixed5_practical_screen"]["large_stable"][
                "both_head_halves_same_strict_direction"
            ]
        )

    def test_harm_gate_is_independently_required(self) -> None:
        result = self._analyze_vector([0.03] * 5, baseline_ieo=0.1)
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        self.assertFalse(result["harm_gate"]["pass"])
        self.assertFalse(
            result["gates"]["fixed5_practical_screen"]["large_stable"][
                "favored_arm_harm_gate"
            ]
        )

    def test_actual_utility_gate_is_independently_required(self) -> None:
        result = self._analyze_vector([0.03] * 5, utility_auroc=0.55)
        self.assertEqual(
            result["decision"]["classification"], "unstable_insufficient"
        )
        gate = result["utility"]["favored_arm_gate"]
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["mean_bounds"]["mean_overall_auroc_gt_0.60"])

    def test_contract_forbids_pseudoreplication(self) -> None:
        contract = analyzer.contract_report()
        self.assertEqual(contract["fm_seeds"], list(analyzer.FM_SEEDS))
        self.assertEqual(contract["independent_fm_seed_units"], 5)
        self.assertEqual(contract["expected_row_count"], 182_700)
        self.assertEqual(contract["expected_combination_count"], 120)
        self.assertEqual(contract["expected_nested_audit_count"], 2_250)
        self.assertIs(
            contract["heads_cancers_folds_targets_patients_are_repeated"],
            True,
        )

    def test_reused_base_seed_scope_is_restored(self) -> None:
        original = base.FM_SEEDS
        with analyzer._fixed5_base_context():
            self.assertEqual(base.FM_SEEDS, analyzer.FM_SEEDS)
        self.assertEqual(base.FM_SEEDS, original)

    def test_paired_summary_rejects_wrong_cardinality(self) -> None:
        with self.assertRaisesRegex(
            analyzer.AnalysisError, "exactly five"
        ):
            analyzer.paired_summary([0.0] * 4)

    def test_inclusive_materiality_boundary_uses_frozen_tolerance(self) -> None:
        boundary = 0.02
        self.assertTrue(analyzer._inclusive_ge(boundary - 5e-13, boundary))
        self.assertTrue(analyzer._inclusive_ge(boundary + 5e-13, boundary))
        self.assertFalse(analyzer._inclusive_ge(boundary - 2e-12, boundary))
        self.assertTrue(analyzer._inclusive_ge(boundary + 2e-12, boundary))
        self.assertTrue(
            analyzer._inclusive_ge(abs(-boundary + 5e-13), boundary)
        )

    def test_strict_margin_boundaries_exclude_close_values(self) -> None:
        boundary = 0.03
        self.assertFalse(analyzer._strict_lt(boundary - 5e-13, boundary))
        self.assertFalse(analyzer._strict_lt(boundary + 5e-13, boundary))
        self.assertTrue(analyzer._strict_lt(boundary - 2e-12, boundary))
        self.assertFalse(analyzer._strict_lt(boundary + 2e-12, boundary))
        self.assertFalse(
            analyzer._strict_gt(-boundary + 5e-13, -boundary)
        )
        self.assertTrue(
            analyzer._strict_gt(-boundary + 2e-12, -boundary)
        )

    def test_zero_sign_and_exact_sign_test_use_frozen_tolerance(self) -> None:
        self.assertEqual(analyzer._sign(5e-13), 0)
        self.assertEqual(analyzer._sign(-5e-13), 0)
        self.assertEqual(analyzer._sign(2e-12), 1)
        self.assertEqual(analyzer._sign(-2e-12), -1)
        summary = analyzer.paired_summary(
            [5e-13, -5e-13, 2e-12, -2e-12, 0.01]
        )
        self.assertEqual(summary["exact_sign_test"]["zero_pairs"], 2)
        self.assertEqual(summary["exact_sign_test"]["nonzero_pairs"], 3)

    def test_secondary_p_boundary_is_strict_with_tolerance(self) -> None:
        boundary = 0.05
        self.assertFalse(analyzer._strict_lt(boundary, boundary))
        self.assertFalse(analyzer._strict_lt(boundary - 5e-13, boundary))
        self.assertFalse(analyzer._strict_lt(boundary + 5e-13, boundary))
        self.assertTrue(analyzer._strict_lt(boundary - 2e-12, boundary))
        self.assertFalse(analyzer._strict_lt(boundary + 2e-12, boundary))

    def test_subtraction_at_materiality_boundary_is_not_small(self) -> None:
        mathematical_boundary = 0.22 - 0.20
        result = self._analyze_vector([mathematical_boundary] * 5)
        self.assertEqual(
            result["decision"]["classification"],
            "large_stable_practical_effect_favoring_H",
        )
        self.assertTrue(
            result["gates"]["fixed5_practical_screen"]["large_stable"][
                "absolute_mean_ge_0.02"
            ]
        )
        self.assertFalse(
            result["gates"]["fixed5_practical_screen"][
                "small_across_five"
            ]["absolute_mean_lt_0.02"]
        )

    def test_subtraction_at_practical_margin_is_not_strictly_small(self) -> None:
        mathematical_boundary = 0.23 - 0.20
        result = self._analyze_vector([mathematical_boundary] * 5)
        self.assertEqual(
            result["decision"]["classification"],
            "large_stable_practical_effect_favoring_H",
        )
        self.assertFalse(
            result["gates"]["fixed5_practical_screen"][
                "small_across_five"
            ]["every_absolute_seed_effect_lt_0.03"]
        )

    def test_strict_utility_boundaries_use_frozen_tolerance(self) -> None:
        for boundary in (0.60, 0.57):
            self.assertFalse(analyzer._strict_gt(boundary, boundary))
            self.assertFalse(
                analyzer._strict_gt(boundary + 5e-13, boundary)
            )
            self.assertTrue(
                analyzer._strict_gt(boundary + 2e-12, boundary)
            )

    def test_inclusive_utility_boundaries_use_frozen_tolerance(self) -> None:
        for boundary in (-0.02, -0.05):
            self.assertTrue(analyzer._inclusive_ge(boundary, boundary))
            self.assertTrue(
                analyzer._inclusive_ge(boundary - 5e-13, boundary)
            )
            self.assertFalse(
                analyzer._inclusive_ge(boundary - 2e-12, boundary)
            )
        for boundary in (0.02, 0.05):
            self.assertTrue(analyzer._inclusive_le(boundary, boundary))
            self.assertTrue(
                analyzer._inclusive_le(boundary + 5e-13, boundary)
            )
            self.assertFalse(
                analyzer._inclusive_le(boundary + 2e-12, boundary)
            )


if __name__ == "__main__":
    unittest.main()
