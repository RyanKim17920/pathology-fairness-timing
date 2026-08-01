from __future__ import annotations

import copy
import unittest

import numpy as np

from . import analyzer


def _record(patient: str, cancer: str, race: str, tss: str, vector: list[float], tile: int = 0) -> dict:
    normalized = np.asarray(vector, dtype=float)
    normalized = normalized / np.linalg.norm(normalized)
    return {
        "metadata": {"patient_id": patient, "cancer": cancer, "race": race, "tss": tss},
        "tile_id": str(tile),
        "embedding": normalized.tolist(),
    }


def _race_dataset() -> list[dict]:
    rows = []
    for tss_index in range(4):
        for race_index, race in enumerate(analyzer.RACES):
            for patient_index in range(2):
                patient = f"P{tss_index}-{race_index}-{patient_index}"
                for tile in range(16):
                    # Both signal and harmless fold-specific variation.
                    rows.append(_record(patient, "BRCA", race, f"T{tss_index}", [1.0, race_index * 3.0 + tile * 0.001, tss_index * 0.01], tile))
    return rows


def _cancer_dataset() -> list[dict]:
    rows = []
    for cancer_index, cancer in enumerate(analyzer.CANCERS):
        for tss_index in range(5):
            for patient_index in range(2):
                patient = f"{cancer}-{tss_index}-{patient_index}"
                race = analyzer.RACES[patient_index % 2]
                vector = [1.0, 0.01 * tss_index] if cancer_index == 0 else [0.01 * tss_index, 1.0]
                for tile in range(16):
                    rows.append(_record(patient, cancer, race, f"{cancer}-T{tss_index}", vector, tile))
    return rows


def _gate_rows(delta: float = 0.06, cancer_loss: float = 0.01) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    fair_race = []
    matched_race = []
    for seed in analyzer.FM_SEEDS:
        for cancer in analyzer.CANCERS:
            for view in analyzer.VIEWS:
                for level in analyzer.LEVELS:
                    key = {"fm_seed": seed, "cancer": cancer, "view": view, "level": level}
                    matched_race.append({**key, "oriented_leakage": 0.20})
                    fair_race.append({**key, "oriented_leakage": 0.20 - delta})
    fair_cancer = []
    matched_cancer = []
    for seed in analyzer.FM_SEEDS:
        for view in analyzer.VIEWS:
            key = {"fm_seed": seed, "view": view}
            matched_cancer.append({**key, "pooled_heldout_patient_auroc": 0.90})
            fair_cancer.append({**key, "pooled_heldout_patient_auroc": 0.90 - cancer_loss})
    return fair_race, matched_race, fair_cancer, matched_cancer


class DiagnosisFreeSchemaTests(unittest.TestCase):
    def test_nested_forbidden_field_is_rejected(self) -> None:
        payload = {"safe": [{"Diagnosis": 1}]}
        with self.assertRaisesRegex(analyzer.AnalysisError, "forbidden diagnosis"):
            analyzer.reject_diagnosis_fields(payload)
        with self.assertRaisesRegex(analyzer.AnalysisError, "forbidden diagnosis"):
            analyzer.reject_diagnosis_fields({"derived_tp53_mutation": 1})

    def test_metadata_is_exact_and_cannot_smuggle_outcome(self) -> None:
        metadata = {"patient_id": "P", "cancer": "BRCA", "race": "Black", "tss": "T", "y_true": 1}
        with self.assertRaises(analyzer.AnalysisError):
            analyzer.diagnosis_free_metadata(metadata)

    def test_probe_rejects_representation_not_normalized_at_cache_boundary(self) -> None:
        rows = _race_dataset()
        rows[0]["embedding"] = [2.0, 0.0, 0.0]
        with self.assertRaisesRegex(analyzer.AnalysisError, "already be per-tile L2 normalized"):
            analyzer.nested_race_probe(rows, level="patient")


class ProbeTests(unittest.TestCase):
    def test_patient_and_patient_weighted_tile_probes_are_deterministic(self) -> None:
        rows = _race_dataset()
        patient = analyzer.nested_race_probe(rows, level="patient")
        tile = analyzer.nested_race_probe(list(reversed(rows)), level="tile")
        patient_again = analyzer.nested_race_probe(list(reversed(copy.deepcopy(rows))), level="patient")
        self.assertEqual(patient, patient_again)
        self.assertAlmostEqual(patient["pooled_heldout_patient_auroc"], 1.0)
        self.assertAlmostEqual(tile["pooled_heldout_patient_auroc"], 1.0)
        self.assertAlmostEqual(patient["oriented_leakage"], 0.5)
        self.assertEqual(patient["patient_count"], 16)
        self.assertTrue(all(fold["valid_inner_folds"] >= 2 for fold in tile["outer_folds"]))
        self.assertTrue(all(fold["selected_c"] == min(analyzer.C_GRID) for fold in patient["outer_folds"]))

    def test_inner_folds_without_both_races_are_counted_then_minimum_enforced(self) -> None:
        rows = [row for row in _race_dataset() if not (row["metadata"]["tss"] == "T0" and row["metadata"]["race"] == "Black")]
        counted = analyzer.nested_race_probe(rows, level="patient")
        self.assertTrue(any(fold["excluded_inner_folds_lacking_both_races"] == 1 for fold in counted["outer_folds"]))
        with self.assertRaisesRegex(analyzer.AnalysisError, "fewer than two valid"):
            # With four outer folds, some outer training sets retain only two race-complete
            # TSS; removing a second makes the nested requirement fail closed.
            reduced = [row for row in rows if not (row["metadata"]["tss"] == "T1" and row["metadata"]["race"] == "Black")]
            analyzer.nested_race_probe(reduced, level="patient")

    def test_pooled_cancer_probe_is_tss_blocked_and_deterministic(self) -> None:
        result = analyzer.pooled_cancer_probe(_cancer_dataset())
        self.assertAlmostEqual(result["pooled_heldout_patient_auroc"], 1.0)
        self.assertEqual(result["patient_count"], 20)
        self.assertEqual(result["outer_block_fold_count"], 5)
        seen_tss = [tss for fold in result["outer_folds"] for tss in fold["heldout_tss"]]
        self.assertEqual(len(seen_tss), len(set(seen_tss)))


class GeometryTests(unittest.TestCase):
    def test_unbiased_energy_distance_uses_distinct_ordered_pairs(self) -> None:
        rows = []
        for cancer in analyzer.CANCERS:
            for race in analyzer.RACES:
                rows.append(_record(f"{cancer}-{race}-0", cancer, race, f"{cancer}-T0", [1.0, 0.0]))
                rows.append(_record(f"{cancer}-{race}-1", cancer, race, f"{cancer}-T1", [0.0, 1.0]))
        result = analyzer.cancer_conditioned_energy_distance(rows)
        # The frozen unbiased U-statistic can be negative for finite samples.
        self.assertAlmostEqual(result["macro_cancer_mean"], -np.sqrt(2.0))

    def test_knn_excludes_same_patient_and_equal_weights_query_races(self) -> None:
        rows = [
            _record("B1", "BRCA", "Black", "T1", [1.0, 0.0], 0),
            _record("B1", "BRCA", "Black", "T1", [1.0, 0.001], 1),
        ]
        rows.extend(_record(f"W{i}", "BRCA", "White", f"TW{i}", [1.0, i * 0.001]) for i in range(1, 7))
        result = analyzer.cosine_knn_cross_race_mixing(rows, level="tile")
        # B1's nearly identical sibling tile is forbidden; both B1 queries choose five Whites.
        self.assertEqual(result["race_query_means"]["Black"], 1.0)
        self.assertEqual(result["query_count_by_race"], {"Black": 2, "White": 6})
        self.assertAlmostEqual(result["equal_race_mean"], (1.0 + result["race_query_means"]["White"]) / 2)

    def test_aligned_displacement_checks_identity_dimension_and_summarizes(self) -> None:
        left = [_record("B", "BRCA", "Black", "T1", [1.0, 0.0]), _record("W", "BRCA", "White", "T2", [0.0, 1.0])]
        right = [_record("B", "BRCA", "Black", "T1", [0.0, 1.0]), _record("W", "BRCA", "White", "T2", [1.0, 0.0])]
        result = analyzer.aligned_representation_displacement(left, right, level="patient")
        self.assertEqual(result["aligned_count"], 2)
        self.assertAlmostEqual(result["mean_l2"], np.sqrt(2.0))
        self.assertAlmostEqual(result["mean_cosine_distance"], 1.0)
        bad = copy.deepcopy(right)
        bad[0]["embedding"] = [2.0, 0.0, 0.0]
        with self.assertRaises(analyzer.AnalysisError):
            analyzer.aligned_representation_displacement(left, bad, level="patient")

    def test_parameter_displacement_uses_aligned_floating_tensors_only(self) -> None:
        result = analyzer.parameter_displacement(
            {"weight": np.asarray([3.0, 4.0]), "counter": np.asarray([1], dtype=np.int64)},
            {"weight": np.asarray([6.0, 8.0]), "counter": np.asarray([2], dtype=np.int64)},
        )
        self.assertAlmostEqual(result["absolute_frobenius"], 5.0)
        self.assertAlmostEqual(result["baseline_frobenius_ratio"], 1.0)
        self.assertEqual(result["floating_tensor_count"], 1)


class GateTests(unittest.TestCase):
    def test_exact_primary_gate_pass_and_semantics(self) -> None:
        rows = _gate_rows()
        result = analyzer.evaluate_primary_gate(*rows)
        self.assertTrue(result["pass"])
        self.assertEqual(result["classification"], "active")
        self.assertEqual(result["exact_dimensions"]["race_cell_count"], 40)
        self.assertEqual(result["exact_dimensions"]["cancer_cell_count"], 10)

    def test_four_of_five_and_median_are_applied_per_stratum(self) -> None:
        fair, matched, fair_cancer, matched_cancer = _gate_rows()
        target = {"cancer": analyzer.CANCERS[0], "view": analyzer.VIEWS[0], "level": analyzer.LEVELS[0]}
        changed = 0
        for row in fair:
            if all(row[key] == value for key, value in target.items()) and changed < 2:
                row["oriented_leakage"] = 0.20
                changed += 1
        result = analyzer.evaluate_primary_gate(fair, matched, fair_cancer, matched_cancer)
        self.assertFalse(result["pass"])
        key = "|".join(target.values())
        self.assertEqual(result["race_leakage_strata"][key]["seeds_at_or_above_threshold"], 3)

    def test_cancer_loss_and_missing_cells_fail_closed(self) -> None:
        fair, matched, fair_cancer, matched_cancer = _gate_rows(cancer_loss=0.03)
        self.assertFalse(analyzer.evaluate_primary_gate(fair, matched, fair_cancer, matched_cancer)["pass"])
        with self.assertRaisesRegex(analyzer.AnalysisError, "exact frozen cells"):
            analyzer.evaluate_primary_gate(fair[:-1], matched, fair_cancer, matched_cancer)
        fair[0]["oriented_leakage"] = 0.6
        with self.assertRaisesRegex(analyzer.AnalysisError, r"\[0.0, 0.5\]"):
            analyzer.evaluate_primary_gate(fair, matched, fair_cancer, matched_cancer)

    def test_semantic_report_rejects_diagnosis_contamination(self) -> None:
        fair, matched, fair_cancer, matched_cancer = _gate_rows()
        race_cells = [
            {
                "fm_seed": f["fm_seed"], "cancer": f["cancer"], "view": f["view"],
                "probe_level": f["level"],
                "baseline_oriented_leakage": m["oriented_leakage"],
                "candidate_oriented_leakage": f["oriented_leakage"],
            }
            for f, m in zip(fair, matched, strict=True)
        ]
        cancer_cells = [
            {
                "fm_seed": f["fm_seed"], "view": f["view"],
                "baseline_auroc": m["pooled_heldout_patient_auroc"],
                "candidate_auroc": f["pooled_heldout_patient_auroc"],
            }
            for f, m in zip(fair_cancer, matched_cancer, strict=True)
        ]
        contrasts = [
            {"candidate": candidate, "baseline": baseline, "race_cells": copy.deepcopy(race_cells), "cancer_cells": copy.deepcopy(cancer_cells)}
            for candidate, baseline in analyzer.GATE_ELIGIBLE_CONTRASTS
        ]
        identity = {"canonical_path": "/tmp/x", "bytes": 1, "sha256": "0" * 64}
        identities = {role: dict(identity) for role in ("metric_input", "lock", "numeric_amendment", "analyzer")}
        report = analyzer.semantic_report(contrasts=contrasts, identities=identities)
        self.assertEqual(report["schema"], analyzer.ANALYSIS_SCHEMA)
        with self.assertRaises(analyzer.AnalysisError):
            contaminated = copy.deepcopy(contrasts)
            contaminated[0]["race_cells"][0]["tp53"] = 1
            analyzer.semantic_report(contrasts=contaminated, identities=identities)


if __name__ == "__main__":
    unittest.main()
