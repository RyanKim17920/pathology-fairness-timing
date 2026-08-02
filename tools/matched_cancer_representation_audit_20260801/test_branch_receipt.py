from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import branch_receipt


def _identity() -> dict:
    return {"canonical_path": "/immutable/input", "bytes": 1, "sha256": "0" * 64}


def _metric() -> dict:
    race = []
    cancer = []
    for layer in branch_receipt.LAYERS:
        for seed in branch_receipt.FM_SEEDS:
            for cancer_name in branch_receipt.CANCERS:
                for view in branch_receipt.VIEWS:
                    for level in branch_receipt.LEVELS:
                        race.append(
                            {
                                "fm_seed": seed,
                                "layer": layer,
                                "cancer": cancer_name,
                                "view": view,
                                "probe_level": level,
                                "result": {"oriented_leakage": 0.20},
                            }
                        )
            for view in branch_receipt.VIEWS:
                cancer.append(
                    {
                        "fm_seed": seed,
                        "layer": layer,
                        "view": view,
                        "result": {"pooled_heldout_patient_auroc": 0.90},
                    }
                )
    return {
        "schema": branch_receipt.METRIC_INPUT_SCHEMA,
        "study_id": branch_receipt.STUDY_ID,
        "status": "complete",
        "diagnosis_free": True,
        "fm_seeds": list(branch_receipt.FM_SEEDS),
        "layers": list(branch_receipt.LAYERS),
        "views": list(branch_receipt.VIEWS),
        "probe_levels": list(branch_receipt.LEVELS),
        "compact_caches": {
            f"{seed}|{layer}": _identity()
            for seed in branch_receipt.FM_SEEDS for layer in branch_receipt.LAYERS
        },
        "preflight_receipt": _identity(),
        "race_probes": race,
        "cancer_probes": cancer,
        "secondary_geometry": {},
        "training_evidence": {},
    }


def _set_race(metric: dict, layer: str, value: float, *, level: str | None = None) -> None:
    for row in metric["race_probes"]:
        if row["layer"] == layer and (level is None or row["probe_level"] == level):
            row["result"]["oriented_leakage"] = value


def _set_cancer(metric: dict, layer: str, value: float) -> None:
    for row in metric["cancer_probes"]:
        if row["layer"] == layer:
            row["result"]["pooled_heldout_patient_auroc"] = value


class BranchPrecedenceTests(unittest.TestCase):
    def test_H_precedes_P_when_both_are_active_and_preserved(self) -> None:
        metric = _metric()
        _set_race(metric, "H", 0.10)
        _set_race(metric, "P", 0.10)
        decision = branch_receipt.evaluate_decision(metric)
        self.assertEqual(decision["selected_route"], 1)
        self.assertEqual(decision["action"], "use_H")

    def test_P_is_selected_when_H_is_inactive(self) -> None:
        metric = _metric()
        _set_race(metric, "P", 0.10)
        decision = branch_receipt.evaluate_decision(metric)
        self.assertEqual(decision["selected_route"], 2)
        self.assertEqual(decision["action"], "use_P")

    def test_any_final_activity_with_failed_preservation_stops_for_harm(self) -> None:
        metric = _metric()
        _set_race(metric, "H", 0.10, level="patient")
        _set_cancer(metric, "H", 0.80)
        decision = branch_receipt.evaluate_decision(metric)
        self.assertEqual(decision["selected_route"], 3)
        self.assertEqual(decision["action"], "stop_utility_harm")

    def test_E_then_A_patient_then_A_tile_precedence(self) -> None:
        e_metric = _metric()
        _set_race(e_metric, "E_fair", 0.10, level="tile")
        self.assertEqual(branch_receipt.evaluate_decision(e_metric)["selected_route"], 4)

        patient_metric = _metric()
        _set_race(patient_metric, "A_temp_fair", 0.10, level="patient")
        patient = branch_receipt.evaluate_decision(patient_metric)
        self.assertEqual(patient["selected_route"], 5)
        self.assertEqual(patient["action"], "run_carry_versus_fresh")

        tile_metric = _metric()
        _set_race(tile_metric, "A_temp_fair", 0.10, level="tile")
        tile = branch_receipt.evaluate_decision(tile_metric)
        self.assertEqual(tile["selected_route"], 6)
        self.assertEqual(tile["action"], "run_patient_mean_training")

    def test_route_7_evaluates_headroom_without_hardcoded_result(self) -> None:
        eligible = branch_receipt.evaluate_decision(_metric())
        self.assertEqual(eligible["selected_route"], 7)
        self.assertTrue(eligible["hmask_feasibility"]["pass"])
        self.assertEqual(eligible["action"], "train_H_mask_32001")

        inadequate_metric = _metric()
        for row in inadequate_metric["race_probes"]:
            if (
                row["layer"] == "B"
                and row["fm_seed"] == 32001
                and row["cancer"] == "BRCA"
                and row["view"] == "A"
                and row["probe_level"] == "patient"
            ):
                row["result"]["oriented_leakage"] = 0.01
        inadequate = branch_receipt.evaluate_decision(inadequate_metric)
        self.assertEqual(inadequate["selected_route"], 7)
        self.assertFalse(inadequate["hmask_feasibility"]["pass"])
        self.assertEqual(inadequate["action"], "no_training_inadequate_headroom")

    def test_route_7_preservation_failure_does_not_open_feasibility(self) -> None:
        metric = _metric()
        _set_cancer(metric, "A_temp_fair", 0.80)
        decision = branch_receipt.evaluate_decision(metric)
        self.assertEqual(decision["selected_route"], 7)
        self.assertEqual(decision["action"], "stop_utility_harm")
        self.assertFalse(decision["hmask_feasibility"]["evaluated"])


class FailClosedAndReceiptTests(unittest.TestCase):
    def test_missing_duplicate_and_diagnosis_fields_fail_closed(self) -> None:
        missing = _metric()
        missing["race_probes"].pop()
        with self.assertRaisesRegex(branch_receipt.BranchReceiptError, "race-probe topology"):
            branch_receipt.evaluate_decision(missing)

        duplicate = _metric()
        duplicate["race_probes"].append(copy.deepcopy(duplicate["race_probes"][0]))
        with self.assertRaisesRegex(branch_receipt.BranchReceiptError, "duplicate race-probe"):
            branch_receipt.evaluate_decision(duplicate)

        diagnosis = _metric()
        diagnosis["training_evidence"]["downstream_diagnosis"] = "forbidden"
        with self.assertRaisesRegex(branch_receipt.BranchReceiptError, "forbidden diagnosis"):
            branch_receipt.evaluate_decision(diagnosis)

    def test_strict_zero_and_inclusive_boundaries_use_amendment_07(self) -> None:
        self.assertFalse(branch_receipt._strict_gt(1e-13, 0.0))
        self.assertTrue(branch_receipt._strict_gt(2e-12, 0.0))
        self.assertTrue(branch_receipt._ge(0.05 - 5e-13, 0.05))
        self.assertFalse(branch_receipt._ge(0.05 - 2e-12, 0.05))
        self.assertTrue(branch_receipt._le(0.02 + 5e-13, 0.02))
        self.assertFalse(branch_receipt._le(0.02 + 2e-12, 0.02))

    def test_source_bound_receipt_and_atomic_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metric_path = root / "metric_input.json"
            metric_path.write_text(json.dumps(_metric(), allow_nan=False))
            receipt = branch_receipt.verify_metric_input(metric_path)
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(
                receipt["identities"]["erratum"]["sha256"],
                branch_receipt.ERRATUM_SHA256,
            )
            self.assertEqual(
                receipt["identities"]["metric_input"],
                branch_receipt.file_identity(metric_path),
            )
            output = root / "receipt.json"
            branch_receipt.write_json_atomic_exclusive(output, receipt)
            parsed = branch_receipt.load_json(output)
            self.assertEqual(parsed, receipt)
            with self.assertRaisesRegex(branch_receipt.BranchReceiptError, "refusing to overwrite"):
                branch_receipt.write_json_atomic_exclusive(output, receipt)

    def test_erratum_identity_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            metric_path = root / "metric_input.json"
            metric_path.write_text(json.dumps(_metric(), allow_nan=False))
            changed_erratum = root / "erratum.md"
            changed_erratum.write_text("changed\n")
            with self.assertRaisesRegex(branch_receipt.BranchReceiptError, "erratum differs"):
                branch_receipt.verify_metric_input(metric_path, erratum=changed_erratum)


if __name__ == "__main__":
    unittest.main()
