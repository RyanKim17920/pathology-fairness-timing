from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import verifier


def _race_cells(delta: float = 0.06) -> list[dict]:
    rows = []
    for seed in verifier.FM_SEEDS:
        for cancer in verifier.CANCERS:
            for view in verifier.VIEWS:
                for level in verifier.PROBE_LEVELS:
                    rows.append(
                        {
                            "fm_seed": seed,
                            "cancer": cancer,
                            "view": view,
                            "probe_level": level,
                            "baseline_oriented_leakage": 0.20,
                            "candidate_oriented_leakage": 0.20 - delta,
                        }
                    )
    return rows


def _cancer_cells(loss: float = 0.01) -> list[dict]:
    return [
        {
            "fm_seed": seed,
            "view": view,
            "baseline_auroc": 0.90,
            "candidate_auroc": 0.90 - loss,
        }
        for seed in verifier.FM_SEEDS
        for view in verifier.VIEWS
    ]


def _contrast(candidate: str, baseline: str, *, delta: float = 0.06, loss: float = 0.01) -> dict:
    race = _race_cells(delta)
    cancer = _cancer_cells(loss)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "race_cells": race,
        "cancer_cells": cancer,
        "reported_gate": verifier.recompute_gate(race, cancer),
    }


class VerifierFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.metric_input = self.root / "metric_input.jsonl"
        self.lock = self.root / "lock.md"
        self.amendment = self.root / "amendment.md"
        self.analyzer = self.root / "analyzer.py"
        self.metric_input.write_text("diagnosis-free metrics\n")
        self.lock.write_text("frozen lock\n")
        self.amendment.write_text("numeric amendment\n")
        self.analyzer.write_text("# independent analyzer source\n")
        self.report_path = self.root / "analysis.json"
        self.receipt_path = self.root / "analysis.receipt.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def identities(self) -> dict:
        return {
            "metric_input": verifier.file_identity(self.metric_input),
            "lock": verifier.file_identity(self.lock),
            "numeric_amendment": verifier.file_identity(self.amendment),
            "analyzer": verifier.file_identity(self.analyzer),
        }

    def report(self) -> dict:
        return {
            "schema": verifier.ANALYSIS_SCHEMA,
            "study_id": verifier.STUDY_ID,
            "status": "complete",
            "diagnosis_free": True,
            "inference_unit": "FM seed",
            "fm_seeds": list(verifier.FM_SEEDS),
            "contrasts": [
                _contrast(candidate, baseline)
                for candidate, baseline in verifier.CONTRASTS
            ],
            "identities": self.identities(),
        }

    def publish(self, report: dict) -> None:
        self.report_path.write_text(json.dumps(report, sort_keys=True))
        receipt = {
            "schema": verifier.ANALYSIS_RECEIPT_SCHEMA,
            "study_id": verifier.STUDY_ID,
            "status": "complete",
            "analysis_report": verifier.file_identity(self.report_path),
            "identities": report["identities"],
        }
        self.receipt_path.write_text(json.dumps(receipt, sort_keys=True))

    def verify(self) -> dict:
        return verifier.verify_analysis_files(
            self.report_path,
            self.receipt_path,
            self.metric_input,
            lock=self.lock,
            numeric_amendment=self.amendment,
            analyzer_source=self.analyzer,
        )


class RepresentationAuditVerifierTests(VerifierFixture):
    def test_valid_report_recomputes_all_four_passing_gates(self) -> None:
        report = self.report()
        self.publish(report)
        result = self.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(result["semantic_report"]["contrasts"]), 4)
        self.assertTrue(all(result["scientific_gate_passes"].values()))

    def test_four_of_five_and_median_are_recomputed_per_race_stratum(self) -> None:
        race = _race_cells()
        target = [
            row
            for row in race
            if row["cancer"] == "BRCA"
            and row["view"] == "A"
            and row["probe_level"] == "patient"
        ]
        for row in target[:2]:
            row["candidate_oriented_leakage"] = row["baseline_oriented_leakage"]
        gate = verifier.recompute_gate(race, _cancer_cells())
        stratum = gate["race_leakage_strata"]["BRCA|A|patient"]
        self.assertEqual(stratum["seeds_at_or_above_threshold"], 3)
        self.assertFalse(stratum["pass"])
        self.assertFalse(gate["pass"])

    def test_cancer_loss_gate_is_per_view_four_of_five_plus_median(self) -> None:
        cancer = _cancer_cells()
        target = [row for row in cancer if row["view"] == "B"]
        for row in target[:2]:
            row["candidate_auroc"] = 0.80
        gate = verifier.recompute_gate(_race_cells(), cancer)
        self.assertEqual(
            gate["cancer_probe_views"]["B"]
            ["seeds_at_or_below_maximum_loss"],
            3,
        )
        self.assertFalse(gate["pass"])

    def test_inclusive_frozen_thresholds_pass_at_exact_boundaries(self) -> None:
        gate = verifier.recompute_gate(
            _race_cells(delta=0.05), _cancer_cells(loss=0.02)
        )
        self.assertTrue(gate["pass"])

    def test_missing_and_duplicate_gate_cells_fail_closed(self) -> None:
        with self.assertRaisesRegex(verifier.VerificationError, "exactly 40"):
            verifier.recompute_gate(_race_cells()[:-1], _cancer_cells())
        duplicate = _race_cells()
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate race"):
            verifier.recompute_gate(duplicate, _cancer_cells())

    def test_nonfinite_and_out_of_range_metrics_fail_closed(self) -> None:
        rows = _race_cells()
        rows[0]["candidate_oriented_leakage"] = float("nan")
        with self.assertRaisesRegex(verifier.VerificationError, "finite"):
            verifier.recompute_gate(rows, _cancer_cells())
        cancer = _cancer_cells()
        cancer[0]["baseline_auroc"] = 1.1
        with self.assertRaisesRegex(verifier.VerificationError, r"\[0.0, 1.0\]"):
            verifier.recompute_gate(_race_cells(), cancer)

    def test_analyzer_gate_tamper_is_detected_semantically(self) -> None:
        report = self.report()
        report["contrasts"][0]["reported_gate"]["pass"] = False
        report["contrasts"][0]["reported_gate"]["classification"] = "inactive"
        self.publish(report)
        with self.assertRaisesRegex(verifier.VerificationError, "reported_gate"):
            self.verify()

    def test_exact_contrast_topology_rejects_descriptive_or_duplicate_pair(self) -> None:
        report = self.report()
        report["contrasts"][0]["candidate"] = "P"
        report["contrasts"][0]["baseline"] = "A_temp_fair"
        self.publish(report)
        with self.assertRaisesRegex(verifier.VerificationError, "invalid or duplicate"):
            self.verify()

        report = self.report()
        report["contrasts"][0], report["contrasts"][1] = (
            report["contrasts"][1],
            report["contrasts"][0],
        )
        self.publish(report)
        with self.assertRaisesRegex(verifier.VerificationError, "invalid or duplicate"):
            self.verify()

    def test_report_receipt_identity_tamper_fails(self) -> None:
        report = self.report()
        self.publish(report)
        with self.report_path.open("a") as output:
            output.write(" \n")
        with self.assertRaisesRegex(verifier.VerificationError, "analysis_report identity"):
            self.verify()

    def test_bound_source_identity_tamper_fails(self) -> None:
        report = self.report()
        report["identities"]["analyzer"]["sha256"] = "0" * 64
        self.publish(report)
        with self.assertRaisesRegex(verifier.VerificationError, "analyzer.*identity"):
            self.verify()

    def test_diagnosis_field_and_extra_cell_field_fail_closed(self) -> None:
        report = self.report()
        report["contrasts"][0]["race_cells"][0]["tp53_status"] = 1
        self.publish(report)
        with self.assertRaisesRegex(verifier.VerificationError, "forbidden diagnosis"):
            self.verify()

    def test_strict_json_rejects_duplicate_keys_and_nan(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate JSON key"):
            verifier.load_json(duplicate)
        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text('{"value":NaN}')
        with self.assertRaisesRegex(verifier.VerificationError, "non-finite"):
            verifier.load_json(nonfinite)

    def test_exclusive_output_will_not_overwrite(self) -> None:
        output = self.root / "verification.json"
        verifier.write_json_exclusive(output, {"status": "pass"})
        with self.assertRaisesRegex(verifier.VerificationError, "already exists"):
            verifier.write_json_exclusive(output, {"status": "pass"})

    def test_identity_sources_may_not_be_symlinks(self) -> None:
        link = self.root / "metric-link"
        link.symlink_to(self.metric_input)
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            verifier.file_identity(link)


if __name__ == "__main__":
    unittest.main()
