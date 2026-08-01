#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools.matched_cancer_representation_audit_20260801 import contract


class RepresentationAuditContractTest(unittest.TestCase):
    def test_frozen_seed_attempt_layer_probe_and_gate_contract(self) -> None:
        self.assertEqual(contract.FM_SEEDS, tuple(range(32001, 32006)))
        self.assertEqual(
            dict(contract.ACCEPTED_ATTEMPTS),
            {32001: "attempt_03", 32002: "attempt_01", 32003: "attempt_01", 32004: "attempt_01", 32005: "attempt_01"},
        )
        self.assertEqual(contract.LAYER_FAMILIES[384], ("E_plain", "E_fair"))
        self.assertEqual(contract.LAYER_FAMILIES[128], ("A_temp_plain", "A_temp_fair", "B", "P", "H"))
        self.assertEqual(contract.REPRESENTATION_NORMALIZATION, "per_tile_l2")
        self.assertEqual(
            set(contract.LAYER_NORMALIZATION.values()), {"per_tile_l2"}
        )
        self.assertEqual(contract.C_GRID, (0.01, 0.1, 1.0, 10.0, 100.0))
        self.assertEqual(contract.SOLVER_SEED, 288850999)
        self.assertEqual(contract.PRIMARY_GATE["race_minimum_reduction"], 0.05)
        self.assertEqual(contract.PRIMARY_GATE["cancer_maximum_loss"], 0.02)
        self.assertEqual(contract.PRIMARY_GATE["race_minimum_passing_seeds_per_cell"], 4)
        self.assertEqual(contract.PRIMARY_GATE["race_seed_comparator"], ">=")
        self.assertEqual(contract.PRIMARY_GATE["race_median_comparator"], ">=")
        self.assertEqual(contract.PRIMARY_GATE["cancer_minimum_passing_seeds_per_view"], 4)
        self.assertEqual(contract.PRIMARY_GATE["cancer_seed_comparator"], "<=")
        self.assertEqual(contract.PRIMARY_GATE["cancer_median_comparator"], "<=")
        self.assertEqual(len(contract.GATE_ELIGIBLE_CONTRASTS), 4)
        self.assertNotIn(("P", "A_temp_fair"), contract.GATE_ELIGIBLE_CONTRASTS)
        self.assertEqual(contract.DESCRIPTIVE_CONTRASTS, (("P", "A_temp_fair"),))
        self.assertEqual(contract.PROBE_CONTRACT["minimum_valid_inner_folds"], 2)
        self.assertEqual(contract.PROBE_CONTRACT["tile_training_weight"], 1 / 16)
        self.assertEqual(contract.KNN_K, 5)
        self.assertEqual(contract.CONTINUATION_LIMITS["maximum_new_fm_seeds_total"], 4)
        for layer in contract.LAYER_DIMENSIONS:
            contract.validate_representation_normalization(layer, "per_tile_l2")
            with self.assertRaisesRegex(ValueError, "per_tile_l2"):
                contract.validate_representation_normalization(layer, "none")
            row = contract.normalize_representation_row(
                layer, [2.0] * contract.LAYER_DIMENSIONS[layer]
            )
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in row)), 1.0)
        self.assertEqual(
            contract.PATIENT_POOLING, "arithmetic_mean_16_no_renormalization"
        )
        with self.assertRaisesRegex(ValueError, "zero"):
            contract.normalize_representation_row("E_plain", [0.0] * 384)
        bad = [1.0] * 128
        bad[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            contract.normalize_representation_row("P", bad)

    def test_production_paths_and_encoder_ancestry(self) -> None:
        paths = contract.production_paths(32001)
        self.assertEqual(paths["root"].name, "attempt_03")
        self.assertEqual(paths["checkpoints"]["E_plain"], paths["checkpoints"]["A_temp_plain"])
        self.assertEqual(paths["checkpoints"]["E_fair"], paths["checkpoints"]["A_temp_fair"])
        self.assertEqual(paths["checkpoints"]["P"].name, "latest.pt")
        digest_a = "a" * 64
        digest_b = "b" * 64
        contract.validate_encoder_state_sharing(
            {"E_plain": digest_a, "E_fair": digest_b, "B": digest_a, "H": digest_a, "P": digest_b}
        )
        with self.assertRaisesRegex(ValueError, "B/H"):
            contract.validate_encoder_state_sharing(
                {"E_plain": digest_a, "E_fair": digest_b, "B": digest_a, "H": digest_b, "P": digest_b}
            )
        contract.validate_equal_dimension("P", "H")
        with self.assertRaisesRegex(ValueError, "equal-dimensional"):
            contract.validate_equal_dimension("E_fair", "P")

    def test_real_sanitizer_emits_only_allowlisted_exact_population(self) -> None:
        rows = contract.load_sanitized_population()
        self.assertEqual(len(rows), 609)
        self.assertTrue(all(set(row) == contract.METADATA_ALLOWLIST for row in rows))
        self.assertEqual(sum(row["cancer"] == "BRCA" for row in rows), 328)
        self.assertEqual(sum(row["cancer"] == "LUAD" for row in rows), 281)
        contract.validate_exclusion_membership(rows)

    def test_sanitizer_filters_non_black_white_and_rejects_count_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for cancer, source in contract.METADATA_PATHS.items():
                target = root / source.name
                target.write_text(source.read_text())
                paths[cancer] = target
            with paths["BRCA"].open(newline="") as source:
                rows = list(csv.DictReader(source))
            for row in rows:
                if row["fold"] == "target" and row["race"].lower() == "white":
                    row["fold"] = "source"
                    break
            with paths["BRCA"].open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "exactly 609|BRCA patient"):
                contract.load_sanitized_population(paths)

    def test_diagnosis_denylist_and_metadata_allowlist_fail_closed(self) -> None:
        for field in ("tp53_status", "diagnosis", "outcome", "target", "y_true", "model_tp53_score"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    contract.assert_diagnosis_free_fields({"patient_id", field})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            contract.validate_metadata_records(
                [{"patient_id": "P", "cancer": "BRCA", "race": "Black", "tss": "A2", "age": "50"}]
            )

    def test_tile_digest_and_even_odd_views_are_exact_and_stable(self) -> None:
        occurrences = [
            {
                "payload_sha256": hashlib.sha256(f"tile-{index}".encode()).hexdigest(),
                "occurrence_index": index,
                "keep_mask": index != 2,
            }
            for index in range(40)
        ]
        expected = hashlib.sha256(
            f"rep-audit/v1|288850999|PATIENT|{occurrences[0]['payload_sha256']}|0".encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            contract.tile_rank_digest("PATIENT", occurrences[0]["payload_sha256"], 0), expected
        )
        views = contract.select_tile_views("PATIENT", occurrences)
        self.assertEqual({key: len(value) for key, value in views.items()}, {"A": 16, "B": 16})
        self.assertFalse(
            {(x["payload_sha256"], x["occurrence_index"]) for x in views["A"]}
            & {(x["payload_sha256"], x["occurrence_index"]) for x in views["B"]}
        )
        repeated = contract.select_tile_views("PATIENT", occurrences)
        self.assertEqual(views, repeated)
        with self.assertRaisesRegex(ValueError, "full cache/input position"):
            contract.select_tile_views("PATIENT", list(reversed(occurrences)))
        contract.validate_shared_tile_views({(32001, "E_plain"): views, (32005, "H"): views})
        altered = {key: list(value) for key, value in views.items()}
        altered["A"][0] = {
            "payload_sha256": occurrences[-1]["payload_sha256"],
            "occurrence_index": occurrences[-1]["occurrence_index"],
        }
        with self.assertRaisesRegex(ValueError, "differ"):
            contract.validate_shared_tile_views({(32001, "E_plain"): views, (32005, "H"): altered})
        with self.assertRaisesRegex(ValueError, "at least 32"):
            contract.select_tile_views("PATIENT", occurrences[:31])
        payloads_read = {
            item["occurrence_index"]: item["payload_sha256"]
            for view in contract.TILE_VIEWS
            for item in views[view]
        }
        contract.validate_selected_payload_reads(views, payloads_read)
        payloads_read[next(iter(payloads_read))] = "f" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            contract.validate_selected_payload_reads(views, payloads_read)

    def test_grouped_cancer_probe_folds_are_exact_and_bicancer(self) -> None:
        tss = {
            cancer: contract.EXPECTED_POPULATION[cancer]["tss"]
            for cancer in contract.CANCERS
        }
        folds = contract.grouped_cancer_probe_folds(tss)
        self.assertEqual(set(folds.values()), set(range(5)))
        for fold in range(5):
            self.assertEqual(
                {cancer for (cancer, _), assigned in folds.items() if assigned == fold},
                set(contract.CANCERS),
            )
        digest = hashlib.sha256(
            b"rep-audit-cancer-fold/v1|288850999|BRCA|A2"
        ).hexdigest()
        self.assertEqual(contract.cancer_fold_digest("BRCA", "A2"), digest)

    def test_exclusion_and_replay_overlap_hooks(self) -> None:
        row = {"patient_id": "TARGET-P", "cancer": "BRCA", "race": "Black", "tss": "A2"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exclusion = root / "excluded.txt"
            exclusion.write_text("\n".join(["TARGET-P"] + [f"OTHER-{i}" for i in range(978)]) + "\n")
            contract.validate_exclusion_membership([row], exclusion)
            replay_paths = []
            for index in range(5):
                path = root / f"replay_{index}.json"
                patient = "TARGET-P" if index == 3 else f"TRAIN-{index}"
                path.write_text(json.dumps({"schema": "matched-cancer-replay-manifest/v1", "occurrences": [{"patient": patient}]}))
                replay_paths.append(path)
            with self.assertRaisesRegex(ValueError, "overlap for seed 32004"):
                contract.validate_replay_nonoverlap([row], replay_paths)
            value = json.loads(replay_paths[3].read_text())
            value["occurrences"][0]["patient"] = "TRAIN-3"
            replay_paths[3].write_text(json.dumps(value))
            contract.validate_replay_nonoverlap([row], replay_paths)


if __name__ == "__main__":
    unittest.main()
