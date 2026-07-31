from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    load_receipt,
    topology_sha256,
)

from . import execution_receipts, final_collector, launch_receipt


class FinalCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = (Path(self.temporary.name) / "production").resolve()
        self.root.mkdir()
        self.fixed5 = self._file(
            "control/FIXED5_SOURCE_MANIFEST_V1.json", "fixed5\n"
        )
        self.fixed48 = self._file(
            "control/FIXED48_SOURCE_MANIFEST_V2.json", "fixed48\n"
        )
        self.adoption = self._file(
            "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
            "adoption\n",
        )
        self.authorization = self._file(
            "authorization/AUTHORIZATION_MANIFEST_V3.json", "auth\n"
        )
        self.feasibility = self._file(
            "control/FEASIBILITY_GATE_RECEIPT_V2.json", "feasible\n"
        )
        self.options = self._file("controls/FIXED5_CONTINUATION_OPTIONS.md")
        self.options_receipt = self._file(
            "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json"
        )
        self.dummy = self._file("control/dummy.txt")
        self.nonce = "00000000000000000000000000000001"
        self.launch = self._receipt(
            f"control/launch/FIXED5_LAUNCH_{self.nonce}_JOB_1.json",
            schema=launch_receipt.LAUNCH_SCHEMA,
            scenario=launch_receipt.LAUNCH_SCENARIO,
            fields={"launch_nonce": self.nonce, "slurm_job_id": "1"},
        )
        self.excluded = self._file(
            f"control/excluded/FIXED5_EXCLUDED_{self.nonce}.json"
        )
        self.collections: dict[int, Path] = {}
        self.successes: dict[int, Path] = {}
        self.chains: dict[int, dict[str, Path]] = {}
        for seed in final_collector.SEEDS:
            source = self.root / f"diagnostic/seed_{seed}/attempt_01/rows.jsonl"
            source.parent.mkdir(parents=True)
            self._write_rows(source, seed)
            source_receipt = self._file(
                f"diagnostic/seed_{seed}/attempt_01/rows.jsonl.receipt.json"
            )
            success = self._file(
                f"diagnostic/seed_{seed}/attempt_01/"
                f"{execution_receipts.SUCCESS_NAME}",
                f"success-{seed}\n",
            )
            self.collections[seed] = source
            self.successes[seed] = success
            if seed == 32001:
                self.chains[seed] = {"success": success}
            else:
                start = self._file(
                    f"fixed5_execution/seed_{seed}/attempt_01/"
                    f"{execution_receipts.START_NAME}"
                )
                complete = self._file(
                    f"fixed5_execution/seed_{seed}/attempt_01/"
                    f"{execution_receipts.COMPLETE_NAME}"
                )
                self.chains[seed] = {
                    "start": start,
                    "complete": complete,
                    "success": success,
                }
        self.state = execution_receipts.StudyState(
            completed=final_collector.SEEDS,
            lowest_incomplete=32006,
            resumable={},
            used_attempts={seed: (1,) for seed in final_collector.SEEDS},
            chains=self.chains,
        )
        self.patches = ExitStack()
        for module in (final_collector, launch_receipt, execution_receipts):
            self.patches.enter_context(
                mock.patch.object(module, "PRODUCTION_ROOT", self.root)
            )
        self.patches.enter_context(
            mock.patch.object(
                final_collector, "CONTINUATION_OPTIONS", self.options
            )
        )
        self.patches.enter_context(
            mock.patch.object(final_collector, "COHORT_SIZES", {
                "BRCA": 1,
                "LUAD": 1,
            })
        )
        self.patches.enter_context(
            mock.patch.object(final_collector, "EXPECTED_PATIENTS", 2)
        )
        self.patches.enter_context(
            mock.patch.object(final_collector, "ROWS_PER_SEED", 120)
        )
        self.patches.enter_context(
            mock.patch.object(final_collector, "EXPECTED_ROWS", 600)
        )

    def tearDown(self) -> None:
        self.patches.close()
        self.temporary.cleanup()

    def _file(self, relative: str, text: str = "sealed\n") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path.resolve()

    def _receipt(
        self, relative: str, *, schema: str, scenario: str, fields: dict
    ) -> Path:
        receipt = build_receipt(
            schema=schema,
            study_id=final_collector.STUDY_ID,
            scenario=scenario,
            identities={
                "dummy": file_identity(self.dummy),
                "prelaunch_receipt": file_identity(self.dummy),
            },
            fields=fields,
        )
        return atomic_write_receipt(self.root / relative, receipt)

    def _write_rows(
        self,
        path: Path,
        seed: int,
        *,
        mutate: bool = False,
        cohort_sizes: dict[str, int] | None = None,
    ) -> None:
        rows = []
        sizes = cohort_sizes or {"BRCA": 1, "LUAD": 1}
        patients = [
            (cancer, f"{cancer}-P{index:04d}", index % 5, index)
            for cancer, count in sizes.items()
            for index in range(count)
        ]
        for cancer, patient, fold, patient_index in patients:
            for arm in final_collector.ARMS:
                for head in final_collector.HEADS:
                    for outer in range(5):
                        y_true = patient_index % 2
                        if mutate and cancer == "BRCA":
                            y_true = 1 - y_true
                        rows.append({
                            "schema": final_collector.ROW_SCHEMA,
                            "fm_seed": seed,
                            "arm": arm,
                            "cancer": cancer,
                            "head_seed": head,
                            "patient_id": patient,
                            "y_true": y_true,
                            "race": (
                                "Black"
                                if patient_index % 2 == 0
                                else "White"
                            ),
                            "fold": fold,
                            "role": (
                                "outer_test"
                                if outer == fold
                                else "inner_calibration"
                            ),
                            "outer_fold": outer,
                            "inner_fold": None if outer == fold else fold,
                            "probability": 0.5,
                        })
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )

    def _success(self, path: Path, *, seed: int, **_kwargs) -> dict:
        self.assertEqual(path, self.successes[seed])
        collection = self.collections[seed]
        return {
            "identities": {
                "per_seed_collection": file_identity(collection),
                "per_seed_collection_receipt": file_identity(
                    collection.with_suffix(
                        collection.suffix + ".receipt.json"
                    )
                ),
            }
        }

    def _collect(self, *, continuation_verifier=None) -> Path:
        return final_collector.collect(
            production_root=self.root,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            launch=self.launch,
            excluded_audit=self.excluded,
            continuation_options=self.options,
            continuation_options_receipt=self.options_receipt,
            destination=(
                self.root
                / "finalization/attempt_01/fixed5_predictions.jsonl"
            ),
            state_scanner=lambda **_kwargs: self.state,
            success_verifier=self._success,
            collection_verifier=lambda *_args, **_kwargs: {},
            manifest_verifier=lambda _path: {},
            excluded_verifier=lambda *_args, **_kwargs: {},
            fixed48_manifest_verifier=lambda _path: {},
            authorization_verifier=lambda _path: {},
            feasibility_verifier=lambda *_args, **_kwargs: {},
            adoption_verifier=lambda *_args, **_kwargs: {},
            launch_verifier=lambda *_args, **_kwargs: {},
            continuation_verifier=(
                continuation_verifier
                or (lambda *_args, **_kwargs: {})
            ),
        )

    def _verify(
        self,
        output: Path,
        *,
        receipt_path: Path | None = None,
    ) -> dict:
        return final_collector.verify_final_collection(
            output,
            receipt_path=receipt_path,
            source_manifest=self.fixed5,
            manifest_verifier=lambda _path: {},
            state_scanner=lambda **_kwargs: self.state,
            success_verifier=self._success,
            collection_verifier=lambda *_args, **_kwargs: {},
            excluded_verifier=lambda *_args, **_kwargs: {},
            fixed48_manifest_verifier=lambda _path: {},
            authorization_verifier=lambda _path: {},
            feasibility_verifier=lambda *_args, **_kwargs: {},
            adoption_verifier=lambda *_args, **_kwargs: {},
            launch_verifier=lambda *_args, **_kwargs: {},
            continuation_verifier=lambda *_args, **_kwargs: {},
        )

    def test_exact_matrix_and_complete_ancestry_are_sealed(self) -> None:
        output = self._collect()
        self.assertEqual(len(output.read_text().splitlines()), 600)
        receipt = load_receipt(
            output.with_suffix(output.suffix + ".receipt.json")
        )
        self.assertEqual(receipt["row_count"], 600)
        self.assertEqual(receipt["combination_count"], 120)
        self.assertEqual(receipt["patient_count"], 2)
        self.assertEqual(
            set(receipt["identities"]["seed_chains"]),
            {str(seed) for seed in final_collector.SEEDS},
        )
        self.assertEqual(
            set(receipt["identities"]["seed_sources"]),
            {str(seed) for seed in final_collector.SEEDS},
        )
        for role in (
            "excluded_seed_audit",
            "launch_receipt",
            "fixed5_source_manifest",
            "adoption_authorization",
            "continuation_options",
            "amendment_08",
        ):
            self.assertIn(role, receipt["identities"])

    def test_cross_seed_metadata_drift_fails_before_publication(self) -> None:
        self._write_rows(self.collections[32005], 32005, mutate=True)
        destination = (
            self.root / "finalization/attempt_01/fixed5_predictions.jsonl"
        )
        with self.assertRaisesRegex(ValueError, "cohort metadata differs"):
            self._collect()
        self.assertFalse(destination.exists())

    def test_excluded_contamination_fails_before_publication(self) -> None:
        (self.root / "diagnostic/seed_32006").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "excluded-seed"):
            self._collect()

    def test_no_overwrite_or_redirect(self) -> None:
        output = self._collect()
        with self.assertRaises(FileExistsError):
            self._collect()
        output.unlink()
        target = self._file("elsewhere.jsonl")
        output.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self._collect()

    def test_production_scale_exact_182700_rows(self) -> None:
        production_sizes = {"BRCA": 328, "LUAD": 281}
        for seed, source in self.collections.items():
            self._write_rows(
                source,
                seed,
                cohort_sizes=production_sizes,
            )
        with (
            mock.patch.object(
                final_collector, "COHORT_SIZES", production_sizes
            ),
            mock.patch.object(final_collector, "EXPECTED_PATIENTS", 609),
            mock.patch.object(final_collector, "ROWS_PER_SEED", 36_540),
            mock.patch.object(final_collector, "EXPECTED_ROWS", 182_700),
        ):
            output = self._collect()
            with output.open("rb") as stream:
                self.assertEqual(sum(1 for _ in stream), 182_700)
            receipt = load_receipt(
                output.with_suffix(output.suffix + ".receipt.json")
            )
            self.assertEqual(receipt["row_count"], 182_700)
            self.assertEqual(receipt["combination_count"], 120)
            self.assertEqual(receipt["patient_count"], 609)

    def test_receipt_rejects_self_consistent_unexpected_top_level_field(
        self,
    ) -> None:
        output = self._collect()
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
        receipt = load_receipt(receipt_path)
        receipt["unexpected_field"] = "forged"
        atomic_write_receipt(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "field topology"):
            self._verify(output)

    def test_receipt_rejects_bogus_continuation_receipt_identity(
        self,
    ) -> None:
        output = self._collect()
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
        receipt = load_receipt(receipt_path)
        bogus = self._file("control/bogus-continuation.json")
        receipt["identities"]["continuation_options_receipt"] = (
            file_identity(bogus)
        )
        receipt["topology_sha256"] = topology_sha256(
            receipt["identities"]
        )
        atomic_write_receipt(receipt_path, receipt)
        with self.assertRaisesRegex(
            ValueError, "continuation_options_receipt identity differs"
        ):
            self._verify(output)

    def test_collection_semantically_verifies_continuation_receipt(
        self,
    ) -> None:
        calls: list[Path] = []

        def continuation(path: Path, **_kwargs) -> dict:
            calls.append(path)
            return {}

        self._collect(continuation_verifier=continuation)
        self.assertGreaterEqual(len(calls), 4)
        self.assertTrue(
            all(path == self.options_receipt for path in calls)
        )

    def test_collection_rejects_invalid_continuation_receipt_before_output(
        self,
    ) -> None:
        def reject(*_args, **_kwargs):
            raise ValueError("invalid continuation receipt")

        with self.assertRaisesRegex(
            ValueError, "invalid continuation receipt"
        ):
            self._collect(continuation_verifier=reject)
        self.assertFalse(
            (
                self.root
                / "finalization/attempt_01/fixed5_predictions.jsonl"
            ).exists()
        )

    def test_verify_rejects_relative_dotdot_and_intermediate_symlink(
        self,
    ) -> None:
        output = self._collect()
        receipt = output.with_suffix(output.suffix + ".receipt.json")
        relative = Path(
            "finalization/attempt_01/fixed5_predictions.jsonl"
        )
        dotdot = (
            self.root
            / "finalization/attempt_01/../attempt_01/"
            "fixed5_predictions.jsonl"
        )
        for candidate in (relative, dotdot):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    self._verify(candidate, receipt_path=receipt)
        receipt_aliases = (
            Path(receipt.name),
            (
                self.root
                / "finalization/attempt_01/../attempt_01/"
                "fixed5_predictions.jsonl.receipt.json"
            ),
        )
        for receipt_alias in receipt_aliases:
            with self.subTest(receipt_alias=receipt_alias):
                with self.assertRaises(ValueError):
                    self._verify(output, receipt_path=receipt_alias)
        attempt = output.parent
        real_attempt = attempt.with_name("real_attempt_01")
        attempt.rename(real_attempt)
        attempt.symlink_to(real_attempt, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ancestry redirected"):
            self._verify(output, receipt_path=receipt)


if __name__ == "__main__":
    unittest.main()
