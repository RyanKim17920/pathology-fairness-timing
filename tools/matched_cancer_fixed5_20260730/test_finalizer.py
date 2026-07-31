from __future__ import annotations

from contextlib import ExitStack
import json
import os
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

from . import (
    analyzer,
    execution_receipts,
    final_collector,
    finalizer,
    launch_receipt,
    verifier,
)


class FinalizerTests(unittest.TestCase):
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
        self.dummy = self._file("control/dummy.txt")
        self.chains: dict[int, dict[str, Path]] = {}
        for seed in execution_receipts.ADOPTED_SEEDS:
            success = self._file(
                f"diagnostic/seed_{seed}/attempt_01/"
                f"{execution_receipts.SUCCESS_NAME}",
                f"success-{seed}\n",
            )
            if seed == 32001:
                self.chains[seed] = {"success": success}
            else:
                self.chains[seed] = {
                    "start": self._file(
                        f"fixed5_execution/seed_{seed}/attempt_01/"
                        f"{execution_receipts.START_NAME}"
                    ),
                    "complete": self._file(
                        f"fixed5_execution/seed_{seed}/attempt_01/"
                        f"{execution_receipts.COMPLETE_NAME}"
                    ),
                    "success": success,
                }
        self.state = execution_receipts.StudyState(
            completed=execution_receipts.ADOPTED_SEEDS,
            lowest_incomplete=32006,
            resumable={},
            used_attempts={
                seed: (1,) for seed in execution_receipts.ADOPTED_SEEDS
            },
            chains=self.chains,
        )
        self.patches = ExitStack()
        for module in (
            finalizer,
            final_collector,
            launch_receipt,
            execution_receipts,
        ):
            self.patches.enter_context(
                mock.patch.object(module, "PRODUCTION_ROOT", self.root)
            )
        self.patches.enter_context(
            mock.patch.object(finalizer, "CONTINUATION_OPTIONS", self.options)
        )
        self.patches.enter_context(
            mock.patch.object(
                final_collector, "CONTINUATION_OPTIONS", self.options
            )
        )
        self.analyzer_calls = 0
        self.verifier_calls = 0
        self.collector_calls = 0

    def tearDown(self) -> None:
        self.patches.close()
        self.temporary.cleanup()

    def _file(self, relative: str, text: str = "sealed\n") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path.resolve()

    def _launch(self, index: int) -> tuple[Path, Path]:
        nonce = f"{index:032x}"
        receipt = build_receipt(
            schema=launch_receipt.LAUNCH_SCHEMA,
            study_id=finalizer.STUDY_ID,
            scenario=launch_receipt.LAUNCH_SCENARIO,
            identities={
                "dummy": file_identity(self.dummy),
                "prelaunch_receipt": file_identity(self.dummy),
            },
            fields={"launch_nonce": nonce, "slurm_job_id": str(index)},
        )
        launch = atomic_write_receipt(
            self.root
            / f"control/launch/FIXED5_LAUNCH_{nonce}_JOB_{index}.json",
            receipt,
        )
        excluded = self._file(
            f"control/excluded/FIXED5_EXCLUDED_{nonce}.json",
            f"excluded-{index}\n",
        )
        return launch, excluded

    def _collector(self, **kwargs) -> Path:
        self.collector_calls += 1
        destination: Path = kwargs["destination"]
        destination.write_text("sealed predictions\n")
        collection_receipt = build_receipt(
            schema="synthetic-fixed5-collection/v1",
            study_id=finalizer.STUDY_ID,
            scenario="synthetic_fixed5_collection",
            identities={
                "launch_receipt": file_identity(kwargs["launch"]),
                "collected_predictions": file_identity(destination),
            },
            fields={"status": "complete"},
        )
        atomic_write_receipt(
            destination.with_suffix(
                destination.suffix + ".receipt.json"
            ),
            collection_receipt,
        )
        return destination

    def _collection_verifier(self, path: Path, **_kwargs) -> dict:
        self.assertTrue(path.is_file())
        self.assertTrue(
            path.with_suffix(path.suffix + ".receipt.json").is_file()
        )
        return {}

    def _analyzer(self, *_args, **_kwargs) -> dict:
        self.analyzer_calls += 1
        return {
            "schema": analyzer.REPORT_SCHEMA,
            "semantic_report": {"classification": "synthetic"},
        }

    def _verifier(self, predictions: Path, *_args, **kwargs) -> dict:
        self.verifier_calls += 1
        analysis_path = kwargs["analyzer_report"]
        collection_receipt = kwargs["collection_receipt"]
        source_manifest = kwargs["source_manifest"]
        return {
            "schema": verifier.REPORT_SCHEMA,
            "semantic_report": {"classification": "synthetic"},
            "analyzer_comparison": {
                "requested": True,
                "match": True,
                "numeric_comparison": {},
            },
            "verification_provenance": {
                "source_manifest": file_identity(source_manifest),
                "collection_receipt": file_identity(collection_receipt),
                "collected_predictions": file_identity(predictions),
                "analyzer_report": file_identity(analysis_path),
                "independent_verifier": file_identity(verifier.__file__),
            },
        }

    def _run(
        self,
        launch: Path,
        excluded: Path,
        *,
        collector=None,
        analyzer_runner=None,
        manifest_verifier=None,
    ) -> Path:
        return finalizer.run(
            production_root=self.root,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            launch_receipt=launch,
            excluded_audit=excluded,
            continuation_options=self.options,
            manifest_verifier=manifest_verifier or (lambda _path: {}),
            state_scanner=lambda **_kwargs: self.state,
            excluded_verifier=lambda *_args, **_kwargs: {},
            collector=collector or self._collector,
            collection_verifier=self._collection_verifier,
            analyzer_runner=analyzer_runner or self._analyzer,
            independent_verifier=self._verifier,
            lock_checker=lambda _root: None,
            launch_verifier=lambda *_args, **_kwargs: {},
        )

    def test_complete_path_binds_full_ancestry(self) -> None:
        launch, excluded = self._launch(1)
        complete = self._run(launch, excluded)
        self.assertTrue(complete.is_file())
        self.assertEqual(self.collector_calls, 1)
        self.assertEqual(self.analyzer_calls, 1)
        self.assertEqual(self.verifier_calls, 1)
        receipt = load_receipt(complete)
        self.assertEqual(receipt["analyzer_invocation_count"], 1)
        self.assertEqual(
            set(receipt["identities"]["seed_chains"]),
            {str(seed) for seed in execution_receipts.ADOPTED_SEEDS},
        )
        for role in (
            "finalization_start",
            "raw_matrix",
            "collection_receipt",
            "analyzer_barrier",
            "analysis_report",
            "independent_verification_report",
            "current_launch",
            "current_excluded_audit",
            "continuation_options",
            "amendment_08",
        ):
            self.assertIn(role, receipt["identities"])
        independent = json.loads(
            (complete.parent / finalizer.VERIFICATION_NAME).read_text()
        )
        self.assertEqual(
            set(independent["finalization_provenance"]["seed_chains"]),
            {str(seed) for seed in execution_receipts.ADOPTED_SEEDS},
        )
        self.assertIn(
            "analyzer_barrier",
            independent["finalization_provenance"],
        )

    def test_double_look_crash_barrier_fails_closed(self) -> None:
        launch, excluded = self._launch(1)

        def crash(*_args, **_kwargs):
            self.analyzer_calls += 1
            raise RuntimeError("injected analyzer crash")

        with self.assertRaisesRegex(RuntimeError, "injected analyzer crash"):
            self._run(launch, excluded, analyzer_runner=crash)
        barrier = (
            self.root / "finalization/attempt_01"
            / finalizer.BARRIER_NAME
        )
        self.assertTrue(barrier.is_file())
        with self.assertRaisesRegex(
            (RuntimeError, ValueError),
            "second invocation|barrier without analysis",
        ):
            self._run(launch, excluded, analyzer_runner=crash)
        self.assertEqual(self.analyzer_calls, 1)

    def test_l1_to_l2_resume_preserves_collection(self) -> None:
        launch1, excluded1 = self._launch(1)
        calls = 0

        def stop_after_collection(_path: Path) -> dict:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("allocation L1 stopped")
            return {}

        with self.assertRaisesRegex(RuntimeError, "allocation L1 stopped"):
            self._run(
                launch1,
                excluded1,
                manifest_verifier=stop_after_collection,
            )
        collection = (
            self.root / "finalization/attempt_01"
            / finalizer.PREDICTIONS_NAME
        )
        identity = file_identity(collection)
        launch2, excluded2 = self._launch(2)
        complete = self._run(launch2, excluded2)
        self.assertTrue(complete.is_file())
        self.assertEqual(file_identity(collection), identity)
        self.assertTrue(
            (
                collection.parent
                / "FINALIZATION_RESUME_00000000000000000000000000000002.json"
            ).is_file()
        )
        receipt = load_receipt(complete)
        self.assertEqual(len(receipt["identities"]["resumes"]), 1)
        self.assertEqual(self.collector_calls, 1)

    def test_partial_collection_uses_new_attempt(self) -> None:
        launch, excluded = self._launch(1)

        def partial(**kwargs):
            destination: Path = kwargs["destination"]
            destination.write_text("partial\n")
            raise RuntimeError("collector interrupted")

        with self.assertRaisesRegex(RuntimeError, "collector interrupted"):
            self._run(launch, excluded, collector=partial)
        complete = self._run(launch, excluded)
        self.assertEqual(complete.parent.name, "attempt_02")
        self.assertTrue(
            (
                self.root / "finalization/attempt_01"
                / finalizer.PREDICTIONS_NAME
            ).is_file()
        )
        self.assertFalse(
            (
                self.root / "finalization/attempt_01"
                / finalizer.COLLECTION_RECEIPT_NAME
            ).exists()
        )

    def test_redirected_report_is_never_overwritten(self) -> None:
        launch, excluded = self._launch(1)
        external = self._file("external-report.json")

        def redirect(*_args, **_kwargs):
            attempt = self.root / "finalization/attempt_01"
            (attempt / finalizer.ANALYSIS_NAME).symlink_to(external)
            return self._analyzer()

        with self.assertRaises(FileExistsError):
            self._run(launch, excluded, analyzer_runner=redirect)
        with self.assertRaisesRegex(ValueError, "redirected"):
            self._run(launch, excluded)
        self.assertEqual(external.read_text(), "sealed\n")

    def test_excluded_contamination_stops_before_collection(self) -> None:
        launch, excluded = self._launch(1)
        (self.root / "fixed5_execution/seed_32006").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "excluded-seed"):
            self._run(launch, excluded)
        self.assertEqual(self.collector_calls, 0)

    def test_resume_sequence_is_chronological_not_nonce_lexical(self) -> None:
        launch_64, excluded_64 = self._launch(0x64)
        complete = self._run(launch_64, excluded_64)
        analysis_identity = file_identity(
            complete.parent / finalizer.ANALYSIS_NAME
        )
        verification_identity = file_identity(
            complete.parent / finalizer.VERIFICATION_NAME
        )
        launch_c8, excluded_c8 = self._launch(0xC8)
        self._run(launch_c8, excluded_c8)
        launch_01, excluded_01 = self._launch(0x01)
        self._run(launch_01, excluded_01)
        attempts = finalizer._scan_attempts(self.root)
        ordered = finalizer._ordered_resume_paths(attempts[0])
        self.assertEqual(
            [path.name for path in ordered],
            [
                "FINALIZATION_RESUME_"
                "000000000000000000000000000000c8.json",
                "FINALIZATION_RESUME_"
                "00000000000000000000000000000001.json",
            ],
        )
        self.assertEqual(
            [load_receipt(path)["resume_sequence"] for path in ordered],
            [1, 2],
        )
        self.assertEqual(
            file_identity(complete.parent / finalizer.ANALYSIS_NAME),
            analysis_identity,
        )
        self.assertEqual(
            file_identity(complete.parent / finalizer.VERIFICATION_NAME),
            verification_identity,
        )
        self.assertEqual(self.analyzer_calls, 1)
        self.assertEqual(self.verifier_calls, 3)

    def test_completed_new_launch_recompute_mismatch_fails_without_regeneration(
        self,
    ) -> None:
        launch1, excluded1 = self._launch(10)
        complete = self._run(launch1, excluded1)
        analysis_identity = file_identity(
            complete.parent / finalizer.ANALYSIS_NAME
        )
        verification_identity = file_identity(
            complete.parent / finalizer.VERIFICATION_NAME
        )
        launch2, excluded2 = self._launch(11)

        def mismatch(predictions: Path, *_args, **kwargs) -> dict:
            report = self._verifier(predictions, **kwargs)
            report["semantic_report"] = {"classification": "different"}
            return report

        with self.assertRaisesRegex(
            ValueError, "independent recomputation differs"
        ):
            finalizer.run(
                production_root=self.root,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
                fixed48_source_manifest=self.fixed48,
                authorization_manifest=self.authorization,
                feasibility_gate=self.feasibility,
                launch_receipt=launch2,
                excluded_audit=excluded2,
                continuation_options=self.options,
                manifest_verifier=lambda _path: {},
                state_scanner=lambda **_kwargs: self.state,
                excluded_verifier=lambda *_args, **_kwargs: {},
                collector=self._collector,
                collection_verifier=self._collection_verifier,
                analyzer_runner=self._analyzer,
                independent_verifier=mismatch,
                lock_checker=lambda _root: None,
                launch_verifier=lambda *_args, **_kwargs: {},
            )
        self.assertEqual(
            file_identity(complete.parent / finalizer.ANALYSIS_NAME),
            analysis_identity,
        )
        self.assertEqual(
            file_identity(complete.parent / finalizer.VERIFICATION_NAME),
            verification_identity,
        )

    def test_launch_path_rejects_relative_dotdot_outside_and_symlink(
        self,
    ) -> None:
        launch, excluded = self._launch(21)
        outside = self._file("outside-launch.json", launch.read_text())
        symlink = (
            self.root
            / "control/launch/"
            "FIXED5_LAUNCH_00000000000000000000000000000016_JOB_22.json"
        )
        symlink.symlink_to(launch)
        dotdot = (
            self.root
            / "control/launch/../launch"
            / launch.name
        )
        cases = (
            Path(launch.name),
            dotdot,
            outside,
            symlink,
        )
        for candidate in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    self._run(candidate, excluded)

    def test_orphan_resume_only_attempt_fails_closed(self) -> None:
        launch, excluded = self._launch(30)
        attempt = self.root / "finalization/attempt_01"
        attempt.mkdir(parents=True)
        (attempt / (
            "FINALIZATION_RESUME_"
            "0000000000000000000000000000001e.json"
        )).write_text("orphan\n")
        with self.assertRaisesRegex(ValueError, "without START"):
            self._run(launch, excluded)

    def test_complete_values_inspected_must_be_exact_true(self) -> None:
        launch, excluded = self._launch(31)
        complete = self._run(launch, excluded)
        receipt = load_receipt(complete)
        receipt["values_inspected"] = False
        atomic_write_receipt(complete, receipt)
        with self.assertRaisesRegex(ValueError, "COMPLETE semantics"):
            self._run(launch, excluded)

    def _complete_after_collection_recovery(self) -> tuple[Path, Path, Path]:
        launch1, excluded1 = self._launch(0x40)
        calls = 0

        def stop_after_collection(_path: Path) -> dict:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("L1 stopped after collection")
            return {}

        with self.assertRaisesRegex(RuntimeError, "L1 stopped"):
            self._run(
                launch1,
                excluded1,
                manifest_verifier=stop_after_collection,
            )
        launch2, excluded2 = self._launch(0x41)
        complete = self._run(launch2, excluded2)
        return complete, launch2, excluded2

    def test_prelook_resume_cutoff_cannot_be_erased_by_forgery(self) -> None:
        complete, _launch2, _excluded2 = (
            self._complete_after_collection_recovery()
        )
        verification_path = (
            complete.parent / finalizer.VERIFICATION_NAME
        )
        verification = json.loads(verification_path.read_text())
        self.assertIn(
            "resumes", verification["finalization_provenance"]
        )
        del verification["finalization_provenance"]["resumes"]
        verification_path.write_text(
            json.dumps(
                verification,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        complete_receipt = load_receipt(complete)
        self.assertIn("resumes", complete_receipt["identities"])
        del complete_receipt["identities"]["resumes"]
        complete_receipt["identities"][
            "independent_verification_report"
        ] = file_identity(verification_path)
        complete_receipt["topology_sha256"] = topology_sha256(
            complete_receipt["identities"]
        )
        atomic_write_receipt(complete, complete_receipt)

        launch3, excluded3 = self._launch(0x42)
        with self.assertRaisesRegex(
            ValueError,
            "verification .*topology|verification RESUME|COMPLETE RESUME",
        ):
            self._run(launch3, excluded3)

    def test_postcomplete_resume_may_extend_immutable_prelook_cutoff(
        self,
    ) -> None:
        complete, _launch2, _excluded2 = (
            self._complete_after_collection_recovery()
        )
        barrier = load_receipt(complete.parent / finalizer.BARRIER_NAME)
        verification = json.loads(
            (complete.parent / finalizer.VERIFICATION_NAME).read_text()
        )
        complete_receipt = load_receipt(complete)
        cutoff = barrier["identities"]["resumes"]
        self.assertEqual(
            verification["finalization_provenance"]["resumes"],
            cutoff,
        )
        self.assertEqual(complete_receipt["identities"]["resumes"], cutoff)

        analysis_identity = file_identity(
            complete.parent / finalizer.ANALYSIS_NAME
        )
        verification_identity = file_identity(
            complete.parent / finalizer.VERIFICATION_NAME
        )
        launch3, excluded3 = self._launch(0x42)
        self._run(launch3, excluded3)
        attempt = finalizer._scan_attempts(self.root)[0]
        self.assertEqual(len(finalizer._ordered_resume_paths(attempt)), 2)
        self.assertEqual(
            file_identity(complete.parent / finalizer.ANALYSIS_NAME),
            analysis_identity,
        )
        self.assertEqual(
            file_identity(complete.parent / finalizer.VERIFICATION_NAME),
            verification_identity,
        )
        self.assertEqual(
            load_receipt(complete)["identities"]["resumes"],
            cutoff,
        )

    def test_resume_artifact_snapshot_cannot_omit_sealed_collection(
        self,
    ) -> None:
        launch1, excluded1 = self._launch(0x50)
        calls = 0

        def stop_after_collection(_path: Path) -> dict:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("L1 collection stop")
            return {}

        with self.assertRaisesRegex(RuntimeError, "collection stop"):
            self._run(
                launch1,
                excluded1,
                manifest_verifier=stop_after_collection,
            )
        launch2, excluded2 = self._launch(0x51)
        with mock.patch.object(
            finalizer,
            "_publish_barrier",
            side_effect=RuntimeError("L2 post-RESUME stop"),
        ):
            with self.assertRaisesRegex(RuntimeError, "post-RESUME"):
                self._run(launch2, excluded2)
        resume_path = (
            self.root / "finalization/attempt_01"
            / "FINALIZATION_RESUME_"
            "00000000000000000000000000000051.json"
        )
        for artifact_name in (
            finalizer.PREDICTIONS_NAME,
            finalizer.COLLECTION_RECEIPT_NAME,
        ):
            os.utime(resume_path.parent / artifact_name, None)
        resume = load_receipt(resume_path)
        resume["recovery_phase"] = "start_only"
        resume["identities"]["existing_finalization_artifacts"] = {
            finalizer.START_NAME: file_identity(
                resume_path.parent / finalizer.START_NAME
            )
        }
        resume["topology_sha256"] = topology_sha256(
            resume["identities"]
        )
        atomic_write_receipt(resume_path, resume)
        with self.assertRaisesRegex(
            ValueError,
            "phase differs|existing artifacts|omitted a pre-existing",
        ):
            self._run(launch2, excluded2)

    def test_valid_start_only_recovery(self) -> None:
        launch1, excluded1 = self._launch(0x60)

        def stop_before_collection(**_kwargs):
            raise RuntimeError("collector never started")

        with self.assertRaisesRegex(RuntimeError, "never started"):
            self._run(
                launch1,
                excluded1,
                collector=stop_before_collection,
            )
        launch2, excluded2 = self._launch(0x61)
        complete = self._run(launch2, excluded2)
        resume = load_receipt(
            complete.parent
            / "FINALIZATION_RESUME_"
            "00000000000000000000000000000061.json"
        )
        self.assertEqual(resume["recovery_phase"], "start_only")
        self.assertTrue(complete.is_file())

    def test_valid_postanalysis_preverifier_recovery(self) -> None:
        launch1, excluded1 = self._launch(0x70)
        verifier_calls = 0

        def crash_verifier(*_args, **_kwargs):
            nonlocal verifier_calls
            verifier_calls += 1
            raise RuntimeError("verifier publication crash")

        with self.assertRaisesRegex(RuntimeError, "verifier publication"):
            finalizer.run(
                production_root=self.root,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
                fixed48_source_manifest=self.fixed48,
                authorization_manifest=self.authorization,
                feasibility_gate=self.feasibility,
                launch_receipt=launch1,
                excluded_audit=excluded1,
                continuation_options=self.options,
                manifest_verifier=lambda _path: {},
                state_scanner=lambda **_kwargs: self.state,
                excluded_verifier=lambda *_args, **_kwargs: {},
                collector=self._collector,
                collection_verifier=self._collection_verifier,
                analyzer_runner=self._analyzer,
                independent_verifier=crash_verifier,
                lock_checker=lambda _root: None,
                launch_verifier=lambda *_args, **_kwargs: {},
            )
        self.assertEqual(self.analyzer_calls, 1)
        launch2, excluded2 = self._launch(0x71)
        complete = self._run(launch2, excluded2)
        self.assertTrue(complete.is_file())
        self.assertEqual(self.analyzer_calls, 1)
        resume = load_receipt(
            complete.parent
            / "FINALIZATION_RESUME_"
            "00000000000000000000000000000071.json"
        )
        self.assertEqual(resume["recovery_phase"], "analysis_sealed")

    def test_valid_postverifier_precomplete_recovery(self) -> None:
        launch1, excluded1 = self._launch(0x80)
        with mock.patch.object(
            finalizer,
            "_publish_complete",
            side_effect=RuntimeError("completion publication crash"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "completion publication"
            ):
                self._run(launch1, excluded1)
        self.assertEqual(self.analyzer_calls, 1)
        self.assertEqual(self.verifier_calls, 1)
        launch2, excluded2 = self._launch(0x81)
        complete = self._run(launch2, excluded2)
        self.assertTrue(complete.is_file())
        self.assertEqual(self.analyzer_calls, 1)
        self.assertEqual(self.verifier_calls, 1)
        resume = load_receipt(
            complete.parent
            / "FINALIZATION_RESUME_"
            "00000000000000000000000000000081.json"
        )
        self.assertEqual(
            resume["recovery_phase"], "verification_sealed"
        )

    def test_verification_prefix_rejects_forged_extra_resume(self) -> None:
        launch, excluded = self._launch(0x90)
        complete = self._run(launch, excluded)
        verification_path = (
            complete.parent / finalizer.VERIFICATION_NAME
        )
        verification = json.loads(verification_path.read_text())
        verification["finalization_provenance"]["resumes"] = {
            "resume_0001_forged": file_identity(self.dummy)
        }
        verification_path.write_text(
            json.dumps(
                verification,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        complete_receipt = load_receipt(complete)
        complete_receipt["identities"][
            "independent_verification_report"
        ] = file_identity(verification_path)
        complete_receipt["topology_sha256"] = topology_sha256(
            complete_receipt["identities"]
        )
        atomic_write_receipt(complete, complete_receipt)
        with self.assertRaisesRegex(
            ValueError, "verification RESUME ancestry"
        ):
            self._run(launch, excluded)


if __name__ == "__main__":
    unittest.main()
