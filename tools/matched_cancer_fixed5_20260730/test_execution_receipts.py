from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import execution_receipts as execution
from . import launch_receipt


class ExecutionReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "production"
        self.fixed5 = self.root / "control/FIXED5_SOURCE_MANIFEST_V1.json"
        self.adoption = (
            self.root
            / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
        )
        self.fixed48 = self.root / "control/FIXED48_SOURCE_MANIFEST_V2.json"
        self.authorization = (
            self.root / "authorization/AUTHORIZATION_MANIFEST_V3.json"
        )
        self.feasibility = (
            self.root / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
        )
        for path in (
            self.fixed5,
            self.adoption,
            self.fixed48,
            self.authorization,
            self.feasibility,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{path.name}\n")
        self.nonce = "0123456789abcdef0123456789abcdef"
        self.patches = ExitStack()
        for module in (launch_receipt, execution):
            self.patches.enter_context(
                mock.patch.object(
                    module, "PRODUCTION_ROOT", self.root.resolve()
                )
            )
        self.prelaunch = launch_receipt.create_prelaunch(
            self.root
            / f"control/prelaunch/FIXED5_PRELAUNCH_{self.nonce}.json",
            launch_nonce=self.nonce,
            production_root=self.root,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            manifest_verifier=lambda _path: {},
            adoption_verifier=lambda *_args, **_kwargs: {},
        )
        self.launch = launch_receipt.create_launch(
            self.root
            / f"control/launch/FIXED5_LAUNCH_{self.nonce}_JOB_123.json",
            launch_nonce=self.nonce,
            slurm_job_id="123",
            prelaunch_receipt=self.prelaunch,
            production_root=self.root,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
        )
        self._make_seed1()

    def tearDown(self) -> None:
        self.patches.close()
        self.temporary.cleanup()

    def _make_seed1(self) -> None:
        name = "attempt_03"
        (self.root / "calibration/seed_32001" / name).mkdir(
            parents=True
        )
        diagnostic = self.root / "diagnostic/seed_32001" / name
        diagnostic.mkdir(parents=True)
        (diagnostic / execution.SUCCESS_NAME).write_text(
            "fixed48-success-32001\n"
        )

    def _success(
        self,
        path: Path,
        *,
        seed: int,
        source_manifest: Path,
        production_root: Path,
    ) -> dict:
        self.assertEqual(source_manifest, self.fixed48.resolve())
        self.assertEqual(production_root, self.root.resolve())
        self.assertEqual(
            path.read_text(), f"fixed48-success-{seed}\n"
        )
        return {"fm_seed": seed}

    def _start(self, seed: int, number: int) -> Path:
        attempt = execution.reserve_attempt(
            production_root=self.root,
            seed=seed,
            attempt_number=number,
        )
        return execution.publish_start(
            production_root=self.root,
            seed=seed,
            attempt_name=attempt.name,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            prelaunch_receipt=self.prelaunch,
            launch=self.launch,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
        )

    def _worker_success(self, seed: int, number: int) -> Path:
        name = f"attempt_{number:02d}"
        (self.root / "calibration" / f"seed_{seed}" / name).mkdir(
            parents=True
        )
        diagnostic = self.root / "diagnostic" / f"seed_{seed}" / name
        diagnostic.mkdir(parents=True)
        success = diagnostic / execution.SUCCESS_NAME
        success.write_text(f"fixed48-success-{seed}\n")
        return success

    def _complete(self, seed: int, number: int) -> Path:
        success = self._worker_success(seed, number)
        return execution.publish_complete(
            production_root=self.root,
            seed=seed,
            attempt_name=f"attempt_{number:02d}",
            fixed48_success=success,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            prelaunch_receipt=self.prelaunch,
            launch=self.launch,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            success_verifier=self._success,
        )

    def _scan(self) -> execution.StudyState:
        return execution.scan_state(
            production_root=self.root,
            fixed48_source_manifest=self.fixed48,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            success_verifier=self._success,
        )

    def test_start_then_success_then_complete_is_required(self) -> None:
        start = self._start(32002, 1)
        self.assertTrue(start.is_file())
        state = self._scan()
        self.assertEqual(state.completed, (32001,))
        self.assertEqual(state.used_attempts[32002], (1,))
        success = self._worker_success(32002, 1)
        state = self._scan()
        self.assertEqual(state.resumable, {32002: 1})
        complete = execution.publish_complete(
            production_root=self.root,
            seed=32002,
            attempt_name="attempt_01",
            fixed48_success=success,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            prelaunch_receipt=self.prelaunch,
            launch=self.launch,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            success_verifier=self._success,
        )
        self.assertTrue(complete.is_file())
        self.assertEqual(self._scan().completed, (32001, 32002))

    def test_orphan_worker_attempt_and_bare_success_fail(self) -> None:
        self._worker_success(32002, 1)
        with self.assertRaisesRegex(ValueError, "orphan"):
            self._scan()

    def test_interrupted_start_consumes_attempt_and_retry_is_new(self) -> None:
        self._start(32002, 1)
        state = self._scan()
        self.assertEqual(execution.next_attempt_number(state, 32002), 2)
        self._start(32002, 2)
        self._complete(32002, 2)
        self.assertEqual(self._scan().completed, (32001, 32002))

    def test_state_after_success_fails_closed(self) -> None:
        self._start(32002, 1)
        self._worker_success(32002, 1)
        execution.reserve_attempt(
            production_root=self.root, seed=32002, attempt_number=2
        )
        with self.assertRaisesRegex(ValueError, "after an already successful"):
            self._scan()

    def test_unexpected_or_temporary_plus_canonical_topology_fails(self) -> None:
        start = self._start(32002, 1)
        attempt = start.parent
        (attempt / "unexpected.txt").write_text("bad\n")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self._scan()

    def test_complete_without_start_or_success_fails(self) -> None:
        attempt = execution.reserve_attempt(
            production_root=self.root,
            seed=32002,
            attempt_number=1,
        )
        (attempt / execution.COMPLETE_NAME).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "without start"):
            self._scan()

    def test_symlinked_start_receipt_fails(self) -> None:
        start = self._start(32002, 1)
        external = self.root / "external-start.json"
        external.write_bytes(start.read_bytes())
        start.unlink()
        start.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "invalid execution-attempt"):
            self._scan()

    def test_excluded_seed_any_path_family_fails(self) -> None:
        for family in ("calibration", "diagnostic", "fixed5_execution"):
            with self.subTest(family=family):
                candidate = self.root / family / "seed_32006"
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.mkdir()
                try:
                    with self.assertRaisesRegex(ValueError, "excluded-seed"):
                        execution.scan_excluded_state(self.root)
                finally:
                    candidate.rmdir()

    def test_excluded_seed_broken_symlink_fails(self) -> None:
        candidate = self.root / "diagnostic/seed_32048"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "excluded-seed"):
            execution.scan_excluded_state(self.root)

    def test_adopted_family_or_seed_root_symlink_fails(self) -> None:
        for family in ("calibration", "diagnostic", "fixed5_execution"):
            with self.subTest(family=family):
                original = self.root / family
                redirected = self.root / f"{family}_redirected"
                if original.exists():
                    original.rename(redirected)
                else:
                    redirected.mkdir()
                original.symlink_to(redirected, target_is_directory=True)
                try:
                    with self.assertRaisesRegex(
                        ValueError, "ancestry is not canonical"
                    ):
                        self._scan()
                finally:
                    original.unlink()
                    if redirected.exists():
                        redirected.rename(original)

    def test_all_five_chains_seal_one_per_launch_excluded_audit(self) -> None:
        for seed in execution.CONTROLLER_SEEDS:
            self._start(seed, 1)
            self._complete(seed, 1)
        state = self._scan()
        self.assertEqual(state.completed, execution.ADOPTED_SEEDS)
        audit = execution.publish_excluded_audit(
            production_root=self.root,
            launch_nonce=self.nonce,
            launch=self.launch,
            state=state,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
        )
        execution.verify_excluded_audit(
            audit,
            production_root=self.root,
            launch_nonce=self.nonce,
            launch=self.launch,
            state=state,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
        )
        with self.assertRaises(FileExistsError):
            execution.publish_excluded_audit(
                production_root=self.root,
                launch_nonce=self.nonce,
                launch=self.launch,
                state=state,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
            )


if __name__ == "__main__":
    unittest.main()
