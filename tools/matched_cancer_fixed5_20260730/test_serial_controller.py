from __future__ import annotations

from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.matched_cancer_stage_20260730.receipts import file_identity

from . import execution_receipts as execution
from . import launch_receipt
from . import serial_controller as controller


BASE_GPU_ENV = {
    "SLURM_GPUS_ON_NODE": "1",
    "SLURM_JOB_GPUS": "0",
    "CUDA_VISIBLE_DEVICES": "0",
    "SLURM_NTASKS": "1",
    "SLURM_JOB_NAME": "main_1gpu",
    "FIXED5_SLURM_COMMENT": "matched_cancer_fixed48_20260730",
}


class SerialControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "production"
        self.fixed48 = self.root / "control/FIXED48_SOURCE_MANIFEST_V2.json"
        self.fixed5 = self.root / "control/FIXED5_SOURCE_MANIFEST_V1.json"
        self.authorization = (
            self.root / "authorization/AUTHORIZATION_MANIFEST_V3.json"
        )
        self.adoption = (
            self.root
            / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
        )
        self.feasibility = (
            self.root / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
        )
        for path in (
            self.fixed48,
            self.fixed5,
            self.authorization,
            self.adoption,
            self.feasibility,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{path.name}\n")
        self._make_seed1()
        self.patches = ExitStack()
        for module in (controller, launch_receipt, execution):
            self.patches.enter_context(
                mock.patch.object(
                    module, "PRODUCTION_ROOT", self.root.resolve()
                )
            )
        self.launch_index = 0
        self.control_calls: list[str] = []
        self.finalizer_calls = 0

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
        (diagnostic / controller.SUCCESS_NAME).write_text(
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

    def _fixed48_verifier(self, _path: Path) -> dict:
        self.control_calls.append("fixed48")
        return {"identities": {"sources": {}}}

    def _fixed5_verifier(self, _path: Path) -> dict:
        self.control_calls.append("fixed5")
        return {}

    def _authorization_verifier(self, _path: Path) -> dict:
        self.control_calls.append("authorization")
        return {}

    def _feasibility_verifier(self, _path: Path, **_kwargs) -> dict:
        self.control_calls.append("feasibility")
        return {}

    def _adoption_verifier(self, _path: Path, **_kwargs) -> dict:
        self.control_calls.append("adoption")
        return {}

    def _new_environment(self) -> dict[str, str]:
        self.launch_index += 1
        nonce = f"{self.launch_index:032x}"
        job_id = str(1000 + self.launch_index)
        prelaunch = (
            self.root
            / f"control/prelaunch/FIXED5_PRELAUNCH_{nonce}.json"
        )
        launch_receipt.create_prelaunch(
            prelaunch,
            launch_nonce=nonce,
            production_root=self.root,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            manifest_verifier=lambda _path: {},
            adoption_verifier=lambda *_args, **_kwargs: {},
        )
        return {
            **BASE_GPU_ENV,
            "SLURM_JOB_ID": job_id,
            "FIXED5_LAUNCH_NONCE": nonce,
            "FIXED5_PRELAUNCH_RECEIPT": str(prelaunch),
        }

    def _seal(
        self,
        seed: int,
        calibration: Path,
        diagnostic: Path,
        source_manifest: Path,
        authorization: Path,
        feasibility: Path,
    ) -> None:
        self.assertIn(seed, controller.CONTROLLER_SEEDS)
        self.assertEqual(source_manifest, self.fixed48.resolve())
        self.assertEqual(authorization, self.authorization.resolve())
        self.assertEqual(feasibility, self.feasibility.resolve())
        self.assertEqual(calibration.name, diagnostic.name)
        calibration.mkdir(parents=True, exist_ok=False)
        diagnostic.mkdir(parents=True, exist_ok=False)
        (diagnostic / controller.SUCCESS_NAME).write_text(
            f"fixed48-success-{seed}\n"
        )

    def _finalize_under_lock(self, **_kwargs) -> None:
        self.finalizer_calls += 1
        lock_path = self.root / "control/serial_controller.lock"
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(
                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
        finally:
            os.close(descriptor)

    def _run(
        self,
        worker=None,
        *,
        environment: dict[str, str] | None = None,
        success_verifier=None,
        finalizer=None,
    ):
        return controller.run_controller(
            production_root=self.root,
            source_manifest=self.fixed48,
            fixed5_source_manifest=self.fixed5,
            authorization_manifest=self.authorization,
            adoption_authorization=self.adoption,
            feasibility_gate=self.feasibility,
            seed_worker=worker or self._seal,
            environment=environment or self._new_environment(),
            cuda_device_count=lambda: 1,
            fixed48_manifest_verifier=self._fixed48_verifier,
            fixed5_manifest_verifier=self._fixed5_verifier,
            authorization_verifier=self._authorization_verifier,
            feasibility_verifier=self._feasibility_verifier,
            adoption_verifier=self._adoption_verifier,
            success_verifier=success_verifier or self._success,
            finalizer=finalizer or self._finalize_under_lock,
        )

    def test_runs_exact_four_seeds_and_finalizes_under_same_lock(self) -> None:
        called: list[int] = []

        def worker(seed, *args):
            called.append(seed)
            self._seal(seed, *args)

        self.assertEqual(
            self._run(worker), (32002, 32003, 32004, 32005)
        )
        self.assertEqual(called, [32002, 32003, 32004, 32005])
        self.assertEqual(self.finalizer_calls, 1)
        self.assertFalse(
            any(
                (self.root / family / "seed_32006").exists()
                for family in (
                    "calibration",
                    "diagnostic",
                    "fixed5_execution",
                )
            )
        )
        self.assertEqual(self._run(worker), ())
        self.assertEqual(called, [32002, 32003, 32004, 32005])
        self.assertEqual(self.finalizer_calls, 2)

    def test_worker_failure_leaves_start_and_retry_uses_new_attempt(self) -> None:
        called: list[tuple[int, str]] = []

        def fail(seed, calibration, diagnostic, *args):
            called.append((seed, calibration.name))
            calibration.mkdir(parents=True)
            diagnostic.mkdir(parents=True)
            raise RuntimeError("injected stop")

        with self.assertRaisesRegex(RuntimeError, "injected stop"):
            self._run(fail)
        self.assertEqual(called, [(32002, "attempt_01")])
        self.assertTrue(
            (
                self.root
                / "fixed5_execution/seed_32002/attempt_01"
                / execution.START_NAME
            ).is_file()
        )
        called.clear()

        def finish(seed, calibration, diagnostic, *args):
            called.append((seed, calibration.name))
            self._seal(seed, calibration, diagnostic, *args)

        self.assertEqual(
            self._run(finish), (32002, 32003, 32004, 32005)
        )
        self.assertEqual(called[0], (32002, "attempt_02"))

    def test_start_success_without_complete_is_sealed_without_rerun(self) -> None:
        calls: list[int] = []
        injected = {"raised": False}

        def worker(seed, *args):
            calls.append(seed)
            self._seal(seed, *args)

        def crash_after_success(path, *, seed, **kwargs):
            result = self._success(path, seed=seed, **kwargs)
            if seed == 32002 and not injected["raised"]:
                injected["raised"] = True
                raise RuntimeError("crash after success")
            return result

        with self.assertRaisesRegex(RuntimeError, "crash after success"):
            self._run(worker, success_verifier=crash_after_success)
        self.assertEqual(calls, [32002])
        calls.clear()
        self.assertEqual(
            self._run(worker), (32002, 32003, 32004, 32005)
        )
        self.assertEqual(calls, [32003, 32004, 32005])

    def test_bare_success_or_orphan_attempt_is_never_adopted(self) -> None:
        calibration = self.root / "calibration/seed_32002/attempt_01"
        diagnostic = self.root / "diagnostic/seed_32002/attempt_01"
        calibration.mkdir(parents=True)
        diagnostic.mkdir(parents=True)
        (diagnostic / controller.SUCCESS_NAME).write_text(
            "fixed48-success-32002\n"
        )
        worker = mock.Mock()
        with self.assertRaisesRegex(ValueError, "orphan"):
            self._run(worker)
        worker.assert_not_called()

    def test_excluded_seed_state_fails_before_worker(self) -> None:
        excluded = self.root / "fixed5_execution/seed_32006"
        excluded.mkdir(parents=True)
        worker = mock.Mock()
        with self.assertRaisesRegex(ValueError, "excluded-seed"):
            self._run(worker)
        worker.assert_not_called()

    def test_control_drift_after_start_prevents_worker(self) -> None:
        worker = mock.Mock()
        original_publish = execution.publish_start

        def tampering_publish(**kwargs):
            result = original_publish(**kwargs)
            self.adoption.write_text("drift\n")
            return result

        with mock.patch.object(
            execution, "publish_start", side_effect=tampering_publish
        ):
            with self.assertRaisesRegex(ValueError, "controls drifted"):
                self._run(worker)
        worker.assert_not_called()

    def test_reused_launch_nonce_and_path_fails(self) -> None:
        environment = self._new_environment()
        self._run(environment=environment)
        with self.assertRaises(FileExistsError):
            self._run(environment=environment)

    def test_lock_symlink_and_contention_fail_closed(self) -> None:
        lock_path = self.root / "control/serial_controller.lock"
        target = self.root / "control/other.lock"
        target.write_text("")
        lock_path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "lock may not be a symlink"):
            self._run()
        lock_path.unlink()
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(controller.ControllerBusyError):
                self._run()
        finally:
            os.close(descriptor)

    def test_environment_requires_legacy_comment_name_task_gpu_nonce(self) -> None:
        base = self._new_environment()
        mutations = {
            "SLURM_JOB_NAME": "other",
            "FIXED5_SLURM_COMMENT": launch_receipt.SUPERSEDED_COMMENT,
            "SLURM_NTASKS": "2",
            "SLURM_GPUS_ON_NODE": "2",
            "FIXED5_LAUNCH_NONCE": "bad",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                environment = dict(base)
                environment[key] = value
                with self.assertRaises(ValueError):
                    self._run(environment=environment)

    def test_preflight_fails_closed_without_finalizer(self) -> None:
        def unavailable() -> None:
            raise RuntimeError("finalizer unavailable")

        with self.assertRaisesRegex(RuntimeError, "finalizer unavailable"):
            controller.submission_preflight(
                production_root=self.root,
                source_manifest=self.fixed48,
                fixed5_source_manifest=self.fixed5,
                authorization_manifest=self.authorization,
                adoption_authorization=self.adoption,
                feasibility_gate=self.feasibility,
                fixed48_manifest_verifier=self._fixed48_verifier,
                fixed5_manifest_verifier=self._fixed5_verifier,
                authorization_verifier=self._authorization_verifier,
                feasibility_verifier=self._feasibility_verifier,
                adoption_verifier=self._adoption_verifier,
                success_verifier=self._success,
                finalizer_checker=unavailable,
            )

    def test_preflight_validates_state_without_creating_launch_state(
        self,
    ) -> None:
        controller.submission_preflight(
            production_root=self.root,
            source_manifest=self.fixed48,
            fixed5_source_manifest=self.fixed5,
            authorization_manifest=self.authorization,
            adoption_authorization=self.adoption,
            feasibility_gate=self.feasibility,
            fixed48_manifest_verifier=self._fixed48_verifier,
            fixed5_manifest_verifier=self._fixed5_verifier,
            authorization_verifier=self._authorization_verifier,
            feasibility_verifier=self._feasibility_verifier,
            adoption_verifier=self._adoption_verifier,
            success_verifier=self._success,
            finalizer_checker=lambda: None,
        )
        self.assertFalse((self.root / "control/prelaunch").exists())
        self.assertFalse((self.root / "control/launch").exists())
        self.assertFalse((self.root / "fixed5_execution").exists())

    def test_direct_production_run_fails_before_worker_without_finalizer(
        self,
    ) -> None:
        worker = mock.Mock()
        with mock.patch.object(
            controller,
            "_require_finalizer",
            side_effect=RuntimeError("finalizer unavailable"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "finalizer unavailable"
            ):
                self._run(
                    worker,
                    finalizer=controller._default_finalizer,
                )
        worker.assert_not_called()

    def test_whole_root_and_control_symlinks_fail(self) -> None:
        alias = Path(self.temporary.name) / "root-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            controller.submission_preflight(
                production_root=alias,
                source_manifest=alias / "control" / self.fixed48.name,
                fixed5_source_manifest=alias / "control" / self.fixed5.name,
                authorization_manifest=alias
                / "authorization"
                / self.authorization.name,
                adoption_authorization=alias
                / "authorization"
                / self.adoption.name,
                feasibility_gate=alias / "control" / self.feasibility.name,
                finalizer_checker=lambda: None,
            )


class SubmissionFilesTests(unittest.TestCase):
    def test_driver_uses_legacy_comment_and_one_serial_gpu_task(self) -> None:
        package = Path(__file__).parent
        driver = (package / "serial_fixed5.sbatch").read_text()
        self.assertEqual(driver.count("#SBATCH --job-name=main_1gpu"), 1)
        self.assertEqual(
            driver.count(
                "#SBATCH --comment=matched_cancer_fixed48_20260730"
            ),
            1,
        )
        self.assertNotIn(
            "#SBATCH --comment=matched_cancer_fixed5_20260730", driver
        )
        self.assertEqual(driver.count("#SBATCH --nodes=1"), 1)
        self.assertEqual(driver.count("#SBATCH --ntasks=1"), 1)
        self.assertEqual(driver.count("#SBATCH --gpus-per-task=1"), 1)
        self.assertNotRegex(driver, r"(?im)^#SBATCH\s+--array")
        self.assertNotRegex(driver, r"(?im)^#SBATCH\s+--dependency")
        self.assertNotIn("srun", driver)
        self.assertEqual(
            driver.count(
                '"$PY" -m tools.matched_cancer_fixed5_20260730.'
                "serial_controller run"
            ),
            1,
        )

    def test_safe_submit_dual_guard_nonce_receipt_and_one_sbatch(self) -> None:
        package = Path(__file__).parent
        submit = (package / "safe_submit.sh").read_text()
        legacy = (
            package.parents[0]
            / "matched_cancer_fixed48_20260730"
            / "safe_submit.sh"
        ).read_text()
        self.assertIn(
            "FIXED48_COMMENT=matched_cancer_fixed48_20260730", submit
        )
        self.assertIn(
            "FIXED5_COMMENT=matched_cancer_fixed5_20260730", submit
        )
        self.assertIn("-t PENDING,RUNNING", submit)
        self.assertIn(
            '[[ "$comment" == "$FIXED48_COMMENT" || '
            '"$comment" == "$FIXED5_COMMENT" ]]',
            submit,
        )
        self.assertIn(
            "fixed48_execution/submission/safe_submit.lock", submit
        )
        self.assertIn('LOCK="$CONTROL/safe_submit.lock"', legacy)
        self.assertIn("launch_receipt new-nonce", submit)
        self.assertIn("launch_receipt \\\n  create-prelaunch", submit)
        self.assertIn("FIXED5_PRELAUNCH_${LAUNCH_NONCE}.json", submit)
        self.assertIn("FIXED5_LAUNCH_NONCE=$LAUNCH_NONCE", submit)
        self.assertIn("FIXED5_PRELAUNCH_RECEIPT=$PRELAUNCH", submit)
        self.assertIn(
            "FIXED5_SLURM_COMMENT=$FIXED48_COMMENT", submit
        )
        self.assertEqual(submit.count('"$SBATCH" --parsable'), 1)
        self.assertNotIn("--array", submit)
        self.assertNotIn("--dependency", submit)

    def test_bound_worker_remains_frozen_fixed48_module(self) -> None:
        executable = (
            Path(__file__).parents[1]
            / "matched_cancer_fixed48_20260730/seed_worker.py"
        )
        self.assertEqual(
            controller.fixed48_controller.SEED_WORKER_MODULE,
            "tools.matched_cancer_fixed48_20260730.seed_worker",
        )
        self.assertTrue(executable.is_file())
        self.assertEqual(execution.FROZEN_WORKER, executable.resolve())
        self.assertEqual(
            file_identity(execution.FROZEN_WORKER),
            file_identity(executable),
        )


if __name__ == "__main__":
    unittest.main()
