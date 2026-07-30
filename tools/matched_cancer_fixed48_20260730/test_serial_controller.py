from __future__ import annotations

import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
)

from .diag_contract import STUDY_ID, scenario_for
from . import serial_controller as controller
from .serial_controller import (
    CANARY_SEEDS,
    ControllerBusyError,
    PROTOCOL,
    REMAINDER_SEEDS,
    SUCCESS_NAME,
    SUCCESS_SCHEMA,
    run_controller,
    seeds_for_mode,
    verify_one_gpu_environment,
)


GPU_ENV = {
    "SLURM_JOB_ID": "123",
    "SLURM_GPUS_ON_NODE": "1",
    "SLURM_JOB_GPUS": "0",
    "CUDA_VISIBLE_DEVICES": "0",
    "SLURM_NTASKS": "1",
}


class SerialControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "production"
        self.manifest = Path(self.temporary.name) / "manifest.json"
        self.authorization = Path(self.temporary.name) / "authorization.json"
        self.manifest.write_text("{}\n")
        self.authorization.write_text("{}\n")
        self.feasibility = (
            self.root / "control/FEASIBILITY_GATE_RECEIPT.json"
        )
        self.feasibility.parent.mkdir(parents=True)
        self.feasibility.write_text("{}\n")
        self.manifest_calls = 0
        self.phase_patch = mock.patch.object(
            controller, "verify_phase", side_effect=self._verify_phase
        )
        self.phase_patch.start()
        self.feasibility_patch = mock.patch.object(
            controller, "verify_feasibility", return_value={}
        )
        self.feasibility_patch.start()

    def tearDown(self) -> None:
        self.feasibility_patch.stop()
        self.phase_patch.stop()
        self.temporary.cleanup()

    def _verify_manifest(self, _: Path) -> dict:
        self.manifest_calls += 1
        return {}

    @staticmethod
    def _verify_phase(path: Path | str, *, expected_fm_seed: int) -> dict:
        parent = Path(path).parent
        return {
            "identities": {
                "deployment_gate": file_identity(
                    parent / "DEPLOYMENT_GATE_RECEIPT.json"
                ),
                "loader_root": file_identity(
                    parent / "run/LOADER_ROOT_RECEIPT.json"
                ),
                "structural_audit": file_identity(
                    parent / "DIAGNOSTIC_STRUCTURAL_AUDIT_RECEIPT.json"
                ),
                "collection": file_identity(
                    parent / f"seed{expected_fm_seed}_predictions.jsonl"
                ),
                "collection_receipt": file_identity(
                    parent
                    / f"seed{expected_fm_seed}_predictions.jsonl.receipt.json"
                ),
            }
        }

    @staticmethod
    def _artifact(parent: Path, name: str) -> dict:
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n")
        return file_identity(path)

    def _seal(
        self,
        seed: int,
        calibration: Path,
        diagnostic: Path,
        _manifest: Path,
        _authorization: Path,
        feasibility: Path,
    ) -> None:
        calibration.mkdir(parents=True, exist_ok=False)
        diagnostic.mkdir(parents=True, exist_ok=False)
        identities = {
            "source_manifest": file_identity(self.manifest),
            "feasibility_gate": file_identity(feasibility),
            "execution_protocol": file_identity(PROTOCOL),
            "calibration_root": self._artifact(
                calibration, "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
            ),
            "calibration_audit": self._artifact(
                calibration, "INDEPENDENT_CALIBRATION_AUDIT_RECEIPT.json"
            ),
            "calibration_completions": {
                arm: self._artifact(
                    calibration, f"{arm}/COMPLETION_RECEIPT.json"
                )
                for arm in ("B", "P", "H")
            },
            "diagnostic_gate": self._artifact(
                diagnostic, "DEPLOYMENT_GATE_RECEIPT.json"
            ),
            "loader_root": self._artifact(
                diagnostic, "run/LOADER_ROOT_RECEIPT.json"
            ),
            "diagnostic_phase": self._artifact(
                diagnostic, "DIAGNOSTIC_PHASE_RECEIPT.json"
            ),
            "diagnostic_structural_audit": self._artifact(
                diagnostic, "DIAGNOSTIC_STRUCTURAL_AUDIT_RECEIPT.json"
            ),
            "per_seed_collection": self._artifact(
                diagnostic, f"seed{seed}_predictions.jsonl"
            ),
            "per_seed_collection_receipt": self._artifact(
                diagnostic, f"seed{seed}_predictions.jsonl.receipt.json"
            ),
        }
        receipt = build_receipt(
            schema=SUCCESS_SCHEMA,
            study_id=STUDY_ID,
            scenario=scenario_for(seed),
            identities=identities,
            fields={
                "fm_seed": seed,
                "status": "complete",
                "legacy_outputs_used": False,
                "values_inspected": False,
            },
        )
        atomic_write_receipt(diagnostic / SUCCESS_NAME, receipt)

    def _run(self, mode: str, worker=None):
        return run_controller(
            mode=mode,
            production_root=self.root,
            source_manifest=self.manifest,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            seed_worker=worker or self._seal,
            environment=GPU_ENV,
            manifest_verifier=self._verify_manifest,
            cuda_device_count=lambda: 1,
        )

    def test_exact_ranges(self) -> None:
        self.assertEqual(seeds_for_mode("canary"), (32001,))
        self.assertEqual(seeds_for_mode("remainder"), tuple(range(32002, 32049)))
        self.assertEqual(CANARY_SEEDS + REMAINDER_SEEDS, tuple(range(32001, 32049)))
        with self.assertRaises(ValueError):
            seeds_for_mode("32001-32048")

    def test_canary_success_and_verified_resume(self) -> None:
        self.assertEqual(self._run("canary"), (32001,))
        first_calls = self.manifest_calls
        self.assertEqual(self._run("canary"), ())
        self.assertGreater(self.manifest_calls, first_calls)
        success = (
            self.root
            / "diagnostic/seed_32001/attempt_01"
            / SUCCESS_NAME
        )
        self.assertTrue(success.is_file())

    def test_failure_stops_and_retry_uses_new_attempt(self) -> None:
        def fail(_seed, calibration, diagnostic, *_args):
            calibration.mkdir(parents=True)
            diagnostic.mkdir(parents=True)
            raise RuntimeError("injected failure")

        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            self._run("canary", fail)
        partial_calibration = self.root / "calibration/seed_32001/attempt_01"
        partial_diagnostic = self.root / "diagnostic/seed_32001/attempt_01"
        self.assertTrue(partial_calibration.is_dir())
        self.assertTrue(partial_diagnostic.is_dir())
        self.assertEqual(self._run("canary"), (32001,))
        self.assertTrue(partial_calibration.is_dir())
        self.assertTrue(partial_diagnostic.is_dir())
        self.assertTrue(
            (self.root / "diagnostic/seed_32001/attempt_02" / SUCCESS_NAME).is_file()
        )

    def test_retry_uses_lowest_unused_attempt_in_gapped_history(self) -> None:
        stale = self.root / "calibration/seed_32001/attempt_02"
        stale.mkdir(parents=True)
        (stale / "partial.txt").write_text("preserved\n")
        self.assertEqual(self._run("canary"), (32001,))
        self.assertTrue(stale.is_dir())
        self.assertTrue(
            (
                self.root
                / "diagnostic/seed_32001/attempt_01"
                / SUCCESS_NAME
            ).is_file()
        )
        self.assertFalse(
            (self.root / "diagnostic/seed_32001/attempt_03").exists()
        )

    def test_remainder_stops_at_first_failure_without_skipping(self) -> None:
        self._run("canary")
        called: list[int] = []

        def worker(seed, *args):
            called.append(seed)
            if seed == 32003:
                args[0].mkdir(parents=True)
                args[1].mkdir(parents=True)
                raise RuntimeError("stop")
            self._seal(seed, *args)

        with self.assertRaisesRegex(RuntimeError, "stop"):
            self._run("remainder", worker)
        self.assertEqual(called, [32002, 32003])
        self.assertFalse((self.root / "calibration/seed_32004").exists())
        self.assertTrue((self.root / "calibration/seed_32003/attempt_01").is_dir())

    def test_requires_verified_canary_before_remainder(self) -> None:
        with self.assertRaisesRegex(ValueError, "canary"):
            self._run("remainder")

    def test_authorization_drift_during_worker_fails(self) -> None:
        def tamper(
            seed, calibration, diagnostic, manifest, authorization, feasibility
        ):
            self._seal(
                seed,
                calibration,
                diagnostic,
                manifest,
                authorization,
                feasibility,
            )
            authorization.write_text('{"changed":true}\n')

        with self.assertRaisesRegex(ValueError, "authorization manifest drifted"):
            self._run("canary", tamper)

    def test_nonblocking_controller_lock(self) -> None:
        control = self.root / "control"
        control.mkdir(parents=True, exist_ok=True)
        lock_path = control / "serial_controller.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(ControllerBusyError):
                self._run("canary")

    def test_gpu_contract_rejects_arrays_and_multiplicity(self) -> None:
        verify_one_gpu_environment(GPU_ENV, cuda_device_count=lambda: 1)
        mutations = {
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_GPUS_ON_NODE": "2",
            "SLURM_JOB_GPUS": "0,1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_NTASKS": "2",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                environment = dict(GPU_ENV)
                environment[key] = value
                with self.assertRaises(ValueError):
                    verify_one_gpu_environment(
                        environment, cuda_device_count=lambda: 1
                    )
        with self.assertRaisesRegex(ValueError, "observed 0"):
            verify_one_gpu_environment(
                GPU_ENV, cuda_device_count=lambda: 0
            )


class SubmissionScriptTests(unittest.TestCase):
    def test_bound_worker_uses_frozen_python_module_invocation(self) -> None:
        package = Path(__file__).parent
        executable = package / "seed_worker.py"
        manifest = package / "test-manifest-placeholder.json"
        receipt = {
            "identities": {
                "sources": {
                    "fixed48.seed_worker": file_identity(executable)
                }
            }
        }
        worker = controller._bound_worker(
            executable,
            manifest=manifest,
            manifest_receipt=receipt,
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            worker(
                32001,
                Path("/tmp/calibration/attempt_01"),
                Path("/tmp/diagnostic/attempt_01"),
                manifest,
                Path("/tmp/authorization.json"),
                Path("/tmp/feasibility.json"),
            )
        command = run.call_args.args[0]
        self.assertEqual(
            command[:3],
            [
                controller.sys.executable,
                "-m",
                controller.SEED_WORKER_MODULE,
            ],
        )
        self.assertNotEqual(command[0], str(executable))
        self.assertEqual(run.call_args.kwargs["cwd"], controller.REPO)

    def test_no_array_dependency_or_bulk_submission(self) -> None:
        package = Path(__file__).parent
        driver = (package / "serial_fixed48.sbatch").read_text()
        submit = (package / "safe_submit.sh").read_text()
        self.assertNotRegex(driver, r"(?im)^#SBATCH\s+--array")
        self.assertNotRegex(driver, r"(?im)^#SBATCH\s+--dependency")
        self.assertEqual(driver.count("#SBATCH --gpus-per-task=1"), 1)
        self.assertEqual(submit.count('"$SBATCH" --parsable'), 1)
        self.assertNotIn("--array", submit)
        self.assertNotIn("--dependency", submit)
        self.assertNotIn("afterok", submit)


if __name__ == "__main__":
    unittest.main()
