from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
)

from .diag_contract import STUDY_ID, scenario_for
from .seed_worker import (
    CALIBRATION_AUDIT_SCHEMA,
    CALIBRATION_ROOT_SCHEMA,
    COMPLETION_SCHEMA,
    _verify_calibration,
)


RUNS = ("slot1_plain", "slot1_fair", "B", "H", "P")


class SeedWorkerCalibrationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = 32001
        self.scenario = scenario_for(self.seed)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _file(self, path: Path, text: str = "fixture\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _completion(self, run: str) -> Path:
        path = self.root / run / "COMPLETION_RECEIPT.json"
        checkpoint = self._file(self.root / run / "latest.pt")
        receipt = build_receipt(
            schema=COMPLETION_SCHEMA,
            study_id=STUDY_ID,
            scenario=self.scenario,
            identities={"latest_checkpoint": file_identity(checkpoint)},
            fields={
                "status": "complete",
                "steps_completed": 781,
                "tile_presentations": 99_968,
            },
        )
        return atomic_write_receipt(path, receipt)

    def _fixture(self, *, wrong_root_role: bool = False) -> None:
        contract = self._file(self.root / "contract.json")
        replay = self._file(self.root / "replay.json")
        completions = {run: self._completion(run) for run in RUNS}
        root_path = self.root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
        root_receipt = build_receipt(
            schema=CALIBRATION_ROOT_SCHEMA,
            study_id=STUDY_ID,
            scenario=self.scenario,
            identities={
                "contract_receipt": file_identity(contract),
                "replay_manifest": file_identity(replay),
                "runs": {
                    run: file_identity(path)
                    for run, path in completions.items()
                },
            },
            fields={
                "status": "fixed48_two_slot_calibration_complete",
                "representation_seed": self.seed,
                "steps_per_run": 781,
                "presentations_per_run": 99_968,
                "total_presentations": 499_840,
            },
        )
        atomic_write_receipt(root_path, root_receipt)
        audit_runs = {}
        for run, completion in completions.items():
            files = {
                role: self._file(self.root / run / f"{role}.fixture")
                for role in (
                    "effective_config",
                    "effective_config_receipt",
                    "checkpoint",
                    "metrics",
                    "summary",
                )
            }
            audit_runs[run] = {
                **{role: file_identity(path) for role, path in files.items()},
                "completion_receipt": file_identity(completion),
            }
        root_role = (
            "calibration_root"
            if wrong_root_role
            else "root_completion_receipt"
        )
        audit_receipt = build_receipt(
            schema=CALIBRATION_AUDIT_SCHEMA,
            study_id=STUDY_ID,
            scenario=self.scenario,
            identities={
                "auditor_source": file_identity(
                    self._file(self.root / "auditor.py")
                ),
                root_role: file_identity(root_path),
                "contract_receipt": file_identity(contract),
                "replay_manifest": file_identity(replay),
                "runs": audit_runs,
            },
            fields={
                "status": "fixed48_calibration_independent_audit_pass",
                "representation_seed": self.seed,
                "values_or_outcomes_accessed": False,
            },
        )
        atomic_write_receipt(
            self.root / "INDEPENDENT_CALIBRATION_AUDIT_RECEIPT.json",
            audit_receipt,
        )

    def test_agreed_audit_role_and_field_integrate(self) -> None:
        self._fixture()
        paths = _verify_calibration(self.root, seed=self.seed)
        self.assertEqual(set(paths), {"root", "audit", "B", "P", "H"})

    def test_old_root_role_fails_closed(self) -> None:
        self._fixture(wrong_root_role=True)
        with self.assertRaisesRegex(ValueError, "identity topology"):
            _verify_calibration(self.root, seed=self.seed)


if __name__ == "__main__":
    unittest.main()
