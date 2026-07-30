from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.matched_cancer_stage_20260730.receipts import file_identity

from . import launch_receipt


class LaunchReceiptTests(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _patch_root(self) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(
                launch_receipt, "PRODUCTION_ROOT", self.root.resolve()
            )
        )
        return stack

    @staticmethod
    def _manifest(_: Path) -> dict:
        return {}

    @staticmethod
    def _adoption(*_args, **_kwargs) -> dict:
        return {}

    def _create_prelaunch(self, nonce: str | None = None) -> Path:
        selected = nonce or self.nonce
        destination = (
            self.root
            / f"control/prelaunch/FIXED5_PRELAUNCH_{selected}.json"
        )
        return launch_receipt.create_prelaunch(
            destination,
            launch_nonce=selected,
            production_root=self.root,
            fixed5_source_manifest=self.fixed5,
            adoption_authorization=self.adoption,
            fixed48_source_manifest=self.fixed48,
            authorization_manifest=self.authorization,
            feasibility_gate=self.feasibility,
            manifest_verifier=self._manifest,
            adoption_verifier=self._adoption,
        )

    def test_nonce_is_128_bit_lowercase_hex(self) -> None:
        for _ in range(16):
            nonce = launch_receipt.new_nonce()
            self.assertRegex(nonce, r"^[0-9a-f]{32}$")
            self.assertEqual(launch_receipt.validate_nonce(nonce), nonce)
        for invalid in ("", "0" * 31, "G" * 32, "0" * 32 + "/"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    launch_receipt.validate_nonce(invalid)

    def test_prelaunch_is_exclusive_and_binds_exact_inputs(self) -> None:
        with self._patch_root():
            path = self._create_prelaunch()
            receipt = launch_receipt.verify_prelaunch(
                path,
                launch_nonce=self.nonce,
                production_root=self.root,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
            )
            self.assertEqual(receipt["matching_job_count"], 0)
            self.assertEqual(
                receipt["comments"],
                [
                    launch_receipt.LEGACY_COMMENT,
                    launch_receipt.SUPERSEDED_COMMENT,
                ],
            )
            with self.assertRaises(FileExistsError):
                self._create_prelaunch()

    def test_prelaunch_rejects_redirect_symlink_and_drift(self) -> None:
        with self._patch_root():
            path = self._create_prelaunch()
            alias = self.root / "control/prelaunch/alias.json"
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                launch_receipt.verify_prelaunch(
                    alias,
                    launch_nonce=self.nonce,
                    production_root=self.root,
                    fixed5_source_manifest=self.fixed5,
                    adoption_authorization=self.adoption,
                )
            original = self.fixed5.read_text()
            self.fixed5.write_text("drift\n")
            with self.assertRaises(ValueError):
                launch_receipt.verify_prelaunch(
                    path,
                    launch_nonce=self.nonce,
                    production_root=self.root,
                    fixed5_source_manifest=self.fixed5,
                    adoption_authorization=self.adoption,
                )
            self.fixed5.write_text(original)

    def test_launch_is_per_nonce_job_and_exclusive(self) -> None:
        with self._patch_root():
            prelaunch = self._create_prelaunch()
            destination = (
                self.root
                / f"control/launch/FIXED5_LAUNCH_{self.nonce}_JOB_123.json"
            )
            launch = launch_receipt.create_launch(
                destination,
                launch_nonce=self.nonce,
                slurm_job_id="123",
                prelaunch_receipt=prelaunch,
                production_root=self.root,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
            )
            receipt = launch_receipt.verify_launch(
                launch,
                launch_nonce=self.nonce,
                slurm_job_id="123",
                prelaunch_receipt=prelaunch,
                production_root=self.root,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
            )
            self.assertEqual(receipt["comment"], launch_receipt.LEGACY_COMMENT)
            self.assertEqual(receipt["job_name"], "main_1gpu")
            self.assertEqual(receipt["tasks"], 1)
            self.assertEqual(receipt["allocated_gpus"], 1)
            self.assertEqual(receipt["visible_gpus"], 1)
            with self.assertRaises(FileExistsError):
                launch_receipt.create_launch(
                    destination,
                    launch_nonce=self.nonce,
                    slurm_job_id="123",
                    prelaunch_receipt=prelaunch,
                    production_root=self.root,
                    fixed5_source_manifest=self.fixed5,
                    adoption_authorization=self.adoption,
                )

    def test_receipts_bind_current_submitter_and_driver(self) -> None:
        with self._patch_root():
            receipt = launch_receipt.verify_prelaunch(
                self._create_prelaunch(),
                launch_nonce=self.nonce,
                production_root=self.root,
                fixed5_source_manifest=self.fixed5,
                adoption_authorization=self.adoption,
            )
        self.assertEqual(
            receipt["identities"]["safe_submit_source"],
            file_identity(launch_receipt.SAFE_SUBMIT),
        )
        self.assertEqual(
            receipt["identities"]["slurm_driver"],
            file_identity(launch_receipt.SLURM_DRIVER),
        )


if __name__ == "__main__":
    unittest.main()
