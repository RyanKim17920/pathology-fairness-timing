#!/usr/bin/env python3

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.matched_cancer_stage_20260730 import contract
from tools.matched_cancer_stage_20260730.receipts import (
    ReceiptVerificationError,
    atomic_write_receipt,
    build_receipt,
    canonical_json_bytes,
    file_identity,
    topology_sha256,
    verify_receipt,
)


class ReceiptTest(unittest.TestCase):
    def test_canonical_json_and_semantic_role_topology(self) -> None:
        self.assertEqual(
            canonical_json_bytes({"z": 1, "a": {"q": 2}}),
            b'{"a":{"q":2},"z":1}',
        )
        with self.assertRaisesRegex(ValueError, "canonical-JSON"):
            canonical_json_bytes({"bad": float("nan")})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right = root / "left", root / "right"
            left.write_bytes(b"left")
            right.write_bytes(b"right")
            identities = {
                "design": file_identity(left),
                "runtime_sources": {"train": file_identity(right)},
            }
            swapped = {
                "design": file_identity(right),
                "runtime_sources": {"train": file_identity(left)},
            }
            self.assertNotEqual(
                topology_sha256(identities), topology_sha256(swapped)
            )

    def test_atomic_round_trip_and_file_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"version one\n")
            receipt = build_receipt(
                schema="test-receipt/v1",
                study_id="study",
                scenario="scenario",
                identities={"runtime_sources": {"source": file_identity(source)}},
            )
            destination = root / "nested" / "receipt.json"
            atomic_write_receipt(destination, receipt)
            self.assertEqual(
                destination.read_bytes(), canonical_json_bytes(receipt) + b"\n"
            )
            self.assertEqual(
                verify_receipt(
                    destination,
                    expected_schema="test-receipt/v1",
                    expected_study_id="study",
                    expected_scenario="scenario",
                ),
                receipt,
            )
            self.assertEqual(
                list(destination.parent.glob(f".{destination.name}.*.tmp")), []
            )

            symlink = root / "receipt-link.json"
            symlink.symlink_to(destination)
            with self.assertRaisesRegex(ValueError, "symlink"):
                atomic_write_receipt(symlink, receipt)

            source.write_bytes(b"version two\n")
            with self.assertRaisesRegex(
                ReceiptVerificationError, "file identity mismatch"
            ):
                verify_receipt(destination)

    def test_receipt_reformatting_and_topology_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_text("content")
            receipt = build_receipt(
                schema="test-receipt/v1",
                study_id="study",
                scenario="scenario",
                identities={"design": file_identity(source)},
            )
            destination = root / "receipt.json"
            destination.write_text(json.dumps(receipt, indent=2))
            with self.assertRaisesRegex(
                ReceiptVerificationError, "not canonically encoded"
            ):
                verify_receipt(destination)

            receipt["topology_sha256"] = "0" * 64
            atomic_write_receipt(destination, receipt)
            with self.assertRaisesRegex(
                ReceiptVerificationError, "topology SHA-256 mismatch"
            ):
                verify_receipt(destination)

    def test_contract_runtime_roles_and_write_receipt_cli(self) -> None:
        required_roles = {
            "contract",
            "config_builder",
            "replay",
            "objectives",
            "train",
            "dataloader",
            "smoke_driver",
        }
        self.assertTrue(required_roles.issubset(contract.RUNTIME_SOURCES))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "runtime.py"
            source.write_text("pass\n")
            receipt = build_receipt(
                schema=contract.RECEIPT_SCHEMA,
                study_id="matched_cancer_stage_20260730",
                scenario="brca_luad_black_white",
                identities={"runtime_sources": {"runtime": file_identity(source)}},
                fields={"status": "valid"},
            )
            destination = root / "receipt.json"
            with mock.patch.object(contract, "validate", return_value=receipt):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        contract.main(["--write-receipt", str(destination)]), 0
                    )
            verify_receipt(
                destination,
                expected_schema=contract.RECEIPT_SCHEMA,
                expected_study_id="matched_cancer_stage_20260730",
                expected_scenario="brca_luad_black_white",
            )


if __name__ == "__main__":
    unittest.main()
