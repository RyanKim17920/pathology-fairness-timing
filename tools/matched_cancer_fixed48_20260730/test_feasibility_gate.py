from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
)

from . import feasibility_gate as gate


class FeasibilityGateLineageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.authorization = self.root / "authorization.json"
        self.legacy_authorization = self.root / "legacy.json"
        self.authorization.write_text("{}\n", encoding="utf-8")
        self.legacy_authorization.write_text("{}\n", encoding="utf-8")
        self.authorized = {}
        self.redirected = {}
        for cancer in gate.CANCERS:
            authorized = self.root / f"{cancer}.authorized.json"
            redirected = self.root / f"{cancer}.redirected.json"
            authorized.write_text(f"{cancer} authorized\n", encoding="utf-8")
            redirected.write_text(f"{cancer} redirected\n", encoding="utf-8")
            self.authorized[cancer] = authorized
            self.redirected[cancer] = redirected
        self.authorization_receipt = {
            "identities": {
                "legacy_authorization_manifest": file_identity(
                    self.legacy_authorization
                ),
                "legacy_cohorts": {
                    cancer: {
                        "source_bundle": file_identity(
                            self.authorized[cancer]
                        )
                    }
                    for cancer in gate.CANCERS
                },
            }
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_gate(self, bundles: dict[str, Path]) -> Path:
        receipt = build_receipt(
            schema=gate.SCHEMA,
            study_id=gate.STUDY_ID,
            scenario=gate.SCENARIO,
            identities={
                "authorization_manifest": file_identity(self.authorization),
                "legacy_authorization_manifest": file_identity(
                    self.legacy_authorization
                ),
                "source_bundles": {
                    cancer: file_identity(bundles[cancer])
                    for cancer in gate.CANCERS
                },
                "gate_source": file_identity(Path(gate.__file__)),
            },
            fields={
                "status": "pass",
                "all_required_denominators_nonempty": True,
                "outcomes_opened_for_feasibility": True,
                "outcome_values_persisted": False,
                "counts_or_labels_exposed": False,
            },
        )
        destination = self.root / "gate.json"
        return atomic_write_receipt(destination, receipt)

    def test_authorized_source_bundles_pass(self) -> None:
        receipt = self._write_gate(self.authorized)
        with mock.patch.object(
            gate,
            "verify_authorization",
            return_value=self.authorization_receipt,
        ):
            gate.verify(
                receipt, authorization_manifest=self.authorization
            )

    def test_self_consistent_redirected_source_bundles_fail(self) -> None:
        receipt = self._write_gate(self.redirected)
        with mock.patch.object(
            gate,
            "verify_authorization",
            return_value=self.authorization_receipt,
        ):
            with self.assertRaisesRegex(
                ValueError, "source bundle differs from authorization"
            ):
                gate.verify(
                    receipt, authorization_manifest=self.authorization
                )


if __name__ == "__main__":
    unittest.main()
