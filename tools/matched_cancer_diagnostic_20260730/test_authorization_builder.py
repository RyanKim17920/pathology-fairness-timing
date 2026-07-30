"""Synthetic tests for outcome-uninterpreted source authorization."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.matched_cancer_diagnostic_20260730.authorization_builder import (
    build_authorization,
)
from tools.matched_cancer_diagnostic_20260730.deployment import (
    DEFAULT_CONTRACT,
    load_authorization_manifest,
)
from tools.matched_cancer_stage_20260730.receipts import verify_receipt
from tools.matched_cancer_diagnostic_20260730.test_deployment import (
    make_synthetic_tile_view,
)


class AuthorizationBuilderTests(unittest.TestCase):
    def test_explicit_sources_and_tile_inventories_are_sealed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demographics = root / "demographics.csv"
            molecular = root / "molecular.csv"
            demographics.write_text("patient_barcode,race\nP1,white\n")
            molecular.write_text("patient_barcode,tp53_status\nP1,1\n")
            folds = {}
            for cancer in ("BRCA", "LUAD"):
                folds[cancer] = root / f"{cancer}.csv"
                folds[cancer].write_text(
                    "patient_barcode,fold\nP1,target\n"
                )
            tile_view, tiles = make_synthetic_tile_view(root, folds)
            manifest_path = build_authorization(
                contract_path=DEFAULT_CONTRACT,
                demographics=demographics,
                molecular=molecular,
                frozen_folds=folds,
                tile_directories=tiles,
                tile_view_receipt=tile_view,
                destination_root=root / "sealed",
            )
            manifest = load_authorization_manifest(manifest_path)
            for cancer in ("BRCA", "LUAD"):
                verify_receipt(
                    manifest["cohorts"][cancer]["patient_records"],
                    expected_schema=(
                        "matched-cancer-diagnostic-source-bundle/v1"
                    ),
                )
                verify_receipt(
                    manifest["cohorts"][cancer]["cohort_ledger"],
                    expected_schema=(
                        "matched-cancer-diagnostic-tile-ledger/v1"
                    ),
                )

    def test_existing_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                build_authorization(
                    contract_path=DEFAULT_CONTRACT,
                    demographics=root / "missing.csv",
                    molecular=root / "missing2.csv",
                    frozen_folds={"BRCA": "a", "LUAD": "b"},
                    tile_directories={"BRCA": "c", "LUAD": "d"},
                    tile_view_receipt="missing-view.json",
                    destination_root=root,
                )


if __name__ == "__main__":
    unittest.main()
