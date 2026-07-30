"""Outcome-blind synthetic tests for the one-seed deployment gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.matched_cancer_diagnostic_20260730.deployment import (
    AUTHORIZATION_SCHEMA,
    AMENDMENT,
    DEFAULT_CONTRACT,
    RUNTIME_SOURCE_PATHS,
    load_contract,
    load_authorization_manifest,
    preflight,
    verify_gate,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    canonical_json_bytes,
    file_identity,
    topology_sha256,
)
from tools.matched_cancer_diagnostic_20260730.test_integration import (
    make_calibration_tree,
)


def make_synthetic_tile_view(
    root: Path,
    frozen_fold_paths: dict[str, Path] | None = None,
) -> tuple[Path, dict[str, Path]]:
    source_root = root / "tile_view_sources"
    view_root = root / "tile_view"
    source_root.mkdir()
    view_root.mkdir()
    frozen = {}
    source_ids = {}
    view_ids = {}
    inventory = {}
    directories = {}
    for index, cancer in enumerate(("BRCA", "LUAD")):
        frozen_file = (
            frozen_fold_paths[cancer]
            if frozen_fold_paths is not None
            else root / f"tile_view_{cancer}_folds.csv"
        )
        if frozen_fold_paths is None:
            frozen_file.write_text(
                "patient_barcode,fold,race\nP1,target,white\n"
            )
        frozen[cancer] = file_identity(frozen_file)
        directory = view_root / cancer
        directory.mkdir()
        directories[cancer] = directory
        source = source_root / f"{cancer.lower()}.parquet"
        source.write_bytes(f"synthetic-{cancer}-hardlink".encode())
        view = directory / source.name
        os.link(source, view)
        key = f"{index:08d}"
        source_ids[key] = file_identity(source)
        view_ids[key] = file_identity(view)
        source_stat = source.stat()
        view_stat = view.stat()
        inventory[key] = {
            "cancer": cancer,
            "destination_relative": f"{cancer}/{view.name}",
            "patient_id": f"{cancer}-P1",
            "slide_path": f"/synthetic/{cancer}.svs",
            "source_basename": source.name,
            "source_device": source_stat.st_dev,
            "source_inode": source_stat.st_ino,
            "view_device": view_stat.st_dev,
            "view_inode": view_stat.st_ino,
        }
    receipt = build_receipt(
        schema="matched-cancer-diagnostic-tile-view/v1",
        study_id="matched_cancer_stage_20260730",
        scenario="brca_luad_black_white_calibration_seed32001",
        identities={
            "frozen_folds": frozen,
            "source_parquets": source_ids,
            "view_parquets": view_ids,
        },
        fields={
            "destination_root": str(view_root.resolve()),
            "source_root": str(source_root.resolve()),
            "eligible_races": ["Black", "White"],
            "target_fold": "target",
            "expected_patient_counts": {"BRCA": 328, "LUAD": 281},
            "patient_counts": {"BRCA": 328, "LUAD": 281},
            "slide_counts": {"BRCA": 1, "LUAD": 1},
            "file_count": 2,
            "cohort_summaries": {
                "BRCA": {
                    "target_rows": 334,
                    "eligible_rows": 328,
                    "excluded_races": {
                        "asian": 5,
                        "american indian or alaska native": 1,
                    },
                    "race_counts": {"Black": 118, "White": 210},
                },
                "LUAD": {
                    "target_rows": 281,
                    "eligible_rows": 281,
                    "excluded_races": {},
                    "race_counts": {"Black": 40, "White": 241},
                },
            },
            "fold_columns_read": ["patient_barcode", "fold", "race"],
            "parquet_columns_read": ["slide_path"],
            "outcomes_opened": False,
            "inventory": inventory,
        },
    )
    receipt_path = atomic_write_receipt(
        view_root / "TILE_VIEW_RECEIPT.json", receipt
    )
    return receipt_path, directories


def make_authorization_manifest(root: Path) -> Path:
    entrypoint = (
        "tools.matched_cancer_diagnostic_20260730.vetted_loader:load"
    )
    loader_source = Path(__file__).with_name("vetted_loader.py").resolve()
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    demographics = root / "synthetic_demographics.csv"
    molecular = root / "synthetic_molecular.csv"
    demographics.write_text("patient_barcode,race\nP1,white\n")
    molecular.write_text("patient_barcode,tp53_status\nP1,1\n")
    tile_view_receipt, tile_directories = make_synthetic_tile_view(root)
    tile_view_value = json.loads(tile_view_receipt.read_text())
    cohorts = {}
    for cancer in ("BRCA", "LUAD"):
        cancer_root = root / f"authorization_{cancer}"
        cancer_root.mkdir()
        folds = Path(
            tile_view_value["identities"]["frozen_folds"][cancer][
                "canonical_path"
            ]
        )
        spec = contract["cohorts"][cancer]
        source = build_receipt(
            schema="matched-cancer-diagnostic-source-bundle/v1",
            study_id="matched_cancer_stage_20260730",
            scenario="brca_luad_black_white_calibration_seed32001",
            identities={
                "demographics_csv": file_identity(demographics),
                "molecular_csv": file_identity(molecular),
                "frozen_folds_csv": file_identity(folds),
                "estimand_amendment": file_identity(AMENDMENT),
            },
            fields={
                "cancer": cancer,
                "task": spec["task"],
                "target_fold": "target",
                "expected_target_rows": spec["expected_target_rows"],
                "expected_eligible_patients": spec[
                    "expected_eligible_patients"
                ],
                "expected_exclusions_by_race": spec[
                    "expected_exclusions_by_race"
                ],
                "expected_race_counts": spec["expected_race_counts"],
                "split_seed": 288_850_999,
            },
        )
        source_path = atomic_write_receipt(
            cancer_root / "SOURCE.json", source
        )
        tile_directory = tile_directories[cancer]
        files = {
            key: identity
            for key, identity in tile_view_value["identities"][
                "view_parquets"
            ].items()
            if Path(identity["canonical_path"]).parent == tile_directory
        }
        tile_ledger = build_receipt(
            schema="matched-cancer-diagnostic-tile-ledger/v1",
            study_id="matched_cancer_stage_20260730",
            scenario="brca_luad_black_white_calibration_seed32001",
            identities={
                "tile_view_receipt": file_identity(tile_view_receipt),
                "files": files,
            },
            fields={
                "tile_directory": str(tile_directory.resolve()),
                "file_count": len(files),
                "cancer": cancer,
            },
        )
        tile_ledger_path = atomic_write_receipt(
            cancer_root / "TILE_LEDGER.json", tile_ledger
        )
        cohorts[cancer] = {
            "patient_records": str(source_path),
            "tile_source": str(tile_directory),
            "cohort_ledger": str(tile_ledger_path),
        }
    manifest = {
        "schema": AUTHORIZATION_SCHEMA,
        "loader_entrypoint": entrypoint,
        "loader_source": str(loader_source),
        "tile_view_receipt": str(tile_view_receipt),
        "cohorts": cohorts,
    }
    path = root / "authorization.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return path


def make_deployment_gate(
    root: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    calibration, completions = make_calibration_tree(root)
    authorization = make_authorization_manifest(root)
    gate = preflight(
        contract_path=DEFAULT_CONTRACT,
        calibration_root_receipt=calibration,
        completion_receipts=completions,
        authorization_manifest=authorization,
        destination=root / "GATE.json",
    )
    return gate, calibration, completions


class DeploymentTests(unittest.TestCase):
    def test_driver_uses_package_module_invocation(self) -> None:
        driver = Path(__file__).with_name(
            "diagnostic_seed32001.sbatch"
        ).read_text()
        self.assertIn(
            '"$PY" -m "$DEPLOY_MODULE" preflight',
            driver,
        )
        self.assertIn(
            '"$PY" -m "$DEPLOY_MODULE" run',
            driver,
        )
        self.assertIn(
            '"$PY" -m "$EXPORTER_MODULE" export',
            driver,
        )
        self.assertIn(
            '"$PY" -m "$EXPORTER_MODULE" collect',
            driver,
        )
        self.assertNotIn('"$PY" "$DEPLOY" preflight', driver)
        self.assertNotIn('"$PY" "$EXPORTER" export', driver)

    def test_contract_locks_two_named_cohorts(self) -> None:
        contract = load_contract(DEFAULT_CONTRACT)
        self.assertEqual(
            contract["cohorts"]["BRCA"]["expected_target_rows"], 334
        )
        self.assertEqual(
            contract["cohorts"]["BRCA"]["expected_eligible_patients"], 328
        )
        self.assertEqual(
            contract["cohorts"]["LUAD"]["expected_eligible_patients"], 281
        )
        self.assertEqual(
            contract["cohorts"]["BRCA"]["logical_name"], "brca_racepanel"
        )

    def test_preflight_is_outcome_blind_and_does_not_resolve_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration, completions = make_calibration_tree(root)
            authorization = make_authorization_manifest(root)
            with mock.patch(
                "tools.matched_cancer_diagnostic_20260730.deployment.resolve_loader"
            ) as loader:
                gate_path = preflight(
                    contract_path=DEFAULT_CONTRACT,
                    calibration_root_receipt=calibration,
                    completion_receipts=completions,
                    authorization_manifest=authorization,
                    destination=root / "GATE.json",
                )
            loader.assert_not_called()
            gate = verify_gate(gate_path)
            self.assertIs(gate["outcomes_opened"], False)
            self.assertEqual(gate["cohort_sizes"], {"BRCA": 328, "LUAD": 281})

    def test_contract_cohort_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            contract = json.loads(DEFAULT_CONTRACT.read_text())
            contract["cohorts"]["LUAD"]["expected_eligible_patients"] = 280
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, "cohort names/tasks/sizes"):
                load_contract(path)

    def test_gate_fails_after_bound_source_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration, completions = make_calibration_tree(root)
            authorization = make_authorization_manifest(root)
            contract = root / "deployment.json"
            contract.write_bytes(DEFAULT_CONTRACT.read_bytes())
            gate = preflight(
                contract_path=contract,
                calibration_root_receipt=calibration,
                completion_receipts=completions,
                authorization_manifest=authorization,
                destination=root / "GATE.json",
            )
            contract.write_text(contract.read_text() + "\n")
            with self.assertRaises(ValueError):
                verify_gate(gate)

    def test_gate_rejects_self_consistent_outcome_source_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration, completions = make_calibration_tree(root)
            authorization_path = make_authorization_manifest(root)
            authorization = json.loads(authorization_path.read_text())
            gate = preflight(
                contract_path=DEFAULT_CONTRACT,
                calibration_root_receipt=calibration,
                completion_receipts=completions,
                authorization_manifest=authorization_path,
                destination=root / "GATE.json",
            )
            replacement = root / "redirected_molecular.csv"
            replacement.write_text(
                "patient_barcode,tp53_status\nP1,0\n"
            )
            source_path = Path(
                authorization["cohorts"]["BRCA"]["patient_records"]
            )
            source = json.loads(source_path.read_text())
            source["identities"]["molecular_csv"] = file_identity(replacement)
            source["topology_sha256"] = topology_sha256(
                source["identities"]
            )
            source_path.write_bytes(canonical_json_bytes(source) + b"\n")
            with self.assertRaisesRegex(
                ValueError, "file identity mismatch|source-bundle identity"
            ):
                verify_gate(gate)

    def test_gate_rejects_transitive_runtime_source_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration, completions = make_calibration_tree(root)
            authorization = make_authorization_manifest(root)
            copied_source = root / "union_objectives.py"
            copied_source.write_bytes(
                RUNTIME_SOURCE_PATHS["union_objectives"].read_bytes()
            )
            with mock.patch.dict(
                RUNTIME_SOURCE_PATHS,
                {"union_objectives": copied_source},
            ):
                gate = preflight(
                    contract_path=DEFAULT_CONTRACT,
                    calibration_root_receipt=calibration,
                    completion_receipts=completions,
                    authorization_manifest=authorization,
                    destination=root / "GATE.json",
                )
                copied_source.write_text(
                    copied_source.read_text() + "\n# tampered\n"
                )
                with self.assertRaisesRegex(
                    ValueError, "file identity mismatch|source identity"
                ):
                    verify_gate(gate)

    def test_gate_rejects_package_initializer_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration, completions = make_calibration_tree(root)
            authorization = make_authorization_manifest(root)
            copied_source = root / "__init__.py"
            copied_source.write_bytes(
                RUNTIME_SOURCE_PATHS["stage_package_init"].read_bytes()
            )
            with mock.patch.dict(
                RUNTIME_SOURCE_PATHS,
                {"stage_package_init": copied_source},
            ):
                gate = preflight(
                    contract_path=DEFAULT_CONTRACT,
                    calibration_root_receipt=calibration,
                    completion_receipts=completions,
                    authorization_manifest=authorization,
                    destination=root / "GATE.json",
                )
                copied_source.write_text(
                    copied_source.read_text() + "\n# tampered\n"
                )
                with self.assertRaisesRegex(
                    ValueError, "file identity mismatch|source identity"
                ):
                    verify_gate(gate)

    def test_loader_allowlist_rejects_entrypoint_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = make_authorization_manifest(root)
            manifest = json.loads(path.read_text())
            manifest["loader_entrypoint"] = (
                "tools.matched_cancer_diagnostic_20260730.deployment:"
                "load_contract"
            )
            path.write_text(json.dumps(manifest))
            with mock.patch("importlib.import_module") as importer:
                with self.assertRaisesRegex(ValueError, "not the vetted loader"):
                    load_authorization_manifest(path)
            importer.assert_not_called()

    def test_minimal_self_consistent_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forged = build_receipt(
                schema="matched-cancer-diagnostic-deployment-gate/v1",
                study_id="matched_cancer_stage_20260730",
                scenario="brca_luad_black_white_calibration_seed32001",
                identities={
                    "deployment_contract": file_identity(DEFAULT_CONTRACT)
                },
                fields={
                    "status": "valid",
                    "representation_seed": 32001,
                    "outcomes_opened": False,
                },
            )
            path = atomic_write_receipt(root / "forged.json", forged)
            with self.assertRaisesRegex(ValueError, "identity topology"):
                verify_gate(path)


if __name__ == "__main__":
    unittest.main()
