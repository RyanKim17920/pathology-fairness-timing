"""Synthetic-only tests for the gate-bound real-data loader mechanics."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import post_hoc_debias
from tools.matched_cancer_diagnostic_20260730.deployment import (
    AMENDMENT,
    DEFAULT_CONTRACT,
    verify_gate,
    verify_loader_result,
)
from tools.matched_cancer_diagnostic_20260730 import vetted_loader
from tools.matched_cancer_diagnostic_20260730.test_deployment import (
    make_deployment_gate,
    make_synthetic_tile_view,
)
from tools.matched_cancer_diagnostic_20260730.vetted_loader import (
    SOURCE_SCHEMA,
    SPLIT_SEED,
    TILE_LEDGER_SCHEMA,
    prepare_cohort,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
)


def synthetic_sources(
    root: Path,
    *,
    count: int = 10,
    invalid_race: bool = False,
) -> tuple[dict[str, str], dict, dict, Path]:
    demographics = root / "demographics.csv"
    molecular = root / "molecular.csv"
    folds = root / "frozen_folds.csv"
    demo: dict[str, dict] = {}
    mol: dict[str, dict] = {}
    with demographics.open("w", newline="") as demo_handle, molecular.open(
        "w", newline=""
    ) as mol_handle, folds.open("w", newline="") as folds_handle:
        demo_writer = csv.DictWriter(
            demo_handle,
            fieldnames=["patient_barcode", "race", "gender", "tss"],
        )
        mol_writer = csv.DictWriter(
            mol_handle,
            fieldnames=[
                "patient_barcode", "tp53_status", "fold_tp53_brca"
            ],
        )
        fold_writer = csv.DictWriter(
            folds_handle,
            fieldnames=["patient_barcode", "fold", "tss", "race"],
        )
        demo_writer.writeheader()
        mol_writer.writeheader()
        fold_writer.writeheader()
        for index in range(count):
            patient = f"TCGA-{index:02d}-0001"
            race = (
                "asian"
                if invalid_race and index == 0
                else "black or african american" if index % 2 else "white"
            )
            demo_row = {
                "patient_barcode": patient,
                "race": race,
                "gender": "male" if index % 2 else "female",
                "tss": f"{index:02d}",
            }
            mol_row = {
                "patient_barcode": patient,
                "tp53_status": str(index % 2),
                "fold_tp53_brca": str(index % 5),
            }
            fold_row = {
                "patient_barcode": patient,
                "fold": "target",
                "tss": f"{index:02d}",
                "race": race,
            }
            demo_writer.writerow(demo_row)
            mol_writer.writerow(mol_row)
            fold_writer.writerow(fold_row)
            demo[patient] = demo_row
            mol[patient] = mol_row

    eligible = count - int(invalid_race)
    race_counts = {
        "Black": count // 2,
        "White": (count + 1) // 2 - int(invalid_race),
    }
    source = build_receipt(
        schema=SOURCE_SCHEMA,
        study_id="matched_cancer_stage_20260730",
        scenario="brca_luad_black_white_calibration_seed32001",
        identities={
            "demographics_csv": file_identity(demographics),
            "molecular_csv": file_identity(molecular),
            "frozen_folds_csv": file_identity(folds),
            "estimand_amendment": file_identity(AMENDMENT),
        },
        fields={
            "cancer": "BRCA",
            "task": "brca_tp53",
            "target_fold": "target",
            "expected_target_rows": count,
            "expected_eligible_patients": eligible,
            "expected_exclusions_by_race": (
                {"Asian": 1} if invalid_race else {}
            ),
            "expected_race_counts": race_counts,
            "split_seed": SPLIT_SEED,
        },
    )
    source_path = atomic_write_receipt(root / "SOURCE.json", source)

    luad_folds = root / "synthetic_luad_folds.csv"
    luad_folds.write_text(
        "patient_barcode,fold,race\nLUAD-P1,target,white\n"
    )
    tile_view_receipt, directories = make_synthetic_tile_view(
        root, {"BRCA": folds, "LUAD": luad_folds}
    )
    tile_directory = directories["BRCA"]
    tile_view = __import__("json").loads(tile_view_receipt.read_text())
    files = {
        key: identity
        for key, identity in tile_view["identities"]["view_parquets"].items()
        if Path(identity["canonical_path"]).parent == tile_directory
    }
    tile_ledger = build_receipt(
        schema=TILE_LEDGER_SCHEMA,
        study_id="matched_cancer_stage_20260730",
        scenario="brca_luad_black_white_calibration_seed32001",
        identities={
            "tile_view_receipt": file_identity(tile_view_receipt),
            "files": files,
        },
        fields={
            "tile_directory": str(tile_directory.resolve()),
            "file_count": len(files),
            "cancer": "BRCA",
        },
    )
    tile_ledger_path = atomic_write_receipt(
        root / "TILE_LEDGER.json", tile_ledger
    )
    authorized = {
        "patient_records": str(source_path),
        "tile_source": str(tile_directory),
        "cohort_ledger": str(tile_ledger_path),
    }
    return authorized, demo, mol, tile_view_receipt


def synthetic_contract(count: int, invalid_race: bool = False) -> dict:
    return {
        "task": "brca_tp53",
        "expected_target_rows": count,
        "expected_eligible_patients": count - int(invalid_race),
        "expected_exclusions_by_race": (
            {"Asian": 1} if invalid_race else {}
        ),
        "expected_race_counts": {
            "Black": count // 2,
            "White": (count + 1) // 2 - int(invalid_race),
        },
    }


def dependency_hooks(demo: dict, mol: dict):
    def load_demographics(path: str, key: str):
        del key
        return demo if "demographics" in Path(path).name else mol

    def collect_tiles(
        tile_directory,
        task_cohort,
        sensitive,
        sensitive_name,
        adversary_data,
        max_task_slides,
        max_pool_slides,
        max_tiles,
        log,
    ):
        del (
            tile_directory, sensitive, sensitive_name, adversary_data,
            max_task_slides, max_pool_slides, max_tiles, log,
        )
        return [
            (patient, f"jpeg-{patient}".encode())
            for patient in sorted(task_cohort)
        ], []

    return load_demographics, collect_tiles


class VettedLoaderTests(unittest.TestCase):
    def test_exact_target_builds_shared_deterministic_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorized, demo, mol, tile_view = synthetic_sources(root)
            load_demo, collect = dependency_hooks(demo, mol)
            patients, tiles, cohort, receipt = prepare_cohort(
                cancer="BRCA",
                cohort_contract=synthetic_contract(10),
                authorized=authorized,
                tile_view_receipt=tile_view,
                destination=root / "prepared",
                load_demographics_fn=load_demo,
                build_task_cohort_fn=post_hoc_debias.build_task_cohort,
                sensitive_of_fn=post_hoc_debias.sensitive_of,
                collect_tiles_fn=collect,
            )
            self.assertEqual(len(patients), 10)
            self.assertEqual(len(tiles), 10)
            self.assertEqual({patient.outer_fold for patient in patients}, set(range(5)))
            self.assertEqual({patient.race for patient in patients}, {"Black", "White"})
            self.assertTrue(cohort.is_file())
            self.assertTrue(receipt.is_file())

    def test_noneligible_race_is_excluded_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorized, demo, mol, tile_view = synthetic_sources(
                root, count=12, invalid_race=True
            )
            load_demo, collect = dependency_hooks(demo, mol)
            patients, _, _, receipt = prepare_cohort(
                cancer="BRCA",
                cohort_contract=synthetic_contract(12, invalid_race=True),
                authorized=authorized,
                tile_view_receipt=tile_view,
                destination=root / "prepared",
                load_demographics_fn=load_demo,
                build_task_cohort_fn=post_hoc_debias.build_task_cohort,
                sensitive_of_fn=post_hoc_debias.sensitive_of,
                collect_tiles_fn=collect,
            )
            import json

            sealed = json.loads(receipt.read_text())
            self.assertEqual(len(patients), 11)
            self.assertEqual(sealed["raw_target_count"], 12)
            self.assertEqual(sealed["exclusions_by_race"], {"Asian": 1})
            self.assertEqual(sealed["race_counts"], {"Black": 6, "White": 5})
            self.assertEqual(len(sealed["eligible_patient_ids_sha256"]), 64)
            self.assertEqual(len(sealed["fold_sha256"]), 64)

    def test_source_tamper_fails_before_metadata_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorized, demo, mol, tile_view = synthetic_sources(root)
            (root / "molecular.csv").write_text("tampered\n")
            called = False

            def forbidden_loader(*args):
                nonlocal called
                called = True
                raise AssertionError(args)

            with self.assertRaises(ValueError):
                prepare_cohort(
                    cancer="BRCA",
                    cohort_contract=synthetic_contract(10),
                    authorized=authorized,
                    tile_view_receipt=tile_view,
                    destination=root / "prepared",
                    load_demographics_fn=forbidden_loader,
                    build_task_cohort_fn=post_hoc_debias.build_task_cohort,
                    sensitive_of_fn=post_hoc_debias.sensitive_of,
                    collect_tiles_fn=lambda *args: ([], []),
                )
            self.assertFalse(called)

    def test_tile_inventory_drift_fails_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorized, demo, mol, tile_view = synthetic_sources(root)
            (Path(authorized["tile_source"]) / "unledgered.parquet").write_bytes(
                b"extra"
            )
            load_demo, _ = dependency_hooks(demo, mol)
            called = False

            def forbidden_collect(*args):
                nonlocal called
                called = True
                raise AssertionError(args)

            with self.assertRaisesRegex(ValueError, "files differ"):
                prepare_cohort(
                    cancer="BRCA",
                    cohort_contract=synthetic_contract(10),
                    authorized=authorized,
                    tile_view_receipt=tile_view,
                    destination=root / "prepared",
                    load_demographics_fn=load_demo,
                    build_task_cohort_fn=post_hoc_debias.build_task_cohort,
                    sensitive_of_fn=post_hoc_debias.sensitive_of,
                    collect_tiles_fn=forbidden_collect,
                )
            self.assertFalse(called)

    def test_replaced_hardlink_view_file_fails_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorized, demo, mol, tile_view = synthetic_sources(root)
            view_file = next(
                Path(authorized["tile_source"]).glob("*.parquet")
            )
            payload = view_file.read_bytes()
            view_file.unlink()
            view_file.write_bytes(payload)
            load_demo, collect = dependency_hooks(demo, mol)
            with self.assertRaisesRegex(ValueError, "hardlink differs"):
                prepare_cohort(
                    cancer="BRCA",
                    cohort_contract=synthetic_contract(10),
                    authorized=authorized,
                    tile_view_receipt=tile_view,
                    destination=root / "prepared",
                    load_demographics_fn=load_demo,
                    build_task_cohort_fn=post_hoc_debias.build_task_cohort,
                    sensitive_of_fn=post_hoc_debias.sensitive_of,
                    collect_tiles_fn=collect,
                )

    def test_full_load_runs_both_cancers_and_exact_bph_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, _, completions = make_deployment_gate(root)
            gate_value = __import__("json").loads(gate.read_text())
            authorization_path = Path(
                gate_value["identities"]["authorization_manifest"][
                    "canonical_path"
                ]
            )
            authorization = __import__("json").loads(
                authorization_path.read_text()
            )
            contract = __import__("json").loads(DEFAULT_CONTRACT.read_text())
            cohort_sources: dict[str, Path] = {}

            def prepare(**kwargs):
                cancer = kwargs["cancer"]
                destination = Path(kwargs["destination"])
                destination.mkdir(parents=True)
                records = destination / "cohort.jsonl"
                records.write_text('{"synthetic":true}\n')
                cohort_sources[cancer] = records
                spec = contract["cohorts"][cancer]
                receipt = build_receipt(
                    schema="matched-cancer-diagnostic-cohort/v1",
                    study_id="matched_cancer_stage_20260730",
                    scenario=(
                        "brca_luad_black_white_calibration_seed32001"
                    ),
                    identities={
                        "source_bundle": file_identity(
                            authorization["cohorts"][cancer][
                                "patient_records"
                            ]
                        ),
                        "tile_ledger": file_identity(
                            authorization["cohorts"][cancer][
                                "cohort_ledger"
                            ]
                        ),
                        "cohort_records": file_identity(records),
                        "loader": file_identity(
                            Path(__file__).with_name("vetted_loader.py")
                        ),
                    },
                    fields={
                        "cancer": cancer,
                        "task": spec["task"],
                        "raw_target_count": spec["expected_target_rows"],
                        "eligible_patient_count": spec[
                            "expected_eligible_patients"
                        ],
                        "patient_count": spec["expected_eligible_patients"],
                        "tile_count": 1,
                        "split_seed": SPLIT_SEED,
                        "exclusions_by_race": spec[
                            "expected_exclusions_by_race"
                        ],
                        "race_counts": spec["expected_race_counts"],
                        "eligible_patient_ids_sha256": "a" * 64,
                        "fold_sha256": "b" * 64,
                    },
                )
                receipt_path = atomic_write_receipt(
                    destination / "COHORT_RECEIPT.json", receipt
                )
                patient = vetted_loader.PatientRecord(
                    patient_id=f"{cancer}-P1",
                    y_true=1,
                    race="Black",
                    tss="01",
                    outer_fold=0,
                )
                return [patient], [(patient.patient_id, b"jpeg")], records, receipt_path

            run_calls = []

            def run_diagnostic(**kwargs):
                task = kwargs["task_id"]
                cancer = "BRCA" if task == "brca_tp53" else "LUAD"
                output = Path(kwargs["output_root"])
                output.mkdir(parents=True)
                source = output / "source.py"
                source.write_text("# synthetic\n")
                cells = {arm: {} for arm in ("B", "P", "H")}
                for arm in ("B", "P", "H"):
                    for head in (42001, 42002, 42003, 42004):
                        cell_dir = output / arm / str(head)
                        cell_dir.mkdir(parents=True)
                        predictions = cell_dir / "nested.jsonl"
                        predictions.write_text('{"synthetic":true}\n')
                        cell = build_receipt(
                            schema=(
                                "matched-cancer-adapter-diagnostic/v1"
                            ),
                            study_id="matched_cancer_stage_20260730",
                            scenario=(
                                "brca_luad_black_white_calibration_seed32001"
                            ),
                            identities={
                                "predictions": file_identity(predictions),
                                "completion_receipt": file_identity(
                                    completions[arm]
                                ),
                                "cohort_source": file_identity(
                                    cohort_sources[cancer]
                                ),
                            },
                            fields={
                                "status": "complete",
                                "arm": arm,
                                "head_seed": head,
                                "task_id": task,
                            },
                        )
                        cell_path = atomic_write_receipt(
                            cell_dir / "DIAGNOSTIC_RECEIPT.json", cell
                        )
                        cells[arm][str(head)] = file_identity(cell_path)
                receipt = build_receipt(
                    schema="matched-cancer-adapter-diagnostic-root/v1",
                    study_id="matched_cancer_stage_20260730",
                    scenario=(
                        "brca_luad_black_white_calibration_seed32001"
                    ),
                    identities={
                        "cohort_source": file_identity(cohort_sources[cancer]),
                        "completion_receipts": {
                            arm: file_identity(completions[arm])
                            for arm in ("B", "P", "H")
                        },
                        "cells": cells,
                        "sources": {"runner": file_identity(source)},
                    },
                    fields={
                        "status": "complete",
                        "task_id": task,
                        "arms": ["B", "P", "H"],
                        "head_seeds": [42001, 42002, 42003, 42004],
                        "cell_count": 12,
                        "race_usage": "output_metadata_only",
                    },
                )
                run_calls.append(cancer)
                return atomic_write_receipt(
                    output / "ROOT_DIAGNOSTIC_RECEIPT.json", receipt
                )

            with (
                mock.patch.object(
                    vetted_loader,
                    "load_frozen_representation",
                    autospec=True,
                    return_value=object(),
                ) as representation_mock,
                mock.patch.object(
                    vetted_loader, "prepare_cohort", side_effect=prepare
                ) as prepare_mock,
                mock.patch.object(
                    vetted_loader,
                    "run_paired_diagnostic",
                    side_effect=run_diagnostic,
                ),
            ):
                result = vetted_loader.load(
                    contract=contract,
                    cancers=("BRCA", "LUAD"),
                    authorized_paths=authorization,
                    output_root=root / "full_load",
                    gate_receipt=gate,
                )
            verify_loader_result(result, gate)
            self.assertEqual(representation_mock.call_count, 3)
            for call in representation_mock.call_args_list:
                self.assertNotIn("expected_amendment_identity", call.kwargs)
            self.assertEqual(prepare_mock.call_count, 2)
            expected_amendment = verify_gate(gate)["identities"][
                "estimand_amendment"
            ]
            for call in prepare_mock.call_args_list:
                self.assertEqual(
                    call.kwargs["expected_amendment_identity"],
                    expected_amendment,
                )
            self.assertEqual(run_calls, ["BRCA", "LUAD"])


if __name__ == "__main__":
    unittest.main()
