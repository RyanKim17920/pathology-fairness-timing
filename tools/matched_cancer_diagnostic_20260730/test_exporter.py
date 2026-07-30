"""Synthetic tests for exact verifier-row export and collection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.matched_cancer_diagnostic_20260730 import verifier
from tools.matched_cancer_diagnostic_20260730.exporter import (
    ARMS,
    CANCERS,
    FIELDS,
    HEAD_SEEDS,
    collect_exports,
    export_cell,
)
from tools.matched_cancer_diagnostic_20260730.test_deployment import (
    make_deployment_gate,
)
from tools.matched_cancer_diagnostic_20260730.deployment import DEFAULT_CONTRACT
from tools.matched_cancer_diagnostic_20260730.vetted_loader import (
    COHORT_SCHEMA,
    LOADER_ROOT_SCHEMA,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
)


def nested_rows(cancer: str, count: int = 10) -> list[dict]:
    patients = [
        {
            "patient_id": f"{cancer}-{index:03d}",
            "y_true": index % 2,
            "race": "Black" if index % 2 else "White",
            "sex": None,
            "age": None,
            "tss": f"TSS-{index:03d}",
            "original_fold": index % 5,
        }
        for index in range(count)
    ]
    rows: list[dict] = []
    for patient in patients:
        rows.append({
            **patient,
            "prediction_role": "outer_test",
            "outer_fold": patient["original_fold"],
            "y_score": 0.25 + 0.5 * patient["y_true"],
        })
        for outer_fold in range(5):
            if outer_fold == patient["original_fold"]:
                continue
            rows.append({
                **patient,
                "prediction_role": "inner_calibration",
                "calibration_outer_fold": outer_fold,
                "inner_fold": patient["original_fold"],
                "y_score": 0.3 + 0.4 * patient["y_true"],
            })
    return rows


def make_loader_fixture(
    root: Path,
    *,
    gate: Path,
    completions: dict[str, Path],
    count: int = 10,
) -> tuple[Path, dict[tuple[str, str, int], tuple[Path, Path]]]:
    fixture = root / "loader_fixture"
    fixture.mkdir()
    contract = json.loads(DEFAULT_CONTRACT.read_text())
    gate_value = json.loads(gate.read_text())
    authorization = json.loads(Path(
        gate_value["identities"]["authorization_manifest"]["canonical_path"]
    ).read_text())
    cohort_receipts = {}
    diagnostic_receipts = {}
    cells_out: dict[tuple[str, str, int], tuple[Path, Path]] = {}
    source_file = fixture / "runner_source.py"
    source_file.write_text("# synthetic runner source\n")
    for cancer in CANCERS:
        cancer_root = fixture / cancer
        cancer_root.mkdir()
        cohort_records = cancer_root / "cohort.jsonl"
        cohort_records.write_text('{"synthetic":true}\n')
        cohort_contract = contract["cohorts"][cancer]
        cohort = build_receipt(
            schema=COHORT_SCHEMA,
            study_id="matched_cancer_stage_20260730",
            scenario="brca_luad_black_white_calibration_seed32001",
            identities={
                "source_bundle": file_identity(
                    authorization["cohorts"][cancer]["patient_records"]
                ),
                "tile_ledger": file_identity(
                    authorization["cohorts"][cancer]["cohort_ledger"]
                ),
                "cohort_records": file_identity(cohort_records),
                "loader": file_identity(
                    Path(__file__).with_name("vetted_loader.py")
                ),
            },
            fields={
                "cancer": cancer,
                "task": cohort_contract["task"],
                "raw_target_count": cohort_contract["expected_target_rows"],
                "eligible_patient_count": cohort_contract[
                    "expected_eligible_patients"
                ],
                "patient_count": cohort_contract[
                    "expected_eligible_patients"
                ],
                "tile_count": 1,
                "split_seed": 288_850_999,
                "exclusions_by_race": cohort_contract[
                    "expected_exclusions_by_race"
                ],
                "race_counts": cohort_contract["expected_race_counts"],
                "eligible_patient_ids_sha256": "a" * 64,
                "fold_sha256": "b" * 64,
            },
        )
        cohort_path = atomic_write_receipt(
            cancer_root / "COHORT_RECEIPT.json", cohort
        )
        cohort_receipts[cancer] = file_identity(cohort_path)
        cell_identities: dict[str, dict[str, dict]] = {
            arm: {} for arm in ARMS
        }
        for arm in ARMS:
            for head_seed in HEAD_SEEDS:
                cell = cancer_root / arm / str(head_seed)
                cell.mkdir(parents=True)
                nested = cell / "nested.jsonl"
                with nested.open("w") as handle:
                    for row in nested_rows(cancer, count):
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                receipt = build_receipt(
                    schema="matched-cancer-adapter-diagnostic/v1",
                    study_id="matched_cancer_stage_20260730",
                    scenario=(
                        "brca_luad_black_white_calibration_seed32001"
                    ),
                    identities={
                        "predictions": file_identity(nested),
                        "completion_receipt": file_identity(completions[arm]),
                        "cohort_source": file_identity(cohort_records),
                    },
                    fields={
                        "status": "complete",
                        "arm": arm,
                        "head_seed": head_seed,
                        "task_id": f"{cancer.lower()}_tp53",
                    },
                )
                receipt_path = atomic_write_receipt(
                    cell / "DIAGNOSTIC_RECEIPT.json", receipt
                )
                cell_identities[arm][str(head_seed)] = file_identity(
                    receipt_path
                )
                cells_out[(arm, cancer, head_seed)] = (
                    nested, receipt_path
                )
        diagnostic = build_receipt(
            schema="matched-cancer-adapter-diagnostic-root/v1",
            study_id="matched_cancer_stage_20260730",
            scenario="brca_luad_black_white_calibration_seed32001",
            identities={
                "cohort_source": file_identity(cohort_records),
                "completion_receipts": {
                    arm: file_identity(completions[arm]) for arm in ARMS
                },
                "cells": cell_identities,
                "sources": {"runner": file_identity(source_file)},
            },
            fields={
                "status": "complete",
                "task_id": f"{cancer.lower()}_tp53",
                "arms": list(ARMS),
                "head_seeds": list(HEAD_SEEDS),
                "cell_count": 12,
                "race_usage": "output_metadata_only",
            },
        )
        diagnostic_path = atomic_write_receipt(
            cancer_root / "ROOT_DIAGNOSTIC_RECEIPT.json", diagnostic
        )
        diagnostic_receipts[cancer] = file_identity(diagnostic_path)
    loader = build_receipt(
        schema=LOADER_ROOT_SCHEMA,
        study_id="matched_cancer_stage_20260730",
        scenario="brca_luad_black_white_calibration_seed32001",
        identities={
            "deployment_gate": file_identity(gate),
            "cohorts": cohort_receipts,
            "diagnostics": diagnostic_receipts,
            "loader": file_identity(
                Path(__file__).with_name("vetted_loader.py")
            ),
        },
        fields={
            "status": "complete",
            "representation_seed": 32001,
            "cancers": list(CANCERS),
            "arms": list(ARMS),
        },
    )
    loader_path = atomic_write_receipt(
        fixture / "LOADER_ROOT_RECEIPT.json", loader
    )
    return loader_path, cells_out


def write_cell(
    root: Path,
    *,
    fm_seed: int,
    arm: str,
    cancer: str,
    head_seed: int,
    gate: Path,
    loader_root: Path,
    cells: dict[tuple[str, str, int], tuple[Path, Path]],
    count: int = 10,
) -> Path:
    nested, receipt_path = cells[(arm, cancer, head_seed)]
    output = root / f"sealed-{arm}-{cancer}-{head_seed}.jsonl"
    return export_cell(
        nested_predictions=nested,
        diagnostic_receipt=receipt_path,
        destination=output,
        fm_seed=fm_seed,
        arm=arm,
        cancer=cancer,
        head_seed=head_seed,
        deployment_gate_receipt=gate,
        loader_root_receipt=loader_root,
        expected_cohort_size=count,
    )


class ExporterTests(unittest.TestCase):
    def test_export_is_exactly_accepted_by_independent_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, _, completions = make_deployment_gate(root)
            loader_root, cells = make_loader_fixture(
                root, gate=gate, completions=completions
            )
            output = write_cell(
                root,
                fm_seed=32001,
                arm="B",
                cancer="BRCA",
                head_seed=42001,
                gate=gate,
                loader_root=loader_root,
                cells=cells,
            )
            rows = [
                json.loads(line) for line in output.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 50)
            for line_number, row in enumerate(rows, 1):
                self.assertEqual(set(row), FIELDS)
                verifier._validate_row(row, line_number)

    def test_diagnostic_receipt_coordinate_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, _, completions = make_deployment_gate(root)
            loader_root, cells = make_loader_fixture(
                root, gate=gate, completions=completions
            )
            nested, receipt_path = cells[("P", "BRCA", 42001)]
            with self.assertRaisesRegex(ValueError, "coordinate mismatch"):
                export_cell(
                    nested_predictions=nested,
                    diagnostic_receipt=receipt_path,
                    destination=root / "sealed.jsonl",
                    fm_seed=32001,
                    arm="B",
                    cancer="BRCA",
                    head_seed=42001,
                    deployment_gate_receipt=gate,
                    loader_root_receipt=loader_root,
                    expected_cohort_size=10,
                )

    def test_seed_and_cancer_cannot_be_relabelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, _, completions = make_deployment_gate(root)
            loader_root, cells = make_loader_fixture(
                root, gate=gate, completions=completions
            )
            nested, receipt_path = cells[("B", "BRCA", 42001)]
            common = {
                "nested_predictions": nested,
                "diagnostic_receipt": receipt_path,
                "destination": root / "sealed.jsonl",
                "arm": "B",
                "head_seed": 42001,
                "deployment_gate_receipt": gate,
                "loader_root_receipt": loader_root,
                "expected_cohort_size": 10,
            }
            with self.assertRaisesRegex(ValueError, "fm_seed"):
                export_cell(fm_seed=32002, cancer="BRCA", **common)
            with self.assertRaisesRegex(ValueError, "task is not cancer-bound"):
                export_cell(fm_seed=32001, cancer="LUAD", **common)

    def test_fresh_forged_cell_is_not_in_loader_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, _, completions = make_deployment_gate(root)
            loader_root, cells = make_loader_fixture(
                root, gate=gate, completions=completions
            )
            nested, bound_receipt = cells[("B", "BRCA", 42001)]
            bound = json.loads(bound_receipt.read_text())
            forged = build_receipt(
                schema=bound["schema"],
                study_id=bound["study_id"],
                scenario=bound["scenario"],
                identities=bound["identities"],
                fields={
                    key: value
                    for key, value in bound.items()
                    if key not in {
                        "schema", "study_id", "scenario", "identities",
                        "topology_sha256",
                    }
                },
            )
            forged_path = atomic_write_receipt(
                root / "FORGED_CELL_RECEIPT.json", forged
            )
            with self.assertRaisesRegex(ValueError, "loader-root-bound"):
                export_cell(
                    nested_predictions=nested,
                    diagnostic_receipt=forged_path,
                    destination=root / "forged.jsonl",
                    fm_seed=32001,
                    arm="B",
                    cancer="BRCA",
                    head_seed=42001,
                    deployment_gate_receipt=gate,
                    loader_root_receipt=loader_root,
                    expected_cohort_size=10,
                )

    def test_validator_matches_independent_verifier_types(self) -> None:
        row = {
            "schema": "matched-cancer-diagnostic-prediction/v1",
            "fm_seed": 32001,
            "arm": "B",
            "cancer": "BRCA",
            "head_seed": 42001,
            "patient_id": "P1",
            "y_true": 1,
            "race": "Black",
            "fold": 0,
            "role": "outer_test",
            "outer_fold": 0,
            "inner_fold": None,
            "probability": 0.5,
        }
        from tools.matched_cancer_diagnostic_20260730.exporter import (
            validate_prediction_row,
        )

        validate_prediction_row(row)
        for key, invalid in (
            ("fm_seed", 32001.0),
            ("head_seed", 42001.0),
            ("y_true", True),
            ("fold", False),
            ("patient_id", ""),
            ("probability", "0.5"),
        ):
            candidate = dict(row)
            candidate[key] = invalid
            with self.assertRaises(ValueError, msg=key):
                validate_prediction_row(candidate)

    def test_complete_one_seed_matrix_collects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, _, completions = make_deployment_gate(root)
            loader_root, cells = make_loader_fixture(
                root, gate=gate, completions=completions
            )
            exports = [
                write_cell(
                    root,
                    fm_seed=32001,
                    arm=arm,
                    cancer=cancer,
                    head_seed=head,
                    gate=gate,
                    loader_root=loader_root,
                    cells=cells,
                )
                for arm in ARMS
                for cancer in CANCERS
                for head in HEAD_SEEDS
            ]
            collected = collect_exports(
                exports,
                destination=root / "collected.jsonl",
                expected_fm_seeds=[32001],
                cohort_sizes={"BRCA": 10, "LUAD": 10},
            )
            lines = collected.read_text().splitlines()
            self.assertEqual(len(lines), 24 * 50)
            database = root / "verifier.sqlite"
            connection, _, count = verifier.load_predictions(
                collected, database
            )
            try:
                self.assertEqual(count, 24 * 50)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
