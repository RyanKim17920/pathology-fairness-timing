"""Synthetic-only tests for the outcome-blind parquet hardlink-view builder."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from tools.matched_cancer_diagnostic_20260730.tile_view_builder import (
    FOLD_COLUMNS,
    PARQUET_COLUMNS,
    build_tile_views,
    verify_tile_view_receipt,
)


def _write_folds(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    lines = ["patient_barcode,fold,race,tp53_status"]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def _write_parquet(
    path: Path,
    slide_path: str,
    *,
    outcome: int = 999,
) -> None:
    pq.write_table(
        pa.table(
            {
                "slide_path": [slide_path, slide_path],
                "jpeg": [b"not-read-1", b"not-read-2"],
                "tp53_status": [outcome, outcome],
            }
        ),
        path,
        row_group_size=1,
    )


class TileViewBuilderTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, Path]]:
        source = root / "source"
        source.mkdir()
        folds = {"BRCA": root / "brca.csv", "LUAD": root / "luad.csv"}
        _write_folds(
            folds["BRCA"],
            [
                ("TCGA-AA-0001", "target", "white", "1"),
                ("TCGA-AA-0002", "target", "black or african american", "0"),
                ("TCGA-AA-0003", "target", "Asian", "1"),
                ("TCGA-AA-0004", "source", "white", "0"),
            ],
        )
        _write_folds(
            folds["LUAD"],
            [("TCGA-BB-0001", "target", "Black", "1")],
        )
        _write_parquet(
            source / "brca-one.parquet",
            "/slides/TCGA-AA-0001-01A.svs",
        )
        _write_parquet(
            source / "brca-two.parquet",
            "/slides/TCGA-AA-0002-01A.svs",
        )
        _write_parquet(
            source / "brca-excluded.parquet",
            "/slides/TCGA-AA-0003-01A.svs",
        )
        _write_parquet(
            source / "luad-one.parquet",
            "/slides/TCGA-BB-0001-01A.svs",
        )
        _write_parquet(
            source / "unrelated.parquet",
            "/slides/TCGA-ZZ-9999-01A.svs",
        )
        return source, folds

    def test_builds_exact_regular_hardlink_views_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, folds = self._fixture(root)
            receipt_path = build_tile_views(
                source_root=source,
                frozen_folds=folds,
                destination_root=root / "view",
                expected_patient_counts={"BRCA": 2, "LUAD": 1},
                study_id="synthetic",
                scenario="synthetic",
            )
            receipt = verify_tile_view_receipt(receipt_path)
            self.assertEqual(receipt["fold_columns_read"], list(FOLD_COLUMNS))
            self.assertEqual(
                receipt["parquet_columns_read"], list(PARQUET_COLUMNS)
            )
            self.assertFalse(receipt["outcomes_opened"])
            self.assertEqual(receipt["patient_counts"], {"BRCA": 2, "LUAD": 1})
            self.assertEqual(receipt["slide_counts"], {"BRCA": 2, "LUAD": 1})
            self.assertEqual(receipt["file_count"], 3)
            self.assertFalse((root / "view/BRCA/brca-excluded.parquet").exists())
            self.assertFalse((root / "view/BRCA/unrelated.parquet").exists())
            for row in receipt["inventory"].values():
                source_path = Path(
                    receipt["identities"]["source_parquets"][
                        next(
                            key
                            for key, value in receipt["inventory"].items()
                            if value == row
                        )
                    ]["canonical_path"]
                )
                view_path = root / "view" / row["destination_relative"]
                self.assertFalse(view_path.is_symlink())
                self.assertEqual(
                    (source_path.stat().st_dev, source_path.stat().st_ino),
                    (view_path.stat().st_dev, view_path.stat().st_ino),
                )

    def test_existing_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, folds = self._fixture(root)
            destination = root / "view"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                build_tile_views(
                    source_root=source,
                    frozen_folds=folds,
                    destination_root=destination,
                    expected_patient_counts={"BRCA": 2, "LUAD": 1},
                )

    def test_missing_patient_fails_without_publishing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, folds = self._fixture(root)
            (source / "brca-two.parquet").unlink()
            destination = root / "view"
            with self.assertRaisesRegex(ValueError, "missing parquet coverage"):
                build_tile_views(
                    source_root=source,
                    frozen_folds=folds,
                    destination_root=destination,
                    expected_patient_counts={"BRCA": 2, "LUAD": 1},
                )
            self.assertFalse(destination.exists())

    def test_cross_cancer_patient_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, folds = self._fixture(root)
            _write_folds(
                folds["LUAD"],
                [("TCGA-AA-0001", "target", "Black", "1")],
            )
            with self.assertRaisesRegex(ValueError, "cross-cancer"):
                build_tile_views(
                    source_root=source,
                    frozen_folds=folds,
                    destination_root=root / "view",
                    expected_patient_counts={"BRCA": 2, "LUAD": 1},
                )

    def test_duplicate_fold_patient_and_duplicate_slide_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, folds = self._fixture(root)
            _write_folds(
                folds["LUAD"],
                [
                    ("TCGA-BB-0001", "target", "Black", "1"),
                    ("TCGA-BB-0001", "source", "Black", "1"),
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate frozen-fold"):
                build_tile_views(
                    source_root=source,
                    frozen_folds=folds,
                    destination_root=root / "view-one",
                    expected_patient_counts={"BRCA": 2, "LUAD": 1},
                )
            _write_folds(
                folds["LUAD"],
                [("TCGA-BB-0001", "target", "Black", "1")],
            )
            _write_parquet(
                source / "duplicate-slide.parquet",
                "/slides/TCGA-BB-0001-01A.svs",
            )
            with self.assertRaisesRegex(ValueError, "duplicate selected slide"):
                build_tile_views(
                    source_root=source,
                    frozen_folds=folds,
                    destination_root=root / "view-two",
                    expected_patient_counts={"BRCA": 2, "LUAD": 1},
                )

    def test_source_symlink_and_published_identity_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, folds = self._fixture(root)
            os.symlink(
                source / "unrelated.parquet",
                source / "linked.parquet",
            )
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                build_tile_views(
                    source_root=source,
                    frozen_folds=folds,
                    destination_root=root / "bad-view",
                    expected_patient_counts={"BRCA": 2, "LUAD": 1},
                )
            (source / "linked.parquet").unlink()
            receipt_path = build_tile_views(
                source_root=source,
                frozen_folds=folds,
                destination_root=root / "view",
                expected_patient_counts={"BRCA": 2, "LUAD": 1},
                study_id="synthetic",
                scenario="synthetic",
            )
            receipt = json.loads(receipt_path.read_text())
            first = next(iter(receipt["inventory"].values()))
            view_path = root / "view" / first["destination_relative"]
            payload = view_path.read_bytes()
            view_path.unlink()
            view_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "hardlink identity"):
                verify_tile_view_receipt(receipt_path)


if __name__ == "__main__":
    unittest.main()
