#!/usr/bin/env python3
"""Seal explicitly supplied diagnostic sources without interpreting outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730.deployment import (
    AMENDMENT,
    AUTHORIZATION_SCHEMA,
    DEFAULT_CONTRACT,
    load_authorization_manifest,
    load_contract,
)
from tools.matched_cancer_diagnostic_20260730.vetted_loader import (
    SOURCE_SCHEMA,
    SPLIT_SEED,
    TARGET_FOLD,
    TILE_LEDGER_SCHEMA,
    verify_tile_view_receipt,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)


ENTRYPOINT = "tools.matched_cancer_diagnostic_20260730.vetted_loader:load"
LOADER_SOURCE = Path(__file__).with_name("vetted_loader.py")


def _seal_source(
    *,
    destination: Path,
    cancer: str,
    cohort_contract: Mapping[str, object],
    demographics: Path,
    molecular: Path,
    frozen_folds: Path,
    study_id: str,
    scenario: str,
) -> Path:
    receipt = build_receipt(
        schema=SOURCE_SCHEMA,
        study_id=study_id,
        scenario=scenario,
        identities={
            "demographics_csv": file_identity(demographics),
            "molecular_csv": file_identity(molecular),
            "frozen_folds_csv": file_identity(frozen_folds),
            "estimand_amendment": file_identity(AMENDMENT),
        },
        fields={
            "cancer": cancer,
            "task": cohort_contract["task"],
            "target_fold": TARGET_FOLD,
            "expected_target_rows": cohort_contract["expected_target_rows"],
            "expected_eligible_patients": cohort_contract[
                "expected_eligible_patients"
            ],
            "expected_exclusions_by_race": cohort_contract[
                "expected_exclusions_by_race"
            ],
            "expected_race_counts": cohort_contract[
                "expected_race_counts"
            ],
            "split_seed": SPLIT_SEED,
        },
    )
    result = atomic_write_receipt(destination, receipt)
    verify_receipt(result, expected_schema=SOURCE_SCHEMA)
    return result


def _seal_tiles(
    *,
    destination: Path,
    tile_directory: Path,
    cancer: str,
    tile_view_receipt: Path,
    tile_view: Mapping[str, object],
    study_id: str,
    scenario: str,
) -> Path:
    if not tile_directory.is_dir() or tile_directory.is_symlink():
        raise ValueError(f"tile directory is invalid: {tile_directory}")
    view_files = tile_view["identities"]["view_parquets"]  # type: ignore[index]
    files = {
        key: identity
        for key, identity in view_files.items()  # type: ignore[union-attr]
        if Path(identity["canonical_path"]).parent == tile_directory.resolve()
    }
    if not files:
        raise ValueError(f"tile directory has no parquet files: {tile_directory}")
    receipt = build_receipt(
        schema=TILE_LEDGER_SCHEMA,
        study_id=study_id,
        scenario=scenario,
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
    result = atomic_write_receipt(destination, receipt)
    verify_receipt(result, expected_schema=TILE_LEDGER_SCHEMA)
    return result


def build_authorization(
    *,
    contract_path: str | Path,
    demographics: str | Path,
    molecular: str | Path,
    frozen_folds: Mapping[str, str | Path],
    tile_directories: Mapping[str, str | Path],
    tile_view_receipt: str | Path,
    destination_root: str | Path,
) -> Path:
    """Hash and seal explicit paths; never parse labels or cohort rows."""
    if set(frozen_folds) != {"BRCA", "LUAD"} or set(tile_directories) != {
        "BRCA", "LUAD"
    }:
        raise ValueError("authorization builder requires exact BRCA/LUAD inputs")
    contract = load_contract(contract_path)
    root = Path(destination_root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"authorization destination exists: {root}")
    root.mkdir(parents=True)
    demographics_path = Path(demographics).resolve()
    molecular_path = Path(molecular).resolve()
    tile_view_path = Path(tile_view_receipt).resolve()
    resolved_tile_directories = {
        cancer: Path(tile_directories[cancer]).resolve()
        for cancer in ("BRCA", "LUAD")
    }
    tile_view = verify_tile_view_receipt(
        tile_view_path, resolved_tile_directories
    )
    cohorts = {}
    for cancer in ("BRCA", "LUAD"):
        if tile_view["patient_counts"].get(cancer) != contract["cohorts"][
            cancer
        ]["expected_eligible_patients"]:
            raise ValueError(
                f"{cancer} tile-view patient count differs from contract"
            )
        frozen_path = Path(frozen_folds[cancer]).resolve()
        if file_identity(frozen_path) != tile_view["identities"][
            "frozen_folds"
        ][cancer]:
            raise ValueError(
                f"{cancer} frozen folds differ from tile-view receipt"
            )
        cancer_root = root / cancer
        cancer_root.mkdir()
        source = _seal_source(
            destination=cancer_root / "SOURCE_BUNDLE_RECEIPT.json",
            cancer=cancer,
            cohort_contract=contract["cohorts"][cancer],
            demographics=demographics_path,
            molecular=molecular_path,
            frozen_folds=frozen_path,
            study_id=contract["study_id"],
            scenario=contract["calibration_scenario"],
        )
        tile_directory = resolved_tile_directories[cancer]
        ledger = _seal_tiles(
            destination=cancer_root / "TILE_LEDGER_RECEIPT.json",
            tile_directory=tile_directory,
            cancer=cancer,
            tile_view_receipt=tile_view_path,
            tile_view=tile_view,
            study_id=contract["study_id"],
            scenario=contract["calibration_scenario"],
        )
        cohorts[cancer] = {
            "patient_records": str(source),
            "tile_source": str(tile_directory),
            "cohort_ledger": str(ledger),
        }
    manifest = {
        "schema": AUTHORIZATION_SCHEMA,
        "loader_entrypoint": ENTRYPOINT,
        "loader_source": str(LOADER_SOURCE.resolve()),
        "tile_view_receipt": str(tile_view_path),
        "cohorts": cohorts,
    }
    destination = atomic_write_receipt(
        root / "AUTHORIZATION_MANIFEST.json", manifest
    )
    load_authorization_manifest(destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--demographics", required=True)
    parser.add_argument("--molecular", required=True)
    parser.add_argument("--brca-frozen-folds", required=True)
    parser.add_argument("--luad-frozen-folds", required=True)
    parser.add_argument("--brca-tile-directory", required=True)
    parser.add_argument("--luad-tile-directory", required=True)
    parser.add_argument("--tile-view-receipt", required=True)
    parser.add_argument("--destination-root", required=True)
    args = parser.parse_args(argv)
    result = build_authorization(
        contract_path=args.contract,
        demographics=args.demographics,
        molecular=args.molecular,
        frozen_folds={
            "BRCA": args.brca_frozen_folds,
            "LUAD": args.luad_frozen_folds,
        },
        tile_directories={
            "BRCA": args.brca_tile_directory,
            "LUAD": args.luad_tile_directory,
        },
        tile_view_receipt=args.tile_view_receipt,
        destination_root=args.destination_root,
    )
    print(json.dumps({"authorization_manifest": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
