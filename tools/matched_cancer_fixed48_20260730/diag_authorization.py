"""Authorization-v2 wrapper for immutable seed-32001 cohort ancestors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730.deployment import (
    load_authorization_manifest as load_legacy_authorization,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_contract import (
    AUTHORIZATION_SCHEMA,
    CANCERS,
    LEGACY_SCENARIO,
    LEGACY_STUDY_ID,
    STUDY_ID,
)

REPO = Path(__file__).resolve().parents[2]
AMENDMENT = (
    REPO
    / "results/matched_cancer_stage_20260730/"
    "DIAGNOSTIC_FIXED_FINAL_AMENDMENT_01.md"
)
SOURCE_SCHEMA = "matched-cancer-diagnostic-source-bundle/v1"

LOADER_ENTRYPOINT = (
    "tools.matched_cancer_fixed48_20260730.diag_loader:load"
)
LOADER_SOURCE = Path(__file__).with_name("diag_loader.py")


def build_authorization(
    legacy_authorization_manifest: str | Path,
    destination: str | Path,
) -> Path:
    """Seal exact legacy paths without opening molecular outcome values."""
    requested = Path(destination)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(
            f"fixed48 authorization destination exists: {requested}"
        )
    legacy_path = Path(legacy_authorization_manifest).resolve()
    legacy = load_legacy_authorization(legacy_path)
    # Local import avoids an import cycle while binding the complete production
    # closure into the authorization itself, not merely into each seed gate.
    from .diag_deployment import RUNTIME_SOURCE_PATHS

    source_receipts = {
        cancer: verify_receipt(
            legacy["cohorts"][cancer]["patient_records"],
            expected_schema=SOURCE_SCHEMA,
            expected_study_id=LEGACY_STUDY_ID,
            expected_scenario=LEGACY_SCENARIO,
        )
        for cancer in CANCERS
    }
    amendment_identity = file_identity(AMENDMENT)
    if any(
        receipt.get("identities", {}).get("estimand_amendment")
        != amendment_identity
        for receipt in source_receipts.values()
    ):
        raise ValueError("legacy source bundle amendment ancestry differs")
    receipt = build_receipt(
        schema=AUTHORIZATION_SCHEMA,
        study_id=STUDY_ID,
        scenario=LEGACY_SCENARIO,
        identities={
            "legacy_authorization_manifest": file_identity(legacy_path),
            "legacy_loader_source": file_identity(legacy["loader_source"]),
            "legacy_tile_view_receipt": file_identity(
                legacy["tile_view_receipt"]
            ),
            "legacy_cohorts": {
                cancer: {
                    "source_bundle": file_identity(
                        legacy["cohorts"][cancer]["patient_records"]
                    ),
                    "tile_ledger": file_identity(
                        legacy["cohorts"][cancer]["cohort_ledger"]
                    ),
                }
                for cancer in CANCERS
            },
            "loader_source": file_identity(LOADER_SOURCE),
            "estimand_amendment": amendment_identity,
            "runtime_sources": {
                name: file_identity(path)
                for name, path in RUNTIME_SOURCE_PATHS.items()
            },
        },
        fields={
            "status": "valid",
            "loader_entrypoint": LOADER_ENTRYPOINT,
            "legacy_study_id": LEGACY_STUDY_ID,
            "legacy_scenario": LEGACY_SCENARIO,
            "outcomes_opened": False,
        },
    )
    output = atomic_write_receipt(destination, receipt)
    verify_authorization(output)
    return output


def verify_authorization(path: str | Path) -> dict[str, Any]:
    from .diag_deployment import RUNTIME_SOURCE_PATHS

    receipt = verify_receipt(
        path, expected_schema=AUTHORIZATION_SCHEMA, expected_study_id=STUDY_ID
    )
    expected_fields = {
        "status": "valid",
        "loader_entrypoint": LOADER_ENTRYPOINT,
        "legacy_study_id": LEGACY_STUDY_ID,
        "legacy_scenario": LEGACY_SCENARIO,
        "outcomes_opened": False,
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            raise ValueError(f"fixed48 authorization {key} differs")
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "legacy_authorization_manifest",
        "legacy_loader_source",
        "legacy_tile_view_receipt",
        "legacy_cohorts",
        "loader_source",
        "estimand_amendment",
        "runtime_sources",
    }:
        raise ValueError("fixed48 authorization identity topology differs")
    if set(identities["legacy_cohorts"]) != set(CANCERS):
        raise ValueError("fixed48 authorization cohort topology differs")
    legacy_path = identities["legacy_authorization_manifest"]["canonical_path"]
    if file_identity(legacy_path) != identities["legacy_authorization_manifest"]:
        raise ValueError("legacy authorization identity drift")
    legacy = load_legacy_authorization(legacy_path)
    if file_identity(legacy["loader_source"]) != identities[
        "legacy_loader_source"
    ]:
        raise ValueError("legacy loader identity drift")
    if file_identity(legacy["tile_view_receipt"]) != identities[
        "legacy_tile_view_receipt"
    ]:
        raise ValueError("legacy tile-view identity drift")
    for cancer in CANCERS:
        bound = identities["legacy_cohorts"][cancer]
        if not isinstance(bound, Mapping) or set(bound) != {
            "source_bundle", "tile_ledger"
        }:
            raise ValueError(f"{cancer} legacy authorization topology differs")
        declaration = legacy["cohorts"][cancer]
        if bound["source_bundle"] != file_identity(
            declaration["patient_records"]
        ):
            raise ValueError(f"{cancer} legacy source bundle was redirected")
        if bound["tile_ledger"] != file_identity(
            declaration["cohort_ledger"]
        ):
            raise ValueError(f"{cancer} legacy tile ledger was redirected")
        source_receipt = verify_receipt(
            declaration["patient_records"],
            expected_schema=SOURCE_SCHEMA,
            expected_study_id=LEGACY_STUDY_ID,
            expected_scenario=LEGACY_SCENARIO,
        )
        if source_receipt.get("identities", {}).get(
            "estimand_amendment"
        ) != identities["estimand_amendment"]:
            raise ValueError(
                f"{cancer} legacy source amendment ancestry differs"
            )
    if identities["estimand_amendment"] != file_identity(AMENDMENT):
        raise ValueError("fixed48 estimand amendment identity drift")
    if identities["loader_source"] != file_identity(LOADER_SOURCE):
        raise ValueError("fixed48 loader source identity drift")
    if set(identities["runtime_sources"]) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("fixed48 authorization runtime topology differs")
    for name, source in RUNTIME_SOURCE_PATHS.items():
        if identities["runtime_sources"][name] != file_identity(source):
            raise ValueError(
                f"fixed48 authorization runtime source {name} drift"
            )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--legacy-authorization-manifest", required=True)
    build.add_argument("--destination", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--authorization-manifest", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build_authorization(
            args.legacy_authorization_manifest, args.destination
        )
    else:
        verify_authorization(args.authorization_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
