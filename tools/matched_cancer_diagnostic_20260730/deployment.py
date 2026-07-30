#!/usr/bin/env python3
"""Outcome-blind deployment gate and explicit real-loader CLI scaffold."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730.exporter import (
    ARMS, CANCERS, COHORT_SIZES, HEAD_SEEDS, ROW_SCHEMA,
)
from tools.matched_cancer_diagnostic_20260730.integration import (
    verify_calibration_ancestry,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)


CONTRACT_SCHEMA = "matched-cancer-diagnostic-deployment/v1"
GATE_SCHEMA = "matched-cancer-diagnostic-deployment-gate/v1"
AUTHORIZATION_SCHEMA = "matched-cancer-diagnostic-authorization/v1"
LOADER_ROOT_SCHEMA = "matched-cancer-diagnostic-loader-root/v1"
COHORT_SCHEMA = "matched-cancer-diagnostic-cohort/v1"
DIAGNOSTIC_ROOT_SCHEMA = "matched-cancer-adapter-diagnostic-root/v1"
DIAGNOSTIC_CELL_SCHEMA = "matched-cancer-adapter-diagnostic/v1"
DEFAULT_CONTRACT = Path(__file__).with_name("deployment_seed32001.json")
VETTED_LOADER_ENTRYPOINT = (
    "tools.matched_cancer_diagnostic_20260730.vetted_loader:load"
)
VETTED_LOADER_SOURCE = Path(__file__).with_name("vetted_loader.py")
REPO = Path(__file__).resolve().parents[2]
AMENDMENT = (
    REPO
    / "results/matched_cancer_stage_20260730/"
    "DIAGNOSTIC_FIXED_FINAL_AMENDMENT_01.md"
)
RUNTIME_SOURCE_PATHS = {
    "diagnostic_package_init": (
        REPO / "tools/matched_cancer_diagnostic_20260730/__init__.py"
    ),
    "stage_package_init": (
        REPO / "tools/matched_cancer_stage_20260730/__init__.py"
    ),
    "union_package_init": (
        REPO / "tools/matched_stage_union_20260730/__init__.py"
    ),
    "deployment": Path(__file__),
    "integration": Path(__file__).with_name("integration.py"),
    "runner": Path(__file__).with_name("runner.py"),
    "cache": Path(__file__).with_name("cache.py"),
    "exporter": Path(__file__).with_name("exporter.py"),
    "driver": Path(__file__).with_name("diagnostic_seed32001.sbatch"),
    "stage_objectives": (
        REPO / "tools/matched_cancer_stage_20260730/objectives.py"
    ),
    "union_objectives": (
        REPO / "tools/matched_stage_union_20260730/objectives.py"
    ),
    "receipts": REPO / "tools/matched_cancer_stage_20260730/receipts.py",
    "completion_receipt": (
        REPO / "tools/matched_cancer_stage_20260730/completion_receipt.py"
    ),
    "reliable_fairness_head": REPO / "tools/reliable_fairness_head.py",
    "encoder_model": REPO / "vendor/matched_stage_train_20260730/model.py",
    "fairness_eval": REPO / "tools/fairness_eval.py",
    "post_hoc_debias": REPO / "tools/post_hoc_debias.py",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def load_contract(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    contract = _load_json_object(source)
    expected = {
        "schema": CONTRACT_SCHEMA,
        "study_id": "matched_cancer_stage_20260730",
        "calibration_scenario": (
            "brca_luad_black_white_calibration_seed32001"
        ),
        "representation_seed": 32001,
        "arms": list(ARMS),
        "head_seeds": list(HEAD_SEEDS),
        "folds": list(range(5)),
        "prediction_schema": ROW_SCHEMA,
        "loader_protocol": (
            "load(contract=..., cancers=..., authorized_paths=..., "
            "output_root=..., gate_receipt=...) -> non-None"
        ),
    }
    if set(contract) != set(expected) | {"cohorts"}:
        raise ValueError("deployment contract top-level fields drift")
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"deployment contract {key} drift")
    cohorts = contract.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(CANCERS):
        raise ValueError("deployment contract cancer topology drift")
    expected_cohorts = {
        "BRCA": {
            "logical_name": "brca_racepanel",
            "expected_target_rows": 334,
            "expected_eligible_patients": 328,
            "expected_exclusions_by_race": {
                "Asian": 5,
                "American Indian or Alaska Native": 1,
            },
            "expected_race_counts": {"Black": 118, "White": 210},
            "task": "brca_tp53",
        },
        "LUAD": {
            "logical_name": "luad_target_hospitals",
            "expected_target_rows": 281,
            "expected_eligible_patients": 281,
            "expected_exclusions_by_race": {},
            "expected_race_counts": {"Black": 40, "White": 241},
            "task": "luad_tp53",
        },
    }
    if cohorts != expected_cohorts:
        raise ValueError("deployment cohort names/tasks/sizes drift")
    return contract


def load_authorization_manifest(path: str | Path) -> dict[str, Any]:
    """Validate path declarations without opening any declared data path."""
    source = Path(path).resolve()
    manifest = _load_json_object(source)
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "loader_entrypoint", "loader_source", "tile_view_receipt",
        "cohorts",
    }:
        raise ValueError("authorization manifest fields differ")
    if manifest["schema"] != AUTHORIZATION_SCHEMA:
        raise ValueError("authorization manifest schema differs")
    entrypoint = manifest["loader_entrypoint"]
    if entrypoint != VETTED_LOADER_ENTRYPOINT:
        raise ValueError("authorization manifest loader is not the vetted loader")
    loader_source = manifest["loader_source"]
    if not isinstance(loader_source, str) or not Path(loader_source).is_absolute():
        raise ValueError("authorization loader_source must be absolute")
    source_path = Path(loader_source).resolve(strict=True)
    if source_path != VETTED_LOADER_SOURCE.resolve(strict=True):
        raise ValueError("authorization loader_source is not the vetted source")
    tile_view_receipt = manifest["tile_view_receipt"]
    if (
        not isinstance(tile_view_receipt, str)
        or not Path(tile_view_receipt).is_absolute()
    ):
        raise ValueError("authorization tile-view receipt must be absolute")
    file_identity(tile_view_receipt)
    cohorts = manifest["cohorts"]
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(CANCERS):
        raise ValueError("authorization cohort topology differs")
    for cancer in CANCERS:
        declaration = cohorts[cancer]
        if not isinstance(declaration, Mapping) or set(declaration) != {
            "patient_records", "tile_source", "cohort_ledger"
        }:
            raise ValueError(f"{cancer} authorization fields differ")
        for role, value in declaration.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{cancer} authorization {role} is invalid")
    return manifest


def preflight(
    *,
    contract_path: str | Path,
    calibration_root_receipt: str | Path,
    completion_receipts: Mapping[str, str | Path],
    authorization_manifest: str | Path,
    destination: str | Path,
) -> Path:
    """Verify deployment constants and ancestry without opening outcomes."""
    contract_source = Path(contract_path).resolve()
    contract = load_contract(contract_source)
    authorization_path = Path(authorization_manifest).resolve()
    authorization = load_authorization_manifest(authorization_path)
    ancestry = verify_calibration_ancestry(
        calibration_root_receipt,
        completion_receipts,
        expected_representation_seed=contract["representation_seed"],
    )
    if ancestry["root"]["scenario"] != contract["calibration_scenario"]:
        raise ValueError("deployment/calibration scenario mismatch")
    receipt = build_receipt(
        schema=GATE_SCHEMA,
        study_id=contract["study_id"],
        scenario=contract["calibration_scenario"],
        identities={
            "deployment_contract": file_identity(contract_source),
            "calibration_root_receipt": file_identity(
                ancestry["root_path"]
            ),
            "completion_receipts": {
                arm: file_identity(completion_receipts[arm]) for arm in ARMS
            },
            "authorization_manifest": file_identity(authorization_path),
            "loader_source": file_identity(authorization["loader_source"]),
            "tile_view_receipt": file_identity(
                authorization["tile_view_receipt"]
            ),
            "cohort_authorizations": {
                cancer: {
                    "source_bundle": file_identity(
                        authorization["cohorts"][cancer]["patient_records"]
                    ),
                    "tile_ledger": file_identity(
                        authorization["cohorts"][cancer]["cohort_ledger"]
                    ),
                }
                for cancer in CANCERS
            },
            "estimand_amendment": file_identity(AMENDMENT),
            "sources": {
                name: file_identity(path)
                for name, path in RUNTIME_SOURCE_PATHS.items()
            },
        },
        fields={
            "status": "valid",
            "representation_seed": contract["representation_seed"],
            "outcomes_opened": False,
            "cohort_sizes": dict(COHORT_SIZES),
            "head_seeds": list(HEAD_SEEDS),
            "loader_entrypoint": VETTED_LOADER_ENTRYPOINT,
        },
    )
    requested = Path(destination)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"deployment gate already exists: {requested}")
    output = atomic_write_receipt(destination, receipt)
    verify_receipt(output, expected_schema=GATE_SCHEMA)
    return output


def resolve_loader(entrypoint: str):
    """Resolve an explicitly authorized loader only after gate verification."""
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("loader entrypoint must be module:function")
    return getattr(importlib.import_module(module_name), function_name)


def verify_gate(path: str | Path) -> dict[str, Any]:
    receipt = verify_receipt(path, expected_schema=GATE_SCHEMA)
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "deployment_contract", "calibration_root_receipt",
        "completion_receipts", "authorization_manifest", "loader_source",
        "tile_view_receipt", "cohort_authorizations",
        "estimand_amendment", "sources",
    }:
        raise ValueError("deployment gate identity topology differs")
    if set(identities["completion_receipts"]) != set(ARMS):
        raise ValueError("deployment gate completion topology differs")
    if set(identities["sources"]) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("deployment gate source topology differs")
    for name, path in RUNTIME_SOURCE_PATHS.items():
        if identities["sources"][name] != file_identity(path):
            raise ValueError(f"deployment gate {name} source identity differs")
    contract = load_contract(
        identities["deployment_contract"]["canonical_path"]
    )
    authorization = load_authorization_manifest(
        identities["authorization_manifest"]["canonical_path"]
    )
    if file_identity(authorization["loader_source"]) != identities["loader_source"]:
        raise ValueError("deployment loader source identity differs")
    if (
        file_identity(authorization["tile_view_receipt"])
        != identities["tile_view_receipt"]
    ):
        raise ValueError("deployment tile-view receipt identity differs")
    if (
        not isinstance(identities["cohort_authorizations"], Mapping)
        or set(identities["cohort_authorizations"]) != set(CANCERS)
    ):
        raise ValueError("deployment cohort-authorization topology differs")
    contract_cohorts = contract["cohorts"]
    for cancer in CANCERS:
        bound = identities["cohort_authorizations"][cancer]
        if not isinstance(bound, Mapping) or set(bound) != {
            "source_bundle", "tile_ledger"
        }:
            raise ValueError(
                f"deployment {cancer} authorization topology differs"
            )
        declaration = authorization["cohorts"][cancer]
        if bound["source_bundle"] != file_identity(
            declaration["patient_records"]
        ):
            raise ValueError(
                f"deployment {cancer} source-bundle identity differs"
            )
        if bound["tile_ledger"] != file_identity(
            declaration["cohort_ledger"]
        ):
            raise ValueError(
                f"deployment {cancer} tile-ledger identity differs"
            )
        source_bundle = verify_receipt(
            declaration["patient_records"],
            expected_schema="matched-cancer-diagnostic-source-bundle/v1",
            expected_study_id=receipt["study_id"],
            expected_scenario=receipt["scenario"],
        )
        if set(source_bundle.get("identities", {})) != {
            "demographics_csv",
            "molecular_csv",
            "frozen_folds_csv",
            "estimand_amendment",
        }:
            raise ValueError(f"deployment {cancer} source topology differs")
        if source_bundle["identities"]["estimand_amendment"] != identities[
            "estimand_amendment"
        ]:
            raise ValueError(
                f"deployment {cancer} source amendment ancestry differs"
            )
        expected_source_fields = {
            "cancer": cancer,
            "task": contract_cohorts[cancer]["task"],
            "target_fold": "target",
            "expected_target_rows": contract_cohorts[cancer][
                "expected_target_rows"
            ],
            "expected_eligible_patients": contract_cohorts[cancer][
                "expected_eligible_patients"
            ],
            "expected_exclusions_by_race": contract_cohorts[cancer][
                "expected_exclusions_by_race"
            ],
            "expected_race_counts": contract_cohorts[cancer][
                "expected_race_counts"
            ],
            "split_seed": 288_850_999,
        }
        for key, value in expected_source_fields.items():
            if source_bundle.get(key) != value:
                raise ValueError(
                    f"deployment {cancer} source {key} differs"
                )
    completions = {
        arm: identities["completion_receipts"][arm]["canonical_path"]
        for arm in ARMS
    }
    ancestry = verify_calibration_ancestry(
        identities["calibration_root_receipt"]["canonical_path"],
        completions,
        expected_representation_seed=contract["representation_seed"],
    )
    expected_fields = {
        "status": "valid",
        "representation_seed": contract["representation_seed"],
        "outcomes_opened": False,
        "cohort_sizes": dict(COHORT_SIZES),
        "head_seeds": list(HEAD_SEEDS),
        "loader_entrypoint": authorization["loader_entrypoint"],
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"deployment gate {key} drift")
    if set(receipt) != {
        "schema", "study_id", "scenario", "identities", "topology_sha256",
        *expected_fields,
    }:
        raise ValueError("deployment gate top-level fields differ")
    if (
        receipt.get("study_id") != contract["study_id"]
        or receipt.get("scenario") != contract["calibration_scenario"]
        or ancestry["root"]["scenario"] != contract["calibration_scenario"]
    ):
        raise ValueError("deployment gate study/scenario drift")
    return receipt


def verify_loader_result(
    path: str | Path, gate_receipt: str | Path
) -> dict[str, Any]:
    """Verify the exact two-cancer, 24-cell loader postcondition."""
    gate = verify_gate(gate_receipt)
    result = verify_receipt(
        path,
        expected_schema=LOADER_ROOT_SCHEMA,
        expected_study_id=gate["study_id"],
        expected_scenario=gate["scenario"],
    )
    identities = result.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "deployment_gate", "cohorts", "diagnostics", "loader"
    }:
        raise ValueError("loader-root identity topology differs")
    if identities["deployment_gate"] != file_identity(gate_receipt):
        raise ValueError("loader root descends from a different deployment gate")
    if identities["loader"] != file_identity(VETTED_LOADER_SOURCE):
        raise ValueError("loader root does not bind the vetted loader")
    if set(identities["cohorts"]) != set(CANCERS) or set(
        identities["diagnostics"]
    ) != set(CANCERS):
        raise ValueError("loader-root cancer topology differs")
    if (
        result.get("status") != "complete"
        or result.get("representation_seed") != gate["representation_seed"]
        or result.get("cancers") != list(CANCERS)
        or result.get("arms") != list(ARMS)
    ):
        raise ValueError("loader-root completion fields differ")
    contract = load_contract(
        gate["identities"]["deployment_contract"]["canonical_path"]
    )
    authorization = load_authorization_manifest(
        gate["identities"]["authorization_manifest"]["canonical_path"]
    )
    for cancer in CANCERS:
        cohort = verify_receipt(
            identities["cohorts"][cancer]["canonical_path"],
            expected_schema=COHORT_SCHEMA,
            expected_study_id=gate["study_id"],
            expected_scenario=gate["scenario"],
        )
        cohort_contract = contract["cohorts"][cancer]
        cohort_ids = cohort.get("identities", {})
        if set(cohort_ids) != {
            "source_bundle", "tile_ledger", "cohort_records", "loader"
        }:
            raise ValueError(f"{cancer} cohort identity topology differs")
        if cohort_ids["source_bundle"] != file_identity(
            authorization["cohorts"][cancer]["patient_records"]
        ):
            raise ValueError(f"{cancer} cohort source authorization differs")
        if cohort_ids["tile_ledger"] != file_identity(
            authorization["cohorts"][cancer]["cohort_ledger"]
        ):
            raise ValueError(f"{cancer} tile authorization differs")
        if cohort_ids["loader"] != file_identity(VETTED_LOADER_SOURCE):
            raise ValueError(f"{cancer} cohort loader identity differs")
        source_bundle = verify_receipt(
            cohort_ids["source_bundle"]["canonical_path"],
            expected_schema="matched-cancer-diagnostic-source-bundle/v1",
            expected_study_id=gate["study_id"],
            expected_scenario=gate["scenario"],
        )
        if source_bundle.get("identities", {}).get(
            "estimand_amendment"
        ) != gate["identities"]["estimand_amendment"]:
            raise ValueError(f"{cancer} source amendment ancestry differs")
        tile_ledger = verify_receipt(
            cohort_ids["tile_ledger"]["canonical_path"],
            expected_schema="matched-cancer-diagnostic-tile-ledger/v1",
            expected_study_id=gate["study_id"],
            expected_scenario=gate["scenario"],
        )
        if tile_ledger.get("identities", {}).get(
            "tile_view_receipt"
        ) != gate["identities"]["tile_view_receipt"]:
            raise ValueError(f"{cancer} tile-view ancestry differs")
        expected_cohort_fields = {
            "cancer": cancer,
            "task": cohort_contract["task"],
            "raw_target_count": cohort_contract["expected_target_rows"],
            "eligible_patient_count": cohort_contract[
                "expected_eligible_patients"
            ],
            "patient_count": cohort_contract["expected_eligible_patients"],
            "exclusions_by_race": cohort_contract[
                "expected_exclusions_by_race"
            ],
            "race_counts": cohort_contract["expected_race_counts"],
            "split_seed": 288_850_999,
        }
        for key, value in expected_cohort_fields.items():
            if cohort.get(key) != value:
                raise ValueError(f"{cancer} cohort postcondition {key} differs")
        for key in ("eligible_patient_ids_sha256", "fold_sha256"):
            value = cohort.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{cancer} cohort {key} is invalid")

        diagnostic_path = identities["diagnostics"][cancer]["canonical_path"]
        diagnostic = verify_receipt(
            diagnostic_path,
            expected_schema=DIAGNOSTIC_ROOT_SCHEMA,
            expected_study_id=gate["study_id"],
            expected_scenario=gate["scenario"],
        )
        if (
            diagnostic.get("status") != "complete"
            or diagnostic.get("task_id") != cohort_contract["task"]
            or diagnostic.get("arms") != list(ARMS)
            or diagnostic.get("head_seeds") != list(HEAD_SEEDS)
            or diagnostic.get("cell_count") != len(ARMS) * len(HEAD_SEEDS)
            or diagnostic.get("race_usage") != "output_metadata_only"
        ):
            raise ValueError(f"{cancer} diagnostic-root fields differ")
        diagnostic_ids = diagnostic.get("identities", {})
        if set(diagnostic_ids) != {
            "cohort_source", "completion_receipts", "cells", "sources"
        }:
            raise ValueError(f"{cancer} diagnostic identity topology differs")
        if diagnostic_ids["cohort_source"] != cohort["identities"].get(
            "cohort_records"
        ):
            raise ValueError(f"{cancer} diagnostic cohort ancestry differs")
        if diagnostic_ids["completion_receipts"] != gate["identities"][
            "completion_receipts"
        ]:
            raise ValueError(f"{cancer} diagnostic completion ancestry differs")
        cells = diagnostic_ids["cells"]
        if set(cells) != set(ARMS) or any(
            set(cells[arm]) != {str(head) for head in HEAD_SEEDS}
            for arm in ARMS
        ):
            raise ValueError(f"{cancer} diagnostic cell topology differs")
        for arm in ARMS:
            for head in HEAD_SEEDS:
                cell = verify_receipt(
                    cells[arm][str(head)]["canonical_path"],
                    expected_schema=DIAGNOSTIC_CELL_SCHEMA,
                    expected_study_id=gate["study_id"],
                    expected_scenario=gate["scenario"],
                )
                if (
                    cell.get("arm") != arm
                    or cell.get("head_seed") != head
                    or cell.get("task_id") != cohort_contract["task"]
                    or cell.get("identities", {}).get("completion_receipt")
                    != gate["identities"]["completion_receipts"][arm]
                    or cell.get("identities", {}).get("cohort_source")
                    != diagnostic_ids["cohort_source"]
                ):
                    raise ValueError(
                        f"{cancer}/{arm}/{head} diagnostic cell ancestry differs"
                    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    check.add_argument("--calibration-root", required=True)
    for arm in ARMS:
        check.add_argument(f"--completion-{arm.lower()}", required=True)
    check.add_argument("--authorization-manifest", required=True)
    check.add_argument("--write-gate", required=True)
    run = sub.add_parser("run")
    run.add_argument("--gate-receipt", required=True)
    run.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(
            contract_path=args.contract,
            calibration_root_receipt=args.calibration_root,
            completion_receipts={
                "B": args.completion_b,
                "P": args.completion_p,
                "H": args.completion_h,
            },
            authorization_manifest=args.authorization_manifest,
            destination=args.write_gate,
        )
        return 0
    gate = verify_gate(args.gate_receipt)
    contract_path = gate["identities"]["deployment_contract"]["canonical_path"]
    contract = load_contract(contract_path)
    authorization_path = gate["identities"]["authorization_manifest"][
        "canonical_path"
    ]
    authorization = load_authorization_manifest(authorization_path)
    loader = resolve_loader(gate["loader_entrypoint"])
    loader_path = Path(inspect.getsourcefile(loader) or "").resolve()
    if file_identity(loader_path) != gate["identities"]["loader_source"]:
        raise ValueError("resolved loader does not match gate-bound source")
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"diagnostic output root exists: {output_root}")
    # The loader is called only after every bound gate file re-verifies. It must
    # return authorized PatientRecord rows, tile bytes, and a sealed cohort ledger.
    result = loader(
        contract=contract,
        cancers=CANCERS,
        authorized_paths=authorization,
        output_root=output_root,
        gate_receipt=Path(args.gate_receipt).resolve(),
    )
    if result is None:
        raise RuntimeError("authorized loader returned no deployment result")
    verify_loader_result(result, args.gate_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
