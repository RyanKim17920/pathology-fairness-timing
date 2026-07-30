"""Seed-parameterized loader with an explicit legacy-cohort adapter."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730.deployment import (
    load_authorization_manifest as load_legacy_authorization,
)
from tools.matched_cancer_diagnostic_20260730.runner import (
    load_frozen_representation,
    run_paired_diagnostic,
)
from tools.matched_cancer_diagnostic_20260730 import runner as legacy_runner
from tools.matched_cancer_diagnostic_20260730 import (
    vetted_loader as legacy_loader_module,
)
from tools.matched_cancer_diagnostic_20260730.vetted_loader import (
    prepare_cohort as legacy_prepare_cohort,
)
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_authorization import verify_authorization
from .diag_contract import (
    ARMS,
    CANCERS,
    COHORT_SCHEMA,
    LEGACY_SCENARIO,
    LEGACY_STUDY_ID,
    LOADER_ROOT_SCHEMA,
    STUDY_ID,
    load_contract,
)
from .diag_deployment import verify_gate


LEGACY_COHORT_SCHEMA = "matched-cancer-diagnostic-cohort/v1"
LEGACY_SOURCE_SCHEMA = "matched-cancer-diagnostic-source-bundle/v1"
DIAGNOSTIC_ROOT_SCHEMA = "matched-cancer-adapter-diagnostic-root/v1"
LEGACY_COHORT_SEMANTIC_FIELDS = {
    "cancer", "task", "patient_count", "raw_target_count",
    "eligible_patient_count", "tile_count", "split_seed", "fold_counts",
    "exclusions_by_race", "race_counts", "eligible_patient_ids_sha256",
    "fold_sha256",
}


def _verify_runtime_callables(gate: Mapping[str, Any]) -> None:
    from tools import fairness_eval, post_hoc_debias

    checks = {
        "legacy_runner": Path(legacy_runner.__file__ or ""),
        "legacy_vetted_loader": Path(legacy_loader_module.__file__ or ""),
        "fairness_eval": Path(fairness_eval.__file__ or ""),
        "post_hoc_debias": Path(post_hoc_debias.__file__ or ""),
    }
    for name, path in checks.items():
        if file_identity(path.resolve()) != gate["identities"]["sources"][name]:
            raise ValueError(f"runtime import {name} was redirected")
    function_sources = {
        load_frozen_representation: "legacy_runner",
        run_paired_diagnostic: "legacy_runner",
        legacy_prepare_cohort: "legacy_vetted_loader",
        legacy_runner.StageAdapter: "stage_objectives",
    }
    for value, source_name in function_sources.items():
        path = Path(inspect.getsourcefile(value) or "").resolve()
        if file_identity(path) != gate["identities"]["sources"][source_name]:
            raise ValueError(f"runtime callable {source_name} was redirected")


def _wrap_cohort(
    *,
    cancer: str,
    legacy_receipt_path: Path,
    cohort_records: Path,
    gate_receipt: Path,
    destination: Path,
) -> Path:
    legacy = verify_receipt(
        legacy_receipt_path,
        expected_schema=LEGACY_COHORT_SCHEMA,
        expected_study_id=LEGACY_STUDY_ID,
        expected_scenario=LEGACY_SCENARIO,
    )
    if set(legacy) != {
        "schema", "study_id", "scenario", "identities", "topology_sha256",
        *LEGACY_COHORT_SEMANTIC_FIELDS,
    }:
        raise ValueError("legacy cohort receipt semantic topology differs")
    if legacy["identities"].get("cohort_records") != file_identity(
        cohort_records
    ):
        raise ValueError("legacy cohort receipt does not bind cohort records")
    gate = verify_gate(gate_receipt)
    receipt = build_receipt(
        schema=COHORT_SCHEMA,
        study_id=STUDY_ID,
        scenario=gate["scenario"],
        identities={
            "deployment_gate": file_identity(gate_receipt),
            "legacy_cohort_receipt": file_identity(legacy_receipt_path),
            "legacy_source_bundle": legacy["identities"]["source_bundle"],
            "legacy_tile_ledger": legacy["identities"]["tile_ledger"],
            "cohort_records": file_identity(cohort_records),
            "loader": file_identity(Path(__file__)),
        },
        fields={
            "status": "complete",
            "representation_seed": gate["representation_seed"],
            "cancer": cancer,
            "task": legacy["task"],
            "patient_count": legacy["patient_count"],
            "raw_target_count": legacy["raw_target_count"],
            "eligible_patient_count": legacy["eligible_patient_count"],
            "tile_count": legacy["tile_count"],
            "split_seed": legacy["split_seed"],
            "fold_counts": legacy["fold_counts"],
            "exclusions_by_race": legacy["exclusions_by_race"],
            "race_counts": legacy["race_counts"],
            "eligible_patient_ids_sha256": legacy[
                "eligible_patient_ids_sha256"
            ],
            "fold_sha256": legacy["fold_sha256"],
            "legacy_study_id": LEGACY_STUDY_ID,
            "legacy_scenario": LEGACY_SCENARIO,
        },
    )
    return atomic_write_receipt(destination, receipt)


def load(
    *,
    contract: Mapping[str, Any],
    cancers: Sequence[str],
    authorization_manifest: str | Path,
    output_root: str | Path,
    gate_receipt: str | Path,
) -> Path:
    """Run a current-seed diagnostic without relabeling legacy data receipts."""
    gate_path = Path(gate_receipt).resolve()
    gate = verify_gate(gate_path)
    _verify_runtime_callables(gate)
    bound_contract = load_contract(
        gate["identities"]["deployment_contract"]["canonical_path"]
    )
    if dict(contract) != bound_contract:
        raise ValueError("loader contract differs from gate")
    if tuple(cancers) != CANCERS:
        raise ValueError("loader requires exact BRCA/LUAD order")
    auth_path = Path(authorization_manifest).resolve()
    if file_identity(auth_path) != gate["identities"][
        "authorization_manifest"
    ]:
        raise ValueError("loader authorization differs from gate")
    authorization = verify_authorization(auth_path)
    legacy_auth_path = authorization["identities"][
        "legacy_authorization_manifest"
    ]["canonical_path"]
    legacy_authorization = load_legacy_authorization(legacy_auth_path)
    root = Path(output_root).resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"diagnostic output root exists: {root}")
    root.mkdir(parents=True)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    representations = {
        arm: load_frozen_representation(
            gate["identities"]["completion_receipts"][arm]["canonical_path"],
            device=device,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        for arm in ARMS
    }
    cohort_receipts: dict[str, dict[str, Any]] = {}
    diagnostic_receipts: dict[str, dict[str, Any]] = {}
    for cancer in cancers:
        patients, tiles, cohort_records, legacy_receipt = (
            legacy_prepare_cohort(
                cancer=cancer,
                cohort_contract=contract["cohorts"][cancer],
                authorized=legacy_authorization["cohorts"][cancer],
                destination=root / "legacy_cohort_adapter" / cancer,
                tile_view_receipt=legacy_authorization["tile_view_receipt"],
                expected_study_id=LEGACY_STUDY_ID,
                expected_scenario=LEGACY_SCENARIO,
                expected_amendment_identity=gate["identities"][
                    "estimand_amendment"
                ],
            )
        )
        current_cohort = _wrap_cohort(
            cancer=cancer,
            legacy_receipt_path=legacy_receipt,
            cohort_records=cohort_records,
            gate_receipt=gate_path,
            destination=root / "cohorts" / cancer / "COHORT_RECEIPT.json",
        )
        diagnostic = run_paired_diagnostic(
            representations=representations,
            tiles=tiles,
            patients=patients,
            task_id=contract["cohorts"][cancer]["task"],
            cohort_source=cohort_records,
            output_root=root / "diagnostics" / cancer,
            cache_dir=root / "cache",
        )
        cohort_receipts[cancer] = file_identity(current_cohort)
        diagnostic_receipts[cancer] = file_identity(diagnostic)

    receipt = build_receipt(
        schema=LOADER_ROOT_SCHEMA,
        study_id=STUDY_ID,
        scenario=gate["scenario"],
        identities={
            "deployment_gate": file_identity(gate_path),
            "authorization_manifest": file_identity(auth_path),
            "cohorts": cohort_receipts,
            "diagnostics": diagnostic_receipts,
            "loader": file_identity(Path(__file__)),
        },
        fields={
            "status": "complete",
            "representation_seed": gate["representation_seed"],
            "cancers": list(CANCERS),
            "arms": list(ARMS),
            "legacy_cohort_scenario": LEGACY_SCENARIO,
        },
    )
    result = atomic_write_receipt(root / "LOADER_ROOT_RECEIPT.json", receipt)
    verify_loader_result(result, gate_path)
    return result


def verify_loader_result(
    path: str | Path, gate_receipt: str | Path
) -> dict[str, Any]:
    gate_path = Path(gate_receipt).resolve()
    gate = verify_gate(gate_path)
    result = verify_receipt(
        path,
        expected_schema=LOADER_ROOT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=gate["scenario"],
    )
    identities = result.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "deployment_gate", "authorization_manifest", "cohorts",
        "diagnostics", "loader",
    }:
        raise ValueError("loader-root identity topology differs")
    if identities["deployment_gate"] != file_identity(gate_path):
        raise ValueError("loader root descends from a different gate")
    if identities["authorization_manifest"] != gate["identities"][
        "authorization_manifest"
    ]:
        raise ValueError("loader root authorization differs")
    if identities["loader"] != file_identity(Path(__file__)):
        raise ValueError("loader root source differs")
    if set(identities["cohorts"]) != set(CANCERS) or set(
        identities["diagnostics"]
    ) != set(CANCERS):
        raise ValueError("loader-root cancer topology differs")
    if (
        result.get("status") != "complete"
        or result.get("representation_seed") != gate["representation_seed"]
        or result.get("cancers") != list(CANCERS)
        or result.get("arms") != list(ARMS)
        or result.get("legacy_cohort_scenario") != LEGACY_SCENARIO
    ):
        raise ValueError("loader-root semantic fields differ")
    contract = load_contract(
        gate["identities"]["deployment_contract"]["canonical_path"]
    )
    for cancer in CANCERS:
        current_cohort_path = Path(
            identities["cohorts"][cancer]["canonical_path"]
        ).resolve()
        if file_identity(current_cohort_path) != identities["cohorts"][cancer]:
            raise ValueError(f"{cancer} current cohort identity drift")
        cohort = verify_receipt(
            current_cohort_path,
            expected_schema=COHORT_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        if set(cohort) != {
            "schema", "study_id", "scenario", "identities",
            "topology_sha256", "status", "representation_seed",
            "legacy_study_id", "legacy_scenario",
            *LEGACY_COHORT_SEMANTIC_FIELDS,
        }:
            raise ValueError(f"{cancer} current cohort field topology differs")
        cohort_ids = cohort.get("identities", {})
        if set(cohort_ids) != {
            "deployment_gate", "legacy_cohort_receipt",
            "legacy_source_bundle", "legacy_tile_ledger",
            "cohort_records", "loader",
        }:
            raise ValueError(f"{cancer} current cohort topology differs")
        if cohort_ids["deployment_gate"] != file_identity(gate_path):
            raise ValueError(f"{cancer} current cohort gate differs")
        legacy = verify_receipt(
            cohort_ids["legacy_cohort_receipt"]["canonical_path"],
            expected_schema=LEGACY_COHORT_SCHEMA,
            expected_study_id=LEGACY_STUDY_ID,
            expected_scenario=LEGACY_SCENARIO,
        )
        if cohort_ids["legacy_cohort_receipt"] != file_identity(
            cohort_ids["legacy_cohort_receipt"]["canonical_path"]
        ):
            raise ValueError(f"{cancer} legacy cohort receipt identity drift")
        if cohort_ids["cohort_records"] != file_identity(
            cohort_ids["cohort_records"]["canonical_path"]
        ):
            raise ValueError(f"{cancer} cohort records identity drift")
        if set(legacy) != {
            "schema", "study_id", "scenario", "identities",
            "topology_sha256", *LEGACY_COHORT_SEMANTIC_FIELDS,
        }:
            raise ValueError(f"{cancer} legacy cohort field topology differs")
        for key in LEGACY_COHORT_SEMANTIC_FIELDS:
            if cohort.get(key) != legacy.get(key):
                raise ValueError(
                    f"{cancer} current/legacy cohort {key} differs"
                )
        source_bundle = verify_receipt(
            cohort_ids["legacy_source_bundle"]["canonical_path"],
            expected_schema=LEGACY_SOURCE_SCHEMA,
            expected_study_id=LEGACY_STUDY_ID,
            expected_scenario=LEGACY_SCENARIO,
        )
        if source_bundle.get("identities", {}).get(
            "estimand_amendment"
        ) != gate["identities"]["estimand_amendment"]:
            raise ValueError(f"{cancer} estimand amendment ancestry differs")
        if (
            cohort_ids["legacy_source_bundle"]
            != gate["identities"]["legacy_cohorts"][cancer]["source_bundle"]
            or cohort_ids["legacy_tile_ledger"]
            != gate["identities"]["legacy_cohorts"][cancer]["tile_ledger"]
            or legacy["identities"]["cohort_records"]
            != cohort_ids["cohort_records"]
        ):
            raise ValueError(f"{cancer} legacy cohort ancestry differs")
        expected_n = contract["cohorts"][cancer][
            "expected_eligible_patients"
        ]
        if (
            cohort.get("representation_seed") != gate["representation_seed"]
            or cohort.get("cancer") != cancer
            or cohort.get("task") != contract["cohorts"][cancer]["task"]
            or cohort.get("patient_count") != expected_n
            or cohort.get("legacy_scenario") != LEGACY_SCENARIO
        ):
            raise ValueError(f"{cancer} current cohort semantic fields differ")
        diagnostic_path = Path(
            identities["diagnostics"][cancer]["canonical_path"]
        ).resolve()
        if file_identity(diagnostic_path) != identities["diagnostics"][cancer]:
            raise ValueError(f"{cancer} diagnostic root identity drift")
        diagnostic = verify_receipt(
            diagnostic_path,
            expected_schema=DIAGNOSTIC_ROOT_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=gate["scenario"],
        )
        diagnostic_ids = diagnostic.get("identities", {})
        if (
            diagnostic.get("status") != "complete"
            or diagnostic.get("task_id") != contract["cohorts"][cancer]["task"]
            or diagnostic.get("arms") != list(ARMS)
            or diagnostic.get("cell_count") != 12
            or diagnostic.get("race_usage") != "output_metadata_only"
            or diagnostic_ids.get("cohort_source")
            != cohort_ids["cohort_records"]
            or diagnostic_ids.get("completion_receipts")
            != gate["identities"]["completion_receipts"]
        ):
            raise ValueError(f"{cancer} diagnostic ancestry/semantics differ")
    return result
