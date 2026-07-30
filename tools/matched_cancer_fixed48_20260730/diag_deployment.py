"""Current-seed deployment gate over current calibration and legacy cohorts."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Mapping

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from .diag_authorization import verify_authorization
from .diag_contract import (
    ARMS,
    CALIBRATION_ROOT_SCHEMA,
    COHORT_SIZES,
COMPLETION_SCHEMA,
    GATE_SCHEMA,
    HEAD_SEEDS,
    STUDY_ID,
    load_contract,
    scenario_for,
)

CALIBRATION_AUDIT_SCHEMA = (
    "matched-cancer-fixed48-calibration-independent-audit/v1"
)

REPO = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE_PATHS = {
    "fixed48_package_init": Path(__file__).with_name("__init__.py"),
    "legacy_diagnostic_package_init": REPO
    / "tools/matched_cancer_diagnostic_20260730/__init__.py",
    "legacy_stage_package_init": REPO
    / "tools/matched_cancer_stage_20260730/__init__.py",
    "legacy_union_package_init": REPO
    / "tools/matched_stage_union_20260730/__init__.py",
    "diag_contract": Path(__file__).with_name("diag_contract.py"),
    "diag_authorization": Path(__file__).with_name("diag_authorization.py"),
    "diag_deployment": Path(__file__),
    "diag_loader": Path(__file__).with_name("diag_loader.py"),
    "diag_exporter": Path(__file__).with_name("diag_exporter.py"),
    "diag_structural_auditor": Path(__file__).with_name(
        "diag_structural_auditor.py"
    ),
    "diag_worker": Path(__file__).with_name("diag_worker.py"),
    "legacy_deployment": REPO
    / "tools/matched_cancer_diagnostic_20260730/deployment.py",
    "legacy_integration": REPO
    / "tools/matched_cancer_diagnostic_20260730/integration.py",
    "legacy_vetted_loader": REPO
    / "tools/matched_cancer_diagnostic_20260730/vetted_loader.py",
    "legacy_runner": REPO
    / "tools/matched_cancer_diagnostic_20260730/runner.py",
    "legacy_cache": REPO
    / "tools/matched_cancer_diagnostic_20260730/cache.py",
    "legacy_exporter": REPO
    / "tools/matched_cancer_diagnostic_20260730/exporter.py",
    "stage_objectives": REPO
    / "tools/matched_cancer_stage_20260730/objectives.py",
    "union_objectives": REPO
    / "tools/matched_stage_union_20260730/objectives.py",
    "receipts": REPO / "tools/matched_cancer_stage_20260730/receipts.py",
    "completion_receipt": REPO
    / "tools/matched_cancer_stage_20260730/completion_receipt.py",
    "reliable_fairness_head": REPO / "tools/reliable_fairness_head.py",
    "fairness_eval": REPO / "tools/fairness_eval.py",
    "post_hoc_debias": REPO / "tools/post_hoc_debias.py",
    "hf_tiles": REPO / "tools/hf_tiles.py",
    "encoder_model": REPO / "vendor/matched_stage_train_20260730/model.py",
}


def verify_calibration(
    root_path: str | Path,
    completion_receipts: Mapping[str, str | Path],
    *,
    seed: int,
) -> dict[str, Any]:
    """Verify the minimal stable interface exported by fixed48 calibration."""
    scenario = scenario_for(seed)
    root_source = Path(root_path).resolve()
    root = verify_receipt(
        root_source,
        expected_schema=CALIBRATION_ROOT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario,
    )
    if root.get("representation_seed", root.get("seed")) != seed:
        raise ValueError("calibration root seed differs")
    if root.get("status") != "fixed48_two_slot_calibration_complete":
        raise ValueError("calibration root is not complete")
    if (
        root.get("steps_per_run") != 781
        or root.get("presentations_per_run") != 99_968
        or root.get("total_presentations") != 499_840
        or root.get("run_names")
        != ["slot1_plain", "slot1_fair", "B", "H", "P"]
        or root.get("legacy_seed32001_disposition") != {
            "disposition": "systems_only_excluded_from_inference",
            "reusable": False,
            "rerun_in_fixed48_namespace": True,
        }
    ):
        raise ValueError("calibration root fixed execution contract differs")
    root_runs = root.get("identities", {}).get("runs")
    if not isinstance(root_runs, Mapping) or not set(ARMS).issubset(root_runs):
        raise ValueError("calibration root lacks B/P/H run ancestry")
    if set(completion_receipts) != set(ARMS):
        raise ValueError("calibration completions must be exactly B/P/H")
    completions: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = Path(completion_receipts[arm]).resolve()
        if file_identity(path) != root_runs[arm]:
            raise ValueError(f"{arm} completion is not calibration-root-bound")
        receipt = verify_receipt(
            path,
            expected_schema=COMPLETION_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=scenario,
        )
        if (
            receipt.get("status") != "complete"
            or receipt.get("representation_seed", seed) != seed
            or receipt.get("mode") != "adapter_only"
            or receipt.get("fair_weight") != {"B": 0.0, "P": 0.0, "H": 0.1}[arm]
        ):
            raise ValueError(f"{arm} completion semantic contract differs")
        completions[arm] = receipt
        paths[arm] = path
    return {"root": root, "root_path": root_source, "receipts": completions,
            "paths": paths}


def verify_calibration_audit(
    path: str | Path,
    calibration: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    audit_path = Path(path).resolve()
    audit = verify_receipt(
        audit_path,
        expected_schema=CALIBRATION_AUDIT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario_for(seed),
    )
    if (
        audit.get("status")
        != "fixed48_calibration_independent_audit_pass"
        or audit.get("representation_seed") != seed
        or audit.get("values_or_outcomes_accessed") is not False
    ):
        raise ValueError("independent calibration audit did not pass")
    if audit.get("identities", {}).get(
        "root_completion_receipt"
    ) != file_identity(calibration["root_path"]):
        raise ValueError("independent calibration audit root ancestry differs")
    audit_runs = audit.get("identities", {}).get("runs")
    if not isinstance(audit_runs, Mapping):
        raise ValueError("independent calibration audit lacks run ancestry")
    for arm in ARMS:
        if audit_runs.get(arm, {}).get(
            "completion_receipt"
        ) != file_identity(calibration["paths"][arm]):
            raise ValueError(
                f"independent calibration audit {arm} ancestry differs"
            )
    return audit


def preflight(
    *,
    contract_path: str | Path,
    calibration_root_receipt: str | Path,
    calibration_audit_receipt: str | Path,
    completion_receipts: Mapping[str, str | Path],
    authorization_manifest: str | Path,
    destination: str | Path,
) -> Path:
    contract_source = Path(contract_path).resolve()
    contract = load_contract(contract_source)
    seed = contract["representation_seed"]
    calibration = verify_calibration(
        calibration_root_receipt, completion_receipts, seed=seed
    )
    calibration_audit_path = Path(calibration_audit_receipt).resolve()
    verify_calibration_audit(
        calibration_audit_path, calibration, seed=seed
    )
    authorization_path = Path(authorization_manifest).resolve()
    authorization = verify_authorization(authorization_path)
    requested = Path(destination)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"fixed48 deployment gate exists: {requested}")
    receipt = build_receipt(
        schema=GATE_SCHEMA,
        study_id=STUDY_ID,
        scenario=scenario_for(seed),
        identities={
            "deployment_contract": file_identity(contract_source),
            "calibration_root_receipt": file_identity(
                calibration["root_path"]
            ),
            "calibration_audit_receipt": file_identity(
                calibration_audit_path
            ),
            "completion_receipts": {
                arm: file_identity(calibration["paths"][arm]) for arm in ARMS
            },
            "authorization_manifest": file_identity(authorization_path),
            "legacy_authorization_manifest": authorization["identities"][
                "legacy_authorization_manifest"
            ],
            "legacy_tile_view_receipt": authorization["identities"][
                "legacy_tile_view_receipt"
            ],
            "legacy_cohorts": authorization["identities"]["legacy_cohorts"],
            "loader_source": authorization["identities"]["loader_source"],
            "estimand_amendment": authorization["identities"][
                "estimand_amendment"
            ],
            "sources": {
                name: file_identity(path)
                for name, path in RUNTIME_SOURCE_PATHS.items()
            },
        },
        fields={
            "status": "valid",
            "representation_seed": seed,
            "outcomes_opened": False,
            "legacy_cohort_scenario": authorization["legacy_scenario"],
            "cohort_sizes": dict(COHORT_SIZES),
            "head_seeds": list(HEAD_SEEDS),
            "loader_entrypoint": authorization["loader_entrypoint"],
        },
    )
    output = atomic_write_receipt(destination, receipt)
    verify_gate(output)
    return output


def verify_gate(path: str | Path) -> dict[str, Any]:
    receipt = verify_receipt(
        path, expected_schema=GATE_SCHEMA, expected_study_id=STUDY_ID
    )
    seed = receipt.get("representation_seed")
    scenario = scenario_for(seed)
    if receipt.get("scenario") != scenario:
        raise ValueError("gate seed/scenario mismatch")
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {
        "deployment_contract", "calibration_root_receipt",
        "calibration_audit_receipt", "completion_receipts",
        "authorization_manifest",
        "legacy_authorization_manifest", "legacy_tile_view_receipt",
        "legacy_cohorts", "loader_source", "sources",
        "estimand_amendment",
    }:
        raise ValueError("gate identity topology differs")
    if set(identities["completion_receipts"]) != set(ARMS):
        raise ValueError("gate completion-receipt topology differs")
    for role in (
        "deployment_contract", "calibration_root_receipt",
        "calibration_audit_receipt", "authorization_manifest",
        "legacy_authorization_manifest", "legacy_tile_view_receipt",
        "loader_source", "estimand_amendment",
    ):
        if identities[role] != file_identity(
            identities[role]["canonical_path"]
        ):
            raise ValueError(f"gate stored identity {role} differs")
    for arm in ARMS:
        identity = identities["completion_receipts"][arm]
        if identity != file_identity(identity["canonical_path"]):
            raise ValueError(f"gate stored completion {arm} differs")
    if set(identities["sources"]) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("gate runtime source topology differs")
    for name, source in RUNTIME_SOURCE_PATHS.items():
        if identities["sources"][name] != file_identity(source):
            raise ValueError(f"gate runtime source {name} was tampered")
    contract = load_contract(
        identities["deployment_contract"]["canonical_path"]
    )
    if contract["representation_seed"] != seed:
        raise ValueError("gate contract seed differs")
    authorization = verify_authorization(
        identities["authorization_manifest"]["canonical_path"]
    )
    if file_identity(
        identities["authorization_manifest"]["canonical_path"]
    ) != identities["authorization_manifest"]:
        raise ValueError("gate authorization identity drift")
    for key in (
        "legacy_authorization_manifest", "legacy_tile_view_receipt",
        "legacy_cohorts", "loader_source", "estimand_amendment",
    ):
        if identities[key] != authorization["identities"][key]:
            raise ValueError(f"gate authorization ancestor {key} differs")
    verify_calibration(
        identities["calibration_root_receipt"]["canonical_path"],
        {
            arm: identities["completion_receipts"][arm]["canonical_path"]
            for arm in ARMS
        },
        seed=seed,
    )
    calibration_audit_path = identities[
        "calibration_audit_receipt"
    ]["canonical_path"]
    calibration = {
        "root_path": Path(
            identities["calibration_root_receipt"]["canonical_path"]
        ),
        "paths": {
            arm: Path(
                identities["completion_receipts"][arm]["canonical_path"]
            )
            for arm in ARMS
        },
    }
    verify_calibration_audit(
        calibration_audit_path,
        calibration,
        seed=seed,
    )
    expected = {
        "status": "valid",
        "outcomes_opened": False,
        "legacy_cohort_scenario": authorization["legacy_scenario"],
        "cohort_sizes": dict(COHORT_SIZES),
        "head_seeds": list(HEAD_SEEDS),
        "loader_entrypoint": authorization["loader_entrypoint"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"gate {key} differs")
    return receipt


def resolve_loader(entrypoint: str):
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator:
        raise ValueError("loader entrypoint must be module:function")
    loader = getattr(importlib.import_module(module_name), function_name)
    source = Path(inspect.getsourcefile(loader) or "").resolve()
    return loader, source
