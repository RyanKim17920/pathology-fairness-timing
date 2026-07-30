#!/usr/bin/env python3
"""Ordered calibration-to-diagnostic worker for one fixed-48 seed.

The worker runs inside the controller's existing one-GPU allocation.  It never
submits a job and never invokes the analyzer or final verifier.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

from . import diag_worker
from .diag_authorization import verify_authorization
from .diag_contract import STUDY_ID, scenario_for, validate_seed
from .feasibility_gate import verify as verify_feasibility
from .source_manifest import REPO, verify_manifest


SUCCESS_SCHEMA = "matched-cancer-fixed48-seed-success/v1"
CALIBRATION_ROOT_SCHEMA = (
    "matched-cancer-fixed48-calibration-root-completion/v1"
)
CALIBRATION_AUDIT_SCHEMA = (
    "matched-cancer-fixed48-calibration-independent-audit/v1"
)
COMPLETION_SCHEMA = "matched-cancer-stage-completion/v1"
PRODUCTION_ROOT = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730"
)
CALIBRATION_DRIVER = (
    REPO
    / "tools/matched_cancer_fixed48_20260730/"
    "calibration_one_seed.sbatch"
)
EXECUTION_PROTOCOL = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/"
    "FIXED48_EXECUTION_PROTOCOL.md"
)
ATTEMPT_RE = re.compile(r"attempt_([0-9]{2,})")


def _attempt_paths(
    seed: int, attempt_name: str, *, production_root: Path
) -> tuple[Path, Path, str]:
    match = ATTEMPT_RE.fullmatch(attempt_name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError("attempt name must be attempt_NN with NN >= 01")
    root = production_root
    if root.is_symlink():
        raise ValueError("production root may not be a symlink")
    root = root.resolve()
    calibration = (
        root / "calibration" / f"seed_{seed}" / attempt_name
    )
    diagnostic = (
        root / "diagnostic" / f"seed_{seed}" / attempt_name
    )
    if calibration.exists() or calibration.is_symlink():
        raise FileExistsError(f"calibration attempt exists: {calibration}")
    if diagnostic.exists() or diagnostic.is_symlink():
        raise FileExistsError(f"diagnostic attempt exists: {diagnostic}")
    return calibration, diagnostic, match.group(1)


def _verify_calibration(
    calibration_root: Path, *, seed: int
) -> dict[str, Path]:
    scenario = scenario_for(seed)
    root_path = (
        calibration_root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
    )
    audit_path = (
        calibration_root / "INDEPENDENT_CALIBRATION_AUDIT_RECEIPT.json"
    )
    root = verify_receipt(
        root_path,
        expected_schema=CALIBRATION_ROOT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario,
    )
    audit = verify_receipt(
        audit_path,
        expected_schema=CALIBRATION_AUDIT_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario,
    )
    if (
        root.get("status") != "fixed48_two_slot_calibration_complete"
        or root.get("representation_seed") != seed
        or root.get("steps_per_run") != 781
        or root.get("presentations_per_run") != 99_968
        or root.get("total_presentations") != 499_840
    ):
        raise ValueError("calibration root semantic fields differ")
    if (
        audit.get("status")
        != "fixed48_calibration_independent_audit_pass"
        or audit.get("representation_seed") != seed
        or audit.get("values_or_outcomes_accessed") is not False
    ):
        raise ValueError("calibration independent audit semantic fields differ")
    audit_ids = audit.get("identities", {})
    if set(audit_ids) != {
        "auditor_source",
        "root_completion_receipt",
        "contract_receipt",
        "replay_manifest",
        "runs",
    }:
        raise ValueError("calibration audit identity topology differs")
    if audit_ids["root_completion_receipt"] != file_identity(root_path):
        raise ValueError("calibration audit does not bind the root receipt")
    if set(audit_ids["runs"]) != {
        "slot1_plain", "slot1_fair", "B", "H", "P"
    }:
        raise ValueError("calibration audit run topology differs")
    paths: dict[str, Path] = {
        "root": root_path,
        "audit": audit_path,
    }
    root_runs = root.get("identities", {}).get("runs", {})
    if set(root_runs) != {"slot1_plain", "slot1_fair", "B", "H", "P"}:
        raise ValueError("calibration root run topology differs")
    for arm in ("B", "P", "H"):
        completion_path = calibration_root / arm / "COMPLETION_RECEIPT.json"
        completion = verify_receipt(
            completion_path,
            expected_schema=COMPLETION_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=scenario,
        )
        if (
            completion.get("status") != "complete"
            or completion.get("steps_completed") != 781
            or completion.get("tile_presentations") != 99_968
            or root_runs.get(arm) != file_identity(completion_path)
            or audit_ids["runs"][arm].get("completion_receipt")
            != file_identity(completion_path)
        ):
            raise ValueError(f"calibration completion {arm} differs")
        paths[arm] = completion_path
    return paths


def _seal_success(
    *,
    seed: int,
    calibration_paths: Mapping[str, Path],
    diagnostic_phase_path: Path,
    source_manifest: Path,
    feasibility_gate: Path,
    production_root: Path,
) -> Path:
    phase = diag_worker.verify_phase(
        diagnostic_phase_path, expected_fm_seed=seed
    )
    phase_ids = phase["identities"]
    destination = diagnostic_phase_path.parent / "SEED_SUCCESS_RECEIPT.json"
    receipt = build_receipt(
        schema=SUCCESS_SCHEMA,
        study_id=STUDY_ID,
        scenario=scenario_for(seed),
        identities={
            "source_manifest": file_identity(source_manifest),
            "feasibility_gate": file_identity(feasibility_gate),
            "execution_protocol": file_identity(EXECUTION_PROTOCOL),
            "calibration_root": file_identity(calibration_paths["root"]),
            "calibration_audit": file_identity(calibration_paths["audit"]),
            "calibration_completions": {
                arm: file_identity(calibration_paths[arm])
                for arm in ("B", "P", "H")
            },
            "diagnostic_gate": dict(phase_ids["deployment_gate"]),
            "loader_root": dict(phase_ids["loader_root"]),
            "diagnostic_phase": file_identity(diagnostic_phase_path),
            "diagnostic_structural_audit": dict(
                phase_ids["structural_audit"]
            ),
            "per_seed_collection": dict(phase_ids["collection"]),
            "per_seed_collection_receipt": dict(
                phase_ids["collection_receipt"]
            ),
        },
        fields={
            "fm_seed": seed,
            "status": "complete",
            "legacy_outputs_used": False,
            "values_inspected": False,
        },
    )
    output = atomic_write_receipt(destination, receipt)
    # Import lazily so controller construction never imports a production
    # worker merely to inspect state.
    from .serial_controller import verify_seed_success

    verify_seed_success(
        output,
        seed=seed,
        source_manifest=source_manifest,
        production_root=production_root,
    )
    return output


def run(
    *,
    seed: int,
    attempt_name: str,
    source_manifest: str | Path,
    authorization_manifest: str | Path,
    feasibility_gate: str | Path,
    production_root: Path = PRODUCTION_ROOT,
    calibration_runner: Callable[..., Any] = subprocess.run,
    diagnostic_runner: Callable[..., Path] = diag_worker.run,
) -> Path:
    seed = validate_seed(seed)
    manifest_path = Path(source_manifest).resolve(strict=True)
    authorization_path = Path(authorization_manifest).resolve(strict=True)
    feasibility_path = Path(feasibility_gate).resolve(strict=True)
    if feasibility_path != (
        production_root.resolve()
        / "control/FEASIBILITY_GATE_RECEIPT.json"
    ):
        raise ValueError("feasibility gate path differs from production contract")
    manifest_identity = file_identity(manifest_path)
    authorization_identity = file_identity(authorization_path)
    feasibility_identity = file_identity(feasibility_path)
    manifest = verify_manifest(manifest_path)
    verify_authorization(authorization_path)
    verify_feasibility(
        feasibility_path, authorization_manifest=authorization_path
    )
    worker_identity = file_identity(Path(__file__))
    if worker_identity not in manifest["identities"]["sources"].values():
        raise ValueError("seed worker is not bound by the source manifest")
    calibration_root, diagnostic_root, attempt_number = _attempt_paths(
        seed, attempt_name, production_root=production_root
    )

    environment = dict(os.environ)
    calibration_runner(
        [
            "/usr/bin/bash",
            str(CALIBRATION_DRIVER),
            str(seed),
            attempt_number,
        ],
        cwd=REPO,
        env=environment,
        check=True,
    )
    if file_identity(manifest_path) != manifest_identity:
        raise ValueError("source manifest drifted during calibration")
    if file_identity(authorization_path) != authorization_identity:
        raise ValueError("authorization drifted during calibration")
    if file_identity(feasibility_path) != feasibility_identity:
        raise ValueError("feasibility gate drifted during calibration")
    verify_manifest(manifest_path)
    calibration_paths = _verify_calibration(calibration_root, seed=seed)

    phase_path = diagnostic_runner(
        seed=seed,
        attempt_name=attempt_name,
        calibration_attempt_root=calibration_root,
        diagnostic_attempt_root=diagnostic_root,
        authorization_manifest=authorization_path,
    )
    if Path(phase_path).resolve() != (
        diagnostic_root / "DIAGNOSTIC_PHASE_RECEIPT.json"
    ).resolve():
        raise ValueError("diagnostic worker returned an unexpected phase path")
    if file_identity(manifest_path) != manifest_identity:
        raise ValueError("source manifest drifted during diagnostic")
    if file_identity(authorization_path) != authorization_identity:
        raise ValueError("authorization drifted during diagnostic")
    if file_identity(feasibility_path) != feasibility_identity:
        raise ValueError("feasibility gate drifted during diagnostic")
    verify_manifest(manifest_path)
    verify_authorization(authorization_path)
    verify_feasibility(
        feasibility_path, authorization_manifest=authorization_path
    )
    return _seal_success(
        seed=seed,
        calibration_paths=calibration_paths,
        diagnostic_phase_path=Path(phase_path),
        source_manifest=manifest_path,
        feasibility_gate=feasibility_path,
        production_root=production_root,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("seed", type=int)
    result.add_argument("attempt_name")
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--authorization-manifest", type=Path, required=True)
    result.add_argument("--feasibility-gate", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(
        seed=args.seed,
        attempt_name=args.attempt_name,
        source_manifest=args.source_manifest,
        authorization_manifest=args.authorization_manifest,
        feasibility_gate=args.feasibility_gate,
    )
    print(str(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
