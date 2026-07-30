"""One current-seed diagnostic phase. It never submits jobs or analyzes scores."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Sequence

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)

from .diag_contract import (
    ARMS,
    CANCERS,
    HEAD_SEEDS,
    PHASE_SCHEMA,
    STUDY_ID,
    build_contract,
    scenario_for,
    validate_seed,
)
from .diag_deployment import preflight, resolve_loader, verify_gate
from .diag_exporter import collect_exports, export_cell, verify_collection
from .diag_loader import verify_loader_result
from .diag_structural_auditor import audit, verify_audit


def _validate_attempt(path: Path, *, seed: int, attempt_name: str) -> Path:
    if re.fullmatch(r"attempt_[0-9]{2,}", attempt_name) is None:
        raise ValueError("attempt name must match attempt_NN")
    if path.name != attempt_name or path.parent.name != f"seed_{seed}":
        raise ValueError("attempt path seed/name mismatch")
    if path.is_symlink():
        raise ValueError("attempt path may not be a symlink")
    return path.resolve()


def _write_contract(path: Path, value: dict[str, Any]) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"diagnostic contract exists: {path}")
    # atomic_write_receipt is a generic canonical exclusive atomic JSON writer.
    output = atomic_write_receipt(path, value)
    if output.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise ValueError("diagnostic contract canonical write failed")
    return output


def run(
    *,
    seed: int,
    attempt_name: str,
    calibration_attempt_root: str | Path,
    diagnostic_attempt_root: str | Path,
    authorization_manifest: str | Path,
) -> Path:
    seed = validate_seed(seed)
    calibration_root = _validate_attempt(
        Path(calibration_attempt_root),
        seed=seed,
        attempt_name=attempt_name,
    )
    diagnostic_root = _validate_attempt(
        Path(diagnostic_attempt_root),
        seed=seed,
        attempt_name=attempt_name,
    )
    if not calibration_root.is_dir():
        raise ValueError("calibration attempt root does not exist")
    if diagnostic_root.exists() or diagnostic_root.is_symlink():
        raise FileExistsError(
            f"diagnostic attempt root exists: {diagnostic_root}"
        )
    auth_path = Path(authorization_manifest).resolve(strict=True)
    diagnostic_root.mkdir(parents=True, exist_ok=False)
    contract_path = _write_contract(
        diagnostic_root / "DIAGNOSTIC_DEPLOYMENT_CONTRACT.json",
        build_contract(seed),
    )
    gate_path = preflight(
        contract_path=contract_path,
        calibration_root_receipt=(
            calibration_root / "ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
        ),
        calibration_audit_receipt=(
            calibration_root
            / "INDEPENDENT_CALIBRATION_AUDIT_RECEIPT.json"
        ),
        completion_receipts={
            arm: calibration_root / arm / "COMPLETION_RECEIPT.json"
            for arm in ARMS
        },
        authorization_manifest=auth_path,
        destination=diagnostic_root / "DEPLOYMENT_GATE_RECEIPT.json",
    )
    gate = verify_gate(gate_path)
    loader, loader_source = resolve_loader(gate["loader_entrypoint"])
    if file_identity(loader_source) != gate["identities"]["loader_source"]:
        raise ValueError("resolved loader source differs from deployment gate")
    loader_output_root = diagnostic_root / "run"
    loader_root = loader(
        contract=build_contract(seed),
        cancers=CANCERS,
        authorization_manifest=auth_path,
        output_root=loader_output_root,
        gate_receipt=gate_path,
    )
    loader_receipt = verify_loader_result(loader_root, gate_path)

    exports = []
    for cancer in CANCERS:
        diagnostic = verify_receipt(
            loader_receipt["identities"]["diagnostics"][cancer][
                "canonical_path"
            ],
            expected_schema="matched-cancer-adapter-diagnostic-root/v1",
            expected_study_id=STUDY_ID,
            expected_scenario=scenario_for(seed),
        )
        for arm in ARMS:
            for head in HEAD_SEEDS:
                cell_path = Path(
                    diagnostic["identities"]["cells"][arm][str(head)][
                        "canonical_path"
                    ]
                )
                cell = verify_receipt(
                    cell_path,
                    expected_schema="matched-cancer-adapter-diagnostic/v1",
                    expected_study_id=STUDY_ID,
                    expected_scenario=scenario_for(seed),
                )
                destination = (
                    diagnostic_root / "sealed_exports" / cancer / arm
                    / f"head_seed_{head}.jsonl"
                )
                exports.append(export_cell(
                    nested_predictions=cell["identities"]["predictions"][
                        "canonical_path"
                    ],
                    diagnostic_receipt=cell_path,
                    destination=destination,
                    fm_seed=seed,
                    arm=arm,
                    cancer=cancer,
                    head_seed=head,
                    deployment_gate_receipt=gate_path,
                    loader_root_receipt=loader_root,
                ))
    collection = collect_exports(
        exports,
        destination=(
            diagnostic_root / f"seed{seed}_predictions.jsonl"
        ),
        expected_fm_seed=seed,
        deployment_gate_receipt=gate_path,
        loader_root_receipt=loader_root,
    )
    structural_audit = audit(
        output_root=loader_output_root,
        deployment_gate_receipt=gate_path,
        loader_root_receipt=loader_root,
        collection=collection,
        expected_fm_seed=seed,
        destination=diagnostic_root / "STRUCTURAL_AUDIT_RECEIPT.json",
    )
    receipt = build_receipt(
        schema=PHASE_SCHEMA,
        study_id=STUDY_ID,
        scenario=scenario_for(seed),
        identities={
            "deployment_gate": file_identity(gate_path),
            "loader_root": file_identity(loader_root),
            "structural_audit": file_identity(structural_audit),
            "collection": file_identity(collection),
            "collection_receipt": file_identity(
                collection.with_suffix(
                    collection.suffix + ".receipt.json"
                )
            ),
            "worker": file_identity(Path(__file__)),
        },
        fields={
            "status": "complete",
            "fm_seed": seed,
            "values_inspected": False,
            "analysis_performed": False,
        },
    )
    destination = atomic_write_receipt(
        diagnostic_root / "DIAGNOSTIC_PHASE_RECEIPT.json", receipt
    )
    verify_phase(destination, expected_fm_seed=seed)
    return destination


def verify_phase(
    path: str | Path, *, expected_fm_seed: int
) -> dict[str, Any]:
    seed = validate_seed(expected_fm_seed)
    receipt = verify_receipt(
        path,
        expected_schema=PHASE_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario_for(seed),
    )
    if (
        receipt.get("status") != "complete"
        or receipt.get("fm_seed") != seed
        or receipt.get("values_inspected") is not False
        or receipt.get("analysis_performed") is not False
    ):
        raise ValueError("diagnostic phase semantic contract differs")
    identities = receipt.get("identities", {})
    if set(identities) != {
        "deployment_gate", "loader_root", "structural_audit",
        "collection", "collection_receipt", "worker",
    }:
        raise ValueError("diagnostic phase identity topology differs")
    gate_path = identities["deployment_gate"]["canonical_path"]
    loader_path = identities["loader_root"]["canonical_path"]
    collection_path = Path(
        identities["collection"]["canonical_path"]
    ).resolve()
    collection_receipt_path = collection_path.with_suffix(
        collection_path.suffix + ".receipt.json"
    )
    audit_path = Path(
        identities["structural_audit"]["canonical_path"]
    ).resolve()
    for role, source in (
        ("deployment_gate", Path(gate_path)),
        ("loader_root", Path(loader_path)),
        ("collection", collection_path),
        ("collection_receipt", collection_receipt_path),
        ("structural_audit", audit_path),
    ):
        if identities[role] != file_identity(source):
            raise ValueError(f"diagnostic phase {role} identity differs")
    gate = verify_gate(gate_path)
    verify_loader_result(loader_path, gate_path)
    collection_receipt = verify_collection(
        collection_path,
        expected_fm_seed=seed,
        deployment_gate_receipt=gate_path,
        loader_root_receipt=loader_path,
    )
    structural_audit = verify_audit(
        audit_path,
        expected_fm_seed=seed,
    )
    if (
        structural_audit["identities"].get("deployment_gate")
        != identities["deployment_gate"]
        or structural_audit["identities"].get("loader_root")
        != identities["loader_root"]
        or structural_audit["identities"].get("collection")
        != identities["collection"]
        or structural_audit["identities"].get("collection_receipt")
        != identities["collection_receipt"]
        or collection_receipt["identities"].get(
            "deployment_gate_receipt"
        ) != identities["deployment_gate"]
        or collection_receipt["identities"].get(
            "loader_root_receipt"
        ) != identities["loader_root"]
    ):
        raise ValueError("diagnostic phase descendant ancestry differs")
    if identities["worker"] != file_identity(Path(__file__)):
        raise ValueError("diagnostic phase worker source differs")
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("seed", type=int)
    result.add_argument("attempt_name")
    result.add_argument("--calibration-attempt-root", type=Path, required=True)
    result.add_argument("--diagnostic-attempt-root", type=Path, required=True)
    result.add_argument("--authorization-manifest", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run(
        seed=args.seed,
        attempt_name=args.attempt_name,
        calibration_attempt_root=args.calibration_attempt_root,
        diagnostic_attempt_root=args.diagnostic_attempt_root,
        authorization_manifest=args.authorization_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
