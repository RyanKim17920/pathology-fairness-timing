"""Adopt the verified seed-32001 canary into the fixed-five study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

from tools.matched_cancer_fixed48_20260730.diag_authorization import (
    verify_authorization as verify_fixed48_authorization,
)
from tools.matched_cancer_fixed48_20260730.feasibility_gate import (
    verify as verify_fixed48_feasibility,
)
from tools.matched_cancer_fixed48_20260730.serial_controller import (
    SUCCESS_NAME,
    verify_seed_success,
)
from tools.matched_cancer_fixed48_20260730.source_manifest import (
    verify_manifest as verify_fixed48_manifest,
)
from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)


REPO = Path(__file__).resolve().parents[2]
STUDY_ID = "matched_cancer_fixed5_20260730"
SCHEMA = "matched-cancer-fixed5-adoption-authorization/v1"
SCENARIO = "fixed5_adoption_after_seed32001"
FIXED5_SEEDS = (32001, 32002, 32003, 32004, 32005)
CONTROLLER_SEEDS = (32002, 32003, 32004, 32005)
ATTEMPT_RE = re.compile(r"attempt_([0-9]{2,})")
PRODUCTION_ROOT = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730"
)
AMENDMENT_01 = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_SAMPLE_SIZE_AMENDMENT_01.md"
)
AMENDMENT_02 = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_SAMPLE_SIZE_AMENDMENT_02.md"
)


def _seed1_success_path(production_root: Path) -> Path:
    parent = production_root / "diagnostic/seed_32001"
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("seed-32001 diagnostic root is missing or invalid")
    successes: list[Path] = []
    for attempt in parent.iterdir():
        if (
            not attempt.is_dir()
            or attempt.is_symlink()
            or ATTEMPT_RE.fullmatch(attempt.name) is None
        ):
            raise ValueError(f"invalid seed-32001 attempt entry: {attempt}")
        candidate = attempt / SUCCESS_NAME
        if candidate.exists() or candidate.is_symlink():
            successes.append(candidate)
    if len(successes) != 1:
        raise ValueError(
            "seed 32001 must have exactly one success receipt; "
            f"observed {len(successes)}"
        )
    # Preserve the leaf path so verify_seed_success can reject a symlink
    # instead of receiving its already-resolved target.
    return successes[0].absolute()


def _verify_ancestors(
    *,
    fixed48_source_manifest: str | Path,
    authorization_manifest: str | Path,
    feasibility_gate: str | Path,
    production_root: str | Path,
) -> dict[str, Path]:
    root_candidate = Path(production_root)
    if root_candidate.is_symlink():
        raise ValueError("production root may not be a symlink")
    root = root_candidate.resolve(strict=True)
    canonical_root = PRODUCTION_ROOT.resolve(strict=True)
    if root != canonical_root:
        raise ValueError("production root differs from the fixed study root")

    def exact_file(raw: str | Path, *, label: str) -> Path:
        candidate = Path(raw)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"{label} must be a non-symlink regular file")
        return candidate.resolve(strict=True)

    manifest = exact_file(
        fixed48_source_manifest, label="fixed48 source manifest"
    )
    authorization = exact_file(
        authorization_manifest, label="fixed48 authorization"
    )
    feasibility = exact_file(
        feasibility_gate, label="fixed48 feasibility gate"
    )
    expected_manifest = root / "control/FIXED48_SOURCE_MANIFEST_V2.json"
    expected_authorization = (
        root / "authorization/AUTHORIZATION_MANIFEST_V3.json"
    )
    expected_feasibility = root / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
    if manifest != expected_manifest:
        raise ValueError("fixed48 source manifest path differs")
    if authorization != expected_authorization:
        raise ValueError("fixed48 authorization path differs")
    if feasibility != expected_feasibility:
        raise ValueError("fixed48 feasibility path differs")

    verify_fixed48_manifest(manifest)
    verify_fixed48_authorization(authorization)
    verify_fixed48_feasibility(
        feasibility, authorization_manifest=authorization
    )
    success = _seed1_success_path(root)
    verify_seed_success(
        success,
        seed=32001,
        source_manifest=manifest,
        production_root=root,
    )
    return {
        "production_root": root,
        "fixed48_source_manifest": manifest,
        "authorization_manifest": authorization,
        "feasibility_gate": feasibility,
        "seed1_success": success,
    }


def _publish_exclusive(path: Path, receipt: dict[str, Any]) -> Path:
    """Publish canonical JSON without ever replacing a concurrent writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def create(
    destination: str | Path,
    *,
    fixed48_source_manifest: str | Path,
    authorization_manifest: str | Path,
    feasibility_gate: str | Path,
    production_root: str | Path,
) -> Path:
    """Create an immutable bridge only after the canary fully succeeds."""
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"adoption authorization exists: {output}")
    ancestors = _verify_ancestors(
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        production_root=production_root,
    )
    expected_output = (
        ancestors["production_root"]
        / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
    )
    if output.resolve() != expected_output:
        raise ValueError("adoption authorization destination differs")
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities={
            "amendment_01": file_identity(AMENDMENT_01),
            "amendment_02": file_identity(AMENDMENT_02),
            "fixed48_source_manifest": file_identity(
                ancestors["fixed48_source_manifest"]
            ),
            "fixed48_authorization": file_identity(
                ancestors["authorization_manifest"]
            ),
            "fixed48_feasibility": file_identity(
                ancestors["feasibility_gate"]
            ),
            "seed1_success": file_identity(ancestors["seed1_success"]),
            "adoption_source": file_identity(Path(__file__)),
        },
        fields={
            "status": "authorized",
            "adopted_seed": 32001,
            "fixed5_seeds": list(FIXED5_SEEDS),
            "controller_seeds": list(CONTROLLER_SEEDS),
            "ancestor_controls_recomputed": False,
            "values_inspected": False,
        },
    )
    written = _publish_exclusive(output, receipt)
    verify(
        written,
        fixed48_source_manifest=ancestors["fixed48_source_manifest"],
        authorization_manifest=ancestors["authorization_manifest"],
        feasibility_gate=ancestors["feasibility_gate"],
        production_root=ancestors["production_root"],
    )
    return written


def verify(
    path: str | Path,
    *,
    fixed48_source_manifest: str | Path,
    authorization_manifest: str | Path,
    feasibility_gate: str | Path,
    production_root: str | Path,
) -> dict[str, Any]:
    """Reverify the bridge and every fixed-48 ancestor it names."""
    ancestors = _verify_ancestors(
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        production_root=production_root,
    )
    source = Path(path).resolve(strict=True)
    expected_source = (
        ancestors["production_root"]
        / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
    )
    if source != expected_source or Path(path).is_symlink():
        raise ValueError("adoption authorization path differs")
    receipt = verify_receipt(
        source,
        expected_schema=SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=SCENARIO,
    )
    expected_fields = {
        "status": "authorized",
        "adopted_seed": 32001,
        "fixed5_seeds": list(FIXED5_SEEDS),
        "controller_seeds": list(CONTROLLER_SEEDS),
        "ancestor_controls_recomputed": False,
        "values_inspected": False,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"adoption authorization {key} differs")
    identities = receipt.get("identities", {})
    expected_roles = {
        "amendment_01",
        "amendment_02",
        "fixed48_source_manifest",
        "fixed48_authorization",
        "fixed48_feasibility",
        "seed1_success",
        "adoption_source",
    }
    if set(identities) != expected_roles:
        raise ValueError("adoption authorization identity topology differs")
    expected_identities = {
        "amendment_01": file_identity(AMENDMENT_01),
        "amendment_02": file_identity(AMENDMENT_02),
        "fixed48_source_manifest": file_identity(
            ancestors["fixed48_source_manifest"]
        ),
        "fixed48_authorization": file_identity(
            ancestors["authorization_manifest"]
        ),
        "fixed48_feasibility": file_identity(
            ancestors["feasibility_gate"]
        ),
        "seed1_success": file_identity(ancestors["seed1_success"]),
        "adoption_source": file_identity(Path(__file__)),
    }
    if identities != expected_identities:
        raise ValueError("adoption authorization ancestry differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    verify_parser = sub.add_parser("verify")
    for command in (create_parser, verify_parser):
        command.add_argument("--adoption-authorization", required=True)
        command.add_argument("--fixed48-source-manifest", required=True)
        command.add_argument("--authorization-manifest", required=True)
        command.add_argument("--feasibility-gate", required=True)
        command.add_argument("--production-root", required=True)
    args = parser.parse_args(argv)
    common = {
        "fixed48_source_manifest": args.fixed48_source_manifest,
        "authorization_manifest": args.authorization_manifest,
        "feasibility_gate": args.feasibility_gate,
        "production_root": args.production_root,
    }
    if args.command == "create":
        create(args.adoption_authorization, **common)
    else:
        verify(args.adoption_authorization, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
