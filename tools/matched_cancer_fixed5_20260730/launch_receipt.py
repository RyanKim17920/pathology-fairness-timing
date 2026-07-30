#!/usr/bin/env python3
"""Immutable prelaunch and allocation receipts for fixed-five execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)


REPO = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730"
)
STUDY_ID = "matched_cancer_fixed5_20260730"
PRELAUNCH_SCHEMA = "matched-cancer-fixed5-prelaunch-queue/v1"
PRELAUNCH_SCENARIO = "fixed5_prelaunch_queue"
LAUNCH_SCHEMA = "matched-cancer-fixed5-launch/v1"
LAUNCH_SCENARIO = "fixed5_allocation_launch"
LEGACY_COMMENT = "matched_cancer_fixed48_20260730"
SUPERSEDED_COMMENT = "matched_cancer_fixed5_20260730"
JOB_NAME = "main_1gpu"
NONCE_RE = re.compile(r"[0-9a-f]{32,}")
JOB_ID_RE = re.compile(r"[0-9]+")

PACKAGE = Path(__file__).resolve().parent
SAFE_SUBMIT = PACKAGE / "safe_submit.sh"
SLURM_DRIVER = PACKAGE / "serial_fixed5.sbatch"
CONTROLLER = PACKAGE / "serial_controller.py"
AMENDMENTS = tuple(
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution"
    / (
        f"FIXED5_SAMPLE_SIZE_AMENDMENT_{number:02d}.md"
        if number <= 2
        else f"FIXED5_EXECUTION_AMENDMENT_{number:02d}.md"
    )
    for number in range(1, 7)
)

ManifestVerifier = Callable[[Path], Mapping[str, Any]]
AdoptionVerifier = Callable[..., Mapping[str, Any]]


def new_nonce() -> str:
    """Return exactly 128 bits encoded as lowercase hexadecimal."""
    return secrets.token_hex(16)


def validate_nonce(nonce: str) -> str:
    if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
        raise ValueError("launch nonce must be lowercase hexadecimal with >=128 bits")
    return nonce


def _exact_root(production_root: Path | str) -> Path:
    candidate = Path(production_root)
    expected_candidate = PRODUCTION_ROOT
    if candidate.is_symlink() or expected_candidate.is_symlink():
        raise ValueError("production root may not be a symlink")
    expected = expected_candidate.resolve(strict=True)
    if candidate.absolute() != expected:
        raise ValueError("production root path differs")
    root = candidate.resolve(strict=True)
    if root != expected:
        raise ValueError("production root canonical path differs")
    return root


def _require_exact_file(
    path: Path | str, expected: Path, label: str
) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a non-symlink file")
    if candidate.absolute() != expected:
        raise ValueError(f"{label} path differs")
    resolved = candidate.resolve(strict=True)
    if resolved != expected:
        raise ValueError(f"{label} canonical path differs")
    return resolved


def _ensure_directory(root: Path, directory: Path) -> Path:
    if directory.absolute() != directory:
        raise ValueError("directory must be absolute")
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise ValueError("directory escaped production root") from error
    current = root
    for part in relative.parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"directory is not canonical: {current}")
        if current.resolve(strict=True) != current:
            raise ValueError(f"directory ancestry redirected: {current}")
    return directory


def publish_exclusive(
    destination: Path,
    receipt: Mapping[str, Any],
    *,
    production_root: Path,
) -> Path:
    """Publish canonical JSON without any overwrite race."""
    root = _exact_root(production_root)
    if destination.absolute() != destination:
        raise ValueError("receipt destination must be absolute")
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError("receipt destination escaped production root") from error
    _ensure_directory(root, destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {destination}")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(destination.parent, directory_flags)
    opened_directory = os.fstat(directory_fd)
    payload = canonical_json_bytes(receipt) + b"\n"
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(
                temporary,
                destination.name,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_fd)
    current_directory = destination.parent.stat()
    if (
        current_directory.st_dev != opened_directory.st_dev
        or current_directory.st_ino != opened_directory.st_ino
        or destination.parent.is_symlink()
        or destination.parent.resolve(strict=True) != destination.parent
    ):
        raise ValueError("receipt directory identity drifted during publication")
    mode = destination.lstat().st_mode
    if (
        not stat.S_ISREG(mode)
        or destination.is_symlink()
        or destination.resolve(strict=True) != destination
    ):
        raise ValueError("published receipt is not a regular file")
    return destination.resolve(strict=True)


def _default_manifest_verifier(path: Path) -> Mapping[str, Any]:
    from .source_manifest import verify_manifest

    return verify_manifest(path)


def _default_adoption_verifier(
    path: Path,
    *,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    production_root: Path,
) -> Mapping[str, Any]:
    from .adoption_authorization import verify

    return verify(
        path,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        production_root=production_root,
    )


def _inputs(
    *,
    production_root: Path | str,
    fixed5_source_manifest: Path | str,
    adoption_authorization: Path | str,
    fixed48_source_manifest: Path | str,
    authorization_manifest: Path | str,
    feasibility_gate: Path | str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    root = _exact_root(production_root)
    fixed5 = _require_exact_file(
        fixed5_source_manifest,
        root / "control/FIXED5_SOURCE_MANIFEST_V1.json",
        "fixed-five source manifest",
    )
    adoption = _require_exact_file(
        adoption_authorization,
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "adoption authorization",
    )
    fixed48 = _require_exact_file(
        fixed48_source_manifest,
        root / "control/FIXED48_SOURCE_MANIFEST_V2.json",
        "fixed-48 source manifest",
    )
    authorization = _require_exact_file(
        authorization_manifest,
        root / "authorization/AUTHORIZATION_MANIFEST_V3.json",
        "fixed-48 authorization",
    )
    feasibility = _require_exact_file(
        feasibility_gate,
        root / "control/FEASIBILITY_GATE_RECEIPT_V2.json",
        "fixed-48 feasibility",
    )
    return root, fixed5, adoption, fixed48, authorization, feasibility


def _verify_controls(
    *,
    root: Path,
    fixed5: Path,
    adoption: Path,
    fixed48: Path,
    authorization: Path,
    feasibility: Path,
    manifest_verifier: ManifestVerifier,
    adoption_verifier: AdoptionVerifier,
) -> None:
    manifest_verifier(fixed5)
    adoption_verifier(
        adoption,
        fixed48_source_manifest=fixed48,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
        production_root=root,
    )


def _amendment_identities(last: int) -> dict[str, dict[str, Any]]:
    return {
        f"amendment_{index:02d}": file_identity(AMENDMENTS[index - 1])
        for index in range(1, last + 1)
    }


def create_prelaunch(
    destination: Path | str,
    *,
    launch_nonce: str,
    production_root: Path | str,
    fixed5_source_manifest: Path | str,
    adoption_authorization: Path | str,
    fixed48_source_manifest: Path | str,
    authorization_manifest: Path | str,
    feasibility_gate: Path | str,
    manifest_verifier: ManifestVerifier = _default_manifest_verifier,
    adoption_verifier: AdoptionVerifier = _default_adoption_verifier,
) -> Path:
    nonce = validate_nonce(launch_nonce)
    root, fixed5, adoption, fixed48, authorization, feasibility = _inputs(
        production_root=production_root,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
    )
    _verify_controls(
        root=root,
        fixed5=fixed5,
        adoption=adoption,
        fixed48=fixed48,
        authorization=authorization,
        feasibility=feasibility,
        manifest_verifier=manifest_verifier,
        adoption_verifier=adoption_verifier,
    )
    expected = root / f"control/prelaunch/FIXED5_PRELAUNCH_{nonce}.json"
    if Path(destination).absolute() != expected:
        raise ValueError("prelaunch destination differs")
    receipt = build_receipt(
        schema=PRELAUNCH_SCHEMA,
        study_id=STUDY_ID,
        scenario=PRELAUNCH_SCENARIO,
        identities={
            "fixed5_source_manifest": file_identity(fixed5),
            "adoption_authorization": file_identity(adoption),
            "safe_submit_source": file_identity(SAFE_SUBMIT),
            "slurm_driver": file_identity(SLURM_DRIVER),
            **_amendment_identities(4),
        },
        fields={
            "status": "pass",
            "launch_nonce": nonce,
            "comments": [LEGACY_COMMENT, SUPERSEDED_COMMENT],
            "matching_job_count": 0,
            "values_inspected": False,
        },
    )
    output = publish_exclusive(expected, receipt, production_root=root)
    verify_prelaunch(
        output,
        launch_nonce=nonce,
        production_root=root,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    return output


def verify_prelaunch(
    path: Path | str,
    *,
    launch_nonce: str,
    production_root: Path | str,
    fixed5_source_manifest: Path | str,
    adoption_authorization: Path | str,
) -> dict[str, Any]:
    nonce = validate_nonce(launch_nonce)
    root = _exact_root(production_root)
    fixed5 = _require_exact_file(
        fixed5_source_manifest,
        root / "control/FIXED5_SOURCE_MANIFEST_V1.json",
        "fixed-five source manifest",
    )
    adoption = _require_exact_file(
        adoption_authorization,
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "adoption authorization",
    )
    expected = root / f"control/prelaunch/FIXED5_PRELAUNCH_{nonce}.json"
    source = _require_exact_file(path, expected, "prelaunch receipt")
    receipt = verify_receipt(
        source,
        expected_schema=PRELAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=PRELAUNCH_SCENARIO,
    )
    expected_fields = {
        "status": "pass",
        "launch_nonce": nonce,
        "comments": [LEGACY_COMMENT, SUPERSEDED_COMMENT],
        "matching_job_count": 0,
        "values_inspected": False,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"prelaunch receipt {key} differs")
    expected_identities = {
        "fixed5_source_manifest": file_identity(fixed5),
        "adoption_authorization": file_identity(adoption),
        "safe_submit_source": file_identity(SAFE_SUBMIT),
        "slurm_driver": file_identity(SLURM_DRIVER),
        **_amendment_identities(4),
    }
    if receipt.get("identities") != expected_identities:
        raise ValueError("prelaunch receipt identities differ")
    expected_keys = {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        *expected_fields,
    }
    if set(receipt) != expected_keys:
        raise ValueError("prelaunch receipt field topology differs")
    return receipt


def create_launch(
    destination: Path | str,
    *,
    launch_nonce: str,
    slurm_job_id: str,
    prelaunch_receipt: Path | str,
    production_root: Path | str,
    fixed5_source_manifest: Path | str,
    adoption_authorization: Path | str,
) -> Path:
    nonce = validate_nonce(launch_nonce)
    if JOB_ID_RE.fullmatch(slurm_job_id) is None:
        raise ValueError("Slurm job ID must be decimal")
    root = _exact_root(production_root)
    fixed5 = _require_exact_file(
        fixed5_source_manifest,
        root / "control/FIXED5_SOURCE_MANIFEST_V1.json",
        "fixed-five source manifest",
    )
    adoption = _require_exact_file(
        adoption_authorization,
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "adoption authorization",
    )
    prelaunch = root / f"control/prelaunch/FIXED5_PRELAUNCH_{nonce}.json"
    verify_prelaunch(
        prelaunch_receipt,
        launch_nonce=nonce,
        production_root=root,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    if Path(prelaunch_receipt).absolute() != prelaunch:
        raise ValueError("prelaunch receipt path differs")
    expected = (
        root
        / f"control/launch/FIXED5_LAUNCH_{nonce}_JOB_{slurm_job_id}.json"
    )
    if Path(destination).absolute() != expected:
        raise ValueError("launch destination differs")
    receipt = build_receipt(
        schema=LAUNCH_SCHEMA,
        study_id=STUDY_ID,
        scenario=LAUNCH_SCENARIO,
        identities={
            "prelaunch_receipt": file_identity(prelaunch),
            "fixed5_source_manifest": file_identity(fixed5),
            "adoption_authorization": file_identity(adoption),
            "controller": file_identity(CONTROLLER),
            "slurm_driver": file_identity(SLURM_DRIVER),
            **_amendment_identities(5),
        },
        fields={
            "status": "running",
            "launch_nonce": nonce,
            "slurm_job_id": slurm_job_id,
            "job_name": JOB_NAME,
            "comment": LEGACY_COMMENT,
            "tasks": 1,
            "allocated_gpus": 1,
            "visible_gpus": 1,
            "values_inspected": False,
        },
    )
    output = publish_exclusive(expected, receipt, production_root=root)
    verify_launch(
        output,
        launch_nonce=nonce,
        slurm_job_id=slurm_job_id,
        prelaunch_receipt=prelaunch,
        production_root=root,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    return output


def verify_launch(
    path: Path | str,
    *,
    launch_nonce: str,
    slurm_job_id: str,
    prelaunch_receipt: Path | str,
    production_root: Path | str,
    fixed5_source_manifest: Path | str,
    adoption_authorization: Path | str,
) -> dict[str, Any]:
    nonce = validate_nonce(launch_nonce)
    if JOB_ID_RE.fullmatch(slurm_job_id) is None:
        raise ValueError("Slurm job ID must be decimal")
    root = _exact_root(production_root)
    fixed5 = _require_exact_file(
        fixed5_source_manifest,
        root / "control/FIXED5_SOURCE_MANIFEST_V1.json",
        "fixed-five source manifest",
    )
    adoption = _require_exact_file(
        adoption_authorization,
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "adoption authorization",
    )
    expected_prelaunch = (
        root / f"control/prelaunch/FIXED5_PRELAUNCH_{nonce}.json"
    )
    prelaunch = _require_exact_file(
        prelaunch_receipt, expected_prelaunch, "prelaunch receipt"
    )
    expected = (
        root
        / f"control/launch/FIXED5_LAUNCH_{nonce}_JOB_{slurm_job_id}.json"
    )
    source = _require_exact_file(path, expected, "launch receipt")
    receipt = verify_receipt(
        source,
        expected_schema=LAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=LAUNCH_SCENARIO,
    )
    expected_fields = {
        "status": "running",
        "launch_nonce": nonce,
        "slurm_job_id": slurm_job_id,
        "job_name": JOB_NAME,
        "comment": LEGACY_COMMENT,
        "tasks": 1,
        "allocated_gpus": 1,
        "visible_gpus": 1,
        "values_inspected": False,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"launch receipt {key} differs")
    expected_identities = {
        "prelaunch_receipt": file_identity(prelaunch),
        "fixed5_source_manifest": file_identity(fixed5),
        "adoption_authorization": file_identity(adoption),
        "controller": file_identity(CONTROLLER),
        "slurm_driver": file_identity(SLURM_DRIVER),
        **_amendment_identities(5),
    }
    if receipt.get("identities") != expected_identities:
        raise ValueError("launch receipt identities differ")
    expected_keys = {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        *expected_fields,
    }
    if set(receipt) != expected_keys:
        raise ValueError("launch receipt field topology differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("new-nonce")
    create = sub.add_parser("create-prelaunch")
    create.add_argument("--destination", required=True)
    create.add_argument("--launch-nonce", required=True)
    create.add_argument("--production-root", required=True)
    create.add_argument("--fixed5-source-manifest", required=True)
    create.add_argument("--adoption-authorization", required=True)
    create.add_argument("--fixed48-source-manifest", required=True)
    create.add_argument("--authorization-manifest", required=True)
    create.add_argument("--feasibility-gate", required=True)
    args = parser.parse_args(argv)
    if args.command == "new-nonce":
        print(new_nonce())
        return 0
    create_prelaunch(
        args.destination,
        launch_nonce=args.launch_nonce,
        production_root=args.production_root,
        fixed5_source_manifest=args.fixed5_source_manifest,
        adoption_authorization=args.adoption_authorization,
        fixed48_source_manifest=args.fixed48_source_manifest,
        authorization_manifest=args.authorization_manifest,
        feasibility_gate=args.feasibility_gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
