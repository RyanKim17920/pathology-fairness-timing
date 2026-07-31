#!/usr/bin/env python3
"""Freeze and reverify the exact fixed-five implementation closure.

The default manifest is an explicit semantic-role allowlist.  It extends the
already-frozen fixed-48 closure with every fixed-five production source, test,
amendment, and outcome-blind prelaunch receipt.  Creation is exclusive and
fail-closed: a partial file from an interrupted creator is never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from tools.matched_cancer_fixed48_20260730 import source_manifest as fixed48
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
STUDY_ID = "matched_cancer_fixed48_20260730"
SCHEMA = "matched-cancer-fixed5-source-manifest/v1"
SCENARIO = "fixed5_frozen_implementation"

FIXED5_ROOT = REPO / "tools/matched_cancer_fixed5_20260730"
PROTOCOL_ROOT = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution"
)


def _ancestor_spec() -> dict[str, Path]:
    return {
        f"ancestor.{role}": path
        for role, path in fixed48.SOURCE_SPEC.items()
    }


# Explicit rather than directory-discovered: adding a new production source or
# test must be a deliberate source-manifest change.
SOURCE_SPEC: dict[str, Path] = {
    **_ancestor_spec(),
    "fixed5.package_init": FIXED5_ROOT / "__init__.py",
    "fixed5.source_manifest": Path(__file__),
    "fixed5.adoption_authorization": (
        FIXED5_ROOT / "adoption_authorization.py"
    ),
    "fixed5.analyzer": FIXED5_ROOT / "analyzer.py",
    "fixed5.continuation_options_receipt": (
        FIXED5_ROOT / "continuation_options_receipt.py"
    ),
    "fixed5.execution_receipts": FIXED5_ROOT / "execution_receipts.py",
    "fixed5.final_collector": FIXED5_ROOT / "final_collector.py",
    "fixed5.finalizer": FIXED5_ROOT / "finalizer.py",
    "fixed5.full_cardinality_preflight": (
        FIXED5_ROOT / "full_cardinality_preflight.py"
    ),
    "fixed5.launch_receipt": FIXED5_ROOT / "launch_receipt.py",
    "fixed5.safe_submit": FIXED5_ROOT / "safe_submit.sh",
    "fixed5.serial_controller": FIXED5_ROOT / "serial_controller.py",
    "fixed5.serial_driver": FIXED5_ROOT / "serial_fixed5.sbatch",
    "fixed5.verifier": FIXED5_ROOT / "verifier.py",
    "fixed5.test_adoption_authorization": (
        FIXED5_ROOT / "test_adoption_authorization.py"
    ),
    "fixed5.test_analyzer": FIXED5_ROOT / "test_analyzer.py",
    "fixed5.test_continuation_options_receipt": (
        FIXED5_ROOT / "test_continuation_options_receipt.py"
    ),
    "fixed5.test_execution_receipts": (
        FIXED5_ROOT / "test_execution_receipts.py"
    ),
    "fixed5.test_final_collector": (
        FIXED5_ROOT / "test_final_collector.py"
    ),
    "fixed5.test_finalizer": FIXED5_ROOT / "test_finalizer.py",
    "fixed5.test_full_cardinality_preflight": (
        FIXED5_ROOT / "test_full_cardinality_preflight.py"
    ),
    "fixed5.test_launch_receipt": FIXED5_ROOT / "test_launch_receipt.py",
    "fixed5.test_serial_controller": (
        FIXED5_ROOT / "test_serial_controller.py"
    ),
    "fixed5.test_source_manifest": (
        FIXED5_ROOT / "test_source_manifest.py"
    ),
    "fixed5.test_verifier": FIXED5_ROOT / "test_verifier.py",
    "protocol.fixed5_amendment_01": (
        PROTOCOL_ROOT / "FIXED5_SAMPLE_SIZE_AMENDMENT_01.md"
    ),
    "protocol.fixed5_amendment_02": (
        PROTOCOL_ROOT / "FIXED5_SAMPLE_SIZE_AMENDMENT_02.md"
    ),
    "protocol.fixed5_amendment_03": (
        PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_03.md"
    ),
    "protocol.fixed5_amendment_04": (
        PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_04.md"
    ),
    "protocol.fixed5_amendment_05": (
        PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_05.md"
    ),
    "protocol.fixed5_amendment_06": (
        PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_06.md"
    ),
    "protocol.fixed5_amendment_07": (
        PROTOCOL_ROOT / "FIXED5_NUMERIC_AMENDMENT_07.md"
    ),
    "protocol.fixed5_amendment_08": (
        PROTOCOL_ROOT / "FIXED5_NUMERIC_AMENDMENT_08.md"
    ),
    "protocol.fixed5_continuation_options": (
        PROTOCOL_ROOT / "FIXED5_CONTINUATION_OPTIONS.md"
    ),
    "protocol.fixed5_full_cardinality_preflight_receipt": (
        PROTOCOL_ROOT
        / "FIXED5_FULL_CARDINALITY_SYNTHETIC_PREFLIGHT_RECEIPT.json"
    ),
    "control.fixed48_source_manifest_v2": (
        PRODUCTION_ROOT / "control/FIXED48_SOURCE_MANIFEST_V2.json"
    ),
    "control.fixed48_authorization_v3": (
        PRODUCTION_ROOT / "authorization/AUTHORIZATION_MANIFEST_V3.json"
    ),
    "control.fixed48_feasibility_v2": (
        PRODUCTION_ROOT / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
    ),
    "control.fixed5_adoption_authorization_v1": (
        PRODUCTION_ROOT
        / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
    ),
    "control.fixed5_continuation_options_receipt": (
        PRODUCTION_ROOT
        / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json"
    ),
}


def _canonical_regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing: {candidate}") from error
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or not stat.S_ISREG(resolved.stat().st_mode)
        or resolved.stat().st_size == 0
    ):
        raise ValueError(
            f"{label} is empty or non-regular: "
            f"{candidate}"
        )
    return resolved


def _normalized_spec(
    spec: Mapping[str, Path | str],
) -> dict[str, Path]:
    if not spec:
        raise ValueError("source spec may not be empty")
    normalized: dict[str, Path] = {}
    paths: dict[Path, str] = {}
    for role, raw_path in spec.items():
        if not isinstance(role, str) or not role:
            raise ValueError("source roles must be nonempty strings")
        if role in normalized:
            raise ValueError(f"duplicate source role: {role}")
        path = _canonical_regular_file(raw_path, label=f"source {role}")
        if path in paths:
            raise ValueError(
                f"source path has multiple semantic roles: "
                f"{paths[path]} and {role}: {path}"
            )
        paths[path] = role
        normalized[role] = path
    return normalized


def validate_import_closure(
    spec: Mapping[str, Path | str],
) -> list[dict[str, str]]:
    """Require every repository-local Python import in the exact allowlist."""
    normalized = _normalized_spec(spec)
    role_by_path = {path: role for role, path in normalized.items()}
    edges: list[dict[str, str]] = []
    for role, source in normalized.items():
        for imported in sorted(fixed48._resolve_local_imports(source)):
            target = role_by_path.get(imported)
            if target is None:
                raise ValueError(
                    f"unallowlisted local import from {role}: {imported}"
                )
            edges.append({"from_role": role, "to_role": target})
    return sorted(
        edges, key=lambda edge: (edge["from_role"], edge["to_role"])
    )


def _canonical_new_destination(destination: Path | str) -> Path:
    requested = Path(destination)
    if os.path.lexists(requested):
        raise FileExistsError(
            f"refusing to overwrite source manifest: {requested}"
        )
    parent = requested.parent.absolute()
    parent.mkdir(parents=True, exist_ok=True)
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise ValueError("source-manifest destination ancestry is redirected")
    output = parent / requested.name
    if output.absolute() != requested.absolute():
        raise ValueError("source-manifest destination is not canonical")
    return output


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o664)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("published source manifest was redirected")


def create_manifest(
    destination: Path | str,
    *,
    spec: Mapping[str, Path | str] = SOURCE_SPEC,
) -> Path:
    output = _canonical_new_destination(destination)
    normalized = _normalized_spec(spec)
    edges = validate_import_closure(normalized)
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities={
            "sources": {
                role: file_identity(path)
                for role, path in normalized.items()
            }
        },
        fields={
            "status": "frozen",
            "source_roles": sorted(normalized),
            "source_role_count": len(normalized),
            "local_import_edges": edges,
            "local_import_edge_count": len(edges),
            "values_inspected": False,
            "scientific_values_opened": False,
        },
    )
    _write_exclusive(output, canonical_json_bytes(receipt) + b"\n")
    verify_manifest(output, spec=normalized)
    return output


def verify_manifest(
    path: Path | str,
    *,
    spec: Mapping[str, Path | str] = SOURCE_SPEC,
) -> dict[str, Any]:
    source = _canonical_regular_file(path, label="source manifest")
    normalized = _normalized_spec(spec)
    receipt = verify_receipt(
        source,
        expected_schema=SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=SCENARIO,
    )
    expected_keys = {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        "status",
        "source_roles",
        "source_role_count",
        "local_import_edges",
        "local_import_edge_count",
        "values_inspected",
        "scientific_values_opened",
    }
    if set(receipt) != expected_keys:
        raise ValueError("source manifest field topology differs")
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping) or set(identities) != {"sources"}:
        raise ValueError("source manifest identity topology differs")
    sources = identities["sources"]
    if not isinstance(sources, Mapping) or set(sources) != set(normalized):
        raise ValueError("source manifest allowlist topology differs")
    for role, expected_path in normalized.items():
        identity = sources[role]
        if Path(identity["canonical_path"]) != expected_path:
            raise ValueError(f"source role {role} was redirected")
        if identity != file_identity(expected_path):
            raise ValueError(f"source role {role} drifted")
    edges = validate_import_closure(normalized)
    expected_fields = {
        "status": "frozen",
        "source_roles": sorted(normalized),
        "source_role_count": len(normalized),
        "local_import_edges": edges,
        "local_import_edge_count": len(edges),
        "values_inspected": False,
        "scientific_values_opened": False,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"source manifest {key} differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_manifest(args.destination)
    else:
        verify_manifest(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
