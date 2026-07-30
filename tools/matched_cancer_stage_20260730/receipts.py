"""Canonical, atomic, tamper-evident provenance receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


TOPOLOGY_SCHEMA = "matched-cancer-stage-topology/v1"
IDENTITY_KEYS = frozenset({"canonical_path", "bytes", "sha256"})


class ReceiptVerificationError(ValueError):
    """A persisted receipt or one of the files it binds has changed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 encoding used for hashes and receipts."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical-JSON serializable: {error}") from error
    return text.encode("utf-8")


def require_regular_file(path: Path | str) -> Path:
    """Resolve a nonempty, non-symlink regular file to its canonical path."""
    candidate = Path(path)
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.stat().st_size == 0
    ):
        raise ValueError(
            f"required regular file is missing/empty/symlink: {candidate}"
        )
    return candidate.resolve(strict=True)


def sha256_file(path: Path | str) -> str:
    """Hash a required regular file without loading it all into memory."""
    canonical = require_regular_file(path)
    digest = hashlib.sha256()
    with canonical.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path | str) -> dict[str, Any]:
    """Return the canonical path, byte count, and content hash of a file."""
    canonical = require_regular_file(path)
    return {
        "canonical_path": str(canonical),
        "bytes": canonical.stat().st_size,
        "sha256": sha256_file(canonical),
    }


def _identity_roles(
    identities: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[dict[str, str]]:
    roles: list[dict[str, str]] = []
    for name, value in identities.items():
        if not isinstance(name, str) or not name:
            raise ValueError("identity role names must be nonempty strings")
        role = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping) and set(value) == IDENTITY_KEYS:
            sha256 = value.get("sha256")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError(f"invalid SHA-256 for identity role {role!r}")
            roles.append({"role": role, "file_sha256": sha256})
        elif isinstance(value, Mapping):
            roles.extend(_identity_roles(value, prefix=role))
        else:
            raise ValueError(f"identity role {role!r} is not a file identity")
    if not roles:
        raise ValueError("at least one file identity is required")
    return roles


def topology_sha256(identities: Mapping[str, Any]) -> str:
    """Bind each semantic file role to its digest in a canonical topology."""
    payload = {
        "schema": TOPOLOGY_SCHEMA,
        "roles": sorted(_identity_roles(identities), key=lambda row: row["role"]),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_receipt(
    *,
    schema: str,
    study_id: str,
    scenario: str,
    identities: Mapping[str, Any],
    fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a receipt and its semantic-role topology digest."""
    for label, value in (
        ("schema", schema),
        ("study_id", study_id),
        ("scenario", scenario),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty string")
    receipt: dict[str, Any] = {
        "schema": schema,
        "study_id": study_id,
        "scenario": scenario,
        "identities": dict(identities),
        "topology_sha256": topology_sha256(identities),
    }
    if fields:
        collision = set(fields) & set(receipt)
        if collision:
            raise ValueError(f"receipt field collision: {sorted(collision)}")
        receipt.update(fields)
    # Validate the complete object now, rather than during persistence.
    canonical_json_bytes(receipt)
    return receipt


def atomic_write_receipt(path: Path | str, receipt: Mapping[str, Any]) -> Path:
    """Persist canonical JSON with file and directory fsync plus atomic replace."""
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"receipt destination may not be a symlink: {requested}")
    destination = requested.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt) + b"\n"
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
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptVerificationError(f"duplicate JSON key in receipt: {key}")
        result[key] = value
    return result


def load_receipt(path: Path | str) -> dict[str, Any]:
    """Load a receipt while requiring its exact canonical persisted encoding."""
    source = require_regular_file(path)
    raw = source.read_bytes()
    try:
        receipt = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ReceiptVerificationError(f"invalid receipt JSON: {error}") from error
    if not isinstance(receipt, dict):
        raise ReceiptVerificationError("receipt JSON root must be an object")
    if raw != canonical_json_bytes(receipt) + b"\n":
        raise ReceiptVerificationError("receipt JSON is not canonically encoded")
    return receipt


def _recorded_identities(
    identities: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in identities.items():
        role = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Mapping) and set(value) == IDENTITY_KEYS:
            records.append((role, value))
        elif isinstance(value, Mapping):
            records.extend(_recorded_identities(value, prefix=role))
        else:
            raise ReceiptVerificationError(
                f"identity role {role!r} is not a file identity"
            )
    return records


def verify_receipt(
    receipt_or_path: Mapping[str, Any] | Path | str,
    *,
    expected_schema: str | None = None,
    expected_study_id: str | None = None,
    expected_scenario: str | None = None,
) -> dict[str, Any]:
    """Verify receipt structure, topology, and every bound file's current bytes."""
    receipt = (
        load_receipt(receipt_or_path)
        if isinstance(receipt_or_path, (Path, str))
        else dict(receipt_or_path)
    )
    for key, expected in (
        ("schema", expected_schema),
        ("study_id", expected_study_id),
        ("scenario", expected_scenario),
    ):
        actual = receipt.get(key)
        if not isinstance(actual, str) or not actual:
            raise ReceiptVerificationError(f"receipt {key} is missing or invalid")
        if expected is not None and actual != expected:
            raise ReceiptVerificationError(
                f"receipt {key}={actual!r}, expected {expected!r}"
            )
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping):
        raise ReceiptVerificationError("receipt identities are missing or invalid")
    try:
        computed_topology = topology_sha256(identities)
    except ValueError as error:
        raise ReceiptVerificationError(str(error)) from error
    if receipt.get("topology_sha256") != computed_topology:
        raise ReceiptVerificationError("receipt topology SHA-256 mismatch")

    records = _recorded_identities(identities)
    if not records:
        raise ReceiptVerificationError("receipt contains no file identities")
    for role, recorded in records:
        path = recorded.get("canonical_path")
        try:
            current = file_identity(path)
        except (OSError, ValueError, TypeError) as error:
            raise ReceiptVerificationError(
                f"cannot verify identity role {role!r}: {error}"
            ) from error
        if current != dict(recorded):
            raise ReceiptVerificationError(
                f"file identity mismatch for role {role!r}: {path}"
            )
    return receipt
