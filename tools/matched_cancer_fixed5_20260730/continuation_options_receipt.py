#!/usr/bin/env python3
"""Seal the independently proposed, pre-result continuation options."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)


REPO = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution"
)
DOCUMENT = PROTOCOL_ROOT / "FIXED5_CONTINUATION_OPTIONS.md"
AMENDMENTS = (
    PROTOCOL_ROOT / "FIXED5_SAMPLE_SIZE_AMENDMENT_01.md",
    PROTOCOL_ROOT / "FIXED5_SAMPLE_SIZE_AMENDMENT_02.md",
    PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_03.md",
    PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_04.md",
    PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_05.md",
    PROTOCOL_ROOT / "FIXED5_EXECUTION_AMENDMENT_06.md",
    PROTOCOL_ROOT / "FIXED5_NUMERIC_AMENDMENT_07.md",
    PROTOCOL_ROOT / "FIXED5_NUMERIC_AMENDMENT_08.md",
)
STUDY_ID = "matched_cancer_fixed5_20260730"
SCHEMA = "matched-cancer-fixed5-continuation-options/v1"
SCENARIO = "fixed5_value_blind_continuation_options"
MAX_NEW_FM_SEEDS = 5
MAX_CONCURRENT_STUDY_GPUS = 1


def _exact_regular(path: Path | str, *, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    try:
        canonical = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if (
        candidate.absolute() != canonical
        or not canonical.is_file()
        or canonical.stat().st_size == 0
    ):
        raise ValueError(f"{label} is empty, non-regular, or redirected")
    return canonical


def _inputs(
    document: Path | str,
    amendments: Sequence[Path | str],
) -> tuple[Path, tuple[Path, ...]]:
    options = _exact_regular(document, label="continuation-options document")
    frozen = tuple(
        _exact_regular(path, label=f"amendment {index:02d}")
        for index, path in enumerate(amendments, 1)
    )
    if len(frozen) != 8 or len(set(frozen)) != 8:
        raise ValueError("continuation receipt requires eight unique amendments")
    return options, frozen


def _validate_document_contract(document: Path) -> None:
    text = " ".join(document.read_text(encoding="utf-8").split())
    required = (
        "before the fixed-five analyzer or any scientific result was opened",
        "at most one study GPU",
        "no optional extension or run-until-significance",
        "at most five new FM seeds",
        "downstream-label firewall",
    )
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        raise ValueError(
            "continuation-options document lacks frozen contract phrase(s): "
            + repr(missing)
        )


def _write_exclusive(path: Path, payload: bytes) -> Path:
    if os.path.lexists(path):
        raise FileExistsError(f"continuation receipt exists: {path}")
    parent = path.parent.absolute()
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("continuation receipt destination ancestry redirected")
    output = parent / path.name
    if output.absolute() != path.absolute():
        raise ValueError("continuation receipt destination is not canonical")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o664)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if output.is_symlink() or output.resolve(strict=True) != output:
        raise ValueError("published continuation receipt redirected")
    return output


def create(
    destination: Path | str,
    *,
    document: Path | str = DOCUMENT,
    amendments: Sequence[Path | str] = AMENDMENTS,
) -> Path:
    options, frozen = _inputs(document, amendments)
    _validate_document_contract(options)
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities={
            "continuation_options": file_identity(options),
            "receipt_builder": file_identity(Path(__file__)),
            "amendments": {
                f"amendment_{index:02d}": file_identity(path)
                for index, path in enumerate(frozen, 1)
            },
        },
        fields={
            "status": "frozen",
            "values_inspected": False,
            "scientific_values_opened": False,
            "result_dependent_branching_authorized": False,
            "run_until_significance_authorized": False,
            "max_new_fm_seeds": MAX_NEW_FM_SEEDS,
            "max_concurrent_study_gpus": MAX_CONCURRENT_STUDY_GPUS,
            "sequential_funnel": True,
        },
    )
    output = _write_exclusive(
        Path(destination), canonical_json_bytes(receipt) + b"\n"
    )
    verify(output, document=options, amendments=frozen)
    return output


def verify(
    path: Path | str,
    *,
    document: Path | str = DOCUMENT,
    amendments: Sequence[Path | str] = AMENDMENTS,
) -> dict[str, Any]:
    source = _exact_regular(path, label="continuation-options receipt")
    options, frozen = _inputs(document, amendments)
    _validate_document_contract(options)
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
        "values_inspected",
        "scientific_values_opened",
        "result_dependent_branching_authorized",
        "run_until_significance_authorized",
        "max_new_fm_seeds",
        "max_concurrent_study_gpus",
        "sequential_funnel",
    }
    if set(receipt) != expected_keys:
        raise ValueError("continuation-options receipt field topology differs")
    expected_identities = {
        "continuation_options": file_identity(options),
        "receipt_builder": file_identity(Path(__file__)),
        "amendments": {
            f"amendment_{index:02d}": file_identity(amendment)
            for index, amendment in enumerate(frozen, 1)
        },
    }
    if receipt.get("identities") != expected_identities:
        raise ValueError("continuation-options receipt identities differ")
    expected_fields: Mapping[str, Any] = {
        "status": "frozen",
        "values_inspected": False,
        "scientific_values_opened": False,
        "result_dependent_branching_authorized": False,
        "run_until_significance_authorized": False,
        "max_new_fm_seeds": MAX_NEW_FM_SEEDS,
        "max_concurrent_study_gpus": MAX_CONCURRENT_STUDY_GPUS,
        "sequential_funnel": True,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"continuation-options receipt {key} differs")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--destination", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create(args.destination)
    else:
        verify(args.receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
