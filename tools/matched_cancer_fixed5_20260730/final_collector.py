#!/usr/bin/env python3
"""Outcome-blind, exact-five collector for fixed-five finalization.

This module only validates ancestry and prediction topology.  It never imports
or invokes the fixed-five analyzer or independent verifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from tools.matched_cancer_diagnostic_20260730.exporter import (
    validate_prediction_row,
)
from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    file_identity,
    verify_receipt,
)
from tools.matched_cancer_fixed48_20260730.diag_exporter import (
    verify_collection as verify_seed_collection,
)
from tools.matched_cancer_fixed48_20260730.diag_authorization import (
    verify_authorization as verify_fixed48_authorization,
)
from tools.matched_cancer_fixed48_20260730.feasibility_gate import (
    verify as verify_fixed48_feasibility,
)
from tools.matched_cancer_fixed48_20260730.serial_controller import (
    verify_seed_success,
)
from tools.matched_cancer_fixed48_20260730.source_manifest import (
    verify_manifest as verify_fixed48_manifest,
)

from . import execution_receipts, launch_receipt


REPO = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = launch_receipt.PRODUCTION_ROOT
STUDY_ID = launch_receipt.STUDY_ID
SEEDS = execution_receipts.ADOPTED_SEEDS
ARMS = ("B", "P", "H")
CANCERS = ("BRCA", "LUAD")
HEADS = (42001, 42002, 42003, 42004)
COHORT_SIZES = {"BRCA": 328, "LUAD": 281}
ROW_SCHEMA = "matched-cancer-diagnostic-prediction/v1"
ROWS_PER_SEED = 36_540
EXPECTED_ROWS = 182_700
EXPECTED_COMBINATIONS = 120
EXPECTED_PATIENTS = 609
SCHEMA = "matched-cancer-fixed5-final-collection/v1"
SCENARIO = "brca_luad_black_white_fixed5_final"
RECEIPT_SUFFIX = ".receipt.json"
CONTINUATION_OPTIONS = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_CONTINUATION_OPTIONS.md"
)
CONTINUATION_OPTIONS_RECEIPT = (
    PRODUCTION_ROOT / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json"
)
AMENDMENTS = (
    *launch_receipt.AMENDMENTS,
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_NUMERIC_AMENDMENT_07.md",
    REPO
    / "results/matched_cancer_stage_20260730/fixed5_execution/"
    "FIXED5_NUMERIC_AMENDMENT_08.md",
)
ATTEMPT_RE = re.compile(r"attempt_[0-9]{2,}")

ManifestVerifier = Callable[[Path], Mapping[str, Any]]
SuccessVerifier = Callable[..., Mapping[str, Any]]
CollectionVerifier = Callable[..., Mapping[str, Any]]
StateScanner = Callable[..., execution_receipts.StudyState]
ExcludedVerifier = Callable[..., Mapping[str, Any]]
ControlVerifier = Callable[..., Mapping[str, Any]]
LaunchVerifier = Callable[..., Mapping[str, Any]]
AdoptionVerifier = Callable[..., Mapping[str, Any]]
ContinuationVerifier = Callable[..., Mapping[str, Any]]


def _default_continuation_verifier(
    path: Path,
    *,
    document: Path,
    amendments: Sequence[Path],
) -> Mapping[str, Any]:
    from .continuation_options_receipt import verify

    return verify(path, document=document, amendments=amendments)


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


def _exact_root(production_root: Path | str) -> Path:
    candidate = Path(production_root)
    expected = PRODUCTION_ROOT
    if candidate.is_symlink() or expected.is_symlink():
        raise ValueError("production root may not be a symlink")
    expected_resolved = expected.resolve(strict=True)
    if candidate.absolute() != expected_resolved:
        raise ValueError("production root path differs")
    resolved = candidate.resolve(strict=True)
    if resolved != expected_resolved:
        raise ValueError("production root canonical path differs")
    return resolved


def _exact_file(path: Path | str, expected: Path, label: str) -> Path:
    candidate = Path(path)
    if (
        candidate.absolute() != expected
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve(strict=True) != expected
    ):
        raise ValueError(f"{label} is missing, redirected, or at the wrong path")
    return expected


def _exact_final_artifact(
    path: Path | str,
    *,
    root: Path,
    expected_name: str,
    label: str,
) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if ".." in candidate.parts:
        raise ValueError(f"{label} path may not contain '..'")
    if (
        candidate.name != expected_name
        or ATTEMPT_RE.fullmatch(candidate.parent.name) is None
        or candidate.parent.parent != root / "finalization"
    ):
        raise ValueError(f"{label} path topology differs")
    for ancestor in (
        root / "finalization",
        candidate.parent,
    ):
        if (
            ancestor.is_symlink()
            or not ancestor.is_dir()
            or ancestor.resolve(strict=True) != ancestor
        ):
            raise ValueError(f"{label} ancestry redirected")
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.resolve(strict=True) != candidate
    ):
        raise ValueError(f"{label} is missing or redirected")
    return candidate


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _amendment_identities() -> dict[str, dict[str, Any]]:
    return {
        f"amendment_{index:02d}": file_identity(path)
        for index, path in enumerate(AMENDMENTS, start=1)
    }


def _publish_exclusive_bytes(
    destination: Path, payload: bytes, *, production_root: Path
) -> Path:
    if destination.absolute() != destination:
        raise ValueError("destination must be absolute")
    try:
        destination.relative_to(production_root)
    except ValueError as error:
        raise ValueError("destination escaped production root") from error
    launch_receipt._ensure_directory(production_root, destination.parent)
    if os.path.lexists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if destination.is_symlink() or destination.resolve(strict=True) != destination:
        raise ValueError("published destination redirected")
    return destination


def _load_seed_rows(
    path: Path, *, seed: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[Any, ...]]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[tuple[str, str], tuple[Any, ...]] = {}
    multiplicity: dict[tuple[str, str], int] = {}
    combinations: set[tuple[int, str, str, int]] = set()
    logical_keys: set[tuple[Any, ...]] = set()
    patient_coverage: dict[
        tuple[str, str], set[tuple[str, int, str, int, int | None]]
    ] = {}
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank row")
            row = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            validate_prediction_row(row)
            if row["fm_seed"] != seed:
                raise ValueError(f"{path}:{line_number}: seed differs")
            key = (row["cancer"], row["patient_id"])
            value = (row["y_true"], row["race"], row["fold"])
            if key in metadata and metadata[key] != value:
                raise ValueError(f"seed {seed} patient metadata differs")
            metadata[key] = value
            multiplicity[key] = multiplicity.get(key, 0) + 1
            combinations.add(
                (seed, row["arm"], row["cancer"], row["head_seed"])
            )
            logical_key = (
                row["fm_seed"],
                row["arm"],
                row["cancer"],
                row["head_seed"],
                row["patient_id"],
                row["role"],
                row["outer_fold"],
                row["inner_fold"],
            )
            if logical_key in logical_keys:
                raise ValueError(f"seed {seed} has a duplicate logical row")
            logical_keys.add(logical_key)
            patient_coverage.setdefault(key, set()).add(
                (
                    row["arm"],
                    row["head_seed"],
                    row["role"],
                    row["outer_fold"],
                    row["inner_fold"],
                )
            )
            rows.append(row)
    if len(rows) != ROWS_PER_SEED:
        raise ValueError(f"seed {seed} row count differs")
    if len(metadata) != EXPECTED_PATIENTS:
        raise ValueError(f"seed {seed} patient count differs")
    if any(count != 60 for count in multiplicity.values()):
        raise ValueError(f"seed {seed} patient row multiplicity differs")
    for key, value in metadata.items():
        fold = value[2]
        expected_coverage = {
            (
                arm,
                head,
                "outer_test" if outer == fold else "inner_calibration",
                outer,
                None if outer == fold else fold,
            )
            for arm in ARMS
            for head in HEADS
            for outer in range(5)
        }
        if patient_coverage.get(key) != expected_coverage:
            raise ValueError(
                f"seed {seed} patient arm/head/fold coverage differs"
            )
    expected_combinations = {
        (seed, arm, cancer, head)
        for arm in ARMS
        for cancer in CANCERS
        for head in HEADS
    }
    if combinations != expected_combinations:
        raise ValueError(f"seed {seed} cell topology differs")
    for cancer, expected in COHORT_SIZES.items():
        observed = sum(key[0] == cancer for key in metadata)
        if observed != expected:
            raise ValueError(f"seed {seed} {cancer} cohort size differs")
    return rows, metadata


def _chain_identities(
    state: execution_receipts.StudyState,
) -> dict[str, dict[str, dict[str, Any]]]:
    if state.completed != SEEDS or set(state.chains) != set(SEEDS):
        raise ValueError("collector requires exactly five completed seed chains")
    return {
        str(seed): {
            role: file_identity(path)
            for role, path in state.chains[seed].items()
        }
        for seed in SEEDS
    }


def _verify_controls(
    *,
    root: Path,
    fixed5: Path,
    adoption: Path,
    fixed48: Path,
    authorization: Path,
    feasibility: Path,
    manifest_verifier: ManifestVerifier,
    fixed48_manifest_verifier: ControlVerifier,
    authorization_verifier: ControlVerifier,
    feasibility_verifier: ControlVerifier,
    adoption_verifier: AdoptionVerifier,
) -> None:
    manifest_verifier(fixed5)
    fixed48_manifest_verifier(fixed48)
    authorization_verifier(authorization)
    feasibility_verifier(
        feasibility, authorization_manifest=authorization
    )
    adoption_verifier(
        adoption,
        fixed48_source_manifest=fixed48,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
        production_root=root,
    )


def collect(
    *,
    production_root: Path | str,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    launch: Path,
    excluded_audit: Path,
    continuation_options: Path = CONTINUATION_OPTIONS,
    continuation_options_receipt: Path = CONTINUATION_OPTIONS_RECEIPT,
    destination: Path,
    state_scanner: StateScanner = execution_receipts.scan_state,
    success_verifier: SuccessVerifier = verify_seed_success,
    collection_verifier: CollectionVerifier = verify_seed_collection,
    manifest_verifier: ManifestVerifier = _default_manifest_verifier,
    excluded_verifier: ExcludedVerifier = execution_receipts.verify_excluded_audit,
    fixed48_manifest_verifier: ControlVerifier = verify_fixed48_manifest,
    authorization_verifier: ControlVerifier = verify_fixed48_authorization,
    feasibility_verifier: ControlVerifier = verify_fixed48_feasibility,
    adoption_verifier: AdoptionVerifier = _default_adoption_verifier,
    launch_verifier: LaunchVerifier = launch_receipt.verify_launch,
    continuation_verifier: ContinuationVerifier = (
        _default_continuation_verifier
    ),
) -> Path:
    """Publish an exact, ancestry-bound five-seed matrix and O_EXCL receipt."""
    root = _exact_root(production_root)
    fixed5 = _exact_file(
        fixed5_source_manifest,
        root / "control/FIXED5_SOURCE_MANIFEST_V1.json",
        "fixed-five source manifest",
    )
    adoption = _exact_file(
        adoption_authorization,
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "adoption authorization",
    )
    fixed48 = _exact_file(
        fixed48_source_manifest,
        root / "control/FIXED48_SOURCE_MANIFEST_V2.json",
        "fixed-48 source manifest",
    )
    authorization = _exact_file(
        authorization_manifest,
        root / "authorization/AUTHORIZATION_MANIFEST_V3.json",
        "fixed-48 authorization",
    )
    feasibility = _exact_file(
        feasibility_gate,
        root / "control/FEASIBILITY_GATE_RECEIPT_V2.json",
        "fixed-48 feasibility",
    )
    options = _exact_file(
        continuation_options,
        CONTINUATION_OPTIONS.resolve(strict=True),
        "continuation-options document",
    )
    options_receipt = _exact_file(
        continuation_options_receipt,
        root / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json",
        "continuation-options receipt",
    )
    continuation_verifier(
        options_receipt,
        document=options,
        amendments=AMENDMENTS,
    )
    _verify_controls(
        root=root,
        fixed5=fixed5,
        adoption=adoption,
        fixed48=fixed48,
        authorization=authorization,
        feasibility=feasibility,
        manifest_verifier=manifest_verifier,
        fixed48_manifest_verifier=fixed48_manifest_verifier,
        authorization_verifier=authorization_verifier,
        feasibility_verifier=feasibility_verifier,
        adoption_verifier=adoption_verifier,
    )
    fixed5_identity = file_identity(fixed5)
    execution_receipts.scan_excluded_state(root)
    state = state_scanner(
        production_root=root,
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
        success_verifier=success_verifier,
    )
    launch_data = verify_receipt(
        launch,
        expected_schema=launch_receipt.LAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=launch_receipt.LAUNCH_SCENARIO,
    )
    nonce = launch_data.get("launch_nonce")
    prelaunch = Path(
        launch_data["identities"]["prelaunch_receipt"]["canonical_path"]
    )
    launch_verifier(
        launch,
        launch_nonce=nonce,
        slurm_job_id=launch_data.get("slurm_job_id"),
        prelaunch_receipt=prelaunch,
        production_root=root,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    excluded_verifier(
        excluded_audit,
        production_root=root,
        launch_nonce=nonce,
        launch=launch,
        state=state,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    chains = _chain_identities(state)

    def reverify_runtime() -> execution_receipts.StudyState:
        _verify_controls(
            root=root,
            fixed5=fixed5,
            adoption=adoption,
            fixed48=fixed48,
            authorization=authorization,
            feasibility=feasibility,
            manifest_verifier=manifest_verifier,
            fixed48_manifest_verifier=fixed48_manifest_verifier,
            authorization_verifier=authorization_verifier,
            feasibility_verifier=feasibility_verifier,
            adoption_verifier=adoption_verifier,
        )
        execution_receipts.scan_excluded_state(root)
        current = state_scanner(
            production_root=root,
            fixed48_source_manifest=fixed48,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
            authorization_manifest=authorization,
            feasibility_gate=feasibility,
            success_verifier=success_verifier,
        )
        if _chain_identities(current) != chains:
            raise ValueError("fixed-five seed chains drifted")
        launch_verifier(
            launch,
            launch_nonce=nonce,
            slurm_job_id=launch_data.get("slurm_job_id"),
            prelaunch_receipt=prelaunch,
            production_root=root,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
        )
        excluded_verifier(
            excluded_audit,
            production_root=root,
            launch_nonce=nonce,
            launch=launch,
            state=current,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
        )
        continuation_verifier(
            options_receipt,
            document=options,
            amendments=AMENDMENTS,
        )
        return current

    reference_metadata: dict[tuple[str, str], tuple[Any, ...]] | None = None
    all_rows: list[list[dict[str, Any]]] = []
    seed_sources: dict[str, Any] = {}
    for seed in SEEDS:
        if file_identity(fixed5) != fixed5_identity:
            raise ValueError("fixed-five source manifest drifted")
        manifest_verifier(fixed5)
        success_path = state.chains[seed]["success"]
        success = success_verifier(
            success_path,
            seed=seed,
            source_manifest=fixed48,
            production_root=root,
        )
        try:
            collection_identity = success["identities"]["per_seed_collection"]
            receipt_identity = success["identities"][
                "per_seed_collection_receipt"
            ]
            collection_path = Path(
                collection_identity["canonical_path"]
            ).resolve(strict=True)
            receipt_path = Path(
                receipt_identity["canonical_path"]
            ).resolve(strict=True)
        except (KeyError, TypeError, OSError) as error:
            raise ValueError(
                f"seed {seed} success lacks collection ancestry"
            ) from error
        if (
            file_identity(collection_path) != collection_identity
            or file_identity(receipt_path) != receipt_identity
            or receipt_path
            != collection_path.with_suffix(
                collection_path.suffix + ".receipt.json"
            )
        ):
            raise ValueError(f"seed {seed} collection ancestry differs")
        collection_verifier(collection_path, expected_fm_seed=seed)
        rows, metadata = _load_seed_rows(collection_path, seed=seed)
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError(f"seed {seed} cohort metadata differs")
        all_rows.append(rows)
        seed_sources[str(seed)] = {
            "success_receipt": file_identity(success_path),
            "per_seed_collection": collection_identity,
            "per_seed_collection_receipt": receipt_identity,
            "fixed5_execution_chain": chains[str(seed)],
        }
    if sum(len(rows) for rows in all_rows) != EXPECTED_ROWS:
        raise ValueError("fixed-five final row count differs")
    reverify_runtime()
    destination_path = Path(destination)
    expected_parent = destination_path.parent
    if (
        not destination_path.is_absolute()
        or ".." in destination_path.parts
        or destination_path.name != "fixed5_predictions.jsonl"
        or ATTEMPT_RE.fullmatch(expected_parent.name) is None
        or expected_parent.parent != root / "finalization"
    ):
        raise ValueError("collection destination path differs")
    launch_receipt._ensure_directory(root, expected_parent)
    rendered = b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        for rows in all_rows
        for row in rows
    )
    output = _publish_exclusive_bytes(
        destination_path, rendered, production_root=root
    )
    reverify_runtime()
    if file_identity(fixed5) != fixed5_identity:
        raise ValueError("fixed-five source manifest drifted before seal")
    identities: dict[str, Any] = {
        "fixed5_source_manifest": fixed5_identity,
        "adoption_authorization": file_identity(adoption),
        "fixed48_source_manifest": file_identity(fixed48),
        "authorization_manifest": file_identity(authorization),
        "feasibility_gate": file_identity(feasibility),
        "launch_receipt": file_identity(launch),
        "excluded_seed_audit": file_identity(excluded_audit),
        "continuation_options": file_identity(options),
        "continuation_options_receipt": file_identity(options_receipt),
        "seed_chains": chains,
        "seed_sources": seed_sources,
        "collected_predictions": file_identity(output),
        "collector": file_identity(Path(__file__)),
        **_amendment_identities(),
    }
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities=identities,
        fields={
            "status": "complete",
            "fm_seeds": list(SEEDS),
            "row_schema": ROW_SCHEMA,
            "row_count": EXPECTED_ROWS,
            "combination_count": EXPECTED_COMBINATIONS,
            "patient_count": EXPECTED_PATIENTS,
            "cohort_sizes": dict(COHORT_SIZES),
            "analysis_performed": False,
            "excluded_state_absent": True,
        },
    )
    receipt_path = output.with_suffix(output.suffix + RECEIPT_SUFFIX)
    launch_receipt.publish_exclusive(
        receipt_path, receipt, production_root=root
    )
    verify_final_collection(
        output,
        receipt_path=receipt_path,
        source_manifest=fixed5,
        verify_rows=False,
        manifest_verifier=manifest_verifier,
        state_scanner=state_scanner,
        success_verifier=success_verifier,
        collection_verifier=collection_verifier,
        excluded_verifier=excluded_verifier,
        fixed48_manifest_verifier=fixed48_manifest_verifier,
        authorization_verifier=authorization_verifier,
        feasibility_verifier=feasibility_verifier,
        adoption_verifier=adoption_verifier,
        launch_verifier=launch_verifier,
        continuation_verifier=continuation_verifier,
    )
    return output


def verify_final_collection(
    path: Path | str,
    *,
    receipt_path: Path | str | None = None,
    source_manifest: Path | str,
    verify_rows: bool = True,
    manifest_verifier: ManifestVerifier = _default_manifest_verifier,
    state_scanner: StateScanner = execution_receipts.scan_state,
    success_verifier: SuccessVerifier = verify_seed_success,
    collection_verifier: CollectionVerifier = verify_seed_collection,
    excluded_verifier: ExcludedVerifier = execution_receipts.verify_excluded_audit,
    fixed48_manifest_verifier: ControlVerifier = verify_fixed48_manifest,
    authorization_verifier: ControlVerifier = verify_fixed48_authorization,
    feasibility_verifier: ControlVerifier = verify_fixed48_feasibility,
    adoption_verifier: AdoptionVerifier = _default_adoption_verifier,
    launch_verifier: LaunchVerifier = launch_receipt.verify_launch,
    continuation_verifier: ContinuationVerifier = (
        _default_continuation_verifier
    ),
) -> dict[str, Any]:
    """Verify the sealed matrix and every currently bound identity."""
    source = Path(path)
    root = _exact_root(PRODUCTION_ROOT)
    source = _exact_final_artifact(
        source,
        root=root,
        expected_name="fixed5_predictions.jsonl",
        label="final collection",
    )
    expected_receipt = source.with_suffix(source.suffix + RECEIPT_SUFFIX)
    receipt_source = _exact_final_artifact(
        receipt_path or expected_receipt,
        root=root,
        expected_name="fixed5_predictions.jsonl.receipt.json",
        label="final collection receipt",
    )
    if receipt_source != expected_receipt:
        raise ValueError("final collection receipt path differs")
    receipt = verify_receipt(
        receipt_source,
        expected_schema=SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=SCENARIO,
    )
    expected_fields = {
        "status": "complete",
        "fm_seeds": list(SEEDS),
        "row_schema": ROW_SCHEMA,
        "row_count": EXPECTED_ROWS,
        "combination_count": EXPECTED_COMBINATIONS,
        "patient_count": EXPECTED_PATIENTS,
        "cohort_sizes": dict(COHORT_SIZES),
        "analysis_performed": False,
        "excluded_state_absent": True,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"final collection {key} differs")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        *expected_fields,
    }:
        raise ValueError("final collection field topology differs")
    manifest = _exact_file(
        source_manifest,
        root / "control/FIXED5_SOURCE_MANIFEST_V1.json",
        "fixed-five source manifest",
    )
    adoption = _exact_file(
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        root / "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "adoption authorization",
    )
    fixed48 = _exact_file(
        root / "control/FIXED48_SOURCE_MANIFEST_V2.json",
        root / "control/FIXED48_SOURCE_MANIFEST_V2.json",
        "fixed-48 source manifest",
    )
    authorization = _exact_file(
        root / "authorization/AUTHORIZATION_MANIFEST_V3.json",
        root / "authorization/AUTHORIZATION_MANIFEST_V3.json",
        "fixed-48 authorization",
    )
    feasibility = _exact_file(
        root / "control/FEASIBILITY_GATE_RECEIPT_V2.json",
        root / "control/FEASIBILITY_GATE_RECEIPT_V2.json",
        "fixed-48 feasibility",
    )
    _verify_controls(
        root=root,
        fixed5=manifest,
        adoption=adoption,
        fixed48=fixed48,
        authorization=authorization,
        feasibility=feasibility,
        manifest_verifier=manifest_verifier,
        fixed48_manifest_verifier=fixed48_manifest_verifier,
        authorization_verifier=authorization_verifier,
        feasibility_verifier=feasibility_verifier,
        adoption_verifier=adoption_verifier,
    )
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("final collection identities are missing")
    required = {
        "fixed5_source_manifest",
        "adoption_authorization",
        "fixed48_source_manifest",
        "authorization_manifest",
        "feasibility_gate",
        "launch_receipt",
        "excluded_seed_audit",
        "continuation_options",
        "continuation_options_receipt",
        "seed_chains",
        "seed_sources",
        "collected_predictions",
        "collector",
        *(f"amendment_{index:02d}" for index in range(1, 9)),
    }
    if set(identities) != required:
        raise ValueError("final collection identity topology differs")
    if identities["fixed5_source_manifest"] != file_identity(manifest):
        raise ValueError("final collection source-manifest identity differs")
    if identities["collected_predictions"] != file_identity(source):
        raise ValueError("final collection prediction identity differs")
    expected_static = {
        "adoption_authorization": file_identity(adoption),
        "fixed48_source_manifest": file_identity(fixed48),
        "authorization_manifest": file_identity(authorization),
        "feasibility_gate": file_identity(feasibility),
        "continuation_options": file_identity(
            CONTINUATION_OPTIONS.resolve(strict=True)
        ),
        "continuation_options_receipt": file_identity(
            _exact_file(
                root
                / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json",
                root
                / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json",
                "continuation-options receipt",
            )
        ),
        "collector": file_identity(Path(__file__)),
        **_amendment_identities(),
    }
    for role, identity in expected_static.items():
        if identities.get(role) != identity:
            raise ValueError(f"final collection {role} identity differs")
    continuation_verifier(
        root / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json",
        document=CONTINUATION_OPTIONS.resolve(strict=True),
        amendments=AMENDMENTS,
    )
    if set(identities["seed_chains"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("final collection seed-chain topology differs")
    if set(identities["seed_sources"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("final collection seed-source topology differs")
    execution_receipts.scan_excluded_state(root)
    state = state_scanner(
        production_root=root,
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=manifest,
        adoption_authorization=adoption,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
        success_verifier=success_verifier,
    )
    rebuilt_chains = _chain_identities(state)
    if identities["seed_chains"] != rebuilt_chains:
        raise ValueError("final collection rebuilt seed chains differ")
    launch_path = Path(identities["launch_receipt"]["canonical_path"])
    launch_data = verify_receipt(
        launch_path,
        expected_schema=launch_receipt.LAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=launch_receipt.LAUNCH_SCENARIO,
    )
    prelaunch = Path(
        launch_data["identities"]["prelaunch_receipt"]["canonical_path"]
    )
    launch_verifier(
        launch_path,
        launch_nonce=launch_data.get("launch_nonce"),
        slurm_job_id=launch_data.get("slurm_job_id"),
        prelaunch_receipt=prelaunch,
        production_root=root,
        fixed5_source_manifest=manifest,
        adoption_authorization=adoption,
    )
    excluded_path = Path(
        identities["excluded_seed_audit"]["canonical_path"]
    )
    excluded_verifier(
        excluded_path,
        production_root=root,
        launch_nonce=launch_data.get("launch_nonce"),
        launch=launch_path,
        state=state,
        fixed5_source_manifest=manifest,
        adoption_authorization=adoption,
    )
    rebuilt_sources: dict[str, Any] = {}
    for seed in SEEDS:
        success_path = state.chains[seed]["success"]
        success = success_verifier(
            success_path,
            seed=seed,
            source_manifest=fixed48,
            production_root=root,
        )
        collection_identity = success["identities"]["per_seed_collection"]
        collection_receipt_identity = success["identities"][
            "per_seed_collection_receipt"
        ]
        collection_path = Path(collection_identity["canonical_path"])
        collection_receipt_path = Path(
            collection_receipt_identity["canonical_path"]
        )
        if (
            file_identity(collection_path) != collection_identity
            or file_identity(collection_receipt_path)
            != collection_receipt_identity
            or collection_receipt_path
            != collection_path.with_suffix(
                collection_path.suffix + ".receipt.json"
            )
        ):
            raise ValueError(
                f"final collection seed {seed} source identity differs"
            )
        collection_verifier(collection_path, expected_fm_seed=seed)
        rebuilt_sources[str(seed)] = {
            "success_receipt": file_identity(success_path),
            "per_seed_collection": collection_identity,
            "per_seed_collection_receipt": collection_receipt_identity,
            "fixed5_execution_chain": rebuilt_chains[str(seed)],
        }
    if identities["seed_sources"] != rebuilt_sources:
        raise ValueError("final collection rebuilt seed sources differ")
    if verify_rows:
        metadata: dict[tuple[str, str], tuple[Any, ...]] | None = None
        by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in SEEDS}
        global_coordinates: set[tuple[int, str, str, int]] = set()
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise ValueError(
                        f"final collection line {line_number} is blank"
                    )
                row = json.loads(raw, object_pairs_hook=_unique_object)
                if not isinstance(row, dict) or row.get("fm_seed") not in by_seed:
                    raise ValueError("final collection has an unexpected seed")
                by_seed[row["fm_seed"]].append(row)
                global_coordinates.add(
                    (
                        row["fm_seed"],
                        row["arm"],
                        row["cancer"],
                        row["head_seed"],
                    )
                )
        expected_coordinates = {
            (seed, arm, cancer, head)
            for seed in SEEDS
            for arm in ARMS
            for cancer in CANCERS
            for head in HEADS
        }
        if global_coordinates != expected_coordinates:
            raise ValueError("final collection global coordinates differ")
        for seed in SEEDS:
            # Avoid creating verification artifacts: validate the rows directly
            # with the same exact topology checks used at collection.
            rows = by_seed[seed]
            # Reuse the exact per-seed topology validator without touching disk.
            observed: dict[tuple[str, str], tuple[Any, ...]] = {}
            logical: set[tuple[Any, ...]] = set()
            coverage: dict[tuple[str, str], set[tuple[Any, ...]]] = {}
            for row in rows:
                validate_prediction_row(row)
                key = (row["cancer"], row["patient_id"])
                value = (row["y_true"], row["race"], row["fold"])
                if key in observed and observed[key] != value:
                    raise ValueError("final collection metadata differs")
                observed[key] = value
                logical_key = (
                    row["arm"],
                    row["cancer"],
                    row["head_seed"],
                    row["patient_id"],
                    row["role"],
                    row["outer_fold"],
                    row["inner_fold"],
                )
                if logical_key in logical:
                    raise ValueError(
                        "final collection duplicate logical row"
                    )
                logical.add(logical_key)
                coverage.setdefault(key, set()).add(
                    (
                        row["arm"],
                        row["head_seed"],
                        row["role"],
                        row["outer_fold"],
                        row["inner_fold"],
                    )
                )
            expected_patient_count = sum(COHORT_SIZES.values())
            if (
                len(rows) != ROWS_PER_SEED
                or len(observed) != expected_patient_count
            ):
                raise ValueError("final collection patient topology differs")
            for cancer, expected_count in COHORT_SIZES.items():
                if (
                    sum(key[0] == cancer for key in observed)
                    != expected_count
                ):
                    raise ValueError(
                        "final collection cancer cohort size differs"
                    )
            for key, value in observed.items():
                fold = value[2]
                expected_coverage = {
                    (
                        arm,
                        head,
                        (
                            "outer_test"
                            if outer == fold
                            else "inner_calibration"
                        ),
                        outer,
                        None if outer == fold else fold,
                    )
                    for arm in ARMS
                    for head in HEADS
                    for outer in range(5)
                }
                if coverage.get(key) != expected_coverage:
                    raise ValueError(
                        "final collection patient coverage differs"
                    )
            if metadata is None:
                metadata = observed
            elif observed != metadata:
                raise ValueError("final collection cross-seed metadata differs")
    _verify_controls(
        root=root,
        fixed5=manifest,
        adoption=adoption,
        fixed48=fixed48,
        authorization=authorization,
        feasibility=feasibility,
        manifest_verifier=manifest_verifier,
        fixed48_manifest_verifier=fixed48_manifest_verifier,
        authorization_verifier=authorization_verifier,
        feasibility_verifier=feasibility_verifier,
        adoption_verifier=adoption_verifier,
    )
    execution_receipts.scan_excluded_state(root)
    final_state = state_scanner(
        production_root=root,
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=manifest,
        adoption_authorization=adoption,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
        success_verifier=success_verifier,
    )
    if _chain_identities(final_state) != rebuilt_chains:
        raise ValueError("final collection seed chains drifted during verify")
    launch_verifier(
        launch_path,
        launch_nonce=launch_data.get("launch_nonce"),
        slurm_job_id=launch_data.get("slurm_job_id"),
        prelaunch_receipt=prelaunch,
        production_root=root,
        fixed5_source_manifest=manifest,
        adoption_authorization=adoption,
    )
    excluded_verifier(
        excluded_path,
        production_root=root,
        launch_nonce=launch_data.get("launch_nonce"),
        launch=launch_path,
        state=final_state,
        fixed5_source_manifest=manifest,
        adoption_authorization=adoption,
    )
    continuation_verifier(
        root / "control/FIXED5_CONTINUATION_OPTIONS_RECEIPT.json",
        document=CONTINUATION_OPTIONS.resolve(strict=True),
        amendments=AMENDMENTS,
    )
    return receipt
