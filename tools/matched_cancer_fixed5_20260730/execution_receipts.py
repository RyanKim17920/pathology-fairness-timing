#!/usr/bin/env python3
"""Attempt lifecycle and excluded-state receipts for fixed-five execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    file_identity,
    verify_receipt,
)
from tools.matched_cancer_fixed48_20260730 import (
    serial_controller as fixed48_controller,
)

from . import launch_receipt


REPO = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = launch_receipt.PRODUCTION_ROOT
STUDY_ID = launch_receipt.STUDY_ID
ADOPTED_SEEDS = (32001, 32002, 32003, 32004, 32005)
CONTROLLER_SEEDS = (32002, 32003, 32004, 32005)
EXCLUDED_SEEDS = tuple(range(32006, 32049))
ATTEMPT_RE = re.compile(r"attempt_([0-9]{2,})")
TEMP_RE = re.compile(
    r"\.FIXED5_(?:START|COMPLETE)_RECEIPT\.json\..+\.tmp"
)
START_NAME = "FIXED5_START_RECEIPT.json"
COMPLETE_NAME = "FIXED5_COMPLETE_RECEIPT.json"
SUCCESS_NAME = fixed48_controller.SUCCESS_NAME
START_SCHEMA = "matched-cancer-fixed5-seed-start/v1"
COMPLETE_SCHEMA = "matched-cancer-fixed5-seed-complete/v1"
EXCLUDED_SCHEMA = "matched-cancer-fixed5-excluded-seed-audit/v1"
START_SCENARIO = "fixed5_seed_start"
COMPLETE_SCENARIO = "fixed5_seed_complete"
EXCLUDED_SCENARIO = "fixed5_excluded_seed_audit"

PACKAGE = Path(__file__).resolve().parent
CONTROLLER = PACKAGE / "serial_controller.py"
FROZEN_WORKER = (
    REPO / "tools/matched_cancer_fixed48_20260730/seed_worker.py"
)
AMENDMENTS = launch_receipt.AMENDMENTS

SuccessVerifier = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class StudyState:
    completed: tuple[int, ...]
    lowest_incomplete: int
    resumable: Mapping[int, int]
    used_attempts: Mapping[int, tuple[int, ...]]
    chains: Mapping[int, Mapping[str, Path]]


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


def _exact_file(path: Path | str, expected: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a non-symlink file")
    if candidate.absolute() != expected:
        raise ValueError(f"{label} path differs")
    resolved = candidate.resolve(strict=True)
    if resolved != expected:
        raise ValueError(f"{label} canonical path differs")
    return resolved


def _canonical_directory(
    root: Path,
    directory: Path,
    *,
    create: bool = False,
) -> Path:
    if directory.absolute() != directory:
        raise ValueError("directory must be absolute")
    try:
        relative = directory.relative_to(root)
    except ValueError as error:
        raise ValueError("directory escaped production root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if create:
            try:
                current.mkdir()
            except FileExistsError:
                pass
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"directory is missing or redirected: {current}")
        if current.resolve(strict=True) != current:
            raise ValueError(f"directory ancestry redirected: {current}")
    return directory


def scan_excluded_state(production_root: Path | str) -> None:
    """Reject even an empty file, directory, or broken symlink for excluded seeds."""
    root = _exact_root(production_root)
    for seed in EXCLUDED_SEEDS:
        for family in ("calibration", "diagnostic", "fixed5_execution"):
            candidate = root / family / f"seed_{seed}"
            if candidate.exists() or candidate.is_symlink():
                raise ValueError(
                    f"excluded-seed state exists for seed {seed}: {candidate}"
                )


def _attempt_directories(seed_root: Path) -> dict[int, Path]:
    for ancestor in (seed_root.parent, seed_root):
        if ancestor.exists() or ancestor.is_symlink():
            if (
                ancestor.is_symlink()
                or not ancestor.is_dir()
                or ancestor.resolve(strict=True) != ancestor
            ):
                raise ValueError(
                    f"seed-root ancestry is not canonical: {ancestor}"
                )
    if not seed_root.exists():
        return {}
    if seed_root.is_symlink() or not seed_root.is_dir():
        raise ValueError(f"seed root is not a canonical directory: {seed_root}")
    result: dict[int, Path] = {}
    for candidate in seed_root.iterdir():
        match = ATTEMPT_RE.fullmatch(candidate.name)
        if match is None:
            raise ValueError(f"unexpected seed-root entry: {candidate}")
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"attempt is not a canonical directory: {candidate}")
        if candidate.resolve(strict=True) != candidate:
            raise ValueError(f"attempt directory redirected: {candidate}")
        number = int(match.group(1))
        if number < 1 or number in result:
            raise ValueError(f"invalid/duplicate attempt number: {candidate}")
        result[number] = candidate
    return result


def _amendment_identities() -> dict[str, dict[str, Any]]:
    return {
        f"amendment_{index:02d}": file_identity(path)
        for index, path in enumerate(AMENDMENTS, start=1)
    }


def _start_identities(
    *,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    prelaunch_receipt: Path,
    launch: Path,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path,
) -> dict[str, Any]:
    return {
        "fixed5_source_manifest": file_identity(fixed5_source_manifest),
        "adoption_authorization": file_identity(adoption_authorization),
        "prelaunch_receipt": file_identity(prelaunch_receipt),
        "launch_receipt": file_identity(launch),
        "fixed48_source_manifest": file_identity(fixed48_source_manifest),
        "fixed48_authorization": file_identity(authorization_manifest),
        "fixed48_feasibility": file_identity(feasibility_gate),
        "controller": file_identity(CONTROLLER),
        "frozen_fixed48_worker": file_identity(frozen_worker),
        **_amendment_identities(),
    }


def _execution_paths(
    *,
    production_root: Path | str,
    seed: int,
    attempt_name: str,
) -> tuple[Path, Path]:
    if seed not in CONTROLLER_SEEDS:
        raise ValueError("execution receipt seed must be 32002..32005")
    match = ATTEMPT_RE.fullmatch(attempt_name)
    if match is None or int(match.group(1)) < 1:
        raise ValueError("attempt name must be canonical attempt_NN")
    root = _exact_root(production_root)
    attempt = (
        root / "fixed5_execution" / f"seed_{seed}" / attempt_name
    )
    return root, attempt


def reserve_attempt(
    *,
    production_root: Path | str,
    seed: int,
    attempt_number: int,
) -> Path:
    """Exclusively reserve one shared attempt number without creating worker roots."""
    if seed not in CONTROLLER_SEEDS:
        raise ValueError("reservation seed must be 32002..32005")
    if isinstance(attempt_number, bool) or attempt_number < 1:
        raise ValueError("attempt number must be positive")
    root = _exact_root(production_root)
    parent = root / "fixed5_execution" / f"seed_{seed}"
    _canonical_directory(root, parent, create=True)
    attempt = parent / f"attempt_{attempt_number:02d}"
    attempt.mkdir()
    _canonical_directory(root, attempt)
    return attempt


def _validate_execution_topology(
    attempt: Path,
    *,
    allow_temporary_only: bool,
) -> tuple[bool, bool]:
    names: set[str] = set()
    for child in attempt.iterdir():
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"invalid execution-attempt entry: {child}")
        name = child.name
        if name in names:
            raise ValueError(f"duplicate execution-attempt entry: {name}")
        names.add(name)
        if name not in {START_NAME, COMPLETE_NAME} and TEMP_RE.fullmatch(name) is None:
            raise ValueError(f"unexpected execution-attempt entry: {child}")
    canonical = names & {START_NAME, COMPLETE_NAME}
    temporary = names - canonical
    if temporary and canonical:
        raise ValueError("temporary receipt remains beside canonical evidence")
    if temporary and not allow_temporary_only:
        raise ValueError("temporary-only execution reservation is not allowed")
    if COMPLETE_NAME in names and START_NAME not in names:
        raise ValueError("completion exists without start")
    return START_NAME in names, COMPLETE_NAME in names


def publish_start(
    *,
    production_root: Path | str,
    seed: int,
    attempt_name: str,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    prelaunch_receipt: Path,
    launch: Path,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path = FROZEN_WORKER,
) -> Path:
    root, attempt = _execution_paths(
        production_root=production_root,
        seed=seed,
        attempt_name=attempt_name,
    )
    _canonical_directory(root, attempt)
    has_start, has_complete = _validate_execution_topology(
        attempt, allow_temporary_only=True
    )
    if has_start or has_complete:
        raise FileExistsError("execution reservation already has canonical evidence")
    if any(attempt.iterdir()):
        raise ValueError("abandoned temporary reservation cannot be reused")
    for family in ("calibration", "diagnostic"):
        worker_attempt = root / family / f"seed_{seed}" / attempt_name
        if worker_attempt.exists() or worker_attempt.is_symlink():
            raise ValueError("worker attempt exists before fixed-five START")
    receipt = build_receipt(
        schema=START_SCHEMA,
        study_id=STUDY_ID,
        scenario=START_SCENARIO,
        identities=_start_identities(
            fixed5_source_manifest=fixed5_source_manifest,
            adoption_authorization=adoption_authorization,
            prelaunch_receipt=prelaunch_receipt,
            launch=launch,
            fixed48_source_manifest=fixed48_source_manifest,
            authorization_manifest=authorization_manifest,
            feasibility_gate=feasibility_gate,
            frozen_worker=frozen_worker,
        ),
        fields={
            "status": "started",
            "fm_seed": seed,
            "attempt_name": attempt_name,
            "values_inspected": False,
            "excluded_seed_state_absent": True,
        },
    )
    destination = attempt / START_NAME
    output = launch_receipt.publish_exclusive(
        destination, receipt, production_root=root
    )
    verify_start(
        output,
        production_root=root,
        seed=seed,
        attempt_name=attempt_name,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        prelaunch_receipt=prelaunch_receipt,
        launch=launch,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        frozen_worker=frozen_worker,
    )
    return output


def verify_start(
    path: Path | str,
    *,
    production_root: Path | str,
    seed: int,
    attempt_name: str,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    prelaunch_receipt: Path | None,
    launch: Path | None,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path = FROZEN_WORKER,
) -> dict[str, Any]:
    root, attempt = _execution_paths(
        production_root=production_root,
        seed=seed,
        attempt_name=attempt_name,
    )
    _canonical_directory(root, attempt)
    source = _exact_file(path, attempt / START_NAME, "START receipt")
    receipt = verify_receipt(
        source,
        expected_schema=START_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=START_SCENARIO,
    )
    expected_fields = {
        "status": "started",
        "fm_seed": seed,
        "attempt_name": attempt_name,
        "values_inspected": False,
        "excluded_seed_state_absent": True,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"START receipt {key} differs")
    recorded_identities = receipt.get("identities", {})
    if not isinstance(recorded_identities, Mapping):
        raise ValueError("START receipt identities are missing")
    try:
        recorded_prelaunch = Path(
            recorded_identities["prelaunch_receipt"]["canonical_path"]
        )
        recorded_launch = Path(
            recorded_identities["launch_receipt"]["canonical_path"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("START launch ancestry is missing") from error
    if prelaunch_receipt is not None and Path(prelaunch_receipt) != recorded_prelaunch:
        raise ValueError("START prelaunch receipt differs")
    if launch is not None and Path(launch) != recorded_launch:
        raise ValueError("START launch receipt differs")
    launch_data = verify_receipt(
        recorded_launch,
        expected_schema=launch_receipt.LAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=launch_receipt.LAUNCH_SCENARIO,
    )
    launch_receipt.verify_launch(
        recorded_launch,
        launch_nonce=launch_data.get("launch_nonce"),
        slurm_job_id=launch_data.get("slurm_job_id"),
        prelaunch_receipt=recorded_prelaunch,
        production_root=root,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
    )
    identities = _start_identities(
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        prelaunch_receipt=recorded_prelaunch,
        launch=recorded_launch,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        frozen_worker=frozen_worker,
    )
    if receipt.get("identities") != identities:
        raise ValueError("START receipt identities differ")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        *expected_fields,
    }:
        raise ValueError("START receipt field topology differs")
    return receipt


def publish_complete(
    *,
    production_root: Path | str,
    seed: int,
    attempt_name: str,
    fixed48_success: Path,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    prelaunch_receipt: Path,
    launch: Path,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path = FROZEN_WORKER,
    success_verifier: SuccessVerifier = fixed48_controller.verify_seed_success,
) -> Path:
    root, attempt = _execution_paths(
        production_root=production_root,
        seed=seed,
        attempt_name=attempt_name,
    )
    start_path = attempt / START_NAME
    start = verify_start(
        start_path,
        production_root=root,
        seed=seed,
        attempt_name=attempt_name,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        prelaunch_receipt=None,
        launch=None,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        frozen_worker=frozen_worker,
    )
    expected_success = (
        root / "diagnostic" / f"seed_{seed}" / attempt_name / SUCCESS_NAME
    )
    success = _exact_file(
        fixed48_success, expected_success, "fixed-48 success receipt"
    )
    success_verifier(
        success,
        seed=seed,
        source_manifest=fixed48_source_manifest,
        production_root=root,
    )
    receipt = build_receipt(
        schema=COMPLETE_SCHEMA,
        study_id=STUDY_ID,
        scenario=COMPLETE_SCENARIO,
        identities={
            "start_receipt": file_identity(start_path),
            "fixed48_success": file_identity(success),
            "start_bound_identities": start["identities"],
        },
        fields={
            "status": "complete",
            "fm_seed": seed,
            "attempt_name": attempt_name,
            "values_inspected": False,
            "excluded_seed_state_absent": True,
        },
    )
    output = launch_receipt.publish_exclusive(
        attempt / COMPLETE_NAME, receipt, production_root=root
    )
    verify_complete(
        output,
        production_root=root,
        seed=seed,
        attempt_name=attempt_name,
        fixed48_success=success,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        prelaunch_receipt=None,
        launch=None,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        frozen_worker=frozen_worker,
        success_verifier=success_verifier,
    )
    return output


def verify_complete(
    path: Path | str,
    *,
    production_root: Path | str,
    seed: int,
    attempt_name: str,
    fixed48_success: Path,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    prelaunch_receipt: Path | None,
    launch: Path | None,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path = FROZEN_WORKER,
    success_verifier: SuccessVerifier = fixed48_controller.verify_seed_success,
) -> dict[str, Any]:
    root, attempt = _execution_paths(
        production_root=production_root,
        seed=seed,
        attempt_name=attempt_name,
    )
    start_path = attempt / START_NAME
    start = verify_start(
        start_path,
        production_root=root,
        seed=seed,
        attempt_name=attempt_name,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        prelaunch_receipt=prelaunch_receipt,
        launch=launch,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        frozen_worker=frozen_worker,
    )
    expected_success = (
        root / "diagnostic" / f"seed_{seed}" / attempt_name / SUCCESS_NAME
    )
    success = _exact_file(
        fixed48_success, expected_success, "fixed-48 success receipt"
    )
    success_verifier(
        success,
        seed=seed,
        source_manifest=fixed48_source_manifest,
        production_root=root,
    )
    source = _exact_file(path, attempt / COMPLETE_NAME, "COMPLETE receipt")
    receipt = verify_receipt(
        source,
        expected_schema=COMPLETE_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=COMPLETE_SCENARIO,
    )
    expected_fields = {
        "status": "complete",
        "fm_seed": seed,
        "attempt_name": attempt_name,
        "values_inspected": False,
        "excluded_seed_state_absent": True,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"COMPLETE receipt {key} differs")
    expected_identities = {
        "start_receipt": file_identity(start_path),
        "fixed48_success": file_identity(success),
        "start_bound_identities": start["identities"],
    }
    if receipt.get("identities") != expected_identities:
        raise ValueError("COMPLETE receipt identities differ")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        *expected_fields,
    }:
        raise ValueError("COMPLETE receipt field topology differs")
    return receipt


def _seed_attempt_state(
    *,
    root: Path,
    seed: int,
    fixed48_source_manifest: Path,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path,
    success_verifier: SuccessVerifier,
) -> tuple[tuple[int, ...], int | None, Mapping[str, Path] | None]:
    calibration = _attempt_directories(
        root / "calibration" / f"seed_{seed}"
    )
    diagnostic = _attempt_directories(root / "diagnostic" / f"seed_{seed}")
    execution = _attempt_directories(
        root / "fixed5_execution" / f"seed_{seed}"
    )
    numbers = tuple(sorted(set(calibration) | set(diagnostic) | set(execution)))
    successes: list[tuple[int, Path, Path]] = []
    completions: list[tuple[int, Path, Path, Path]] = []
    for number in numbers:
        attempt_name = f"attempt_{number:02d}"
        calibration_attempt = calibration.get(number)
        diagnostic_attempt = diagnostic.get(number)
        execution_attempt = execution.get(number)
        if execution_attempt is None:
            if calibration_attempt is not None or diagnostic_attempt is not None:
                raise ValueError(
                    f"orphan worker attempt without START: seed {seed} {attempt_name}"
                )
            continue
        has_start, has_complete = _validate_execution_topology(
            execution_attempt, allow_temporary_only=True
        )
        worker_state = (
            calibration_attempt is not None or diagnostic_attempt is not None
        )
        if not has_start:
            if worker_state:
                raise ValueError(
                    f"worker state exists without START: seed {seed} {attempt_name}"
                )
            continue
        start_path = execution_attempt / START_NAME
        verify_start(
            start_path,
            production_root=root,
            seed=seed,
            attempt_name=attempt_name,
            fixed5_source_manifest=fixed5_source_manifest,
            adoption_authorization=adoption_authorization,
            prelaunch_receipt=None,
            launch=None,
            fixed48_source_manifest=fixed48_source_manifest,
            authorization_manifest=authorization_manifest,
            feasibility_gate=feasibility_gate,
            frozen_worker=frozen_worker,
        )
        success_path = (
            diagnostic_attempt / SUCCESS_NAME
            if diagnostic_attempt is not None
            else None
        )
        success_exists = success_path is not None and (
            success_path.exists() or success_path.is_symlink()
        )
        if has_complete and not success_exists:
            raise ValueError("COMPLETE exists without matching fixed-48 success")
        if success_exists:
            success_verifier(
                success_path,
                seed=seed,
                source_manifest=fixed48_source_manifest,
                production_root=root,
            )
            successes.append((number, start_path, success_path))
            if has_complete:
                complete_path = execution_attempt / COMPLETE_NAME
                verify_complete(
                    complete_path,
                    production_root=root,
                    seed=seed,
                    attempt_name=attempt_name,
                    fixed48_success=success_path,
                    fixed5_source_manifest=fixed5_source_manifest,
                    adoption_authorization=adoption_authorization,
                    prelaunch_receipt=None,
                    launch=None,
                    fixed48_source_manifest=fixed48_source_manifest,
                    authorization_manifest=authorization_manifest,
                    feasibility_gate=feasibility_gate,
                    frozen_worker=frozen_worker,
                    success_verifier=success_verifier,
                )
                completions.append(
                    (number, start_path, complete_path, success_path)
                )
    if len(successes) > 1:
        raise ValueError(f"seed {seed} has multiple fixed-48 successes")
    if len(completions) > 1:
        raise ValueError(f"seed {seed} has multiple fixed-five completions")
    successful_number = (
        successes[0][0] if successes else None
    )
    if successful_number is not None and any(
        number > successful_number for number in numbers
    ):
        raise ValueError("state exists after an already successful attempt")
    if completions:
        number, start, complete, success = completions[0]
        return numbers, None, {
            "start": start,
            "complete": complete,
            "success": success,
        }
    if successes:
        return numbers, successes[0][0], None
    return numbers, None, None


def scan_state(
    *,
    production_root: Path | str,
    fixed48_source_manifest: Path,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    frozen_worker: Path = FROZEN_WORKER,
    success_verifier: SuccessVerifier = fixed48_controller.verify_seed_success,
) -> StudyState:
    root = _exact_root(production_root)
    scan_excluded_state(root)
    if (
        root / "fixed5_execution/seed_32001"
    ).exists() or (
        root / "fixed5_execution/seed_32001"
    ).is_symlink():
        raise ValueError("seed 32001 may not have fixed-five execution state")

    seed1_calibration = _attempt_directories(root / "calibration/seed_32001")
    seed1_diagnostic = _attempt_directories(root / "diagnostic/seed_32001")
    seed1_successes: list[Path] = []
    for number, attempt in seed1_diagnostic.items():
        success = attempt / SUCCESS_NAME
        if success.exists() or success.is_symlink():
            if number not in seed1_calibration:
                raise ValueError("seed-32001 success lacks calibration attempt")
            seed1_successes.append(success)
    if len(seed1_successes) != 1:
        raise ValueError("seed 32001 must have exactly one fixed-48 success")
    success_verifier(
        seed1_successes[0],
        seed=32001,
        source_manifest=fixed48_source_manifest,
        production_root=root,
    )

    completed: list[int] = [32001]
    first_incomplete: int | None = None
    resumable: dict[int, int] = {}
    used: dict[int, tuple[int, ...]] = {32001: tuple(
        sorted(set(seed1_calibration) | set(seed1_diagnostic))
    )}
    chains: dict[int, Mapping[str, Path]] = {
        32001: {"success": seed1_successes[0]}
    }
    for seed in CONTROLLER_SEEDS:
        numbers, resumable_attempt, chain = _seed_attempt_state(
            root=root,
            seed=seed,
            fixed48_source_manifest=fixed48_source_manifest,
            fixed5_source_manifest=fixed5_source_manifest,
            adoption_authorization=adoption_authorization,
            authorization_manifest=authorization_manifest,
            feasibility_gate=feasibility_gate,
            frozen_worker=frozen_worker,
            success_verifier=success_verifier,
        )
        used[seed] = numbers
        if chain is not None:
            if first_incomplete is not None:
                raise ValueError("completed seeds are not a contiguous prefix")
            completed.append(seed)
            chains[seed] = chain
        else:
            if first_incomplete is None:
                first_incomplete = seed
            elif numbers:
                raise ValueError("state exists after lowest incomplete seed")
            if resumable_attempt is not None:
                resumable[seed] = resumable_attempt
    return StudyState(
        completed=tuple(completed),
        lowest_incomplete=first_incomplete or 32006,
        resumable=resumable,
        used_attempts=used,
        chains=chains,
    )


def next_attempt_number(state: StudyState, seed: int) -> int:
    if seed not in CONTROLLER_SEEDS:
        raise ValueError("next-attempt seed must be 32002..32005")
    used = set(state.used_attempts.get(seed, ()))
    return next(
        candidate
        for candidate in range(1, max(used, default=0) + 2)
        if candidate not in used
    )


def publish_excluded_audit(
    *,
    production_root: Path | str,
    launch_nonce: str,
    launch: Path,
    state: StudyState,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
) -> Path:
    nonce = launch_receipt.validate_nonce(launch_nonce)
    root = _exact_root(production_root)
    scan_excluded_state(root)
    if state.completed != ADOPTED_SEEDS or set(state.chains) != set(ADOPTED_SEEDS):
        raise ValueError("excluded audit requires all five verified seed chains")
    chain_ids: dict[str, Any] = {}
    for seed, chain in state.chains.items():
        chain_ids[f"seed_{seed}"] = {
            role: file_identity(path) for role, path in chain.items()
        }
    receipt = build_receipt(
        schema=EXCLUDED_SCHEMA,
        study_id=STUDY_ID,
        scenario=EXCLUDED_SCENARIO,
        identities={
            "fixed5_source_manifest": file_identity(fixed5_source_manifest),
            "adoption_authorization": file_identity(adoption_authorization),
            "launch_receipt": file_identity(launch),
            "controller": file_identity(CONTROLLER),
            "seed_chains": chain_ids,
            **_amendment_identities(),
        },
        fields={
            "status": "pass",
            "launch_nonce": nonce,
            "excluded_seeds": list(EXCLUDED_SEEDS),
            "excluded_state_present": False,
            "values_inspected": False,
        },
    )
    destination = (
        root / f"control/excluded/FIXED5_EXCLUDED_{nonce}.json"
    )
    output = launch_receipt.publish_exclusive(
        destination, receipt, production_root=root
    )
    verify_excluded_audit(
        output,
        production_root=root,
        launch_nonce=nonce,
        launch=launch,
        state=state,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
    )
    return output


def verify_excluded_audit(
    path: Path | str,
    *,
    production_root: Path | str,
    launch_nonce: str,
    launch: Path,
    state: StudyState,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
) -> dict[str, Any]:
    nonce = launch_receipt.validate_nonce(launch_nonce)
    root = _exact_root(production_root)
    scan_excluded_state(root)
    if state.completed != ADOPTED_SEEDS or set(state.chains) != set(ADOPTED_SEEDS):
        raise ValueError("excluded audit requires all five verified seed chains")
    expected = root / f"control/excluded/FIXED5_EXCLUDED_{nonce}.json"
    source = _exact_file(path, expected, "excluded-seed audit")
    receipt = verify_receipt(
        source,
        expected_schema=EXCLUDED_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=EXCLUDED_SCENARIO,
    )
    expected_fields = {
        "status": "pass",
        "launch_nonce": nonce,
        "excluded_seeds": list(EXCLUDED_SEEDS),
        "excluded_state_present": False,
        "values_inspected": False,
    }
    for key, value in expected_fields.items():
        if receipt.get(key) != value:
            raise ValueError(f"excluded-seed audit {key} differs")
    launch_data = verify_receipt(
        launch,
        expected_schema=launch_receipt.LAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=launch_receipt.LAUNCH_SCENARIO,
    )
    try:
        prelaunch = Path(
            launch_data["identities"]["prelaunch_receipt"]["canonical_path"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("excluded audit launch ancestry is missing") from error
    launch_receipt.verify_launch(
        launch,
        launch_nonce=launch_data.get("launch_nonce"),
        slurm_job_id=launch_data.get("slurm_job_id"),
        prelaunch_receipt=prelaunch,
        production_root=root,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
    )
    chain_ids: dict[str, Any] = {}
    for seed, chain in state.chains.items():
        chain_ids[f"seed_{seed}"] = {
            role: file_identity(chain_path)
            for role, chain_path in chain.items()
        }
    expected_identities = {
        "fixed5_source_manifest": file_identity(fixed5_source_manifest),
        "adoption_authorization": file_identity(adoption_authorization),
        "launch_receipt": file_identity(launch),
        "controller": file_identity(CONTROLLER),
        "seed_chains": chain_ids,
        **_amendment_identities(),
    }
    if receipt.get("identities") != expected_identities:
        raise ValueError("excluded-seed audit identities differ")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        *expected_fields,
    }:
        raise ValueError("excluded-seed audit field topology differs")
    return receipt
