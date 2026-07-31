#!/usr/bin/env python3
"""Crash-safe, at-most-once fixed-five finalization state machine."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)

from . import (
    analyzer,
    execution_receipts,
    final_collector,
    launch_receipt as launch_module,
    verifier,
)


PRODUCTION_ROOT = launch_module.PRODUCTION_ROOT
STUDY_ID = launch_module.STUDY_ID
AMENDMENTS = final_collector.AMENDMENTS
CONTINUATION_OPTIONS = final_collector.CONTINUATION_OPTIONS
ATTEMPT_RE = re.compile(r"attempt_([0-9]{2,})")
RESUME_RE = re.compile(r"FINALIZATION_RESUME_([0-9a-f]{32,})\.json")

START_NAME = "FINALIZATION_START_RECEIPT.json"
PREDICTIONS_NAME = "fixed5_predictions.jsonl"
COLLECTION_RECEIPT_NAME = "fixed5_predictions.jsonl.receipt.json"
BARRIER_NAME = "ANALYZER_START_RECEIPT.json"
ANALYSIS_NAME = "analysis_report.json"
VERIFICATION_NAME = "independent_verification_report.json"
COMPLETE_NAME = "FINALIZATION_COMPLETE_RECEIPT.json"

START_SCHEMA = "matched-cancer-fixed5-finalization-start/v1"
RESUME_SCHEMA = "matched-cancer-fixed5-finalization-resume/v1"
BARRIER_SCHEMA = "matched-cancer-fixed5-analyzer-start/v1"
COMPLETE_SCHEMA = "matched-cancer-fixed5-finalization-complete/v1"
START_SCENARIO = "fixed5_finalization_start"
RESUME_SCENARIO = "fixed5_finalization_resume"
BARRIER_SCENARIO = "fixed5_analyzer_start"
COMPLETE_SCENARIO = "fixed5_finalization_complete"

ManifestVerifier = Callable[[Path], Mapping[str, Any]]
StateScanner = Callable[..., execution_receipts.StudyState]
ExcludedVerifier = Callable[..., Mapping[str, Any]]
Collector = Callable[..., Path]
CollectionVerifier = Callable[..., Mapping[str, Any]]
AnalyzerRunner = Callable[..., Mapping[str, Any]]
IndependentVerifier = Callable[..., Mapping[str, Any]]
LockChecker = Callable[[Path], None]
LaunchVerifier = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class Context:
    root: Path
    fixed5: Path
    adoption: Path
    fixed48: Path
    authorization: Path
    feasibility: Path
    launch: Path
    excluded: Path
    nonce: str
    job_id: str
    options: Path
    state: execution_receipts.StudyState
    launch_verifier: LaunchVerifier
    excluded_verifier: ExcludedVerifier


@dataclass(frozen=True)
class AttemptState:
    number: int
    path: Path
    names: frozenset[str]
    resumes: tuple[Path, ...]

    @property
    def has_partial_collection(self) -> bool:
        return (
            PREDICTIONS_NAME in self.names
            and COLLECTION_RECEIPT_NAME not in self.names
        )


def _default_manifest_verifier(path: Path) -> Mapping[str, Any]:
    from .source_manifest import verify_manifest

    return verify_manifest(path)


def _exact_root(production_root: Path | str) -> Path:
    candidate = Path(production_root)
    expected = PRODUCTION_ROOT
    if candidate.is_symlink() or expected.is_symlink():
        raise ValueError("production root may not be a symlink")
    canonical = expected.resolve(strict=True)
    if candidate.absolute() != canonical:
        raise ValueError("production root path differs")
    if candidate.resolve(strict=True) != canonical:
        raise ValueError("production root canonical path differs")
    return canonical


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


def _require_controller_lock(root: Path) -> None:
    """Fail unless another descriptor already holds the canonical flock."""
    lock = root / "control/serial_controller.lock"
    if lock.is_symlink() or not lock.is_file():
        raise ValueError("canonical controller lock is missing or redirected")
    descriptor = os.open(lock, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        current = lock.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
        ):
            raise ValueError("controller lock identity differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            raise ValueError("fixed-five finalizer requires controller-held lock")
    finally:
        os.close(descriptor)


def _amendment_identities() -> dict[str, dict[str, Any]]:
    return {
        f"amendment_{index:02d}": file_identity(path)
        for index, path in enumerate(AMENDMENTS, start=1)
    }


def _chain_identities(
    state: execution_receipts.StudyState,
) -> dict[str, dict[str, dict[str, Any]]]:
    if (
        state.completed != execution_receipts.ADOPTED_SEEDS
        or set(state.chains) != set(execution_receipts.ADOPTED_SEEDS)
    ):
        raise ValueError("finalization requires exactly five completed chains")
    return {
        str(seed): {
            role: file_identity(path)
            for role, path in state.chains[seed].items()
        }
        for seed in execution_receipts.ADOPTED_SEEDS
    }


def _scan_attempts(root: Path) -> tuple[AttemptState, ...]:
    parent = root / "finalization"
    launch_module._ensure_directory(root, parent)
    attempts: list[AttemptState] = []
    allowed = {
        START_NAME,
        PREDICTIONS_NAME,
        COLLECTION_RECEIPT_NAME,
        BARRIER_NAME,
        ANALYSIS_NAME,
        VERIFICATION_NAME,
        COMPLETE_NAME,
    }
    for child in parent.iterdir():
        match = ATTEMPT_RE.fullmatch(child.name)
        if match is None:
            raise ValueError(f"unexpected finalization-root entry: {child}")
        if (
            child.is_symlink()
            or not child.is_dir()
            or child.resolve(strict=True) != child
        ):
            raise ValueError(f"finalization attempt redirected: {child}")
        names: set[str] = set()
        resumes: list[Path] = []
        for artifact in child.iterdir():
            if artifact.is_symlink() or not artifact.is_file():
                raise ValueError(f"finalization artifact redirected: {artifact}")
            resume_match = RESUME_RE.fullmatch(artifact.name)
            if artifact.name not in allowed and resume_match is None:
                raise ValueError(f"unexpected finalization artifact: {artifact}")
            names.add(artifact.name)
            if resume_match is not None:
                resumes.append(artifact)
        if COMPLETE_NAME in names and not {
            START_NAME,
            PREDICTIONS_NAME,
            COLLECTION_RECEIPT_NAME,
            BARRIER_NAME,
            ANALYSIS_NAME,
            VERIFICATION_NAME,
        }.issubset(names):
            raise ValueError("final completion lacks required artifacts")
        if VERIFICATION_NAME in names and ANALYSIS_NAME not in names:
            raise ValueError("verification exists without analysis")
        if ANALYSIS_NAME in names and BARRIER_NAME not in names:
            raise ValueError("analysis exists without analyzer barrier")
        if COLLECTION_RECEIPT_NAME in names and PREDICTIONS_NAME not in names:
            raise ValueError("collection receipt exists without predictions")
        if names and START_NAME not in names:
            raise ValueError(
                "finalization artifact or RESUME exists without START"
            )
        attempts.append(
            AttemptState(
                int(match.group(1)),
                child,
                frozenset(names),
                tuple(sorted(resumes, key=lambda path: path.name)),
            )
        )
    attempts.sort(key=lambda attempt: attempt.number)
    if len({attempt.number for attempt in attempts}) != len(attempts):
        raise ValueError("duplicate finalization attempt number")
    look_attempts = [
        attempt
        for attempt in attempts
        if BARRIER_NAME in attempt.names or ANALYSIS_NAME in attempt.names
    ]
    if len(look_attempts) > 1:
        raise ValueError("multiple finalization scientific-look attempts")
    if sum(BARRIER_NAME in attempt.names for attempt in attempts) > 1:
        raise ValueError("multiple analyzer barriers")
    if sum(ANALYSIS_NAME in attempt.names for attempt in attempts) > 1:
        raise ValueError("multiple analysis reports")
    return tuple(attempts)


def _new_attempt(root: Path, attempts: tuple[AttemptState, ...]) -> AttemptState:
    number = max((attempt.number for attempt in attempts), default=0) + 1
    path = root / "finalization" / f"attempt_{number:02d}"
    path.mkdir()
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("new finalization attempt redirected")
    return AttemptState(number, path, frozenset(), ())


def _launch_data(path: Path) -> tuple[str, str]:
    receipt = verify_receipt(
        path,
        expected_schema=launch_module.LAUNCH_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=launch_module.LAUNCH_SCENARIO,
    )
    nonce = launch_module.validate_nonce(receipt.get("launch_nonce", ""))
    job_id = receipt.get("slurm_job_id")
    if not isinstance(job_id, str) or launch_module.JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError("launch job ID differs")
    return nonce, job_id


def _secure_launch_path(raw: Path | str, root: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError("launch receipt path must be absolute")
    if ".." in candidate.parts:
        raise ValueError("launch receipt path may not contain '..'")
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("launch receipt must be a regular non-symlink file")
    expected_parent = root / "control/launch"
    if candidate.parent != expected_parent:
        raise ValueError("launch receipt escaped canonical launch directory")
    if candidate.resolve(strict=True) != candidate:
        raise ValueError("launch receipt ancestry redirected")
    nonce, job_id = _launch_data(candidate)
    expected = (
        expected_parent / f"FIXED5_LAUNCH_{nonce}_JOB_{job_id}.json"
    )
    return _exact_file(candidate, expected, "launch receipt")


def _secure_excluded_path(
    raw: Path | str, root: Path, nonce: str
) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError("excluded audit path must be absolute")
    if ".." in candidate.parts:
        raise ValueError("excluded audit path may not contain '..'")
    expected = root / f"control/excluded/FIXED5_EXCLUDED_{nonce}.json"
    return _exact_file(candidate, expected, "excluded-seed audit")


def _context(
    *,
    production_root: Path | str,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    launch_receipt: Path,
    excluded_audit: Path,
    continuation_options: Path,
    state_scanner: StateScanner,
    excluded_verifier: ExcludedVerifier,
    launch_verifier: LaunchVerifier,
    manifest_verifier: ManifestVerifier,
) -> Context:
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
    launch = _secure_launch_path(launch_receipt, root)
    nonce, job_id = _launch_data(launch)
    excluded = _secure_excluded_path(excluded_audit, root, nonce)
    options_expected = CONTINUATION_OPTIONS.resolve(strict=True)
    options = _exact_file(
        continuation_options,
        options_expected,
        "continuation-options document",
    )
    manifest_verifier(fixed5)
    execution_receipts.scan_excluded_state(root)
    state = state_scanner(
        production_root=root,
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
    )
    excluded_verifier(
        excluded,
        production_root=root,
        launch_nonce=nonce,
        launch=launch,
        state=state,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    prelaunch = Path(
        verify_receipt(launch)["identities"]["prelaunch_receipt"][
            "canonical_path"
        ]
    )
    launch_verifier(
        launch,
        launch_nonce=nonce,
        slurm_job_id=job_id,
        prelaunch_receipt=prelaunch,
        production_root=root,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
    )
    _chain_identities(state)
    return Context(
        root,
        fixed5,
        adoption,
        fixed48,
        authorization,
        feasibility,
        launch,
        excluded,
        nonce,
        job_id,
        options,
        state,
        launch_verifier,
        excluded_verifier,
    )


def _runtime(
    context: Context,
    *,
    manifest_verifier: ManifestVerifier,
    excluded_verifier: ExcludedVerifier,
    launch_verifier: LaunchVerifier,
) -> None:
    manifest_verifier(context.fixed5)
    execution_receipts.scan_excluded_state(context.root)
    excluded_verifier(
        context.excluded,
        production_root=context.root,
        launch_nonce=context.nonce,
        launch=context.launch,
        state=context.state,
        fixed5_source_manifest=context.fixed5,
        adoption_authorization=context.adoption,
    )
    prelaunch = Path(
        verify_receipt(context.launch)["identities"]["prelaunch_receipt"][
            "canonical_path"
        ]
    )
    launch_verifier(
        context.launch,
        launch_nonce=context.nonce,
        slurm_job_id=context.job_id,
        prelaunch_receipt=prelaunch,
        production_root=context.root,
        fixed5_source_manifest=context.fixed5,
        adoption_authorization=context.adoption,
    )
    _chain_identities(context.state)


def _replay_launch_and_audit(
    launch: Path,
    excluded: Path,
    context: Context,
    *,
    launch_verifier: LaunchVerifier,
    excluded_verifier: ExcludedVerifier,
) -> tuple[str, str]:
    secure_launch = _secure_launch_path(launch, context.root)
    nonce, job_id = _launch_data(secure_launch)
    secure_excluded = _secure_excluded_path(excluded, context.root, nonce)
    launch_data = verify_receipt(secure_launch)
    prelaunch = Path(
        launch_data["identities"]["prelaunch_receipt"]["canonical_path"]
    )
    launch_verifier(
        secure_launch,
        launch_nonce=nonce,
        slurm_job_id=job_id,
        prelaunch_receipt=prelaunch,
        production_root=context.root,
        fixed5_source_manifest=context.fixed5,
        adoption_authorization=context.adoption,
    )
    excluded_verifier(
        secure_excluded,
        production_root=context.root,
        launch_nonce=nonce,
        launch=secure_launch,
        state=context.state,
        fixed5_source_manifest=context.fixed5,
        adoption_authorization=context.adoption,
    )
    return nonce, job_id


def _origin_identities(context: Context) -> dict[str, Any]:
    return {
        "fixed5_source_manifest": file_identity(context.fixed5),
        "adoption_authorization": file_identity(context.adoption),
        "originating_launch": file_identity(context.launch),
        "originating_excluded_audit": file_identity(context.excluded),
        "continuation_options": file_identity(context.options),
        "seed_chains": _chain_identities(context.state),
        "collector": file_identity(final_collector.__file__),
        "analyzer": file_identity(analyzer.__file__),
        "independent_verifier": file_identity(verifier.__file__),
        **_amendment_identities(),
    }


def _publish_start(attempt: AttemptState, context: Context) -> Path:
    receipt = build_receipt(
        schema=START_SCHEMA,
        study_id=STUDY_ID,
        scenario=START_SCENARIO,
        identities=_origin_identities(context),
        fields={
            "status": "started",
            "attempt_name": attempt.path.name,
            "originating_launch_nonce": context.nonce,
            "originating_slurm_job_id": context.job_id,
            "values_inspected": False,
        },
    )
    return launch_module.publish_exclusive(
        attempt.path / START_NAME, receipt, production_root=context.root
    )


def _verify_start(attempt: AttemptState, context: Context) -> dict[str, Any]:
    start = attempt.path / START_NAME
    receipt = verify_receipt(
        start,
        expected_schema=START_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=START_SCENARIO,
    )
    if (
        receipt.get("status") != "started"
        or receipt.get("attempt_name") != attempt.path.name
        or receipt.get("values_inspected") is not False
    ):
        raise ValueError("finalization START semantics differ")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        "status",
        "attempt_name",
        "originating_launch_nonce",
        "originating_slurm_job_id",
        "values_inspected",
    }:
        raise ValueError("finalization START field topology differs")
    identities = receipt.get("identities", {})
    try:
        origin_launch = Path(
            identities["originating_launch"]["canonical_path"]
        )
        origin_excluded = Path(
            identities["originating_excluded_audit"]["canonical_path"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("finalization START launch ancestry is missing") from error
    origin_nonce, origin_job = _replay_launch_and_audit(
        origin_launch,
        origin_excluded,
        context,
        launch_verifier=context.launch_verifier,
        excluded_verifier=context.excluded_verifier,
    )
    if receipt.get("originating_launch_nonce") != origin_nonce:
        raise ValueError("finalization START launch nonce differs")
    if receipt.get("originating_slurm_job_id") != origin_job:
        raise ValueError("finalization START job ID differs")
    expected_static = {
        "fixed5_source_manifest": file_identity(context.fixed5),
        "adoption_authorization": file_identity(context.adoption),
        "continuation_options": file_identity(context.options),
        "seed_chains": _chain_identities(context.state),
        "collector": file_identity(final_collector.__file__),
        "analyzer": file_identity(analyzer.__file__),
        "independent_verifier": file_identity(verifier.__file__),
        **_amendment_identities(),
    }
    expected_identities = {
        **expected_static,
        "originating_launch": file_identity(origin_launch),
        "originating_excluded_audit": file_identity(origin_excluded),
    }
    if identities != expected_identities:
        raise ValueError("finalization START identities differ")
    return receipt


def _resume_identities(
    attempt: AttemptState,
    context: Context,
    ordered_resumes: tuple[Path, ...],
    historical_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    earlier = {
        f"resume_{index:04d}_{RESUME_RE.fullmatch(path.name).group(1)}": (
            file_identity(path)
        )
        for index, path in enumerate(ordered_resumes, start=1)
    }
    start = verify_receipt(
        attempt.path / START_NAME,
        expected_schema=START_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=START_SCENARIO,
    )
    identities = {
        "finalization_start": file_identity(attempt.path / START_NAME),
        "originating_launch": start["identities"]["originating_launch"],
        "originating_excluded_audit": start["identities"][
            "originating_excluded_audit"
        ],
        "current_recovery_launch": file_identity(context.launch),
        "current_recovery_excluded_audit": file_identity(context.excluded),
        "existing_finalization_artifacts": dict(historical_artifacts),
        "fixed5_source_manifest": file_identity(context.fixed5),
        **_amendment_identities(),
    }
    if earlier:
        identities["earlier_resumes"] = earlier
    return identities


PHASE_ARTIFACTS = {
    "start_only": frozenset({START_NAME}),
    "collection_sealed": frozenset({
        START_NAME,
        PREDICTIONS_NAME,
        COLLECTION_RECEIPT_NAME,
    }),
    "analysis_sealed": frozenset({
        START_NAME,
        PREDICTIONS_NAME,
        COLLECTION_RECEIPT_NAME,
        BARRIER_NAME,
        ANALYSIS_NAME,
    }),
    "verification_sealed": frozenset({
        START_NAME,
        PREDICTIONS_NAME,
        COLLECTION_RECEIPT_NAME,
        BARRIER_NAME,
        ANALYSIS_NAME,
        VERIFICATION_NAME,
    }),
    "complete_sealed": frozenset({
        START_NAME,
        PREDICTIONS_NAME,
        COLLECTION_RECEIPT_NAME,
        BARRIER_NAME,
        ANALYSIS_NAME,
        VERIFICATION_NAME,
        COMPLETE_NAME,
    }),
}


def _phase_for_artifacts(names: frozenset[str]) -> str:
    for phase, expected in PHASE_ARTIFACTS.items():
        if names == expected:
            return phase
    raise ValueError("recovery artifact snapshot has no canonical phase")


def _identity_index(
    identity: Mapping[str, Any],
    ancestry: list[dict[str, Any]],
    *,
    label: str,
) -> int:
    matches = [
        index
        for index, ancestor in enumerate(ancestry)
        if ancestor == dict(identity)
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} launch ancestry is missing or ambiguous")
    return matches[0]


def _historical_artifacts(
    attempt: AttemptState,
    *,
    recovery_sequence: int,
    ordered_resumes: tuple[Path, ...],
    prospective_launch: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reconstruct artifacts sealed strictly before one recovery launch."""
    start = verify_receipt(attempt.path / START_NAME)
    ancestry = [dict(start["identities"]["originating_launch"])]
    for resume_path in ordered_resumes:
        resume = verify_receipt(
            resume_path,
            expected_schema=RESUME_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=RESUME_SCENARIO,
        )
        ancestry.append(
            dict(resume["identities"]["current_recovery_launch"])
        )
    if prospective_launch is not None:
        ancestry.append(dict(prospective_launch))
    if len(ancestry) <= recovery_sequence:
        raise ValueError("recovery launch sequence is incomplete")
    if len({identity["sha256"] for identity in ancestry}) != len(ancestry):
        raise ValueError("recovery launch ancestry contains a duplicate")

    publication: dict[str, int] = {START_NAME: 0}
    collection_receipt = attempt.path / COLLECTION_RECEIPT_NAME
    predictions = attempt.path / PREDICTIONS_NAME
    if collection_receipt.exists() or predictions.exists():
        if (
            collection_receipt.is_symlink()
            or predictions.is_symlink()
            or not collection_receipt.is_file()
            or not predictions.is_file()
        ):
            raise ValueError("collection snapshot is partial or redirected")
        collection = verify_receipt(collection_receipt)
        collection_index = _identity_index(
            collection["identities"]["launch_receipt"],
            ancestry,
            label="collection",
        )
        publication[COLLECTION_RECEIPT_NAME] = collection_index
        publication[PREDICTIONS_NAME] = collection_index

    barrier_path = attempt.path / BARRIER_NAME
    analysis_path = attempt.path / ANALYSIS_NAME
    if barrier_path.exists() or analysis_path.exists():
        if (
            barrier_path.is_symlink()
            or not barrier_path.is_file()
        ):
            raise ValueError("analyzer barrier snapshot is redirected")
        barrier = verify_receipt(
            barrier_path,
            expected_schema=BARRIER_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=BARRIER_SCENARIO,
        )
        barrier_index = _identity_index(
            barrier["identities"]["current_launch"],
            ancestry,
            label="analyzer barrier",
        )
        if analysis_path.exists():
            if analysis_path.is_symlink() or not analysis_path.is_file():
                raise ValueError("analysis snapshot is redirected")
            publication[BARRIER_NAME] = barrier_index
            publication[ANALYSIS_NAME] = barrier_index
        elif barrier_index < recovery_sequence:
            raise ValueError(
                "pre-existing analyzer barrier lacks analysis"
            )

    verification_path = attempt.path / VERIFICATION_NAME
    if verification_path.exists():
        if verification_path.is_symlink() or not verification_path.is_file():
            raise ValueError("verification snapshot is redirected")
        verification = _load_unique_json(verification_path)
        try:
            verification_launch = verification[
                "finalization_provenance"
            ]["current_launch"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "verification snapshot lacks launch ancestry"
            ) from error
        publication[VERIFICATION_NAME] = _identity_index(
            verification_launch,
            ancestry,
            label="independent verification",
        )

    complete_path = attempt.path / COMPLETE_NAME
    if complete_path.exists():
        if complete_path.is_symlink() or not complete_path.is_file():
            raise ValueError("completion snapshot is redirected")
        complete = verify_receipt(
            complete_path,
            expected_schema=COMPLETE_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=COMPLETE_SCENARIO,
        )
        publication[COMPLETE_NAME] = _identity_index(
            complete["identities"]["current_launch"],
            ancestry,
            label="completion",
        )

    names = frozenset(
        name
        for name, launch_index in publication.items()
        if launch_index < recovery_sequence
    )
    _phase_for_artifacts(names)
    return {
        name: file_identity(attempt.path / name)
        for name in names
    }


def _publish_resume(
    attempt: AttemptState,
    context: Context,
) -> Path:
    ordered = _verify_resumes(
        attempt,
        context,
    )
    sequence = len(ordered) + 1
    historical = _historical_artifacts(
        attempt,
        recovery_sequence=sequence,
        ordered_resumes=ordered,
        prospective_launch=file_identity(context.launch),
    )
    phase = _phase_for_artifacts(frozenset(historical))
    destination = (
        attempt.path / f"FINALIZATION_RESUME_{context.nonce}.json"
    )
    receipt = build_receipt(
        schema=RESUME_SCHEMA,
        study_id=STUDY_ID,
        scenario=RESUME_SCENARIO,
        identities=_resume_identities(
            attempt, context, ordered, historical
        ),
        fields={
            "status": "authorized",
            "attempt_name": attempt.path.name,
            "recovery_launch_nonce": context.nonce,
            "recovery_slurm_job_id": context.job_id,
            "resume_sequence": sequence,
            "earlier_resume_count": len(ordered),
            "recovery_phase": phase,
            "values_inspected": False,
        },
    )
    return launch_module.publish_exclusive(
        destination, receipt, production_root=context.root
    )


def _ordered_resume_paths(attempt: AttemptState) -> tuple[Path, ...]:
    indexed: dict[int, Path] = {}
    for path in attempt.resumes:
        receipt = verify_receipt(
            path,
            expected_schema=RESUME_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=RESUME_SCENARIO,
        )
        sequence = receipt.get("resume_sequence")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 1
            or sequence in indexed
        ):
            raise ValueError("finalization RESUME sequence is invalid")
        indexed[sequence] = path
    if set(indexed) != set(range(1, len(indexed) + 1)):
        raise ValueError("finalization RESUME sequence is not contiguous")
    return tuple(indexed[index] for index in range(1, len(indexed) + 1))


def _verify_resumes(
    attempt: AttemptState,
    context: Context,
) -> tuple[Path, ...]:
    earlier: dict[str, dict[str, Any]] = {}
    previous_artifacts: set[str] = set()
    ordered = _ordered_resume_paths(attempt)
    for sequence, path in enumerate(ordered, start=1):
        match = RESUME_RE.fullmatch(path.name)
        assert match is not None
        receipt = verify_receipt(
            path,
            expected_schema=RESUME_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=RESUME_SCENARIO,
        )
        if (
            receipt.get("status") != "authorized"
            or receipt.get("attempt_name") != attempt.path.name
            or receipt.get("recovery_launch_nonce") != match.group(1)
            or receipt.get("resume_sequence") != sequence
            or receipt.get("earlier_resume_count") != len(earlier)
            or receipt.get("recovery_phase") not in PHASE_ARTIFACTS
            or receipt.get("values_inspected") is not False
        ):
            raise ValueError("finalization RESUME semantics differ")
        if set(receipt) != {
            "schema",
            "study_id",
            "scenario",
            "identities",
            "topology_sha256",
            "status",
            "attempt_name",
            "recovery_launch_nonce",
            "recovery_slurm_job_id",
            "resume_sequence",
            "earlier_resume_count",
            "recovery_phase",
            "values_inspected",
        }:
            raise ValueError("finalization RESUME field topology differs")
        if receipt["identities"].get("earlier_resumes", {}) != earlier:
            raise ValueError("finalization RESUME chain differs")
        identities = receipt["identities"]
        required_identity_roles = {
            "finalization_start",
            "originating_launch",
            "originating_excluded_audit",
            "current_recovery_launch",
            "current_recovery_excluded_audit",
            "existing_finalization_artifacts",
            "fixed5_source_manifest",
            *(f"amendment_{index:02d}" for index in range(1, 9)),
        }
        if earlier:
            required_identity_roles.add("earlier_resumes")
        if set(identities) != required_identity_roles:
            raise ValueError("finalization RESUME identity topology differs")
        start_receipt = verify_receipt(attempt.path / START_NAME)
        expected_static = {
            "finalization_start": file_identity(
                attempt.path / START_NAME
            ),
            "originating_launch": start_receipt["identities"][
                "originating_launch"
            ],
            "originating_excluded_audit": start_receipt["identities"][
                "originating_excluded_audit"
            ],
            "fixed5_source_manifest": file_identity(context.fixed5),
            **_amendment_identities(),
        }
        for role, identity in expected_static.items():
            if identities.get(role) != identity:
                raise ValueError(
                    f"finalization RESUME {role} differs"
                )
        recorded_artifacts = identities.get(
            "existing_finalization_artifacts"
        )
        phase = receipt["recovery_phase"]
        expected_artifacts = _historical_artifacts(
            attempt,
            recovery_sequence=sequence,
            ordered_resumes=ordered,
        )
        if (
            not isinstance(recorded_artifacts, Mapping)
            or dict(recorded_artifacts) != expected_artifacts
            or phase
            != _phase_for_artifacts(frozenset(expected_artifacts))
            or not previous_artifacts.issubset(recorded_artifacts)
        ):
            raise ValueError(
                "finalization RESUME existing artifacts differ"
            )
        previous_artifacts = set(recorded_artifacts)
        recovery_launch = Path(
            receipt["identities"]["current_recovery_launch"][
                "canonical_path"
            ]
        )
        recovery_excluded = Path(
            receipt["identities"]["current_recovery_excluded_audit"][
                "canonical_path"
            ]
        )
        launch_nonce, job_id = _replay_launch_and_audit(
            recovery_launch,
            recovery_excluded,
            context,
            launch_verifier=context.launch_verifier,
            excluded_verifier=context.excluded_verifier,
        )
        if (
            launch_nonce != match.group(1)
            or job_id != receipt.get("recovery_slurm_job_id")
            or file_identity(recovery_launch)
            != identities["current_recovery_launch"]
            or file_identity(recovery_excluded)
            != identities["current_recovery_excluded_audit"]
        ):
            raise ValueError("finalization RESUME launch identity differs")
        earlier[
            f"resume_{sequence:04d}_{match.group(1)}"
        ] = file_identity(path)
    return ordered


def _current_ancestry(
    attempt: AttemptState,
    context: Context,
) -> dict[str, Any]:
    ancestry = {
        "finalization_start": file_identity(attempt.path / START_NAME),
    }
    resumes = {
        f"resume_{index:04d}_{RESUME_RE.fullmatch(path.name).group(1)}": (
            file_identity(path)
        )
        for index, path in enumerate(
            _verify_resumes(
                attempt,
                context,
            ),
            start=1,
        )
    }
    if resumes:
        ancestry["resumes"] = resumes
    return ancestry


def _publish_barrier(attempt: AttemptState, context: Context) -> Path:
    receipt = build_receipt(
        schema=BARRIER_SCHEMA,
        study_id=STUDY_ID,
        scenario=BARRIER_SCENARIO,
        identities={
            **_current_ancestry(attempt, context),
            "raw_matrix": file_identity(attempt.path / PREDICTIONS_NAME),
            "collection_receipt": file_identity(
                attempt.path / COLLECTION_RECEIPT_NAME
            ),
            "fixed5_source_manifest": file_identity(context.fixed5),
            "analyzer": file_identity(analyzer.__file__),
            "current_launch": file_identity(context.launch),
            "current_excluded_audit": file_identity(context.excluded),
            "continuation_options": file_identity(context.options),
            **_amendment_identities(),
        },
        fields={
            "status": "started",
            "attempt_name": attempt.path.name,
            "launch_nonce": context.nonce,
            "slurm_job_id": context.job_id,
            "scientific_values_opened": False,
            "analyzer_invocations_before": 0,
        },
    )
    return launch_module.publish_exclusive(
        attempt.path / BARRIER_NAME,
        receipt,
        production_root=context.root,
    )


def _verify_barrier(
    attempt: AttemptState, context: Context
) -> Mapping[str, Any]:
    receipt = verify_receipt(
        attempt.path / BARRIER_NAME,
        expected_schema=BARRIER_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=BARRIER_SCENARIO,
    )
    if (
        receipt.get("status") != "started"
        or receipt.get("attempt_name") != attempt.path.name
        or receipt.get("scientific_values_opened") is not False
        or receipt.get("analyzer_invocations_before") != 0
    ):
        raise ValueError("analyzer barrier semantics differ")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        "status",
        "attempt_name",
        "launch_nonce",
        "slurm_job_id",
        "scientific_values_opened",
        "analyzer_invocations_before",
    }:
        raise ValueError("analyzer barrier field topology differs")
    identities = receipt["identities"]
    expected = {
        "finalization_start": file_identity(attempt.path / START_NAME),
        "raw_matrix": file_identity(attempt.path / PREDICTIONS_NAME),
        "collection_receipt": file_identity(
            attempt.path / COLLECTION_RECEIPT_NAME
        ),
        "fixed5_source_manifest": file_identity(context.fixed5),
        "analyzer": file_identity(analyzer.__file__),
        "continuation_options": file_identity(context.options),
        **_amendment_identities(),
    }
    for role, identity in expected.items():
        if identities.get(role) != identity:
            raise ValueError(f"analyzer barrier {role} differs")
    recorded_resumes = identities.get("resumes", {})
    current_resumes = _current_ancestry(attempt, context).get(
        "resumes", {}
    )
    if not isinstance(recorded_resumes, Mapping) or any(
        current_resumes.get(name) != identity
        for name, identity in recorded_resumes.items()
    ):
        raise ValueError("analyzer barrier RESUME ancestry differs")
    current_launch = Path(identities["current_launch"]["canonical_path"])
    current_excluded = Path(
        identities["current_excluded_audit"]["canonical_path"]
    )
    nonce, job_id = _replay_launch_and_audit(
        current_launch,
        current_excluded,
        context,
        launch_verifier=context.launch_verifier,
        excluded_verifier=context.excluded_verifier,
    )
    if (
        receipt.get("launch_nonce") != nonce
        or receipt.get("slurm_job_id") != job_id
    ):
        raise ValueError("analyzer barrier launch differs")
    if recorded_resumes:
        last_identity = next(reversed(recorded_resumes.values()))
        last_resume = verify_receipt(
            Path(last_identity["canonical_path"]),
            expected_schema=RESUME_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=RESUME_SCENARIO,
        )
        expected_cutoff_launch = last_resume["identities"][
            "current_recovery_launch"
        ]
        expected_cutoff_excluded = last_resume["identities"][
            "current_recovery_excluded_audit"
        ]
    else:
        start = verify_receipt(attempt.path / START_NAME)
        expected_cutoff_launch = start["identities"][
            "originating_launch"
        ]
        expected_cutoff_excluded = start["identities"][
            "originating_excluded_audit"
        ]
    if (
        identities["current_launch"] != expected_cutoff_launch
        or identities["current_excluded_audit"]
        != expected_cutoff_excluded
    ):
        raise ValueError("analyzer barrier cutoff launch ancestry differs")
    expected_identities = {
        **expected,
        "current_launch": file_identity(current_launch),
        "current_excluded_audit": file_identity(current_excluded),
    }
    if recorded_resumes:
        expected_identities["resumes"] = dict(recorded_resumes)
    if identities != expected_identities:
        raise ValueError("analyzer barrier identities differ")
    return receipt


def _barrier_cutoff(
    attempt: AttemptState, context: Context
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the immutable pre-look RESUME, launch, and audit identities."""
    barrier = _verify_barrier(attempt, context)
    identities = barrier["identities"]
    resumes = identities.get("resumes", {})
    if not isinstance(resumes, Mapping):
        raise ValueError("analyzer barrier RESUME cutoff differs")
    return (
        dict(resumes),
        dict(identities["current_launch"]),
        dict(identities["current_excluded_audit"]),
    )


def _is_ordered_prefix(
    prefix: Mapping[str, Any], extension: Mapping[str, Any]
) -> bool:
    prefix_items = list(prefix.items())
    extension_items = list(extension.items())
    return extension_items[: len(prefix_items)] == prefix_items


def _resume_endpoint(
    attempt: AttemptState,
    resumes: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if resumes:
        last_identity = list(resumes.values())[-1]
        resume = verify_receipt(
            Path(last_identity["canonical_path"]),
            expected_schema=RESUME_SCHEMA,
            expected_study_id=STUDY_ID,
            expected_scenario=RESUME_SCENARIO,
        )
        return (
            dict(resume["identities"]["current_recovery_launch"]),
            dict(
                resume["identities"]["current_recovery_excluded_audit"]
            ),
        )
    start = verify_receipt(attempt.path / START_NAME)
    return (
        dict(start["identities"]["originating_launch"]),
        dict(start["identities"]["originating_excluded_audit"]),
    )


def _write_json_exclusive(
    path: Path, value: Mapping[str, Any], *, root: Path
) -> Path:
    payload = canonical_json_bytes(value) + b"\n"
    if os.path.lexists(path):
        raise FileExistsError(f"final artifact already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o664)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("final artifact redirected")
    return path


def _load_unique_json(path: Path) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"final report missing or redirected: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique)
    if not isinstance(value, Mapping):
        raise ValueError("final report root must be an object")
    return value


def _verify_analysis_envelope(path: Path) -> Mapping[str, Any]:
    value = _load_unique_json(path)
    if set(value) != {"schema", "semantic_report"}:
        raise ValueError("analysis report envelope differs")
    if value["schema"] != analyzer.REPORT_SCHEMA:
        raise ValueError("analysis report schema differs")
    return value


def _verification_provenance(
    attempt: AttemptState, context: Context
) -> dict[str, Any]:
    return {
        **_current_ancestry(attempt, context),
        "raw_matrix": file_identity(attempt.path / PREDICTIONS_NAME),
        "collection_receipt": file_identity(
            attempt.path / COLLECTION_RECEIPT_NAME
        ),
        "analyzer_barrier": file_identity(attempt.path / BARRIER_NAME),
        "analysis_report": file_identity(attempt.path / ANALYSIS_NAME),
        "seed_chains": _chain_identities(context.state),
        "current_launch": file_identity(context.launch),
        "current_excluded_audit": file_identity(context.excluded),
        "fixed5_source_manifest": file_identity(context.fixed5),
        "adoption_authorization": file_identity(context.adoption),
        "continuation_options": file_identity(context.options),
        **_amendment_identities(),
    }


def _verify_verification_envelope(
    path: Path, attempt: AttemptState, context: Context
) -> Mapping[str, Any]:
    value = _load_unique_json(path)
    if set(value) != {
        "schema",
        "semantic_report",
        "analyzer_comparison",
        "verification_provenance",
        "finalization_provenance",
    }:
        raise ValueError("independent-verification field topology differs")
    if value.get("schema") != verifier.REPORT_SCHEMA:
        raise ValueError("independent-verification schema differs")
    comparison = value.get("analyzer_comparison")
    if (
        not isinstance(comparison, Mapping)
        or comparison.get("requested") is not True
        or comparison.get("match") is not True
    ):
        raise ValueError("independent verifier did not match analyzer")
    expected_verifier_provenance = {
        "source_manifest": file_identity(context.fixed5),
        "collection_receipt": file_identity(
            attempt.path / COLLECTION_RECEIPT_NAME
        ),
        "collected_predictions": file_identity(
            attempt.path / PREDICTIONS_NAME
        ),
        "analyzer_report": file_identity(attempt.path / ANALYSIS_NAME),
        "independent_verifier": file_identity(verifier.__file__),
    }
    if value.get("verification_provenance") != expected_verifier_provenance:
        raise ValueError("independent-verification provenance differs")
    finalization = value.get("finalization_provenance")
    if not isinstance(finalization, Mapping):
        raise ValueError(
            "independent-verification finalization ancestry is missing"
        )
    expected = _verification_provenance(attempt, context)
    for role in (
        "finalization_start",
        "raw_matrix",
        "collection_receipt",
        "analyzer_barrier",
        "analysis_report",
        "seed_chains",
        "fixed5_source_manifest",
        "adoption_authorization",
        "continuation_options",
        *(f"amendment_{index:02d}" for index in range(1, 9)),
    ):
        if finalization.get(role) != expected[role]:
            raise ValueError(
                "independent-verification finalization ancestry differs"
            )
    recorded_resumes = finalization.get("resumes", {})
    current_resumes = expected.get("resumes", {})
    barrier_resumes, _barrier_launch, _barrier_excluded = _barrier_cutoff(
        attempt, context
    )
    if (
        not isinstance(recorded_resumes, Mapping)
        or not isinstance(current_resumes, Mapping)
        or not _is_ordered_prefix(barrier_resumes, recorded_resumes)
        or not _is_ordered_prefix(recorded_resumes, current_resumes)
    ):
        raise ValueError(
            "independent-verification RESUME ancestry differs"
        )
    required_finalization_roles = {
        "finalization_start",
        "raw_matrix",
        "collection_receipt",
        "analyzer_barrier",
        "analysis_report",
        "seed_chains",
        "current_launch",
        "current_excluded_audit",
        "fixed5_source_manifest",
        "adoption_authorization",
        "continuation_options",
        *(f"amendment_{index:02d}" for index in range(1, 9)),
    }
    if recorded_resumes:
        required_finalization_roles.add("resumes")
    if set(finalization) != required_finalization_roles:
        raise ValueError(
            "independent-verification finalization topology differs"
        )
    verification_launch, verification_excluded = _resume_endpoint(
        attempt, recorded_resumes
    )
    if (
        finalization.get("current_launch") != verification_launch
        or finalization.get("current_excluded_audit")
        != verification_excluded
    ):
        raise ValueError(
            "independent-verification cutoff launch ancestry differs"
        )
    recorded_launch = Path(
        finalization["current_launch"]["canonical_path"]
    )
    recorded_excluded = Path(
        finalization["current_excluded_audit"]["canonical_path"]
    )
    _replay_launch_and_audit(
        recorded_launch,
        recorded_excluded,
        context,
        launch_verifier=context.launch_verifier,
        excluded_verifier=context.excluded_verifier,
    )
    if (
        file_identity(recorded_launch) != finalization["current_launch"]
        or file_identity(recorded_excluded)
        != finalization["current_excluded_audit"]
    ):
        raise ValueError(
            "independent-verification excluded ancestry differs"
        )
    return value


def _publish_complete(attempt: AttemptState, context: Context) -> Path:
    receipt = build_receipt(
        schema=COMPLETE_SCHEMA,
        study_id=STUDY_ID,
        scenario=COMPLETE_SCENARIO,
        identities={
            **_current_ancestry(attempt, context),
            "raw_matrix": file_identity(attempt.path / PREDICTIONS_NAME),
            "collection_receipt": file_identity(
                attempt.path / COLLECTION_RECEIPT_NAME
            ),
            "analyzer_barrier": file_identity(attempt.path / BARRIER_NAME),
            "analysis_report": file_identity(attempt.path / ANALYSIS_NAME),
            "independent_verification_report": file_identity(
                attempt.path / VERIFICATION_NAME
            ),
            "seed_chains": _chain_identities(context.state),
            "current_launch": file_identity(context.launch),
            "current_excluded_audit": file_identity(context.excluded),
            "fixed5_source_manifest": file_identity(context.fixed5),
            "adoption_authorization": file_identity(context.adoption),
            "continuation_options": file_identity(context.options),
            **_amendment_identities(),
        },
        fields={
            "status": "complete",
            "attempt_name": attempt.path.name,
            "fm_seeds": list(execution_receipts.ADOPTED_SEEDS),
            "row_count": final_collector.EXPECTED_ROWS,
            "combination_count": final_collector.EXPECTED_COMBINATIONS,
            "values_inspected": True,
            "analyzer_invocation_count": 1,
            "independent_verification_passed": True,
            "excluded_state_absent": True,
        },
    )
    return launch_module.publish_exclusive(
        attempt.path / COMPLETE_NAME,
        receipt,
        production_root=context.root,
    )


def _verify_complete(
    attempt: AttemptState,
    context: Context,
    *,
    collection_verifier: CollectionVerifier = (
        final_collector.verify_final_collection
    ),
) -> Mapping[str, Any]:
    collection_verifier(
        attempt.path / PREDICTIONS_NAME,
        receipt_path=attempt.path / COLLECTION_RECEIPT_NAME,
        source_manifest=context.fixed5,
    )
    _verify_analysis_envelope(attempt.path / ANALYSIS_NAME)
    _verify_barrier(attempt, context)
    _verify_verification_envelope(
        attempt.path / VERIFICATION_NAME, attempt, context
    )
    receipt = verify_receipt(
        attempt.path / COMPLETE_NAME,
        expected_schema=COMPLETE_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=COMPLETE_SCENARIO,
    )
    if (
        receipt.get("status") != "complete"
        or receipt.get("attempt_name") != attempt.path.name
        or receipt.get("fm_seeds")
        != list(execution_receipts.ADOPTED_SEEDS)
        or receipt.get("row_count") != final_collector.EXPECTED_ROWS
        or receipt.get("combination_count")
        != final_collector.EXPECTED_COMBINATIONS
        or receipt.get("values_inspected") is not True
        or receipt.get("analyzer_invocation_count") != 1
        or receipt.get("independent_verification_passed") is not True
        or receipt.get("excluded_state_absent") is not True
    ):
        raise ValueError("finalization COMPLETE semantics differ")
    if set(receipt) != {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        "status",
        "attempt_name",
        "fm_seeds",
        "row_count",
        "combination_count",
        "values_inspected",
        "analyzer_invocation_count",
        "independent_verification_passed",
        "excluded_state_absent",
    }:
        raise ValueError("finalization COMPLETE field topology differs")
    recorded_identities = receipt["identities"]
    recorded_resumes = recorded_identities.get("resumes", {})
    current_resumes = _current_ancestry(attempt, context).get(
        "resumes", {}
    )
    verification_report = _load_unique_json(
        attempt.path / VERIFICATION_NAME
    )
    verification_resumes = verification_report[
        "finalization_provenance"
    ].get("resumes", {})
    if (
        not isinstance(recorded_resumes, Mapping)
        or not isinstance(verification_resumes, Mapping)
        or not _is_ordered_prefix(
            verification_resumes, recorded_resumes
        )
        or not _is_ordered_prefix(recorded_resumes, current_resumes)
    ):
        raise ValueError("finalization COMPLETE RESUME ancestry differs")
    recorded_launch = Path(
        recorded_identities["current_launch"]["canonical_path"]
    )
    recorded_excluded = Path(
        recorded_identities["current_excluded_audit"]["canonical_path"]
    )
    _replay_launch_and_audit(
        recorded_launch,
        recorded_excluded,
        context,
        launch_verifier=context.launch_verifier,
        excluded_verifier=context.excluded_verifier,
    )
    completion_launch, completion_excluded = _resume_endpoint(
        attempt, recorded_resumes
    )
    if (
        recorded_identities["current_launch"] != completion_launch
        or recorded_identities["current_excluded_audit"]
        != completion_excluded
    ):
        raise ValueError(
            "finalization COMPLETE cutoff launch ancestry differs"
        )
    expected = {
        "finalization_start": file_identity(attempt.path / START_NAME),
        "raw_matrix": file_identity(attempt.path / PREDICTIONS_NAME),
        "collection_receipt": file_identity(
            attempt.path / COLLECTION_RECEIPT_NAME
        ),
        "analyzer_barrier": file_identity(attempt.path / BARRIER_NAME),
        "analysis_report": file_identity(attempt.path / ANALYSIS_NAME),
        "independent_verification_report": file_identity(
            attempt.path / VERIFICATION_NAME
        ),
        "seed_chains": _chain_identities(context.state),
        "current_launch": file_identity(recorded_launch),
        "current_excluded_audit": file_identity(recorded_excluded),
        "fixed5_source_manifest": file_identity(context.fixed5),
        "adoption_authorization": file_identity(context.adoption),
        "continuation_options": file_identity(context.options),
        **_amendment_identities(),
    }
    if recorded_resumes:
        expected["resumes"] = dict(recorded_resumes)
    if recorded_identities != expected:
        raise ValueError("finalization COMPLETE identities differ")
    return receipt


def run(
    *,
    production_root: Path | str,
    fixed5_source_manifest: Path,
    adoption_authorization: Path,
    fixed48_source_manifest: Path,
    authorization_manifest: Path,
    feasibility_gate: Path,
    launch_receipt: Path,
    excluded_audit: Path,
    environment: Mapping[str, str] | None = None,
    continuation_options: Path = CONTINUATION_OPTIONS,
    manifest_verifier: ManifestVerifier = _default_manifest_verifier,
    state_scanner: StateScanner = execution_receipts.scan_state,
    excluded_verifier: ExcludedVerifier = execution_receipts.verify_excluded_audit,
    collector: Collector = final_collector.collect,
    collection_verifier: CollectionVerifier = (
        final_collector.verify_final_collection
    ),
    analyzer_runner: AnalyzerRunner = analyzer.run_sealed,
    independent_verifier: IndependentVerifier = verifier.run_sealed,
    lock_checker: LockChecker = _require_controller_lock,
    launch_verifier: LaunchVerifier = launch_module.verify_launch,
) -> Path:
    """Advance or recover the sole finalization attempt under controller lock."""
    del environment  # Launch identity is receipt-bound; controller checks env.
    context = _context(
        production_root=production_root,
        fixed5_source_manifest=fixed5_source_manifest,
        adoption_authorization=adoption_authorization,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        launch_receipt=launch_receipt,
        excluded_audit=excluded_audit,
        continuation_options=continuation_options,
        state_scanner=state_scanner,
        excluded_verifier=excluded_verifier,
        launch_verifier=launch_verifier,
        manifest_verifier=manifest_verifier,
    )
    lock_checker(context.root)
    _runtime(
        context,
        manifest_verifier=manifest_verifier,
        excluded_verifier=excluded_verifier,
        launch_verifier=launch_verifier,
    )
    attempts = _scan_attempts(context.root)
    complete = [attempt for attempt in attempts if COMPLETE_NAME in attempt.names]
    if complete:
        if len(complete) != 1 or complete[0] != attempts[-1]:
            raise ValueError("invalid state after final completion")
        attempt = complete[0]
        _verify_start(attempt, context)
        known_nonces = {
            RESUME_RE.fullmatch(path.name).group(1)  # type: ignore[union-attr]
            for path in _ordered_resume_paths(attempt)
        }
        origin = verify_receipt(
            attempt.path / START_NAME
        ).get("originating_launch_nonce")
        if context.nonce != origin and context.nonce not in known_nonces:
            _publish_resume(attempt, context)
            attempt = next(
                item
                for item in _scan_attempts(context.root)
                if item.number == attempt.number
            )
        _verify_complete(
            attempt,
            context,
            collection_verifier=collection_verifier,
        )
        recomputed = dict(independent_verifier(
            attempt.path / PREDICTIONS_NAME,
            analyzer_report=attempt.path / ANALYSIS_NAME,
            collection_receipt=attempt.path / COLLECTION_RECEIPT_NAME,
            source_manifest=context.fixed5,
        ))
        stored = _load_unique_json(attempt.path / VERIFICATION_NAME)
        for role in (
            "schema",
            "semantic_report",
            "analyzer_comparison",
            "verification_provenance",
        ):
            if recomputed.get(role) != stored.get(role):
                raise ValueError(
                    "completed-state independent recomputation differs"
                )
        _runtime(
            context,
            manifest_verifier=manifest_verifier,
            excluded_verifier=excluded_verifier,
            launch_verifier=launch_verifier,
        )
        return attempt.path / COMPLETE_NAME

    looked = [
        attempt for attempt in attempts if BARRIER_NAME in attempt.names
    ]
    reusable = [
        attempt
        for attempt in attempts
        if START_NAME in attempt.names and not attempt.has_partial_collection
    ]
    if looked:
        attempt = looked[0]
    elif reusable:
        if len(reusable) != 1 or any(
            candidate.number > reusable[0].number for candidate in attempts
        ):
            raise ValueError("multiple or nonterminal reusable finalizations")
        attempt = reusable[0]
    else:
        attempt = _new_attempt(context.root, attempts)
        _publish_start(attempt, context)
        attempt = _scan_attempts(context.root)[-1]

    _verify_start(attempt, context)
    start = verify_receipt(attempt.path / START_NAME)
    origin_nonce = start.get("originating_launch_nonce")
    resume_nonces = {
        RESUME_RE.fullmatch(path.name).group(1)  # type: ignore[union-attr]
        for path in attempt.resumes
    }
    if context.nonce != origin_nonce and context.nonce not in resume_nonces:
        _publish_resume(attempt, context)
        attempt = next(
            item
            for item in _scan_attempts(context.root)
            if item.number == attempt.number
        )
    _verify_resumes(attempt, context)

    if attempt.has_partial_collection:
        if BARRIER_NAME in attempt.names:
            raise ValueError("partial collection exists after analyzer barrier")
        attempt = _new_attempt(context.root, _scan_attempts(context.root))
        _publish_start(attempt, context)
        attempt = _scan_attempts(context.root)[-1]

    predictions = attempt.path / PREDICTIONS_NAME
    collection_receipt = attempt.path / COLLECTION_RECEIPT_NAME
    if COLLECTION_RECEIPT_NAME in attempt.names:
        collection_verifier(
            predictions,
            receipt_path=collection_receipt,
            source_manifest=context.fixed5,
        )
    else:
        collector(
            production_root=context.root,
            fixed5_source_manifest=context.fixed5,
            adoption_authorization=context.adoption,
            fixed48_source_manifest=context.fixed48,
            authorization_manifest=context.authorization,
            feasibility_gate=context.feasibility,
            launch=context.launch,
            excluded_audit=context.excluded,
            continuation_options=context.options,
            destination=predictions,
        )
        collection_verifier(
            predictions,
            receipt_path=collection_receipt,
            source_manifest=context.fixed5,
        )
    _runtime(
        context,
        manifest_verifier=manifest_verifier,
        excluded_verifier=excluded_verifier,
        launch_verifier=launch_verifier,
    )
    attempt = next(
        item
        for item in _scan_attempts(context.root)
        if item.number == attempt.number
    )

    barrier_preexisting = BARRIER_NAME in attempt.names
    if not barrier_preexisting:
        _publish_barrier(attempt, context)
        barrier_preexisting = False
    _verify_barrier(attempt, context)
    analysis_path = attempt.path / ANALYSIS_NAME
    if ANALYSIS_NAME not in attempt.names:
        if barrier_preexisting:
            raise RuntimeError(
                "analyzer barrier exists without a valid analysis report; "
                "second invocation is forbidden"
            )
        report = analyzer_runner(
            predictions,
            collection_receipt=collection_receipt,
            source_manifest=context.fixed5,
        )
        _write_json_exclusive(analysis_path, report, root=context.root)
    _verify_analysis_envelope(analysis_path)
    _runtime(
        context,
        manifest_verifier=manifest_verifier,
        excluded_verifier=excluded_verifier,
        launch_verifier=launch_verifier,
    )

    verification_path = attempt.path / VERIFICATION_NAME
    attempt = next(
        item
        for item in _scan_attempts(context.root)
        if item.number == attempt.number
    )
    if VERIFICATION_NAME not in attempt.names:
        verification = dict(independent_verifier(
            predictions,
            analyzer_report=analysis_path,
            collection_receipt=collection_receipt,
            source_manifest=context.fixed5,
        ))
        verification["finalization_provenance"] = _verification_provenance(
            attempt, context
        )
        _write_json_exclusive(
            verification_path, verification, root=context.root
        )
    _verify_verification_envelope(
        verification_path, attempt, context
    )
    _runtime(
        context,
        manifest_verifier=manifest_verifier,
        excluded_verifier=excluded_verifier,
        launch_verifier=launch_verifier,
    )
    attempt = next(
        item
        for item in _scan_attempts(context.root)
        if item.number == attempt.number
    )
    if COMPLETE_NAME not in attempt.names:
        _publish_complete(attempt, context)
    attempt = next(
        item
        for item in _scan_attempts(context.root)
        if item.number == attempt.number
    )
    _verify_complete(
        attempt,
        context,
        collection_verifier=collection_verifier,
    )
    _runtime(
        context,
        manifest_verifier=manifest_verifier,
        excluded_verifier=excluded_verifier,
        launch_verifier=launch_verifier,
    )
    return attempt.path / COMPLETE_NAME
