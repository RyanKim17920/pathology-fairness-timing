#!/usr/bin/env python3
"""Fail-closed, one-GPU, ascending fixed-48 production controller."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from tools.matched_cancer_stage_20260730.receipts import (
    file_identity,
    verify_receipt,
)

from .diag_contract import STUDY_ID, scenario_for
from .diag_worker import verify_phase
from .feasibility_gate import verify as verify_feasibility
from .source_manifest import REPO, verify_manifest


SEEDS = tuple(range(32001, 32049))
CANARY_SEEDS = (32001,)
REMAINDER_SEEDS = tuple(range(32002, 32049))
SUCCESS_SCHEMA = "matched-cancer-fixed48-seed-success/v1"
PROTOCOL = (
    REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/"
    "FIXED48_EXECUTION_PROTOCOL.md"
)
ATTEMPT_RE = re.compile(r"attempt_([0-9]{2,})")
SUCCESS_NAME = "SEED_SUCCESS_RECEIPT.json"
SEED_WORKER_MODULE = "tools.matched_cancer_fixed48_20260730.seed_worker"
Worker = Callable[[int, Path, Path, Path, Path, Path], None]
ManifestVerifier = Callable[[Path], Mapping[str, Any]]


class ControllerBusyError(RuntimeError):
    """Another fixed-48 controller owns the allocation-wide lock."""


def seeds_for_mode(mode: str) -> tuple[int, ...]:
    if mode == "canary":
        return CANARY_SEEDS
    if mode == "remainder":
        return REMAINDER_SEEDS
    raise ValueError("mode must be exactly canary or remainder")


def verify_one_gpu_environment(
    environment: Mapping[str, str] | None = None,
    *,
    cuda_device_count: Callable[[], int] | None = None,
) -> None:
    env = os.environ if environment is None else environment
    if not env.get("SLURM_JOB_ID"):
        raise ValueError("controller must run inside a Slurm allocation")
    for name in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        if env.get(name):
            raise ValueError("Slurm array execution is prohibited")
    if env.get("SLURM_GPUS_ON_NODE") != "1":
        raise ValueError("allocation must expose exactly one GPU on the node")
    allocated = env.get("SLURM_JOB_GPUS", "")
    if (
        not allocated
        or allocated.strip() != allocated
        or "," in allocated
        or any(character.isspace() for character in allocated)
    ):
        raise ValueError("allocation must contain exactly one GPU identifier")
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if (
        not visible
        or visible in {"NoDevFiles", "-1"}
        or visible.strip() != visible
        or "," in visible
        or any(character.isspace() for character in visible)
    ):
        raise ValueError("exactly one CUDA device must be visible")
    if env.get("SLURM_NTASKS", "1") != "1":
        raise ValueError("controller allocation must contain exactly one task")
    if cuda_device_count is None:
        import torch

        count = torch.cuda.device_count()
    else:
        count = cuda_device_count()
    if isinstance(count, bool) or count != 1:
        raise ValueError(
            f"runtime must see exactly one CUDA device; observed {count!r}"
        )


def _attempts(seed_root: Path) -> dict[int, Path]:
    if not seed_root.exists():
        return {}
    if not seed_root.is_dir() or seed_root.is_symlink():
        raise ValueError(f"seed root is not a real directory: {seed_root}")
    result: dict[int, Path] = {}
    for candidate in seed_root.iterdir():
        match = ATTEMPT_RE.fullmatch(candidate.name)
        if match is None:
            raise ValueError(f"unexpected entry in seed root: {candidate}")
        if not candidate.is_dir() or candidate.is_symlink():
            raise ValueError(f"attempt is not a real directory: {candidate}")
        number = int(match.group(1))
        if number < 1 or number in result:
            raise ValueError(f"invalid/duplicate attempt number: {candidate}")
        result[number] = candidate.resolve()
    return result


def _identity_is_within(identity: Mapping[str, Any], directory: Path) -> bool:
    try:
        Path(identity["canonical_path"]).resolve().relative_to(directory.resolve())
    except (KeyError, TypeError, ValueError):
        return False
    return True


def verify_seed_success(
    path: Path | str,
    *,
    seed: int,
    source_manifest: Path | str,
    production_root: Path | str,
) -> dict[str, Any]:
    """Verify one exact, append-only seed-success receipt and its ancestry."""
    if seed not in SEEDS:
        raise ValueError("success seed must be exactly one of 32001..32048")
    source = Path(path)
    if source.name != SUCCESS_NAME or source.is_symlink():
        raise ValueError("seed success must use the canonical non-symlink name")
    receipt = verify_receipt(
        source,
        expected_schema=SUCCESS_SCHEMA,
        expected_study_id=STUDY_ID,
        expected_scenario=scenario_for(seed),
    )
    expected_fields = {
        "schema",
        "study_id",
        "scenario",
        "identities",
        "topology_sha256",
        "fm_seed",
        "status",
        "legacy_outputs_used",
        "values_inspected",
    }
    if set(receipt) != expected_fields:
        raise ValueError("seed-success field topology differs")
    if (
        receipt["fm_seed"] != seed
        or receipt["status"] != "complete"
        or receipt["legacy_outputs_used"] is not False
        or receipt["values_inspected"] is not False
    ):
        raise ValueError("seed-success semantic fields differ")
    identities = receipt["identities"]
    expected_identities = {
        "source_manifest",
        "feasibility_gate",
        "execution_protocol",
        "calibration_root",
        "calibration_audit",
        "calibration_completions",
        "diagnostic_gate",
        "loader_root",
        "diagnostic_phase",
        "diagnostic_structural_audit",
        "per_seed_collection",
        "per_seed_collection_receipt",
    }
    if set(identities) != expected_identities:
        raise ValueError("seed-success identity topology differs")
    if set(identities["calibration_completions"]) != {"B", "P", "H"}:
        raise ValueError("seed-success calibration completions differ")
    manifest_path = Path(source_manifest).resolve(strict=True)
    if identities["source_manifest"] != file_identity(manifest_path):
        raise ValueError("seed-success source manifest differs")
    expected_feasibility = (
        Path(production_root).resolve()
        / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
    )
    if identities["feasibility_gate"] != file_identity(expected_feasibility):
        raise ValueError("seed-success feasibility gate differs")
    verify_feasibility(expected_feasibility)
    if identities["execution_protocol"] != file_identity(PROTOCOL):
        raise ValueError("seed-success execution protocol differs")

    root = Path(production_root).resolve()
    diagnostic_attempt = source.parent.resolve()
    expected_diagnostic_parent = root / "diagnostic" / f"seed_{seed}"
    if diagnostic_attempt.parent != expected_diagnostic_parent:
        raise ValueError("seed-success diagnostic attempt location differs")
    match = ATTEMPT_RE.fullmatch(diagnostic_attempt.name)
    if match is None:
        raise ValueError("seed-success parent is not an attempt directory")
    calibration_attempt = (
        root / "calibration" / f"seed_{seed}" / diagnostic_attempt.name
    )
    calibration_roles = {
        "calibration_root",
        "calibration_audit",
    }
    for role in calibration_roles:
        if not _identity_is_within(identities[role], calibration_attempt):
            raise ValueError(f"seed-success {role} escaped its attempt")
    for arm, identity in identities["calibration_completions"].items():
        if not _identity_is_within(identity, calibration_attempt):
            raise ValueError(f"seed-success completion {arm} escaped its attempt")
    diagnostic_roles = {
        "diagnostic_gate",
        "loader_root",
        "diagnostic_phase",
        "diagnostic_structural_audit",
        "per_seed_collection",
        "per_seed_collection_receipt",
    }
    for role in diagnostic_roles:
        if not _identity_is_within(identities[role], diagnostic_attempt):
            raise ValueError(f"seed-success {role} escaped its attempt")
        if identities[role] != file_identity(
            identities[role]["canonical_path"]
        ):
            raise ValueError(f"seed-success {role} identity drift")
    for role in calibration_roles:
        if identities[role] != file_identity(
            identities[role]["canonical_path"]
        ):
            raise ValueError(f"seed-success {role} identity drift")
    for arm, identity in identities["calibration_completions"].items():
        if identity != file_identity(identity["canonical_path"]):
            raise ValueError(f"seed-success completion {arm} identity drift")
    phase = verify_phase(
        identities["diagnostic_phase"]["canonical_path"],
        expected_fm_seed=seed,
    )
    phase_links = {
        "deployment_gate": "diagnostic_gate",
        "loader_root": "loader_root",
        "structural_audit": "diagnostic_structural_audit",
        "collection": "per_seed_collection",
        "collection_receipt": "per_seed_collection_receipt",
    }
    for phase_role, success_role in phase_links.items():
        if phase["identities"][phase_role] != identities[success_role]:
            raise ValueError(
                f"seed-success diagnostic phase {phase_role} differs"
            )
    return receipt


def _scan_state(
    root: Path,
    *,
    source_manifest: Path,
) -> tuple[tuple[int, ...], int]:
    completed: list[int] = []
    first_incomplete: int | None = None
    for seed in SEEDS:
        calibration_attempts = _attempts(
            root / "calibration" / f"seed_{seed}"
        )
        diagnostic_attempts = _attempts(
            root / "diagnostic" / f"seed_{seed}"
        )
        all_numbers = sorted(set(calibration_attempts) | set(diagnostic_attempts))
        successes: list[Path] = []
        for number in all_numbers:
            diagnostic = diagnostic_attempts.get(number)
            if diagnostic is not None:
                candidate = diagnostic / SUCCESS_NAME
                if candidate.exists() or candidate.is_symlink():
                    successes.append(candidate)
        if len(successes) > 1:
            raise ValueError(f"seed {seed} has multiple success receipts")
        if successes:
            verify_seed_success(
                successes[0],
                seed=seed,
                source_manifest=source_manifest,
                production_root=root,
            )
            if first_incomplete is not None:
                raise ValueError("completed seeds are not a contiguous prefix")
            completed.append(seed)
        else:
            if first_incomplete is None:
                first_incomplete = seed
            elif all_numbers:
                raise ValueError("attempt exists after the lowest incomplete seed")
    return tuple(completed), first_incomplete or (SEEDS[-1] + 1)


def _next_attempt(root: Path, seed: int) -> tuple[Path, Path]:
    calibration_parent = root / "calibration" / f"seed_{seed}"
    diagnostic_parent = root / "diagnostic" / f"seed_{seed}"
    calibration_parent.mkdir(parents=True, exist_ok=True)
    diagnostic_parent.mkdir(parents=True, exist_ok=True)
    used = set(_attempts(calibration_parent)) | set(_attempts(diagnostic_parent))
    number = next(
        candidate
        for candidate in range(1, max(used, default=0) + 2)
        if candidate not in used
    )
    name = f"attempt_{number:02d}"
    calibration = calibration_parent / name
    diagnostic = diagnostic_parent / name
    if calibration.exists() or calibration.is_symlink():
        raise FileExistsError(f"calibration attempt already exists: {calibration}")
    if diagnostic.exists() or diagnostic.is_symlink():
        raise FileExistsError(f"diagnostic attempt already exists: {diagnostic}")
    # Phase workers own exclusive leaf creation because their existing,
    # independently tested contracts reject pre-created attempt roots.
    return calibration.absolute(), diagnostic.absolute()


def run_controller(
    *,
    mode: str,
    production_root: Path | str,
    source_manifest: Path | str,
    authorization_manifest: Path | str,
    feasibility_gate: Path | str,
    seed_worker: Worker,
    environment: Mapping[str, str] | None = None,
    manifest_verifier: ManifestVerifier = verify_manifest,
    cuda_device_count: Callable[[], int] | None = None,
) -> tuple[int, ...]:
    """Run an exact canary/remainder range, one lowest-incomplete seed at a time."""
    requested = seeds_for_mode(mode)
    root = Path(production_root)
    if root.is_symlink():
        raise ValueError("production root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    manifest = Path(source_manifest).resolve(strict=True)
    authorization = Path(authorization_manifest)
    if authorization.is_symlink() or not authorization.is_file():
        raise ValueError("authorization manifest must be a non-symlink file")
    authorization = authorization.resolve(strict=True)
    authorization_identity = file_identity(authorization)
    feasibility = Path(feasibility_gate)
    if feasibility.is_symlink() or not feasibility.is_file():
        raise ValueError("feasibility gate must be a non-symlink file")
    feasibility = feasibility.resolve(strict=True)
    expected_feasibility = root / "control/FEASIBILITY_GATE_RECEIPT_V2.json"
    if feasibility != expected_feasibility:
        raise ValueError("feasibility gate path differs from production contract")
    verify_feasibility(
        feasibility, authorization_manifest=authorization
    )
    feasibility_identity = file_identity(feasibility)
    lock_root = root / "control"
    lock_root.mkdir(exist_ok=True)
    lock_path = lock_root / "serial_controller.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControllerBusyError(
                "another fixed-48 controller is active"
            ) from error
        verify_one_gpu_environment(
            environment, cuda_device_count=cuda_device_count
        )
        manifest_verifier(manifest)
        completed, lowest = _scan_state(root, source_manifest=manifest)
        if mode == "canary":
            if lowest > 32001:
                return ()
            if lowest != 32001:
                raise ValueError("canary may process only seed 32001")
        else:
            if 32001 not in completed:
                raise ValueError("remainder requires a verified seed-32001 canary")
            if lowest > SEEDS[-1]:
                return ()
            if lowest not in REMAINDER_SEEDS:
                raise ValueError("remainder seed range differs")

        newly_completed: list[int] = []
        for seed in requested:
            manifest_verifier(manifest)
            if file_identity(authorization) != authorization_identity:
                raise ValueError("authorization manifest drifted during execution")
            if file_identity(feasibility) != feasibility_identity:
                raise ValueError("feasibility gate drifted during execution")
            verify_feasibility(
                feasibility, authorization_manifest=authorization
            )
            completed, lowest = _scan_state(root, source_manifest=manifest)
            if seed in completed:
                continue
            if seed != lowest:
                raise ValueError(
                    f"seed {seed} is not the lowest incomplete seed {lowest}"
                )
            calibration_attempt, diagnostic_attempt = _next_attempt(root, seed)
            # The bound seed worker owns the ordered calibration -> calibration
            # audit -> diagnostic -> structural audit -> success-seal phases.
            seed_worker(
                seed,
                calibration_attempt,
                diagnostic_attempt,
                manifest,
                authorization,
                feasibility,
            )
            manifest_verifier(manifest)
            if file_identity(authorization) != authorization_identity:
                raise ValueError("authorization manifest drifted during seed")
            if file_identity(feasibility) != feasibility_identity:
                raise ValueError("feasibility gate drifted during seed")
            success = diagnostic_attempt / SUCCESS_NAME
            verify_seed_success(
                success,
                seed=seed,
                source_manifest=manifest,
                production_root=root,
            )
            newly_completed.append(seed)
        return tuple(newly_completed)


def submission_preflight(
    *,
    mode: str,
    production_root: Path | str,
    source_manifest: Path | str,
    authorization_manifest: Path | str,
    feasibility_gate: Path | str,
) -> None:
    """Outcome-blind submit guard; it never creates attempts or needs a GPU."""
    requested = seeds_for_mode(mode)
    root = Path(production_root).resolve()
    manifest = Path(source_manifest).resolve(strict=True)
    verify_manifest(manifest)
    authorization = Path(authorization_manifest).resolve(strict=True)
    feasibility = Path(feasibility_gate).resolve(strict=True)
    verify_feasibility(
        feasibility, authorization_manifest=authorization
    )
    completed, lowest = _scan_state(root, source_manifest=manifest)
    if mode == "canary":
        if 32001 in completed:
            raise ValueError("canary is already complete")
        if lowest != 32001:
            raise ValueError("canary state differs")
    else:
        if 32001 not in completed:
            raise ValueError("verified canary is required before remainder")
        if all(seed in completed for seed in requested):
            raise ValueError("remainder is already complete")
        if lowest not in requested:
            raise ValueError("remainder state differs")


def _bound_worker(
    executable: Path,
    *,
    manifest: Path,
    manifest_receipt: Mapping[str, Any],
) -> Worker:
    if executable.is_symlink() or not executable.is_file():
        raise ValueError(f"worker must be a non-symlink file: {executable}")
    executable = executable.resolve(strict=True)
    source_identities = manifest_receipt["identities"]["sources"]
    if source_identities.get("fixed48.seed_worker") != file_identity(
        executable
    ):
        raise ValueError(f"worker is not bound by the source manifest: {executable}")

    def run(
        seed: int,
        calibration_attempt: Path,
        diagnostic_attempt: Path,
        _: Path,
        authorization: Path,
        feasibility: Path,
    ) -> None:
        if calibration_attempt.name != diagnostic_attempt.name:
            raise ValueError("calibration and diagnostic attempt names differ")
        environment = dict(os.environ)
        subprocess.run(
            [
                sys.executable,
                "-m",
                SEED_WORKER_MODULE,
                str(seed),
                calibration_attempt.name,
                "--source-manifest",
                str(manifest),
                "--authorization-manifest",
                str(authorization),
                "--feasibility-gate",
                str(feasibility),
            ],
            cwd=REPO,
            env=environment,
            check=True,
        )

    return run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("canary", "remainder"), required=True)
    run.add_argument("--production-root", type=Path, required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--authorization-manifest", type=Path, required=True)
    run.add_argument("--feasibility-gate", type=Path, required=True)
    run.add_argument("--seed-worker", type=Path, required=True)
    preflight = subparsers.add_parser("submission-preflight")
    preflight.add_argument(
        "--mode", choices=("canary", "remainder"), required=True
    )
    preflight.add_argument("--production-root", type=Path, required=True)
    preflight.add_argument("--source-manifest", type=Path, required=True)
    preflight.add_argument("--authorization-manifest", type=Path, required=True)
    preflight.add_argument("--feasibility-gate", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "submission-preflight":
        submission_preflight(
            mode=args.mode,
            production_root=args.production_root,
            source_manifest=args.source_manifest,
            authorization_manifest=args.authorization_manifest,
            feasibility_gate=args.feasibility_gate,
        )
        return 0
    manifest = args.source_manifest.resolve(strict=True)
    receipt = verify_manifest(manifest)
    worker = _bound_worker(
        args.seed_worker, manifest=manifest, manifest_receipt=receipt
    )
    run_controller(
        mode=args.mode,
        production_root=args.production_root,
        source_manifest=manifest,
        authorization_manifest=args.authorization_manifest,
        feasibility_gate=args.feasibility_gate,
        seed_worker=worker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
