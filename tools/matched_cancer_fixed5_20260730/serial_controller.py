#!/usr/bin/env python3
"""One-allocation fixed-five execution, provenance, and finalization controller."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
from typing import Any, Callable, Iterator, Mapping, Sequence

from tools.matched_cancer_stage_20260730.receipts import file_identity
from tools.matched_cancer_fixed48_20260730 import (
    serial_controller as fixed48_controller,
)
from tools.matched_cancer_fixed48_20260730.diag_authorization import (
    verify_authorization as verify_fixed48_authorization,
)
from tools.matched_cancer_fixed48_20260730.feasibility_gate import (
    verify as verify_fixed48_feasibility,
)
from tools.matched_cancer_fixed48_20260730.source_manifest import (
    REPO,
    verify_manifest as verify_fixed48_manifest,
)

from . import execution_receipts, launch_receipt


PRODUCTION_ROOT = launch_receipt.PRODUCTION_ROOT
ADOPTED_SEEDS = execution_receipts.ADOPTED_SEEDS
CONTROLLER_SEEDS = execution_receipts.CONTROLLER_SEEDS
EXCLUDED_SEEDS = execution_receipts.EXCLUDED_SEEDS
CANARY_SEED = 32001
SUCCESS_NAME = fixed48_controller.SUCCESS_NAME
FIXED48_MANIFEST_NAME = "FIXED48_SOURCE_MANIFEST_V2.json"
FIXED5_MANIFEST_NAME = "FIXED5_SOURCE_MANIFEST_V1.json"
AUTHORIZATION_NAME = "AUTHORIZATION_MANIFEST_V3.json"
ADOPTION_NAME = "FIXED5_ADOPTION_AUTHORIZATION_V1.json"
FEASIBILITY_NAME = "FEASIBILITY_GATE_RECEIPT_V2.json"

Worker = Callable[[int, Path, Path, Path, Path, Path], None]
Verifier = Callable[..., Mapping[str, Any]]
Finalizer = Callable[..., Any]


class ControllerBusyError(RuntimeError):
    """Another fixed-48/fixed-five controller owns the canonical lock."""


def _verify_fixed5_manifest(path: Path) -> Mapping[str, Any]:
    from .source_manifest import verify_manifest

    return verify_manifest(path)


def _verify_adoption(
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


def _default_finalizer(**kwargs: Any) -> Any:
    from .finalizer import run

    return run(**kwargs)


def _require_finalizer() -> None:
    try:
        from . import finalizer
    except ImportError as error:
        raise RuntimeError(
            "fixed-five finalizer is unavailable; submission is prohibited"
        ) from error
    if not callable(getattr(finalizer, "run", None)):
        raise RuntimeError("fixed-five finalizer.run is unavailable")


def _resolve_exact_file(raw: Path | str, expected: Path, label: str) -> Path:
    candidate = Path(raw)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a non-symlink file")
    if candidate.absolute() != expected:
        raise ValueError(f"{label} path differs from the fixed-five contract")
    resolved = candidate.resolve(strict=True)
    if resolved != expected:
        raise ValueError(f"{label} canonical path differs")
    return resolved


def _resolve_inputs(
    *,
    production_root: Path | str,
    source_manifest: Path | str,
    fixed5_source_manifest: Path | str,
    authorization_manifest: Path | str,
    adoption_authorization: Path | str,
    feasibility_gate: Path | str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    root_candidate = Path(production_root)
    expected_candidate = PRODUCTION_ROOT
    if root_candidate.is_symlink() or expected_candidate.is_symlink():
        raise ValueError("production root may not be a symlink")
    expected = expected_candidate.resolve(strict=True)
    if root_candidate.absolute() != expected:
        raise ValueError("production root path differs from the fixed-five contract")
    root = root_candidate.resolve(strict=True)
    if root != expected:
        raise ValueError("production root canonical path differs")
    fixed48 = _resolve_exact_file(
        source_manifest,
        root / "control" / FIXED48_MANIFEST_NAME,
        "fixed-48 source manifest",
    )
    fixed5 = _resolve_exact_file(
        fixed5_source_manifest,
        root / "control" / FIXED5_MANIFEST_NAME,
        "fixed-five source manifest",
    )
    authorization = _resolve_exact_file(
        authorization_manifest,
        root / "authorization" / AUTHORIZATION_NAME,
        "fixed-48 authorization",
    )
    adoption = _resolve_exact_file(
        adoption_authorization,
        root / "authorization" / ADOPTION_NAME,
        "fixed-five adoption authorization",
    )
    feasibility = _resolve_exact_file(
        feasibility_gate,
        root / "control" / FEASIBILITY_NAME,
        "fixed-48 feasibility gate",
    )
    return root, fixed48, fixed5, authorization, adoption, feasibility


def _control_identities(
    *,
    fixed48_source_manifest: Path,
    fixed5_source_manifest: Path,
    authorization_manifest: Path,
    adoption_authorization: Path,
    feasibility_gate: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "fixed48_source_manifest": file_identity(fixed48_source_manifest),
        "fixed5_source_manifest": file_identity(fixed5_source_manifest),
        "authorization_manifest": file_identity(authorization_manifest),
        "adoption_authorization": file_identity(adoption_authorization),
        "feasibility_gate": file_identity(feasibility_gate),
    }


def verify_controls(
    *,
    production_root: Path,
    fixed48_source_manifest: Path,
    fixed5_source_manifest: Path,
    authorization_manifest: Path,
    adoption_authorization: Path,
    feasibility_gate: Path,
    fixed48_manifest_verifier: Verifier = verify_fixed48_manifest,
    fixed5_manifest_verifier: Verifier = _verify_fixed5_manifest,
    authorization_verifier: Verifier = verify_fixed48_authorization,
    feasibility_verifier: Verifier = verify_fixed48_feasibility,
    adoption_verifier: Verifier = _verify_adoption,
) -> None:
    fixed48_manifest_verifier(fixed48_source_manifest)
    fixed5_manifest_verifier(fixed5_source_manifest)
    authorization_verifier(authorization_manifest)
    feasibility_verifier(
        feasibility_gate, authorization_manifest=authorization_manifest
    )
    adoption_verifier(
        adoption_authorization,
        fixed48_source_manifest=fixed48_source_manifest,
        authorization_manifest=authorization_manifest,
        feasibility_gate=feasibility_gate,
        production_root=production_root,
    )


def _launch_environment(
    environment: Mapping[str, str] | None,
    *,
    cuda_device_count: Callable[[], int] | None,
) -> tuple[Mapping[str, str], str, str, Path]:
    env = os.environ if environment is None else environment
    fixed48_controller.verify_one_gpu_environment(
        env, cuda_device_count=cuda_device_count
    )
    if env.get("SLURM_JOB_NAME") != launch_receipt.JOB_NAME:
        raise ValueError("Slurm job name differs")
    if env.get("FIXED5_SLURM_COMMENT") != launch_receipt.LEGACY_COMMENT:
        raise ValueError("Slurm comment differs")
    if env.get("SLURM_NTASKS") != "1":
        raise ValueError("Slurm task count must be exactly one")
    nonce = launch_receipt.validate_nonce(env.get("FIXED5_LAUNCH_NONCE", ""))
    job_id = env.get("SLURM_JOB_ID", "")
    if launch_receipt.JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError("Slurm job ID must be decimal")
    prelaunch_raw = env.get("FIXED5_PRELAUNCH_RECEIPT", "")
    if not prelaunch_raw:
        raise ValueError("exported prelaunch receipt is missing")
    prelaunch = Path(prelaunch_raw)
    return env, nonce, job_id, prelaunch


@contextmanager
def _controller_lock(root: Path) -> Iterator[None]:
    control = root / "control"
    if (
        control.is_symlink()
        or not control.is_dir()
        or control.resolve(strict=True) != control
    ):
        raise ValueError("control directory is missing or redirected")
    lock_path = control / "serial_controller.lock"
    if lock_path.is_symlink():
        raise ValueError("controller lock may not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o664)
    except OSError as error:
        raise ValueError("cannot open canonical controller lock") from error
    try:
        opened = os.fstat(descriptor)
        current = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or lock_path.is_symlink()
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
            or lock_path.resolve(strict=True) != lock_path
        ):
            raise ValueError("controller lock identity or ancestry differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ControllerBusyError(
                "another fixed-48/fixed-five controller is active"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _verify_identity_snapshot(
    snapshot: Mapping[str, Mapping[str, Any]],
    *,
    fixed48: Path,
    fixed5: Path,
    authorization: Path,
    adoption: Path,
    feasibility: Path,
) -> None:
    current = _control_identities(
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=fixed5,
        authorization_manifest=authorization,
        adoption_authorization=adoption,
        feasibility_gate=feasibility,
    )
    if current != snapshot:
        raise ValueError("fixed-five controls drifted")


def run_controller(
    *,
    production_root: Path | str,
    source_manifest: Path | str,
    fixed5_source_manifest: Path | str,
    authorization_manifest: Path | str,
    adoption_authorization: Path | str,
    feasibility_gate: Path | str,
    seed_worker: Worker,
    environment: Mapping[str, str] | None = None,
    cuda_device_count: Callable[[], int] | None = None,
    fixed48_manifest_verifier: Verifier = verify_fixed48_manifest,
    fixed5_manifest_verifier: Verifier = _verify_fixed5_manifest,
    authorization_verifier: Verifier = verify_fixed48_authorization,
    feasibility_verifier: Verifier = verify_fixed48_feasibility,
    adoption_verifier: Verifier = _verify_adoption,
    success_verifier: Verifier = fixed48_controller.verify_seed_success,
    prelaunch_verifier: Verifier = launch_receipt.verify_prelaunch,
    launch_publisher: Callable[..., Path] = launch_receipt.create_launch,
    launch_verifier: Verifier = launch_receipt.verify_launch,
    finalizer: Finalizer = _default_finalizer,
) -> tuple[int, ...]:
    (
        root,
        fixed48,
        fixed5,
        authorization,
        adoption,
        feasibility,
    ) = _resolve_inputs(
        production_root=production_root,
        source_manifest=source_manifest,
        fixed5_source_manifest=fixed5_source_manifest,
        authorization_manifest=authorization_manifest,
        adoption_authorization=adoption_authorization,
        feasibility_gate=feasibility_gate,
    )
    control_verifiers = {
        "production_root": root,
        "fixed48_source_manifest": fixed48,
        "fixed5_source_manifest": fixed5,
        "authorization_manifest": authorization,
        "adoption_authorization": adoption,
        "feasibility_gate": feasibility,
        "fixed48_manifest_verifier": fixed48_manifest_verifier,
        "fixed5_manifest_verifier": fixed5_manifest_verifier,
        "authorization_verifier": authorization_verifier,
        "feasibility_verifier": feasibility_verifier,
        "adoption_verifier": adoption_verifier,
    }
    frozen_worker = (
        REPO / "tools/matched_cancer_fixed48_20260730/seed_worker.py"
    ).resolve(strict=True)

    with _controller_lock(root):
        env, nonce, job_id, prelaunch = _launch_environment(
            environment, cuda_device_count=cuda_device_count
        )
        expected_prelaunch = (
            root / f"control/prelaunch/FIXED5_PRELAUNCH_{nonce}.json"
        )
        prelaunch = _resolve_exact_file(
            prelaunch, expected_prelaunch, "prelaunch queue receipt"
        )
        if finalizer is _default_finalizer:
            _require_finalizer()
        snapshot = _control_identities(
            fixed48_source_manifest=fixed48,
            fixed5_source_manifest=fixed5,
            authorization_manifest=authorization,
            adoption_authorization=adoption,
            feasibility_gate=feasibility,
        )
        verify_controls(**control_verifiers)
        _verify_identity_snapshot(
            snapshot,
            fixed48=fixed48,
            fixed5=fixed5,
            authorization=authorization,
            adoption=adoption,
            feasibility=feasibility,
        )
        execution_receipts.scan_excluded_state(root)
        prelaunch_verifier(
            prelaunch,
            launch_nonce=nonce,
            production_root=root,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
        )
        launch_path = (
            root
            / f"control/launch/FIXED5_LAUNCH_{nonce}_JOB_{job_id}.json"
        )
        launch = launch_publisher(
            launch_path,
            launch_nonce=nonce,
            slurm_job_id=job_id,
            prelaunch_receipt=prelaunch,
            production_root=root,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
        )

        def verify_runtime() -> None:
            verify_controls(**control_verifiers)
            _verify_identity_snapshot(
                snapshot,
                fixed48=fixed48,
                fixed5=fixed5,
                authorization=authorization,
                adoption=adoption,
                feasibility=feasibility,
            )
            execution_receipts.scan_excluded_state(root)
            prelaunch_verifier(
                prelaunch,
                launch_nonce=nonce,
                production_root=root,
                fixed5_source_manifest=fixed5,
                adoption_authorization=adoption,
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

        newly_completed: list[int] = []
        verify_runtime()
        state = execution_receipts.scan_state(
            production_root=root,
            fixed48_source_manifest=fixed48,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
            authorization_manifest=authorization,
            feasibility_gate=feasibility,
            frozen_worker=frozen_worker,
            success_verifier=success_verifier,
        )
        for seed in CONTROLLER_SEEDS:
            verify_runtime()
            state = execution_receipts.scan_state(
                production_root=root,
                fixed48_source_manifest=fixed48,
                fixed5_source_manifest=fixed5,
                adoption_authorization=adoption,
                authorization_manifest=authorization,
                feasibility_gate=feasibility,
                frozen_worker=frozen_worker,
                success_verifier=success_verifier,
            )
            if seed in state.completed:
                continue
            if seed != state.lowest_incomplete:
                raise ValueError(
                    f"seed {seed} is not lowest incomplete "
                    f"{state.lowest_incomplete}"
                )
            if seed in state.resumable:
                attempt_number = state.resumable[seed]
                attempt_name = f"attempt_{attempt_number:02d}"
                success = (
                    root
                    / "diagnostic"
                    / f"seed_{seed}"
                    / attempt_name
                    / SUCCESS_NAME
                )
                execution_receipts.publish_complete(
                    production_root=root,
                    seed=seed,
                    attempt_name=attempt_name,
                    fixed48_success=success,
                    fixed5_source_manifest=fixed5,
                    adoption_authorization=adoption,
                    prelaunch_receipt=prelaunch,
                    launch=launch,
                    fixed48_source_manifest=fixed48,
                    authorization_manifest=authorization,
                    feasibility_gate=feasibility,
                    frozen_worker=frozen_worker,
                    success_verifier=success_verifier,
                )
                verify_runtime()
                newly_completed.append(seed)
                continue

            attempt_number = execution_receipts.next_attempt_number(state, seed)
            attempt_name = f"attempt_{attempt_number:02d}"
            reservation = execution_receipts.reserve_attempt(
                production_root=root,
                seed=seed,
                attempt_number=attempt_number,
            )
            # The empty reservation consumes its number after a crash. Recheck
            # every runtime invariant immediately before START publication.
            verify_runtime()
            start = execution_receipts.publish_start(
                production_root=root,
                seed=seed,
                attempt_name=attempt_name,
                fixed5_source_manifest=fixed5,
                adoption_authorization=adoption,
                prelaunch_receipt=prelaunch,
                launch=launch,
                fixed48_source_manifest=fixed48,
                authorization_manifest=authorization,
                feasibility_gate=feasibility,
                frozen_worker=frozen_worker,
            )
            verify_runtime()
            execution_receipts.verify_start(
                start,
                production_root=root,
                seed=seed,
                attempt_name=attempt_name,
                fixed5_source_manifest=fixed5,
                adoption_authorization=adoption,
                prelaunch_receipt=prelaunch,
                launch=launch,
                fixed48_source_manifest=fixed48,
                authorization_manifest=authorization,
                feasibility_gate=feasibility,
                frozen_worker=frozen_worker,
            )
            calibration_attempt = (
                root / "calibration" / f"seed_{seed}" / attempt_name
            )
            diagnostic_attempt = (
                root / "diagnostic" / f"seed_{seed}" / attempt_name
            )
            if (
                calibration_attempt.exists()
                or calibration_attempt.is_symlink()
                or diagnostic_attempt.exists()
                or diagnostic_attempt.is_symlink()
            ):
                raise ValueError("worker attempt appeared before invocation")
            if reservation.name != attempt_name:
                raise ValueError("attempt reservation name differs")
            seed_worker(
                seed,
                calibration_attempt,
                diagnostic_attempt,
                fixed48,
                authorization,
                feasibility,
            )
            verify_runtime()
            success = diagnostic_attempt / SUCCESS_NAME
            success_verifier(
                success,
                seed=seed,
                source_manifest=fixed48,
                production_root=root,
            )
            execution_receipts.publish_complete(
                production_root=root,
                seed=seed,
                attempt_name=attempt_name,
                fixed48_success=success,
                fixed5_source_manifest=fixed5,
                adoption_authorization=adoption,
                prelaunch_receipt=prelaunch,
                launch=launch,
                fixed48_source_manifest=fixed48,
                authorization_manifest=authorization,
                feasibility_gate=feasibility,
                frozen_worker=frozen_worker,
                success_verifier=success_verifier,
            )
            verify_runtime()
            newly_completed.append(seed)

        state = execution_receipts.scan_state(
            production_root=root,
            fixed48_source_manifest=fixed48,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
            authorization_manifest=authorization,
            feasibility_gate=feasibility,
            frozen_worker=frozen_worker,
            success_verifier=success_verifier,
        )
        if state.completed != ADOPTED_SEEDS:
            raise ValueError("all five seeds must complete before finalization")
        excluded_audit = execution_receipts.publish_excluded_audit(
            production_root=root,
            launch_nonce=nonce,
            launch=launch,
            state=state,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
        )
        execution_receipts.verify_excluded_audit(
            excluded_audit,
            production_root=root,
            launch_nonce=nonce,
            launch=launch,
            state=state,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
        )
        verify_runtime()
        finalizer(
            production_root=root,
            fixed5_source_manifest=fixed5,
            adoption_authorization=adoption,
            fixed48_source_manifest=fixed48,
            authorization_manifest=authorization,
            feasibility_gate=feasibility,
            launch_receipt=launch,
            excluded_audit=excluded_audit,
            environment=dict(env),
        )
        # This final check still runs under the canonical execution lock.
        verify_runtime()
        execution_receipts.scan_excluded_state(root)
        return tuple(newly_completed)


def submission_preflight(
    *,
    production_root: Path | str,
    source_manifest: Path | str,
    fixed5_source_manifest: Path | str,
    authorization_manifest: Path | str,
    adoption_authorization: Path | str,
    feasibility_gate: Path | str,
    fixed48_manifest_verifier: Verifier = verify_fixed48_manifest,
    fixed5_manifest_verifier: Verifier = _verify_fixed5_manifest,
    authorization_verifier: Verifier = verify_fixed48_authorization,
    feasibility_verifier: Verifier = verify_fixed48_feasibility,
    adoption_verifier: Verifier = _verify_adoption,
    success_verifier: Verifier = fixed48_controller.verify_seed_success,
    finalizer_checker: Callable[[], None] = _require_finalizer,
) -> None:
    (
        root,
        fixed48,
        fixed5,
        authorization,
        adoption,
        feasibility,
    ) = _resolve_inputs(
        production_root=production_root,
        source_manifest=source_manifest,
        fixed5_source_manifest=fixed5_source_manifest,
        authorization_manifest=authorization_manifest,
        adoption_authorization=adoption_authorization,
        feasibility_gate=feasibility_gate,
    )
    finalizer_checker()
    verify_controls(
        production_root=root,
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=fixed5,
        authorization_manifest=authorization,
        adoption_authorization=adoption,
        feasibility_gate=feasibility,
        fixed48_manifest_verifier=fixed48_manifest_verifier,
        fixed5_manifest_verifier=fixed5_manifest_verifier,
        authorization_verifier=authorization_verifier,
        feasibility_verifier=feasibility_verifier,
        adoption_verifier=adoption_verifier,
    )
    execution_receipts.scan_excluded_state(root)
    # Historical START/COMPLETE receipts replay their own immutable launch
    # ancestry, so no new launch receipt is needed for this state audit.
    execution_receipts.scan_state(
        production_root=root,
        fixed48_source_manifest=fixed48,
        fixed5_source_manifest=fixed5,
        adoption_authorization=adoption,
        authorization_manifest=authorization,
        feasibility_gate=feasibility,
        frozen_worker=execution_receipts.FROZEN_WORKER,
        success_verifier=success_verifier,
    )


def _add_control_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--fixed5-source-manifest", type=Path, required=True)
    parser.add_argument("--authorization-manifest", type=Path, required=True)
    parser.add_argument("--adoption-authorization", type=Path, required=True)
    parser.add_argument("--feasibility-gate", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    _add_control_arguments(run)
    run.add_argument("--seed-worker", type=Path, required=True)
    preflight = subparsers.add_parser("submission-preflight")
    _add_control_arguments(preflight)
    args = parser.parse_args(argv)
    common = {
        "production_root": args.production_root,
        "source_manifest": args.source_manifest,
        "fixed5_source_manifest": args.fixed5_source_manifest,
        "authorization_manifest": args.authorization_manifest,
        "adoption_authorization": args.adoption_authorization,
        "feasibility_gate": args.feasibility_gate,
    }
    if args.command == "submission-preflight":
        submission_preflight(**common)
        return 0
    fixed48_manifest = args.source_manifest.resolve(strict=True)
    manifest_receipt = verify_fixed48_manifest(fixed48_manifest)
    worker = fixed48_controller._bound_worker(
        args.seed_worker,
        manifest=fixed48_manifest,
        manifest_receipt=manifest_receipt,
    )
    run_controller(seed_worker=worker, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
