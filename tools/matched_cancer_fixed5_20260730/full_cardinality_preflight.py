#!/usr/bin/env python3
"""Full-cardinality synthetic preflight for the fixed-five final stack.

No caller can supply predictions, labels, probabilities, or a production
root.  The module creates its own deterministic synthetic 182,700-row matrix
and drives the real fixed-five analyzer, independent verifier, and finalizer
state machine through explicitly injected synthetic controls.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)

from . import (
    analyzer,
    execution_receipts,
    final_collector,
    finalizer,
    launch_receipt,
    verifier,
)
from .continuation_options_receipt import AMENDMENTS, DOCUMENT


SCHEMA = "matched-cancer-fixed5-full-cardinality-preflight/v1"
STUDY_ID = "matched_cancer_fixed5_20260730"
SCENARIO = "fixed5_full_cardinality_synthetic"
EXPECTED_ROWS = 182_700
EXPECTED_COMBINATIONS = 120
EXPECTED_NESTED_AUDITS = 2_250
EXPECTED_CLASSIFICATION = "small_across_five_tested_seeds"
REAL_PRODUCTION_ROOT = Path(
    "/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730"
)
RACE_COUNTS = {
    "BRCA": {"Black": 118, "White": 210},
    "LUAD": {"Black": 40, "White": 241},
}
HEAD_OFFSETS = {
    42001: -0.015,
    42002: -0.005,
    42003: 0.005,
    42004: 0.015,
}

FinalizerRunner = Callable[..., Path]


def _canonical(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _patients(cancer: str) -> Iterator[tuple[str, int, str, int]]:
    for race in ("Black", "White"):
        for index in range(RACE_COUNTS[cancer][race]):
            fold = index % 5
            outcome = (index // 5) % 2
            yield (
                f"SYN-{cancer}-{race.upper()}-{index:04d}",
                outcome,
                race,
                fold,
            )


def _probability(*, seed: int, head: int, outcome: int) -> float:
    seed_offset = ((seed - analyzer.FM_SEEDS[0]) % 5 - 2) * 0.0002
    value = 0.15 + 0.70 * outcome + seed_offset + HEAD_OFFSETS[head]
    if not 0.0 < value < 1.0:
        raise AssertionError("synthetic probability escaped unit interval")
    return value


def _canonical_new_root(output_root: Path | str) -> Path:
    requested = Path(output_root)
    if os.path.lexists(requested):
        raise FileExistsError(f"preflight output root exists: {requested}")
    parent = requested.parent.absolute()
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("preflight output ancestry is redirected")
    root = parent / requested.name
    if root.absolute() != requested.absolute():
        raise ValueError("preflight output root is not canonical")
    real = REAL_PRODUCTION_ROOT
    if real.exists():
        real = real.resolve(strict=True)
        try:
            root.resolve(strict=False).relative_to(real)
        except ValueError:
            pass
        else:
            raise ValueError(
                "synthetic preflight may not write inside production root"
            )
    root.mkdir()
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("preflight output root was redirected")
    return root


def write_predictions(destination: Path) -> tuple[int, str]:
    """Write the exact fixed-five matrix from deterministic synthetic records."""
    if os.path.lexists(destination):
        raise FileExistsError(f"synthetic predictions exist: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o664)
    digest = hashlib.sha256()
    count = 0
    with os.fdopen(descriptor, "wb") as stream:
        for seed in analyzer.FM_SEEDS:
            for cancer in analyzer.CANCERS:
                patients = tuple(_patients(cancer))
                if len(patients) != analyzer.COHORT_SIZES[cancer]:
                    raise AssertionError(
                        f"{cancer} synthetic cohort cardinality drift"
                    )
                for arm in analyzer.ARMS:
                    for head in analyzer.HEADS:
                        for patient, outcome, race, fold in patients:
                            common = {
                                "schema": (
                                    "matched-cancer-diagnostic-prediction/v1"
                                ),
                                "fm_seed": seed,
                                "arm": arm,
                                "cancer": cancer,
                                "head_seed": head,
                                "patient_id": patient,
                                "y_true": outcome,
                                "race": race,
                                "fold": fold,
                                "probability": _probability(
                                    seed=seed,
                                    head=head,
                                    outcome=outcome,
                                ),
                            }
                            outer_row = {
                                **common,
                                "role": "outer_test",
                                "outer_fold": fold,
                                "inner_fold": None,
                            }
                            payload = _canonical(outer_row)
                            stream.write(payload)
                            digest.update(payload)
                            count += 1
                            for outer in analyzer.FOLDS:
                                if outer == fold:
                                    continue
                                inner_row = {
                                    **common,
                                    "role": "inner_calibration",
                                    "outer_fold": outer,
                                    "inner_fold": fold,
                                }
                                payload = _canonical(inner_row)
                                stream.write(payload)
                                digest.update(payload)
                                count += 1
        stream.flush()
        os.fsync(stream.fileno())
    if count != EXPECTED_ROWS:
        raise AssertionError(
            f"synthetic row count {count} != {EXPECTED_ROWS}"
        )
    return count, digest.hexdigest()


def _exclusive_copy(source: Path, destination: Path) -> Path:
    if os.path.lexists(destination):
        raise FileExistsError(f"synthetic collection exists: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o664)
    with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return destination


def _synthetic_state(root: Path) -> execution_receipts.StudyState:
    chains: dict[int, dict[str, Path]] = {}
    for seed in execution_receipts.ADOPTED_SEEDS:
        seed_root = root / "synthetic_chains" / f"seed_{seed}"
        seed_root.mkdir(parents=True)
        success = seed_root / execution_receipts.SUCCESS_NAME
        success.write_text(f"synthetic success {seed}\n")
        if seed == 32001:
            chains[seed] = {"success": success}
        else:
            start = seed_root / execution_receipts.START_NAME
            complete = seed_root / execution_receipts.COMPLETE_NAME
            start.write_text(f"synthetic start {seed}\n")
            complete.write_text(f"synthetic complete {seed}\n")
            chains[seed] = {
                "start": start,
                "complete": complete,
                "success": success,
            }
    return execution_receipts.StudyState(
        completed=execution_receipts.ADOPTED_SEEDS,
        lowest_incomplete=32006,
        resumable={},
        used_attempts={
            seed: (1,) for seed in execution_receipts.ADOPTED_SEEDS
        },
        chains=chains,
    )


def _write_control(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path.resolve()


@contextmanager
def _injected_module_roots(root: Path) -> Iterator[None]:
    replacements = (
        (finalizer, "PRODUCTION_ROOT", root),
        (final_collector, "PRODUCTION_ROOT", root),
        (launch_receipt, "PRODUCTION_ROOT", root),
        (execution_receipts, "PRODUCTION_ROOT", root),
        (finalizer, "CONTINUATION_OPTIONS", DOCUMENT.resolve(strict=True)),
        (
            final_collector,
            "CONTINUATION_OPTIONS",
            DOCUMENT.resolve(strict=True),
        ),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    try:
        for module, name, value in replacements:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def _synthetic_finalization(
    *,
    root: Path,
    generated_predictions: Path,
    finalizer_runner: FinalizerRunner,
) -> tuple[Path, int, int]:
    """Drive the real finalizer with fake ancestry and real science engines."""
    fixed5 = _write_control(
        root, "control/FIXED5_SOURCE_MANIFEST_V1.json", "synthetic fixed5\n"
    )
    fixed48 = _write_control(
        root, "control/FIXED48_SOURCE_MANIFEST_V2.json", "synthetic fixed48\n"
    )
    adoption = _write_control(
        root,
        "authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json",
        "synthetic adoption\n",
    )
    authorization = _write_control(
        root,
        "authorization/AUTHORIZATION_MANIFEST_V3.json",
        "synthetic authorization\n",
    )
    feasibility = _write_control(
        root,
        "control/FEASIBILITY_GATE_RECEIPT_V2.json",
        "synthetic feasibility\n",
    )
    prelaunch = _write_control(
        root,
        "control/prelaunch/FIXED5_PRELAUNCH_"
        "00000000000000000000000000000001.json",
        "synthetic prelaunch\n",
    )
    dummy = _write_control(root, "control/synthetic_identity.txt", "synthetic\n")
    nonce = "00000000000000000000000000000001"
    job_id = "1"
    launch = atomic_write_receipt(
        root / f"control/launch/FIXED5_LAUNCH_{nonce}_JOB_{job_id}.json",
        build_receipt(
            schema=launch_receipt.LAUNCH_SCHEMA,
            study_id=finalizer.STUDY_ID,
            scenario=launch_receipt.LAUNCH_SCENARIO,
            identities={
                "prelaunch_receipt": file_identity(prelaunch),
                "synthetic_control": file_identity(dummy),
            },
            fields={"launch_nonce": nonce, "slurm_job_id": job_id},
        ),
    )
    excluded = _write_control(
        root,
        f"control/excluded/FIXED5_EXCLUDED_{nonce}.json",
        "synthetic excluded-state audit\n",
    )
    state = _synthetic_state(root)
    analyzer_calls = 0
    verifier_calls = 0

    def collector(**kwargs: Any) -> Path:
        destination = Path(kwargs["destination"])
        output = _exclusive_copy(generated_predictions, destination)
        receipt_path = output.with_suffix(output.suffix + ".receipt.json")
        receipt_path.write_text("synthetic collection receipt\n")
        return output

    prediction_identity = file_identity(generated_predictions)

    def collection_verifier(path: Path, **kwargs: Any) -> dict[str, Any]:
        receipt_path = Path(kwargs["receipt_path"])
        if path.is_symlink() or not path.is_file() or not receipt_path.is_file():
            raise ValueError("synthetic collection topology differs")
        if file_identity(path)["sha256"] != prediction_identity["sha256"]:
            raise ValueError("synthetic collection bytes differ")
        return {"status": "pass", "synthetic_only": True}

    def analyzer_runner(
        predictions: Path, **_kwargs: Any
    ) -> Mapping[str, Any]:
        nonlocal analyzer_calls
        analyzer_calls += 1
        return analyzer.analyze_predictions(predictions)

    def independent_verifier(
        predictions: Path,
        *,
        analyzer_report: Path,
        collection_receipt: Path,
        source_manifest: Path,
        **_kwargs: Any,
    ) -> Mapping[str, Any]:
        nonlocal verifier_calls
        verifier_calls += 1
        report = verifier.verify(predictions, analyzer_report)
        report["verification_provenance"] = {
            "source_manifest": file_identity(source_manifest),
            "collection_receipt": file_identity(collection_receipt),
            "collected_predictions": file_identity(predictions),
            "analyzer_report": file_identity(analyzer_report),
            "independent_verifier": file_identity(verifier.__file__),
        }
        return report

    kwargs: dict[str, Any] = {
        "production_root": root,
        "fixed5_source_manifest": fixed5,
        "adoption_authorization": adoption,
        "fixed48_source_manifest": fixed48,
        "authorization_manifest": authorization,
        "feasibility_gate": feasibility,
        "launch_receipt": launch,
        "excluded_audit": excluded,
        "continuation_options": DOCUMENT.resolve(strict=True),
        "manifest_verifier": lambda _path: {"status": "synthetic"},
        "state_scanner": lambda **_kwargs: state,
        "excluded_verifier": lambda *_args, **_kwargs: {
            "status": "synthetic"
        },
        "collector": collector,
        "collection_verifier": collection_verifier,
        "analyzer_runner": analyzer_runner,
        "independent_verifier": independent_verifier,
        "lock_checker": lambda _root: None,
    }
    if "launch_verifier" in inspect.signature(finalizer_runner).parameters:
        kwargs["launch_verifier"] = lambda *_args, **_kwargs: {
            "status": "synthetic"
        }
    with _injected_module_roots(root):
        complete = finalizer_runner(**kwargs)
    return complete, analyzer_calls, verifier_calls


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _receipt_identities() -> dict[str, Any]:
    return {
        "preflight": file_identity(Path(__file__)),
        "analyzer": file_identity(analyzer.__file__),
        "independent_verifier": file_identity(verifier.__file__),
        "finalizer": file_identity(finalizer.__file__),
        "continuation_options": file_identity(DOCUMENT),
        "amendments": {
            f"amendment_{index:02d}": file_identity(path)
            for index, path in enumerate(AMENDMENTS, 1)
        },
    }


def run(
    output_root: Path | str,
    *,
    receipt_destination: Path | str | None = None,
    finalizer_runner: FinalizerRunner = finalizer.run,
) -> Path:
    root = _canonical_new_root(output_root)
    generated = root / "synthetic_fixed5_predictions.jsonl"
    row_count, generated_sha = write_predictions(generated)
    synthetic_production = (root / "synthetic_production").resolve()
    synthetic_production.mkdir()
    complete, analyzer_calls, verifier_calls = _synthetic_finalization(
        root=synthetic_production,
        generated_predictions=generated,
        finalizer_runner=finalizer_runner,
    )
    attempt = complete.parent
    predictions = attempt / finalizer.PREDICTIONS_NAME
    analysis_path = attempt / finalizer.ANALYSIS_NAME
    verification_path = attempt / finalizer.VERIFICATION_NAME
    barrier_path = attempt / finalizer.BARRIER_NAME
    analysis = _load_json(analysis_path)
    verification = _load_json(verification_path)
    completion = verify_receipt(
        complete,
        expected_schema=finalizer.COMPLETE_SCHEMA,
        expected_study_id=finalizer.STUDY_ID,
        expected_scenario=finalizer.COMPLETE_SCENARIO,
    )
    semantic = analysis.get("semantic_report", {})
    verification_semantic = verification.get("semantic_report", {})
    if semantic != verification_semantic:
        raise AssertionError("analyzer/verifier semantic reports differ")
    if (
        not verification.get("analyzer_comparison", {}).get("match")
        or semantic.get("row_count") != EXPECTED_ROWS
        or semantic.get("counts", {}).get("combination_count")
        != EXPECTED_COMBINATIONS
        or len(semantic.get("nested_audit", ()))
        != EXPECTED_NESTED_AUDITS
        or semantic.get("decision", {}).get("classification")
        != EXPECTED_CLASSIFICATION
    ):
        raise AssertionError("fixed-five synthetic semantics differ")
    if (
        analyzer_calls != 1
        or verifier_calls != 1
        or completion.get("analyzer_invocation_count") != 1
        or completion.get("independent_verification_passed") is not True
    ):
        raise AssertionError("synthetic finalization invocation topology differs")
    exact_names = {
        finalizer.START_NAME,
        finalizer.PREDICTIONS_NAME,
        finalizer.COLLECTION_RECEIPT_NAME,
        finalizer.BARRIER_NAME,
        finalizer.ANALYSIS_NAME,
        finalizer.VERIFICATION_NAME,
        finalizer.COMPLETE_NAME,
    }
    if {path.name for path in attempt.iterdir()} != exact_names:
        raise AssertionError("synthetic finalization artifact topology differs")
    prediction_sha = file_identity(predictions)["sha256"]
    if prediction_sha != generated_sha:
        raise AssertionError("synthetic collector changed prediction bytes")

    fields = {
        "status": "pass",
        "synthetic_only": True,
        "values_inspected": False,
        "scientific_values_opened": False,
        "real_data_paths_accepted": False,
        "row_count": row_count,
        "combination_count": EXPECTED_COMBINATIONS,
        "nested_audit_count": EXPECTED_NESTED_AUDITS,
        "expected_classification": EXPECTED_CLASSIFICATION,
        "prediction_sha256": prediction_sha,
        "analysis_input_sha256": semantic.get("input_sha256"),
        "verification_input_sha256": verification_semantic.get(
            "input_sha256"
        ),
        "analyzer_invocation_count": analyzer_calls,
        "independent_verifier_invocation_count": verifier_calls,
        "independent_verifier_match": True,
        "finalization_complete": True,
        "finalization_artifact_names": sorted(exact_names),
        "analyzer_barrier_sha256": file_identity(barrier_path)["sha256"],
        "finalization_complete_sha256": file_identity(complete)["sha256"],
    }
    if len({
        fields["prediction_sha256"],
        fields["analysis_input_sha256"],
        fields["verification_input_sha256"],
    }) != 1:
        raise AssertionError("synthetic analyzer/verifier input identities differ")
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities=_receipt_identities(),
        fields=fields,
    )
    destination = Path(
        receipt_destination
        if receipt_destination is not None
        else root / "PREFLIGHT_RECEIPT.json"
    )
    if os.path.lexists(destination):
        raise FileExistsError(f"preflight receipt exists: {destination}")
    parent = destination.parent.absolute()
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise ValueError("preflight receipt destination ancestry redirected")
    destination = parent / destination.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o664)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(receipt) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if destination.is_symlink() or destination.resolve(strict=True) != destination:
        raise ValueError("published preflight receipt redirected")
    verify_preflight_receipt(destination)
    return destination.resolve(strict=True)


def verify_preflight_receipt(path: Path | str) -> dict[str, Any]:
    receipt = verify_receipt(
        path,
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
        "synthetic_only",
        "values_inspected",
        "scientific_values_opened",
        "real_data_paths_accepted",
        "row_count",
        "combination_count",
        "nested_audit_count",
        "expected_classification",
        "prediction_sha256",
        "analysis_input_sha256",
        "verification_input_sha256",
        "analyzer_invocation_count",
        "independent_verifier_invocation_count",
        "independent_verifier_match",
        "finalization_complete",
        "finalization_artifact_names",
        "analyzer_barrier_sha256",
        "finalization_complete_sha256",
    }
    if set(receipt) != expected_keys:
        raise ValueError("preflight receipt field topology differs")
    if receipt.get("identities") != _receipt_identities():
        raise ValueError("preflight receipt source identities differ")
    expected = {
        "status": "pass",
        "synthetic_only": True,
        "values_inspected": False,
        "scientific_values_opened": False,
        "real_data_paths_accepted": False,
        "row_count": EXPECTED_ROWS,
        "combination_count": EXPECTED_COMBINATIONS,
        "nested_audit_count": EXPECTED_NESTED_AUDITS,
        "expected_classification": EXPECTED_CLASSIFICATION,
        "analyzer_invocation_count": 1,
        "independent_verifier_invocation_count": 1,
        "independent_verifier_match": True,
        "finalization_complete": True,
        "finalization_artifact_names": sorted({
            finalizer.START_NAME,
            finalizer.PREDICTIONS_NAME,
            finalizer.COLLECTION_RECEIPT_NAME,
            finalizer.BARRIER_NAME,
            finalizer.ANALYSIS_NAME,
            finalizer.VERIFICATION_NAME,
            finalizer.COMPLETE_NAME,
        }),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"preflight receipt {key} differs")
    for key in (
        "prediction_sha256",
        "analysis_input_sha256",
        "verification_input_sha256",
        "analyzer_barrier_sha256",
        "finalization_complete_sha256",
    ):
        value = receipt.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"preflight receipt {key} is not SHA-256")
    if len({
        receipt["prediction_sha256"],
        receipt["analysis_input_sha256"],
        receipt["verification_input_sha256"],
    }) != 1:
        raise ValueError("preflight receipt input identities differ")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-destination", type=Path)
    args = parser.parse_args(argv)
    result = run(
        args.output_root,
        receipt_destination=args.receipt_destination,
    )
    print(json.dumps({"status": "pass", "receipt": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
