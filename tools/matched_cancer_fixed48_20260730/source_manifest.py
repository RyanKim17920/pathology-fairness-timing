#!/usr/bin/env python3
"""Freeze and reverify the exact fixed-48 implementation source closure."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from tools.matched_cancer_stage_20260730.receipts import (
    build_receipt,
    canonical_json_bytes,
    file_identity,
    verify_receipt,
)


REPO = Path(__file__).resolve().parents[2]
STUDY_ID = "matched_cancer_fixed48_20260730"
SCHEMA = "matched-cancer-fixed48-source-manifest/v1"
SCENARIO = "fixed48_frozen_implementation"

# This is an explicit semantic-role allowlist, not a directory walk.  A source
# is added only after it is part of the frozen production call graph.
SOURCE_SPEC: dict[str, Path] = {
    "fixed48.package_init": REPO
    / "tools/matched_cancer_fixed48_20260730/__init__.py",
    "fixed48.source_manifest": Path(__file__),
    "fixed48.contract": REPO
    / "tools/matched_cancer_fixed48_20260730/contract.py",
    "fixed48.runner": REPO
    / "tools/matched_cancer_fixed48_20260730/runner.py",
    "fixed48.calibration_auditor": REPO
    / "tools/matched_cancer_fixed48_20260730/auditor.py",
    "fixed48.calibration_driver": REPO
    / "tools/matched_cancer_fixed48_20260730/calibration_one_seed.sbatch",
    "fixed48.diag_worker": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_worker.py",
    "fixed48.seed_worker": REPO
    / "tools/matched_cancer_fixed48_20260730/seed_worker.py",
    "fixed48.diag_contract": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_contract.py",
    "fixed48.diag_authorization": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_authorization.py",
    "fixed48.diag_deployment": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_deployment.py",
    "fixed48.diag_loader": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_loader.py",
    "fixed48.diag_exporter": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_exporter.py",
    "fixed48.diag_structural_auditor": REPO
    / "tools/matched_cancer_fixed48_20260730/diag_structural_auditor.py",
    "fixed48.serial_controller": REPO
    / "tools/matched_cancer_fixed48_20260730/serial_controller.py",
    "fixed48.serial_driver": REPO
    / "tools/matched_cancer_fixed48_20260730/serial_fixed48.sbatch",
    "fixed48.safe_submit": REPO
    / "tools/matched_cancer_fixed48_20260730/safe_submit.sh",
    "fixed48.feasibility_gate": REPO
    / "tools/matched_cancer_fixed48_20260730/feasibility_gate.py",
    "fixed48.final_collector": REPO
    / "tools/matched_cancer_fixed48_20260730/final_collector.py",
    "fixed48.full_cardinality_preflight": REPO
    / "tools/matched_cancer_fixed48_20260730/full_cardinality_preflight.py",
    "runtime.python_binary": Path(
        "/admin/home/ryan.kim/.local/share/uv/python/"
        "cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
    ),
    "runtime.pyvenv_config": Path(
        "/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg"
    ),
    "config.calibration_plan": REPO
    / "configs_vendor/matched_cancer_fixed48_20260730/calibration_contract.yaml",
    "config.seed_plan": REPO
    / "configs_vendor/matched_cancer_fixed48_20260730/calibration_seed_plan.yaml",
    "config.base_template": REPO
    / "configs_vendor/matched_cancer_fixed48_20260730/calibration_base_template.yaml",
    "config.population": REPO
    / "configs_vendor/matched_cancer_stage_20260730/population_cancer_race.csv",
    "config.exclusions": REPO
    / "configs_vendor/matched_stage_union_20260730/exclude_union_target_hospitals.txt",
    "pretrained.dinov2_vits14_reg": Path(
        "/data/ryan.kim/nanopath/reruns/"
        "matched_cancer_fixed48_20260730/control/torch_home/"
        "hub/checkpoints/"
        "dinov2_vits14_reg4_pretrain.pth"
    ),
    "protocol.execution": REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/FIXED48_EXECUTION_PROTOCOL.md",
    "protocol.final_lock": REPO
    / "results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_LOCK.md",
    "protocol.amendment_01": REPO
    / "results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_AMENDMENT_01.md",
    "protocol.full_cardinality_preflight_receipt": REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/"
    "FULL_CARDINALITY_SYNTHETIC_PREFLIGHT_RECEIPT.json",
    "protocol.prelaunch_incident_record": REPO
    / "results/matched_cancer_stage_20260730/fixed48_execution/"
    "PRELAUNCH_INCIDENT_RECORD.md",
    "legacy.stage_config_builder": REPO
    / "tools/matched_cancer_stage_20260730/config_builder.py",
    "legacy.stage_package_init": REPO
    / "tools/matched_cancer_stage_20260730/__init__.py",
    "legacy.stage_manifest_builder": REPO
    / "tools/matched_cancer_stage_20260730/manifest_builder.py",
    "legacy.stage_completion_receipt": REPO
    / "tools/matched_cancer_stage_20260730/completion_receipt.py",
    "legacy.stage_replay": REPO
    / "tools/matched_cancer_stage_20260730/replay.py",
    "legacy.stage_objectives": REPO
    / "tools/matched_cancer_stage_20260730/objectives.py",
    "legacy.stage_receipts": REPO
    / "tools/matched_cancer_stage_20260730/receipts.py",
    "legacy.union_package_init": REPO
    / "tools/matched_stage_union_20260730/__init__.py",
    "legacy.union_objectives": REPO
    / "tools/matched_stage_union_20260730/objectives.py",
    "legacy.union_instrumentation": REPO
    / "tools/matched_stage_union_20260730/instrumentation.py",
    "legacy.diagnostic_package_init": REPO
    / "tools/matched_cancer_diagnostic_20260730/__init__.py",
    "legacy.diagnostic_deployment": REPO
    / "tools/matched_cancer_diagnostic_20260730/deployment.py",
    "legacy.diagnostic_integration": REPO
    / "tools/matched_cancer_diagnostic_20260730/integration.py",
    "legacy.diagnostic_loader": REPO
    / "tools/matched_cancer_diagnostic_20260730/vetted_loader.py",
    "legacy.diagnostic_runner": REPO
    / "tools/matched_cancer_diagnostic_20260730/runner.py",
    "legacy.diagnostic_cache": REPO
    / "tools/matched_cancer_diagnostic_20260730/cache.py",
    "legacy.diagnostic_exporter": REPO
    / "tools/matched_cancer_diagnostic_20260730/exporter.py",
    "legacy.reliable_fairness_head": REPO / "tools/reliable_fairness_head.py",
    "legacy.fairness_eval": REPO / "tools/fairness_eval.py",
    "legacy.post_hoc_debias": REPO / "tools/post_hoc_debias.py",
    "legacy.hf_tiles": REPO / "tools/hf_tiles.py",
    "legacy.final_analyzer": REPO
    / "tools/matched_cancer_diagnostic_20260730/analyzer.py",
    "legacy.independent_final_verifier": REPO
    / "tools/matched_cancer_diagnostic_20260730/verifier.py",
    "vendor.train": REPO / "vendor/matched_stage_train_20260730/train.py",
    "vendor.dataloader": REPO
    / "vendor/matched_stage_train_20260730/dataloader.py",
    "vendor.model": REPO / "vendor/matched_stage_train_20260730/model.py",
    "vendor.probe": REPO / "vendor/matched_stage_train_20260730/probe.py",
}


def _normalized_spec(spec: Mapping[str, Path | str]) -> dict[str, Path]:
    if not spec:
        raise ValueError("source spec may not be empty")
    result: dict[str, Path] = {}
    seen: set[Path] = set()
    for role, raw_path in spec.items():
        if not isinstance(role, str) or not role:
            raise ValueError("source roles must be nonempty strings")
        candidate = Path(raw_path)
        if candidate.is_symlink():
            raise ValueError(f"source may not be a symlink: {candidate}")
        path = candidate.resolve(strict=True)
        if path in seen:
            raise ValueError(f"source path has multiple semantic roles: {path}")
        seen.add(path)
        result[role] = path
    return result


def _module_candidates(module: str) -> tuple[Path, Path]:
    stem = REPO.joinpath(*module.split("."))
    return stem.with_suffix(".py"), stem / "__init__.py"


def _resolve_local_imports(source: Path) -> set[Path]:
    """Resolve imports that target this repository; external imports are ignored."""
    if source.suffix != ".py":
        return set()
    tree = ast.parse(source.read_text(), filename=str(source))
    local: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[tuple[str, bool]] = []
        if isinstance(node, ast.Import):
            modules.extend((alias.name, True) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                try:
                    package_parts = list(source.relative_to(REPO).with_suffix("").parts)
                except ValueError as error:
                    raise ValueError(f"Python source is outside repository: {source}") from error
                # Drop the module filename. One dot means the current package.
                package_parts = package_parts[:-node.level]
                if node.module:
                    package_parts.extend(node.module.split("."))
                base = ".".join(package_parts)
            elif node.module:
                base = node.module
            else:
                base = ""
            if base:
                modules.append((base, False))
                modules.extend(
                    (f"{base}.{alias.name}", False)
                    for alias in node.names
                    if alias.name != "*"
                )
        resolved_any = False
        considered_local = False
        for module, required in modules:
            if not module:
                continue
            root_name = module.split(".", 1)[0]
            if root_name in {"tools", "vendor"}:
                candidates = _module_candidates(module)
                considered_local = True
            else:
                relative = Path(*module.split("."))
                candidates = (
                    (source.parent / relative).with_suffix(".py"),
                    source.parent / relative / "__init__.py",
                    (REPO / relative).with_suffix(".py"),
                    REPO / relative / "__init__.py",
                    (REPO / "tools" / relative).with_suffix(".py"),
                    REPO / "tools" / relative / "__init__.py",
                    (
                        REPO
                        / "vendor/matched_stage_train_20260730"
                        / relative
                    ).with_suffix(".py"),
                    (
                        REPO
                        / "vendor/matched_stage_train_20260730"
                        / relative
                        / "__init__.py"
                    ),
                )
            resolved = next(
                (path.resolve() for path in candidates if path.is_file()), None
            )
            if resolved is None:
                if required and root_name in {"tools", "vendor"}:
                    raise ValueError(
                        f"repository import does not resolve to a regular file: "
                        f"{source}: {module}"
                    )
                continue
            considered_local = True
            resolved_any = True
            local.add(resolved)
        if considered_local and not resolved_any:
            raise ValueError(
                f"repository import does not resolve to a regular file: {source}"
            )
    return local


def validate_import_closure(spec: Mapping[str, Path | str]) -> list[dict[str, str]]:
    normalized = _normalized_spec(spec)
    allowed = set(normalized.values())
    edges: list[dict[str, str]] = []
    for role, source in normalized.items():
        for imported in sorted(_resolve_local_imports(source)):
            if imported not in allowed:
                raise ValueError(
                    f"unallowlisted local import from {role}: {imported}"
                )
            edges.append(
                {
                    "from_role": role,
                    "to_role": next(
                        target_role
                        for target_role, target in normalized.items()
                        if target == imported
                    ),
                }
            )
    return sorted(edges, key=lambda edge: (edge["from_role"], edge["to_role"]))


def create_manifest(
    destination: Path | str,
    *,
    spec: Mapping[str, Path | str] = SOURCE_SPEC,
) -> Path:
    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite source manifest: {output}")
    normalized = _normalized_spec(spec)
    edges = validate_import_closure(normalized)
    receipt = build_receipt(
        schema=SCHEMA,
        study_id=STUDY_ID,
        scenario=SCENARIO,
        identities={
            "sources": {
                role: file_identity(path) for role, path in normalized.items()
            }
        },
        fields={
            "status": "frozen",
            "source_roles": sorted(normalized),
            "local_import_edges": edges,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is exclusive: a concurrent creator cannot be
        # overwritten between the initial existence check and publication.
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    verify_manifest(output, spec=normalized)
    return output.resolve()


def verify_manifest(
    path: Path | str,
    *,
    spec: Mapping[str, Path | str] = SOURCE_SPEC,
) -> dict[str, Any]:
    normalized = _normalized_spec(spec)
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
        "source_roles",
        "local_import_edges",
    }
    if set(receipt) != expected_keys:
        raise ValueError("source manifest field topology differs")
    identities = receipt["identities"]
    if set(identities) != {"sources"}:
        raise ValueError("source manifest identity topology differs")
    sources = identities["sources"]
    if set(sources) != set(normalized):
        raise ValueError("source manifest allowlist topology differs")
    for role, source in normalized.items():
        identity = sources[role]
        if Path(identity["canonical_path"]) != source:
            raise ValueError(f"source role {role} was redirected")
        if identity != file_identity(source):
            raise ValueError(f"source role {role} drifted")
    edges = validate_import_closure(normalized)
    if receipt["status"] != "frozen":
        raise ValueError("source manifest status differs")
    if receipt["source_roles"] != sorted(normalized):
        raise ValueError("source manifest role list differs")
    if receipt["local_import_edges"] != edges:
        raise ValueError("source manifest import closure differs")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--destination", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        create_manifest(args.destination)
    else:
        verify_manifest(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
