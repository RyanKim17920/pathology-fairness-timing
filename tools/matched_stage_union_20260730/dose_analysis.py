#!/usr/bin/env python3
"""Validate and summarize noninferential matched-stage dose calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any


REPO = Path("/admin/home/ryan.kim/nt")
CONFIG = REPO / "configs_vendor/matched_stage_union_20260730"
EXPECTED_DOSES = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4)
EXPECTED_TARGETS = ("brca", "luad", "ucec", "coad")
EXPECTED_PATIENTS = {"brca": 334, "luad": 281, "ucec": 181, "coad": 176}
OBJECTIVE = REPO / "tools/matched_stage_union_20260730/objectives.py"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        fail(f"missing/empty/symlink JSON: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON root is not an object: {path}")
    return value


def finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        fail(f"{label} must be finite and positive, got {value!r}")
    return number


def validate_result(
    path: Path,
    *,
    target: str,
    task: str,
    dose: float,
    checkpoint: Path,
    head_seed: int,
) -> dict[str, float]:
    result = load_json(path)
    expected = {
        "task": task,
        "sensitive": "race",
        "method": "contrastive",
        "proto_temp": 0.2,
        "race_weight": "none",
        "condition_on_label": False,
        "dose_logging": True,
        "hospital_fold": "target",
        "lambda_adv": dose,
        "n_pool_tiles": 0,
        "epochs": 20,
        "hidden": 128,
        "lr": 0.001,
        "batch_size": 256,
        "embed_batch_size": 128,
        "adversary_data": "task_only",
        "n_cohort_patients": EXPECTED_PATIENTS[target],
    }
    for key, wanted in expected.items():
        if result.get(key) != wanted:
            fail(f"{path}: {key}={result.get(key)!r}, expected {wanted!r}")
    reliable = result.get("reliable_fairness") or {}
    if int(reliable.get("head_seed", -1)) != head_seed:
        fail(f"{path}: head seed mismatch")
    if (reliable.get("study_task") or "") != target:
        fail(f"{path}: study task mismatch")
    objective = reliable.get("shared_objective") or {}
    if Path(objective.get("canonical_path", "")).resolve() != OBJECTIVE.resolve():
        fail(f"{path}: shared objective path mismatch")
    if objective.get("sha256") != sha256_file(OBJECTIVE):
        fail(f"{path}: shared objective hash mismatch")
    if int(objective.get("bytes", -1)) != OBJECTIVE.stat().st_size:
        fail(f"{path}: shared objective byte count mismatch")
    identity = reliable.get("checkpoint_identity") or {}
    if Path(identity.get("canonical_path", "")).resolve() != checkpoint.resolve():
        fail(f"{path}: checkpoint path mismatch")
    if identity.get("sha256") != sha256_file(checkpoint):
        fail(f"{path}: checkpoint hash mismatch")
    manifest_path = checkpoint.parent / "manifest.json"
    manifest_identity = reliable.get("checkpoint_manifest") or {}
    if (
        Path(manifest_identity.get("canonical_path", "")).resolve()
        != manifest_path.resolve()
    ):
        fail(f"{path}: checkpoint manifest path mismatch")
    if manifest_identity.get("sha256") != sha256_file(manifest_path):
        fail(f"{path}: checkpoint manifest hash mismatch")
    manifest = load_json(manifest_path)
    checkpoint_entry = (manifest.get("files") or {}).get(checkpoint.name) or {}
    if checkpoint_entry.get("sha256") != sha256_file(checkpoint):
        fail(f"{path}: checkpoint no longer matches its FM manifest")
    diagnostic = (result.get("runs") or {}).get("debiased", {}).get(
        "dose_diagnostic"
    )
    if not isinstance(diagnostic, dict):
        fail(f"{path}: missing debiased dose diagnostic")
    if diagnostic.get("scope") != "shared_adapter_lin1":
        fail(f"{path}: wrong gradient scope")
    debiased = (result.get("runs") or {}).get("debiased") or {}
    n_train = int(debiased.get("n_train_task_tiles", -1))
    expected_batches = int(result["epochs"]) * math.ceil(
        n_train / int(result["batch_size"])
    )
    if n_train <= 0 or int(diagnostic.get("n_batches", 0)) != expected_batches:
        fail(
            f"{path}: dose batch count mismatch; "
            f"expected {expected_batches}, "
            f"found {diagnostic.get('n_batches')!r}"
        )
    return {
        "grad_ratio": finite_positive(
            diagnostic.get("cumulative_fair_main_grad_ratio"),
            f"{path}: cumulative grad ratio",
        ),
        "loss_ratio": finite_positive(
            diagnostic.get("weighted_fair_main_loss_ratio"),
            f"{path}: weighted loss ratio",
        ),
        "cosine": float(diagnostic["grad_cosine_mean"]),
        "conflict_fraction": float(diagnostic["grad_conflict_fraction"]),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize_seed(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    seed = int(args.fm_seed)
    head_seed = int(args.head_seed)
    rows = []
    task_by_target = {
        "brca": "brca_tp53",
        "luad": "luad_tp53",
        "ucec": "ucec_tp53",
        "coad": "coad_tp53",
    }
    for target in EXPECTED_TARGETS:
        for dose in EXPECTED_DOSES:
            path = root / target / f"dose_{dose:g}" / "result.json"
            diagnostic = validate_result(
                path,
                target=target,
                task=task_by_target[target],
                dose=dose,
                checkpoint=checkpoint,
                head_seed=head_seed,
            )
            rows.append(
                {
                    "target": target,
                    "dose": dose,
                    "result": str(path),
                    "result_sha256": sha256_file(path),
                    **diagnostic,
                }
            )
    by_dose = {}
    for dose in EXPECTED_DOSES:
        selected = [row for row in rows if row["dose"] == dose]
        by_dose[f"{dose:g}"] = {
            "equal_target_median_grad_ratio": statistics.median(
                row["grad_ratio"] for row in selected
            ),
            "equal_target_median_loss_ratio": statistics.median(
                row["loss_ratio"] for row in selected
            ),
            "targets": len(selected),
        }
    payload = {
        "schema": "matched-stage-posthoc-dose-seed/v1",
        "fm_seed": seed,
        "head_seed": head_seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "shared_objective": {
            "canonical_path": str(OBJECTIVE.resolve()),
            "sha256": sha256_file(OBJECTIVE),
            "bytes": OBJECTIVE.stat().st_size,
        },
        "rows": rows,
        "by_dose": by_dose,
    }
    atomic_json(root / "dose_summary.json", payload)
    print(json.dumps({"status": "valid", "rows": len(rows), "by_dose": by_dose}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fm-seed", type=int, required=True)
    parser.add_argument("--head-seed", type=int, required=True)
    return parser


if __name__ == "__main__":
    summarize_seed(build_parser().parse_args())
