"""Receipt-bound adapter diagnostic with task-only nested cross-fitting.

Real outcome discovery is intentionally outside this module. Callers provide
already-authorized patient records and tile bytes; race is attached to output
rows only and is never passed to the task optimizer.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.matched_cancer_diagnostic_20260730.cache import (
    cached_adapter_embeddings,
)
from tools.matched_cancer_stage_20260730.objectives import StageAdapter
from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)
from tools.matched_cancer_stage_20260730.completion_receipt import (
    state_dict_sha256,
)
from tools import reliable_fairness_head as reliable


RUNNER_SCHEMA = "matched-cancer-adapter-diagnostic/v1"
ROOT_SCHEMA = "matched-cancer-adapter-diagnostic-root/v1"
HEAD_SEEDS = (42001, 42002, 42003, 42004)
REPO = Path(__file__).resolve().parents[2]


def _source_identities() -> dict[str, dict[str, Any]]:
    return {
        "runner": file_identity(Path(__file__)),
        "cache": file_identity(Path(__file__).with_name("cache.py")),
        "stage_objectives": file_identity(
            REPO / "tools/matched_cancer_stage_20260730/objectives.py"
        ),
        "receipts": file_identity(
            REPO / "tools/matched_cancer_stage_20260730/receipts.py"
        ),
        "completion_receipt": file_identity(
            REPO / "tools/matched_cancer_stage_20260730/completion_receipt.py"
        ),
        "reliable_cache_rows": file_identity(
            REPO / "tools/reliable_fairness_head.py"
        ),
        "encoder_model": file_identity(
            REPO / "vendor/matched_stage_train_20260730/model.py"
        ),
    }


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    y_true: int
    race: str
    tss: str
    outer_fold: int
    sex: str | None = None
    age: float | None = None
    site: str | None = None


@dataclass
class FrozenRepresentation:
    encoder: nn.Module
    adapter: StageAdapter
    checkpoint: Path
    completion_receipt: Path
    device: torch.device
    mean: torch.Tensor
    std: torch.Tensor
    encoder_sha256: str
    adapter_sha256: str

    def assert_unchanged(self) -> None:
        if state_dict_sha256(self.encoder.state_dict()) != self.encoder_sha256:
            raise RuntimeError("diagnostic mutated the frozen encoder")
        if state_dict_sha256(self.adapter.state_dict()) != self.adapter_sha256:
            raise RuntimeError("diagnostic mutated the frozen stage adapter")


class TaskProbe(nn.Module):
    """Exact diagnostic head: 128 -> 64 -> ReLU -> Dropout(.1) -> 1."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x)))).squeeze(-1)


def _default_encoder_factory(checkpoint: Mapping[str, Any]) -> nn.Module:
    from vendor.matched_stage_train_20260730.model import DinoV2ViT

    variant = (
        checkpoint.get("config", {}).get("model", {}).get(
            "type", "dinov2_vits14_reg"
        )
    )
    return DinoV2ViT(variant=variant)


def load_frozen_representation(
    completion_receipt: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    encoder_factory: Callable[[Mapping[str, Any]], nn.Module] | None = None,
    expected_study_id: str | None = None,
    expected_scenario: str | None = None,
) -> FrozenRepresentation:
    """Load exactly the E+A checkpoint bound by a completion receipt."""
    receipt_path = Path(completion_receipt).resolve()
    receipt = verify_receipt(
        receipt_path,
        expected_schema="matched-cancer-stage-completion/v1",
        expected_study_id=expected_study_id,
        expected_scenario=expected_scenario,
    )
    checkpoint_identity = receipt.get("identities", {}).get("latest_checkpoint")
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("completion receipt lacks latest_checkpoint identity")
    checkpoint_path = Path(checkpoint_identity["canonical_path"])
    if file_identity(checkpoint_path) != dict(checkpoint_identity):
        raise ValueError("completion-bound checkpoint identity drift")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "model" not in checkpoint or "stage_adapter" not in checkpoint:
        raise ValueError("checkpoint must contain model and stage_adapter")
    factory = encoder_factory or _default_encoder_factory
    encoder = factory(checkpoint)
    encoder.load_state_dict(checkpoint["model"], strict=True)
    adapter = StageAdapter()
    adapter.load_state_dict(checkpoint["stage_adapter"], strict=True)
    encoder_sha = state_dict_sha256(encoder.state_dict())
    adapter_sha = state_dict_sha256(adapter.state_dict())
    if encoder_sha != receipt.get("encoder_post_sha256"):
        raise ValueError("encoder state differs from completion receipt")
    if adapter_sha != receipt.get("adapter_post_sha256"):
        raise ValueError("adapter state differs from completion receipt")
    target = torch.device(device)
    encoder.to(target).eval()
    adapter.to(target).eval()
    for module in (encoder, adapter):
        for parameter in module.parameters():
            parameter.requires_grad = False
    data = checkpoint.get("config", {}).get("data", {})
    mean = torch.tensor(
        data.get("mean", [0.485, 0.456, 0.406]), device=target
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        data.get("std", [0.229, 0.224, 0.225]), device=target
    ).view(1, 3, 1, 1)
    result = FrozenRepresentation(
        encoder=encoder,
        adapter=adapter,
        checkpoint=checkpoint_path,
        completion_receipt=receipt_path,
        device=target,
        mean=mean,
        std=std,
        encoder_sha256=encoder_sha,
        adapter_sha256=adapter_sha,
    )
    result.assert_unchanged()
    return result


def embed_tiles(
    representation: FrozenRepresentation,
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode tiles and return per-tile normalized A(E(tile)) plus keep mask."""
    from PIL import Image
    from torchvision.transforms import v2

    transform = v2.Compose(
        [v2.ToImage(), v2.Resize((224, 224), antialias=True),
         v2.ToDtype(torch.float32, scale=True)]
    )
    output = np.zeros((len(tiles), 128), dtype=np.float32)
    keep = np.zeros(len(tiles), dtype=np.bool_)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if representation.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            images, indices = [], []
            for offset, (_, payload) in enumerate(tiles[start:start + batch_size]):
                try:
                    image = Image.open(io.BytesIO(bytes(payload))).convert("RGB")
                except Exception:
                    continue
                images.append(transform(image))
                indices.append(start + offset)
            if not images:
                continue
            x = torch.stack(images).to(representation.device)
            with autocast:
                encoded = representation.encoder.probe_features(
                    (x - representation.mean) / representation.std
                )
                adapted = F.normalize(
                    representation.adapter(encoded.float()), dim=1
                )
            output[indices] = adapted.float().cpu().numpy()
            keep[indices] = True
    representation.assert_unchanged()
    return output, keep


def cache_representation(
    representation: FrozenRepresentation,
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
    cache_dir: str | os.PathLike[str],
    *,
    tag: str,
    embed_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, Path, str]:
    """Cache normalized adapter embeddings with complete E+A provenance."""
    source = {
        "completion_receipt": file_identity(representation.completion_receipt),
        "checkpoint": file_identity(representation.checkpoint),
        "encoder_state_sha256": representation.encoder_sha256,
        "adapter_state_sha256": representation.adapter_sha256,
        "sources": _source_identities(),
    }
    implementation = embed_fn or embed_tiles
    result = cached_adapter_embeddings(
        tag=tag,
        tiles=tiles,
        embed_fn=lambda rows: implementation(representation, rows),
        cache_dir=cache_dir,
        source_identity=source,
    )
    representation.assert_unchanged()
    return result


def pool_patients(
    embeddings: np.ndarray,
    barcodes: Sequence[Any],
    patients: Sequence[PatientRecord],
) -> np.ndarray:
    """Mean-pool cached normalized tile rows in frozen patient order."""
    positions = {patient.patient_id: index for index, patient in enumerate(patients)}
    sums = np.zeros((len(patients), 128), dtype=np.float64)
    counts = np.zeros(len(patients), dtype=np.int64)
    for embedding, barcode in zip(embeddings, barcodes, strict=True):
        patient = str(barcode)
        if patient in positions:
            index = positions[patient]
            sums[index] += embedding
            counts[index] += 1
    if np.any(counts == 0):
        missing = [
            patients[index].patient_id
            for index in np.flatnonzero(counts == 0)
        ]
        raise ValueError(f"patients lack valid cached tiles: {missing[:5]}")
    return (sums / counts[:, None]).astype(np.float32)


def train_task_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    seed: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> np.ndarray:
    """Fit BCE task-only probe; sensitive metadata is not accepted."""
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_eval_t = torch.as_tensor(x_eval, dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = TaskProbe()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(int(seed) + 1)
        for _ in range(int(epochs)):
            order = torch.randperm(len(x_train_t), generator=generator)
            model.train()
            for start in range(0, len(order), int(batch_size)):
                rows = order[start:start + int(batch_size)]
                logits = model(x_train_t[rows])
                loss = F.binary_cross_entropy_with_logits(
                    logits, y_train_t[rows]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(x_eval_t)).numpy()


def _fit_seed(head_seed: int, role: str, outer: int, inner: int | None) -> int:
    payload = f"{head_seed}|{role}|{outer}|{inner}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def nested_crossfit_predictions(
    features: np.ndarray,
    patients: Sequence[PatientRecord],
    *,
    head_seed: int,
    epochs: int = 50,
    fit_fn: Callable[..., np.ndarray] = train_task_probe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit the reliable-fairness N outer + 4N inner patient-row topology."""
    if int(head_seed) not in HEAD_SEEDS:
        raise ValueError(f"head_seed must be one of {HEAD_SEEDS}")
    if features.shape != (len(patients), 128):
        raise ValueError("features must have shape [patients,128]")
    folds = np.asarray([patient.outer_fold for patient in patients], dtype=int)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("patient outer folds must cover exactly 0..4")
    y = np.asarray([patient.y_true for patient in patients], dtype=np.float32)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("patient task labels must be binary")
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    def checked_scores(value: Any, count: int) -> np.ndarray:
        scores = np.asarray(value, dtype=np.float64)
        if (
            scores.shape != (count,)
            or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("probe scores must be finite probabilities")
        return scores

    def metadata(index: int) -> dict[str, Any]:
        patient = patients[index]
        result = {
            "patient_id": patient.patient_id,
            "y_true": int(patient.y_true),
            "race": patient.race,
            "sex": patient.sex,
            "age": patient.age,
            "tss": patient.tss,
        }
        if patient.site is not None:
            result["site"] = patient.site
        return result

    for outer in range(5):
        outer_train = np.flatnonzero(folds != outer)
        outer_eval = np.flatnonzero(folds == outer)
        scores = checked_scores(
            fit_fn(
                features[outer_train], y[outer_train], features[outer_eval],
                seed=_fit_seed(head_seed, "outer", outer, None), epochs=epochs,
            ),
            len(outer_eval),
        )
        for index, score in zip(outer_eval, scores, strict=True):
            rows.append({
                **metadata(int(index)),
                "y_score": float(score),
                "prediction_role": "outer_test",
                "original_fold": int(folds[index]),
                "outer_fold": outer,
            })
        inner_audit = []
        for inner in sorted(set(range(5)) - {outer}):
            inner_train = np.flatnonzero((folds != outer) & (folds != inner))
            inner_eval = np.flatnonzero(folds == inner)
            scores = checked_scores(
                fit_fn(
                    features[inner_train], y[inner_train], features[inner_eval],
                    seed=_fit_seed(head_seed, "inner", outer, inner),
                    epochs=epochs,
                ),
                len(inner_eval),
            )
            for index, score in zip(inner_eval, scores, strict=True):
                rows.append({
                    **metadata(int(index)),
                    "y_score": float(score),
                    "prediction_role": "inner_calibration",
                    "original_fold": int(folds[index]),
                    "calibration_outer_fold": outer,
                    "inner_fold": inner,
                })
            inner_audit.append({
                "inner_fold": inner,
                "excluded_folds": sorted((outer, inner)),
                "train_count": len(inner_train),
                "eval_count": len(inner_eval),
            })
        audit.append({
            "calibration_outer_fold": outer,
            "outer_test": {
                "excluded_folds": [outer],
                "train_count": len(outer_train),
                "eval_count": len(outer_eval),
            },
            "inner_fits": inner_audit,
        })
    reliable._validate_nested_prediction_records(rows)
    reliable._validate_nested_training_audit(audit)
    return rows, audit


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def write_cell(
    *,
    output_dir: str | os.PathLike[str],
    arm: str,
    head_seed: int,
    rows: Sequence[Mapping[str, Any]],
    audit: Sequence[Mapping[str, Any]],
    representation: FrozenRepresentation,
    cache_path: Path,
    cache_entry_sha256: str,
    task_id: str,
    cohort_source: Path,
) -> Path:
    """Atomically publish artifacts and write the receipt last."""
    if arm not in {"B", "P", "H"}:
        raise ValueError("arm must be B, P, or H")
    if head_seed not in HEAD_SEEDS:
        raise ValueError(f"head_seed must be one of {HEAD_SEEDS}")
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"diagnostic cell already exists: {target}")
    target.mkdir(parents=True)
    predictions = target / "nested_predictions.jsonl"
    audit_path = target / "training_audit.jsonl"
    _atomic_jsonl(predictions, rows)
    _atomic_jsonl(audit_path, audit)
    representation.assert_unchanged()
    receipt = build_receipt(
        schema=RUNNER_SCHEMA,
        study_id=verify_receipt(representation.completion_receipt)["study_id"],
        scenario=verify_receipt(representation.completion_receipt)["scenario"],
        identities={
            "completion_receipt": file_identity(
                representation.completion_receipt
            ),
            "checkpoint": file_identity(representation.checkpoint),
            "adapter_cache": file_identity(cache_path),
            "predictions": file_identity(predictions),
            "training_audit": file_identity(audit_path),
            "cohort_source": file_identity(cohort_source),
            "sources": _source_identities(),
        },
        fields={
            "status": "complete",
            "arm": arm,
            "head_seed": int(head_seed),
            "task_id": task_id,
            "head_seeds_contract": list(HEAD_SEEDS),
            "probe": "128->Linear64->ReLU->Dropout0.1->Linear1",
            "optimizer_objective": "BCEWithLogits_task_only",
            "prediction_schema": "nested-crossfit-predictions/v1",
            "prediction_rows": len(rows),
            "outer_rows": sum(
                row["prediction_role"] == "outer_test" for row in rows
            ),
            "inner_rows": sum(
                row["prediction_role"] == "inner_calibration" for row in rows
            ),
            "fit_topology": {"outer": 5, "inner": 20},
            "cache_entry_sha256": cache_entry_sha256,
            "encoder_pre_sha256": representation.encoder_sha256,
            "encoder_post_sha256": state_dict_sha256(
                representation.encoder.state_dict()
            ),
            "adapter_pre_sha256": representation.adapter_sha256,
            "adapter_post_sha256": state_dict_sha256(
                representation.adapter.state_dict()
            ),
        },
    )
    destination = atomic_write_receipt(
        target / "DIAGNOSTIC_RECEIPT.json", receipt
    )
    verify_receipt(destination, expected_schema=RUNNER_SCHEMA)
    return destination


def run_paired_diagnostic(
    *,
    representations: Mapping[str, FrozenRepresentation],
    tiles: Sequence[tuple[Any, bytes | bytearray | memoryview]],
    patients: Sequence[PatientRecord],
    task_id: str,
    cohort_source: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    epochs: int = 50,
    fit_fn: Callable[..., np.ndarray] = train_task_probe,
    embed_fns: Mapping[str, Callable[..., tuple[np.ndarray, np.ndarray]]]
    | None = None,
) -> Path:
    """Run all 3 arms × 4 paired head seeds and seal a root receipt.

    The patient order, labels, folds, fit seeds, and raw tile sequence are one
    shared object across B/P/H. Only the receipt-bound E+A representation differs.
    """
    if set(representations) != {"B", "P", "H"}:
        raise ValueError("representations must contain exactly B, P, and H")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a nonempty string")
    cohort_path = Path(cohort_source).resolve()
    cohort_identity = file_identity(cohort_path)
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"diagnostic output root already exists: {root}")
    root.mkdir(parents=True)

    completion_metadata = {
        arm: verify_receipt(representation.completion_receipt)
        for arm, representation in representations.items()
    }
    studies = {value["study_id"] for value in completion_metadata.values()}
    scenarios = {value["scenario"] for value in completion_metadata.values()}
    if len(studies) != 1 or len(scenarios) != 1:
        raise ValueError("B/P/H completion receipts differ in study or scenario")

    cell_receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("B", "P", "H"):
        representation = representations[arm]
        embed_fn = embed_fns.get(arm) if embed_fns else None
        embeddings, barcodes, cache_path, cache_sha = cache_representation(
            representation,
            tiles,
            cache_dir,
            tag=f"{task_id}-{arm}",
            embed_fn=embed_fn,
        )
        features = pool_patients(embeddings, barcodes, patients)
        cell_receipts[arm] = {}
        for head_seed in HEAD_SEEDS:
            rows, audit = nested_crossfit_predictions(
                features,
                patients,
                head_seed=head_seed,
                epochs=epochs,
                fit_fn=fit_fn,
            )
            destination = write_cell(
                output_dir=root / arm / f"head_seed_{head_seed}",
                arm=arm,
                head_seed=head_seed,
                rows=rows,
                audit=audit,
                representation=representation,
                cache_path=cache_path,
                cache_entry_sha256=cache_sha,
                task_id=task_id,
                cohort_source=cohort_path,
            )
            cell_receipts[arm][str(head_seed)] = file_identity(destination)
        representation.assert_unchanged()

    receipt = build_receipt(
        schema=ROOT_SCHEMA,
        study_id=next(iter(studies)),
        scenario=next(iter(scenarios)),
        identities={
            "cohort_source": cohort_identity,
            "completion_receipts": {
                arm: file_identity(
                    representations[arm].completion_receipt
                )
                for arm in ("B", "P", "H")
            },
            "cells": cell_receipts,
            "sources": _source_identities(),
        },
        fields={
            "status": "complete",
            "task_id": task_id,
            "arms": ["B", "P", "H"],
            "head_seeds": list(HEAD_SEEDS),
            "paired_patient_order_sha256": hashlib.sha256(
                json.dumps(
                    [patient.patient_id for patient in patients],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "cell_count": 12,
            "race_usage": "output_metadata_only",
        },
    )
    destination = atomic_write_receipt(
        root / "ROOT_DIAGNOSTIC_RECEIPT.json", receipt
    )
    verify_receipt(destination, expected_schema=ROOT_SCHEMA)
    return destination


def assert_task_only_api() -> None:
    parameters = set(inspect.signature(train_task_probe).parameters)
    forbidden = {"race", "sex", "age", "tss", "sensitive", "tp53_status"}
    if parameters & forbidden:
        raise RuntimeError("task probe API accepts sensitive metadata")
