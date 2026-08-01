"""Pure, deterministic analysis primitives for the fixed-five representation audit.

This module deliberately has no checkpoint, extraction, filesystem, or command-line
code.  Its inputs are in-memory diagnosis-free embedding records and its outputs are
JSON-compatible semantic reports that can be recomputed by an independent verifier.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .contract import (
    ANALYSIS_SCHEMA,
    CANCERS,
    CANCER_FOLD_HASH_PREFIX,
    CANCER_PROBE_OUTER_FOLDS,
    C_GRID,
    DIAGNOSIS_FIELD_DENYLIST,
    FM_SEEDS,
    GATE_ELIGIBLE_CONTRASTS,
    KNN_K,
    METADATA_ALLOWLIST,
    PRIMARY_GATE,
    PROBE_LEVELS,
    RACES,
    SCHEMA,
    SOLVER_SEED,
    STUDY_ID,
    TILE_VIEWS,
    TILES_PER_VIEW,
)


PROBE_SCHEMA = "matched-cancer-representation-probe/v1"
GEOMETRY_SCHEMA = "matched-cancer-representation-geometry/v1"
GATE_SCHEMA = "matched-cancer-representation-primary-gate/v1"
ALLOWED_METADATA_FIELDS = METADATA_ALLOWLIST
FORBIDDEN_DIAGNOSIS_FIELDS = DIAGNOSIS_FIELD_DENYLIST
PROBE_SEED = SOLVER_SEED
VIEWS = TILE_VIEWS
LEVELS = PROBE_LEVELS
LEAKAGE_DELTA = float(PRIMARY_GATE["race_minimum_reduction"])
CANCER_LOSS_MAX = float(PRIMARY_GATE["cancer_maximum_loss"])
_TIE_TOLERANCE = 1e-12


class AnalysisError(ValueError):
    """Raised when an input violates the frozen analysis contract."""


def _canonical_field(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def reject_diagnosis_fields(payload: object, *, path: str = "input") -> None:
    """Recursively reject diagnosis/outcome fields before any metric is computed."""
    forbidden = {_canonical_field(field) for field in FORBIDDEN_DIAGNOSIS_FIELDS}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = _canonical_field(key)
            if (
                (normalized in forbidden and normalized != "diagnosis_free")
                or "tp53" in normalized
                or ("diagnosis" in normalized and normalized != "diagnosis_free")
                or "outcome" in normalized
                or normalized.endswith("y_true")
            ):
                raise AnalysisError(f"{path}: forbidden diagnosis field {key!r}")
            reject_diagnosis_fields(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            reject_diagnosis_fields(value, path=f"{path}[{index}]")


def diagnosis_free_metadata(record: Mapping[str, object]) -> dict[str, str]:
    """Return exactly the four allowed sanitized metadata fields.

    The contract's allowed-field declaration is checked as well as the concrete
    audit fields so a contract drift cannot silently broaden the analyzer schema.
    """
    reject_diagnosis_fields(record)
    allowed = set(ALLOWED_METADATA_FIELDS)
    keys = set(record)
    if keys != allowed:
        raise AnalysisError(
            "metadata must contain exactly the diagnosis-free fields; "
            f"missing={sorted(allowed - keys)!r}, extra={sorted(keys - allowed)!r}"
        )
    required = {"patient_id", "cancer", "race", "tss"}
    if allowed != required:
        raise AnalysisError(
            "contract ALLOWED_METADATA_FIELDS must be exactly "
            "patient_id/cancer/race/tss"
        )
    result = {field: str(record[field]).strip() for field in sorted(required)}
    if any(not value for value in result.values()):
        raise AnalysisError("diagnosis-free metadata fields must be nonempty")
    if result["cancer"] not in CANCERS:
        raise AnalysisError(f"unexpected cancer {result['cancer']!r}")
    if result["race"] not in RACES:
        raise AnalysisError(f"unexpected race {result['race']!r}")
    return result


def _finite_matrix(values: object, *, context: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context}: embeddings must have one common dimension") from error
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise AnalysisError(f"{context}: embeddings must be a nonempty 2-D matrix")
    if not np.isfinite(array).all():
        raise AnalysisError(f"{context}: embeddings must be finite")
    return array


def _records_to_arrays(
    records: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, list[dict[str, str]], list[str]]:
    reject_diagnosis_fields(records)
    if not records:
        raise AnalysisError("at least one embedding record is required")
    metadata: list[dict[str, str]] = []
    embeddings: list[object] = []
    tile_keys: list[str] = []
    for index, record in enumerate(records):
        if "metadata" in record:
            raw_metadata = record["metadata"]
            if not isinstance(raw_metadata, Mapping):
                raise AnalysisError(f"record {index}: metadata must be a mapping")
            meta = diagnosis_free_metadata(raw_metadata)
        else:
            try:
                raw_metadata = {field: record[field] for field in ALLOWED_METADATA_FIELDS}
            except KeyError as error:
                raise AnalysisError(f"record {index}: missing {error.args[0]!r}") from error
            meta = diagnosis_free_metadata(raw_metadata)
        if "embedding" not in record:
            raise AnalysisError(f"record {index}: missing embedding")
        embeddings.append(record["embedding"])
        metadata.append(meta)
        explicit = record.get("tile_id", record.get("tile_key"))
        if explicit is None:
            digest = hashlib.sha256(
                np.asarray(record["embedding"], dtype=np.float64).tobytes()
            ).hexdigest()
            explicit = digest
        tile_keys.append(f"{meta['patient_id']}|{explicit}")
    matrix = _finite_matrix(embeddings, context="records")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise AnalysisError("every tile representation must have finite nonzero L2 norm")
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=0.0):
        raise AnalysisError("every tile representation must already be per-tile L2 normalized")
    # Remove only floating-point roundoff after enforcing the cache boundary.
    matrix = matrix / norms[:, None]
    if len(set(tile_keys)) != len(tile_keys):
        raise AnalysisError("tile identities must be unique within an analysis input")
    order = sorted(range(len(tile_keys)), key=lambda index: tile_keys[index])
    matrix = matrix[order]
    metadata = [metadata[index] for index in order]
    tile_keys = [tile_keys[index] for index in order]
    by_patient: dict[str, tuple[str, str, str]] = {}
    for meta in metadata:
        identity = (meta["cancer"], meta["race"], meta["tss"])
        old = by_patient.setdefault(meta["patient_id"], identity)
        if old != identity:
            raise AnalysisError("patient metadata differs across embedding records")
    return matrix, metadata, tile_keys


def _patient_means(
    matrix: np.ndarray, metadata: Sequence[Mapping[str, str]]
) -> tuple[np.ndarray, list[dict[str, str]]]:
    indices: dict[str, list[int]] = defaultdict(list)
    exemplar: dict[str, dict[str, str]] = {}
    for index, meta in enumerate(metadata):
        patient = meta["patient_id"]
        indices[patient].append(index)
        exemplar[patient] = dict(meta)
    patients = sorted(indices)
    means = np.stack(
        [np.mean(matrix[indices[patient]], axis=0) for patient in patients]
    )
    return means, [exemplar[patient] for patient in patients]


def _patient_weights(metadata: Sequence[Mapping[str, str]]) -> np.ndarray:
    counts = Counter(meta["patient_id"] for meta in metadata)
    return np.asarray([1.0 / counts[meta["patient_id"]] for meta in metadata])


def _weighted_standardize_fit(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(matrix) != len(weights) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise AnalysisError("standardization weights must be finite and positive")
    mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    return mean, scale


def _fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    c_value: float,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    if len(np.unique(train_y)) != 2:
        raise AnalysisError("logistic training partition must contain both classes")
    mean, scale = _weighted_standardize_fit(train_x, weights)
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        random_state=PROBE_SEED,
        max_iter=10_000,
        tol=1e-10,
    )
    model.fit((train_x - mean) / scale, train_y, sample_weight=weights)
    return model, mean, scale


def _predict(model: LogisticRegression, mean: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba((x - mean) / scale)[:, 1], dtype=float)


def _aggregate_patient_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Mapping[str, str]],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, meta in enumerate(metadata):
        grouped[meta["patient_id"]].append(index)
    patients = sorted(grouped)
    patient_scores: list[float] = []
    patient_labels: list[int] = []
    patient_tss: list[str] = []
    for patient in patients:
        positions = grouped[patient]
        unique_labels = set(int(labels[index]) for index in positions)
        if len(unique_labels) != 1:
            raise AnalysisError("class differs across tiles for one patient")
        patient_scores.append(float(np.mean(scores[positions])))
        patient_labels.append(unique_labels.pop())
        patient_tss.append(metadata[positions[0]]["tss"])
    return patients, np.asarray(patient_labels), np.asarray(patient_scores), patient_tss


def _auc(labels: Sequence[int], scores: Sequence[float], *, context: str) -> float:
    if len(set(int(value) for value in labels)) != 2:
        raise AnalysisError(f"{context}: pooled predictions do not contain both classes")
    return float(roc_auc_score(np.asarray(labels), np.asarray(scores)))


def _smallest_best_c(scores: Mapping[float, float]) -> float:
    if set(scores) != set(float(value) for value in C_GRID):
        raise AnalysisError("C-selection score set differs from the frozen grid")
    best = max(scores.values())
    return min(c for c, score in scores.items() if math.isclose(score, best, rel_tol=0.0, abs_tol=_TIE_TOLERANCE))


def _race_inner_score(
    x: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Mapping[str, str]],
    weights: np.ndarray,
    c_value: float,
) -> tuple[float, int, int]:
    predictions: list[tuple[str, int, float]] = []
    valid = 0
    excluded = 0
    for inner_tss in sorted({meta["tss"] for meta in metadata}):
        test = np.asarray([meta["tss"] == inner_tss for meta in metadata])
        train = ~test
        if len(np.unique(labels[test])) != 2 or len(np.unique(labels[train])) != 2:
            excluded += 1
            continue
        model, mean, scale = _fit_logistic(x[train], labels[train], weights[train], c_value)
        raw = _predict(model, mean, scale, x[test])
        pids, ys, ps, _ = _aggregate_patient_predictions(
            raw, labels[test], [metadata[index] for index in np.flatnonzero(test)]
        )
        predictions.extend(zip(pids, ys.tolist(), ps.tolist()))
        valid += 1
    if valid < 2:
        raise AnalysisError("outer fit retained fewer than two valid inner TSS folds")
    predictions.sort(key=lambda item: item[0])
    score = _auc([item[1] for item in predictions], [item[2] for item in predictions], context="nested race selection")
    return score, valid, excluded


def nested_race_probe(
    records: Sequence[Mapping[str, object]], *, level: str
) -> dict[str, Any]:
    """Cancer-conditioned nested leave-one-TSS-out race probe.

    Patient probes mean-pool first. Tile probes train with total weight one per
    patient and average held-out tile probabilities back to patients. In both
    cases, only pooled outer held-out patient predictions are scored.
    """
    reject_diagnosis_fields(records)
    if level not in LEVELS:
        raise AnalysisError(f"level must be one of {tuple(LEVELS)!r}")
    matrix, metadata, _ = _records_to_arrays(records)
    cancers = {meta["cancer"] for meta in metadata}
    if len(cancers) != 1:
        raise AnalysisError("race probe input must contain exactly one cancer")
    counts = Counter(meta["patient_id"] for meta in metadata)
    if set(counts.values()) != {TILES_PER_VIEW}:
        raise AnalysisError(
            f"race probe requires exactly {TILES_PER_VIEW} tiles per patient"
        )
    if level == "patient":
        matrix, metadata = _patient_means(matrix, metadata)
        weights = np.ones(len(matrix), dtype=float)
    else:
        weights = _patient_weights(metadata)
    labels = np.asarray([1 if meta["race"] == RACES[1] else 0 for meta in metadata])
    outer_tss = sorted({meta["tss"] for meta in metadata})
    pooled: list[tuple[str, str, int, float]] = []
    audits: list[dict[str, Any]] = []
    for heldout_tss in outer_tss:
        test = np.asarray([meta["tss"] == heldout_tss for meta in metadata])
        train = ~test
        if len(np.unique(labels[train])) != 2:
            raise AnalysisError(f"outer training partition {heldout_tss!r} lacks both races")
        train_meta = [metadata[index] for index in np.flatnonzero(train)]
        c_scores: dict[float, float] = {}
        valid_folds: int | None = None
        excluded_folds: int | None = None
        for raw_c in C_GRID:
            c_value = float(raw_c)
            score, valid, excluded = _race_inner_score(
                matrix[train], labels[train], train_meta, weights[train], c_value
            )
            c_scores[c_value] = score
            valid_folds = valid
            excluded_folds = excluded
        selected = _smallest_best_c(c_scores)
        model, mean, scale = _fit_logistic(matrix[train], labels[train], weights[train], selected)
        raw = _predict(model, mean, scale, matrix[test])
        pids, ys, ps, patient_tss = _aggregate_patient_predictions(
            raw, labels[test], [metadata[index] for index in np.flatnonzero(test)]
        )
        pooled.extend(zip(pids, patient_tss, ys.tolist(), ps.tolist()))
        audits.append(
            {
                "heldout_tss": heldout_tss,
                "selected_c": selected,
                "inner_pooled_patient_auroc_by_c": {
                    str(c): c_scores[c] for c in sorted(c_scores)
                },
                "valid_inner_folds": valid_folds,
                "excluded_inner_folds_lacking_both_races": excluded_folds,
                "train_patient_count": len({meta["patient_id"] for meta in train_meta}),
                "heldout_patient_count": len(pids),
            }
        )
    pooled.sort(key=lambda item: item[0])
    if len({item[0] for item in pooled}) != len(pooled):
        raise AnalysisError("each patient must receive exactly one outer held-out prediction")
    auc = _auc([item[2] for item in pooled], [item[3] for item in pooled], context="outer race probe")
    return {
        "schema": PROBE_SCHEMA,
        "probe": "race",
        "cancer": next(iter(cancers)),
        "level": level,
        "positive_class": RACES[1],
        "pooled_heldout_patient_auroc": auc,
        "oriented_leakage": max(auc, 1.0 - auc) - 0.5,
        "patient_count": len(pooled),
        "input_embedding_count": len(records),
        "outer_tss_count": len(outer_tss),
        "outer_folds": audits,
        "predictions": [
            {"patient_id": patient, "tss": tss, "race_index": label, "score_white": score}
            for patient, tss, label, score in pooled
        ],
    }


def _cancer_block_assignment(
    metadata: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    """Assign whole TSSs to the five frozen, cancer-stratified hash folds."""
    by_class: dict[str, set[str]] = defaultdict(set)
    tss_class: dict[str, str] = {}
    for meta in metadata:
        label = meta["cancer"]
        tss = meta["tss"]
        old = tss_class.setdefault(tss, label)
        if old != label:
            raise AnalysisError(f"TSS {tss!r} spans multiple cancers")
        by_class[label].add(tss)
    if set(by_class) != set(CANCERS):
        raise AnalysisError("blocked probe requires the exact cancer set")
    assignment: dict[str, int] = {}
    for cancer in CANCERS:
        ranked = sorted(
            by_class[cancer],
            key=lambda tss: (
                hashlib.sha256(
                    f"{CANCER_FOLD_HASH_PREFIX}|{PROBE_SEED}|{cancer}|{tss}".encode("utf-8")
                ).hexdigest(),
                tss,
            ),
        )
        for rank, tss in enumerate(ranked):
            assignment[tss] = rank % CANCER_PROBE_OUTER_FOLDS
    if set(assignment) != set(tss_class):
        raise AnalysisError("cancer fold assignment lost a TSS")
    return assignment


def _blocked_inner_score(
    x: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Mapping[str, str]],
    fold_labels: Sequence[int],
    c_value: float,
) -> tuple[float, int, int]:
    predictions: list[tuple[str, int, float]] = []
    weights = np.ones(len(x), dtype=float)
    valid = 0
    excluded = 0
    for heldout_label in sorted(set(fold_labels)):
        test = np.asarray([label == heldout_label for label in fold_labels])
        train = ~test
        if len(np.unique(labels[test])) != 2 or len(np.unique(labels[train])) != 2:
            excluded += 1
            continue
        model, mean, scale = _fit_logistic(x[train], labels[train], weights[train], c_value)
        scores = _predict(model, mean, scale, x[test])
        predictions.extend(
            (metadata[index]["patient_id"], int(labels[index]), float(score))
            for index, score in zip(np.flatnonzero(test), scores, strict=True)
        )
        valid += 1
    if valid < 2:
        raise AnalysisError("cancer probe retained fewer than two valid inner grouped folds")
    predictions.sort(key=lambda item: item[0])
    return (
        _auc([p[1] for p in predictions], [p[2] for p in predictions], context="nested cancer selection"),
        valid,
        excluded,
    )


def pooled_cancer_probe(records: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Nested, patient-level BRCA-vs-LUAD probe with all splits blocked by TSS."""
    reject_diagnosis_fields(records)
    matrix, metadata, _ = _records_to_arrays(records)
    matrix, metadata = _patient_means(matrix, metadata)
    if set(meta["cancer"] for meta in metadata) != set(CANCERS):
        raise AnalysisError("cancer probe requires the exact cancer set")
    labels = np.asarray([1 if meta["cancer"] == CANCERS[1] else 0 for meta in metadata])
    assignment = _cancer_block_assignment(metadata)
    folds = [
        sorted(tss for tss, fold in assignment.items() if fold == fold_index)
        for fold_index in range(CANCER_PROBE_OUTER_FOLDS)
    ]
    if any(not fold for fold in folds):
        raise AnalysisError("each frozen cancer-probe fold must contain a TSS")
    pooled: list[tuple[str, str, int, float]] = []
    audits: list[dict[str, Any]] = []
    weights = np.ones(len(matrix), dtype=float)
    for outer_fold, heldout in enumerate(folds):
        test = np.asarray([meta["tss"] in heldout for meta in metadata])
        train = ~test
        train_meta = [metadata[index] for index in np.flatnonzero(train)]
        train_fold_labels = [assignment[meta["tss"]] for meta in train_meta]
        scores_by_c: dict[float, float] = {}
        valid_inner_count: int | None = None
        excluded_inner_count: int | None = None
        for raw_c in C_GRID:
            score, valid_inner, excluded_inner = _blocked_inner_score(
                matrix[train], labels[train], train_meta, train_fold_labels, float(raw_c)
            )
            scores_by_c[float(raw_c)] = score
            valid_inner_count = valid_inner
            excluded_inner_count = excluded_inner
        selected = _smallest_best_c(scores_by_c)
        model, mean, scale = _fit_logistic(matrix[train], labels[train], weights[train], selected)
        scores = _predict(model, mean, scale, matrix[test])
        for index, score in zip(np.flatnonzero(test), scores, strict=True):
            pooled.append((metadata[index]["patient_id"], metadata[index]["tss"], int(labels[index]), float(score)))
        audits.append(
            {
                "heldout_tss": heldout,
                "outer_fold": outer_fold,
                "selected_c": selected,
                "inner_pooled_patient_auroc_by_c": {str(c): scores_by_c[c] for c in sorted(scores_by_c)},
                "valid_inner_grouped_folds": valid_inner_count,
                "excluded_inner_grouped_folds": excluded_inner_count,
                "heldout_patient_count": int(test.sum()),
            }
        )
    pooled.sort(key=lambda item: item[0])
    if len({item[0] for item in pooled}) != len(pooled):
        raise AnalysisError("cancer probe patients were not evaluated exactly once")
    auc = _auc([p[2] for p in pooled], [p[3] for p in pooled], context="outer cancer probe")
    return {
        "schema": PROBE_SCHEMA,
        "probe": "cancer",
        "positive_class": CANCERS[1],
        "pooled_heldout_patient_auroc": auc,
        "patient_count": len(pooled),
        "outer_block_fold_count": CANCER_PROBE_OUTER_FOLDS,
        "outer_folds": audits,
        "predictions": [
            {"patient_id": patient, "tss": tss, "cancer_index": label, "score_second_cancer": score}
            for patient, tss, label, score in pooled
        ],
    }


def cancer_conditioned_energy_distance(records: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    """Patient-level multivariate energy distance between races, by cancer."""
    reject_diagnosis_fields(records)
    matrix, metadata, _ = _records_to_arrays(records)
    matrix, metadata = _patient_means(matrix, metadata)
    per_cancer: dict[str, dict[str, Any]] = {}
    for cancer in CANCERS:
        black = matrix[[meta["cancer"] == cancer and meta["race"] == RACES[0] for meta in metadata]]
        white = matrix[[meta["cancer"] == cancer and meta["race"] == RACES[1] for meta in metadata]]
        if not len(black) or not len(white):
            raise AnalysisError(f"{cancer}: energy distance requires both races")
        if len(black) < 2 or len(white) < 2:
            raise AnalysisError(f"{cancer}: unbiased energy distance requires at least two patients per race")
        black_dist = cdist(black, black)
        white_dist = cdist(white, white)
        black_within = float(black_dist.sum() / (len(black) * (len(black) - 1)))
        white_within = float(white_dist.sum() / (len(white) * (len(white) - 1)))
        value = float(2 * np.mean(cdist(black, white)) - black_within - white_within)
        if value < 0 and abs(value) <= _TIE_TOLERANCE:
            value = 0.0
        per_cancer[cancer] = {
            "energy_distance": value,
            "black_patient_count": len(black),
            "white_patient_count": len(white),
        }
    return {
        "schema": GEOMETRY_SCHEMA,
        "metric": "cancer_conditioned_patient_energy_distance",
        "per_cancer": per_cancer,
        "macro_cancer_mean": float(np.mean([entry["energy_distance"] for entry in per_cancer.values()])),
    }


def cosine_knn_cross_race_mixing(
    records: Sequence[Mapping[str, object]], *, level: str, k: int = KNN_K
) -> dict[str, Any]:
    """Cosine kNN cross-race fraction, excluding the query patient's samples."""
    reject_diagnosis_fields(records)
    if level not in LEVELS:
        raise AnalysisError(f"level must be one of {tuple(LEVELS)!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k != KNN_K:
        raise AnalysisError(f"k must equal the frozen value {KNN_K}")
    matrix, metadata, tile_keys = _records_to_arrays(records)
    cancers = {meta["cancer"] for meta in metadata}
    if len(cancers) != 1:
        raise AnalysisError("cross-race mixing must be computed separately per cancer")
    if level == "patient":
        matrix, metadata = _patient_means(matrix, metadata)
        stable_keys = [meta["patient_id"] for meta in metadata]
    else:
        stable_keys = tile_keys
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise AnalysisError("cosine mixing is undefined for zero-norm embeddings")
    normalized = matrix / norms[:, None]
    distance = 1.0 - normalized @ normalized.T
    values_by_race: dict[str, list[float]] = {race: [] for race in RACES}
    for query, meta in enumerate(metadata):
        candidates = [
            index for index, other in enumerate(metadata)
            if other["cancer"] == meta["cancer"]
            and other["patient_id"] != meta["patient_id"]
        ]
        if len(candidates) < k:
            raise AnalysisError("fewer than k eligible neighbors after same-patient exclusion")
        candidates.sort(key=lambda index: (float(distance[query, index]), stable_keys[index]))
        neighbors = candidates[:k]
        values_by_race[meta["race"]].append(
            sum(metadata[index]["race"] != meta["race"] for index in neighbors) / k
        )
    if any(not values_by_race[race] for race in RACES):
        raise AnalysisError("mixing requires queries from both races")
    race_means = {race: float(np.mean(values_by_race[race])) for race in RACES}
    return {
        "schema": GEOMETRY_SCHEMA,
        "metric": "cosine_knn_cross_race_mixing",
        "cancer": next(iter(cancers)),
        "level": level,
        "k": k,
        "same_patient_excluded": True,
        "race_query_means": race_means,
        "equal_race_mean": float(np.mean([race_means[race] for race in RACES])),
        "query_count_by_race": {race: len(values_by_race[race]) for race in RACES},
    }


def _aligned_map(
    records: Sequence[Mapping[str, object]], level: str
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, str]]]:
    matrix, metadata, tile_keys = _records_to_arrays(records)
    if level == "patient":
        matrix, metadata = _patient_means(matrix, metadata)
        keys = [meta["patient_id"] for meta in metadata]
    elif level == "tile":
        keys = tile_keys
    else:
        raise AnalysisError(f"level must be one of {tuple(LEVELS)!r}")
    return (
        {key: matrix[index] for index, key in enumerate(keys)},
        {key: dict(metadata[index]) for index, key in enumerate(keys)},
    )


def aligned_representation_displacement(
    reference: Sequence[Mapping[str, object]],
    comparison: Sequence[Mapping[str, object]],
    *,
    level: str,
) -> dict[str, Any]:
    """Displacement for exactly aligned tile vectors or patient-mean vectors."""
    reject_diagnosis_fields(reference)
    reject_diagnosis_fields(comparison)
    left, left_meta = _aligned_map(reference, level)
    right, right_meta = _aligned_map(comparison, level)
    if set(left) != set(right):
        raise AnalysisError("aligned representations must have identical identities")
    keys = sorted(left)
    rows: list[dict[str, Any]] = []
    for key in keys:
        if left_meta[key] != right_meta[key]:
            raise AnalysisError(f"aligned metadata mismatch for {key!r}")
        if left[key].shape != right[key].shape:
            raise AnalysisError("only equal-dimensional layers may be contrasted")
        delta = right[key] - left[key]
        left_norm = float(np.linalg.norm(left[key]))
        right_norm = float(np.linalg.norm(right[key]))
        cosine_distance = (
            0.0 if left_norm == 0.0 and right_norm == 0.0
            else None if left_norm == 0.0 or right_norm == 0.0
            else 1.0 - float(np.dot(left[key], right[key]) / (left_norm * right_norm))
        )
        rows.append(
            {
                "identity": key,
                "cancer": left_meta[key]["cancer"],
                "race": left_meta[key]["race"],
                "l2": float(np.linalg.norm(delta)),
                "cosine_distance": cosine_distance,
            }
        )
    l2_values = [row["l2"] for row in rows]
    cosine_values = [row["cosine_distance"] for row in rows if row["cosine_distance"] is not None]
    strata: dict[str, dict[str, float | int | None]] = {}
    for cancer in CANCERS:
        for race in RACES:
            selected = [row for row in rows if row["cancer"] == cancer and row["race"] == race]
            if selected:
                valid_cos = [row["cosine_distance"] for row in selected if row["cosine_distance"] is not None]
                strata[f"{cancer}|{race}"] = {
                    "count": len(selected),
                    "mean_l2": float(np.mean([row["l2"] for row in selected])),
                    "mean_cosine_distance": float(np.mean(valid_cos)) if valid_cos else None,
                }
    return {
        "schema": GEOMETRY_SCHEMA,
        "metric": "aligned_representation_displacement",
        "level": level,
        "aligned_count": len(rows),
        "dimension": len(left[keys[0]]),
        "mean_l2": float(np.mean(l2_values)),
        "median_l2": float(np.median(l2_values)),
        "rms_l2": float(np.sqrt(np.mean(np.square(l2_values)))),
        "mean_cosine_distance": float(np.mean(cosine_values)) if cosine_values else None,
        "undefined_cosine_count": len(rows) - len(cosine_values),
        "strata": strata,
    }


def parameter_displacement(
    baseline: Mapping[str, object], comparison: Mapping[str, object]
) -> dict[str, Any]:
    """Aligned floating-tensor Frobenius displacement (descriptive only)."""
    reject_diagnosis_fields(baseline)
    reject_diagnosis_fields(comparison)
    if set(baseline) != set(comparison):
        raise AnalysisError("parameter states must contain identical tensor names")
    squared_delta = 0.0
    squared_baseline = 0.0
    element_count = 0
    tensor_count = 0
    for name in sorted(baseline):
        left = np.asarray(baseline[name])
        right = np.asarray(comparison[name])
        if left.shape != right.shape:
            raise AnalysisError(f"parameter shape mismatch for {name!r}")
        if left.dtype.kind not in "fc" or right.dtype.kind not in "fc":
            continue
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise AnalysisError(f"nonfinite parameter tensor {name!r}")
        squared_delta += float(np.sum(np.square(right - left)))
        squared_baseline += float(np.sum(np.square(left)))
        element_count += left.size
        tensor_count += 1
    if tensor_count == 0:
        raise AnalysisError("no aligned floating-point parameter tensors")
    absolute = math.sqrt(squared_delta)
    baseline_norm = math.sqrt(squared_baseline)
    ratio = None if baseline_norm == 0.0 else absolute / baseline_norm
    return {
        "schema": GEOMETRY_SCHEMA,
        "metric": "parameter_displacement",
        "descriptive_only": True,
        "floating_tensor_count": tensor_count,
        "floating_element_count": element_count,
        "absolute_frobenius": absolute,
        "baseline_frobenius": baseline_norm,
        "baseline_frobenius_ratio": ratio,
    }


def _exact_metric_map(
    rows: Sequence[Mapping[str, object]],
    *,
    value_field: str,
    dimensions: Sequence[str],
    low: float,
    high: float,
) -> dict[tuple[object, ...], float]:
    reject_diagnosis_fields(rows)
    observed: dict[tuple[object, ...], float] = {}
    for index, row in enumerate(rows):
        try:
            key = tuple(row[field] for field in dimensions)
            raw_value = row[value_field]
        except KeyError as error:
            raise AnalysisError(f"metric row {index}: missing {error.args[0]!r}") from error
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise AnalysisError(f"metric row {index}: value must be numeric")
        value = float(raw_value)
        if not math.isfinite(value) or value < low or value > high:
            raise AnalysisError(
                f"metric row {index}: value must be finite and in [{low}, {high}]"
            )
        if key in observed:
            raise AnalysisError(f"duplicate metric cell {key!r}")
        observed[key] = value
    return observed


def _summary(values: Iterable[float]) -> dict[str, Any]:
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise AnalysisError("summary requires finite nonempty values")
    return {
        "count": len(vector),
        "mean": float(math.fsum(vector) / len(vector)),
        "median": float(np.median(vector)),
        "minimum": min(vector),
        "maximum": max(vector),
        "values": vector,
    }


def evaluate_primary_gate(
    fair_race_rows: Sequence[Mapping[str, object]],
    matched_race_rows: Sequence[Mapping[str, object]],
    fair_cancer_rows: Sequence[Mapping[str, object]],
    matched_cancer_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Evaluate the locked fixed-five activity gate with exact-cell rejection.

    Race rows are keyed by seed/cancer/view/level and use ``oriented_leakage``.
    Cancer rows are keyed by seed/view and use ``pooled_heldout_patient_auroc``.
    Leakage reduction is matched minus fair; cancer loss is matched minus fair.
    """
    race_dims = ("fm_seed", "cancer", "view", "level")
    cancer_dims = ("fm_seed", "view")
    fair_race = _exact_metric_map(
        fair_race_rows, value_field="oriented_leakage", dimensions=race_dims,
        low=0.0, high=0.5,
    )
    matched_race = _exact_metric_map(
        matched_race_rows, value_field="oriented_leakage", dimensions=race_dims,
        low=0.0, high=0.5,
    )
    expected_race = {
        (seed, cancer, view, level)
        for seed in FM_SEEDS for cancer in CANCERS for view in VIEWS for level in LEVELS
    }
    if set(fair_race) != expected_race or set(matched_race) != expected_race:
        raise AnalysisError("race gate rows do not contain the exact frozen cells")
    fair_cancer = _exact_metric_map(
        fair_cancer_rows, value_field="pooled_heldout_patient_auroc",
        dimensions=cancer_dims, low=0.0, high=1.0,
    )
    matched_cancer = _exact_metric_map(
        matched_cancer_rows, value_field="pooled_heldout_patient_auroc",
        dimensions=cancer_dims, low=0.0, high=1.0,
    )
    expected_cancer = {(seed, view) for seed in FM_SEEDS for view in VIEWS}
    if set(fair_cancer) != expected_cancer or set(matched_cancer) != expected_cancer:
        raise AnalysisError("cancer gate rows do not contain the exact frozen cells")

    strata: dict[str, dict[str, Any]] = {}
    all_leakage_pass = True
    for cancer in CANCERS:
        for view in VIEWS:
            for level in LEVELS:
                values = [matched_race[(seed, cancer, view, level)] - fair_race[(seed, cancer, view, level)] for seed in FM_SEEDS]
                summary = _summary(values)
                passing = sum(value + _TIE_TOLERANCE >= LEAKAGE_DELTA for value in values)
                median_pass = summary["median"] + _TIE_TOLERANCE >= LEAKAGE_DELTA
                stratum_pass = passing >= 4 and median_pass
                all_leakage_pass &= stratum_pass
                strata[f"{cancer}|{view}|{level}"] = {
                    "reduction_summary": summary,
                    "seeds_at_or_above_threshold": passing,
                    "four_of_five_pass": passing >= 4,
                    "median_at_or_above_threshold": median_pass,
                    "pass": stratum_pass,
                }

    cancer_views: dict[str, dict[str, Any]] = {}
    all_cancer_pass = True
    for view in VIEWS:
        losses = [matched_cancer[(seed, view)] - fair_cancer[(seed, view)] for seed in FM_SEEDS]
        summary = _summary(losses)
        passing = sum(loss <= CANCER_LOSS_MAX + _TIE_TOLERANCE for loss in losses)
        median_pass = summary["median"] <= CANCER_LOSS_MAX + _TIE_TOLERANCE
        view_pass = passing >= 4 and median_pass
        all_cancer_pass &= view_pass
        cancer_views[view] = {
            "loss_summary": summary,
            "seeds_at_or_below_maximum_loss": passing,
            "four_of_five_pass": passing >= 4,
            "median_at_or_below_maximum_loss": median_pass,
            "pass": view_pass,
        }
    passed = bool(all_leakage_pass and all_cancer_pass)
    return {
        "schema": GATE_SCHEMA,
        "audit_contract_schema": SCHEMA,
        "semantics": {
            "leakage_reduction": "matched_oriented_leakage_minus_fair_oriented_leakage",
            "cancer_probe_loss": "matched_auroc_minus_fair_auroc",
            "seed_rule": "each stratum/view requires threshold in >=4/5 seeds and threshold-consistent median",
            "missing_or_duplicate_cells": "fail_closed_with_AnalysisError",
        },
        "thresholds": {
            "minimum_oriented_leakage_reduction": LEAKAGE_DELTA,
            "maximum_cancer_probe_auroc_loss": CANCER_LOSS_MAX,
        },
        "exact_dimensions": {
            "fm_seeds": list(FM_SEEDS),
            "cancers": list(CANCERS),
            "views": list(VIEWS),
            "levels": list(LEVELS),
            "race_cell_count": len(expected_race),
            "cancer_cell_count": len(expected_cancer),
        },
        "race_leakage_strata": strata,
        "cancer_probe_views": cancer_views,
        "all_race_leakage_strata_pass": bool(all_leakage_pass),
        "all_cancer_probe_views_pass": bool(all_cancer_pass),
        "pass": passed,
        "classification": "active" if passed else "inactive",
    }


def semantic_report(
    *,
    contrasts: Sequence[Mapping[str, object]],
    identities: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Build the exact raw-cell report consumed by the independent verifier."""
    reject_diagnosis_fields(contrasts)
    reject_diagnosis_fields(identities)
    if len(contrasts) != len(GATE_ELIGIBLE_CONTRASTS):
        raise AnalysisError("report requires exactly four gate-eligible contrasts")
    output_contrasts: list[dict[str, Any]] = []
    for index, ((expected_candidate, expected_baseline), raw) in enumerate(
        zip(GATE_ELIGIBLE_CONTRASTS, contrasts, strict=True)
    ):
        if set(raw) != {"candidate", "baseline", "race_cells", "cancer_cells"}:
            raise AnalysisError(f"contrast {index}: raw input keys differ from semantic schema")
        if raw["candidate"] != expected_candidate or raw["baseline"] != expected_baseline:
            raise AnalysisError("contrasts must use the frozen candidate/baseline order")
        race_cells = sorted(
            list(raw["race_cells"]),  # type: ignore[arg-type]
            key=lambda row: (
                int(row["fm_seed"]), str(row["cancer"]),
                str(row["view"]), str(row["probe_level"]),
            ),
        )
        cancer_cells = sorted(
            list(raw["cancer_cells"]),  # type: ignore[arg-type]
            key=lambda row: (int(row["fm_seed"]), str(row["view"])),
        )
        required_race_keys = {
            "fm_seed", "cancer", "view", "probe_level",
            "baseline_oriented_leakage", "candidate_oriented_leakage",
        }
        required_cancer_keys = {
            "fm_seed", "view", "baseline_auroc", "candidate_auroc",
        }
        if any(set(row) != required_race_keys for row in race_cells):
            raise AnalysisError("race_cells rows differ from the semantic schema")
        if any(set(row) != required_cancer_keys for row in cancer_cells):
            raise AnalysisError("cancer_cells rows differ from the semantic schema")
        fair_race = [
            {
                "fm_seed": row["fm_seed"], "cancer": row["cancer"],
                "view": row["view"], "level": row["probe_level"],
                "oriented_leakage": row["candidate_oriented_leakage"],
            }
            for row in race_cells
        ]
        matched_race = [
            {
                "fm_seed": row["fm_seed"], "cancer": row["cancer"],
                "view": row["view"], "level": row["probe_level"],
                "oriented_leakage": row["baseline_oriented_leakage"],
            }
            for row in race_cells
        ]
        fair_cancer = [
            {"fm_seed": row["fm_seed"], "view": row["view"],
             "pooled_heldout_patient_auroc": row["candidate_auroc"]}
            for row in cancer_cells
        ]
        matched_cancer = [
            {"fm_seed": row["fm_seed"], "view": row["view"],
             "pooled_heldout_patient_auroc": row["baseline_auroc"]}
            for row in cancer_cells
        ]
        gate = evaluate_primary_gate(fair_race, matched_race, fair_cancer, matched_cancer)
        output_contrasts.append(
            {
                "candidate": expected_candidate,
                "baseline": expected_baseline,
                "race_cells": race_cells,
                "cancer_cells": cancer_cells,
                "reported_gate": gate,
            }
        )
    required_roles = {"metric_input", "lock", "numeric_amendment", "analyzer"}
    if set(identities) != required_roles:
        raise AnalysisError("identity roles differ from the semantic report contract")
    for role, identity in identities.items():
        if set(identity) != {"canonical_path", "bytes", "sha256"}:
            raise AnalysisError(f"identity {role!r} is not an exact file identity")
        digest = identity["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AnalysisError(f"identity {role!r} has invalid SHA-256")
        if not isinstance(identity["bytes"], int) or identity["bytes"] <= 0:
            raise AnalysisError(f"identity {role!r} has invalid byte count")
    return {
        "schema": ANALYSIS_SCHEMA,
        "study_id": STUDY_ID,
        "status": "complete",
        "diagnosis_free": True,
        "inference_unit": "FM seed",
        "fm_seeds": list(FM_SEEDS),
        "contrasts": output_contrasts,
        "identities": {role: dict(identities[role]) for role in sorted(identities)},
    }
