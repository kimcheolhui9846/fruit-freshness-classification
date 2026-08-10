"""Deterministic post-holdout development, validation, and CV helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


POSTHOLDOUT_SPLIT_SCHEMA_VERSION = 1
POSTHOLDOUT_CV_SCHEMA_VERSION = 1


def _sha256_int64(values) -> str:
    """Return the portable little-endian int64 digest used by frozen indices."""
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest.") from error
    return value


def _as_indices(value: object, name: str, *, upper_bound: int) -> np.ndarray:
    if not isinstance(value, list) or any(type(index) is not int for index in value):
        raise ValueError(f"{name} must be a JSON list of integer indices.")
    indices = np.asarray(value, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(indices < 0) or np.any(indices >= upper_bound):
        raise ValueError(f"{name} contains an out-of-range index.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicate indices.")
    return indices


def _load_json_object(path: str | Path, description: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {description}: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return resolved, payload


def load_frozen_postholdout_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen development/locked-test split manifest."""
    _, manifest = _load_json_object(path, "post-holdout split manifest")
    required_values = {
        "schema_version": POSTHOLDOUT_SPLIT_SCHEMA_VERSION,
        "protocol": "DEV_PLUS_LOCKED_TEST",
        "source_pool_identity": "HISTORICAL_CANONICAL_TRAIN_ONLY",
        "canonical_holdout_usage": "HISTORICAL_EVIDENCE_ONLY",
        "canonical_holdout_overlap": 0,
        "locked_test_status": "FROZEN_UNOBSERVED_BY_MODEL",
        "locked_test_model_evaluation": "NO",
    }
    for key, expected in required_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Unexpected frozen split manifest value: {key}.")

    source_count = manifest.get("source_pool_size")
    development_count = manifest.get("development_count")
    locked_test_count = manifest.get("locked_test_count")
    if any(type(value) is not int or value <= 0 for value in (source_count, development_count, locked_test_count)):
        raise ValueError("Frozen split sizes must be positive integers.")
    if source_count != development_count + locked_test_count:
        raise ValueError("Frozen split sizes do not partition the source pool.")

    development_indices = _as_indices(
        manifest.get("development_indices"),
        "development_indices",
        upper_bound=source_count,
    )
    locked_test_indices = _as_indices(
        manifest.get("locked_test_indices"),
        "locked_test_indices",
        upper_bound=source_count,
    )
    if len(development_indices) != development_count or len(locked_test_indices) != locked_test_count:
        raise ValueError("Frozen split index counts do not match the manifest.")
    if len(np.intersect1d(development_indices, locked_test_indices)) != 0:
        raise ValueError("Frozen development and locked-test indices overlap.")
    if not np.array_equal(
        np.sort(np.concatenate((development_indices, locked_test_indices))),
        np.arange(source_count, dtype=np.int64),
    ):
        raise ValueError("Frozen split indices do not exhaust the source pool.")

    for field, indices in (
        ("development_indices_sha256", development_indices),
        ("locked_test_indices_sha256", locked_test_indices),
    ):
        expected_digest = _require_sha256(manifest.get(field), field)
        if _sha256_int64(indices) != expected_digest:
            raise ValueError(f"Frozen split {field} hash does not match indices.")
    _require_sha256(manifest.get("source_label_sequence_sha256"), "source_label_sequence_sha256")
    return manifest


def validate_postholdout_source_pool(
    historical_train,
    historical_holdout,
    manifest: dict[str, Any],
) -> None:
    """Validate the canonical reconstruction before selecting model-visible rows."""
    if len(historical_train) != manifest["source_pool_size"]:
        raise ValueError("Historical canonical train size does not match frozen split.")
    if len(historical_holdout) != manifest["canonical_holdout_size"]:
        raise ValueError("Historical canonical holdout size does not match frozen split.")
    source_labels = np.asarray(historical_train["label"], dtype=np.int64)
    if _sha256_int64(source_labels) != manifest["source_label_sequence_sha256"]:
        raise ValueError("Historical canonical train labels do not match frozen split.")


def select_frozen_development_pool(
    historical_train,
    historical_holdout,
    manifest: dict[str, Any],
):
    """Return only the frozen model-visible development subset.

    The locked test and historical holdout are read solely for identity checks;
    neither split is returned to callers that construct models or loaders.
    """
    validate_postholdout_source_pool(historical_train, historical_holdout, manifest)
    development = historical_train.select(manifest["development_indices"])
    if len(development) != manifest["development_count"]:
        raise ValueError("Selected development size does not match frozen split.")
    labels = np.asarray(development["label"], dtype=np.int64)
    names = development.features["label"].names
    actual_counts = {
        names[index]: int(np.count_nonzero(labels == index))
        for index in range(len(names))
    }
    if actual_counts != manifest["development_class_counts"]:
        raise ValueError("Selected development class counts do not match frozen split.")
    return development


def build_postholdout_split(
    labels,
    *,
    locked_test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint stratified source-pool indices for Phase 9 development.

    Returned indices are positions relative to the supplied source pool. The
    function does not mutate the supplied labels or NumPy's global RNG state.
    """
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional sequence")
    if label_array.size == 0:
        raise ValueError("labels must not be empty")
    if not 0.0 < locked_test_fraction < 1.0:
        raise ValueError("locked_test_fraction must be strictly between 0 and 1")

    _, class_counts = np.unique(label_array, return_counts=True)
    if class_counts.size < 2:
        raise ValueError("stratified splitting requires at least two classes")
    if np.any(class_counts < 2):
        raise ValueError("each class must contain at least two examples")

    source_indices = np.arange(label_array.size, dtype=np.int64)
    development_indices, locked_test_indices = train_test_split(
        source_indices,
        test_size=locked_test_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=label_array,
    )
    return (
        np.sort(np.asarray(development_indices, dtype=np.int64)),
        np.sort(np.asarray(locked_test_indices, dtype=np.int64)),
    )


def build_postholdout_cv_manifest(
    labels,
    *,
    experiment_id: str,
    parent_experiment_id: str,
    development_manifest_sha256: str,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> dict[str, Any]:
    """Create a deterministic, portable CV identity over development positions."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("development labels must be a non-empty one-dimensional sequence.")
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be boolean.")
    if type(random_state) is not int or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")
    _require_sha256(development_manifest_sha256, "development_manifest_sha256")

    positions = np.arange(label_array.size, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    folds = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(positions, label_array),
        start=1,
    ):
        train_indices = np.asarray(train_indices, dtype=np.int64)
        validation_indices = np.asarray(validation_indices, dtype=np.int64)
        folds.append(
            {
                "fold": fold_number,
                "train_count": len(train_indices),
                "validation_count": len(validation_indices),
                "train_indices": train_indices.tolist(),
                "train_indices_sha256": _sha256_int64(train_indices),
                "validation_indices": validation_indices.tolist(),
                "validation_indices_sha256": _sha256_int64(validation_indices),
            }
        )
    manifest = {
        "schema_version": POSTHOLDOUT_CV_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "development_manifest_sha256": development_manifest_sha256,
        "development_count": len(label_array),
        "cross_validation": {
            "n_splits": n_splits,
            "shuffle": shuffle,
            "random_state": random_state,
        },
        "folds": folds,
    }
    return validate_postholdout_cv_manifest(
        manifest,
        development_manifest_sha256=development_manifest_sha256,
        development_count=len(label_array),
    )


def load_postholdout_cv_manifest(
    path: str | Path,
    *,
    development_manifest_sha256: str,
    development_count: int,
) -> dict[str, Any]:
    """Load and validate a tracked CV identity without accessing dataset rows."""
    _, manifest = _load_json_object(path, "post-holdout CV manifest")
    return validate_postholdout_cv_manifest(
        manifest,
        development_manifest_sha256=development_manifest_sha256,
        development_count=development_count,
    )


def validate_postholdout_cv_manifest(
    manifest: dict[str, Any],
    *,
    development_manifest_sha256: str,
    development_count: int,
) -> dict[str, Any]:
    """Validate fold hashes, disjoint validation coverage, and split identity."""
    if manifest.get("schema_version") != POSTHOLDOUT_CV_SCHEMA_VERSION:
        raise ValueError("Unexpected post-holdout CV manifest schema version.")
    if manifest.get("development_manifest_sha256") != development_manifest_sha256:
        raise ValueError("CV manifest does not match the frozen development manifest hash.")
    if manifest.get("development_count") != development_count:
        raise ValueError("CV manifest does not match the frozen development count.")
    cv = manifest.get("cross_validation")
    if not isinstance(cv, dict) or cv.get("n_splits") != 3 or cv.get("shuffle") is not True or cv.get("random_state") != 42:
        raise ValueError("CV manifest does not preserve the approved three-fold identity.")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != cv["n_splits"]:
        raise ValueError("CV manifest fold count does not match its configuration.")

    validation_parts = []
    for expected_fold, fold in enumerate(folds, start=1):
        if not isinstance(fold, dict) or fold.get("fold") != expected_fold:
            raise ValueError("CV manifest fold ordering is invalid.")
        train_indices = _as_indices(fold.get("train_indices"), "train_indices", upper_bound=development_count)
        validation_indices = _as_indices(fold.get("validation_indices"), "validation_indices", upper_bound=development_count)
        if fold.get("train_count") != len(train_indices) or fold.get("validation_count") != len(validation_indices):
            raise ValueError("CV manifest fold counts do not match indices.")
        if len(np.intersect1d(train_indices, validation_indices)) != 0:
            raise ValueError("CV manifest train and validation indices overlap.")
        if not np.array_equal(
            np.sort(np.concatenate((train_indices, validation_indices))),
            np.arange(development_count, dtype=np.int64),
        ):
            raise ValueError("CV manifest fold does not partition development positions.")
        for field, indices in (
            ("train_indices_sha256", train_indices),
            ("validation_indices_sha256", validation_indices),
        ):
            if _sha256_int64(indices) != _require_sha256(fold.get(field), field):
                raise ValueError(f"CV manifest {field} hash does not match indices.")
        validation_parts.append(validation_indices)
    if not np.array_equal(
        np.sort(np.concatenate(validation_parts)),
        np.arange(development_count, dtype=np.int64),
    ):
        raise ValueError("CV validation folds are not disjoint and exhaustive.")
    return manifest


def cv_folds_from_manifest(manifest: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return model-ready fold positions after a manifest has been validated."""
    return [
        (
            np.asarray(fold["train_indices"], dtype=np.int64),
            np.asarray(fold["validation_indices"], dtype=np.int64),
        )
        for fold in manifest["folds"]
    ]