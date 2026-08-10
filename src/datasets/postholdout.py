"""Deterministic post-holdout development and locked-test split helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


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