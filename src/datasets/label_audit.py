"""Selection and scoring for the frozen Phase 9.5 label quality audit.

Pure logic only. This module never touches the dataset, the filesystem, or a
model, so the audit's determinism can be tested without either.
"""

from __future__ import annotations

import numpy as np


SUBJECT_CLASS = "freshpotato"
CONTROL_CLASS = "rottenpotato"

JUDGMENT_CATEGORIES = ("FRESH", "ROTTEN", "NOT_A_POTATO", "UNDECIDABLE")
SUBJECT_ERROR_CATEGORIES = ("ROTTEN", "NOT_A_POTATO")
CONTROL_ERROR_CATEGORIES = ("FRESH", "NOT_A_POTATO")


def select_review_set(
    development_indices: np.ndarray,
    development_labels: np.ndarray,
    class_names: list[str],
    *,
    control_seed: int,
    order_seed: int,
    subject_count: int,
    control_count: int,
) -> dict:
    """Choose the blinded review set deterministically from frozen seeds."""
    development_indices = np.asarray(development_indices, dtype=np.int64)
    development_labels = np.asarray(development_labels, dtype=np.int64)
    if development_indices.shape != development_labels.shape:
        raise ValueError("Development indices and labels must align.")

    subject_label = class_names.index(SUBJECT_CLASS)
    control_label = class_names.index(CONTROL_CLASS)

    subject_indices = np.sort(development_indices[development_labels == subject_label])
    control_pool = np.sort(development_indices[development_labels == control_label])

    if len(subject_indices) != subject_count:
        raise ValueError(
            f"Expected {subject_count} {SUBJECT_CLASS} indices, found {len(subject_indices)}."
        )
    if len(control_pool) < control_count:
        raise ValueError(
            f"Expected at least {control_count} {CONTROL_CLASS} indices, found {len(control_pool)}."
        )

    control_indices = np.sort(
        np.random.default_rng(control_seed).choice(control_pool, size=control_count, replace=False)
    )

    presentation = np.concatenate([subject_indices, control_indices])
    np.random.default_rng(order_seed).shuffle(presentation)

    return {
        "subject_indices": subject_indices,
        "control_indices": control_indices,
        "presentation": presentation,
    }
