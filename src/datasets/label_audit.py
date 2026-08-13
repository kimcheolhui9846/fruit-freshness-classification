"""Selection and scoring for the frozen Phase 9.5 label quality audit.

Pure logic only. This module never touches the dataset, the filesystem, or a
model, so the audit's determinism can be tested without either.
"""

from __future__ import annotations

import numpy as np


SUBJECT_CLASS = "freshpotato"
CONTROL_CLASS = "rottenpotato"

# Frozen review-set sizing from the protocol. This is the single source that
# both the build and analysis CLIs check group sizes against, so a change to
# one cannot silently drift from the other.
SUBJECT_COUNT = 347
CONTROL_COUNT = 150
REVIEW_SET_COUNT = SUBJECT_COUNT + CONTROL_COUNT

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


DECISION_THRESHOLD = 0.15


def _tally(judgments: dict[int, str], indices: np.ndarray) -> dict[str, int]:
    """Count each judgment category over one group. Single point of validation."""
    counts = {category: 0 for category in JUDGMENT_CATEGORIES}
    for index in indices.tolist():
        try:
            call = judgments[int(index)]
        except KeyError:
            raise ValueError(f"Missing judgment for review index {index}.") from None
        if call not in JUDGMENT_CATEGORIES:
            raise ValueError(f"Unknown judgment category {call!r} for index {index}.")
        counts[call] += 1
    return counts


def _rate(counts: dict[str, int], total: int, error_categories) -> tuple[float, float]:
    errors = sum(counts[category] for category in error_categories)
    undecidable = counts["UNDECIDABLE"]
    return errors / total, undecidable / total


def score_reviewer(
    judgments: dict[int, str],
    subject_indices: np.ndarray,
    control_indices: np.ndarray,
) -> dict:
    """Score one reviewer. UNDECIDABLE stays in the denominator, never an error."""
    subject_indices = np.asarray(subject_indices)
    control_indices = np.asarray(control_indices)
    subject_counts = _tally(judgments, subject_indices)
    control_counts = _tally(judgments, control_indices)
    subject_error, subject_undecidable = _rate(
        subject_counts, len(subject_indices), SUBJECT_ERROR_CATEGORIES
    )
    control_error, control_undecidable = _rate(
        control_counts, len(control_indices), CONTROL_ERROR_CATEGORIES
    )
    return {
        "subject_error_rate": subject_error,
        "control_error_rate": control_error,
        "difference": subject_error - control_error,
        "subject_undecidable_rate": subject_undecidable,
        "control_undecidable_rate": control_undecidable,
        "subject_counts": subject_counts,
        "control_counts": control_counts,
    }


def apply_decision_rule(reviewer_scores: list[dict], *, threshold: float = DECISION_THRESHOLD) -> dict:
    """Apply the pre-committed rule. Each reviewer is judged against their own control."""
    clears = [
        (s["subject_error_rate"] - s["control_error_rate"]) >= threshold
        for s in reviewer_scores
    ]
    if all(clears):
        outcome = "DEFECT_CONFIRMED"
        next_phase = "Phase 9.6 remediation decision: relabel, exclude, or retain (separate authorization)"
    elif not any(clears):
        outcome = "DEFECT_NOT_CONFIRMED"
        next_phase = "Phase 9.6 is H1 loss and class imbalance, as pre-registered"
    else:
        outcome = "SPLIT_OUTCOME"
        next_phase = "No phase selected automatically; owner decides after reviewing disagreements"
    return {"outcome": outcome, "next_phase": next_phase, "clears_threshold": clears}
