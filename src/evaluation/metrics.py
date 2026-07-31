"""Evaluation metrics that preserve the active notebook calculation contract."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, top_k_accuracy_score


def compute_validation_metrics(labels, predictions, logits):
    """Compute the notebook's macro F1, balanced accuracy, and top-k scores."""
    va_f1 = f1_score(labels, predictions, average="macro")
    va_bal = balanced_accuracy_score(labels, predictions)

    try:
        va_top2 = top_k_accuracy_score(
            labels,
            logits,
            k=2,
            labels=np.arange(logits.shape[1]),
        )
        va_top3 = top_k_accuracy_score(
            labels,
            logits,
            k=3,
            labels=np.arange(logits.shape[1]),
        )
    except:
        va_top2 = va_top3 = None

    return va_f1, va_bal, va_top2, va_top3
