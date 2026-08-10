"""Structured classification diagnostics for development-only evaluation."""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.evaluation.metrics import compute_validation_metrics


def compute_classification_diagnostics(labels, predictions, logits, class_names):
    """Return JSON-safe aggregate, per-class, and confusion diagnostics."""
    label_array = np.asarray(labels, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    logit_array = np.asarray(logits)
    names = list(class_names)
    class_indices = np.arange(len(names), dtype=np.int64)

    if label_array.ndim != 1 or prediction_array.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional.")
    if len(label_array) != len(prediction_array):
        raise ValueError("Labels and predictions must have the same length.")
    if logit_array.ndim != 2 or logit_array.shape != (len(label_array), len(names)):
        raise ValueError("Logits must have one row per sample and one column per class.")

    macro_f1, balanced_accuracy, top2_accuracy, top3_accuracy = compute_validation_metrics(
        label_array,
        prediction_array,
        logit_array,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        label_array,
        prediction_array,
        labels=class_indices,
        zero_division=0,
    )
    per_class = [
        {
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(names)
    ]

    return {
        "sample_count": int(len(label_array)),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "top1_accuracy": float(np.mean(label_array == prediction_array)),
        "top2_accuracy": None if top2_accuracy is None else float(top2_accuracy),
        "top3_accuracy": None if top3_accuracy is None else float(top3_accuracy),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            label_array,
            prediction_array,
            labels=class_indices,
        ).astype(int).tolist(),
    }