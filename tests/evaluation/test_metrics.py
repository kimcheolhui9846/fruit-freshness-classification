import unittest
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, top_k_accuracy_score

from src.evaluation.diagnostics import compute_classification_diagnostics
from src.evaluation.metrics import compute_validation_metrics


def legacy_validation_metrics(labels, predictions, logits):
    """Test-only reference copied from the pre-Phase 4.8 notebook metric region."""
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


class ValidationMetricParityTest(unittest.TestCase):
    def assert_metric_parity(self, labels, predictions, logits):
        with warnings.catch_warnings(record=True) as legacy_warnings:
            warnings.simplefilter("always")
            expected = legacy_validation_metrics(labels, predictions, logits)
        with warnings.catch_warnings(record=True) as extracted_warnings:
            warnings.simplefilter("always")
            actual = compute_validation_metrics(labels, predictions, logits)

        self.assertEqual(type(expected), type(actual))
        self.assertEqual(len(expected), len(actual))
        for expected_value, actual_value in zip(expected, actual):
            self.assertIs(type(expected_value), type(actual_value))
            self.assertEqual(expected_value, actual_value)
        self.assertEqual(
            [(type(item.message), str(item.message)) for item in legacy_warnings],
            [(type(item.message), str(item.message)) for item in extracted_warnings],
        )

    def test_representative_classification_cases_match_the_legacy_region(self):
        cases = {
            "perfect_predictions": (
                [0, 1, 2],
                [0, 1, 2],
                np.array([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 5.0]]),
            ),
            "all_predictions_incorrect": (
                [0, 1, 2],
                [1, 2, 0],
                np.array([[0.0, 5.0, 1.0], [1.0, 0.0, 5.0], [5.0, 1.0, 0.0]]),
            ),
            "class_imbalance": (
                [0, 0, 0, 1, 2],
                [0, 0, 1, 1, 2],
                np.array(
                    [[5.0, 1.0, 0.0], [4.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 5.0, 1.0], [0.0, 1.0, 5.0]]
                ),
            ),
            "class_absent_from_predictions": (
                [0, 1, 2],
                [0, 0, 1],
                np.array([[5.0, 1.0, 0.0], [5.0, 1.0, 0.0], [1.0, 5.0, 0.0]]),
            ),
            "class_absent_from_labels": (
                [0, 1, 1],
                [0, 1, 0],
                np.array([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [5.0, 1.0, 0.0]]),
            ),
            "single_sample": (
                [1],
                [1],
                np.array([[0.0, 5.0, 1.0]]),
            ),
        }
        for name, (labels, predictions, logits) in cases.items():
            with self.subTest(name=name):
                self.assert_metric_parity(labels, predictions, logits)

    def test_topk_error_path_matches_when_three_classes_are_unavailable(self):
        labels = [0, 1]
        predictions = [0, 1]
        logits = np.array([[5.0, 1.0], [1.0, 5.0]])

        self.assert_metric_parity(labels, predictions, logits)
        _, _, top2, top3 = compute_validation_metrics(labels, predictions, logits)
        self.assertIsNone(top2)
        self.assertIsNone(top3)

    def test_metric_arguments_preserve_the_notebook_contract(self):
        labels = [0, 1, 2]
        predictions = [0, 1, 2]
        logits = np.array([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 5.0]])

        actual = compute_validation_metrics(labels, predictions, logits)
        self.assertEqual(actual[0], f1_score(labels, predictions, average="macro"))
        self.assertEqual(actual[1], balanced_accuracy_score(labels, predictions))
        self.assertEqual(
            actual[2],
            top_k_accuracy_score(labels, logits, k=2, labels=np.arange(logits.shape[1])),
        )
        self.assertEqual(
            actual[3],
            top_k_accuracy_score(labels, logits, k=3, labels=np.arange(logits.shape[1])),
        )


class ClassificationDiagnosticTest(unittest.TestCase):
    def test_diagnostics_include_oof_summary_per_class_and_confusion_matrix(self):
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        predictions = np.array([0, 2, 1, 3], dtype=np.int64)
        logits = np.array(
            [
                [4.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 4.0, -1.0],
                [0.0, 4.0, 1.0, -1.0],
                [0.0, 1.0, 0.0, 4.0],
            ]
        )

        diagnostics = compute_classification_diagnostics(
            labels,
            predictions,
            logits,
            ["fresh", "ripe", "rotten", "other"],
        )

        self.assertEqual(diagnostics["sample_count"], 4)
        self.assertEqual(diagnostics["top1_accuracy"], 0.5)
        self.assertEqual(
            diagnostics["confusion_matrix"],
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        )
        self.assertEqual(
            [row["class_name"] for row in diagnostics["per_class"]],
            ["fresh", "ripe", "rotten", "other"],
        )
        self.assertIn("macro_f1", diagnostics)
        self.assertIn("balanced_accuracy", diagnostics)
        self.assertIn("top2_accuracy", diagnostics)
        self.assertIn("top3_accuracy", diagnostics)

if __name__ == "__main__":
    unittest.main()
