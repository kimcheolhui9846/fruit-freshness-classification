"""Offline contract tests for Phase 9.3 development-only OOF evaluation."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = ROOT / "scripts" / "evaluate_postholdout_baseline.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("phase93_oof_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PostHoldoutOofEvaluationContractTest(unittest.TestCase):
    def test_development_only_evaluator_entrypoint_exists(self):
        self.assertTrue(EVALUATOR_PATH.is_file())

    def test_oof_assembly_is_exhaustive_and_preserves_frozen_labels(self):
        evaluator = load_evaluator()
        labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        fold_outputs = [
            {
                "fold": 1,
                "validation_indices": np.array([0, 3], dtype=np.int64),
                "labels": [0, 0],
                "predictions": [0, 1],
                "logits": np.array([[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]]),
            },
            {
                "fold": 2,
                "validation_indices": np.array([1, 4], dtype=np.int64),
                "labels": [1, 1],
                "predictions": [1, 1],
                "logits": np.array([[1.0, 4.0, 0.0], [0.0, 4.0, 1.0]]),
            },
            {
                "fold": 3,
                "validation_indices": np.array([2, 5], dtype=np.int64),
                "labels": [2, 2],
                "predictions": [2, 0],
                "logits": np.array([[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]]),
            },
        ]

        assembled = evaluator.assemble_oof_predictions(
            expected_labels=labels,
            num_classes=3,
            fold_outputs=fold_outputs,
        )

        np.testing.assert_array_equal(assembled["labels"], labels)
        np.testing.assert_array_equal(
            assembled["predictions"],
            np.array([0, 1, 2, 1, 1, 0], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            assembled["fold_assignments"],
            np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
        )
        self.assertEqual(assembled["logits"].shape, (6, 3))

    def test_oof_assembly_rejects_duplicate_or_incomplete_validation_coverage(self):
        evaluator = load_evaluator()
        labels = np.array([0, 1, 2], dtype=np.int64)
        duplicate_fold_outputs = [
            {
                "fold": 1,
                "validation_indices": np.array([0, 1], dtype=np.int64),
                "labels": [0, 1],
                "predictions": [0, 1],
                "logits": np.eye(3)[[0, 1]],
            },
            {
                "fold": 2,
                "validation_indices": np.array([1], dtype=np.int64),
                "labels": [1],
                "predictions": [1],
                "logits": np.eye(3)[[1]],
            },
        ]

        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluator.assemble_oof_predictions(
                expected_labels=labels,
                num_classes=3,
                fold_outputs=duplicate_fold_outputs,
            )


class PostHoldoutOofArtifactTest(unittest.TestCase):
    def test_writer_records_only_local_development_oof_artifacts(self):
        evaluator = load_evaluator()
        payload = {
            "experiment_id": "deep3-postholdout-research-01-baseline",
            "metrics": {
                "sample_count": 2,
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": 1.0,
                "per_class": [
                    {
                        "class_name": "fresh",
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "support": 1,
                    },
                    {
                        "class_name": "rotten",
                        "precision": 1.0,
                        "recall": 1.0,
                        "f1": 1.0,
                        "support": 1,
                    },
                ],
                "confusion_matrix": [[1, 0], [0, 1]],
            },
            "labels": np.array([0, 1], dtype=np.int64),
            "predictions": np.array([0, 1], dtype=np.int64),
            "logits": np.eye(2, dtype=np.float32),
            "fold_assignments": np.array([1, 2], dtype=np.int64),
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "development-oof"
            paths = evaluator.write_development_oof_artifacts(output_dir, payload)

            self.assertEqual(
                sorted(path.name for path in paths.values()),
                sorted(
                    [
                        "development_oof_confusion_matrix.csv",
                        "development_oof_metrics.json",
                        "development_oof_per_class_metrics.csv",
                        "development_oof_predictions.npz",
                    ]
                ),
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))

            with self.assertRaisesRegex(FileExistsError, "must be empty"):
                evaluator.write_development_oof_artifacts(output_dir, payload)

class DevelopmentOnlyFoldEvaluationBoundaryTest(unittest.TestCase):
    def test_fold_evaluation_constructs_loaders_and_models_from_development_only(self):
        evaluator = load_evaluator()
        events = []

        class DevelopmentDataset:
            features = {"label": SimpleNamespace(names=["fresh", "rotten"])}

            def __getitem__(self, key):
                if key != "label":
                    raise AssertionError("Only development labels may be requested.")
                return [0, 1, 0, 1]

        development = DevelopmentDataset()
        folds = [
            (np.array([1, 3], dtype=np.int64), np.array([0, 2], dtype=np.int64)),
            (np.array([0, 2], dtype=np.int64), np.array([1, 3], dtype=np.int64)),
        ]

        class WrappedDataset:
            def __init__(self, split, transform=None):
                events.append(("wrapped", split, transform))
                self.split = split

        def select_fold_datasets(dataset, train_indices, validation_indices):
            self.assertIs(dataset, development)
            events.append(("select", train_indices.tolist(), validation_indices.tolist()))
            return "development-train", "development-validation-" + str(validation_indices[0])

        def validate_one_epoch(model, loader, criterion, device, progress_description):
            events.append(("validate", model, loader, progress_description))
            if model == "model-1":
                return 1.0, 0.0, [0, 0], [0, 0], [np.array([[4.0, 0.0], [4.0, 0.0]])]
            return 1.0, 0.0, [1, 1], [1, 1], [np.array([[0.0, 4.0], [0.0, 4.0]])]

        dependencies = SimpleNamespace(
            FruitHFDataset=WrappedDataset,
            build_class_balanced_alpha=lambda counts, beta, classes: torch.ones(classes),
            build_holdout_dataloader=lambda dataset, batch_size: (
                events.append(("loader", dataset.split, batch_size)) or dataset.split
            ),
            build_validation_transform=lambda: "validation-transform",
            compute_classification_diagnostics=lambda labels, predictions, logits, names: {
                "sample_count": len(labels),
                "macro_f1": 1.0,
                "balanced_accuracy": 1.0,
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": None,
                "per_class": [],
                "confusion_matrix": [],
            },
            load_fold_model=lambda classes, device, directory, fold: (
                events.append(("model", classes, fold)) or "model-" + str(fold)
            ),
            select_fold_datasets=select_fold_datasets,
            validate_one_epoch=validate_one_epoch,
        )
        config = {
            "loss": {
                "class_balanced_beta": 0.999,
                "use_ce_label_smoothing": True,
                "label_smoothing": 0.01,
            },
            "training": {"batch_size": 64},
            "cross_validation": {"n_splits": 2},
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            for fold in (1, 2):
                (checkpoint_directory / ("best_model_fold" + str(fold) + ".pt")).touch()
            with unittest.mock.patch.object(evaluator.torch.cuda, "is_available", return_value=False):
                result = evaluator.evaluate_development_cv(
                    config,
                    development,
                    folds,
                    checkpoint_directory,
                    "cpu",
                    dependencies,
                )

        self.assertEqual(result["metrics"]["sample_count"], 4)
        self.assertEqual(
            [event for event in events if event[0] == "model"],
            [("model", 2, 1), ("model", 2, 2)],
        )
        self.assertEqual(
            [event[1] for event in events if event[0] == "wrapped"],
            ["development-validation-0", "development-validation-1"],
        )
        self.assertEqual(
            [event[1] for event in events if event[0] == "loader"],
            ["development-validation-0", "development-validation-1"],
        )
        self.assertEqual(result["labels"].tolist(), [0, 1, 0, 1])
        self.assertEqual(result["fold_assignments"].tolist(), [1, 2, 1, 2])

if __name__ == "__main__":
    unittest.main()
