import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from src.engine.checkpoint import save_model_state
from src.inference.loading import load_fold_models


EVALUATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location("phase54_evaluate_orchestration", EVALUATE_PATH)
evaluate = importlib.util.module_from_spec(EVALUATE_SPEC)
sys.modules[EVALUATE_SPEC.name] = evaluate
EVALUATE_SPEC.loader.exec_module(evaluate)


class FakeSplit:
    def __init__(self):
        self.features = {"label": SimpleNamespace(names=["fresh", "rotten"])}


class SyntheticEvaluationHarness:
    def __init__(self):
        self.events = []
        harness = self

        class FinalDataset(dict):
            def __getitem__(inner_self, key):
                harness.events.append(("dataset_split", key))
                return super(FinalDataset, inner_self).__getitem__(key)

        self.final_dataset = FinalDataset(train=FakeSplit(), test="holdout-split")

    def dependencies(self):
        harness = self

        class WrappedDataset:
            def __init__(self, split, transform=None):
                self.split = split
                self.transform = transform
                harness.events.append(("wrapped_dataset", split, transform))

        def load_dataset():
            harness.events.append("load_dataset")
            return harness.final_dataset

        def build_transform():
            harness.events.append("validation_transform")
            return "validation-transform"

        def build_loader(dataset, batch_size):
            harness.events.append(("holdout_loader", dataset, batch_size))
            return "holdout-loader"

        def load_models(num_folds, num_classes, device, checkpoint_dir):
            harness.events.append(
                ("load_models", num_folds, num_classes, device, Path(checkpoint_dir)),
            )
            return ["fold-1", "fold-2", "fold-3"]

        def run_ensemble(models, dataloader, device):
            harness.events.append(("run_ensemble", models, dataloader, device))
            return 5, 8

        return SimpleNamespace(
            FruitHFDataset=WrappedDataset,
            build_holdout_dataloader=build_loader,
            build_validation_transform=build_transform,
            load_fold_models=load_models,
            load_fruit_freshness_dataset=load_dataset,
            run_ensemble_holdout=run_ensemble,
        )


def make_config():
    return {
        "runtime": {"cudnn_benchmark": True},
        "training": {"batch_size": 192},
        "cross_validation": {"n_splits": 3},
    }


class EvaluateOrchestrationTest(unittest.TestCase):
    def test_synthetic_orchestration_reuses_labeled_holdout_ensemble_in_order(self):
        harness = SyntheticEvaluationHarness()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            expected_paths = []
            for fold in range(1, 4):
                path = checkpoint_dir / f"best_model_fold{fold}.pt"
                path.touch()
                expected_paths.append(path)

            args = argparse.Namespace(
                config=Path("configs/deep3.toml"),
                checkpoint_dir=checkpoint_dir,
            )
            previous_benchmark = torch.backends.cudnn.benchmark
            output = io.StringIO()
            try:
                with (
                    patch.object(
                        evaluate,
                        "resolve_device",
                        side_effect=lambda: harness.events.append("device") or "cpu",
                    ),
                    patch.object(
                        evaluate,
                        "load_experiment_config",
                        side_effect=lambda path: harness.events.append(("config", Path(path))) or make_config(),
                    ),
                    patch.object(
                        evaluate,
                        "_load_evaluation_dependencies",
                        side_effect=lambda: harness.events.append("dependencies") or harness.dependencies(),
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    summary = evaluate.run_evaluation(args)
            finally:
                torch.backends.cudnn.benchmark = previous_benchmark

        # Phase 9.7 reversed these two: the config must be read before the
        # device is resolved, so the determinism policy can set
        # CUBLAS_WORKSPACE_CONFIG before any CUDA work reads it.
        self.assertEqual(
            harness.events[0],
            ("config", evaluate.REPOSITORY_ROOT / "configs/deep3.toml"),
        )
        self.assertEqual(harness.events[1], "device")
        self.assertEqual(harness.events[2], "dependencies")
        self.assertEqual(harness.events[3], "load_dataset")
        self.assertEqual(
            [event for event in harness.events if event[0] == "dataset_split"],
            [("dataset_split", "train"), ("dataset_split", "test")],
        )
        self.assertIn(("validation_transform"), harness.events)
        self.assertIn(("holdout_loader", unittest.mock.ANY, 192), harness.events)
        self.assertIn(
            ("load_models", 3, 2, "cpu", checkpoint_dir),
            harness.events,
        )
        self.assertIn(
            ("run_ensemble", ["fold-1", "fold-2", "fold-3"], "holdout-loader", "cpu"),
            harness.events,
        )
        self.assertEqual(summary["checkpoint_paths"], expected_paths)
        self.assertEqual(summary["correct"], 5)
        self.assertEqual(summary["total"], 8)
        self.assertEqual(summary["accuracy"], 0.625)
        self.assertIn("Final Holdout Acc: 0.625", output.getvalue())

class TinyCheckpointClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_classes))

    def forward(self, images):
        return self.logits.expand(images.shape[0], -1)


class EvaluateCheckpointIntegrationTest(unittest.TestCase):
    def test_entrypoint_loads_temporary_fold_states_and_preserves_ensemble_order(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            for fold in range(1, 4):
                source_model = TinyCheckpointClassifier(2)
                with torch.no_grad():
                    source_model.logits[0] = fold
                save_model_state(source_model, checkpoint_dir / f"best_model_fold{fold}.pt")

            events = []
            train_split = FakeSplit()
            final_dataset = {"train": train_split, "test": "holdout-split"}

            class WrappedDataset:
                def __init__(self, split, transform=None):
                    events.append(("wrapped_dataset", split, transform))

            def load_dataset():
                events.append("load_dataset")
                return final_dataset

            def build_loader(dataset, batch_size):
                events.append(("holdout_loader", batch_size))
                return "holdout-loader"

            def run_ensemble(models, dataloader, device):
                events.append(
                    (
                        "run_ensemble",
                        [float(model.logits[0]) for model in models],
                        [model.training for model in models],
                        dataloader,
                        device,
                    )
                )
                return 7, 9

            dependencies = SimpleNamespace(
                FruitHFDataset=WrappedDataset,
                build_holdout_dataloader=build_loader,
                build_validation_transform=lambda: "validation-transform",
                load_fold_models=load_fold_models,
                load_fruit_freshness_dataset=load_dataset,
                run_ensemble_holdout=run_ensemble,
            )
            args = argparse.Namespace(
                config=Path("configs/deep3.toml"),
                checkpoint_dir=checkpoint_dir,
            )

            with (
                patch.object(evaluate, "resolve_device", return_value="cpu"),
                patch.object(evaluate, "load_experiment_config", return_value=make_config()),
                patch.object(evaluate, "_load_evaluation_dependencies", return_value=dependencies),
                patch(
                    "src.inference.loading.build_cmt_classifier",
                    side_effect=TinyCheckpointClassifier,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = evaluate.run_evaluation(args)

        self.assertEqual(summary["correct"], 7)
        self.assertEqual(summary["total"], 9)
        self.assertEqual(summary["accuracy"], 7 / 9)
        self.assertIn(
            ("run_ensemble", [1.0, 2.0, 3.0], [False, False, False], "holdout-loader", "cpu"),
            events,
        )

if __name__ == "__main__":
    unittest.main()

class EvaluationDeterminismTest(unittest.TestCase):
    def _source(self, name: str) -> str:
        return (
            Path(__file__).resolve().parents[2] / "scripts" / name
        ).read_text(encoding="utf-8")

    def test_evaluation_module_applies_the_shared_policy(self):
        source = self._source("evaluate.py")

        # The bare assignment seeds nothing and leaves the autotuner in
        # whatever state the config names.
        self.assertIn("resolve_policy(config)", source)
        self.assertNotIn(
            'torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]',
            source,
        )

    def test_postholdout_evaluation_records_the_policy(self):
        source = self._source("evaluate_postholdout_baseline.py")

        # Evaluation computes the Macro F1 every decision rests on. A
        # nondeterministic evaluation would leave the phase half-served.
        self.assertIn("resolve_policy(config)", source)
        self.assertIn('"determinism": determinism', source)
        self.assertNotIn(
            'torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]',
            source,
        )
