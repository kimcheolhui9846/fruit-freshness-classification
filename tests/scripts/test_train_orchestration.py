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

import numpy as np
import torch


TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("phase53_train_orchestration", TRAIN_PATH)
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


class FakeSplit:
    def __init__(self, labels):
        self.features = {"label": SimpleNamespace(names=["fresh", "rotten", "mixed"])}
        self.labels = labels

    def __getitem__(self, key):
        if key == "label":
            return self.labels
        raise KeyError(key)


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class FakeEma:
    def __init__(self, model, decay, device):
        self.module = SimpleNamespace(
            name=f"ema-{model.name}",
            source=model,
            decay=decay,
            device=device,
        )


class FakeScheduler:
    def __init__(self, events):
        self.events = events

    def step(self):
        self.events.append("scheduler_step")


class SyntheticTrainingHarness:
    def __init__(self, config, validation_scores):
        self.config = config
        self.validation_scores = iter(validation_scores)
        self.events = []
        self.saved_models = []
        self.train_datasets = []
        self.final_dataset_accesses = []
        self.models = []

        train_split = FakeSplit([0, 1, 2, 0, 1, 2])
        test_split = FakeSplit([0, 1, 2])

        class FinalDataset(dict):
            def __getitem__(inner_self, key):
                self.final_dataset_accesses.append(key)
                return super(FinalDataset, inner_self).__getitem__(key)

        self.final_dataset = FinalDataset(train=train_split, test=test_split)

    def dependencies(self):
        harness = self

        class WrappedDataset:
            def __init__(self, split, transform=None):
                self.split = split
                self.tf = transform
                if transform == "train-transform":
                    harness.train_datasets.append(self)

        def load_dataset():
            harness.events.append("load_dataset")
            return harness.final_dataset

        def iter_folds(dataset, n_splits, shuffle, random_state):
            harness.events.append(("iter_folds", n_splits, shuffle, random_state))
            splits = (([0, 1, 2], [3, 4, 5]), ([3, 4, 5], [0, 1, 2]))
            return iter(splits[:n_splits])

        def select_folds(dataset, train_indices, validation_indices):
            harness.events.append(("select", tuple(train_indices), tuple(validation_indices)))
            return "train-split", "validation-split"

        def build_loaders(train_dataset, validation_dataset, batch_size):
            harness.events.append(("loaders", batch_size))
            return "train-loader", "validation-loader"

        def build_model(num_classes):
            model = FakeModel(f"model-{len(harness.models) + 1}")
            harness.models.append(model)
            harness.events.append(("model", num_classes))
            return model

        def build_ema(model, decay, device):
            harness.events.append(("ema", model.name, decay, device))
            return FakeEma(model, decay, device)

        def build_alpha(class_counts, beta, num_classes):
            harness.events.append(("alpha", tuple(class_counts), beta, num_classes))
            return torch.ones(num_classes)

        def build_optimizer(model, lr_cnn, lr_trans, weight_decay):
            harness.events.append(("optimizer", model.name, lr_cnn, lr_trans, weight_decay))
            return SimpleNamespace(model=model)

        def build_scheduler(optimizer, t_max):
            harness.events.append(("scheduler", optimizer.model.name, t_max))
            return FakeScheduler(harness.events)

        def train_epoch(*args, **kwargs):
            harness.events.append(("train", args[7], args[8], args[9]))
            return 0.25, 0.75

        def validate_epoch(*args, **kwargs):
            score = next(harness.validation_scores)
            harness.events.append(("validate", args[0].name, kwargs["progress_description"]))
            return score, 0.5, [0, 1, 2], [0, 1, 2], [np.eye(3)]

        def compute_metrics(labels, predictions, logits):
            harness.events.append(("metrics", tuple(labels), tuple(predictions), tuple(logits.shape)))
            return 0.4, 0.5, 0.6, 0.7

        def ensure_output(path):
            harness.events.append(("ensure_output", Path(path)))
            return path

        def save_labels(names, output_dir):
            harness.events.append(("save_labels", tuple(names), Path(output_dir)))

        def checkpoint_path(directory, fold):
            return str(Path(directory) / f"best_model_fold{fold}.pt")

        def save_model(model, path):
            harness.saved_models.append((model, Path(path)))
            harness.events.append(("save_model", getattr(model, "name", None), Path(path).name))

        return SimpleNamespace(
            FruitHFDataset=WrappedDataset,
            FocalLoss=object,
            ModelEma=build_ema,
            build_class_balanced_alpha=build_alpha,
            build_cmt_classifier=build_model,
            build_finetune_transform=lambda: "finetune-transform",
            build_fold_checkpoint_path=checkpoint_path,
            build_fold_dataloaders=build_loaders,
            build_optimizer=build_optimizer,
            build_scheduler=build_scheduler,
            build_train_transform=lambda: "train-transform",
            build_validation_transform=lambda: "validation-transform",
            compute_validation_metrics=compute_metrics,
            ensure_output_directory=ensure_output,
            iter_stratified_folds=iter_folds,
            load_fruit_freshness_dataset=load_dataset,
            save_label_names=save_labels,
            save_model_state=save_model,
            select_fold_datasets=select_folds,
            train_one_epoch=train_epoch,
            validate_one_epoch=validate_epoch,
        )


def make_config(*, epochs=3, finetune_epochs=1, folds=2):
    return {
        "runtime": {"cudnn_benchmark": True},
        "loss": {
            "class_balanced_beta": 0.999,
            "use_ce_label_smoothing": True,
            "label_smoothing": 0.01,
            "focal_gamma": 2.0,
        },
        "training": {"epochs": epochs, "batch_size": 192},
        "fine_tuning": {"epochs": finetune_epochs},
        "cross_validation": {"n_splits": folds, "shuffle": True, "random_state": 42},
        "mixup": {"alpha": 0.8, "probability": 0.5},
        "optimization": {"lr_cnn": 5e-5, "lr_trans": 1e-4, "weight_decay": 1e-4},
        "ema": {"decay": 0.999},
        "checkpoint": {"final_model_filename": "last_model_weights.pt"},
        "reporting": {"figure_size": [10, 4]},
    }


class TrainOrchestrationTest(unittest.TestCase):
    def run_with_harness(self, harness, output_dir):
        args = argparse.Namespace(
            config=Path("configs/deep3.toml"),
            output_dir=Path(output_dir),
        )
        previous_benchmark = torch.backends.cudnn.benchmark
        try:
            with (
                patch.object(
                    train,
                    "resolve_device",
                    side_effect=lambda: harness.events.append("device") or "cpu",
                ),
                patch.object(
                    train,
                    "load_experiment_config",
                    side_effect=lambda path: harness.events.append(("config", Path(path))) or harness.config,
                ),
                patch.object(train, "_load_training_dependencies", return_value=harness.dependencies()),
                patch.object(train, "GradScaler", side_effect=lambda: SimpleNamespace(name="scaler")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return train.run_training(args)
        finally:
            torch.backends.cudnn.benchmark = previous_benchmark

    def test_synthetic_orchestration_preserves_fold_epoch_and_finetuning_order(self):
        harness = SyntheticTrainingHarness(
            make_config(epochs=3, finetune_epochs=1, folds=2),
            validation_scores=[0.5] * 6,
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_with_harness(harness, directory)

            # Phase 9.7 reversed these two: the config must be read before
            # the device is resolved, so the determinism policy can set
            # CUBLAS_WORKSPACE_CONFIG before any CUDA work reads it.
            self.assertEqual(
                harness.events[0],
                ("config", train.REPOSITORY_ROOT / "configs/deep3.toml"),
            )
            self.assertEqual(harness.events[1], "device")
            self.assertIn(("iter_folds", 2, True, 42), harness.events)
            self.assertEqual(len([event for event in harness.events if event[0] == "model"]), 2)
            self.assertEqual(len([event for event in harness.events if event[0] == "ema"]), 2)
            self.assertEqual(len([event for event in harness.events if event[0] == "optimizer"]), 2)
            self.assertEqual(len([event for event in harness.events if event[0] == "scheduler"]), 2)
            train_events = [event for event in harness.events if event[0] == "train"]
            self.assertEqual(len(train_events), 6)
            self.assertEqual([event[1] for event in train_events], [False, False, True] * 2)
            self.assertTrue(all(event[2:] == (0.5, 0.8) for event in train_events))
            self.assertEqual(len([event for event in harness.events if event == "scheduler_step"]), 6)
            self.assertEqual(len(summary["histories"]), 2)
            self.assertEqual([len(history["val_acc"]) for history in summary["histories"]], [3, 3])
            self.assertEqual(len(harness.train_datasets), 2)
            self.assertTrue(all(dataset.tf == "finetune-transform" for dataset in harness.train_datasets))
            self.assertNotIn("test", harness.final_dataset_accesses)

    def test_checkpoint_policy_uses_ema_for_improvements_and_raw_model_for_final_save(self):
        harness = SyntheticTrainingHarness(
            make_config(epochs=3, finetune_epochs=1, folds=1),
            validation_scores=[0.5, 0.5, 0.6],
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_with_harness(harness, directory)

            self.assertEqual(
                [(model.name, path.name) for model, path in harness.saved_models],
                [
                    ("ema-model-1", "best_model_fold1.pt"),
                    ("ema-model-1", "best_model_fold1.pt"),
                    ("model-1", "last_model_weights.pt"),
                ],
            )
            self.assertEqual(summary["final_model_path"], Path(directory) / "last_model_weights.pt")
            self.assertEqual(len([event for event in harness.events if event == "scheduler_step"]), 3)

    def test_output_directory_failure_is_propagated_without_checkpoint_writes(self):
        harness = SyntheticTrainingHarness(
            make_config(epochs=1, finetune_epochs=1, folds=1),
            validation_scores=[0.5],
        )
        dependencies = harness.dependencies()

        def reject_output_directory(path):
            raise PermissionError("selected output directory is unavailable")

        dependencies.ensure_output_directory = reject_output_directory
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "unavailable"
            args = argparse.Namespace(
                config=Path("configs/deep3.toml"),
                output_dir=output_dir,
            )
            with (
                patch.object(train, "resolve_device", return_value="cpu"),
                patch.object(train, "load_experiment_config", return_value=harness.config),
                patch.object(train, "_load_training_dependencies", return_value=dependencies),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(PermissionError),
            ):
                train.run_training(args)

            self.assertFalse(output_dir.exists())
            self.assertEqual(harness.saved_models, [])

if __name__ == "__main__":
    unittest.main()


def _manifest_metadata() -> dict:
    return {
        "run_id": "determinism-check-a",
        "repository_commit": "0" * 40,
        "config_path": "configs/deep3_postholdout_determinism_check.toml",
        "config_sha256": "1" * 64,
        "dataset_repository": "Densu341/Fresh-rotten-fruit",
        "dataset_revision": "2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
        "dataset_archive_sha256": "2" * 64,
        "num_classes": 14,
        "num_folds": 3,
        "epochs": 2,
        "fine_tuning_epochs": 1,
        "batch_size": 64,
    }


def _manifest_config() -> dict:
    return {
        "optimization": {"lr_cnn": 5e-5, "lr_trans": 1e-4, "weight_decay": 1e-4},
        "mixup": {"alpha": 0.8, "probability": 0.5},
        "ema": {"decay": 0.999},
    }


class DeterminismManifestTest(unittest.TestCase):
    def test_manifest_schema_version_is_two(self):
        # Adding a field without bumping the version would let a manifest
        # with a determinism block and one without both claim version 1.
        self.assertEqual(train.RUN_MANIFEST_SCHEMA_VERSION, 2)

    def test_manifest_records_an_unseeded_run_explicitly(self):
        manifest = train._build_run_manifest(
            metadata=_manifest_metadata(),
            config=_manifest_config(),
            device=torch.device("cpu"),
            resume_enabled=True,
            determinism={
                "seed": None,
                "level": None,
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
                "use_deterministic_algorithms": False,
                "cublas_workspace_config": None,
            },
        )

        # "This run was unseeded" is the fact Phase 9.7 exists to make
        # visible. It must be recorded, not omitted.
        self.assertIn("determinism", manifest)
        self.assertIsNone(manifest["determinism"]["seed"])
        self.assertIsNone(manifest["determinism"]["level"])

    def test_manifest_records_an_applied_policy(self):
        manifest = train._build_run_manifest(
            metadata=_manifest_metadata(),
            config=_manifest_config(),
            device=torch.device("cpu"),
            resume_enabled=True,
            determinism={
                "seed": 20260815,
                "level": "A_STRICT",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "use_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
            },
        )

        self.assertEqual(manifest["determinism"]["seed"], 20260815)
        self.assertEqual(manifest["determinism"]["level"], "A_STRICT")
        self.assertEqual(manifest["schema_version"], 2)

    def test_manifest_determinism_block_is_a_copy(self):
        record = {
            "seed": 20260815,
            "level": "B_CUDNN",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "use_deterministic_algorithms": False,
            "cublas_workspace_config": None,
        }
        manifest = train._build_run_manifest(
            metadata=_manifest_metadata(),
            config=_manifest_config(),
            device=torch.device("cpu"),
            resume_enabled=True,
            determinism=record,
        )
        record["seed"] = 999

        # A manifest records what happened; a later mutation of the caller's
        # dict must not rewrite it.
        self.assertEqual(manifest["determinism"]["seed"], 20260815)

    def test_policy_is_applied_before_the_device_is_resolved(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        policy_at = source.index("determinism = resolve_policy(config)")
        device_at = source.index("device = resolve_device()")

        # CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle is created,
        # so a policy applied after any CUDA work is ignored, not refused.
        self.assertLess(policy_at, device_at)

    def test_per_fold_cudnn_assignment_is_gone(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")

        # The policy is applied once at start-up. Leaving the per-fold
        # assignment would silently re-enable the autotuner mid-run.
        self.assertNotIn(
            'torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]',
            source,
        )
