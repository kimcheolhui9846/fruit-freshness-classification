"""Synthetic integration tests for optional epoch-boundary training resume."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.engine.ema import ModelEma
from src.engine.training_state import load_training_state


TRAIN_PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'
TRAIN_SPEC = importlib.util.spec_from_file_location('phase82_resumable_training', TRAIN_PATH)
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


class StopAfterStateSave(RuntimeError):
    """Test-only interruption raised after a durable epoch-boundary state."""


class FakeSplit:
    def __init__(self, labels=(0, 1, 0, 1)):
        self.features = {"label": type("Label", (), {"names": ["fresh", "rotten"]})()}
        self._labels = list(labels)

    def __getitem__(self, key):
        if key == "label":
            return self._labels
        raise KeyError(key)

    def select(self, indices):
        return FakeSplit([self._labels[index] for index in indices])


class ResumeHarness:
    def __init__(self):
        self.events = []
        self.saved_models = []
        self.schedulers = []
        self.dataset_loads = 0
        self.final_dataset = {"train": FakeSplit(), "test": FakeSplit()}

    @staticmethod
    def config():
        return {
            "runtime": {"cudnn_benchmark": True},
            "loss": {
                "class_balanced_beta": 0.999,
                "use_ce_label_smoothing": True,
                "label_smoothing": 0.01,
                "focal_gamma": 2.0,
            },
            "training": {"epochs": 3, "batch_size": 64},
            "fine_tuning": {"epochs": 1},
            "cross_validation": {"n_splits": 2, "shuffle": True, "random_state": 42},
            "mixup": {"alpha": 0.8, "probability": 0.5},
            "optimization": {"lr_cnn": 5e-5, "lr_trans": 1e-4, "weight_decay": 1e-4},
            "ema": {"decay": 0.999},
            "checkpoint": {"final_model_filename": "last_model_weights.pt"},
            "reporting": {"figure_size": [10, 4]},
        }

    def dependencies(self):
        harness = self

        class WrappedDataset:
            def __init__(self, split, transform=None):
                self.split = split
                self.tf = transform

        class Scaler:
            def __init__(self):
                self.value = 0

            def state_dict(self):
                return {"value": self.value}

            def load_state_dict(self, state):
                self.value = state["value"]

        def load_dataset():
            harness.dataset_loads += 1
            harness.events.append("load_dataset")
            return harness.final_dataset

        def iter_folds(dataset, n_splits, shuffle, random_state):
            harness.events.append(("folds", n_splits, shuffle, random_state))
            return iter(
                (
                    (np.asarray([0, 1]), np.asarray([2, 3])),
                    (np.asarray([2, 3]), np.asarray([0, 1])),
                )
            )

        def build_model(num_classes):
            harness.events.append(("model", num_classes))
            return torch.nn.Linear(1, num_classes)

        def build_ema(model, decay, device):
            return ModelEma(model, decay=decay, device=device)

        def build_optimizer(model, lr_cnn, lr_trans, weight_decay):
            return torch.optim.AdamW(model.parameters(), lr=lr_cnn, weight_decay=weight_decay)

        def build_scheduler(optimizer, t_max):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            harness.schedulers.append(scheduler)
            return scheduler

        def train_epoch(model, dataloader, criterion, optimizer, device, scaler, ema, is_finetuning, mixup_probability, mixup_alpha, progress_description):
            optimizer.zero_grad(set_to_none=True)
            output = model(torch.ones(2, 1))
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            ema.update(model)
            harness.events.append(("train", progress_description, is_finetuning))
            return 0.5, float(loss.detach())

        def validate_epoch(model, dataloader, criterion, device, progress_description):
            fold = int(progress_description.split()[1])
            epoch = int(progress_description.split()[3])
            score = fold * 0.1 + epoch * 0.01
            harness.events.append(("validate", fold, epoch))
            return score, 1.0 - score, [0, 1], [0, 1], [np.eye(2)]

        def compute_metrics(labels, predictions, logits):
            return 0.5, 0.5, 1.0, 1.0

        def ensure_output(path):
            Path(path).mkdir(parents=True, exist_ok=True)
            return path

        def save_labels(names, directory):
            Path(directory, "label_names.json").write_text("[\"fresh\", \"rotten\"]", encoding="utf-8")

        def save_model(model, path):
            harness.saved_models.append((Path(path).name, copy.deepcopy(model.state_dict())))

        return SimpleNamespace(
            FruitHFDataset=WrappedDataset,
            FocalLoss=object,
            ModelEma=build_ema,
            build_class_balanced_alpha=lambda counts, beta, classes: torch.ones(classes),
            build_cmt_classifier=build_model,
            build_finetune_transform=lambda: "finetune",
            build_fold_checkpoint_path=lambda directory, fold: str(Path(directory) / f"best_model_fold{fold}.pt"),
            build_fold_dataloaders=lambda train_ds, val_ds, batch: ("train", "validation"),
            build_optimizer=build_optimizer,
            build_scheduler=build_scheduler,
            build_train_transform=lambda: "train",
            build_validation_transform=lambda: "validation",
            compute_validation_metrics=compute_metrics,
            ensure_output_directory=ensure_output,
            iter_stratified_folds=iter_folds,
            load_fruit_freshness_dataset=load_dataset,
            save_label_names=save_labels,
            save_model_state=save_model,
            select_fold_datasets=lambda dataset, train_indices, validation_indices: (dataset.select(train_indices), dataset.select(validation_indices)),
            train_one_epoch=train_epoch,
            validate_one_epoch=validate_epoch,
            dataset_repository="Densu341/Fresh-rotten-fruit",
            dataset_revision="2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
            dataset_archive_sha256="a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e",
            scaler_factory=Scaler,
        )

    @staticmethod
    def args(output_dir, *, resume_state=None, save=True, require_empty=False):
        return argparse.Namespace(
            config=Path("configs/deep3_canonical.toml"),
            output_dir=Path(output_dir),
            resume_state=Path(resume_state) if resume_state else None,
            save_training_state=save,
            require_empty_output_dir=require_empty,
            run_id="deep3-canonical-reference-01",
        )

    def run(self, args, *, interrupt_after=None):
        dependencies = self.dependencies()
        original_save = train.save_training_state_atomic

        def save_then_interrupt(state, path):
            original_save(state, path)
            if interrupt_after is not None and state["status"] == interrupt_after:
                raise StopAfterStateSave(interrupt_after)

        with (
            patch.object(train, "resolve_device", return_value=torch.device("cpu")),
            patch.object(train, "load_experiment_config", return_value=self.config()),
            patch.object(train, "_load_training_dependencies", return_value=dependencies),
            patch.object(train, "_repository_commit", return_value="a" * 40),
            patch.object(train, "GradScaler", side_effect=dependencies.scaler_factory),
            patch.object(train, "save_training_state_atomic", side_effect=save_then_interrupt),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return train.run_training(args)


class ResumableTrainingIntegrationTest(unittest.TestCase):
    def test_non_empty_canonical_output_is_rejected_before_dataset_loading(self):
        harness = ResumeHarness()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "occupied"
            output_dir.mkdir()
            (output_dir / ".hidden").write_text("do not overwrite", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "must be empty"):
                harness.run(harness.args(output_dir, require_empty=True))

            self.assertEqual(harness.dataset_loads, 0)

    def test_epoch_boundary_resume_matches_uninterrupted_history_and_final_state(self):
        torch.manual_seed(314)
        with tempfile.TemporaryDirectory() as directory:
            uninterrupted_dir = Path(directory) / "uninterrupted"
            uninterrupted = ResumeHarness()
            uninterrupted_summary = uninterrupted.run(
                uninterrupted.args(uninterrupted_dir, require_empty=True)
            )

            torch.manual_seed(314)
            interrupted_dir = Path(directory) / "interrupted"
            interrupted = ResumeHarness()
            with self.assertRaises(StopAfterStateSave):
                interrupted.run(
                    interrupted.args(interrupted_dir, require_empty=True),
                    interrupt_after="RUNNING",
                )

            manifest = json.loads(
                (interrupted_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["resume_enabled"])

            state_path = interrupted_dir / "training_state.pt"
            state = load_training_state(
                state_path,
                trusted_local=True,
                allow_completed=True,
            )
            self.assertEqual(state["completed_epoch"], 1)
            self.assertEqual(state["next_epoch"], 2)

            resumed = ResumeHarness()
            resumed_summary = resumed.run(
                resumed.args(interrupted_dir, resume_state=state_path)
            )

            self.assertEqual(resumed_summary["histories"], uninterrupted_summary["histories"])
            self.assertEqual(
                resumed_summary["fold_accuracies"],
                uninterrupted_summary["fold_accuracies"],
            )
            self.assertEqual(
                [name for name, _ in resumed.saved_models][-1],
                "last_model_weights.pt",
            )
            for actual, expected in zip(
                resumed.saved_models[-1][1].values(),
                uninterrupted.saved_models[-1][1].values(),
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(
                [scheduler.last_epoch for scheduler in resumed.schedulers],
                [3, 3],
            )

            uninterrupted_state = load_training_state(
                uninterrupted_dir / "training_state.pt",
                trusted_local=True,
                allow_completed=True,
            )
            resumed_state = load_training_state(
                interrupted_dir / "training_state.pt",
                trusted_local=True,
                allow_completed=True,
            )
            self.assertEqual(uninterrupted_state["scheduler_state_dict"], resumed_state["scheduler_state_dict"])
            self.assertEqual(uninterrupted_state["optimizer_state_dict"].keys(), resumed_state["optimizer_state_dict"].keys())
            self.assertEqual(uninterrupted_state["optimizer_state_dict"]["param_groups"], resumed_state["optimizer_state_dict"]["param_groups"])
            for parameter_id, expected_state in uninterrupted_state["optimizer_state_dict"]["state"].items():
                actual_state = resumed_state["optimizer_state_dict"]["state"][parameter_id]
                for key, expected_value in expected_state.items():
                    actual_value = actual_state[key]
                    if torch.is_tensor(expected_value):
                        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual_value, expected_value)
            for key, expected_value in uninterrupted_state["ema_state_dict"].items():
                actual_value = resumed_state["ema_state_dict"][key]
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
                else:
                    self.assertEqual(actual_value, expected_value)
            self.assertEqual(set(uninterrupted_summary), set(resumed_summary))
    def test_fold_boundary_resume_starts_the_next_fold_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "fold-boundary"
            interrupted = ResumeHarness()
            with self.assertRaises(StopAfterStateSave):
                interrupted.run(
                    interrupted.args(output_dir, require_empty=True),
                    interrupt_after="FOLD_COMPLETE",
                )

            resumed = ResumeHarness()
            resumed.run(
                resumed.args(output_dir, resume_state=output_dir / "training_state.pt")
            )

            started_folds = [event for event in resumed.events if event[0] == "model"]
            self.assertEqual(len(started_folds), 1)
            self.assertEqual(
                [event for event in resumed.events if event[0] == "validate"],
                [( "validate", 2, 1), ("validate", 2, 2), ("validate", 2, 3)],
            )

    def test_finetuning_boundary_helper_uses_epochs_100_101_and_119(self):
        self.assertFalse(train._is_finetuning_epoch(100, 120, 20))
        self.assertTrue(train._is_finetuning_epoch(101, 120, 20))
        self.assertTrue(train._is_finetuning_epoch(119, 120, 20))

    def test_legacy_invocation_remains_stateless(self):
        harness = ResumeHarness()
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                config=Path("configs/deep3.toml"),
                output_dir=Path(directory) / "legacy",
            )
            harness.run(args)

            self.assertFalse((Path(directory) / "legacy" / "training_state.pt").exists())
            self.assertFalse((Path(directory) / "legacy" / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
