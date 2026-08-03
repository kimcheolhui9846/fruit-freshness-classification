"""Behavioral tests for trusted epoch-boundary training state."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.engine.ema import ModelEma
from src.engine.training_state import (
    STATE_SCHEMA_VERSION,
    build_training_state,
    capture_rng_state,
    load_training_state,
    restore_rng_state,
    save_training_state_atomic,
    validate_training_state,
)


def make_metadata(**overrides):
    metadata = {
        "run_id": "deep3-canonical-reference-01",
        "repository_commit": "a" * 40,
        "config_path": "configs/deep3_canonical.toml",
        "config_sha256": "b" * 64,
        "dataset_repository": "Densu341/Fresh-rotten-fruit",
        "dataset_revision": "2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
        "dataset_archive_sha256": "c" * 64,
        "label_names": ["fresh", "rotten"],
        "label_names_sha256": "d" * 64,
        "num_classes": 2,
        "num_folds": 2,
        "epochs": 3,
        "fine_tuning_epochs": 1,
        "batch_size": 64,
    }
    metadata.update(overrides)
    return metadata


def make_runtime_objects():
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    ema = ModelEma(model, decay=0.999, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler("cpu")
    return model, ema, optimizer, scheduler, scaler


def make_state(**overrides):
    model, ema, optimizer, scheduler, scaler = make_runtime_objects()
    state = build_training_state(
        metadata=make_metadata(),
        status="RUNNING",
        current_fold=1,
        completed_epoch=1,
        next_fold=1,
        next_epoch=2,
        best_accuracy_current_fold=0.4,
        current_fold_history={
            "train_loss": [0.9],
            "train_acc": [0.5],
            "val_loss": [0.8],
            "val_acc": [0.4],
        },
        completed_fold_histories=[],
        completed_fold_accuracies=[],
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_indices=[0, 2, 4],
        validation_indices=[1, 3, 5],
    )
    state.update(overrides)
    return state


class TrainingStateTest(unittest.TestCase):
    def test_schema_captures_every_operational_component(self):
        state = make_state()

        self.assertEqual(state["schema_version"], STATE_SCHEMA_VERSION)
        self.assertEqual(state["status"], "RUNNING")
        self.assertEqual(state["next_epoch"], 2)
        self.assertEqual(state["train_indices_sha256"], "a955c6c593a4f03328fb14905a735f8b66b60b5c75756d5ebbb30b81ba47b2ba")
        self.assertEqual(state["validation_indices_sha256"], "a5e64abb5dacf3e7febe111383d2f1935cebbcb6ae50465e134a2ba1c26e6e0d")
        for field in (
            "model_state_dict",
            "ema_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "grad_scaler_state_dict",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
        ):
            self.assertIn(field, state)

    def test_rng_restore_replays_python_numpy_and_torch_sequences(self):
        random.seed(101)
        np.random.seed(101)
        torch.manual_seed(101)
        captured = capture_rng_state()

        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
        )
        random.random()
        np.random.random()
        torch.rand(3)

        restore_rng_state(captured)

        self.assertEqual(random.random(), expected[0])
        self.assertEqual(float(np.random.random()), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_atomic_save_replaces_state_without_leaving_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_state.pt"
            state = make_state()
            save_training_state_atomic(state, path)

            self.assertTrue(path.is_file())
            self.assertEqual(
                [candidate.name for candidate in Path(directory).iterdir()],
                ["training_state.pt"],
            )
            loaded = load_training_state(path, trusted_local=True)
            self.assertEqual(loaded["next_epoch"], 2)

    def test_failed_atomic_write_keeps_previous_valid_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_state.pt"
            previous = make_state(next_epoch=2)
            save_training_state_atomic(previous, path)
            previous_bytes = path.read_bytes()

            with patch("src.engine.training_state.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_training_state_atomic(make_state(next_epoch=3), path)

            self.assertEqual(path.read_bytes(), previous_bytes)
            self.assertEqual(
                [candidate.name for candidate in Path(directory).iterdir()],
                ["training_state.pt"],
            )

    def test_trusted_local_load_is_explicit_and_corrupt_state_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_state.pt"
            save_training_state_atomic(make_state(), path)

            with self.assertRaisesRegex(PermissionError, "trusted local"):
                load_training_state(path, trusted_local=False)

            path.write_bytes(b"not a torch state")
            with self.assertRaisesRegex(ValueError, "Could not load"):
                load_training_state(path, trusted_local=True)

    def test_validation_rejects_schema_metadata_and_completed_state_mismatches(self):
        state = make_state()
        validate_training_state(
            state,
            expected_metadata=make_metadata(),
            expected_train_indices=[0, 2, 4],
            expected_validation_indices=[1, 3, 5],
        )

        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_training_state({**state, "schema_version": 999})

        for field, value in (
            ("config_sha256", "wrong"),
            ("run_id", "other-run"),
            ("batch_size", 32),
            ("dataset_revision", "other-revision"),
            ("label_names", ["fresh", "other"]),
        ):
            mismatched = copy.deepcopy(state)
            mismatched[field] = value
            with self.assertRaisesRegex(ValueError, field):
                validate_training_state(mismatched, expected_metadata=make_metadata())

        with self.assertRaisesRegex(ValueError, "train_indices_sha256"):
            validate_training_state(
                state,
                expected_metadata=make_metadata(),
                expected_train_indices=[0, 1, 2],
            )

        completed = copy.deepcopy(state)
        completed["status"] = "COMPLETED"
        with self.assertRaisesRegex(ValueError, "completed"):
            validate_training_state(completed, expected_metadata=make_metadata())

    def test_restore_moves_cuda_rng_states_back_to_cpu_after_cuda_map_location(self):
        state = capture_rng_state()
        mapped_rng_state = unittest.mock.Mock()
        restored_cpu_state = torch.get_rng_state()
        mapped_rng_state.cpu.return_value = restored_cpu_state
        state["torch_cuda_rng_states"] = [mapped_rng_state]

        with (
            patch("src.engine.training_state.torch.cuda.is_available", return_value=True),
            patch("src.engine.training_state.torch.cuda.set_rng_state_all") as set_rng_state_all,
        ):
            restore_rng_state(state)

        mapped_rng_state.cpu.assert_called_once_with()
        set_rng_state_all.assert_called_once_with([restored_cpu_state])
    def test_restore_moves_cpu_rng_state_back_to_cpu_after_cuda_map_location(self):
        state = capture_rng_state()
        mapped_rng_state = unittest.mock.Mock()
        restored_cpu_state = torch.get_rng_state()
        mapped_rng_state.cpu.return_value = restored_cpu_state
        state["torch_cpu_rng_state"] = mapped_rng_state

        with patch("src.engine.training_state.torch.set_rng_state") as set_rng_state:
            restore_rng_state(state)

        mapped_rng_state.cpu.assert_called_once_with()
        set_rng_state.assert_called_once_with(restored_cpu_state)

    def test_cpu_rng_capture_and_restore_handles_absent_cuda_state(self):
        with patch("src.engine.training_state.torch.cuda.is_available", return_value=False):
            captured = capture_rng_state()
            self.assertIsNone(captured["torch_cuda_rng_states"])
            restore_rng_state(captured)


if __name__ == "__main__":
    unittest.main()
