import copy
import json
import random
import subprocess
import textwrap
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

import src.trainers.loops as loops_module
from src.engine.ema import ModelEma
from src.losses.mixup import mixup_criterion
from src.trainers.loops import train_one_epoch
from src.transforms.mixup import mixup_data


LEGACY_NOTEBOOK_COMMIT = "0f89baa"

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")


def load_legacy_train_source():
    notebook_text = subprocess.check_output(
        ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:deep3.ipynb"],
        text=True,
        encoding="utf-8",
    )
    source = "".join(json.loads(notebook_text)["cells"][4]["source"])
    start = source.index("            # ---- [Train] ----\n")
    end = source.index("            # ---- [Validation] ----\n")
    body = textwrap.dedent(source[start:end])
    return "def legacy_train_one_epoch():\n" + textwrap.indent(body, "    ") + "    return tr_acc, tr_loss\n"


def assert_nested_equal(test_case, actual, expected):
    if isinstance(actual, torch.Tensor):
        test_case.assertIsInstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        test_case.assertEqual(list(actual.keys()), list(expected.keys()))
        for key in actual:
            assert_nested_equal(test_case, actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        test_case.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_equal(test_case, actual_item, expected_item)
    else:
        test_case.assertEqual(actual, expected)


def snapshot_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def assert_rng_state_equal(test_case, actual, expected):
    test_case.assertEqual(actual["python"], expected["python"])
    test_case.assertEqual(actual["numpy"][0], expected["numpy"][0])
    np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
    test_case.assertEqual(actual["numpy"][2:], expected["numpy"][2:])
    test_case.assertTrue(torch.equal(actual["torch"], expected["torch"]))
    if actual["cuda"] is not None:
        test_case.assertEqual(len(actual["cuda"]), len(expected["cuda"]))
        for actual_state, expected_state in zip(actual["cuda"], expected["cuda"]):
            test_case.assertTrue(torch.equal(actual_state, expected_state))


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(4, 6)
        self.activation = nn.GELU()
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        return self.head(self.activation(self.hidden(self.flatten(x))))


class RecordingCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, logits, targets):
        self.calls.append(tuple(targets.detach().cpu().tolist()))
        return functional.cross_entropy(logits, targets)


def recording_tqdm(records):
    def iterate(iterable, desc=None, ncols=None):
        records.append((desc, ncols))
        return iterable

    return iterate


def record_method(instance, method_name, event_name, events):
    original = getattr(instance, method_name)

    def wrapped(*args, **kwargs):
        events.append(event_name)
        return original(*args, **kwargs)

    setattr(instance, method_name, wrapped)


class TrainEpochParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_source = load_legacy_train_source()

    @staticmethod
    def build_loader():
        inputs = torch.tensor(
            [
                [[[0.0, 1.0], [2.0, 3.0]]],
                [[[1.0, 0.0], [3.0, 2.0]]],
                [[[2.0, 3.0], [0.0, 1.0]]],
                [[[3.0, 2.0], [1.0, 0.0]]],
                [[[1.5, 2.5], [0.5, 3.5]]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
        return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)

    def build_pair(self, device):
        torch.manual_seed(701)
        legacy_model = TinyClassifier().to(device)
        extracted_model = copy.deepcopy(legacy_model).to(device)
        legacy_optimizer = torch.optim.AdamW(legacy_model.parameters(), lr=1e-3)
        extracted_optimizer = torch.optim.AdamW(extracted_model.parameters(), lr=1e-3)
        legacy_ema = ModelEma(legacy_model, decay=0.999, device=device)
        extracted_ema = ModelEma(extracted_model, decay=0.999, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            legacy_scaler = GradScaler()
            extracted_scaler = GradScaler()
        return (
            legacy_model,
            extracted_model,
            legacy_optimizer,
            extracted_optimizer,
            legacy_ema,
            extracted_ema,
            legacy_scaler,
            extracted_scaler,
        )

    def run_pair(self, *, device, is_finetuning, mixup_probability):
        (
            legacy_model,
            extracted_model,
            legacy_optimizer,
            extracted_optimizer,
            legacy_ema,
            extracted_ema,
            legacy_scaler,
            extracted_scaler,
        ) = self.build_pair(device)
        legacy_criterion = RecordingCriterion()
        extracted_criterion = RecordingCriterion()
        legacy_events, extracted_events = [], []
        for optimizer, ema, scaler, events in (
            (legacy_optimizer, legacy_ema, legacy_scaler, legacy_events),
            (extracted_optimizer, extracted_ema, extracted_scaler, extracted_events),
        ):
            record_method(optimizer, "zero_grad", "zero_grad", events)
            record_method(optimizer, "step", "optimizer_step", events)
            record_method(scaler, "scale", "scaler_scale", events)
            record_method(scaler, "step", "scaler_step", events)
            record_method(scaler, "update", "scaler_update", events)
            record_method(ema, "update", "ema_update", events)

        random.seed(811)
        np.random.seed(821)
        torch.manual_seed(831)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(841)
        initial_rng = snapshot_rng_state()

        legacy_progress, extracted_progress = [], []
        legacy_namespace = {
            "model": legacy_model,
            "train_loader": self.build_loader(),
            "criterion": legacy_criterion,
            "optimizer": legacy_optimizer,
            "device": device,
            "scaler": legacy_scaler,
            "ema": legacy_ema,
            "is_finetuning": is_finetuning,
            "MIXUP_P": mixup_probability,
            "MIXUP_ALPHA": 0.8,
            "np": np,
            "autocast": autocast,
            "mixup_data": mixup_data,
            "mixup_criterion": mixup_criterion,
            "tqdm": recording_tqdm(legacy_progress),
            "fold": 1,
            "epoch": 2,
        }
        exec(self.legacy_source, legacy_namespace)
        with patch("builtins.print") as legacy_print:
            legacy_result = legacy_namespace["legacy_train_one_epoch"]()
        legacy_rng = snapshot_rng_state()

        restore_rng_state(initial_rng)
        with patch.object(loops_module, "tqdm", recording_tqdm(extracted_progress)):
            with patch("builtins.print") as extracted_print:
                extracted_result = train_one_epoch(
                    extracted_model,
                    self.build_loader(),
                    extracted_criterion,
                    extracted_optimizer,
                    device,
                    extracted_scaler,
                    extracted_ema,
                    is_finetuning,
                    mixup_probability,
                    0.8,
                    progress_description="Fold 1 Epoch 2 [Train]",
                )
        extracted_rng = snapshot_rng_state()

        self.assertEqual(legacy_result, extracted_result)
        self.assertEqual(legacy_progress, extracted_progress)
        self.assertEqual(legacy_print.call_args_list, extracted_print.call_args_list)
        self.assertEqual(legacy_criterion.calls, extracted_criterion.calls)
        self.assertEqual(legacy_events, extracted_events)
        assert_nested_equal(self, legacy_model.state_dict(), extracted_model.state_dict())
        assert_nested_equal(self, legacy_optimizer.state_dict(), extracted_optimizer.state_dict())
        assert_nested_equal(self, legacy_scaler.state_dict(), extracted_scaler.state_dict())
        assert_nested_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())
        assert_rng_state_equal(self, legacy_rng, extracted_rng)
        self.assertEqual(legacy_result[1], extracted_result[1])
        return legacy_scaler, extracted_scaler, legacy_events

    def test_mixed_training_parity_with_final_incomplete_batch(self):
        _, _, events = self.run_pair(
            device="cpu", is_finetuning=False, mixup_probability=1.0
        )
        self.assertEqual(events.count("ema_update"), 3)
        self.assertEqual(events.count("optimizer_step"), 3)

    def test_finetuning_disables_mixup_without_consuming_mixup_rng(self):
        self.run_pair(device="cpu", is_finetuning=True, mixup_probability=1.0)

    def test_cuda_amp_and_scaler_parity_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        legacy_scaler, extracted_scaler, _ = self.run_pair(
            device="cuda", is_finetuning=False, mixup_probability=1.0
        )
        self.assertTrue(legacy_scaler.is_enabled())
        self.assertTrue(extracted_scaler.is_enabled())


if __name__ == "__main__":
    unittest.main()
