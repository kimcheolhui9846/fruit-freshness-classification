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
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

import src.trainers.loops as loops_module
from src.engine.ema import ModelEma
from src.trainers.loops import validate_one_epoch


LEGACY_NOTEBOOK_COMMIT = "0f89baa"

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")


def load_legacy_validation_source():
    notebook_text = subprocess.check_output(
        ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:deep3.ipynb"],
        text=True,
        encoding="utf-8",
    )
    source = "".join(json.loads(notebook_text)["cells"][4]["source"])
    start = source.index("            # ---- [Validation] ----\n")
    end = source.index('            history["train_loss"].append(tr_loss)\n')
    body = textwrap.dedent(source[start:end])
    return (
        "def legacy_validate_one_epoch():\n"
        + textwrap.indent(body, "    ")
        + "    return va_acc, va_loss, all_preds, all_labels, all_logits\n"
    )


def assert_state_equal(test_case, actual, expected):
    test_case.assertEqual(list(actual.keys()), list(expected.keys()))
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


def assert_rng_equal(test_case, actual, expected):
    test_case.assertEqual(actual[0], expected[0])
    test_case.assertEqual(actual[1][0], expected[1][0])
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    test_case.assertEqual(actual[1][2:], expected[1][2:])
    test_case.assertTrue(torch.equal(actual[2], expected[2]))


class TinyValidationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(4, 5)
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        return self.head(torch.tanh(self.hidden(self.flatten(x))))


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


class ValidateEpochParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_source = load_legacy_validation_source()

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

    def test_validation_return_state_mode_and_rng_parity(self):
        torch.manual_seed(907)
        legacy_model = TinyValidationModel()
        extracted_model = copy.deepcopy(legacy_model)
        legacy_ema = ModelEma(legacy_model, decay=0.999, device="cpu")
        extracted_ema = ModelEma(extracted_model, decay=0.999, device="cpu")
        legacy_criterion = RecordingCriterion()
        extracted_criterion = RecordingCriterion()
        legacy_before = copy.deepcopy(legacy_ema.module.state_dict())
        extracted_before = copy.deepcopy(extracted_ema.module.state_dict())
        random.seed(911)
        np.random.seed(913)
        torch.manual_seed(917)
        rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

        legacy_progress, extracted_progress = [], []
        namespace = {
            "ema": legacy_ema,
            "val_loader": self.build_loader(),
            "criterion": legacy_criterion,
            "device": "cpu",
            "torch": torch,
            "autocast": autocast,
            "tqdm": recording_tqdm(legacy_progress),
            "fold": 1,
            "epoch": 2,
        }
        exec(self.legacy_source, namespace)
        legacy_result = namespace["legacy_validate_one_epoch"]()
        legacy_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

        random.setstate(rng_before[0])
        np.random.set_state(rng_before[1])
        torch.set_rng_state(rng_before[2])
        with patch.object(loops_module, "tqdm", recording_tqdm(extracted_progress)):
            extracted_result = validate_one_epoch(
                extracted_ema.module,
                self.build_loader(),
                extracted_criterion,
                "cpu",
                progress_description="Fold 1 Epoch 2 [Val]",
            )
        extracted_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

        self.assertEqual(legacy_result[:4], extracted_result[:4])
        self.assertEqual(len(legacy_result[4]), len(extracted_result[4]))
        for legacy_logits, extracted_logits in zip(legacy_result[4], extracted_result[4]):
            np.testing.assert_array_equal(legacy_logits, extracted_logits)
        self.assertEqual(legacy_criterion.calls, extracted_criterion.calls)
        self.assertEqual(legacy_progress, extracted_progress)
        assert_state_equal(self, legacy_ema.module.state_dict(), legacy_before)
        assert_state_equal(self, extracted_ema.module.state_dict(), extracted_before)
        self.assertFalse(legacy_ema.module.training)
        self.assertFalse(extracted_ema.module.training)
        self.assertTrue(all(parameter.grad is None for parameter in legacy_ema.module.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in extracted_ema.module.parameters()))
        assert_rng_equal(self, legacy_rng, extracted_rng)


if __name__ == "__main__":
    unittest.main()
