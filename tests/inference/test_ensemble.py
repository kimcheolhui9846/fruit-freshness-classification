import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from src.inference.ensemble import ensemble_logits_tta_hflip, run_ensemble_holdout


class RecordingFlipModel(nn.Module):
    def __init__(self, name, scale, bias, trace):
        super().__init__()
        self.name = name
        self.trace = trace
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("bias", torch.tensor(bias))

    def forward(self, images):
        self.trace.append((self.name, images.detach().clone()))
        left = images[:, 0, 0, 0]
        right = images[:, 0, 0, -1]
        return torch.stack((left * self.scale + self.bias, right * self.scale - self.bias), dim=1)


def build_models(count, trace, device="cpu"):
    models = []
    for index in range(count):
        model = RecordingFlipModel(f"model-{index}", index + 1.0, index / 10, trace).to(device)
        model.eval()
        models.append(model)
    return models


@torch.inference_mode()
def legacy_ensemble_logits(models, x):
    logits_sum = 0
    for model in models:
        logits_sum = logits_sum + model(x)
    return logits_sum / len(models)


@torch.inference_mode()
def legacy_ensemble_logits_tta_hflip(models, x):
    x_flip = torch.flip(x, dims=[3])
    logits = legacy_ensemble_logits(models, x)
    logits_flip = legacy_ensemble_logits(models, x_flip)
    return (logits + logits_flip) / 2


def legacy_run_ensemble_holdout(models, dataloader, device):
    t_total = t_correct = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        with autocast("cuda"):
            logits = legacy_ensemble_logits_tta_hflip(models, x)
        pred = logits.argmax(1)
        t_correct += (pred == y).sum().item()
        t_total += y.size(0)
    return t_correct, t_total


def assert_trace_equal(test_case, actual, expected):
    test_case.assertEqual([name for name, _ in actual], [name for name, _ in expected])
    for (_, actual_images), (_, expected_images) in zip(actual, expected):
        torch.testing.assert_close(actual_images, expected_images, rtol=0, atol=0)


class HoldoutEnsembleParityTest(unittest.TestCase):
    def make_dataset(self, model_count):
        images = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],
                [[[4.0, 3.0, 2.0, 1.0]]],
                [[[2.0, 1.0, 4.0, 3.0]]],
                [[[3.0, 4.0, 1.0, 2.0]]],
                [[[5.0, 1.0, 2.0, 4.0]]],
            ]
        )
        models = build_models(model_count, [])
        labels = legacy_ensemble_logits_tta_hflip(models, images).argmax(1).cpu()
        labels[-1] = (labels[-1] + 1) % 2
        return images, labels

    def test_holdout_counts_ordering_and_final_incomplete_batch_match_legacy(self):
        for model_count in (1, 2, 3):
            with self.subTest(model_count=model_count):
                images, labels = self.make_dataset(model_count)
                legacy_trace, actual_trace = [], []
                legacy_models = build_models(model_count, legacy_trace)
                actual_models = build_models(model_count, actual_trace)
                legacy_loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
                actual_loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

                rng_start = torch.get_rng_state().clone()
                expected = legacy_run_ensemble_holdout(legacy_models, legacy_loader, "cpu")
                legacy_rng_end = torch.get_rng_state().clone()
                torch.set_rng_state(rng_start)
                with patch("src.inference.ensemble.tqdm", side_effect=lambda iterable, ncols: iterable):
                    actual = run_ensemble_holdout(actual_models, actual_loader, "cpu")
                self.assertTrue(torch.equal(legacy_rng_end, torch.get_rng_state()))

                self.assertIs(type(actual), tuple)
                self.assertEqual(actual, expected)
                self.assertEqual(actual, (4, 5))
                self.assertTrue(all(isinstance(value, int) for value in actual))
                assert_trace_equal(self, actual_trace, legacy_trace)
                for model in actual_models:
                    model_sizes = [images.shape[0] for name, images in actual_trace if name == model.name]
                    self.assertEqual(model_sizes, [2, 2, 2, 2, 1, 1])
                    self.assertFalse(model.training)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this parity check")
    def test_synthetic_cuda_holdout_preserves_counts_order_and_no_grad(self):
        device = torch.device("cuda")
        images, labels = self.make_dataset(3)
        legacy_trace, actual_trace = [], []
        legacy_models = build_models(3, legacy_trace, device)
        actual_models = build_models(3, actual_trace, device)
        legacy_loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
        actual_loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

        cuda_rng = torch.cuda.get_rng_state(device).clone()
        expected = legacy_run_ensemble_holdout(legacy_models, legacy_loader, device)
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        with patch("src.inference.ensemble.tqdm", side_effect=lambda iterable, ncols: iterable):
            actual = run_ensemble_holdout(actual_models, actual_loader, device)
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))

        self.assertEqual(actual, expected)
        assert_trace_equal(self, actual_trace, legacy_trace)
        self.assertTrue(all(images.device.type == "cuda" for _, images in actual_trace))
        self.assertTrue(all(not images.requires_grad for _, images in actual_trace))


if __name__ == "__main__":
    unittest.main()
