import unittest

import torch
import torch.nn as nn

from src.inference.ensemble import ensemble_logits, ensemble_logits_tta_hflip


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


def assert_trace_equal(test_case, actual, expected):
    test_case.assertEqual([name for name, _ in actual], [name for name, _ in expected])
    for (_, actual_images), (_, expected_images) in zip(actual, expected):
        torch.testing.assert_close(actual_images, expected_images, rtol=0, atol=0)


class EnsembleTtaParityTest(unittest.TestCase):
    def test_ordered_raw_logit_ensemble_matches_for_one_two_and_three_models(self):
        images = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],
                [[[4.0, 3.0, 2.0, 1.0]]],
            ]
        )
        for model_count in (1, 2, 3):
            with self.subTest(model_count=model_count):
                legacy_trace, actual_trace = [], []
                legacy_models = build_models(model_count, legacy_trace)
                actual_models = build_models(model_count, actual_trace)
                rng_before = torch.get_rng_state().clone()
                expected = legacy_ensemble_logits(legacy_models, images)
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                rng_before = torch.get_rng_state().clone()
                actual = ensemble_logits(actual_models, images)
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert_trace_equal(self, actual_trace, legacy_trace)
                self.assertEqual([name for name, _ in actual_trace], [f"model-{index}" for index in range(model_count)])
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.device, expected.device)
                self.assertFalse(actual.requires_grad)

    def test_horizontal_flip_tta_preserves_calls_flip_dimension_and_formula(self):
        images = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],
                [[[5.0, 6.0, 7.0, 8.0]]],
            ],
            requires_grad=True,
        )
        legacy_trace, actual_trace = [], []
        legacy_models = build_models(2, legacy_trace)
        actual_models = build_models(2, actual_trace)
        initial_states = [model.state_dict() for model in actual_models]

        rng_before = torch.get_rng_state().clone()
        expected = legacy_ensemble_logits_tta_hflip(legacy_models, images)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        rng_before = torch.get_rng_state().clone()
        actual = ensemble_logits_tta_hflip(actual_models, images)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_trace_equal(self, actual_trace, legacy_trace)
        self.assertEqual([name for name, _ in actual_trace], ["model-0", "model-1", "model-0", "model-1"])
        flipped = torch.flip(images, dims=[3])
        for index, (_, observed) in enumerate(actual_trace):
            reference = images if index < 2 else flipped
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)
        for model, initial_state in zip(actual_models, initial_states):
            self.assertFalse(model.training)
            self.assertEqual(list(model.state_dict().keys()), list(initial_state.keys()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, initial_state[key], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this parity check")
    def test_synthetic_cuda_tta_preserves_device_dtype_and_no_grad(self):
        device = torch.device("cuda")
        images = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], device=device)
        legacy_trace, actual_trace = [], []
        legacy_models = build_models(3, legacy_trace, device)
        actual_models = build_models(3, actual_trace, device)
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        expected = legacy_ensemble_logits_tta_hflip(legacy_models, images)
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        actual = ensemble_logits_tta_hflip(actual_models, images)
        self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_trace_equal(self, actual_trace, legacy_trace)
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertFalse(actual.requires_grad)


if __name__ == "__main__":
    unittest.main()
