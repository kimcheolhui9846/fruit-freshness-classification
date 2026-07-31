import unittest

import numpy as np
import torch

from src.transforms.mixup import mixup_data


def _legacy_mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MixupParityTest(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        self.targets = torch.tensor([0, 1, 2, 3])

    def test_non_positive_alpha_preserves_not_mixed_behavior(self):
        mixed_x, y_a, y_b, lam = mixup_data(self.inputs, self.targets, alpha=0)

        self.assertIs(mixed_x, self.inputs)
        self.assertIs(y_a, self.targets)
        self.assertIs(y_b, self.targets)
        self.assertEqual(lam, 1.0)

    def test_seeded_positive_alpha_matches_notebook_implementation(self):
        np.random.seed(321)
        torch.manual_seed(654)
        expected = _legacy_mixup_data(self.inputs, self.targets, alpha=0.8)
        np.random.seed(321)
        torch.manual_seed(654)
        observed = mixup_data(self.inputs, self.targets, alpha=0.8)

        expected_x, expected_y_a, expected_y_b, expected_lam = expected
        observed_x, observed_y_a, observed_y_b, observed_lam = observed
        self.assertIsInstance(observed_lam, type(expected_lam))
        self.assertEqual(observed_lam, expected_lam)
        self.assertEqual(observed_x.device, self.inputs.device)
        self.assertEqual(observed_x.dtype, self.inputs.dtype)
        torch.testing.assert_close(observed_x, expected_x)
        torch.testing.assert_close(observed_y_a, expected_y_a)
        torch.testing.assert_close(observed_y_b, expected_y_b)
