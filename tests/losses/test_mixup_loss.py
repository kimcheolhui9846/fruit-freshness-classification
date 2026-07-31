import unittest

import numpy as np
import torch
import torch.nn as nn

from src.losses.focal import FocalLoss
from src.losses.mixup import mixup_criterion


def legacy_mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def loss_and_gradient(combine, criterion, logits, y_a, y_b, lam):
    independent_logits = logits.detach().clone().requires_grad_(True)
    loss = combine(criterion, independent_logits, y_a, y_b, lam)
    loss.backward()
    return loss.detach(), independent_logits.grad.detach()


class MixupCriterionParityTest(unittest.TestCase):
    def test_cross_entropy_value_and_gradient_match(self):
        logits = torch.tensor([[0.2, -0.1, 0.8], [1.0, 0.3, -0.4]], dtype=torch.float32)
        y_a = torch.tensor([2, 0])
        y_b = torch.tensor([1, 2])
        lam = np.float64(0.35)
        legacy_loss, legacy_gradient = loss_and_gradient(
            legacy_mixup_criterion,
            nn.CrossEntropyLoss(label_smoothing=0.01),
            logits,
            y_a,
            y_b,
            lam,
        )
        extracted_loss, extracted_gradient = loss_and_gradient(
            mixup_criterion,
            nn.CrossEntropyLoss(label_smoothing=0.01),
            logits,
            y_a,
            y_b,
            lam,
        )
        torch.testing.assert_close(legacy_loss, extracted_loss, rtol=0, atol=0)
        torch.testing.assert_close(legacy_gradient, extracted_gradient, rtol=0, atol=0)

    def test_focal_value_and_gradient_match(self):
        logits = torch.tensor([[0.6, 0.1, -0.2], [-0.5, 0.3, 1.1]], dtype=torch.float32)
        y_a = torch.tensor([0, 2])
        y_b = torch.tensor([2, 1])
        lam = 0.8
        legacy_loss, legacy_gradient = loss_and_gradient(
            legacy_mixup_criterion,
            FocalLoss(alpha=[0.2, 0.7, 1.1], gamma=2.0),
            logits,
            y_a,
            y_b,
            lam,
        )
        extracted_loss, extracted_gradient = loss_and_gradient(
            mixup_criterion,
            FocalLoss(alpha=[0.2, 0.7, 1.1], gamma=2.0),
            logits,
            y_a,
            y_b,
            lam,
        )
        torch.testing.assert_close(legacy_loss, extracted_loss, rtol=0, atol=0)
        torch.testing.assert_close(legacy_gradient, extracted_gradient, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(extracted_gradient).all())


if __name__ == "__main__":
    unittest.main()
