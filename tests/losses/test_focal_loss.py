import inspect
import json
import subprocess
import unittest

import torch

from src.losses.focal import FocalLoss, build_class_balanced_alpha


LEGACY_NOTEBOOK_COMMIT = "97f3a37"


def load_legacy_focal_loss():
    notebook_text = subprocess.check_output(
        ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:deep3.ipynb"],
        text=True,
        encoding="utf-8",
    )
    cell_source = "".join(json.loads(notebook_text)["cells"][3]["source"])
    focal_start = cell_source.index("class FocalLoss")
    focal_end = cell_source.index("# ------------------------------------------------------------", focal_start)
    namespace = {}
    exec(
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        + cell_source[focal_start:focal_end],
        namespace,
    )
    return namespace["FocalLoss"]


def legacy_class_balanced_alpha(class_counts, beta, num_classes):
    effective_num = [1.0 - (beta ** count) for count in class_counts]
    raw_alpha = torch.tensor(
        [(1.0 - beta) / (value if value > 0 else 1e-8) for value in effective_num],
        dtype=torch.float32,
    )
    return (raw_alpha / raw_alpha.sum()) * num_classes


def loss_and_gradient(loss_class, logits, targets, **kwargs):
    independent_logits = logits.detach().clone().requires_grad_(True)
    loss = loss_class(**kwargs)(independent_logits, targets)
    loss.sum().backward()
    return loss.detach(), independent_logits.grad.detach()


class FocalLossParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_class = load_legacy_focal_loss()

    def assert_loss_parity(self, logits, targets, **kwargs):
        legacy = self.legacy_class(**kwargs)
        extracted = FocalLoss(**kwargs)
        legacy_output = legacy(logits.detach().clone(), targets)
        extracted_output = extracted(logits.detach().clone(), targets)
        self.assertIs(type(legacy_output), type(extracted_output))
        self.assertEqual(legacy_output.shape, extracted_output.shape)
        self.assertEqual(legacy_output.dtype, extracted_output.dtype)
        self.assertEqual(legacy_output.device, extracted_output.device)
        torch.testing.assert_close(legacy_output, extracted_output, rtol=0, atol=0)

    def test_constructor_and_module_state_match(self):
        alpha = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
        legacy = self.legacy_class(alpha=alpha, gamma=1.5, reduction="sum", eps=1e-6)
        extracted = FocalLoss(alpha=alpha, gamma=1.5, reduction="sum", eps=1e-6)
        self.assertEqual(
            inspect.signature(self.legacy_class.__init__),
            inspect.signature(FocalLoss.__init__),
        )
        self.assertEqual(legacy.gamma, extracted.gamma)
        self.assertEqual(legacy.reduction, extracted.reduction)
        self.assertEqual(legacy.eps, extracted.eps)
        self.assertEqual(list(legacy.state_dict().keys()), list(extracted.state_dict().keys()))
        self.assertEqual(
            [name for name, _ in legacy.named_parameters()],
            [name for name, _ in extracted.named_parameters()],
        )
        self.assertEqual(
            [name for name, _ in legacy.named_buffers()],
            [name for name, _ in extracted.named_buffers()],
        )
        torch.testing.assert_close(legacy.alpha, extracted.alpha, rtol=0, atol=0)

    def test_loss_values_cover_active_reductions_and_target_cases(self):
        cases = [
            (
                torch.tensor([[1.5, -0.2, 0.4], [-0.4, 0.2, 2.1], [0.0, 0.0, 0.0]]),
                torch.tensor([0, 2, 1]),
                [0.5, 1.0, 1.5],
                2.0,
            ),
            (torch.tensor([[9.0, -4.0, -4.0]]), torch.tensor([0]), None, 2.0),
            (torch.tensor([[9.0, -4.0, -4.0]]), torch.tensor([2]), None, 2.0),
            (torch.zeros(2, 3), torch.tensor([1, 1]), [1.0, 2.0, 3.0], 1.5),
            (torch.tensor([[0.3, 0.2, -0.1]]), torch.tensor([1]), [0.2, 0.4, 0.6], 2.0),
        ]
        for logits, targets, alpha, gamma in cases:
            for reduction in ("mean", "sum", "none"):
                with self.subTest(reduction=reduction, alpha=alpha, gamma=gamma):
                    self.assert_loss_parity(
                        logits,
                        targets,
                        alpha=alpha,
                        gamma=gamma,
                        reduction=reduction,
                    )

    def test_default_and_alpha_gradient_parity(self):
        logits = torch.tensor(
            [[0.2, -0.5, 1.1], [1.2, 0.4, -0.8]],
            dtype=torch.float32,
        )
        targets = torch.tensor([2, 0])
        for kwargs in (
            {},
            {"alpha": [0.3, 0.7, 1.4], "gamma": 1.5},
        ):
            with self.subTest(kwargs=kwargs):
                legacy_loss, legacy_gradient = loss_and_gradient(
                    self.legacy_class,
                    logits,
                    targets,
                    **kwargs,
                )
                extracted_loss, extracted_gradient = loss_and_gradient(
                    FocalLoss,
                    logits,
                    targets,
                    **kwargs,
                )
                torch.testing.assert_close(legacy_loss, extracted_loss, rtol=0, atol=0)
                self.assertEqual(legacy_gradient.dtype, extracted_gradient.dtype)
                self.assertEqual(legacy_gradient.device, extracted_gradient.device)
                torch.testing.assert_close(legacy_gradient, extracted_gradient, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(extracted_gradient).all())

    def test_class_balanced_alpha_matches_the_notebook_formula(self):
        for counts in ([5, 3, 2], [10, 0, 1, 4]):
            with self.subTest(counts=counts):
                legacy = legacy_class_balanced_alpha(counts, beta=0.999, num_classes=len(counts))
                extracted = build_class_balanced_alpha(counts, beta=0.999, num_classes=len(counts))
                self.assertEqual(legacy.dtype, extracted.dtype)
                self.assertEqual(legacy.device, extracted.device)
                torch.testing.assert_close(legacy, extracted, rtol=0, atol=0)

    def test_error_behavior_matches_for_invalid_indices_and_batch_shapes(self):
        cases = [
            (torch.zeros(2, 3), torch.tensor([0, 3]), {}),
            (torch.zeros(2, 3), torch.tensor([0]), {}),
            (torch.zeros(1, 3), torch.tensor([2]), {"alpha": [1.0, 1.0]}),
        ]
        for logits, targets, kwargs in cases:
            with self.subTest(shape=tuple(logits.shape), kwargs=kwargs):
                try:
                    legacy_output = self.legacy_class(**kwargs)(logits, targets)
                    legacy_error = None
                except Exception as error:
                    legacy_output = None
                    legacy_error = error
                try:
                    extracted_output = FocalLoss(**kwargs)(logits, targets)
                    extracted_error = None
                except Exception as error:
                    extracted_output = None
                    extracted_error = error
                self.assertEqual(type(legacy_error), type(extracted_error))
                if legacy_error is None:
                    self.assertEqual(legacy_output.shape, extracted_output.shape)
                    self.assertEqual(legacy_output.dtype, extracted_output.dtype)
                    torch.testing.assert_close(legacy_output, extracted_output, rtol=0, atol=0)

    def test_cpu_and_cuda_device_dtype_parity(self):
        logits = torch.tensor([[0.1, 0.8, -0.4], [0.5, -0.3, 1.2]], dtype=torch.float32)
        targets = torch.tensor([1, 2])
        self.assert_loss_parity(logits, targets, alpha=[0.2, 0.6, 1.0])

        if torch.cuda.is_available():
            device = torch.device("cuda")
            legacy = self.legacy_class(alpha=[0.2, 0.6, 1.0]).to(device)
            extracted = FocalLoss(alpha=[0.2, 0.6, 1.0]).to(device)
            cuda_targets = targets.to(device)
            legacy_logits = logits.to(device).detach().clone().requires_grad_(True)
            extracted_logits = logits.to(device).detach().clone().requires_grad_(True)
            legacy_output = legacy(legacy_logits, cuda_targets)
            extracted_output = extracted(extracted_logits, cuda_targets)
            legacy_output.backward()
            extracted_output.backward()
            self.assertEqual(legacy_output.device.type, "cuda")
            self.assertEqual(extracted_output.dtype, torch.float32)
            torch.testing.assert_close(legacy_output, extracted_output, rtol=0, atol=0)
            torch.testing.assert_close(legacy_logits.grad, extracted_logits.grad, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(extracted_logits.grad).all())

    def test_cmt_output_contract_smoke(self):
        logits = torch.randn(2, 13, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0, 12])
        loss = FocalLoss()(logits, targets)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertEqual(logits.grad.shape, logits.shape)


if __name__ == "__main__":
    unittest.main()
