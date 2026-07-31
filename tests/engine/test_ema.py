import copy
import inspect
import json
import subprocess
import unittest

import torch
import torch.nn as nn

from src.engine.ema import ModelEma
from src.models.factory import build_cmt_classifier


LEGACY_NOTEBOOK_COMMIT = "c2eebb1"


def load_legacy_model_ema():
    notebook_text = subprocess.check_output(
        ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:deep3.ipynb"],
        text=True,
        encoding="utf-8",
    )
    source = "".join(json.loads(notebook_text)["cells"][2]["source"])
    namespace = {}
    exec(source, namespace)
    return namespace["ModelEma"]


def assert_state_dict_equal(test_case, actual, expected):
    test_case.assertEqual(list(actual.keys()), list(expected.keys()))
    for key in actual:
        test_case.assertEqual(actual[key].shape, expected[key].shape, key)
        test_case.assertEqual(actual[key].dtype, expected[key].dtype, key)
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.batch_norm = nn.BatchNorm1d(4)
        self.register_buffer("integer_counter", torch.tensor(7, dtype=torch.long))

    def forward(self, x):
        return self.batch_norm(self.linear(x))


class ModelEmaParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_class = load_legacy_model_ema()

    def make_pair(self, device="cpu"):
        torch.manual_seed(1729)
        legacy_model = StatefulModel().to(device)
        torch.manual_seed(1729)
        extracted_model = StatefulModel().to(device)
        legacy_ema = self.legacy_class(legacy_model, decay=0.999, device=device)
        extracted_ema = ModelEma(extracted_model, decay=0.999, device=device)
        return legacy_model, extracted_model, legacy_ema, extracted_ema

    @staticmethod
    def mutate_source(model):
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters(), 1):
                parameter.add_(0.125 * index)
            model.batch_norm.running_mean.add_(0.25)
            model.batch_norm.running_var.mul_(1.5)
            model.batch_norm.num_batches_tracked.add_(2)
            model.integer_counter.add_(3)

    def test_interface_and_initial_state_match(self):
        legacy_model, extracted_model, legacy_ema, extracted_ema = self.make_pair()
        self.assertEqual(
            inspect.signature(self.legacy_class.__init__),
            inspect.signature(ModelEma.__init__),
        )
        self.assertEqual(list(legacy_ema._modules.keys()), list(extracted_ema._modules.keys()))
        self.assertEqual(list(legacy_ema._modules.keys()), ["module"])
        self.assertFalse(legacy_ema.module.training)
        self.assertFalse(extracted_ema.module.training)
        self.assertEqual(legacy_ema.decay, extracted_ema.decay)
        self.assertEqual(legacy_ema.device, extracted_ema.device)
        assert_state_dict_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())
        assert_state_dict_equal(self, legacy_ema.module.state_dict(), legacy_model.state_dict())
        assert_state_dict_equal(self, extracted_ema.module.state_dict(), extracted_model.state_dict())
        self.assertEqual(
            [name for name, _ in legacy_ema.named_parameters()],
            [name for name, _ in extracted_ema.named_parameters()],
        )
        self.assertEqual(
            [tensor.requires_grad for _, tensor in legacy_ema.named_parameters()],
            [tensor.requires_grad for _, tensor in extracted_ema.named_parameters()],
        )
        self.assertEqual(
            [name for name, _ in legacy_ema.named_buffers()],
            [name for name, _ in extracted_ema.named_buffers()],
        )

    def test_update_and_set_match_without_mutating_source(self):
        legacy_model, extracted_model, legacy_ema, extracted_ema = self.make_pair()
        self.mutate_source(legacy_model)
        self.mutate_source(extracted_model)
        legacy_source_before = copy.deepcopy(legacy_model.state_dict())
        extracted_source_before = copy.deepcopy(extracted_model.state_dict())

        legacy_ema.update(legacy_model)
        extracted_ema.update(extracted_model)
        assert_state_dict_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())
        assert_state_dict_equal(self, legacy_model.state_dict(), legacy_source_before)
        assert_state_dict_equal(self, extracted_model.state_dict(), extracted_source_before)

        self.mutate_source(legacy_model)
        self.mutate_source(extracted_model)
        legacy_ema.update(legacy_model)
        extracted_ema.update(extracted_model)
        assert_state_dict_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())

        legacy_ema.set(legacy_model)
        extracted_ema.set(extracted_model)
        assert_state_dict_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())
        for tensor in extracted_ema.module.state_dict().values():
            self.assertIsNone(tensor.grad_fn)
        for parameter in extracted_ema.module.parameters():
            self.assertIsNone(parameter.grad)

    def test_cmt_module_path_and_state_order_match(self):
        torch.manual_seed(91)
        legacy_model = build_cmt_classifier(13)
        torch.manual_seed(91)
        extracted_model = build_cmt_classifier(13)
        legacy_ema = self.legacy_class(legacy_model, decay=0.999, device="cpu")
        extracted_ema = ModelEma(extracted_model, decay=0.999, device="cpu")
        self.assertEqual(
            list(legacy_ema.module.state_dict().keys()),
            list(extracted_ema.module.state_dict().keys()),
        )
        self.assertEqual(
            list(legacy_ema.state_dict().keys()),
            list(extracted_ema.state_dict().keys()),
        )
        assert_state_dict_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())

    def test_cuda_update_parity_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        legacy_model, extracted_model, legacy_ema, extracted_ema = self.make_pair("cuda")
        self.mutate_source(legacy_model)
        self.mutate_source(extracted_model)
        legacy_ema.update(legacy_model)
        extracted_ema.update(extracted_model)
        assert_state_dict_equal(self, legacy_ema.state_dict(), extracted_ema.state_dict())
        self.assertTrue(all(tensor.device.type == "cuda" for tensor in extracted_ema.module.state_dict().values()))


if __name__ == "__main__":
    unittest.main()
