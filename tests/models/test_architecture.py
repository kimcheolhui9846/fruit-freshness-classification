import inspect
import json
import subprocess
import unittest

import torch

from src.models.cmt_classifier import CMTClassifier
from src.models.factory import build_cmt_classifier


LEGACY_NOTEBOOK_COMMIT = "7eb6e2a"
NUM_CLASSES = 13


def load_legacy_cmt_classifier():
    notebook_text = subprocess.check_output(
        ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:deep3.ipynb"],
        text=True,
        encoding="utf-8",
    )
    cell_source = "".join(json.loads(notebook_text)["cells"][2]["source"])
    ema_start = cell_source.index("from copy import deepcopy")
    classifier_start = cell_source.index("class CMTClassifier")
    legacy_source = cell_source[:ema_start] + cell_source[classifier_start:]
    namespace = {}
    exec(legacy_source, namespace)
    return namespace["CMTClassifier"]


class CMTArchitectureParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_class = load_legacy_cmt_classifier()
        torch.manual_seed(1729)
        cls.initial_rng_state = torch.get_rng_state().clone()
        cls.legacy_model = cls.legacy_class(NUM_CLASSES)
        cls.legacy_rng_state = torch.get_rng_state().clone()

        torch.set_rng_state(cls.initial_rng_state)
        cls.extracted_model = build_cmt_classifier(NUM_CLASSES)
        cls.extracted_rng_state = torch.get_rng_state().clone()

    def test_constructor_signature_matches(self):
        self.assertEqual(
            inspect.signature(self.legacy_class.__init__),
            inspect.signature(CMTClassifier.__init__),
        )

    def test_factory_returns_the_active_classifier_type(self):
        self.assertIsInstance(self.extracted_model, CMTClassifier)
        self.assertEqual(self.extracted_model.num_classes, NUM_CLASSES)

    def test_module_structure_and_registration_order_match(self):
        legacy_modules = [
            (name, type(module).__name__)
            for name, module in self.legacy_model.named_modules()
        ]
        extracted_modules = [
            (name, type(module).__name__)
            for name, module in self.extracted_model.named_modules()
        ]
        self.assertEqual(legacy_modules, extracted_modules)
        self.assertEqual(
            list(self.legacy_model._modules.keys()),
            list(self.extracted_model._modules.keys()),
        )

    def test_state_parameters_and_buffers_match_exactly(self):
        legacy_state = self.legacy_model.state_dict()
        extracted_state = self.extracted_model.state_dict()
        self.assertEqual(list(legacy_state.keys()), list(extracted_state.keys()))
        for key in legacy_state:
            self.assertEqual(legacy_state[key].shape, extracted_state[key].shape)
            self.assertEqual(legacy_state[key].dtype, extracted_state[key].dtype)
            torch.testing.assert_close(legacy_state[key], extracted_state[key], rtol=0, atol=0)

        legacy_parameters = list(self.legacy_model.named_parameters())
        extracted_parameters = list(self.extracted_model.named_parameters())
        self.assertEqual(
            [name for name, _ in legacy_parameters],
            [name for name, _ in extracted_parameters],
        )
        self.assertEqual(
            [parameter.requires_grad for _, parameter in legacy_parameters],
            [parameter.requires_grad for _, parameter in extracted_parameters],
        )
        self.assertEqual(
            sum(parameter.numel() for _, parameter in legacy_parameters),
            sum(parameter.numel() for _, parameter in extracted_parameters),
        )
        self.assertEqual(
            sum(parameter.numel() for _, parameter in legacy_parameters if parameter.requires_grad),
            sum(parameter.numel() for _, parameter in extracted_parameters if parameter.requires_grad),
        )

        legacy_buffers = list(self.legacy_model.named_buffers())
        extracted_buffers = list(self.extracted_model.named_buffers())
        self.assertEqual(
            [name for name, _ in legacy_buffers],
            [name for name, _ in extracted_buffers],
        )
        for (_, legacy_buffer), (_, extracted_buffer) in zip(legacy_buffers, extracted_buffers):
            self.assertEqual(legacy_buffer.shape, extracted_buffer.shape)
            self.assertEqual(legacy_buffer.dtype, extracted_buffer.dtype)

    def test_initialization_and_cpu_rng_consumption_match(self):
        self.assertTrue(torch.equal(self.legacy_rng_state, self.extracted_rng_state))

    def test_forward_contract_matches_in_evaluation_and_training_modes(self):
        torch.manual_seed(31415)
        inputs = torch.randn(1, 3, 224, 224)

        self.legacy_model.eval()
        self.extracted_model.eval()
        with torch.inference_mode():
            legacy_output = self.legacy_model(inputs)
            extracted_output = self.extracted_model(inputs)
        self.assertIs(type(legacy_output), type(extracted_output))
        self.assertEqual(legacy_output.shape, extracted_output.shape)
        self.assertEqual(legacy_output.dtype, extracted_output.dtype)
        self.assertEqual(legacy_output.device, extracted_output.device)
        torch.testing.assert_close(legacy_output, extracted_output, rtol=0, atol=0)

        self.legacy_model.train()
        self.extracted_model.train()
        torch.manual_seed(27182)
        legacy_train_output = self.legacy_model(inputs)
        torch.manual_seed(27182)
        extracted_train_output = self.extracted_model(inputs)
        torch.testing.assert_close(legacy_train_output, extracted_train_output, rtol=0, atol=0)

    def test_strict_state_dict_loading_is_bidirectionally_compatible(self):
        extracted_result = self.extracted_model.load_state_dict(
            self.legacy_model.state_dict(),
            strict=True,
        )
        legacy_result = self.legacy_model.load_state_dict(
            self.extracted_model.state_dict(),
            strict=True,
        )
        self.assertEqual(extracted_result.missing_keys, [])
        self.assertEqual(extracted_result.unexpected_keys, [])
        self.assertEqual(legacy_result.missing_keys, [])
        self.assertEqual(legacy_result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
