import copy
import unittest

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.engine.optimization import build_optimizer, build_scheduler
from src.models.factory import build_cmt_classifier


LR_CNN = 5e-5
LR_TRANS = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 120


class CmtLayoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 3)
        self.stage1 = nn.Linear(3, 3)
        self.stage2 = nn.Linear(3, 3)
        self.stage3 = nn.Linear(3, 3)
        self.to_embed1 = nn.Linear(3, 3)
        self.trans1 = nn.Linear(3, 3)
        self.down_tokens = nn.Linear(3, 3)
        self.trans2 = nn.Linear(3, 3)
        self.head_norm = nn.LayerNorm(3)
        self.fc = nn.Linear(3, 2)


def legacy_build_optimizer(model):
    cnn_modules = [model.stem, model.stage1, model.stage2, model.stage3]
    trans_modules = [
        model.to_embed1,
        model.trans1,
        model.down_tokens,
        model.trans2,
        model.head_norm,
        model.fc,
    ]
    cnn_params = []
    trans_params = []
    for module in cnn_modules:
        cnn_params += list(module.parameters())
    for module in trans_modules:
        trans_params += list(module.parameters())
    return torch.optim.AdamW(
        [
            {"params": cnn_params, "lr": LR_CNN},
            {"params": trans_params, "lr": LR_TRANS},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def parameter_group_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]


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
        for actual_value, expected_value in zip(actual, expected):
            assert_nested_equal(test_case, actual_value, expected_value)
    else:
        test_case.assertEqual(actual, expected)


class OptimizationParityTest(unittest.TestCase):
    def make_optimizer_pair(self):
        torch.manual_seed(53)
        legacy_model = CmtLayoutModel()
        torch.manual_seed(53)
        extracted_model = CmtLayoutModel()
        legacy_optimizer = legacy_build_optimizer(legacy_model)
        extracted_optimizer = build_optimizer(
            extracted_model,
            lr_cnn=LR_CNN,
            lr_trans=LR_TRANS,
            weight_decay=WEIGHT_DECAY,
        )
        return legacy_model, extracted_model, legacy_optimizer, extracted_optimizer

    def test_optimizer_class_groups_and_parameter_order_match(self):
        legacy_model, extracted_model, legacy_optimizer, extracted_optimizer = self.make_optimizer_pair()
        self.assertIsInstance(extracted_optimizer, torch.optim.AdamW)
        self.assertEqual(type(legacy_optimizer), type(extracted_optimizer))
        self.assertEqual(legacy_optimizer.defaults, extracted_optimizer.defaults)
        self.assertEqual(
            parameter_group_names(legacy_model, legacy_optimizer),
            parameter_group_names(extracted_model, extracted_optimizer),
        )
        self.assertEqual(len(legacy_optimizer.param_groups), len(extracted_optimizer.param_groups))
        for legacy_group, extracted_group in zip(legacy_optimizer.param_groups, extracted_optimizer.param_groups):
            self.assertEqual(set(legacy_group.keys()), set(extracted_group.keys()))
            for key in legacy_group:
                if key != "params":
                    self.assertEqual(legacy_group[key], extracted_group[key])
        assert_nested_equal(self, legacy_optimizer.state_dict(), extracted_optimizer.state_dict())

    def test_optimizer_step_and_serialized_state_match(self):
        legacy_model, extracted_model, legacy_optimizer, extracted_optimizer = self.make_optimizer_pair()
        legacy_loss = sum(parameter.square().sum() for parameter in legacy_model.parameters())
        extracted_loss = sum(parameter.square().sum() for parameter in extracted_model.parameters())
        legacy_loss.backward()
        extracted_loss.backward()
        legacy_optimizer.step()
        extracted_optimizer.step()
        for (_, legacy_parameter), (_, extracted_parameter) in zip(
            legacy_model.named_parameters(), extracted_model.named_parameters()
        ):
            torch.testing.assert_close(legacy_parameter, extracted_parameter, rtol=0, atol=0)
        assert_nested_equal(self, legacy_optimizer.state_dict(), extracted_optimizer.state_dict())

    def test_cmt_parameter_group_order_matches_the_notebook_boundary(self):
        torch.manual_seed(59)
        legacy_model = build_cmt_classifier(13)
        torch.manual_seed(59)
        extracted_model = build_cmt_classifier(13)
        legacy_optimizer = legacy_build_optimizer(legacy_model)
        extracted_optimizer = build_optimizer(
            extracted_model,
            lr_cnn=LR_CNN,
            lr_trans=LR_TRANS,
            weight_decay=WEIGHT_DECAY,
        )
        self.assertEqual(
            parameter_group_names(legacy_model, legacy_optimizer),
            parameter_group_names(extracted_model, extracted_optimizer),
        )

    def test_scheduler_sequence_and_state_match(self):
        legacy_model, extracted_model, legacy_optimizer, extracted_optimizer = self.make_optimizer_pair()
        legacy_scheduler = CosineAnnealingLR(legacy_optimizer, T_max=EPOCHS)
        extracted_scheduler = build_scheduler(extracted_optimizer, t_max=EPOCHS)
        self.assertEqual(type(legacy_scheduler), type(extracted_scheduler))
        self.assertEqual(legacy_scheduler.state_dict(), extracted_scheduler.state_dict())
        legacy_lrs = []
        extracted_lrs = []
        for _ in range(6):
            legacy_optimizer.step()
            extracted_optimizer.step()
            legacy_scheduler.step()
            extracted_scheduler.step()
            legacy_lrs.append([group["lr"] for group in legacy_optimizer.param_groups])
            extracted_lrs.append([group["lr"] for group in extracted_optimizer.param_groups])
        self.assertEqual(legacy_lrs, extracted_lrs)
        self.assertEqual(legacy_scheduler.state_dict(), extracted_scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
