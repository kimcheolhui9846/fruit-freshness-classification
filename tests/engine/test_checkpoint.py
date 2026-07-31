import copy
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from src.engine.checkpoint import load_model_state, save_model_state
from src.engine.ema import ModelEma


def assert_state_dict_equal(test_case, actual, expected):
    test_case.assertEqual(list(actual.keys()), list(expected.keys()))
    for key in actual:
        test_case.assertEqual(actual[key].shape, expected[key].shape, key)
        test_case.assertEqual(actual[key].dtype, expected[key].dtype, key)
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


def build_model():
    return nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
        nn.GELU(),
        nn.Linear(4, 3),
    )


def legacy_save(model, path):
    torch.save(model.state_dict(), path)


def legacy_load(model, path, map_location):
    return model.load_state_dict(torch.load(path, map_location=map_location))


class CheckpointParityTest(unittest.TestCase):
    def test_legacy_save_and_extracted_load_round_trip(self):
        torch.manual_seed(31)
        source = build_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model_fold1.pt"
            legacy_save(source, path)
            saved = torch.load(path, map_location="cpu")
            self.assertEqual(list(saved.keys()), list(source.state_dict().keys()))

            torch.manual_seed(37)
            target = build_model()
            result = load_model_state(target, path, map_location="cpu")
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            assert_state_dict_equal(self, target.state_dict(), source.state_dict())

    def test_extracted_save_and_legacy_load_round_trip(self):
        torch.manual_seed(41)
        source = build_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_model_weights.pt"
            save_result = save_model_state(source, path)
            self.assertIsNone(save_result)
            self.assertTrue(path.exists())

            torch.manual_seed(43)
            target = build_model()
            result = legacy_load(target, path, map_location="cpu")
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            assert_state_dict_equal(self, target.state_dict(), source.state_dict())

    def test_ema_module_state_uses_the_same_active_format(self):
        torch.manual_seed(47)
        source = build_model()
        ema = ModelEma(source, decay=0.999, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model_fold2.pt"
            save_model_state(ema.module, path)
            target = build_model()
            result = legacy_load(target, path, map_location="cpu")
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            assert_state_dict_equal(self, target.state_dict(), ema.module.state_dict())

    def test_default_strict_loading_matches_the_legacy_call(self):
        source = build_model()
        incomplete_state = OrderedDict(list(source.state_dict().items())[:1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.pt"
            torch.save(incomplete_state, path)
            legacy_target = build_model()
            extracted_target = copy.deepcopy(legacy_target)
            with self.assertRaises(RuntimeError):
                legacy_load(legacy_target, path, map_location="cpu")
            with self.assertRaises(RuntimeError):
                load_model_state(extracted_target, path, map_location="cpu")


if __name__ == "__main__":
    unittest.main()
