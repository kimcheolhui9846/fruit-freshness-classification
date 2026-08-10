import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from src.engine.checkpoint import load_model_state, save_model_state
from src.inference.loading import load_fold_model, load_fold_models
from src.utils.paths import build_fold_checkpoint_path


def build_tiny_classifier(num_classes):
    return nn.Sequential(nn.Flatten(), nn.Linear(4, num_classes))


def assert_state_dict_equal(test_case, actual, expected):
    test_case.assertEqual(list(actual.keys()), list(expected.keys()))
    for key in actual:
        test_case.assertEqual(actual[key].dtype, expected[key].dtype, key)
        test_case.assertEqual(actual[key].shape, expected[key].shape, key)
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


def legacy_load_fold_models(num_folds, num_classes, device, ckpt_dir, factory):
    """Test-only reference for the pre-Phase 4.9 notebook loader."""
    models = []
    for fold in range(1, num_folds + 1):
        model = factory(num_classes).to(device)
        path = build_fold_checkpoint_path(ckpt_dir, fold)
        load_model_state(model, path, map_location=device)
        model.eval()
        models.append(model)
    return models


class FoldModelLoadingParityTest(unittest.TestCase):
    def create_checkpoints(self, directory, num_folds, num_classes):
        source_models = []
        for fold in range(1, num_folds + 1):
            torch.manual_seed(100 + fold)
            model = build_tiny_classifier(num_classes)
            path = build_fold_checkpoint_path(directory, fold)
            save_model_state(model, path)
            source_models.append(model)
        return source_models

    def test_checkpoint_paths_states_order_and_eval_mode_match_legacy(self):
        num_folds, num_classes, device = 3, 3, "cpu"
        with tempfile.TemporaryDirectory() as directory:
            source_models = self.create_checkpoints(directory, num_folds, num_classes)
            factory_calls = []
            with patch(
                "src.inference.loading.build_cmt_classifier",
                side_effect=lambda classes: factory_calls.append(classes)
                or build_tiny_classifier(classes),
            ), patch(
                "src.inference.loading.build_fold_checkpoint_path",
                wraps=build_fold_checkpoint_path,
            ) as path_helper, patch(
                "src.inference.loading.load_model_state",
                wraps=load_model_state,
            ) as checkpoint_loader:
                extracted = load_fold_models(num_folds, num_classes, device, directory)

            legacy = legacy_load_fold_models(
                num_folds,
                num_classes,
                device,
                directory,
                build_tiny_classifier,
            )
            expected_paths = [
                build_fold_checkpoint_path(directory, fold)
                for fold in range(1, num_folds + 1)
            ]
            self.assertEqual(factory_calls, [num_classes] * num_folds)
            self.assertEqual(
                [call.args for call in path_helper.call_args_list],
                [(directory, fold) for fold in range(1, num_folds + 1)],
            )
            self.assertEqual(
                [call.args[1] for call in checkpoint_loader.call_args_list],
                expected_paths,
            )
            self.assertEqual(
                [call.kwargs["map_location"] for call in checkpoint_loader.call_args_list],
                [device] * num_folds,
            )
            self.assertIsInstance(extracted, list)
            self.assertEqual(len(extracted), num_folds)

            inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
            for source, actual, expected in zip(source_models, extracted, legacy):
                assert_state_dict_equal(self, actual.state_dict(), source.state_dict())
                assert_state_dict_equal(self, actual.state_dict(), expected.state_dict())
                self.assertFalse(actual.training)
                self.assertFalse(expected.training)
                torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)

    def test_missing_and_incomplete_checkpoints_preserve_exception_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "src.inference.loading.build_cmt_classifier",
                side_effect=build_tiny_classifier,
            ):
                with self.assertRaises(FileNotFoundError):
                    load_fold_models(1, 3, "cpu", directory)
            with self.assertRaises(FileNotFoundError):
                legacy_load_fold_models(1, 3, "cpu", directory, build_tiny_classifier)

            source = build_tiny_classifier(3)
            incomplete_path = Path(build_fold_checkpoint_path(directory, 1))
            torch.save(dict(list(source.state_dict().items())[:1]), incomplete_path)
            with patch(
                "src.inference.loading.build_cmt_classifier",
                side_effect=build_tiny_classifier,
            ):
                with self.assertRaises(RuntimeError):
                    load_fold_models(1, 3, "cpu", directory)
            with self.assertRaises(RuntimeError):
                legacy_load_fold_models(1, 3, "cpu", directory, build_tiny_classifier)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this parity check")
    def test_synthetic_cuda_loading_preserves_state_mode_device_and_output(self):
        device = torch.device("cuda")
        with tempfile.TemporaryDirectory() as directory:
            source = self.create_checkpoints(directory, 1, 3)[0]
            with patch(
                "src.inference.loading.build_cmt_classifier",
                side_effect=build_tiny_classifier,
            ):
                loaded = load_fold_models(1, 3, device, directory)[0]

            self.assertFalse(loaded.training)
            self.assertEqual(next(loaded.parameters()).device.type, "cuda")
            assert_state_dict_equal(
                self,
                {key: value.cpu() for key, value in loaded.state_dict().items()},
                source.state_dict(),
            )
            with torch.inference_mode():
                output = loaded(torch.ones(2, 1, 2, 2, device=device))
            self.assertEqual(output.device.type, "cuda")
            self.assertFalse(output.requires_grad)


class SingleFoldModelLoadingTest(unittest.TestCase):
    def test_single_fold_loader_reuses_checkpoint_contract_without_loading_other_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = build_tiny_classifier(3)
            checkpoint = Path(build_fold_checkpoint_path(directory, 2))
            save_model_state(source, checkpoint)

            with patch(
                "src.inference.loading.build_cmt_classifier",
                side_effect=build_tiny_classifier,
            ), patch(
                "src.inference.loading.build_fold_checkpoint_path",
                wraps=build_fold_checkpoint_path,
            ) as path_helper:
                loaded = load_fold_model(3, "cpu", directory, 2)

        self.assertFalse(loaded.training)
        self.assertEqual([call.args for call in path_helper.call_args_list], [(directory, 2)])
        assert_state_dict_equal(self, loaded.state_dict(), source.state_dict())

if __name__ == "__main__":
    unittest.main()
