import unittest

import torch
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset

from src.datasets.loaders import build_fold_dataloaders, build_holdout_dataloader


class DataLoaderContractTest(unittest.TestCase):
    def setUp(self):
        features = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        self.dataset = TensorDataset(features, targets)

    def test_fold_loader_parameters_and_batch_contract(self):
        train_loader, validation_loader = build_fold_dataloaders(
            self.dataset,
            self.dataset,
            batch_size=3,
        )

        self.assertEqual(train_loader.batch_size, 3)
        self.assertEqual(validation_loader.batch_size, 3)
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertIsInstance(validation_loader.sampler, SequentialSampler)
        self.assertEqual(train_loader.num_workers, 0)
        self.assertEqual(validation_loader.num_workers, 0)
        self.assertTrue(train_loader.pin_memory)
        self.assertTrue(validation_loader.pin_memory)
        self.assertFalse(train_loader.drop_last)
        self.assertFalse(validation_loader.drop_last)

        first_features, first_targets = next(iter(validation_loader))
        self.assertEqual(first_features.shape, (3, 4))
        self.assertEqual(first_targets.dtype, torch.long)
        self.assertEqual(first_features.device.type, "cpu")
        self.assertEqual(len(list(validation_loader)), 2)
        self.assertEqual(list(validation_loader)[-1][0].shape[0], 2)

    def test_holdout_loader_preserves_notebook_parameters(self):
        loader = build_holdout_dataloader(self.dataset, batch_size=3)

        self.assertEqual(loader.batch_size, 3)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.pin_memory)
        self.assertFalse(loader.drop_last)
