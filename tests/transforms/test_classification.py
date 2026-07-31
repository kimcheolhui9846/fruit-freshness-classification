import random
import unittest

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.transforms.classification import (
    build_finetune_transform,
    build_train_transform,
    build_validation_transform,
)


def _legacy_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])


def _legacy_validation_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _legacy_finetune_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _seed_random_sources():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)


class ClassificationTransformParityTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (320, 280), color=(20, 40, 60))

    def test_train_transform_structure_and_seeded_parity(self):
        actual = build_train_transform()
        legacy = _legacy_train_transform()

        self.assertEqual(repr(actual), repr(legacy))
        _seed_random_sources()
        expected = legacy(self.image)
        _seed_random_sources()
        observed = actual(self.image)
        self.assertEqual(observed.shape, (3, 224, 224))
        self.assertEqual(observed.dtype, torch.float32)
        torch.testing.assert_close(observed, expected)

    def test_validation_transform_structure_and_determinism(self):
        actual = build_validation_transform()
        legacy = _legacy_validation_transform()

        self.assertEqual(repr(actual), repr(legacy))
        expected = legacy(self.image)
        observed = actual(self.image)
        self.assertEqual(observed.shape, (3, 224, 224))
        self.assertEqual(observed.dtype, torch.float32)
        torch.testing.assert_close(observed, expected)

    def test_finetune_transform_structure_and_seeded_parity(self):
        actual = build_finetune_transform()
        legacy = _legacy_finetune_transform()

        self.assertEqual(repr(actual), repr(legacy))
        _seed_random_sources()
        expected = legacy(self.image)
        _seed_random_sources()
        observed = actual(self.image)
        self.assertEqual(observed.shape, (3, 224, 224))
        self.assertEqual(observed.dtype, torch.float32)
        torch.testing.assert_close(observed, expected)
