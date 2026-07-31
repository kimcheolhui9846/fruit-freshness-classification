import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch
from PIL import Image


REMOVED_LABELS = [18, 20, 16, 13, 2, 5, 7, 9]


class FakeClassLabel:
    def __init__(self, num_classes, names):
        self.num_classes = num_classes
        self.names = names

    def int2str(self, index):
        return self.names[index]


class FakeSplit:
    def __init__(self, items, label_feature):
        self.items = items
        self.features = {"label": label_feature}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        if key == "label":
            return [item["label"] for item in self.items]
        return self.items[key]

    def select(self, indices):
        return FakeSplit([self.items[index].copy() for index in indices], self.features["label"])

    def train_test_split(self, test_size, seed):
        test_count = int(len(self.items) * test_size)
        return {
            "train": FakeSplit(self.items[test_count:], self.features["label"]),
            "test": FakeSplit(self.items[:test_count], self.features["label"]),
        }

    def map(self, function, **_):
        return FakeSplit([function(item.copy()) for item in self.items], self.features["label"])

    def cast_column(self, name, feature):
        self.features[name] = feature
        return self


class FakeDatasetDict(dict):
    pass


def _load_dataset_module():
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.ClassLabel = FakeClassLabel
    fake_datasets.DatasetDict = FakeDatasetDict
    fake_datasets.load_dataset = lambda *_: None
    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        sys.modules.pop("src.datasets.fruit_freshness", None)
        return importlib.import_module("src.datasets.fruit_freshness")


def _build_source_dataset():
    label_names = [f"class_{index}" for index in range(21)]
    items = [
        {
            "image": Image.new("L" if index % 2 else "RGB", (4, 4), color=index % 255),
            "label": label,
        }
        for label in range(21)
        for index in range(5)
    ]
    return FakeDatasetDict({"train": FakeSplit(items, FakeClassLabel(21, label_names))}), label_names


class FruitFreshnessDatasetTest(unittest.TestCase):
    def test_loading_filtering_remapping_and_item_contract(self):
        module = _load_dataset_module()
        source, original_names = _build_source_dataset()
        with patch.object(module, "load_dataset", return_value=source) as mocked_load_dataset:
            with patch.object(module.os, "cpu_count", return_value=2):
                prepared = module.load_fruit_freshness_dataset()

        mocked_load_dataset.assert_called_once_with("Densu341/Fresh-rotten-fruit")
        expected_names = [
            name for index, name in enumerate(original_names) if index not in REMOVED_LABELS
        ]
        self.assertEqual(prepared.keys(), {"train", "test"})
        self.assertEqual(prepared["train"].features["label"].names, expected_names)
        self.assertEqual(prepared["test"].features["label"].names, expected_names)
        self.assertEqual(
            sorted(set(prepared["train"]["label"]) | set(prepared["test"]["label"])),
            list(range(len(expected_names))),
        )
        self.assertEqual(
            len(prepared["train"]) + len(prepared["test"]),
            len(original_names) * 5 - len(REMOVED_LABELS) * 5,
        )
        self.assertEqual(prepared["train"][0]["image"].mode, "RGB")

        wrapped = module.FruitHFDataset(prepared["train"].select([0]))
        image, target = wrapped[0]
        self.assertEqual(image.mode, "RGB")
        self.assertTrue(torch.is_tensor(target))
        self.assertEqual(target.dtype, torch.long)

        transformed = module.FruitHFDataset(
            prepared["train"].select([0]),
            transform=lambda image: (image.mode, image.size),
        )
        image, target = transformed[0]
        self.assertEqual(image, ("RGB", (4, 4)))
        self.assertEqual(target.dtype, torch.long)
