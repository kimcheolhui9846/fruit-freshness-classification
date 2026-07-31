import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class FakeStratifiedKFold:
    last_instance = None

    def __init__(self, n_splits, shuffle, random_state):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.received_indices = None
        self.received_labels = None
        FakeStratifiedKFold.last_instance = self

    def split(self, indices, labels):
        self.received_indices = np.asarray(indices)
        self.received_labels = np.asarray(labels)
        yield np.array([1, 4, 7, 10]), np.array([0, 2, 3, 5, 6, 8, 9, 11])
        yield np.array([0, 2, 3, 5]), np.array([1, 4, 6, 7, 8, 9, 10, 11])
        yield np.array([6, 8, 9, 11]), np.array([0, 1, 2, 3, 4, 5, 7, 10])


def load_folds_module():
    sklearn_module = types.ModuleType("sklearn")
    model_selection_module = types.ModuleType("sklearn.model_selection")
    model_selection_module.StratifiedKFold = FakeStratifiedKFold
    sklearn_module.model_selection = model_selection_module

    with patch.dict(
        sys.modules,
        {"sklearn": sklearn_module, "sklearn.model_selection": model_selection_module},
    ):
        sys.modules.pop("src.datasets.folds", None)
        return importlib.import_module("src.datasets.folds")


class LabelDataset:
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if key == "label":
            return self.labels
        raise TypeError("Only the label column is required by this test double.")

    def select(self, indices):
        return LabelDataset([self.labels[index] for index in indices])


class FoldPreparationTest(unittest.TestCase):
    def test_stratified_fold_parameters_and_source_indices_are_preserved(self):
        module = load_folds_module()
        dataset = LabelDataset([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

        folds = list(
            module.iter_stratified_folds(
                dataset,
                n_splits=3,
                shuffle=True,
                random_state=42,
            )
        )

        self.assertEqual(len(folds), 3)
        splitter = FakeStratifiedKFold.last_instance
        self.assertEqual(splitter.n_splits, 3)
        self.assertTrue(splitter.shuffle)
        self.assertEqual(splitter.random_state, 42)
        np.testing.assert_array_equal(splitter.received_indices, np.arange(len(dataset)))
        np.testing.assert_array_equal(splitter.received_labels, np.asarray(dataset.labels))
        np.testing.assert_array_equal(folds[0][0], np.array([1, 4, 7, 10]))
        np.testing.assert_array_equal(folds[0][1], np.array([0, 2, 3, 5, 6, 8, 9, 11]))

    def test_selected_subsets_preserve_index_order(self):
        module = load_folds_module()
        dataset = LabelDataset([10, 11, 12, 13, 14, 15])

        train_dataset, validation_dataset = module.select_fold_datasets(
            dataset,
            np.array([4, 1, 5]),
            np.array([0, 3, 2]),
        )

        self.assertEqual(train_dataset.labels, [14, 11, 15])
        self.assertEqual(validation_dataset.labels, [10, 13, 12])


if __name__ == "__main__":
    unittest.main()
