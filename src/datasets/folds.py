"""Fold split helpers for fruit-freshness classification."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def iter_stratified_folds(dataset, n_splits, shuffle, random_state):
    """Return the notebook's stratified fold iterator for a dataset split."""
    labels = np.asarray(dataset["label"], dtype=np.int64)
    indices = np.arange(len(dataset), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    return splitter.split(indices, labels)


def select_fold_datasets(dataset, train_indices, validation_indices):
    """Select the notebook's train and validation subsets from fold indices."""
    train_dataset = dataset.select(list(train_indices))
    validation_dataset = dataset.select(list(validation_indices))
    return train_dataset, validation_dataset
