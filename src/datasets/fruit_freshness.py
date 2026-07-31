"""Fruit-freshness Hugging Face dataset preparation."""

import os

import numpy as np
from datasets import ClassLabel, DatasetDict, load_dataset
from torch.utils.data import Dataset


def load_fruit_freshness_dataset():
    """Load and prepare the notebook's fruit-freshness dataset splits."""
    dataset = load_dataset("Densu341/Fresh-rotten-fruit")

    remove_labels = [18, 20, 16, 13, 2, 5, 7, 9]
    labels = np.array(dataset["train"]["label"])
    mask = ~np.isin(labels, remove_labels)
    clean = dataset["train"].select(np.where(mask)[0])

    split = clean.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    uniq = sorted(set(train_ds["label"]) | set(val_ds["label"]))
    names = [train_ds.features["label"].int2str(index) for index in uniq]
    new_lbl = ClassLabel(num_classes=len(names), names=names)

    def remap(example):
        name = train_ds.features["label"].int2str(example["label"])
        example["label"] = names.index(name)
        return example

    train_ds = train_ds.map(
        remap,
        num_proc=os.cpu_count() // 2,
        load_from_cache_file=True,
        desc="Remap train",
    )
    val_ds = val_ds.map(
        remap,
        num_proc=os.cpu_count() // 2,
        load_from_cache_file=True,
        desc="Remap val",
    )

    train_ds = train_ds.cast_column("label", new_lbl)
    val_ds = val_ds.cast_column("label", new_lbl)

    def to_rgb(example):
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    train_ds = train_ds.map(
        to_rgb,
        num_proc=os.cpu_count() // 2,
        load_from_cache_file=True,
        desc="RGB train",
    )
    val_ds = val_ds.map(
        to_rgb,
        num_proc=os.cpu_count() // 2,
        load_from_cache_file=True,
        desc="RGB val",
    )

    return DatasetDict({"train": train_ds, "test": val_ds})


class FruitHFDataset(Dataset):
    """Wrap a Hugging Face split with the notebook's item contract."""

    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.tf = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if self.tf is not None:
            image = self.tf(image)
        label = item["label"]
        import torch

        if not torch.is_tensor(label):
            import torch

            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.to(dtype=torch.long)
        return image, label
