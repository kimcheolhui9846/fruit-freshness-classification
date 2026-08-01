"""Fruit-freshness Hugging Face dataset preparation."""

import os
from pathlib import Path, PurePosixPath
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
from datasets import ClassLabel, DatasetDict, config as datasets_config, load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset


DATASET_REPOSITORY_ID = "Densu341/Fresh-rotten-fruit"
DATASET_REVISION = "2077850adc575aa1e8d6029e6cd6cefe9e403a1c"
DATASET_ARCHIVE_FILENAME = "freshness_fruit.zip"
DATASET_CONTENT_DIRECTORY = "dataset"


def _validate_archive_members(archive: ZipFile) -> None:
    """Reject archive members that would escape the managed cache directory."""
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe dataset archive member: {member.filename}")


def _resolve_imagefolder_data_dir() -> Path:
    """Download the pinned archive and expose its image-directory root."""
    extraction_parent = (
        Path(datasets_config.HF_DATASETS_CACHE)
        / "fruit_freshness"
        / DATASET_REVISION
    )
    data_dir = extraction_parent / DATASET_CONTENT_DIRECTORY
    if data_dir.is_dir():
        return data_dir
    if extraction_parent.exists():
        raise RuntimeError(
            "Fruit-freshness extraction cache is incomplete: "
            f"{extraction_parent}"
        )

    archive_path = Path(
        hf_hub_download(
            repo_id=DATASET_REPOSITORY_ID,
            repo_type="dataset",
            filename=DATASET_ARCHIVE_FILENAME,
            revision=DATASET_REVISION,
        )
    )
    staging_parent = extraction_parent.with_name(
        f".{extraction_parent.name}.tmp-{uuid4().hex}"
    )
    staging_parent.mkdir(parents=True)
    with ZipFile(archive_path) as archive:
        _validate_archive_members(archive)
        archive.extractall(staging_parent)

    staging_data_dir = staging_parent / DATASET_CONTENT_DIRECTORY
    if not staging_data_dir.is_dir():
        raise ValueError(
            "Fruit-freshness archive does not contain the expected "
            f"{DATASET_CONTENT_DIRECTORY!r} directory."
        )
    try:
        staging_parent.rename(extraction_parent)
    except FileExistsError:
        if data_dir.is_dir():
            return data_dir
        raise
    return data_dir


def load_fruit_freshness_dataset():
    """Load and prepare the notebook's fruit-freshness dataset splits."""
    dataset = load_dataset("imagefolder", data_dir=str(_resolve_imagefolder_data_dir()))

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
