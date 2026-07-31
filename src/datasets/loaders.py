"""DataLoader builders for fruit-freshness experiments."""

from torch.utils.data import DataLoader


def build_fold_dataloaders(train_dataset, validation_dataset, batch_size):
    """Build the notebook's train and validation fold DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, validation_loader


def build_holdout_dataloader(dataset, batch_size):
    """Build the notebook's final holdout DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
