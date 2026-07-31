"""Classification image transform builders."""

from torchvision import transforms


def build_train_transform():
    """Build the notebook's training image transform pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])


def build_validation_transform():
    """Build the notebook's deterministic validation image transform pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_finetune_transform():
    """Build the notebook's fine-tuning image transform pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
