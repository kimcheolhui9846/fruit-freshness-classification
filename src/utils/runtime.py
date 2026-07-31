"""Runtime environment helpers."""

import torch


def resolve_device() -> torch.device:
    """Return the notebook's CUDA-first PyTorch device selection."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
