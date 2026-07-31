"""Output-directory helpers."""

import os


def ensure_output_directory(path: str) -> str:
    """Create an output directory using the notebook's existing semantics."""
    os.makedirs(path, exist_ok=True)
    return path


def build_fold_checkpoint_path(directory: str, fold: int) -> str:
    """Return the notebook's existing per-fold checkpoint path."""
    return os.path.join(directory, f"best_model_fold{fold}.pt")
