"""Saved-fold model loading used by the notebook's final ensemble."""

from src.engine.checkpoint import load_model_state
from src.models.factory import build_cmt_classifier
from src.utils.paths import build_fold_checkpoint_path


def load_fold_model(num_classes, device, ckpt_dir, fold):
    """Load one evaluation-ready fold checkpoint without loading other folds."""
    model = build_cmt_classifier(num_classes).to(device)
    path = build_fold_checkpoint_path(ckpt_dir, fold)
    load_model_state(model, path, map_location=device)
    model.eval()
    return model


def load_fold_models(num_folds, num_classes, device, ckpt_dir):
    """Load the notebook's fold checkpoints in ascending fold order."""
    return [
        load_fold_model(num_classes, device, ckpt_dir, fold)
        for fold in range(1, num_folds + 1)
    ]
