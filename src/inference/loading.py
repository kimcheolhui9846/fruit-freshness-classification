"""Saved-fold model loading used by the notebook's final ensemble."""

from src.engine.checkpoint import load_model_state
from src.models.factory import build_cmt_classifier
from src.utils.paths import build_fold_checkpoint_path


def load_fold_models(num_folds, num_classes, device, ckpt_dir):
    """Load the notebook's fold checkpoints in ascending fold order."""
    models = []
    for fold in range(1, num_folds + 1):
        model = build_cmt_classifier(num_classes).to(device)
        path = build_fold_checkpoint_path(ckpt_dir, fold)
        load_model_state(model, path, map_location=device)
        model.eval()
        models.append(model)
    return models
