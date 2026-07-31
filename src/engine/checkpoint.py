"""State-dict checkpoint primitives used by the notebook."""

import torch


def save_model_state(model, path):
    """Save the model state dict with the notebook's torch.save call."""
    torch.save(model.state_dict(), path)


def load_model_state(model, path, *, map_location=None):
    """Load a state dict with the notebook's default strict behavior."""
    return model.load_state_dict(torch.load(path, map_location=map_location))
