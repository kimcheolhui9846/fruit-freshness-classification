"""Optimizer and scheduler construction used by the notebook."""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_optimizer(model, lr_cnn, lr_trans, weight_decay):
    """Build the notebook's two-group AdamW optimizer."""
    cnn_modules = [model.stem, model.stage1, model.stage2, model.stage3]
    trans_modules = [
        model.to_embed1,
        model.trans1,
        model.down_tokens,
        model.trans2,
        model.head_norm,
        model.fc,
    ]

    cnn_params = []
    trans_params = []
    for module in cnn_modules:
        cnn_params += list(module.parameters())
    for module in trans_modules:
        trans_params += list(module.parameters())

    return torch.optim.AdamW(
        [
            {"params": cnn_params, "lr": lr_cnn},
            {"params": trans_params, "lr": lr_trans},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, t_max):
    """Build the notebook's epoch-based cosine annealing scheduler."""
    return CosineAnnealingLR(optimizer, T_max=t_max)
