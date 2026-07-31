"""Loss functions used by the fruit-freshness notebook."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_class_balanced_alpha(class_counts, beta, num_classes):
    """Build the notebook's class-balanced alpha tensor from precomputed counts."""
    effective_num = [1.0 - (beta ** count) for count in class_counts]
    raw_alpha = torch.tensor(
        [(1.0 - beta) / (value if value > 0 else 1e-8) for value in effective_num],
        dtype=torch.float32,
    )
    return (raw_alpha / raw_alpha.sum()) * num_classes


class FocalLoss(nn.Module):
    """Multi-class focal loss with the notebook's alpha semantics."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", eps=1e-8):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.eps = float(eps)

        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        pt = pt.clamp(min=self.eps, max=1.0 - self.eps)
        log_pt = torch.log(pt)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = torch.ones_like(pt)

        focal = (1.0 - pt).pow(self.gamma)
        loss = -alpha_t * focal * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
