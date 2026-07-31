"""Final holdout ensemble inference used by the notebook."""

import torch
from torch.amp import autocast
from tqdm import tqdm


@torch.inference_mode()
def ensemble_logits(models, x):
    """Average raw model logits in the notebook's supplied model order."""
    logits_sum = 0
    for model in models:
        logits_sum = logits_sum + model(x)
    return logits_sum / len(models)


@torch.inference_mode()
def ensemble_logits_tta_hflip(models, x):
    """Average original and width-flipped ensemble logits."""
    x_flip = torch.flip(x, dims=[3])
    logits = ensemble_logits(models, x)
    logits_flip = ensemble_logits(models, x_flip)
    return (logits + logits_flip) / 2


def run_ensemble_holdout(models, dataloader, device):
    """Run the notebook's final holdout loop and return correct and total counts."""
    t_total = t_correct = 0
    for x, y in tqdm(dataloader, ncols=100):
        x = x.to(device)
        y = y.to(device)
        with autocast("cuda"):
            logits = ensemble_logits_tta_hflip(models, x)
        pred = logits.argmax(1)
        t_correct += (pred == y).sum().item()
        t_total += y.size(0)
    return t_correct, t_total
