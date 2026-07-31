"""Single-epoch training and validation routines used by the notebook."""

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from src.losses.mixup import mixup_criterion
from src.transforms.mixup import mixup_data


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    ema,
    is_finetuning,
    mixup_probability,
    mixup_alpha,
    progress_description,
):
    """Run the notebook's single training epoch without scheduler stepping."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(dataloader, desc=progress_description, ncols=100)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if is_finetuning:
            do_mix = False
        else:
            do_mix = np.random.rand() < mixup_probability

        if do_mix:
            x_in, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
        else:
            x_in, y_a, y_b, lam = x, y, y, 1.0

        with autocast("cuda"):
            out = model(x_in)
            if do_mix:
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        pred = out.argmax(1)
        if do_mix:
            correct += (
                lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()
            ).sum().item()
        else:
            correct += (pred == y).sum().item()
        total += bs

    tr_acc = correct / max(1, total)
    tr_loss = loss_sum / max(1, total)
    print(f"Train ▶ acc: {tr_acc:.4f} | loss: {tr_loss:.4f}")
    return tr_acc, tr_loss


def validate_one_epoch(model, dataloader, criterion, device, progress_description):
    """Run the notebook's single EMA-model validation epoch with horizontal-flip TTA."""
    model.eval()

    v_total, v_correct, v_loss_sum = 0, 0, 0.0
    all_preds, all_labels, all_logits = [], [], []

    with torch.inference_mode():
        for x, y in tqdm(dataloader, desc=progress_description, ncols=100):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast("cuda"):
                out = (model(x) + model(torch.flip(x, dims=[3]))) / 2
                v_loss = criterion(out, y)

            bs = x.size(0)
            v_loss_sum += v_loss.item() * bs
            preds = out.argmax(1)
            v_correct += (preds == y).sum().item()
            v_total += bs

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            all_logits.append(out.detach().cpu().numpy())

    va_acc = v_correct / max(1, v_total)
    va_loss = v_loss_sum / max(1, v_total)
    return va_acc, va_loss, all_preds, all_labels, all_logits
