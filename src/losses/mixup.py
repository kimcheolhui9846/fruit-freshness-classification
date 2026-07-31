"""Loss-only Mixup composition helper."""


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Apply the notebook's Mixup criterion combination."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
