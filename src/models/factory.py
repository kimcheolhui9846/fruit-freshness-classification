"""Construction helpers for the active CMT classifier."""

from src.models.cmt_classifier import CMTClassifier


def build_cmt_classifier(num_classes: int) -> CMTClassifier:
    """Construct the notebook's CMT classifier with its default arguments."""
    return CMTClassifier(num_classes)
