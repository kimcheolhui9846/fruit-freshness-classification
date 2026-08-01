"""Reproducible holdout evaluation entry point for fruit-freshness classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src.utils.config import load_experiment_config
from src.utils.paths import build_fold_checkpoint_path
from src.utils.runtime import resolve_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal CLI for the committed holdout evaluation flow."""
    parser = argparse.ArgumentParser(
        description="Evaluate the configured fruit-freshness fold ensemble.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deep3.toml"),
        help="Experiment TOML path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing every required best_model_fold{fold}.pt checkpoint.",
    )
    return parser


def _resolve_repository_path(path: str | Path) -> Path:
    """Resolve relative CLI paths from the repository rather than the caller CWD."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPOSITORY_ROOT / candidate


def resolve_fold_checkpoint_paths(checkpoint_dir: Path, num_folds: int) -> list[Path]:
    """Validate the complete, ordered checkpoint set used by the notebook ensemble."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_dir}")
    if not any(checkpoint_dir.iterdir()):
        raise FileNotFoundError(f"Checkpoint directory is empty: {checkpoint_dir}")

    checkpoint_paths = [
        Path(build_fold_checkpoint_path(str(checkpoint_dir), fold))
        for fold in range(1, num_folds + 1)
    ]
    missing_paths = [path for path in checkpoint_paths if not path.is_file()]
    if missing_paths:
        formatted_paths = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing required fold checkpoint(s): {formatted_paths}")
    return checkpoint_paths


def _load_evaluation_dependencies() -> SimpleNamespace:
    """Import production evaluation APIs only when evaluation is requested."""
    from src.datasets.fruit_freshness import (
        FruitHFDataset,
        load_fruit_freshness_dataset,
    )
    from src.datasets.loaders import build_holdout_dataloader
    from src.inference.ensemble import run_ensemble_holdout
    from src.inference.loading import load_fold_models
    from src.transforms.classification import build_validation_transform

    return SimpleNamespace(
        FruitHFDataset=FruitHFDataset,
        build_holdout_dataloader=build_holdout_dataloader,
        build_validation_transform=build_validation_transform,
        load_fold_models=load_fold_models,
        load_fruit_freshness_dataset=load_fruit_freshness_dataset,
        run_ensemble_holdout=run_ensemble_holdout,
    )


def run_evaluation(args: argparse.Namespace) -> dict:
    """Run the active notebook's final labeled holdout ensemble evaluation."""
    config_path = _resolve_repository_path(args.config)
    checkpoint_directory = _resolve_repository_path(args.checkpoint_dir)

    device = resolve_device()
    print("device:", device)
    config = load_experiment_config(config_path)
    torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]

    num_folds = config["cross_validation"]["n_splits"]
    checkpoint_paths = resolve_fold_checkpoint_paths(checkpoint_directory, num_folds)
    dependencies = _load_evaluation_dependencies()

    final_dataset = dependencies.load_fruit_freshness_dataset()
    num_classes = len(final_dataset["train"].features["label"].names)
    validation_transform = dependencies.build_validation_transform()
    test_dataset = dependencies.FruitHFDataset(
        final_dataset["test"],
        transform=validation_transform,
    )
    test_loader = dependencies.build_holdout_dataloader(
        test_dataset,
        config["training"]["batch_size"],
    )
    models = dependencies.load_fold_models(
        num_folds,
        num_classes,
        device,
        str(checkpoint_directory),
    )

    print("\n[최종 평가] Holdout Test Set (Ensemble + TTA)")
    correct, total = dependencies.run_ensemble_holdout(models, test_loader, device)
    accuracy = correct / total
    print("Final Holdout Acc:", accuracy)

    return {
        "accuracy": accuracy,
        "checkpoint_paths": checkpoint_paths,
        "correct": correct,
        "total": total,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse evaluation arguments, run the holdout flow, and return success."""
    args = build_parser().parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())