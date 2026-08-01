"""Reproducible training entry point for fruit-freshness classification."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.amp import GradScaler

from src.utils.config import load_experiment_config
from src.utils.runtime import resolve_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal command-line interface for the committed experiment."""
    parser = argparse.ArgumentParser(
        description="Train the configured fruit-freshness classifier.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deep3.toml"),
        help="Experiment TOML path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Checkpoint directory, relative to the repository root by default.",
    )
    return parser


def _resolve_repository_path(path: str | Path) -> Path:
    """Resolve relative CLI paths from the repository rather than the caller CWD."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPOSITORY_ROOT / candidate


def _load_training_dependencies() -> SimpleNamespace:
    """Import production training APIs only when training is requested.

    Keeping these imports lazy allows ``python -m scripts.train --help`` and a
    normal module import to work without the optional Hugging Face dependency.
    """
    from src.datasets.folds import iter_stratified_folds, select_fold_datasets
    from src.datasets.fruit_freshness import (
        FruitHFDataset,
        load_fruit_freshness_dataset,
    )
    from src.datasets.loaders import build_fold_dataloaders
    from src.engine.checkpoint import save_model_state
    from src.engine.ema import ModelEma
    from src.engine.optimization import build_optimizer, build_scheduler
    from src.evaluation.metrics import compute_validation_metrics
    from src.losses.focal import FocalLoss, build_class_balanced_alpha
    from src.models.factory import build_cmt_classifier
    from src.trainers.loops import train_one_epoch, validate_one_epoch
    from src.transforms.classification import (
        build_finetune_transform,
        build_train_transform,
        build_validation_transform,
    )
    from src.utils.labels import save_label_names
    from src.utils.paths import build_fold_checkpoint_path, ensure_output_directory

    return SimpleNamespace(
        FruitHFDataset=FruitHFDataset,
        FocalLoss=FocalLoss,
        ModelEma=ModelEma,
        build_class_balanced_alpha=build_class_balanced_alpha,
        build_cmt_classifier=build_cmt_classifier,
        build_finetune_transform=build_finetune_transform,
        build_fold_checkpoint_path=build_fold_checkpoint_path,
        build_fold_dataloaders=build_fold_dataloaders,
        build_optimizer=build_optimizer,
        build_scheduler=build_scheduler,
        build_train_transform=build_train_transform,
        build_validation_transform=build_validation_transform,
        compute_validation_metrics=compute_validation_metrics,
        ensure_output_directory=ensure_output_directory,
        iter_stratified_folds=iter_stratified_folds,
        load_fruit_freshness_dataset=load_fruit_freshness_dataset,
        save_label_names=save_label_names,
        save_model_state=save_model_state,
        select_fold_datasets=select_fold_datasets,
        train_one_epoch=train_one_epoch,
        validate_one_epoch=validate_one_epoch,
    )


def run_training(args: argparse.Namespace) -> dict:
    """Run the active notebook training flow with portable output paths."""
    config_path = _resolve_repository_path(args.config)
    output_directory = _resolve_repository_path(args.output_dir)

    device = resolve_device()
    print("device:", device)
    config = load_experiment_config(config_path)
    dependencies = _load_training_dependencies()

    train_transform = dependencies.build_train_transform()
    val_transform = dependencies.build_validation_transform()

    final_dataset = dependencies.load_fruit_freshness_dataset()
    names = final_dataset["train"].features["label"].names
    save_dir = dependencies.ensure_output_directory(str(output_directory))
    dependencies.save_label_names(names, save_dir)

    num_classes = len(final_dataset["train"].features["label"].names)
    train_labels = [int(label) for label in final_dataset["train"]["label"]]
    counts = Counter(train_labels)
    class_counts = [counts[index] for index in range(num_classes)]
    beta = config["loss"]["class_balanced_beta"]
    alpha = dependencies.build_class_balanced_alpha(class_counts, beta, num_classes)
    print("alpha:", alpha.tolist())

    epochs = config["training"]["epochs"]
    finetune_epochs = config["fine_tuning"]["epochs"]
    batch_size = config["training"]["batch_size"]
    num_folds = config["cross_validation"]["n_splits"]
    mixup_alpha = config["mixup"]["alpha"]
    mixup_probability = config["mixup"]["probability"]
    lr_cnn = config["optimization"]["lr_cnn"]
    lr_trans = config["optimization"]["lr_trans"]
    weight_decay = config["optimization"]["weight_decay"]
    ema_decay = config["ema"]["decay"]
    use_ce_label_smoothing = config["loss"]["use_ce_label_smoothing"]
    label_smoothing = config["loss"]["label_smoothing"]
    finetune_transform = dependencies.build_finetune_transform()

    fold_accuracies = []
    histories = []
    start_time = time.time()

    folds = dependencies.iter_stratified_folds(
        final_dataset["train"],
        n_splits=num_folds,
        shuffle=config["cross_validation"]["shuffle"],
        random_state=config["cross_validation"]["random_state"],
    )
    for fold, (train_indices, validation_indices) in enumerate(folds, 1):
        best_acc_fold = 0.0
        print(f"\n================ Fold {fold}/{num_folds} starting ================")
        fold_start = time.time()
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        train_split, validation_split = dependencies.select_fold_datasets(
            final_dataset["train"],
            train_indices,
            validation_indices,
        )
        train_dataset = dependencies.FruitHFDataset(
            train_split,
            transform=train_transform,
        )
        validation_dataset = dependencies.FruitHFDataset(
            validation_split,
            transform=val_transform,
        )
        train_loader, validation_loader = dependencies.build_fold_dataloaders(
            train_dataset,
            validation_dataset,
            batch_size,
        )

        torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]
        model = dependencies.build_cmt_classifier(num_classes).to(device)
        ema = dependencies.ModelEma(model, decay=ema_decay, device=device)

        if use_ce_label_smoothing:
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
            ).to(device)
        else:
            criterion = dependencies.FocalLoss(
                alpha=alpha.to(device),
                gamma=config["loss"]["focal_gamma"],
            ).to(device)

        optimizer = dependencies.build_optimizer(
            model,
            lr_cnn=lr_cnn,
            lr_trans=lr_trans,
            weight_decay=weight_decay,
        )
        scheduler = dependencies.build_scheduler(optimizer, t_max=epochs)
        scaler = GradScaler()

        validation_accuracies = []
        validation_losses = []
        validation_f1_scores = []

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            is_finetuning = epoch > epochs - finetune_epochs
            if is_finetuning:
                print(
                    f"Fold {fold} | Epoch {epoch} "
                    "[Fine-tuning: Mixup OFF, weak augmentation]",
                )
                train_dataset.tf = finetune_transform
            else:
                print(f"\nFold {fold} | Epoch {epoch}/{epochs}")

            train_accuracy, train_loss = dependencies.train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                ema,
                is_finetuning,
                mixup_probability,
                mixup_alpha,
                progress_description=f"Fold {fold} Epoch {epoch} [Train]",
            )
            validation_model = ema.module
            validation_accuracy, validation_loss, predictions, labels, logits = (
                dependencies.validate_one_epoch(
                    validation_model,
                    validation_loader,
                    criterion,
                    device,
                    progress_description=f"Fold {fold} Epoch {epoch} [Val]",
                )
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_accuracy)
            history["val_loss"].append(validation_loss)
            history["val_acc"].append(validation_accuracy)

            logits = np.concatenate(logits, axis=0)
            validation_f1, validation_balanced_accuracy, validation_top2, validation_top3 = (
                dependencies.compute_validation_metrics(labels, predictions, logits)
            )
            validation_accuracies.append(validation_accuracy)
            validation_losses.append(validation_loss)
            validation_f1_scores.append(validation_f1)
            print(
                "Val (EMA) "
                f"acc: {validation_accuracy:.4f} | "
                f"f1: {validation_f1:.4f} | loss: {validation_loss:.4f}",
            )

            if validation_accuracy > best_acc_fold + 1e-6:
                best_acc_fold = validation_accuracy
                save_path = dependencies.build_fold_checkpoint_path(save_dir, fold)
                dependencies.save_model_state(ema.module, save_path)
                print(
                    "New best model (EMA) saved! "
                    f"(fold={fold}, acc={best_acc_fold:.4f})",
                )

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} complete ({epoch_time:.2f} seconds)")
            scheduler.step()

        fold_time = time.time() - fold_start
        histories.append(history)
        print(f"Fold {fold} complete ({fold_time / 60:.2f} minutes)")
        fold_accuracies.append(validation_accuracies)

    final_model_path = Path(save_dir) / config["checkpoint"]["final_model_filename"]
    dependencies.save_model_state(model, final_model_path)
    total_time = time.time() - start_time
    print(f"Training complete ({total_time / 60:.2f} minutes)")

    return {
        "elapsed_seconds": total_time,
        "final_model_path": final_model_path,
        "fold_accuracies": fold_accuracies,
        "histories": histories,
        "output_dir": Path(save_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run training, and return a process exit status."""
    args = build_parser().parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
