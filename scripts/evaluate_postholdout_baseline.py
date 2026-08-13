"""Development-only OOF evaluation for the Phase 9 post-holdout baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.datasets.postholdout import sha256_json_identity_file

from src.utils.config import (
    load_experiment_config,
    resolve_experiment_validation,
    validate_postholdout_baseline_config,
)
from src.utils.paths import build_fold_checkpoint_path
from src.utils.runtime import resolve_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "deep3_canonical.toml"
DEFAULT_BASELINE_CONFIG_PATH = Path("configs/deep3_postholdout_baseline.toml")
DEFAULT_OUTPUT_DIRECTORY = Path("results/deep3-postholdout-research-01-baseline")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for local-only development-CV OOF evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 9 baseline checkpoints on development CV only.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASELINE_CONFIG_PATH,
        help="Phase 9 baseline TOML path, relative to the repository root.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 9 fold-best checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Empty local-only directory for OOF metrics and prediction artifacts.",
    )
    return parser


def _resolve_repository_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository rather than the caller CWD."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPOSITORY_ROOT / candidate



def resolve_fold_checkpoint_paths(checkpoint_dir: Path, num_folds: int) -> list[Path]:
    """Require one best checkpoint for every frozen development fold."""
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Checkpoint directory does not exist or is not a directory: {checkpoint_dir}"
        )
    checkpoint_paths = [
        Path(build_fold_checkpoint_path(str(checkpoint_dir), fold))
        for fold in range(1, num_folds + 1)
    ]
    missing = [path for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required development fold checkpoint(s): "
            + ", ".join(str(path) for path in missing)
        )
    return checkpoint_paths


def _load_evaluation_dependencies() -> SimpleNamespace:
    """Import model and data dependencies only for an evaluation invocation."""
    from src.datasets.folds import select_fold_datasets
    from src.datasets.fruit_freshness import FruitHFDataset, load_fruit_freshness_dataset
    from src.datasets.loaders import build_holdout_dataloader
    from src.datasets.postholdout import (
        cv_folds_from_manifest,
        load_frozen_postholdout_manifest,
        load_postholdout_cv_manifest,
        select_frozen_development_pool,
    )
    from src.evaluation.diagnostics import compute_classification_diagnostics
    from src.inference.loading import load_fold_model
    from src.losses.focal import FocalLoss, build_class_balanced_alpha
    from src.trainers.loops import validate_one_epoch
    from src.transforms.classification import build_validation_transform

    return SimpleNamespace(
        FocalLoss=FocalLoss,
        FruitHFDataset=FruitHFDataset,
        build_class_balanced_alpha=build_class_balanced_alpha,
        build_holdout_dataloader=build_holdout_dataloader,
        build_validation_transform=build_validation_transform,
        compute_classification_diagnostics=compute_classification_diagnostics,
        cv_folds_from_manifest=cv_folds_from_manifest,
        load_fold_model=load_fold_model,
        load_frozen_postholdout_manifest=load_frozen_postholdout_manifest,
        load_fruit_freshness_dataset=load_fruit_freshness_dataset,
        load_postholdout_cv_manifest=load_postholdout_cv_manifest,
        select_fold_datasets=select_fold_datasets,
        select_frozen_development_pool=select_frozen_development_pool,
        validate_one_epoch=validate_one_epoch,
    )


def prepare_development_dataset_and_folds(
    config: dict,
    dependencies: SimpleNamespace,
) -> tuple[object, list[tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    """Return only the frozen development pool and its tracked CV positions."""
    post_holdout = config.get("post_holdout")
    if post_holdout is None:
        raise ValueError("Development-only OOF evaluation requires a post-holdout config.")

    split_path = _resolve_repository_path(post_holdout["split_manifest_path"])
    cv_path = _resolve_repository_path(post_holdout["cv_manifest_path"])
    final_dataset = dependencies.load_fruit_freshness_dataset()
    frozen_manifest = dependencies.load_frozen_postholdout_manifest(split_path)
    development_dataset = dependencies.select_frozen_development_pool(
        final_dataset["train"],
        final_dataset["test"],
        frozen_manifest,
    )
    cv_manifest = dependencies.load_postholdout_cv_manifest(
        cv_path,
        development_manifest_sha256=sha256_json_identity_file(split_path),
        development_count=frozen_manifest["development_count"],
    )
    folds = dependencies.cv_folds_from_manifest(cv_manifest)
    protocol = {
        "experiment_id": post_holdout["experiment_id"],
        "parent_experiment_id": post_holdout["parent_experiment_id"],
        "data_protocol": "DEV_PLUS_LOCKED_TEST",
        "development_count": frozen_manifest["development_count"],
        "locked_test_count": frozen_manifest["locked_test_count"],
        "split_manifest_sha256": sha256_json_identity_file(split_path),
        "cv_manifest_sha256": sha256_json_identity_file(cv_path),
        "locked_test_model_access": "NO",
        "canonical_holdout_model_access": "NO",
    }
    return development_dataset, folds, protocol


def assemble_oof_predictions(*, expected_labels, num_classes: int, fold_outputs) -> dict[str, np.ndarray]:
    """Assemble one and only one validation prediction for every development row."""
    expected = np.asarray(expected_labels, dtype=np.int64)
    if expected.ndim != 1 or len(expected) == 0:
        raise ValueError("Expected development labels must be a non-empty one-dimensional array.")

    count = len(expected)
    labels = np.full(count, -1, dtype=np.int64)
    predictions = np.full(count, -1, dtype=np.int64)
    logits = np.full((count, num_classes), np.nan, dtype=np.float32)
    fold_assignments = np.zeros(count, dtype=np.int64)

    for output in fold_outputs:
        fold = int(output["fold"])
        indices = np.asarray(output["validation_indices"], dtype=np.int64)
        fold_labels = np.asarray(output["labels"], dtype=np.int64)
        fold_predictions = np.asarray(output["predictions"], dtype=np.int64)
        fold_logits = np.asarray(output["logits"], dtype=np.float32)

        if (
            indices.ndim != 1
            or len(indices) != len(fold_labels)
            or len(indices) != len(fold_predictions)
            or fold_logits.shape != (len(indices), num_classes)
        ):
            raise ValueError("Fold OOF output does not match its validation positions.")
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("Fold OOF validation positions are outside the development pool.")
        if np.any(fold_assignments[indices] != 0):
            raise ValueError("Every development example must receive exactly once OOF prediction.")
        if not np.array_equal(fold_labels, expected[indices]):
            raise ValueError("OOF prediction labels do not match frozen development labels.")

        labels[indices] = fold_labels
        predictions[indices] = fold_predictions
        logits[indices] = fold_logits
        fold_assignments[indices] = fold

    if np.any(fold_assignments == 0):
        raise ValueError("Every development example must receive exactly once OOF prediction.")
    if not np.array_equal(labels, expected) or np.any(~np.isfinite(logits)):
        raise ValueError("OOF assembly did not preserve every frozen development example.")

    return {
        "labels": labels,
        "predictions": predictions,
        "logits": logits,
        "fold_assignments": fold_assignments,
    }


def evaluate_development_cv(
    config: dict,
    development_dataset,
    folds: list[tuple[np.ndarray, np.ndarray]],
    checkpoint_directory: Path,
    device,
    dependencies: SimpleNamespace,
) -> dict:
    """Evaluate each fold only on its matching development validation partition."""
    names = list(development_dataset.features["label"].names)
    expected_labels = np.asarray(development_dataset["label"], dtype=np.int64)
    class_counts = [
        int(np.count_nonzero(expected_labels == index))
        for index in range(len(names))
    ]
    alpha = dependencies.build_class_balanced_alpha(
        class_counts,
        config["loss"]["class_balanced_beta"],
        len(names),
    )
    if config["loss"]["use_ce_label_smoothing"]:
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config["loss"]["label_smoothing"],
        ).to(device)
    else:
        criterion = dependencies.FocalLoss(
            alpha=alpha.to(device),
            gamma=config["loss"]["focal_gamma"],
        ).to(device)

    checkpoint_paths = resolve_fold_checkpoint_paths(
        checkpoint_directory,
        config["cross_validation"]["n_splits"],
    )
    if len(folds) != len(checkpoint_paths):
        raise ValueError("Tracked CV folds do not match the required checkpoint count.")

    validation_transform = dependencies.build_validation_transform()
    fold_outputs = []
    fold_metrics = []
    for fold, ((train_indices, validation_indices), checkpoint_path) in enumerate(
        zip(folds, checkpoint_paths),
        start=1,
    ):
        _, validation_split = dependencies.select_fold_datasets(
            development_dataset,
            train_indices,
            validation_indices,
        )
        validation_dataset = dependencies.FruitHFDataset(
            validation_split,
            transform=validation_transform,
        )
        validation_loader = dependencies.build_holdout_dataloader(
            validation_dataset,
            config["training"]["batch_size"],
        )
        model = dependencies.load_fold_model(
            len(names),
            device,
            str(checkpoint_directory),
            fold,
        )
        accuracy, loss, predictions, labels, logits = dependencies.validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device,
            progress_description=f"Phase 9.3 development fold {fold}",
        )
        fold_logits = np.concatenate(logits, axis=0)
        fold_diagnostics = dependencies.compute_classification_diagnostics(
            labels,
            predictions,
            fold_logits,
            names,
        )
        fold_diagnostics.update(
            {
                "fold": fold,
                "validation_count": int(len(validation_indices)),
                "validation_loss": float(loss),
                "validation_top1_accuracy": float(accuracy),
                "checkpoint_filename": checkpoint_path.name,
            }
        )
        fold_metrics.append(fold_diagnostics)
        fold_outputs.append(
            {
                "fold": fold,
                "validation_indices": validation_indices,
                "labels": labels,
                "predictions": predictions,
                "logits": fold_logits,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assembled = assemble_oof_predictions(
        expected_labels=expected_labels,
        num_classes=len(names),
        fold_outputs=fold_outputs,
    )
    return {
        "metrics": dependencies.compute_classification_diagnostics(
            assembled["labels"],
            assembled["predictions"],
            assembled["logits"],
            names,
        ),
        "fold_metrics": fold_metrics,
        **assembled,
    }


def write_development_oof_artifacts(output_directory: Path, payload: dict) -> dict[str, Path]:
    """Write local-only OOF outputs once, refusing any collision or replacement."""
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"OOF output directory must be empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    metrics_path = output_directory / "development_oof_metrics.json"
    per_class_path = output_directory / "development_oof_per_class_metrics.csv"
    confusion_path = output_directory / "development_oof_confusion_matrix.csv"
    predictions_path = output_directory / "development_oof_predictions.npz"

    metadata = {
        "experiment_id": payload["experiment_id"],
        "data_protocol": payload.get("data_protocol", "DEV_PLUS_LOCKED_TEST"),
        "metrics": payload["metrics"],
        "fold_metrics": payload.get("fold_metrics", []),
        "integrity": payload.get("integrity", {}),
    }
    metrics_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    per_class_rows = payload["metrics"]["per_class"]
    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    names = [row["class_name"] for row in per_class_rows]
    with confusion_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *names])
        for name, row in zip(names, payload["metrics"]["confusion_matrix"]):
            writer.writerow([name, *row])

    np.savez_compressed(
        predictions_path,
        labels=np.asarray(payload["labels"], dtype=np.int64),
        predictions=np.asarray(payload["predictions"], dtype=np.int64),
        logits=np.asarray(payload["logits"], dtype=np.float32),
        fold_assignments=np.asarray(payload["fold_assignments"], dtype=np.int64),
    )
    return {
        "metrics": metrics_path,
        "per_class": per_class_path,
        "confusion_matrix": confusion_path,
        "predictions": predictions_path,
    }


def run_evaluation(args: argparse.Namespace) -> dict:
    """Run the approved development-only OOF evaluation flow."""
    config_path = _resolve_repository_path(args.config)
    checkpoint_directory = _resolve_repository_path(args.checkpoint_dir)
    output_directory = _resolve_repository_path(args.output_dir)
    config = load_experiment_config(config_path)
    # Evaluation must not accept a config training would have rejected. A
    # baseline-parented experiment is checked against the baseline; anything
    # parented to the research identity keeps the canonical comparison
    # untouched, so the existing guard is added beside rather than relaxed.
    experiment_validation = resolve_experiment_validation(config, config_path)
    if experiment_validation is None:
        validation = validate_postholdout_baseline_config(
            CANONICAL_CONFIG_PATH,
            config_path,
        )
        if not validation["recipe_equivalent"]:
            raise RuntimeError(
                "Baseline recipe equivalence validation failed before OOF evaluation."
            )
    torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]
    device = resolve_device()
    dependencies = _load_evaluation_dependencies()
    development_dataset, folds, protocol = prepare_development_dataset_and_folds(
        config,
        dependencies,
    )
    results = evaluate_development_cv(
        config,
        development_dataset,
        folds,
        checkpoint_directory,
        device,
        dependencies,
    )
    payload = {
        "experiment_id": protocol["experiment_id"],
        "data_protocol": protocol["data_protocol"],
        "integrity": {
            "post_holdout_locked_test_model_forward_passes": 0,
            "post_holdout_locked_test_predictions": 0,
            "post_holdout_locked_test_metrics": 0,
            "canonical_holdout_model_forward_passes": 0,
            "canonical_holdout_new_metrics": 0,
        },
        **results,
    }
    artifacts = write_development_oof_artifacts(output_directory, payload)
    return {
        "artifacts": artifacts,
        "metrics": payload["metrics"],
        "fold_metrics": payload["fold_metrics"],
        "protocol": protocol,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute local-only development-CV evaluation."""
    args = build_parser().parse_args(argv)
    summary = run_evaluation(args)
    print("OOF development Macro F1:", summary["metrics"]["macro_f1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
