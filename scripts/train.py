"""Reproducible training entry point for fruit-freshness classification."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.amp import GradScaler

from src.engine.training_state import (
    build_training_state,
    load_training_state,
    restore_rng_state,
    save_training_state_atomic,
    validate_training_state,
)
from src.utils.config import load_experiment_config, resolve_experiment_validation
from src.utils.determinism import resolve_policy
from src.utils.runtime import resolve_device


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_MANIFEST_SCHEMA_VERSION = 2
TRAINING_STATE_FILENAME = "training_state.pt"
RUN_MANIFEST_FILENAME = "run_manifest.json"
DATASET_ARCHIVE_SHA256 = (
    "a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e"
)
_EXPECTED_CANONICAL_ARTIFACTS = {
    RUN_MANIFEST_FILENAME,
    TRAINING_STATE_FILENAME,
    "label_names.json",
    "best_model_fold1.pt",
    "best_model_fold2.pt",
    "best_model_fold3.pt",
    "last_model_weights.pt",
}


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
    parser.add_argument(
        "--resume-state",
        type=Path,
        default=None,
        help="Trusted local epoch-boundary state file for an existing run.",
    )
    parser.add_argument(
        "--save-training-state",
        action="store_true",
        help="Save trusted local operational state after completed epochs.",
    )
    parser.add_argument(
        "--require-empty-output-dir",
        action="store_true",
        help="Reject a non-empty output directory before dataset preparation.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Portable identifier required for stateful canonical runs.",
    )
    return parser


def _resolve_repository_path(path: str | Path) -> Path:
    """Resolve relative CLI paths from the repository rather than caller CWD."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _repository_commit() -> str:
    """Return the checked-out repository commit for run identity."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_label_names(label_names: list[str]) -> str:
    payload = json.dumps(
        label_names,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_config_path(config_path: Path) -> str:
    try:
        return config_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(
            "Stateful canonical training requires a config inside the repository."
        ) from error


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _optional_argument(args: argparse.Namespace, name: str, default):
    return getattr(args, name, default)


def _is_finetuning_epoch(epoch: int, epochs: int, fine_tuning_epochs: int) -> bool:
    """Preserve the active notebook fine-tuning boundary."""
    return epoch > epochs - fine_tuning_epochs


def _stateful_options(args: argparse.Namespace) -> dict[str, object]:
    resume_state = _optional_argument(args, "resume_state", None)
    save_training_state = bool(_optional_argument(args, "save_training_state", False))
    require_empty_output_dir = bool(
        _optional_argument(args, "require_empty_output_dir", False)
    )
    run_id = _optional_argument(args, "run_id", None)
    enabled = any(
        (
            resume_state is not None,
            save_training_state,
            require_empty_output_dir,
            run_id is not None,
        )
    )
    if not enabled:
        return {
            "enabled": False,
            "resume_state": None,
            "save_training_state": False,
            "require_empty_output_dir": False,
            "run_id": None,
        }
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        run_id,
    ):
        raise ValueError("Stateful canonical training requires a portable --run-id.")
    if not save_training_state:
        raise ValueError("Stateful canonical training requires --save-training-state.")
    if resume_state is not None and require_empty_output_dir:
        raise ValueError(
            "--resume-state and --require-empty-output-dir are incompatible."
        )
    if resume_state is not None and "://" in str(resume_state):
        raise ValueError("Remote resume-state locations are not accepted.")
    return {
        "enabled": True,
        "resume_state": resume_state,
        "save_training_state": True,
        "require_empty_output_dir": require_empty_output_dir,
        "run_id": run_id,
    }


def _prepare_stateful_output_directory(
    output_directory: Path,
    options: dict[str, object],
) -> None:
    """Apply fresh-run collision checks before any dataset preparation."""
    resume_state = options["resume_state"]
    if resume_state is None:
        if options["require_empty_output_dir"]:
            if output_directory.exists() and any(output_directory.iterdir()):
                raise RuntimeError(
                    "Canonical output directory must be empty before dataset "
                    f"preparation: {output_directory}"
                )
            output_directory.mkdir(parents=True, exist_ok=True)
        return

    if not output_directory.is_dir():
        raise FileNotFoundError(
            f"Resume output directory does not exist: {output_directory}"
        )
    expected_state = output_directory / TRAINING_STATE_FILENAME
    if _resolve_repository_path(resume_state) != expected_state:
        raise ValueError(
            "Resume state must be the approved training_state.pt inside "
            "the selected output directory."
        )
    if not expected_state.is_file() or not (
        output_directory / RUN_MANIFEST_FILENAME
    ).is_file():
        raise FileNotFoundError(
            "Resume requires both training_state.pt and run_manifest.json."
        )
    unexpected = sorted(
        path.name
        for path in output_directory.iterdir()
        if path.name not in _EXPECTED_CANONICAL_ARTIFACTS
    )
    if unexpected:
        raise RuntimeError(
            "Resume output directory contains unexpected files: "
            f"{', '.join(unexpected)}"
        )


def _build_run_manifest(
    *,
    metadata: dict[str, object],
    config: dict,
    device: torch.device,
    resume_enabled: bool,
    determinism: dict[str, object],
) -> dict[str, object]:
    gpu_model = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": metadata["run_id"],
        "repository_commit": metadata["repository_commit"],
        "config_path": metadata["config_path"],
        "config_sha256": metadata["config_sha256"],
        "dataset_repository": metadata["dataset_repository"],
        "dataset_revision": metadata["dataset_revision"],
        "dataset_archive_sha256": metadata["dataset_archive_sha256"],
        "class_count": metadata["num_classes"],
        "fold_count": metadata["num_folds"],
        "epochs": metadata["epochs"],
        "fine_tuning_epochs": metadata["fine_tuning_epochs"],
        "batch_size": metadata["batch_size"],
        "learning_rates": {
            "cnn": config["optimization"]["lr_cnn"],
            "transformer": config["optimization"]["lr_trans"],
        },
        "weight_decay": config["optimization"]["weight_decay"],
        "mixup": copy.deepcopy(config["mixup"]),
        "ema_decay": config["ema"]["decay"],
        "device_type": device.type,
        "gpu_model": gpu_model,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "determinism": copy.deepcopy(determinism),
        "start_time": _utc_timestamp(),
        "output_artifact_names": sorted(_EXPECTED_CANONICAL_ARTIFACTS),
        "resume_enabled": resume_enabled,
        "publication_policy": {
            "dataset": "NO",
            "weights": "NO",
            "checkpoints": "NO",
            "other_binary_artifacts": "NO",
        },
    }
    if "post_holdout" in metadata:
        manifest["post_holdout"] = copy.deepcopy(metadata["post_holdout"])
    return manifest


def _write_run_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_and_validate_manifest(
    path: Path,
    *,
    metadata: dict[str, object],
) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Could not load the trusted local run manifest.") from error
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Run manifest schema version does not match.")
    for state_field, manifest_field in (
        ("run_id", "run_id"),
        ("repository_commit", "repository_commit"),
        ("config_path", "config_path"),
        ("config_sha256", "config_sha256"),
        ("dataset_repository", "dataset_repository"),
        ("dataset_revision", "dataset_revision"),
        ("dataset_archive_sha256", "dataset_archive_sha256"),
        ("num_classes", "class_count"),
        ("num_folds", "fold_count"),
        ("epochs", "epochs"),
        ("fine_tuning_epochs", "fine_tuning_epochs"),
        ("batch_size", "batch_size"),
    ):
        if manifest.get(manifest_field) != metadata[state_field]:
            raise ValueError(
                "Run manifest does not match the current invocation: "
                f"{manifest_field}."
            )
    return manifest


def _build_state_metadata(
    *,
    run_id: str,
    config_path: Path,
    config: dict,
    label_names: list[str],
    repository_commit: str,
    dependencies: SimpleNamespace,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "repository_commit": repository_commit,
        "config_path": _portable_config_path(config_path),
        "config_sha256": _sha256_file(config_path),
        "dataset_repository": dependencies.dataset_repository,
        "dataset_revision": dependencies.dataset_revision,
        "dataset_archive_sha256": dependencies.dataset_archive_sha256,
        "label_names": list(label_names),
        "label_names_sha256": _sha256_label_names(label_names),
        "num_classes": len(label_names),
        "num_folds": config["cross_validation"]["n_splits"],
        "epochs": config["training"]["epochs"],
        "fine_tuning_epochs": config["fine_tuning"]["epochs"],
        "batch_size": config["training"]["batch_size"],
    }


def _move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def _restore_runtime_state(
    state: dict,
    *,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> None:
    model.load_state_dict(state["model_state_dict"])
    ema.load_state_dict(state["ema_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["grad_scaler_state_dict"])
    restore_rng_state(state)


def _load_training_dependencies() -> SimpleNamespace:
    """Import production training dependencies only when training is requested."""
    from src.datasets.folds import iter_stratified_folds, select_fold_datasets
    from src.datasets.postholdout import (
        cv_folds_from_manifest,
        load_frozen_postholdout_manifest,
        load_postholdout_cv_manifest,
        select_frozen_development_pool,
        sha256_json_identity_file,
    )
    from src.datasets.fruit_freshness import (
        DATASET_REPOSITORY_ID,
        DATASET_REVISION,
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
        cv_folds_from_manifest=cv_folds_from_manifest,
        build_cmt_classifier=build_cmt_classifier,
        build_finetune_transform=build_finetune_transform,
        build_fold_checkpoint_path=build_fold_checkpoint_path,
        build_fold_dataloaders=build_fold_dataloaders,
        build_optimizer=build_optimizer,
        build_scheduler=build_scheduler,
        build_train_transform=build_train_transform,
        build_validation_transform=build_validation_transform,
        compute_validation_metrics=compute_validation_metrics,
        dataset_archive_sha256=DATASET_ARCHIVE_SHA256,
        dataset_repository=DATASET_REPOSITORY_ID,
        dataset_revision=DATASET_REVISION,
        ensure_output_directory=ensure_output_directory,
        iter_stratified_folds=iter_stratified_folds,
        load_frozen_postholdout_manifest=load_frozen_postholdout_manifest,
        load_fruit_freshness_dataset=load_fruit_freshness_dataset,
        load_postholdout_cv_manifest=load_postholdout_cv_manifest,
        save_label_names=save_label_names,
        save_model_state=save_model_state,
        select_fold_datasets=select_fold_datasets,
        select_frozen_development_pool=select_frozen_development_pool,
        sha256_json_identity_file=sha256_json_identity_file,
        train_one_epoch=train_one_epoch,
        validate_one_epoch=validate_one_epoch,
    )


def _save_operational_state(
    *,
    state_path: Path,
    metadata: dict[str, object],
    status: str,
    current_fold: int,
    completed_epoch: int,
    next_fold: int,
    next_epoch: int,
    best_accuracy_current_fold: float,
    current_fold_history: dict[str, list[float]],
    completed_fold_histories: list[dict[str, list[float]]],
    completed_fold_accuracies: list[list[float]],
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    train_indices,
    validation_indices,
    created_at: str | None,
) -> dict:
    state = build_training_state(
        metadata=metadata,
        status=status,
        current_fold=current_fold,
        completed_epoch=completed_epoch,
        next_fold=next_fold,
        next_epoch=next_epoch,
        best_accuracy_current_fold=best_accuracy_current_fold,
        current_fold_history=current_fold_history,
        completed_fold_histories=completed_fold_histories,
        completed_fold_accuracies=completed_fold_accuracies,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_indices=train_indices,
        validation_indices=validation_indices,
        created_at=created_at,
    )
    save_training_state_atomic(state, state_path)
    return state



def prepare_training_dataset_and_folds(
    config: dict,
    dependencies: SimpleNamespace,
) -> tuple[object, list[tuple[np.ndarray, np.ndarray]], dict[str, object] | None]:
    """Select the canonical train route or the frozen post-holdout development route."""
    historical_dataset = dependencies.load_fruit_freshness_dataset()
    post_holdout = config.get("post_holdout")
    if post_holdout is None:
        training_dataset = historical_dataset["train"]
        folds = list(
            dependencies.iter_stratified_folds(
                training_dataset,
                n_splits=config["cross_validation"]["n_splits"],
                shuffle=config["cross_validation"]["shuffle"],
                random_state=config["cross_validation"]["random_state"],
            )
        )
        return training_dataset, folds, None

    split_path = _resolve_repository_path(post_holdout["split_manifest_path"])
    cv_path = _resolve_repository_path(post_holdout["cv_manifest_path"])
    frozen_manifest = dependencies.load_frozen_postholdout_manifest(split_path)
    training_dataset = dependencies.select_frozen_development_pool(
        historical_dataset["train"],
        historical_dataset["test"],
        frozen_manifest,
    )
    cv_manifest = dependencies.load_postholdout_cv_manifest(
        cv_path,
        development_manifest_sha256=dependencies.sha256_json_identity_file(split_path),
        development_count=frozen_manifest["development_count"],
    )
    folds = dependencies.cv_folds_from_manifest(cv_manifest)
    protocol = {
        "experiment_id": post_holdout["experiment_id"],
        "parent_experiment_id": post_holdout["parent_experiment_id"],
        "artifact_namespace": post_holdout["artifact_namespace"],
        "split_manifest_path": post_holdout["split_manifest_path"],
        "split_manifest_sha256": dependencies.sha256_json_identity_file(split_path),
        "cv_manifest_path": post_holdout["cv_manifest_path"],
        "cv_manifest_sha256": dependencies.sha256_json_identity_file(cv_path),
        "development_count": frozen_manifest["development_count"],
        "locked_test_count": frozen_manifest["locked_test_count"],
        "locked_test_model_access": "NO",
        "canonical_holdout_model_access": "NO",
    }
    return training_dataset, folds, protocol

def run_training(args: argparse.Namespace) -> dict:
    """Run the notebook-equivalent training flow with optional stateful resume."""
    config_path = _resolve_repository_path(args.config)
    output_directory = _resolve_repository_path(args.output_dir)
    options = _stateful_options(args)

    config = load_experiment_config(config_path)
    # Before resolve_device, because A_STRICT sets CUBLAS_WORKSPACE_CONFIG and
    # that variable is read when the cuBLAS handle is created. Applying it
    # after any CUDA work would be ignored rather than refused.
    determinism = resolve_policy(config)
    print("determinism:", determinism["level"], "seed:", determinism["seed"])

    device = resolve_device()
    print("device:", device)
    # Before any dataset preparation or model construction: a config whose
    # lineage names a registered ancestor must change exactly its one factor.
    experiment_validation = resolve_experiment_validation(config, config_path)
    if experiment_validation is not None:
        print(
            "Single-factor validation passed: "
            f"{sorted(experiment_validation['differences'])}"
        )
    if options["enabled"]:
        _prepare_stateful_output_directory(output_directory, options)

    dependencies = _load_training_dependencies()
    train_transform = dependencies.build_train_transform()
    validation_transform = dependencies.build_validation_transform()
    training_dataset, folds, post_holdout_protocol = prepare_training_dataset_and_folds(
        config,
        dependencies,
    )
    names = list(training_dataset.features["label"].names)

    state_path = output_directory / TRAINING_STATE_FILENAME
    manifest_path = output_directory / RUN_MANIFEST_FILENAME
    metadata = None
    resume_state = None
    if options["enabled"]:
        metadata = _build_state_metadata(
            run_id=options["run_id"],
            config_path=config_path,
            config=config,
            label_names=names,
            repository_commit=_repository_commit(),
            dependencies=dependencies,
        )
        if post_holdout_protocol is not None:
            metadata["post_holdout"] = post_holdout_protocol
        if options["resume_state"] is None:
            dependencies.save_label_names(names, str(output_directory))
            _write_run_manifest(
                manifest_path,
                _build_run_manifest(
                    metadata=metadata,
                    config=config,
                    device=device,
                    resume_enabled=True,
                    determinism=determinism,
                ),
            )
        else:
            _load_and_validate_manifest(manifest_path, metadata=metadata)
            resume_state = load_training_state(
                state_path,
                trusted_local=True,
                map_location=device,
                expected_metadata=metadata,
            )
    else:
        save_dir = dependencies.ensure_output_directory(str(output_directory))
        dependencies.save_label_names(names, save_dir)

    save_dir = str(output_directory)
    num_classes = len(names)
    train_labels = [int(label) for label in training_dataset["label"]]
    counts = Counter(train_labels)
    class_counts = [counts[index] for index in range(num_classes)]
    alpha = dependencies.build_class_balanced_alpha(
        class_counts,
        config["loss"]["class_balanced_beta"],
        num_classes,
    )
    print("alpha:", alpha.tolist())

    epochs = config["training"]["epochs"]
    fine_tuning_epochs = config["fine_tuning"]["epochs"]
    batch_size = config["training"]["batch_size"]
    num_folds = config["cross_validation"]["n_splits"]

    if len(folds) != num_folds:
        raise RuntimeError("The configured fold generator returned an unexpected count.")

    histories: list[dict[str, list[float]]] = []
    fold_accuracies: list[list[float]] = []
    resume_fold = 1
    resume_epoch = 1
    state_created_at = None
    if resume_state is not None:
        state_fold = resume_state["current_fold"]
        if not 1 <= state_fold <= num_folds:
            raise ValueError("Training state current_fold is outside configured folds.")
        stored_train_indices, stored_validation_indices = folds[state_fold - 1]
        validate_training_state(
            resume_state,
            expected_metadata=metadata,
            expected_train_indices=stored_train_indices,
            expected_validation_indices=stored_validation_indices,
        )
        histories = copy.deepcopy(resume_state["completed_fold_histories"])
        fold_accuracies = copy.deepcopy(resume_state["completed_fold_accuracies"])
        resume_fold = resume_state["next_fold"]
        resume_epoch = resume_state["next_epoch"]
        state_created_at = resume_state["created_at"]
        if resume_fold > num_folds:
            raise ValueError("Training state does not identify a remaining fold.")

    mixup_alpha = config["mixup"]["alpha"]
    mixup_probability = config["mixup"]["probability"]
    lr_cnn = config["optimization"]["lr_cnn"]
    lr_trans = config["optimization"]["lr_trans"]
    weight_decay = config["optimization"]["weight_decay"]
    ema_decay = config["ema"]["decay"]
    fine_tuning_transform = dependencies.build_finetune_transform()
    start_time = time.time()
    model = None
    last_train_indices = None
    last_validation_indices = None

    for fold, (train_indices, validation_indices) in enumerate(folds, 1):
        if fold < resume_fold:
            continue

        print(f"\n================ Fold {fold}/{num_folds} starting ================")
        fold_start = time.time()
        train_split, validation_split = dependencies.select_fold_datasets(
            training_dataset,
            train_indices,
            validation_indices,
        )
        train_dataset = dependencies.FruitHFDataset(
            train_split,
            transform=train_transform,
        )
        validation_dataset = dependencies.FruitHFDataset(
            validation_split,
            transform=validation_transform,
        )
        train_loader, validation_loader = dependencies.build_fold_dataloaders(
            train_dataset,
            validation_dataset,
            batch_size,
        )
        model = dependencies.build_cmt_classifier(num_classes).to(device)
        ema = dependencies.ModelEma(model, decay=ema_decay, device=device)
        if config["loss"]["use_ce_label_smoothing"]:
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=config["loss"]["label_smoothing"],
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
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        validation_accuracies: list[float] = []
        best_accuracy = 0.0
        first_epoch = 1
        restoring_current_fold = (
            resume_state is not None
            and resume_state["status"] == "RUNNING"
            and fold == resume_state["current_fold"]
        )
        if restoring_current_fold:
            _restore_runtime_state(
                resume_state,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            history = copy.deepcopy(resume_state["current_fold_history"])
            validation_accuracies = list(history["val_acc"])
            best_accuracy = resume_state["best_accuracy_current_fold"]
            first_epoch = resume_epoch

        for epoch in range(first_epoch, epochs + 1):
            epoch_start = time.time()
            is_finetuning = _is_finetuning_epoch(
                epoch,
                epochs,
                fine_tuning_epochs,
            )
            if is_finetuning:
                print(
                    f"Fold {fold} | Epoch {epoch} "
                    "[Fine-tuning: Mixup OFF, weak augmentation]",
                )
                train_dataset.tf = fine_tuning_transform
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
            validation_accuracy, validation_loss, predictions, labels, logits = (
                dependencies.validate_one_epoch(
                    ema.module,
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
            print(
                "Val (EMA) "
                f"acc: {validation_accuracy:.4f} | "
                f"f1: {validation_f1:.4f} | loss: {validation_loss:.4f}",
            )
            if validation_accuracy > best_accuracy + 1e-6:
                best_accuracy = validation_accuracy
                save_path = dependencies.build_fold_checkpoint_path(save_dir, fold)
                dependencies.save_model_state(ema.module, save_path)
                print(
                    "New best model (EMA) saved! "
                    f"(fold={fold}, acc={best_accuracy:.4f})",
                )
            scheduler.step()
            if options["enabled"] and epoch < epochs:
                saved_state = _save_operational_state(
                    state_path=state_path,
                    metadata=metadata,
                    status="RUNNING",
                    current_fold=fold,
                    completed_epoch=epoch,
                    next_fold=fold,
                    next_epoch=epoch + 1,
                    best_accuracy_current_fold=best_accuracy,
                    current_fold_history=history,
                    completed_fold_histories=histories,
                    completed_fold_accuracies=fold_accuracies,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    created_at=state_created_at,
                )
                state_created_at = saved_state["created_at"]
            print(f"Epoch {epoch} complete ({time.time() - epoch_start:.2f} seconds)")

        histories.append(history)
        fold_accuracies.append(validation_accuracies)
        if options["enabled"]:
            saved_state = _save_operational_state(
                state_path=state_path,
                metadata=metadata,
                status="FOLD_COMPLETE",
                current_fold=fold,
                completed_epoch=epochs,
                next_fold=fold + 1,
                next_epoch=1,
                best_accuracy_current_fold=best_accuracy,
                current_fold_history=history,
                completed_fold_histories=histories,
                completed_fold_accuracies=fold_accuracies,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                train_indices=train_indices,
                validation_indices=validation_indices,
                created_at=state_created_at,
            )
            state_created_at = saved_state["created_at"]
        print(f"Fold {fold} complete ({(time.time() - fold_start) / 60:.2f} minutes)")
        last_train_indices = train_indices
        last_validation_indices = validation_indices
        resume_state = None
        resume_epoch = 1

    if model is None or last_train_indices is None or last_validation_indices is None:
        raise RuntimeError("No remaining fold was available for training.")

    final_model_path = Path(save_dir) / config["checkpoint"]["final_model_filename"]
    dependencies.save_model_state(model, final_model_path)
    if options["enabled"]:
        _save_operational_state(
            state_path=state_path,
            metadata=metadata,
            status="COMPLETED",
            current_fold=num_folds,
            completed_epoch=epochs,
            next_fold=num_folds + 1,
            next_epoch=1,
            best_accuracy_current_fold=best_accuracy,
            current_fold_history=history,
            completed_fold_histories=histories,
            completed_fold_accuracies=fold_accuracies,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_indices=last_train_indices,
            validation_indices=last_validation_indices,
            created_at=state_created_at,
        )

    elapsed_seconds = time.time() - start_time
    print(f"Training complete ({elapsed_seconds / 60:.2f} minutes)")
    return {
        "elapsed_seconds": elapsed_seconds,
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
