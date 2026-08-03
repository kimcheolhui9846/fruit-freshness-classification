"""Trusted local operational state for epoch-boundary training resume."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any
from uuid import uuid4

import numpy as np
import torch


STATE_SCHEMA_VERSION = 1
_STATE_STATUSES = {"RUNNING", "FOLD_COMPLETE", "COMPLETED"}
_METADATA_FIELDS = (
    "run_id",
    "repository_commit",
    "config_path",
    "config_sha256",
    "dataset_repository",
    "dataset_revision",
    "dataset_archive_sha256",
    "label_names",
    "label_names_sha256",
    "num_classes",
    "num_folds",
    "epochs",
    "fine_tuning_epochs",
    "batch_size",
)
_STATE_FIELDS = (
    "schema_version",
    "status",
    *_METADATA_FIELDS,
    "current_fold",
    "completed_epoch",
    "next_fold",
    "next_epoch",
    "best_accuracy_current_fold",
    "current_fold_history",
    "completed_fold_histories",
    "completed_fold_accuracies",
    "model_state_dict",
    "ema_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "grad_scaler_state_dict",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "train_indices_sha256",
    "validation_indices_sha256",
    "created_at",
    "updated_at",
)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _hash_indices(indices) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state after an epoch has fully completed."""
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": cuda_states,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state only after the runtime objects are reconstructed."""
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())

    cuda_states = state.get("torch_cuda_rng_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in cuda_states])


def build_training_state(
    *,
    metadata: dict[str, Any],
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
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a versioned, self-contained trusted-local resume state."""
    if status not in _STATE_STATUSES:
        raise ValueError(f"Unsupported training-state status: {status}")

    missing_metadata = [field for field in _METADATA_FIELDS if field not in metadata]
    if missing_metadata:
        joined = ", ".join(missing_metadata)
        raise ValueError(f"Missing training-state metadata: {joined}")

    timestamp = _utc_timestamp()
    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": status,
        **{field: copy.deepcopy(metadata[field]) for field in _METADATA_FIELDS},
        "current_fold": current_fold,
        "completed_epoch": completed_epoch,
        "next_fold": next_fold,
        "next_epoch": next_epoch,
        "best_accuracy_current_fold": best_accuracy_current_fold,
        "current_fold_history": copy.deepcopy(current_fold_history),
        "completed_fold_histories": copy.deepcopy(completed_fold_histories),
        "completed_fold_accuracies": copy.deepcopy(completed_fold_accuracies),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "ema_state_dict": copy.deepcopy(ema.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
        "grad_scaler_state_dict": copy.deepcopy(scaler.state_dict()),
        "train_indices_sha256": _hash_indices(train_indices),
        "validation_indices_sha256": _hash_indices(validation_indices),
        "created_at": created_at or timestamp,
        "updated_at": timestamp,
    }
    state.update(capture_rng_state())
    return state


def validate_training_state(
    state: dict[str, Any],
    *,
    expected_metadata: dict[str, Any] | None = None,
    expected_train_indices=None,
    expected_validation_indices=None,
    allow_completed: bool = False,
) -> None:
    """Validate schema and immutable run identity before state is applied."""
    if not isinstance(state, dict):
        raise ValueError("Training state must be a dictionary.")

    missing_fields = [field for field in _STATE_FIELDS if field not in state]
    if missing_fields:
        raise ValueError(f"Training state missing required fields: {', '.join(missing_fields)}")

    if state["schema_version"] != STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported training-state schema_version: "
            f"{state['schema_version']}."
        )
    if state["status"] not in _STATE_STATUSES:
        raise ValueError(f"Unsupported training-state status: {state['status']}")
    if state["status"] == "COMPLETED" and not allow_completed:
        raise ValueError("A completed training state cannot be resumed.")

    if expected_metadata is not None:
        for field in _METADATA_FIELDS:
            if field not in expected_metadata:
                raise ValueError(f"Expected metadata missing field: {field}")
            if state[field] != expected_metadata[field]:
                raise ValueError(f"Training-state metadata mismatch for {field}.")

    if expected_train_indices is not None:
        expected_hash = _hash_indices(expected_train_indices)
        if state["train_indices_sha256"] != expected_hash:
            raise ValueError("Training-state metadata mismatch for train_indices_sha256.")

    if expected_validation_indices is not None:
        expected_hash = _hash_indices(expected_validation_indices)
        if state["validation_indices_sha256"] != expected_hash:
            raise ValueError(
                "Training-state metadata mismatch for validation_indices_sha256."
            )


def save_training_state_atomic(state: dict[str, Any], path: str | Path) -> Path:
    """Atomically replace a state file while preserving a prior valid state."""
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"Training-state directory does not exist: {target.parent}"
        )

    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.{uuid4().hex}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        torch.save(state, temporary_path)
        os.replace(temporary_path, target)
        return target
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def _is_remote_location(path: str | Path) -> bool:
    return isinstance(path, str) and "://" in path


def load_training_state(
    path: str | Path,
    *,
    trusted_local: bool = False,
    map_location: str | torch.device = "cpu",
    expected_metadata: dict[str, Any] | None = None,
    expected_train_indices=None,
    expected_validation_indices=None,
    allow_completed: bool = False,
) -> dict[str, Any]:
    """Load only a locally generated state after explicit trust acknowledgement."""
    if not trusted_local:
        raise PermissionError(
            "Training state may be loaded only from a trusted local file."
        )
    if _is_remote_location(path):
        raise ValueError("Remote training-state locations are not accepted.")

    state_path = Path(path)
    if not state_path.is_file():
        raise FileNotFoundError(f"Training state does not exist: {state_path}")

    try:
        state = torch.load(
            state_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(
            f"Could not load trusted training state: {state_path.name}"
        ) from error

    validate_training_state(
        state,
        expected_metadata=expected_metadata,
        expected_train_indices=expected_train_indices,
        expected_validation_indices=expected_validation_indices,
        allow_completed=allow_completed,
    )
    return state
