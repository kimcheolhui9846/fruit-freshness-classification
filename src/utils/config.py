"""Standard-library loader for the active deep3 experiment configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path, PureWindowsPath
import tomllib


_REQUIRED_TYPES = {
    "runtime": {"cudnn_benchmark": bool},
    "loss": {
        "class_balanced_beta": float,
        "use_ce_label_smoothing": bool,
        "label_smoothing": float,
        "focal_gamma": float,
    },
    "training": {"epochs": int, "batch_size": int},
    "fine_tuning": {"epochs": int},
    "cross_validation": {
        "n_splits": int,
        "shuffle": bool,
        "random_state": int,
    },
    "mixup": {"alpha": float, "probability": float},
    "optimization": {
        "lr_cnn": float,
        "lr_trans": float,
        "weight_decay": float,
    },
    "ema": {"decay": float},
    "checkpoint": {"final_model_filename": str},
    "reporting": {"figure_size": list},
}


def load_experiment_config(path: str | Path) -> dict:
    """Load and validate an explicitly supplied TOML experiment configuration."""
    with Path(path).open("rb") as file:
        config = tomllib.load(file)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    for section_name, keys in _REQUIRED_TYPES.items():
        section = _require_section(config, section_name)
        for key, expected_type in keys.items():
            value = _require_key(section, section_name, key)
            if type(value) is not expected_type:
                raise TypeError(
                    f"Configuration value [{section_name}].{key} must be "
                    f"{expected_type.__name__}, got {type(value).__name__}."
                )

    _require_positive(config["training"]["epochs"], "[training].epochs")
    _require_positive(config["training"]["batch_size"], "[training].batch_size")
    _require_positive(config["fine_tuning"]["epochs"], "[fine_tuning].epochs")
    _require_positive(config["cross_validation"]["n_splits"], "[cross_validation].n_splits")
    _require_positive(config["cross_validation"]["random_state"], "[cross_validation].random_state")
    _require_positive(config["optimization"]["lr_cnn"], "[optimization].lr_cnn")
    _require_positive(config["optimization"]["lr_trans"], "[optimization].lr_trans")
    _require_positive(config["optimization"]["weight_decay"], "[optimization].weight_decay")
    _require_positive(config["mixup"]["alpha"], "[mixup].alpha", allow_zero=True)
    _require_probability(config["mixup"]["probability"], "[mixup].probability")
    _require_probability(config["loss"]["label_smoothing"], "[loss].label_smoothing")
    _require_positive(config["loss"]["focal_gamma"], "[loss].focal_gamma", allow_zero=True)
    _require_open_unit_interval(config["loss"]["class_balanced_beta"], "[loss].class_balanced_beta")
    _require_open_unit_interval(config["ema"]["decay"], "[ema].decay")
    _validate_filename(config["checkpoint"]["final_model_filename"])
    _validate_figure_size(config["reporting"]["figure_size"])
    if "post_holdout" in config:
        _validate_postholdout_section(config["post_holdout"])


def _require_section(config: dict, section_name: str) -> dict:
    try:
        section = config[section_name]
    except KeyError as error:
        raise KeyError(f"Missing configuration section: [{section_name}]") from error
    if type(section) is not dict:
        raise TypeError(f"Configuration section [{section_name}] must be a table.")
    return section


def _require_key(section: dict, section_name: str, key: str):
    try:
        return section[key]
    except KeyError as error:
        raise KeyError(f"Missing configuration value: [{section_name}].{key}") from error


def _require_positive(value: int | float, name: str, *, allow_zero: bool = False) -> None:
    if value < 0 or (value == 0 and not allow_zero):
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"Configuration value {name} must be {comparison}.")


def _require_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Configuration value {name} must be between 0.0 and 1.0.")


def _require_open_unit_interval(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"Configuration value {name} must be between 0.0 and 1.0 exclusively.")


def _validate_filename(value: str) -> None:
    if not value or Path(value).name != value or Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("[checkpoint].final_model_filename must be a portable filename.")


def _validate_figure_size(value: list) -> None:
    if len(value) != 2 or any(type(dimension) is not int or dimension <= 0 for dimension in value):
        raise ValueError("[reporting].figure_size must be a two-item list of positive integers.")


def _validate_postholdout_section(section: dict) -> None:
    required = {
        "experiment_id": str,
        "parent_experiment_id": str,
        "split_manifest_path": str,
        "cv_manifest_path": str,
        "artifact_namespace": str,
    }
    if type(section) is not dict:
        raise TypeError("Configuration section [post_holdout] must be a table.")
    for key, expected_type in required.items():
        value = _require_key(section, "post_holdout", key)
        if type(value) is not expected_type or not value:
            raise TypeError(f"Configuration value [post_holdout].{key} must be a non-empty string.")
    for key in ("split_manifest_path", "cv_manifest_path"):
        _validate_repository_relative_path(section[key], f"[post_holdout].{key}")
    for key in ("experiment_id", "parent_experiment_id", "artifact_namespace"):
        if any(character in section[key] for character in "\\/:"):
            raise ValueError(f"Configuration value [post_holdout].{key} must be portable.")


def _validate_repository_relative_path(value: str, name: str) -> None:
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in path.parts:
        raise ValueError(f"Configuration value {name} must be repository-relative.")


def baseline_recipe_differences(canonical: dict, baseline: dict) -> dict:
    """Return every semantic difference between canonical and baseline configs."""
    differences = {}
    for section in sorted(set(canonical) | set(baseline)):
        canonical_value = canonical.get(section)
        baseline_value = baseline.get(section)
        if canonical_value == baseline_value:
            continue
        if section == "post_holdout" and section not in canonical:
            differences[section] = baseline_value
        else:
            differences[section] = {
                "canonical": canonical_value,
                "baseline": baseline_value,
            }
    return differences


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_postholdout_baseline_config(
    canonical_path: str | Path,
    baseline_path: str | Path,
) -> dict[str, object]:
    """Validate that a Phase 9 baseline changes only protocol identity fields."""
    canonical_resolved = Path(canonical_path)
    baseline_resolved = Path(baseline_path)
    canonical = load_experiment_config(canonical_resolved)
    baseline = load_experiment_config(baseline_resolved)
    differences = baseline_recipe_differences(canonical, baseline)
    if set(differences) != {"post_holdout"}:
        raise ValueError("Post-holdout baseline recipe differs from canonical values.")
    return {
        "recipe_equivalent": True,
        "canonical_config_sha256": _sha256_file(canonical_resolved),
        "baseline_config_sha256": _sha256_file(baseline_resolved),
        "allowed_differences": differences,
    }