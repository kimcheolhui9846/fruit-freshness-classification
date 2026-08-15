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

# Phase 9.7 determinism level names. They live here, not in
# src/utils/determinism.py, because this module is a standard-library-only
# loader and must validate the name without importing torch or numpy.
A_STRICT = "A_STRICT"
B_CUDNN = "B_CUDNN"
C_SEED_ONLY = "C_SEED_ONLY"
DETERMINISM_LEVELS = (A_STRICT, B_CUDNN, C_SEED_ONLY)
_CUDNN_CONSTRAINED_LEVELS = (A_STRICT, B_CUDNN)


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
    _validate_runtime_determinism(config["runtime"])
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


def _validate_runtime_determinism(runtime: dict) -> None:
    """Validate the optional Phase 9.7 determinism keys.

    Both keys are optional so that every configuration frozen before Phase 9.7
    keeps its recorded SHA-256 and its original meaning. They must appear
    together: a seed with no level, or a level with no seed, describes a
    behaviour nobody can read off the file.
    """
    has_seed = "seed" in runtime
    has_level = "determinism_level" in runtime
    if has_seed != has_level:
        raise ValueError(
            "[runtime].seed and [runtime].determinism_level must be set together."
        )
    if not has_seed:
        return

    seed = runtime["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("[runtime].seed must be a non-negative integer.")

    level = runtime["determinism_level"]
    if type(level) is not str or level not in DETERMINISM_LEVELS:
        expected = ", ".join(DETERMINISM_LEVELS)
        raise ValueError(f"[runtime].determinism_level must be one of {expected}.")

    if level in _CUDNN_CONSTRAINED_LEVELS and runtime["cudnn_benchmark"]:
        raise ValueError(
            f"[runtime].determinism_level {level} requires cudnn_benchmark = false; "
            "the cuDNN autotuner selects kernels nondeterministically."
        )


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

LOSS001_ALLOWED_DIFFERENCES = frozenset(
    {
        "loss.class_balanced_beta",
        "post_holdout.experiment_id",
        "post_holdout.parent_experiment_id",
        "post_holdout.artifact_namespace",
    }
)


def flatten_experiment_config(config: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a config to dotted keys so a single key can be compared exactly.

    `baseline_recipe_differences` compares whole sections, which suffices when
    the only permitted change is the presence of a `post_holdout` section. A
    loss experiment changes one key *inside* `[loss]`, and a section-level
    check would accept any other change in that same section.
    """
    flat: dict[str, object] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_experiment_config(value, path))
        else:
            flat[path] = value
    return flat


def validate_loss_experiment_config(
    baseline_path: str | Path,
    experiment_path: str | Path,
    *,
    allowed_differences: frozenset[str],
    expected_values: dict[str, object],
) -> dict[str, object]:
    """Verify an experiment config changes exactly its registered factor.

    The name records the first caller, the Phase 9.6 loss experiment. The
    function is lineage-generic and is also used by the Phase 9.7 determinism
    check. It is not renamed because docs/postholdout-loss001-runbook.md
    records the Phase 9.6 implementation and names it throughout.
    """
    baseline_resolved = Path(baseline_path)
    experiment_resolved = Path(experiment_path)
    baseline = flatten_experiment_config(load_experiment_config(baseline_resolved))
    experiment = flatten_experiment_config(load_experiment_config(experiment_resolved))

    differing = {
        key
        for key in set(baseline) | set(experiment)
        if baseline.get(key) != experiment.get(key)
    }
    unexpected = sorted(differing - allowed_differences)
    if unexpected:
        raise ValueError(
            "Experiment config changes fields outside its registered factor: "
            f"{', '.join(unexpected)}."
        )
    missing = sorted(allowed_differences - differing)
    if missing:
        raise ValueError(
            "Experiment config does not differ from the baseline where it must: "
            f"{', '.join(missing)}."
        )
    for key, expected in expected_values.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"Experiment config {key} is {experiment.get(key)!r}, "
                f"but the frozen protocol pins it to {expected!r}."
            )

    return {
        "single_factor_verified": True,
        "baseline_config_sha256": _sha256_file(baseline_resolved),
        "experiment_config_sha256": _sha256_file(experiment_resolved),
        "differences": {
            key: {"baseline": baseline.get(key), "experiment": experiment.get(key)}
            for key in sorted(differing)
        },
    }


BASELINE_EXPERIMENT_ID = "deep3-postholdout-research-01-baseline"
RESEARCH_PARENT_ID = "deep3-postholdout-research-01"
BASELINE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "deep3_postholdout_baseline.toml"
)
LOSS001_EXPECTED_VALUES = {"loss.class_balanced_beta": 0.9999}

DETERMINISM_CHECK_PARENT_ID = "deep3-postholdout-determinism-check"
DETERMINISM_CHECK_ALLOWED_DIFFERENCES = frozenset(
    {
        "runtime.cudnn_benchmark",
        "runtime.seed",
        "runtime.determinism_level",
        "training.epochs",
        "fine_tuning.epochs",
        "post_holdout.experiment_id",
        "post_holdout.parent_experiment_id",
        "post_holdout.artifact_namespace",
    }
)
# The level is deliberately absent: the frozen ladder descends from A_STRICT to
# B_CUDNN, and pinning it here would make the registered descent unexecutable.
DETERMINISM_CHECK_EXPECTED_VALUES = {
    "runtime.seed": 20260815,
    "runtime.cudnn_benchmark": False,
    "training.epochs": 2,
    "fine_tuning.epochs": 1,
    "post_holdout.split_manifest_path": (
        "configs/splits/deep3-postholdout-research-01.json"
    ),
    "post_holdout.cv_manifest_path": (
        "configs/splits/deep3-postholdout-research-01-baseline-cv.json"
    ),
}


def resolve_experiment_validation(config: dict, config_path: str | Path) -> dict | None:
    """Validate a post-holdout config against the right ancestor, by lineage.

    Returns None for configs parented to the research identity, which keep the
    existing canonical comparison at their call sites. An unrecognized parent
    raises rather than falling through unchecked: a config that names no known
    ancestor has no registered factor to be held to, and silently skipping the
    check is how an unregistered experiment would get to run.

    `config_path` is separate because `load_experiment_config` returns the
    parsed mapping without recording where it read from, and the validator
    needs the file itself to hash it.
    """
    post_holdout = config.get("post_holdout")
    if post_holdout is None:
        return None
    parent = post_holdout.get("parent_experiment_id")
    if parent == RESEARCH_PARENT_ID:
        return None
    if parent == BASELINE_EXPERIMENT_ID:
        return validate_loss_experiment_config(
            BASELINE_CONFIG_PATH,
            config_path,
            allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
            expected_values=LOSS001_EXPECTED_VALUES,
        )
    if parent == DETERMINISM_CHECK_PARENT_ID:
        return validate_loss_experiment_config(
            BASELINE_CONFIG_PATH,
            config_path,
            allowed_differences=DETERMINISM_CHECK_ALLOWED_DIFFERENCES,
            expected_values=DETERMINISM_CHECK_EXPECTED_VALUES,
        )
    raise ValueError(
        f"Config declares an unregistered parent experiment {parent!r}; "
        "every experiment must name the ancestor its single factor is measured against."
    )
