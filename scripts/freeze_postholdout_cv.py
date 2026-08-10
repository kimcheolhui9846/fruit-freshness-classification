"""Freeze the approved Phase 9.3 development-CV identity exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets.postholdout import (
    build_postholdout_cv_manifest,
    load_frozen_postholdout_manifest,
    select_frozen_development_pool,
    sha256_json_identity_file,
)
from src.utils.config import load_experiment_config, validate_postholdout_baseline_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/deep3_postholdout_baseline.toml")
DEFAULT_OUTPUT = Path("configs/splits/deep3-postholdout-research-01-baseline-cv.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the approved Phase 9.3 development-CV identity exactly once.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate



def write_cv_manifest(path: Path, manifest: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Frozen CV manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def freeze_cv_manifest(config_path: str | Path, output_path: str | Path) -> Path:
    config_path = _resolve_repository_path(config_path)
    output_path = _resolve_repository_path(output_path)
    validation = validate_postholdout_baseline_config(
        REPOSITORY_ROOT / "configs/deep3_canonical.toml",
        config_path,
    )
    if not validation["recipe_equivalent"]:
        raise ValueError("Post-holdout baseline recipe equivalence failed.")
    config = load_experiment_config(config_path)
    protocol = config["post_holdout"]
    split_path = _resolve_repository_path(protocol["split_manifest_path"])
    manifest = load_frozen_postholdout_manifest(split_path)

    from src.datasets.fruit_freshness import load_fruit_freshness_dataset

    historical_dataset = load_fruit_freshness_dataset()
    development = select_frozen_development_pool(
        historical_dataset["train"],
        historical_dataset["test"],
        manifest,
    )
    cv_manifest = build_postholdout_cv_manifest(
        development["label"],
        experiment_id=protocol["experiment_id"],
        parent_experiment_id=protocol["parent_experiment_id"],
        development_manifest_sha256=sha256_json_identity_file(split_path),
        n_splits=config["cross_validation"]["n_splits"],
        shuffle=config["cross_validation"]["shuffle"],
        random_state=config["cross_validation"]["random_state"],
    )
    if output_path != _resolve_repository_path(protocol["cv_manifest_path"]):
        raise ValueError("CV manifest output must match the approved baseline config.")
    write_cv_manifest(output_path, cv_manifest)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    freeze_cv_manifest(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())