"""Freeze the Phase 9 post-holdout development and locked-test boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.datasets.postholdout import build_postholdout_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = Path("configs/splits/deep3-postholdout-research-01.json")
EXPERIMENT_ID = "deep3-postholdout-research-01"
PARENT_EXPERIMENT_ID = "deep3-canonical-reference-01"
EXPECTED_DATASET_NAME = "Densu341/Fresh-rotten-fruit"
EXPECTED_DATASET_REVISION = "2077850adc575aa1e8d6029e6cd6cefe9e403a1c"
EXPECTED_FILTERED_SIZE = 26858
EXPECTED_SOURCE_POOL_SIZE = 21486
EXPECTED_CANONICAL_HOLDOUT_SIZE = 5372
EXPECTED_CLASS_COUNT = 14
LOCKED_TEST_FRACTION = 0.20
SPLIT_SEED = 20260810


class DatasetIdentityMismatchError(RuntimeError):
    """Raised when the pinned canonical source cannot be reconstructed exactly."""


def build_parser() -> argparse.ArgumentParser:
    """Build the safe, parameter-free CLI for the pre-registered split."""
    parser = argparse.ArgumentParser(
        description="Freeze the approved Phase 9 post-holdout split exactly once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Frozen JSON manifest path, relative to the repository root by default.",
    )
    return parser


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPOSITORY_ROOT / candidate


def _sha256_int64(values) -> str:
    encoded = np.asarray(values, dtype="<i8").tobytes()
    return hashlib.sha256(encoded).hexdigest()


def class_names_from_raw_label_ids(label_feature, raw_label_ids) -> list[str]:
    """Resolve label names while normalizing NumPy scalar IDs for datasets."""
    return [label_feature.int2str(int(raw_label_id)) for raw_label_id in raw_label_ids]

def _class_count_map(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(class_names))
    if counts.size != len(class_names):
        raise ValueError("labels contain values outside the declared class names")
    return {name: int(count) for name, count in zip(class_names, counts, strict=True)}


def _validate_split(
    development_indices: np.ndarray,
    locked_test_indices: np.ndarray,
    source_pool_size: int,
) -> None:
    development = np.asarray(development_indices, dtype=np.int64)
    locked_test = np.asarray(locked_test_indices, dtype=np.int64)
    if development.size + locked_test.size != source_pool_size:
        raise ValueError("split counts do not cover the approved source pool size")
    if np.unique(development).size != development.size:
        raise ValueError("development indices contain duplicates")
    if np.unique(locked_test).size != locked_test.size:
        raise ValueError("locked-test indices contain duplicates")
    if np.intersect1d(development, locked_test).size:
        raise ValueError("development and locked-test indices overlap")
    combined = np.sort(np.concatenate((development, locked_test)))
    if not np.array_equal(combined, np.arange(source_pool_size, dtype=np.int64)):
        raise ValueError("split indices are not exhaustive source-pool positions")


def build_split_manifest(
    *,
    labels: np.ndarray,
    class_names: list[str],
    development_indices: np.ndarray,
    locked_test_indices: np.ndarray,
    repository_sha: str,
    locked_test_fraction: float,
    split_seed: int,
    canonical_holdout_size: int = EXPECTED_CANONICAL_HOLDOUT_SIZE,
) -> dict:
    """Build a portable manifest containing only label and index identities."""
    source_labels = np.asarray(labels, dtype=np.int64)
    development = np.asarray(development_indices, dtype=np.int64)
    locked_test = np.asarray(locked_test_indices, dtype=np.int64)
    _validate_split(development, locked_test, source_labels.size)
    if len(class_names) != np.unique(source_labels).size:
        raise ValueError("class names do not match the reconstructed source labels")

    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "dataset_name": EXPECTED_DATASET_NAME,
        "dataset_revision": EXPECTED_DATASET_REVISION,
        "source_pool_identity": "HISTORICAL_CANONICAL_TRAIN_ONLY",
        "source_pool_size": int(source_labels.size),
        "canonical_holdout_size": int(canonical_holdout_size),
        "canonical_holdout_overlap": 0,
        "protocol": "DEV_PLUS_LOCKED_TEST",
        "locked_test_fraction": locked_test_fraction,
        "split_seed": split_seed,
        "stratified": True,
        "index_identity": (
            "Zero-based positions relative to the reconstructed "
            "historical canonical training pool."
        ),
        "index_encoding": "little-endian signed 64-bit integer bytes before SHA-256",
        "class_names": list(class_names),
        "source_class_counts": _class_count_map(source_labels, class_names),
        "development_count": int(development.size),
        "locked_test_count": int(locked_test.size),
        "development_class_counts": _class_count_map(
            source_labels[development], class_names
        ),
        "locked_test_class_counts": _class_count_map(
            source_labels[locked_test], class_names
        ),
        "development_indices": development.tolist(),
        "locked_test_indices": locked_test.tolist(),
        "source_label_sequence_sha256": _sha256_int64(source_labels),
        "development_indices_sha256": _sha256_int64(development),
        "locked_test_indices_sha256": _sha256_int64(locked_test),
        "created_from_repository_sha": repository_sha,
        "primary_selection_metric": "Macro F1",
        "internal_cv": {"n_splits": 3, "shuffle": True, "random_state": 42},
        "canonical_holdout_usage": "HISTORICAL_EVIDENCE_ONLY",
        "locked_test_status": "FROZEN_UNOBSERVED_BY_MODEL",
        "model_training": "NO",
        "model_evaluation": "NO",
        "locked_test_model_evaluation": "NO",
    }


def write_split_manifest(output_path: Path, manifest: dict) -> None:
    """Write one frozen manifest, refusing any overwrite or regeneration."""
    if output_path.exists():
        raise FileExistsError(f"Frozen split manifest already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _repository_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _reconstruct_canonical_pool_with_images():
    """Reconstruct the canonical training split, its labels, and its class names.

    Single definition of the filter, the 0.2/seed-42 canonical split, and every
    identity check. Returning the split object decodes no image; `datasets`
    decodes a row only when it is indexed.
    """
    from datasets import load_dataset

    from src.datasets.fruit_freshness import (
        DATASET_ARCHIVE_FILENAME,
        DATASET_REPOSITORY_ID,
        DATASET_REVISION,
        _resolve_imagefolder_data_dir,
    )

    if (
        DATASET_REPOSITORY_ID != EXPECTED_DATASET_NAME
        or DATASET_REVISION != EXPECTED_DATASET_REVISION
        or DATASET_ARCHIVE_FILENAME != "freshness_fruit.zip"
    ):
        raise DatasetIdentityMismatchError("Pinned canonical dataset identity does not match Phase 9.2")

    dataset = load_dataset("imagefolder", data_dir=str(_resolve_imagefolder_data_dir()))
    raw_labels = np.asarray(dataset["train"]["label"], dtype=np.int64)
    remove_labels = np.asarray([18, 20, 16, 13, 2, 5, 7, 9], dtype=np.int64)
    clean = dataset["train"].select(np.flatnonzero(~np.isin(raw_labels, remove_labels)))
    if len(clean) != EXPECTED_FILTERED_SIZE:
        raise DatasetIdentityMismatchError(
            f"Filtered canonical dataset size mismatch: {len(clean)} != {EXPECTED_FILTERED_SIZE}"
        )

    canonical_split = clean.train_test_split(test_size=0.2, seed=42)
    canonical_train = canonical_split["train"]
    canonical_holdout = canonical_split["test"]
    if len(canonical_train) != EXPECTED_SOURCE_POOL_SIZE:
        raise DatasetIdentityMismatchError(
            "Canonical training size mismatch: "
            f"{len(canonical_train)} != {EXPECTED_SOURCE_POOL_SIZE}"
        )
    if len(canonical_holdout) != EXPECTED_CANONICAL_HOLDOUT_SIZE:
        raise DatasetIdentityMismatchError(
            "Canonical holdout size mismatch: "
            f"{len(canonical_holdout)} != {EXPECTED_CANONICAL_HOLDOUT_SIZE}"
        )

    source_raw_labels = np.asarray(canonical_train["label"], dtype=np.int64)
    holdout_raw_labels = np.asarray(canonical_holdout["label"], dtype=np.int64)
    raw_label_ids = sorted(set(source_raw_labels) | set(holdout_raw_labels))
    class_names = class_names_from_raw_label_ids(
        canonical_train.features["label"],
        raw_label_ids,
    )
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise DatasetIdentityMismatchError(
            f"Canonical class count mismatch: {len(class_names)} != {EXPECTED_CLASS_COUNT}"
        )
    remap = {raw_label_id: index for index, raw_label_id in enumerate(raw_label_ids)}
    source_labels = np.asarray(
        [remap[raw_label_id] for raw_label_id in source_raw_labels],
        dtype=np.int64,
    )
    return canonical_train, source_labels, class_names, len(canonical_holdout)


def _reconstruct_canonical_source_pool() -> tuple[np.ndarray, list[str], int]:
    """Reconstruct canonical split labels without requesting image samples."""
    _, source_labels, class_names, holdout_size = _reconstruct_canonical_pool_with_images()
    return source_labels, class_names, holdout_size


def freeze_postholdout_split(output_path: Path) -> dict:
    """Materialize the sole approved Phase 9.2 split and freeze its identity."""
    if output_path.exists():
        raise FileExistsError(f"Frozen split manifest already exists: {output_path}")

    source_labels, class_names, canonical_holdout_size = _reconstruct_canonical_source_pool()
    development_indices, locked_test_indices = build_postholdout_split(
        source_labels,
        locked_test_fraction=LOCKED_TEST_FRACTION,
        random_state=SPLIT_SEED,
    )
    manifest = build_split_manifest(
        labels=source_labels,
        class_names=class_names,
        development_indices=development_indices,
        locked_test_indices=locked_test_indices,
        repository_sha=_repository_sha(),
        locked_test_fraction=LOCKED_TEST_FRACTION,
        split_seed=SPLIT_SEED,
        canonical_holdout_size=canonical_holdout_size,
    )
    write_split_manifest(output_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Print the approved parameters, then materialize the frozen manifest once."""
    args = build_parser().parse_args(argv)
    output_path = _resolve_repository_path(args.output)
    print("OWNER_PHASE_9_2_APPROVAL: APPROVED")
    print("protocol: DEV_PLUS_LOCKED_TEST")
    print("source size: 21486")
    print("locked fraction: 0.20")
    print("seed: 20260810")
    print("model training: NO")
    print("model evaluation: NO")
    try:
        manifest = freeze_postholdout_split(output_path)
    except DatasetIdentityMismatchError as error:
        print("PHASE_9_2_SPLIT_STATUS: BLOCKED_DATA_IDENTITY_MISMATCH")
        print(f"reason: {error}")
        return 2
    print(f"frozen manifest: {output_path}")
    print(f"development count: {manifest['development_count']}")
    print(f"locked-test count: {manifest['locked_test_count']}")
    print(f"development indices sha256: {manifest['development_indices_sha256']}")
    print(f"locked-test indices sha256: {manifest['locked_test_indices_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())