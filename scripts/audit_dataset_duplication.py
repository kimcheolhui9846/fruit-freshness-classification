"""Report byte-identical duplicate images and how they cross a split boundary.

The source dataset stores the same file under both its `Train` and `Test`
directories, and the project's loader concatenates the two before splitting by
row. A random row split therefore scatters copies of one image across both
sides of every boundary derived from it.

This script measures that. It loads no model, trains nothing, and consumes no
GPU. It reads image files and compares SHA-256 digests of their bytes, so a
"duplicate" here means the files are identical, not merely similar.

The finding is recorded in docs/dataset-duplication-audit.md.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def hash_file(path: Path) -> str:
    """Digest a file's bytes, not its decoded pixels.

    Byte equality is the strict reading: two files that decode to the same
    picture through different encodings are not counted here, so every match
    this reports is beyond argument.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_image_hashes(root: Path, classes: set[str] | None = None) -> dict[str, list[Path]]:
    """Map each digest to every file carrying it, optionally limited to classes."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if classes is not None and path.parent.name not in classes:
            continue
        groups[hash_file(path)].append(path)
    return dict(groups)


def duplicate_summary(groups: dict[str, list[Path]]) -> dict:
    """Describe how much of a file collection is redundant."""
    duplicated = {digest: paths for digest, paths in groups.items() if len(paths) > 1}
    extra_copies = sum(len(paths) - 1 for paths in duplicated.values())
    cross_class = sum(
        1 for paths in duplicated.values() if len({p.parent.name for p in paths}) > 1
    )
    per_class: dict[str, int] = defaultdict(int)
    for paths in duplicated.values():
        for path in paths[1:]:
            per_class[path.parent.name] += 1
    return {
        "files": sum(len(paths) for paths in groups.values()),
        "unique_images": len(groups),
        "duplicate_groups": len(duplicated),
        "extra_copies": extra_copies,
        # A group spanning two class directories would mean contradictory
        # labels on identical pixels, which is a different and worse problem.
        "cross_class_groups": cross_class,
        "extra_copies_per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
    }


def split_contamination(train_digests, evaluation_digests) -> dict:
    """Count evaluation rows that are byte-identical to some training row.

    This is the quantity that matters for a leak: not how many duplicates
    exist, but how many of them land on opposite sides of a boundary that is
    supposed to separate what the model saw from what it is scored on.
    """
    train_set = set(train_digests)
    evaluation = list(evaluation_digests)
    contaminated = [digest for digest in evaluation if digest in train_set]
    return {
        "train_rows": len(list(train_digests)),
        "evaluation_rows": len(evaluation),
        "contaminated_rows": len(contaminated),
        "contaminated_fraction": (
            len(contaminated) / len(evaluation) if evaluation else 0.0
        ),
        "distinct_images_on_both_sides": len(set(contaminated)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report duplicate images in an image-folder dataset.",
    )
    parser.add_argument("--root", required=True, help="Image-folder dataset root.")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Limit to these class directories; omit to scan all.",
    )
    parser.add_argument("--output", required=True, help="Audit record JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Audit record already exists: {output_path}.")

    classes = set(args.classes) if args.classes else None
    groups = collect_image_hashes(Path(args.root), classes)
    summary = duplicate_summary(groups)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"files {summary['files']}, unique {summary['unique_images']}, "
        f"duplicate groups {summary['duplicate_groups']}, "
        f"extra copies {summary['extra_copies']}, "
        f"cross-class groups {summary['cross_class_groups']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
