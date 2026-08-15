"""Describe how much the freshpotato error set moves between identical runs.

Phase 9.8, exploratory and descriptive. The variance decomposition in
docs/postholdout-measurement-floor-protocol.md shows that this one class
accounts for 90.56% of Macro F1's run-to-run variance. This script asks the
next question: are the images it gets wrong the same images each run, or
different ones?

It loads no model, touches no dataset, and consumes no GPU. It reads
development OOF prediction files that already exist. No rule about what its
output means is frozen anywhere, because none was chosen before looking, so
it can support no claim and advance no candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


FRESHPOTATO_INDEX = 5


def error_positions(labels, predictions, class_index: int) -> np.ndarray:
    """Development positions where this class's examples were misclassified.

    Examples of *other* classes predicted as this one are excluded: they are
    errors, but not errors of this class, and counting them would conflate
    recall failures with precision failures.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")
    return np.flatnonzero((labels == class_index) & (predictions != class_index))


def compare_error_sets(runs: dict, class_index: int = FRESHPOTATO_INDEX) -> dict:
    """Describe how far the error set moves across runs of the same recipe."""
    if len(runs) < 2:
        raise ValueError("Comparing error sets needs at least two runs.")

    names = list(runs)
    reference_labels = np.asarray(runs[names[0]][0])
    for name in names[1:]:
        if not np.array_equal(np.asarray(runs[name][0]), reference_labels):
            raise ValueError(
                f"Run {name!r} has different labels from {names[0]!r}; "
                "runs that disagree about which examples belong to the class "
                "are not comparable."
            )

    sets = {
        name: set(error_positions(labels, predictions, class_index).tolist())
        for name, (labels, predictions) in runs.items()
    }
    union = set().union(*sets.values())
    intersection = set(sets[names[0]])
    for name in names[1:]:
        intersection &= sets[name]

    counts = {position: 0 for position in union}
    for positions in sets.values():
        for position in positions:
            counts[position] += 1
    histogram = {
        str(k): sum(1 for value in counts.values() if value == k)
        for k in range(1, len(names) + 1)
    }

    return {
        "class_index": class_index,
        "class_support": int(np.count_nonzero(reference_labels == class_index)),
        "run_names": names,
        "per_run_error_counts": {
            name: len(positions) for name, positions in sets.items()
        },
        "stable_error_count": len(intersection),
        "union_error_count": len(union),
        # An empty union means no run erred at all, which is total agreement.
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "error_frequency_histogram": histogram,
    }


def load_run(path: Path) -> tuple:
    archive = np.load(Path(path))
    return archive["labels"], archive["predictions"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe error-set movement for one class across runs.",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Two or more development OOF prediction .npz files.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=FRESHPOTATO_INDEX,
        help="Frozen label index; freshpotato is 5.",
    )
    parser.add_argument("--output", required=True, help="Descriptive record JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Record already exists: {output_path}.")

    runs = {}
    for path in args.predictions:
        resolved = Path(path)
        name = resolved.parent.name or resolved.stem
        runs[name] = load_run(resolved)
    if len(runs) != len(args.predictions):
        raise ValueError("Prediction files must have distinguishable run names.")

    result = compare_error_sets(runs, class_index=args.class_index)
    result.update(
        {
            "status": "EXPLORATORY_DESCRIPTIVE",
            "may_advance_a_candidate": False,
            "may_support_a_claim": False,
        }
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"class {result['class_index']}: support {result['class_support']}, "
        f"errors {result['per_run_error_counts']}, "
        f"stable {result['stable_error_count']}, union {result['union_error_count']}, "
        f"jaccard {result['jaccard']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
