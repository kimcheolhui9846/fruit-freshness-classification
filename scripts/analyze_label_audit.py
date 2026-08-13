"""Unblind the Phase 9.5 audit and apply the frozen decision rule.

This is the only reader of the sealed key. No model is constructed or loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.datasets.label_audit import JUDGMENT_CATEGORIES, apply_decision_rule, score_reviewer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unblind and score the Phase 9.5 label quality audit."
    )
    parser.add_argument("--key", required=True, help="Sealed review_set_key.json.")
    parser.add_argument(
        "--reviewer",
        action="append",
        required=True,
        help="Reviewer judgment CSV. Pass exactly twice.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for findings artifacts.")
    parser.add_argument(
        "--baseline-predictions",
        default="results/deep3-postholdout-research-01-baseline/development_oof_predictions.npz",
        help="Stored Phase 9.4 OOF predictions, read for a diagnostic comparison only.",
    )
    parser.add_argument(
        "--label-names",
        default="weights/deep3-postholdout-research-01-baseline/label_names.json",
        help="Class ordering used by the stored predictions.",
    )
    parser.add_argument(
        "--split-manifest",
        default="configs/splits/deep3-postholdout-research-01.json",
        help="Frozen split manifest, for mapping source indices to prediction rows.",
    )
    return parser


def load_label_names(path: Path) -> list[str]:
    """Class ordering is derived, never hardcoded, so a reordering cannot pass silently."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload["label_names"] if isinstance(payload, dict) else payload
    if len(names) != 14:
        raise ValueError(f"Expected 14 class names, found {len(names)}.")
    return list(names)


def model_agreement(
    predictions_path: Path,
    development_indices: np.ndarray,
    entries: list[dict],
    judgments: dict[int, str],
    rotten_label_index: int,
) -> dict:
    """Compare reviewer calls against stored predictions. Reads a file; runs nothing.

    Diagnostic only: this never feeds the decision rule, which depends on
    reviewer judgments alone.
    """
    stored = np.load(predictions_path)
    predicted = stored["predictions"]
    position_in_development = {
        int(source): row for row, source in enumerate(development_indices.tolist())
    }

    agree = 0
    compared = 0
    for entry in entries:
        if entry["group"] != "SUBJECT":
            continue
        call = judgments[entry["position"]]
        if call not in ("FRESH", "ROTTEN"):
            continue
        row = position_in_development[int(entry["source_index"])]
        model_says_rotten = bool(predicted[row] == rotten_label_index)
        compared += 1
        agree += int(model_says_rotten == (call == "ROTTEN"))

    return {
        "compared": compared,
        "agreement": (agree / compared) if compared else None,
    }


def load_judgments(handle) -> dict[int, str]:
    """Read one reviewer CSV. A blank cell is an incomplete review, not a category."""
    judgments: dict[int, str] = {}
    for row in csv.DictReader(handle):
        position = int(row["position"])
        call = (row["judgment"] or "").strip().upper()
        if not call:
            raise ValueError(f"Position {position:03d} has no judgment.")
        if call not in JUDGMENT_CATEGORIES:
            raise ValueError(f"Position {position:03d} has unknown category {call!r}.")
        judgments[position] = call
    return judgments


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.reviewer) != 2:
        raise SystemExit("The protocol fixes two independent reviewers.")

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    entries = key["entries"]
    subject_positions = np.array(
        [e["position"] for e in entries if e["group"] == "SUBJECT"], dtype=np.int64
    )
    control_positions = np.array(
        [e["position"] for e in entries if e["group"] == "CONTROL"], dtype=np.int64
    )

    reviewers = []
    for path in args.reviewer:
        with Path(path).open(encoding="utf-8") as handle:
            reviewers.append(load_judgments(handle))

    scores = [score_reviewer(j, subject_positions, control_positions) for j in reviewers]
    decision = apply_decision_rule(scores)

    positions = [e["position"] for e in entries]
    first = [reviewers[0][p] for p in positions]
    second = [reviewers[1][p] for p in positions]
    agreement = float(np.mean([a == b for a, b in zip(first, second)]))
    kappa = float(cohen_kappa_score(first, second))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = load_label_names(REPOSITORY_ROOT / args.label_names)
    rotten_label_index = label_names.index("rottenpotato")
    development_indices = np.asarray(
        json.loads((REPOSITORY_ROOT / args.split_manifest).read_text(encoding="utf-8"))[
            "development_indices"
        ],
        dtype=np.int64,
    )
    model_comparison = [
        model_agreement(
            REPOSITORY_ROOT / args.baseline_predictions,
            development_indices,
            entries,
            judgments,
            rotten_label_index,
        )
        for judgments in reviewers
    ]

    findings = {
        "schema_version": 1,
        "review_set_count": key["review_set_count"],
        "reviewer_files": list(args.reviewer),
        "reviewer_scores": scores,
        "raw_agreement": agreement,
        "cohen_kappa": kappa,
        "baseline_model_comparison": model_comparison,
        "decision": decision,
        "integrity": {
            "model_forward_passes": 0,
            "locked_test_images_reviewed": 0,
            "labels_modified": 0,
        },
    }
    (output_dir / "label_audit_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )

    with (output_dir / "label_audit_disagreements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "source_index", "group", "reviewer_1", "reviewer_2"])
        for entry in entries:
            p = entry["position"]
            if reviewers[0][p] != reviewers[1][p]:
                writer.writerow(
                    [f"{p:03d}", entry["source_index"], entry["group"],
                     reviewers[0][p], reviewers[1][p]]
                )

    print(f"Outcome: {decision['outcome']}")
    print(f"Next: {decision['next_phase']}")
    print(f"Raw agreement {agreement:.4f}, Cohen's kappa {kappa:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
