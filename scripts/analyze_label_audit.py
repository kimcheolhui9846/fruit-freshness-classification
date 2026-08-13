"""Unblind the Phase 9.5 audit and apply the frozen decision rule.

This is the only reader of the sealed key. No model is constructed or loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.datasets.label_audit import (
    CONTROL_COUNT,
    CONTROL_SAMPLE_SEED,
    PRESENTATION_ORDER_SEED,
    JUDGMENT_CATEGORIES,
    REVIEW_SET_COUNT,
    SUBJECT_COUNT,
    apply_decision_rule,
    score_reviewer,
    select_review_set,
)

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
    parser.add_argument(
        "--expected-hash",
        default=None,
        help=(
            "Review-set presentation_indices_sha256 recorded before the review "
            "began. Without it the recompute only proves the key is internally "
            "consistent, never that it is the key the operator started from."
        ),
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
    fresh_label_index: int,
) -> dict:
    """Compare reviewer calls against stored predictions. Reads a file; runs nothing.

    Diagnostic only: this never feeds the decision rule, which depends on
    reviewer judgments alone.

    Restricted to the two potato classes: a prediction of some other produce
    entirely (e.g. `freshbanana`) is not model/reviewer agreement or
    disagreement about potato freshness, so it is counted separately as
    `off_class` rather than folded into `agreement` as a mismatch would be.
    """
    stored = np.load(predictions_path)
    predicted = stored["predictions"]
    position_in_development = {
        int(source): row for row, source in enumerate(development_indices.tolist())
    }

    agree = 0
    compared = 0
    off_class = 0
    for entry in entries:
        if entry["group"] != "SUBJECT":
            continue
        call = judgments[entry["position"]]
        if call not in ("FRESH", "ROTTEN"):
            continue
        row = position_in_development[int(entry["source_index"])]
        predicted_label = int(predicted[row])
        if predicted_label not in (rotten_label_index, fresh_label_index):
            off_class += 1
            continue
        model_says_rotten = predicted_label == rotten_label_index
        compared += 1
        agree += int(model_says_rotten == (call == "ROTTEN"))

    return {
        "compared": compared,
        "agreement": (agree / compared) if compared else None,
        "off_class": off_class,
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
        if position in judgments:
            raise ValueError(f"Position {position:03d} is duplicated in this file.")
        judgments[position] = call
    return judgments


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.reviewer) != 2:
        raise SystemExit("The protocol fixes two independent reviewers.")

    reviewer_paths = [Path(path) for path in args.reviewer]
    if reviewer_paths[0].resolve() == reviewer_paths[1].resolve():
        raise SystemExit(
            "The two --reviewer paths resolve to the same file; the protocol "
            "requires two independent reviewers, not one file counted twice."
        )

    output_dir = Path(args.output_dir)
    findings_path = output_dir / "label_audit_findings.json"
    disagreements_path = output_dir / "label_audit_disagreements.csv"
    if findings_path.exists() or disagreements_path.exists():
        raise SystemExit(
            f"Findings already exist in {output_dir}; refusing to overwrite them. "
            "Re-running would silently replace one recorded outcome with another. "
            "Use a fresh --output-dir."
        )

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    entries = key["entries"]
    subject_positions = np.array(
        [e["position"] for e in entries if e["group"] == "SUBJECT"], dtype=np.int64
    )
    control_positions = np.array(
        [e["position"] for e in entries if e["group"] == "CONTROL"], dtype=np.int64
    )
    if len(subject_positions) != SUBJECT_COUNT or len(control_positions) != CONTROL_COUNT:
        raise SystemExit(
            "Sealed key group sizes do not match the frozen protocol: "
            f"subject {len(subject_positions)} (expected {SUBJECT_COUNT}), "
            f"control {len(control_positions)} (expected {CONTROL_COUNT})."
        )
    if len(entries) != REVIEW_SET_COUNT:
        raise SystemExit(
            f"Sealed key has {len(entries)} entries, expected {REVIEW_SET_COUNT}."
        )
    if key.get("review_set_count") != len(entries):
        raise SystemExit(
            f"Sealed key review_set_count ({key.get('review_set_count')!r}) does "
            f"not match its own entry count ({len(entries)})."
        )

    ordered_source_indices = [
        entry["source_index"] for entry in sorted(entries, key=lambda e: e["position"])
    ]
    recomputed_hash = hashlib.sha256(
        np.asarray(ordered_source_indices, dtype="<i8").tobytes()
    ).hexdigest()
    if recomputed_hash != key.get("presentation_indices_sha256"):
        raise SystemExit(
            "Sealed key presentation_indices_sha256 does not match the hash "
            "recomputed from its entries; the key may be corrupted or tampered with."
        )
    if args.expected_hash is not None and args.expected_hash != recomputed_hash:
        raise SystemExit(
            f"Review set hash {recomputed_hash} does not match the expected value "
            f"{args.expected_hash} recorded before the review began."
        )

    # The key's `group` field is not trusted. The checks above cover the group
    # sizes and the presentation indices, but a balanced SUBJECT<->CONTROL swap
    # keeps both intact while changing each error rate and therefore the
    # decision. Group membership is derivable from the published seeds, so
    # recompute it and refuse to proceed if the key disagrees. Tampering with
    # the key alone fails this check; tampering with the labels alone fails it
    # from the other side.
    label_names = load_label_names(REPOSITORY_ROOT / args.label_names)
    development_indices = np.asarray(
        json.loads((REPOSITORY_ROOT / args.split_manifest).read_text(encoding="utf-8"))[
            "development_indices"
        ],
        dtype=np.int64,
    )
    stored_predictions = np.load(REPOSITORY_ROOT / args.baseline_predictions)
    if "labels" not in stored_predictions:
        raise SystemExit(
            "Stored predictions carry no `labels` array, so review-set group "
            "membership cannot be recomputed and the sealed key cannot be checked."
        )
    derived_subject = set(
        select_review_set(
            development_indices,
            np.asarray(stored_predictions["labels"], dtype=np.int64),
            label_names,
            control_seed=CONTROL_SAMPLE_SEED,
            order_seed=PRESENTATION_ORDER_SEED,
            subject_count=SUBJECT_COUNT,
            control_count=CONTROL_COUNT,
        )["subject_indices"].tolist()
    )
    for entry in entries:
        derived = "SUBJECT" if int(entry["source_index"]) in derived_subject else "CONTROL"
        if derived != entry["group"]:
            raise SystemExit(
                "Sealed key group disagrees with the seeded recomputation at "
                f"position {entry['position']:03d}: key says {entry['group']}, "
                f"recomputation says {derived}."
            )
    subject_positions = np.array(
        [e["position"] for e in entries if int(e["source_index"]) in derived_subject],
        dtype=np.int64,
    )
    control_positions = np.array(
        [e["position"] for e in entries if int(e["source_index"]) not in derived_subject],
        dtype=np.int64,
    )

    reviewers = []
    for path in reviewer_paths:
        with path.open(encoding="utf-8") as handle:
            reviewers.append(load_judgments(handle))

    scores = [score_reviewer(j, subject_positions, control_positions) for j in reviewers]
    decision = apply_decision_rule(scores)

    positions = [e["position"] for e in entries]
    first = [reviewers[0][p] for p in positions]
    second = [reviewers[1][p] for p in positions]
    agreement = float(np.mean([a == b for a, b in zip(first, second)]))
    kappa_raw = cohen_kappa_score(first, second)
    # A degenerate agreement matrix (e.g. only one category ever used) makes
    # sklearn return NaN. json.dumps would emit a bare NaN token, which
    # strict JSON parsers reject, so store None (-> JSON null) instead.
    kappa = float(kappa_raw) if math.isfinite(kappa_raw) else None

    output_dir.mkdir(parents=True, exist_ok=True)

    rotten_label_index = label_names.index("rottenpotato")
    fresh_label_index = label_names.index("freshpotato")
    model_comparison = [
        model_agreement(
            REPOSITORY_ROOT / args.baseline_predictions,
            development_indices,
            entries,
            judgments,
            rotten_label_index,
            fresh_label_index,
        )
        for judgments in reviewers
    ]

    findings = {
        "schema_version": 2,
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
    findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )

    with disagreements_path.open("w", newline="", encoding="utf-8") as handle:
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
    kappa_display = f"{kappa:.4f}" if kappa is not None else "NaN (degenerate agreement matrix)"
    print(f"Raw agreement {agreement:.4f}, Cohen's kappa {kappa_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
