# Post-Holdout Label Audit Runbook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this runbook task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the frozen Phase 9.5 label quality audit and produce a finding that selects Phase 9.6 under the pre-committed decision rule.

**Architecture:** Three command-line entry points over one pure selection module. `build_label_audit_set` materializes a seeded, blinded 497-image review set plus a sealed answer key. Reviewers record judgments in a flat CSV keyed by review position. `analyze_label_audit` is the only consumer of the sealed key: it unblinds, computes per-reviewer error rates and inter-rater agreement, and applies the decision rule. No model is constructed or loaded at any point.

**Tech Stack:** Python 3.12, `datasets` 5.0.1 for the pinned local archive, NumPy, Pillow for contact sheets, scikit-learn for Cohen's kappa, `unittest` for tests.

## Global Constraints

Every task inherits these. Values are copied verbatim from [postholdout-label-audit-protocol.md](postholdout-label-audit-protocol.md); none may be recomputed or adjusted.

- `SUBJECT_COUNT` is 347, all development `freshpotato` indices.
- `CONTROL_COUNT` is 150, sampled from development `rottenpotato` indices.
- `REVIEW_SET_COUNT` is 497.
- `CONTROL_SAMPLE_SEED` is 20260813. `PRESENTATION_ORDER_SEED` is 20260813.
- Judgment categories are exactly `FRESH`, `ROTTEN`, `NOT_A_POTATO`, `UNDECIDABLE`.
- `SUBJECT_ERROR_RATE` is `count(ROTTEN or NOT_A_POTATO) / 347`.
- `CONTROL_ERROR_RATE` is `count(FRESH or NOT_A_POTATO) / 150`.
- `UNDECIDABLE` counts in the denominator and is never an error.
- The threshold is 15 percentage points, evaluated for each reviewer against that reviewer's own control rate.
- No locked-test index may enter the review set. No canonical-holdout index may enter it.
- No model is constructed, loaded, or run. No label is modified. No image is published.
- Indices are zero-based positions into the reconstructed historical canonical training pool, the same identity used by `configs/splits/deep3-postholdout-research-01.json`.

## File Structure

| File | Responsibility |
|---|---|
| `src/datasets/label_audit.py` | Pure selection and scoring logic: choose the review set, shuffle it, score judgments, apply the decision rule. No I/O, no dataset access. |
| `scripts/build_label_audit_set.py` | CLI. Reconstructs the source pool, materializes blinded images and contact sheets, writes the sealed key. |
| `scripts/analyze_label_audit.py` | CLI. The only reader of the sealed key. Unblinds, computes rates and agreement, applies the rule, writes findings. |
| `tests/datasets/test_label_audit.py` | Unit tests for selection, scoring, and the decision rule. |
| `tests/scripts/test_label_audit_cli.py` | CLI contract tests: argument surface, leakage guards, refusal to overwrite. |

Blinding is enforced by file layout, not by discipline. The review directory contains only position-named images; the answer key is written outside it.

---

### Task 1: Deterministic review-set selection

**Files:**
- Create: `src/datasets/label_audit.py`
- Test: `tests/datasets/test_label_audit.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `select_review_set(development_indices: np.ndarray, development_labels: np.ndarray, class_names: list[str], *, control_seed: int, order_seed: int, subject_count: int, control_count: int) -> dict` returning keys `subject_indices` (np.ndarray int64, 347), `control_indices` (np.ndarray int64, 150), `presentation` (np.ndarray int64, 497 source indices in review order). All five keyword arguments are required.

- [ ] **Step 1: Write the failing test**

```python
# tests/datasets/test_label_audit.py
import unittest

import numpy as np

from src.datasets.label_audit import select_review_set


CLASS_NAMES = ["freshpotato", "rottenpotato", "freshapples"]


def _synthetic_pool():
    """400 freshpotato, 200 rottenpotato, 100 freshapples at known source indices."""
    labels = np.array([0] * 400 + [1] * 200 + [2] * 100, dtype=np.int64)
    indices = np.arange(700, dtype=np.int64) + 1000
    return indices, labels


class SelectReviewSetTest(unittest.TestCase):
    def test_subject_group_is_every_freshpotato_index(self):
        indices, labels = _synthetic_pool()

        result = select_review_set(
            indices, labels, CLASS_NAMES,
            control_seed=20260813, order_seed=20260813,
            subject_count=400, control_count=150,
        )

        self.assertEqual(len(result["subject_indices"]), 400)
        self.assertEqual(set(result["subject_indices"].tolist()), set(indices[labels == 0].tolist()))

    def test_control_group_is_seeded_and_reproducible(self):
        indices, labels = _synthetic_pool()
        kwargs = dict(
            control_seed=20260813, order_seed=20260813,
            subject_count=400, control_count=150,
        )

        first = select_review_set(indices, labels, CLASS_NAMES, **kwargs)
        second = select_review_set(indices, labels, CLASS_NAMES, **kwargs)

        np.testing.assert_array_equal(first["control_indices"], second["control_indices"])
        self.assertEqual(len(first["control_indices"]), 150)
        self.assertTrue(set(first["control_indices"].tolist()) <= set(indices[labels == 1].tolist()))

    def test_presentation_interleaves_both_groups(self):
        indices, labels = _synthetic_pool()

        result = select_review_set(
            indices, labels, CLASS_NAMES,
            control_seed=20260813, order_seed=20260813,
            subject_count=400, control_count=150,
        )
        presentation = result["presentation"]

        self.assertEqual(len(presentation), 550)
        self.assertEqual(len(set(presentation.tolist())), 550)
        # A sorted or grouped order would make group membership guessable from position.
        controls = set(result["control_indices"].tolist())
        first_half = sum(1 for i in presentation[:275] if i in controls)
        self.assertGreater(first_half, 40)

    def test_wrong_subject_count_is_rejected(self):
        indices, labels = _synthetic_pool()

        with self.assertRaises(ValueError):
            select_review_set(
                indices, labels, CLASS_NAMES,
                control_seed=1, order_seed=1,
                subject_count=347, control_count=150,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.datasets.test_label_audit -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.datasets.label_audit'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/datasets/label_audit.py
"""Selection and scoring for the frozen Phase 9.5 label quality audit.

Pure logic only. This module never touches the dataset, the filesystem, or a
model, so the audit's determinism can be tested without either.
"""

from __future__ import annotations

import numpy as np


SUBJECT_CLASS = "freshpotato"
CONTROL_CLASS = "rottenpotato"

JUDGMENT_CATEGORIES = ("FRESH", "ROTTEN", "NOT_A_POTATO", "UNDECIDABLE")
SUBJECT_ERROR_CATEGORIES = ("ROTTEN", "NOT_A_POTATO")
CONTROL_ERROR_CATEGORIES = ("FRESH", "NOT_A_POTATO")


def select_review_set(
    development_indices: np.ndarray,
    development_labels: np.ndarray,
    class_names: list[str],
    *,
    control_seed: int,
    order_seed: int,
    subject_count: int,
    control_count: int,
) -> dict:
    """Choose the blinded review set deterministically from frozen seeds."""
    development_indices = np.asarray(development_indices, dtype=np.int64)
    development_labels = np.asarray(development_labels, dtype=np.int64)
    if development_indices.shape != development_labels.shape:
        raise ValueError("Development indices and labels must align.")

    subject_label = class_names.index(SUBJECT_CLASS)
    control_label = class_names.index(CONTROL_CLASS)

    subject_indices = np.sort(development_indices[development_labels == subject_label])
    control_pool = np.sort(development_indices[development_labels == control_label])

    if len(subject_indices) != subject_count:
        raise ValueError(
            f"Expected {subject_count} {SUBJECT_CLASS} indices, found {len(subject_indices)}."
        )
    if len(control_pool) < control_count:
        raise ValueError(
            f"Expected at least {control_count} {CONTROL_CLASS} indices, found {len(control_pool)}."
        )

    control_indices = np.sort(
        np.random.default_rng(control_seed).choice(control_pool, size=control_count, replace=False)
    )

    presentation = np.concatenate([subject_indices, control_indices])
    np.random.default_rng(order_seed).shuffle(presentation)

    return {
        "subject_indices": subject_indices,
        "control_indices": control_indices,
        "presentation": presentation,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.datasets.test_label_audit -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/datasets/label_audit.py tests/datasets/test_label_audit.py
git commit -m "feat: add deterministic label audit review-set selection"
```

---

### Task 2: Scoring and the frozen decision rule

**Files:**
- Modify: `src/datasets/label_audit.py`
- Modify: `tests/datasets/test_label_audit.py`

**Interfaces:**
- Consumes: `SUBJECT_ERROR_CATEGORIES`, `CONTROL_ERROR_CATEGORIES`, `JUDGMENT_CATEGORIES` from Task 1.
- Produces:
  - `score_reviewer(judgments: dict[int, str], subject_indices: np.ndarray, control_indices: np.ndarray) -> dict` with keys `subject_error_rate`, `control_error_rate`, `difference`, `subject_undecidable_rate`, `control_undecidable_rate`.
  - `apply_decision_rule(reviewer_scores: list[dict], *, threshold: float = 0.15) -> dict` with keys `outcome` (one of `DEFECT_CONFIRMED`, `DEFECT_NOT_CONFIRMED`, `SPLIT_OUTCOME`), `next_phase`, `clears_threshold` (list[bool]).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/datasets/test_label_audit.py
from src.datasets.label_audit import apply_decision_rule, score_reviewer


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.subject = np.arange(0, 10, dtype=np.int64)
        self.control = np.arange(100, 110, dtype=np.int64)

    def _judgments(self, subject_calls, control_calls):
        j = {int(i): c for i, c in zip(self.subject, subject_calls)}
        j.update({int(i): c for i, c in zip(self.control, control_calls)})
        return j

    def test_undecidable_counts_in_denominator_but_is_not_an_error(self):
        judgments = self._judgments(
            ["ROTTEN"] * 3 + ["UNDECIDABLE"] * 2 + ["FRESH"] * 5,
            ["ROTTEN"] * 10,
        )

        scores = score_reviewer(judgments, self.subject, self.control)

        # 3 errors over 10, not 3 over 8.
        self.assertAlmostEqual(scores["subject_error_rate"], 0.3)
        self.assertAlmostEqual(scores["subject_undecidable_rate"], 0.2)
        self.assertAlmostEqual(scores["control_error_rate"], 0.0)

    def test_not_a_potato_is_an_error_for_both_groups(self):
        judgments = self._judgments(
            ["NOT_A_POTATO"] * 10,
            ["NOT_A_POTATO"] * 10,
        )

        scores = score_reviewer(judgments, self.subject, self.control)

        self.assertAlmostEqual(scores["subject_error_rate"], 1.0)
        self.assertAlmostEqual(scores["control_error_rate"], 1.0)

    def test_missing_judgment_is_rejected(self):
        judgments = self._judgments(["FRESH"] * 10, ["ROTTEN"] * 9)

        with self.assertRaises(ValueError):
            score_reviewer(judgments, self.subject, self.control)

    def test_unknown_category_is_rejected(self):
        judgments = self._judgments(["SPOILED"] * 10, ["ROTTEN"] * 10)

        with self.assertRaises(ValueError):
            score_reviewer(judgments, self.subject, self.control)


class DecisionRuleTest(unittest.TestCase):
    def test_both_reviewers_clearing_confirms_the_defect(self):
        scores = [
            {"subject_error_rate": 0.70, "control_error_rate": 0.05},
            {"subject_error_rate": 0.60, "control_error_rate": 0.10},
        ]

        result = apply_decision_rule(scores)

        self.assertEqual(result["outcome"], "DEFECT_CONFIRMED")
        self.assertEqual(result["clears_threshold"], [True, True])

    def test_neither_reviewer_clearing_returns_to_the_loss_experiment(self):
        scores = [
            {"subject_error_rate": 0.12, "control_error_rate": 0.05},
            {"subject_error_rate": 0.18, "control_error_rate": 0.10},
        ]

        result = apply_decision_rule(scores)

        self.assertEqual(result["outcome"], "DEFECT_NOT_CONFIRMED")
        self.assertIn("H1", result["next_phase"])

    def test_one_reviewer_clearing_is_a_split_outcome(self):
        scores = [
            {"subject_error_rate": 0.70, "control_error_rate": 0.05},
            {"subject_error_rate": 0.12, "control_error_rate": 0.10},
        ]

        result = apply_decision_rule(scores)

        self.assertEqual(result["outcome"], "SPLIT_OUTCOME")
        self.assertEqual(result["clears_threshold"], [True, False])

    def test_exactly_fifteen_points_clears(self):
        # 0.15 - 0.0 is exactly 0.15 in IEEE-754, so this distinguishes >= from >.
        # Do not use 0.20 - 0.05: that is 0.15000000000000002, which passes under
        # both comparisons and would leave the frozen threshold unprotected.
        scores = [
            {"subject_error_rate": 0.15, "control_error_rate": 0.0},
            {"subject_error_rate": 0.15, "control_error_rate": 0.0},
        ]

        self.assertEqual(apply_decision_rule(scores)["outcome"], "DEFECT_CONFIRMED")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.datasets.test_label_audit -v`
Expected: FAIL with `ImportError: cannot import name 'apply_decision_rule'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/datasets/label_audit.py

DECISION_THRESHOLD = 0.15


def _rate(judgments: dict[int, str], indices: np.ndarray, error_categories) -> tuple[float, float]:
    total = len(indices)
    errors = 0
    undecidable = 0
    for index in indices.tolist():
        try:
            call = judgments[int(index)]
        except KeyError:
            raise ValueError(f"Missing judgment for review index {index}.") from None
        if call not in JUDGMENT_CATEGORIES:
            raise ValueError(f"Unknown judgment category {call!r} for index {index}.")
        if call in error_categories:
            errors += 1
        elif call == "UNDECIDABLE":
            undecidable += 1
    return errors / total, undecidable / total


def score_reviewer(
    judgments: dict[int, str],
    subject_indices: np.ndarray,
    control_indices: np.ndarray,
) -> dict:
    """Score one reviewer. UNDECIDABLE stays in the denominator, never an error."""
    subject_error, subject_undecidable = _rate(
        judgments, np.asarray(subject_indices), SUBJECT_ERROR_CATEGORIES
    )
    control_error, control_undecidable = _rate(
        judgments, np.asarray(control_indices), CONTROL_ERROR_CATEGORIES
    )
    return {
        "subject_error_rate": subject_error,
        "control_error_rate": control_error,
        "difference": subject_error - control_error,
        "subject_undecidable_rate": subject_undecidable,
        "control_undecidable_rate": control_undecidable,
    }


def apply_decision_rule(reviewer_scores: list[dict], *, threshold: float = DECISION_THRESHOLD) -> dict:
    """Apply the pre-committed rule. Each reviewer is judged against their own control."""
    clears = [
        (s["subject_error_rate"] - s["control_error_rate"]) >= threshold
        for s in reviewer_scores
    ]
    if all(clears):
        outcome = "DEFECT_CONFIRMED"
        next_phase = "Phase 9.6 remediation decision: relabel, exclude, or retain (separate authorization)"
    elif not any(clears):
        outcome = "DEFECT_NOT_CONFIRMED"
        next_phase = "Phase 9.6 is H1 loss and class imbalance, as pre-registered"
    else:
        outcome = "SPLIT_OUTCOME"
        next_phase = "No phase selected automatically; owner decides after reviewing disagreements"
    return {"outcome": outcome, "next_phase": next_phase, "clears_threshold": clears}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.datasets.test_label_audit -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add src/datasets/label_audit.py tests/datasets/test_label_audit.py
git commit -m "feat: add label audit scoring and frozen decision rule"
```

---

### Task 3: Blinded review-set materialization CLI

**Files:**
- Create: `scripts/build_label_audit_set.py`
- Test: `tests/scripts/test_label_audit_cli.py`

**Interfaces:**
- Consumes: `select_review_set` from Task 1.
- Produces: a CLI writing `review/000.jpg` … `review/496.jpg`, `review/contact_sheets/sheet_00.jpg` …, `review_set_key.json`, and `judgment_template.csv`.

The answer key is written to the parent directory, never inside `review/`. Images are named by review position only, so no filename carries class information.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_label_audit_cli.py
import unittest

from scripts.build_label_audit_set import build_parser, partition_outputs


class BuildLabelAuditSetCliTest(unittest.TestCase):
    def test_parser_requires_an_output_directory(self):
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_the_documented_arguments(self):
        parser = build_parser()

        args = parser.parse_args([
            "--split-manifest", "configs/splits/deep3-postholdout-research-01.json",
            "--output-dir", "results/label-audit",
        ])

        self.assertEqual(args.output_dir, "results/label-audit")

    def test_answer_key_is_written_outside_the_review_directory(self):
        outputs = partition_outputs("results/label-audit")

        # Blinding is enforced by layout: a reviewer opening review/ must not
        # be able to reach the key.
        self.assertNotIn(str(outputs["review_dir"]), str(outputs["key_path"].parent))
        self.assertEqual(outputs["key_path"].name, "review_set_key.json")
        self.assertEqual(outputs["review_dir"].name, "review")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_label_audit_cli -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_label_audit_set'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/build_label_audit_set.py
"""Materialize the blinded Phase 9.5 review set from the frozen protocol.

Writes position-named images and a sealed key. No model is constructed, and no
locked-test index may enter the review set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.datasets.label_audit import select_review_set

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROL_SAMPLE_SEED = 20260813
PRESENTATION_ORDER_SEED = 20260813
SUBJECT_COUNT = 347
CONTROL_COUNT = 150
SHEET_COLUMNS = 4
SHEET_ROWS = 4
CELL_PIXELS = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the blinded Phase 9.5 label audit review set."
    )
    parser.add_argument(
        "--split-manifest",
        default="configs/splits/deep3-postholdout-research-01.json",
        help="Frozen Phase 9 split manifest.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Empty local-only directory for the review set and sealed key.",
    )
    return parser


def partition_outputs(output_dir: str | Path) -> dict[str, Path]:
    """Place the key outside review/ so blinding survives a careless reviewer."""
    root = Path(output_dir)
    return {
        "root": root,
        "review_dir": root / "review",
        "sheet_dir": root / "review" / "contact_sheets",
        "key_path": root / "review_set_key.json",
        "template_path": root / "judgment_template.csv",
    }


def _load_development(split_manifest: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    return (
        np.asarray(manifest["development_indices"], dtype=np.int64),
        np.asarray(manifest["locked_test_indices"], dtype=np.int64),
    )


def _write_contact_sheets(images, positions, sheet_dir: Path) -> None:
    from PIL import Image, ImageDraw

    sheet_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for sheet_index in range(0, len(positions), per_sheet):
        chunk = positions[sheet_index : sheet_index + per_sheet]
        sheet = Image.new(
            "RGB",
            (SHEET_COLUMNS * CELL_PIXELS, SHEET_ROWS * (CELL_PIXELS + 24)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for cell, position in enumerate(chunk):
            thumb = images[position].convert("RGB").resize((CELL_PIXELS, CELL_PIXELS))
            x = (cell % SHEET_COLUMNS) * CELL_PIXELS
            y = (cell // SHEET_COLUMNS) * (CELL_PIXELS + 24)
            sheet.paste(thumb, (x, y))
            draw.text((x + 4, y + CELL_PIXELS + 4), f"{position:03d}", fill="black")
        sheet.save(sheet_dir / f"sheet_{sheet_index // per_sheet:02d}.jpg", quality=92)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = partition_outputs(REPOSITORY_ROOT / args.output_dir)
    if outputs["root"].exists() and any(outputs["root"].iterdir()):
        raise SystemExit(f"Output directory is not empty: {outputs['root']}")

    from scripts.freeze_postholdout_split import _reconstruct_canonical_pool_with_images

    dataset, labels, class_names, _ = _reconstruct_canonical_pool_with_images()
    development_indices, locked_test_indices = _load_development(
        REPOSITORY_ROOT / args.split_manifest
    )

    selection = select_review_set(
        development_indices,
        labels[development_indices],
        class_names,
        control_seed=CONTROL_SAMPLE_SEED,
        order_seed=PRESENTATION_ORDER_SEED,
        subject_count=SUBJECT_COUNT,
        control_count=CONTROL_COUNT,
    )
    presentation = selection["presentation"]

    locked = set(locked_test_indices.tolist())
    leaked = sorted(locked.intersection(presentation.tolist()))
    if leaked:
        raise SystemExit(f"Locked-test indices reached the review set: {leaked[:5]}")

    outputs["review_dir"].mkdir(parents=True, exist_ok=True)
    images = {}
    for position, source_index in enumerate(presentation.tolist()):
        image = dataset[int(source_index)]["image"]
        images[position] = image
        image.convert("RGB").save(outputs["review_dir"] / f"{position:03d}.jpg", quality=95)

    _write_contact_sheets(images, list(range(len(presentation))), outputs["sheet_dir"])

    subject = set(selection["subject_indices"].tolist())
    key = {
        "schema_version": 1,
        "control_sample_seed": CONTROL_SAMPLE_SEED,
        "presentation_order_seed": PRESENTATION_ORDER_SEED,
        "review_set_count": int(len(presentation)),
        "presentation_indices_sha256": hashlib.sha256(
            presentation.astype("<i8").tobytes()
        ).hexdigest(),
        "entries": [
            {
                "position": position,
                "source_index": int(source_index),
                "group": "SUBJECT" if int(source_index) in subject else "CONTROL",
            }
            for position, source_index in enumerate(presentation.tolist())
        ],
    }
    outputs["key_path"].write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")

    with outputs["template_path"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "judgment"])
        for position in range(len(presentation)):
            writer.writerow([f"{position:03d}", ""])

    print(f"Review set: {len(presentation)} images")
    print(f"Sealed key: {outputs['key_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Refactor the pool reconstruction so both callers share one definition**

The audit needs images; the split freeze needs labels only. Do not copy the filter and split logic — a duplicated `remove_labels` or seed that later diverges would silently audit different images than the split describes. Extract one implementation and make the existing entry point delegate to it.

Rename the body of `_reconstruct_canonical_source_pool` into a new function that additionally returns the split object, then reduce the original to a wrapper. Its no-image guarantee is preserved: `datasets` decodes an image only when a row is indexed, and returning the split object decodes nothing.

```python
# in scripts/freeze_postholdout_split.py, replacing _reconstruct_canonical_source_pool

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
```

In `scripts/build_label_audit_set.py`, import `_reconstruct_canonical_pool_with_images` and unpack four values:

```python
from scripts.freeze_postholdout_split import _reconstruct_canonical_pool_with_images

dataset, labels, class_names, _ = _reconstruct_canonical_pool_with_images()
```

The existing split-freeze tests cover the wrapper's contract, so a regression in the extraction fails them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_label_audit_cli -v`
Expected: PASS, 3 tests

- [ ] **Step 6: Commit**

```bash
git add scripts/build_label_audit_set.py scripts/freeze_postholdout_split.py tests/scripts/test_label_audit_cli.py
git commit -m "feat: add blinded label audit review-set materialization"
```

---

### Task 4: Unblinding, agreement, and findings

**Files:**
- Create: `scripts/analyze_label_audit.py`
- Modify: `tests/scripts/test_label_audit_cli.py`

**Interfaces:**
- Consumes: `score_reviewer` and `apply_decision_rule` from Task 2; `review_set_key.json` from Task 3; the existing `results/deep3-postholdout-research-01-baseline/development_oof_predictions.npz`.
- Produces: `label_audit_findings.json` and `label_audit_disagreements.csv`.

The protocol lists agreement between reviewer judgments and the baseline model's predictions on the subject group as an output. That comparison reads stored predictions from Phase 9.4; it runs no model. It is diagnostic only and feeds no part of the decision rule, which depends on reviewer judgments alone.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/scripts/test_label_audit_cli.py
from scripts.analyze_label_audit import build_parser as analyze_parser
from scripts.analyze_label_audit import load_judgments


class AnalyzeLabelAuditCliTest(unittest.TestCase):
    def test_main_refuses_any_count_other_than_two_reviewers(self):
        # Drive the guard in main(), not the parser. A parser-level test here
        # exits for the missing required --output-dir instead, which would pass
        # even with the two-reviewer guard deleted.
        for reviewer_files in (["a.csv"], ["a.csv", "b.csv", "c.csv"]):
            argv = ["--key", "k.json", "--output-dir", "out"]
            for path in reviewer_files:
                argv += ["--reviewer", path]
            with self.subTest(count=len(reviewer_files)):
                with self.assertRaises(SystemExit):
                    main(argv)

    def test_parser_accepts_exactly_two_reviewers(self):
        parser = analyze_parser()

        args = parser.parse_args([
            "--key", "k.json",
            "--reviewer", "owner.csv",
            "--reviewer", "assistant.csv",
            "--output-dir", "results/label-audit",
        ])

        self.assertEqual(len(args.reviewer), 2)

    def test_blank_judgment_rows_are_rejected(self):
        import io

        handle = io.StringIO("position,judgment\n000,FRESH\n001,\n")

        with self.assertRaises(ValueError):
            load_judgments(handle)

    def test_model_comparison_skips_undecidable_and_not_a_potato(self):
        import numpy as np
        import tempfile
        from pathlib import Path

        from scripts.analyze_label_audit import model_agreement

        development_indices = np.array([10, 11, 12, 13], dtype=np.int64)
        entries = [
            {"position": 0, "source_index": 10, "group": "SUBJECT"},
            {"position": 1, "source_index": 11, "group": "SUBJECT"},
            {"position": 2, "source_index": 12, "group": "SUBJECT"},
            {"position": 3, "source_index": 13, "group": "CONTROL"},
        ]
        judgments = {0: "ROTTEN", 1: "UNDECIDABLE", 2: "NOT_A_POTATO", 3: "ROTTEN"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.npz"
            np.savez(path, predictions=np.array([12, 12, 12, 12], dtype=np.int64))
            result = model_agreement(path, development_indices, entries, judgments, 12)

        # Only position 0 is a FRESH/ROTTEN subject call.
        self.assertEqual(result["compared"], 1)
        self.assertAlmostEqual(result["agreement"], 1.0)
```

Add one end-to-end test over synthetic fixtures in a temporary directory: a key JSON mixing `SUBJECT` and `CONTROL` entries, two reviewer CSVs, a `.npz` holding a `predictions` array, a `label_names.json` with the fourteen class names, and a split manifest carrying `development_indices`. Run `main()` against them and assert both artifacts are written with the expected content.

Then make that test prove the isolation claim directly. Run `main()` twice with **different** `predictions` arrays and everything else identical, and assert the `decision` block is identical across both runs while `baseline_model_comparison` differs:

```python
    def test_decision_is_unaffected_by_the_stored_predictions(self):
        # The audit's credibility rests on nothing learned at unblinding time
        # changing the outcome. Assert that mechanically rather than by reading.
        first = self._run_main_with_predictions([12] * 4)
        second = self._run_main_with_predictions([0] * 4)

        self.assertEqual(first["decision"], second["decision"])
        self.assertNotEqual(
            first["baseline_model_comparison"], second["baseline_model_comparison"]
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_label_audit_cli -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.analyze_label_audit'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/analyze_label_audit.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_label_audit_cli -v`
Expected: PASS, 7 tests (3 from Task 3, 4 added here)

- [ ] **Step 5: Run the full suite and compileall**

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
.venv/Scripts/python.exe -m compileall -q src scripts tests
```

Expected: OK, and the count rises from 268 by the tests added in Tasks 1 through 4.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_label_audit.py tests/scripts/test_label_audit_cli.py
git commit -m "feat: add label audit unblinding and findings analysis"
```

---

### Task 5: Execute the audit

This task is procedural rather than code, and its ordering is what the protocol depends on. Do not reorder these steps.

- [ ] **Step 1: Materialize the review set**

```powershell
.venv\Scripts\python.exe -m scripts.build_label_audit_set --output-dir results/deep3-postholdout-research-01-label-audit
```

Expected: `Review set: 497 images`. The command aborts if the output directory is non-empty or if any locked-test index reaches the review set.

- [ ] **Step 2: Verify the review set before anyone looks at it**

Confirm 497 files in `review/`, 32 contact sheets, and that `review_set_key.json` sits outside `review/`. Record `presentation_indices_sha256` in the phase handoff. Do not open the key.

- [ ] **Step 3: Both reviewers judge independently**

Each reviewer copies `judgment_template.csv` to `judgment_owner.csv` or `judgment_assistant.csv` and fills every row with one of `FRESH`, `ROTTEN`, `NOT_A_POTATO`, `UNDECIDABLE`, working only from `review/` or `review/contact_sheets/`. Neither reviewer opens the key, the other's file, the split manifest, or the baseline predictions until both files are complete. Unsure means `UNDECIDABLE`.

- [ ] **Step 4: Unblind and score**

```powershell
.venv\Scripts\python.exe -m scripts.analyze_label_audit `
  --key results/deep3-postholdout-research-01-label-audit/review_set_key.json `
  --reviewer results/deep3-postholdout-research-01-label-audit/judgment_owner.csv `
  --reviewer results/deep3-postholdout-research-01-label-audit/judgment_assistant.csv `
  --output-dir results/deep3-postholdout-research-01-label-audit
```

- [ ] **Step 5: Record the finding**

Report the aggregate outcome, both reviewers' subject and control rates separately, the `UNDECIDABLE` shares, raw agreement, and Cohen's kappa. Report the decision-rule outcome exactly as computed, including `SPLIT_OUTCOME`. Do not substitute a narrative reading for the rule.

Update `docs/postholdout-label-audit-protocol.md` to `AUDIT_EXECUTION_STATUS: COMPLETED`, add the Phase 9.5 handoff block to `SESSION_HANDOFF.md`, update `CHANGELOG.md`, and record the selected Phase 9.6 in the registry and governance documents.

- [ ] **Step 6: Commit and open the execution PR**

Artifacts under `results/` are gitignored and stay local. Only the judgment record and the findings summary are quoted into tracked documents; images are never published.

## Stop Conditions

Stop and report rather than continuing if the review set does not resolve to exactly 347 subject and 150 control images, if any locked-test or canonical-holdout index appears, if a reviewer sees the key or the other reviewer's file before both are complete, if a judgment file has a blank or unknown category, or if the decision rule returns `SPLIT_OUTCOME`.

Do not adjust the threshold, the denominators, the seeds, or the category definitions during execution. Those are frozen in the protocol, and changing them after seeing any image voids the audit.
