"""Offline contract for the dataset duplication audit and its disclosures."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "dataset-duplication-audit.md"
README = ROOT / "README.md"
MODEL_CARD = ROOT / "docs" / "model-card.md"
CANONICAL_RESULTS = ROOT / "docs" / "canonical-results.md"
CANONICAL_EVAL = ROOT / "docs" / "canonical-holdout-evaluation.md"


class DuplicationAuditContractTest(unittest.TestCase):
    def test_audit_records_the_measured_counts(self):
        document = AUDIT.read_text(encoding="utf-8")

        for token in (
            "FILTERED_ROWS:\n26858",
            "UNIQUE_IMAGES_AFTER_FILTER:\n21413",
            "EXTRA_COPIES:\n5445",
            "CROSS_CLASS_DUPLICATE_GROUPS:\n0",
            "HOLDOUT_ROWS_DUPLICATING_A_TRAIN_ROW:\n1618",
            "LOCKED_ROWS_DUPLICATING_A_DEVELOPMENT_ROW:\n1140",
            "DEVELOPMENT_ROWS_WHOSE_COPY_SITS_IN_ANOTHER_CV_FOLD:\n3312",
        ):
            self.assertIn(token, document)

    def test_audit_states_that_no_metric_was_revised(self):
        document = AUDIT.read_text(encoding="utf-8")

        # The decision was to record without revising. A later edit that
        # quietly swaps in the cleaned figures would change what this
        # repository claims.
        self.assertIn("RECORDED_METRICS_REVISED:\nNO", document)
        self.assertIn("DECISION:\nRECORD_WITHOUT_REVISING", document)
        self.assertIn("RESPLIT_ON_UNIQUE_IMAGES:\nNOT_AUTHORIZED", document)
        self.assertIn("**No recorded metric is revised.**", document)

    def test_audit_keeps_the_uncomfortable_comparison(self):
        document = AUDIT.read_text(encoding="utf-8")

        # The leak being the size of the effect Phase 9.6 chased is the part
        # most worth losing and least comfortable to keep.
        self.assertIn(
            "The leak is the same size as the signal that phase was chasing",
            document,
        )

    def test_audit_records_what_is_unaffected_without_overclaiming(self):
        document = AUDIT.read_text(encoding="utf-8")

        self.assertIn(
            "`freshpotato` and `rottenpotato` contain no duplicates at all",
            document,
        )
        # Byte hashing cannot see re-encoded near-duplicates, so the counts
        # are a floor and the document must say so.
        self.assertIn("a lower bound on the true redundancy", document)

    def test_audit_records_that_the_class_filter_is_correct(self):
        document = AUDIT.read_text(encoding="utf-8")

        # The audit began as a search for unused data. It found none, and
        # that negative result is the reason nobody should look again.
        self.assertIn("The filter is deduplicating, not discarding.", document)
        self.assertIn("| `freshpatato` | 270 | 270 |", document)

    def test_locked_test_invariant_is_restated_not_weakened(self):
        document = AUDIT.read_text(encoding="utf-8")

        self.assertIn("LOCKED_TEST_MODEL_FORWARD_PASSES:\n0", document)
        self.assertIn("still has zero model forward passes", document)

    def test_reader_facing_documents_disclose_the_overlap(self):
        for path in (README, MODEL_CARD, CANONICAL_RESULTS, CANONICAL_EVAL):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                # Every document that reports the inflated number must point
                # at the audit, or a reader meets the figure without it.
                self.assertIn("dataset-duplication-audit.md", text)
                self.assertIn("0.9414", text)


if __name__ == "__main__":
    unittest.main()
