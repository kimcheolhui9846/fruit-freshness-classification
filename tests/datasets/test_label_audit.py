import unittest

import numpy as np

from src.datasets.label_audit import apply_decision_rule, score_reviewer, select_review_set


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
        scores = [
            {"subject_error_rate": 0.20, "control_error_rate": 0.05},
            {"subject_error_rate": 0.20, "control_error_rate": 0.05},
        ]

        self.assertEqual(apply_decision_rule(scores)["outcome"], "DEFECT_CONFIRMED")


if __name__ == "__main__":
    unittest.main()
