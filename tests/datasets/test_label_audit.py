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
