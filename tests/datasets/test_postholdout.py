"""Unit tests for the Phase 9 post-holdout split utility."""

import unittest

import numpy as np

try:
    from src.datasets.postholdout import build_postholdout_split
except ModuleNotFoundError:
    build_postholdout_split = None


class PostHoldoutSplitTest(unittest.TestCase):
    def _split(self, labels, *, fraction=0.2, seed=20260810):
        self.assertIsNotNone(
            build_postholdout_split,
            "Phase 9.2 must provide build_postholdout_split().",
        )
        return build_postholdout_split(
            labels,
            locked_test_fraction=fraction,
            random_state=seed,
        )

    def test_same_seed_returns_identical_indices(self):
        labels = np.repeat(np.arange(4), 25)

        first = self._split(labels, seed=20260810)
        second = self._split(labels, seed=20260810)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_different_synthetic_seed_changes_membership(self):
        labels = np.repeat(np.arange(4), 25)

        first_development, first_locked = self._split(labels, seed=17)
        second_development, second_locked = self._split(labels, seed=29)

        self.assertFalse(
            np.array_equal(first_development, second_development)
            and np.array_equal(first_locked, second_locked)
        )

    def test_indices_are_disjoint_and_exhaustive(self):
        labels = np.repeat(np.arange(5), 20)

        development, locked_test = self._split(labels)

        self.assertEqual(np.intersect1d(development, locked_test).size, 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate((development, locked_test))),
            np.arange(labels.size),
        )

    def test_stratification_is_preserved(self):
        labels = np.repeat(np.arange(4), 50)

        development, locked_test = self._split(labels, fraction=0.2)

        source_counts = np.bincount(labels)
        development_counts = np.bincount(labels[development], minlength=4)
        locked_counts = np.bincount(labels[locked_test], minlength=4)
        np.testing.assert_array_equal(development_counts + locked_counts, source_counts)
        np.testing.assert_array_equal(locked_counts, np.full(4, 10))

    def test_invalid_fraction_is_rejected(self):
        labels = np.repeat(np.arange(2), 10)

        for fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    self._split(labels, fraction=fraction)

    def test_empty_and_non_vector_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            self._split([])
        with self.assertRaises(ValueError):
            self._split(np.zeros((2, 2), dtype=np.int64))

    def test_input_labels_and_global_numpy_rng_are_not_mutated(self):
        labels = np.repeat(np.arange(3), 30)
        original_labels = labels.copy()
        np.random.seed(12345)
        before = np.random.get_state()

        self._split(labels)

        after = np.random.get_state()
        np.testing.assert_array_equal(labels, original_labels)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == "__main__":
    unittest.main()