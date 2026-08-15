"""Set arithmetic and CLI behaviour of the instability diagnostic."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnose_freshpotato_instability.py"

# tests/scripts shadows scripts on the import path, so the CLI is loaded by
# file location rather than by module name.
_spec = importlib.util.spec_from_file_location("diagnose_instability_cli", SCRIPT)
diagnose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnose)


LABELS = np.array([5, 5, 5, 5, 12, 12], dtype=np.int64)


def _write_run(root: Path, run_name: str, predictions) -> Path:
    """Mirror the real layout: results/<run-id>/development_oof_predictions.npz.

    The CLI derives each run's name from its parent directory, so fixtures
    written side by side in one directory would all collide on the same name.
    """
    directory = root / run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "development_oof_predictions.npz"
    np.savez(
        path,
        labels=LABELS,
        predictions=np.asarray(predictions, dtype=np.int64),
        logits=np.zeros((LABELS.size, 14), dtype=np.float32),
        fold_assignments=np.ones(LABELS.size, dtype=np.int64),
    )
    return path


class ErrorPositionTest(unittest.TestCase):
    def test_only_the_named_class_counts_as_an_error(self):
        predictions = np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)
        positions = diagnose.error_positions(LABELS, predictions, 5)

        # Position 4 is a rottenpotato predicted freshpotato. That is an
        # error, but not an error *of* freshpotato.
        np.testing.assert_array_equal(positions, np.array([1, 3]))

    def test_a_perfect_class_has_no_error_positions(self):
        predictions = np.array([5, 5, 5, 5, 12, 12], dtype=np.int64)
        positions = diagnose.error_positions(LABELS, predictions, 5)

        self.assertEqual(positions.size, 0)


class CompareErrorSetsTest(unittest.TestCase):
    def test_identical_runs_are_fully_stable(self):
        runs = {
            "a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
            "b": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        self.assertEqual(result["stable_error_count"], 2)
        self.assertEqual(result["union_error_count"], 2)
        self.assertEqual(result["jaccard"], 1.0)

    def test_disjoint_error_sets_have_zero_overlap(self):
        runs = {
            "a": (LABELS, np.array([12, 5, 5, 5, 12, 12], dtype=np.int64)),
            "b": (LABELS, np.array([5, 12, 5, 5, 12, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        # Same error count, no shared example: the boundary moved.
        self.assertEqual(result["per_run_error_counts"], {"a": 1, "b": 1})
        self.assertEqual(result["stable_error_count"], 0)
        self.assertEqual(result["union_error_count"], 2)
        self.assertEqual(result["jaccard"], 0.0)

    def test_frequency_histogram_counts_examples_by_how_often_they_fail(self):
        runs = {
            "a": (LABELS, np.array([12, 12, 5, 5, 12, 12], dtype=np.int64)),
            "b": (LABELS, np.array([12, 5, 5, 5, 12, 12], dtype=np.int64)),
            "c": (LABELS, np.array([12, 5, 12, 5, 12, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        # Position 0 fails in all three, position 1 in one, position 2 in one.
        self.assertEqual(result["error_frequency_histogram"], {"1": 2, "2": 0, "3": 1})

    def test_no_errors_anywhere_gives_a_defined_jaccard(self):
        perfect = np.array([5, 5, 5, 5, 12, 12], dtype=np.int64)
        runs = {"a": (LABELS, perfect), "b": (LABELS, perfect)}
        result = diagnose.compare_error_sets(runs)

        # An empty union must not divide by zero.
        self.assertEqual(result["union_error_count"], 0)
        self.assertEqual(result["jaccard"], 1.0)

    def test_disagreeing_label_arrays_are_refused(self):
        other = np.array([5, 5, 5, 12, 12, 12], dtype=np.int64)
        runs = {
            "a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
            "b": (other, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
        }
        # Runs that disagree about which examples belong to the class are not
        # comparable, and averaging over them would be meaningless.
        with self.assertRaises(ValueError):
            diagnose.compare_error_sets(runs)

    def test_fewer_than_two_runs_is_refused(self):
        runs = {"a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64))}
        with self.assertRaises(ValueError):
            diagnose.compare_error_sets(runs)

    def test_class_support_is_reported(self):
        runs = {
            "a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
            "b": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        self.assertEqual(result["class_support"], 4)
        self.assertEqual(result["class_index"], 5)


class CliTest(unittest.TestCase):
    def test_main_writes_a_record_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            second = _write_run(Path(root), "b", [12, 12, 5, 5, 5, 12])
            output = Path(root) / "record.json"
            code = diagnose.main(
                ["--predictions", str(first), str(second), "--output", str(output)]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["class_index"], 5)
        self.assertEqual(payload["run_names"], ["a", "b"])

    def test_main_refuses_a_single_predictions_file(self):
        with tempfile.TemporaryDirectory() as root:
            only = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            output = Path(root) / "record.json"
            with self.assertRaises(ValueError):
                diagnose.main(["--predictions", str(only), "--output", str(output)])

    def test_main_refuses_to_overwrite_an_existing_record(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            second = _write_run(Path(root), "b", [12, 12, 5, 5, 5, 12])
            output = Path(root) / "record.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                diagnose.main(
                    ["--predictions", str(first), str(second), "--output", str(output)]
                )

    def test_record_carries_its_exploratory_status(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            second = _write_run(Path(root), "b", [12, 12, 5, 5, 5, 12])
            output = Path(root) / "record.json"
            diagnose.main(
                ["--predictions", str(first), str(second), "--output", str(output)]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        # A descriptive record that does not say it is descriptive will be
        # read as a result.
        self.assertEqual(payload["status"], "EXPLORATORY_DESCRIPTIVE")
        self.assertFalse(payload["may_advance_a_candidate"])
        self.assertFalse(payload["may_support_a_claim"])


if __name__ == "__main__":
    unittest.main()
