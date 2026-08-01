import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location("phase54_evaluate_checkpoints", EVALUATE_PATH)
evaluate = importlib.util.module_from_spec(EVALUATE_SPEC)
sys.modules[EVALUATE_SPEC.name] = evaluate
EVALUATE_SPEC.loader.exec_module(evaluate)


class EvaluateCheckpointValidationTest(unittest.TestCase):
    def test_missing_directory_is_rejected_without_loading_optional_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                evaluate.resolve_fold_checkpoint_paths(missing, 3)

    def test_file_and_empty_directory_have_distinct_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file_path = root / "checkpoint.pt"
            file_path.touch()
            with self.assertRaisesRegex(NotADirectoryError, "not a directory"):
                evaluate.resolve_fold_checkpoint_paths(file_path, 3)

            empty_directory = root / "empty"
            empty_directory.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "empty"):
                evaluate.resolve_fold_checkpoint_paths(empty_directory, 3)

    def test_incomplete_directory_is_rejected_in_required_fold_order(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            (checkpoint_dir / "best_model_fold1.pt").touch()
            with self.assertRaisesRegex(FileNotFoundError, "best_model_fold2.pt") as error:
                evaluate.resolve_fold_checkpoint_paths(checkpoint_dir, 3)
            self.assertIn("best_model_fold3.pt", str(error.exception))

    def test_complete_directory_returns_existing_template_paths_in_ascending_order(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            expected_paths = []
            for fold in range(1, 4):
                path = checkpoint_dir / f"best_model_fold{fold}.pt"
                path.touch()
                expected_paths.append(path)

            self.assertEqual(
                evaluate.resolve_fold_checkpoint_paths(checkpoint_dir, 3),
                expected_paths,
            )


if __name__ == "__main__":
    unittest.main()