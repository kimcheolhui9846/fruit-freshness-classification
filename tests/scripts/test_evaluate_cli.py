import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


EVALUATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location("phase54_evaluate_cli", EVALUATE_PATH)
evaluate = importlib.util.module_from_spec(EVALUATE_SPEC)
sys.modules[EVALUATE_SPEC.name] = evaluate
EVALUATE_SPEC.loader.exec_module(evaluate)


class EvaluateCliTest(unittest.TestCase):
    def test_parser_uses_the_committed_config_and_requires_checkpoints(self):
        parser = evaluate.build_parser()
        with self.assertRaises(SystemExit) as missing_checkpoint_exit:
            parser.parse_args([])
        self.assertEqual(missing_checkpoint_exit.exception.code, 2)

        args = parser.parse_args(["--checkpoint-dir", "weights/run-a"])
        self.assertEqual(args.config, Path("configs/deep3.toml"))
        self.assertEqual(args.checkpoint_dir, Path("weights/run-a"))
        self.assertEqual(set(vars(args)), {"config", "checkpoint_dir"})
        self.assertEqual(
            evaluate._resolve_repository_path(args.config),
            evaluate.REPOSITORY_ROOT / "configs/deep3.toml",
        )
        self.assertEqual(
            evaluate._resolve_repository_path(args.checkpoint_dir),
            evaluate.REPOSITORY_ROOT / "weights/run-a",
        )

    def test_parser_accepts_only_evaluation_level_arguments(self):
        args = evaluate.build_parser().parse_args(
            ["--config", "configs/custom.toml", "--checkpoint-dir", "results/run-a"],
        )
        self.assertEqual(args.config, Path("configs/custom.toml"))
        self.assertEqual(args.checkpoint_dir, Path("results/run-a"))
        self.assertEqual(set(vars(args)), {"config", "checkpoint_dir"})

    def test_help_and_unknown_arguments_follow_argparse_contract(self):
        parser = evaluate.build_parser()
        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output), self.assertRaises(SystemExit) as help_exit:
            parser.parse_args(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertIn("--config", help_output.getvalue())
        self.assertIn("--checkpoint-dir", help_output.getvalue())

        with self.assertRaises(SystemExit) as unknown_exit:
            parser.parse_args(["--checkpoint-dir", "weights", "--device", "cpu"])
        self.assertEqual(unknown_exit.exception.code, 2)

    def test_invalid_config_fails_before_checkpoint_or_dataset_access(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    evaluate.main(
                        [
                            "--config",
                            "configs/missing.toml",
                            "--checkpoint-dir",
                            str(checkpoint_dir),
                        ]
                    )

    def test_absolute_checkpoint_path_is_preserved_as_an_explicit_user_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory)
            self.assertEqual(evaluate._resolve_repository_path(selected), selected)


if __name__ == "__main__":
    unittest.main()