import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("phase53_train_cli", TRAIN_PATH)
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


class TrainCliTest(unittest.TestCase):
    def test_parser_defaults_are_repository_relative_experiment_paths(self):
        args = train.build_parser().parse_args([])

        self.assertEqual(args.config, Path("configs/deep3.toml"))
        self.assertEqual(args.output_dir, Path("weights"))
        self.assertEqual(
            train._resolve_repository_path(args.config),
            train.REPOSITORY_ROOT / "configs/deep3.toml",
        )
        self.assertEqual(
            train._resolve_repository_path(args.output_dir),
            train.REPOSITORY_ROOT / "weights",
        )

    def test_parser_accepts_only_execution_level_arguments(self):
        args = train.build_parser().parse_args(
            ["--config", "configs/custom.toml", "--output-dir", "results/run-a"],
        )

        self.assertEqual(args.config, Path("configs/custom.toml"))
        self.assertEqual(args.output_dir, Path("results/run-a"))
        self.assertEqual(set(vars(args)), {"config", "output_dir"})

    def test_help_and_unknown_arguments_follow_argparse_contract(self):
        parser = train.build_parser()
        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output), self.assertRaises(SystemExit) as help_exit:
            parser.parse_args(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        self.assertIn("--config", help_output.getvalue())
        self.assertIn("--output-dir", help_output.getvalue())

        with self.assertRaises(SystemExit) as unknown_exit:
            parser.parse_args(["--epochs", "1"])
        self.assertEqual(unknown_exit.exception.code, 2)

    def test_invalid_config_path_fails_before_creating_the_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    train.main(
                        [
                            "--config",
                            "configs/missing.toml",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )
            self.assertFalse(output_dir.exists())

    def test_absolute_output_path_is_preserved_as_an_explicit_user_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory)
            self.assertEqual(train._resolve_repository_path(selected), selected)


if __name__ == "__main__":
    unittest.main()
