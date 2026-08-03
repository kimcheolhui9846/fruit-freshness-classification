import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("phase82_train_cli", TRAIN_PATH)
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


class TrainCliTest(unittest.TestCase):
    def test_parser_defaults_keep_legacy_execution_without_resume_state(self):
        args = train.build_parser().parse_args([])

        self.assertEqual(args.config, Path("configs/deep3.toml"))
        self.assertEqual(args.output_dir, Path("weights"))
        self.assertIsNone(args.resume_state)
        self.assertFalse(args.save_training_state)
        self.assertFalse(args.require_empty_output_dir)
        self.assertIsNone(args.run_id)
        self.assertEqual(
            train._resolve_repository_path(args.config),
            train.REPOSITORY_ROOT / "configs/deep3.toml",
        )
        self.assertEqual(
            train._resolve_repository_path(args.output_dir),
            train.REPOSITORY_ROOT / "weights",
        )

    def test_parser_accepts_only_approved_execution_level_resume_controls(self):
        args = train.build_parser().parse_args(
            [
                "--config",
                "configs/deep3_canonical.toml",
                "--output-dir",
                "weights/run-a",
                "--resume-state",
                "weights/run-a/training_state.pt",
                "--save-training-state",
                "--require-empty-output-dir",
                "--run-id",
                "run-a",
            ]
        )

        self.assertEqual(args.config, Path("configs/deep3_canonical.toml"))
        self.assertEqual(args.output_dir, Path("weights/run-a"))
        self.assertEqual(args.resume_state, Path("weights/run-a/training_state.pt"))
        self.assertTrue(args.save_training_state)
        self.assertTrue(args.require_empty_output_dir)
        self.assertEqual(args.run_id, "run-a")
        self.assertEqual(
            set(vars(args)),
            {
                "config",
                "output_dir",
                "resume_state",
                "save_training_state",
                "require_empty_output_dir",
                "run_id",
            },
        )

    def test_help_and_unknown_arguments_follow_argparse_contract(self):
        parser = train.build_parser()
        help_output = io.StringIO()
        with contextlib.redirect_stdout(help_output), self.assertRaises(SystemExit) as help_exit:
            parser.parse_args(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        for option in (
            "--config",
            "--output-dir",
            "--resume-state",
            "--save-training-state",
            "--require-empty-output-dir",
            "--run-id",
        ):
            self.assertIn(option, help_output.getvalue())

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
                            "--require-empty-output-dir",
                            "--save-training-state",
                            "--run-id",
                            "run-a",
                        ]
                    )
            self.assertFalse(output_dir.exists())

    def test_absolute_output_path_is_preserved_as_an_explicit_user_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory)
            self.assertEqual(train._resolve_repository_path(selected), selected)


if __name__ == "__main__":
    unittest.main()
