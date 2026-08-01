import re
import unittest
from pathlib import Path

from src.utils.config import load_experiment_config


CONFIG_PATH = Path("configs/deep3.toml")

# Baseline captured from deep3.ipynb before Phase 5.2 wiring.
BASELINE_EXPERIMENT_VALUES = {
    "runtime": {"cudnn_benchmark": True},
    "loss": {
        "class_balanced_beta": 0.999,
        "use_ce_label_smoothing": True,
        "label_smoothing": 0.01,
        "focal_gamma": 2.0,
    },
    "training": {"epochs": 120, "batch_size": 192},
    "fine_tuning": {"epochs": 20},
    "cross_validation": {"n_splits": 3, "shuffle": True, "random_state": 42},
    "mixup": {"alpha": 0.8, "probability": 0.5},
    "optimization": {"lr_cnn": 5e-5, "lr_trans": 1e-4, "weight_decay": 1e-4},
    "ema": {"decay": 0.999},
    "checkpoint": {"final_model_filename": "last_model_weights.pt"},
    "reporting": {"figure_size": [10, 4]},
}

EXPECTED_TYPES = {
    "runtime": {"cudnn_benchmark": bool},
    "loss": {
        "class_balanced_beta": float,
        "use_ce_label_smoothing": bool,
        "label_smoothing": float,
        "focal_gamma": float,
    },
    "training": {"epochs": int, "batch_size": int},
    "fine_tuning": {"epochs": int},
    "cross_validation": {"n_splits": int, "shuffle": bool, "random_state": int},
    "mixup": {"alpha": float, "probability": float},
    "optimization": {"lr_cnn": float, "lr_trans": float, "weight_decay": float},
    "ema": {"decay": float},
    "checkpoint": {"final_model_filename": str},
    "reporting": {"figure_size": list},
}


class ExperimentValueParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_experiment_config(CONFIG_PATH)

    def test_config_matches_the_pre_wiring_value_baseline_exactly(self):
        self.assertEqual(self.config, BASELINE_EXPERIMENT_VALUES)

    def test_config_preserves_every_baseline_scalar_type(self):
        for section, keys in EXPECTED_TYPES.items():
            for key, expected_type in keys.items():
                self.assertIs(type(self.config[section][key]), expected_type, f"[{section}].{key}")
        self.assertTrue(all(type(value) is int for value in self.config["reporting"]["figure_size"]))

    def test_configuration_excludes_derived_and_runtime_state(self):
        forbidden_paths = {
            ("runtime", "device"),
            ("loss", "class_counts"),
            ("loss", "class_balanced_alpha"),
            ("training", "num_classes"),
            ("training", "current_epoch"),
            ("cross_validation", "fold_indices"),
            ("checkpoint", "save_dir"),
            ("checkpoint", "ckpt_dir"),
            ("checkpoint", "resolved_path"),
            ("evaluation", "best_acc_fold"),
            ("evaluation", "history"),
            ("inference", "models"),
            ("inference", "dataloader_length"),
        }
        actual_paths = {
            (section_name, key)
            for section_name, section in self.config.items()
            for key in section
        }
        self.assertTrue(forbidden_paths.isdisjoint(actual_paths))

    def test_committed_config_has_no_secret_or_machine_specific_path(self):
        text = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"(?i)(ctx7sk-|ghp_|github_pat_|hf_[A-Za-z0-9]|C:\\Users\\|file://)")
        self.assertNotRegex(text, r"(?i)(localhost|127\.0\.0\.1|cuda:\d|cache|temporary)")


if __name__ == "__main__":
    unittest.main()
