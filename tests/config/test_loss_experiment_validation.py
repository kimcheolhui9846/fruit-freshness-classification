"""Unit tests for key-level single-factor experiment config validation."""

from pathlib import Path
import tempfile
import unittest

from src.utils.config import (
    LOSS001_ALLOWED_DIFFERENCES,
    flatten_experiment_config,
    validate_loss_experiment_config,
)


BASELINE = "configs/deep3_postholdout_baseline.toml"
EXPERIMENT = "configs/deep3_postholdout_loss001.toml"
EXPECTED = {"loss.class_balanced_beta": 0.9999}


class FlattenExperimentConfigTest(unittest.TestCase):
    def test_nested_sections_become_dotted_keys(self):
        flat = flatten_experiment_config({"loss": {"beta": 0.999}, "training": {"epochs": 120}})

        self.assertEqual(flat, {"loss.beta": 0.999, "training.epochs": 120})

    def test_scalar_at_top_level_is_preserved(self):
        self.assertEqual(flatten_experiment_config({"seed": 42}), {"seed": 42})


class ValidateLossExperimentConfigTest(unittest.TestCase):
    def _tampered(self, tmp, old, new):
        path = Path(tmp) / "tampered.toml"
        path.write_text(
            Path(EXPERIMENT).read_text(encoding="utf-8").replace(old, new),
            encoding="utf-8",
        )
        return path

    def test_frozen_pair_passes_with_exactly_the_allowed_differences(self):
        result = validate_loss_experiment_config(
            BASELINE,
            EXPERIMENT,
            allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
            expected_values=EXPECTED,
        )

        self.assertTrue(result["single_factor_verified"])
        self.assertEqual(set(result["differences"]), set(LOSS001_ALLOWED_DIFFERENCES))
        self.assertEqual(
            result["differences"]["loss.class_balanced_beta"],
            {"baseline": 0.999, "experiment": 0.9999},
        )

    def test_an_extra_changed_key_is_rejected(self):
        # A second factor is exactly what "one factor at a time" forbids, and a
        # section-level comparison would not see it.
        with tempfile.TemporaryDirectory() as tmp:
            tampered = self._tampered(tmp, "focal_gamma = 2.0", "focal_gamma = 3.0")

            with self.assertRaises(ValueError) as caught:
                validate_loss_experiment_config(
                    BASELINE,
                    tampered,
                    allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
                    expected_values=EXPECTED,
                )
        self.assertIn("focal_gamma", str(caught.exception))

    def test_the_experimental_value_itself_is_pinned(self):
        # The allowed key carrying an unfrozen value is still an unregistered
        # experiment.
        with tempfile.TemporaryDirectory() as tmp:
            tampered = self._tampered(
                tmp, "class_balanced_beta = 0.9999", "class_balanced_beta = 0.99999"
            )

            with self.assertRaises(ValueError) as caught:
                validate_loss_experiment_config(
                    BASELINE,
                    tampered,
                    allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
                    expected_values=EXPECTED,
                )
        self.assertIn("0.9999", str(caught.exception))

    def test_an_identical_config_is_rejected_as_no_experiment(self):
        with self.assertRaises(ValueError):
            validate_loss_experiment_config(
                BASELINE,
                BASELINE,
                allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
                expected_values=EXPECTED,
            )


if __name__ == "__main__":
    unittest.main()
