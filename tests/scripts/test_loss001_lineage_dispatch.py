"""Contract for routing a post-holdout config to the right ancestor check."""

import unittest

from src.utils.config import load_experiment_config, resolve_experiment_validation


class ResolveExperimentValidationTest(unittest.TestCase):
    LOSS001 = "configs/deep3_postholdout_loss001.toml"
    BASELINE = "configs/deep3_postholdout_baseline.toml"

    def test_baseline_parented_config_is_validated_against_the_baseline(self):
        config = load_experiment_config(self.LOSS001)

        result = resolve_experiment_validation(config, self.LOSS001)

        self.assertIsNotNone(result)
        self.assertTrue(result["single_factor_verified"])
        self.assertIn("loss.class_balanced_beta", result["differences"])

    def test_canonical_parented_config_keeps_the_existing_path(self):
        config = load_experiment_config(self.BASELINE)

        # The baseline is parented to the research identity, not to itself, so
        # it must not be routed through the experiment validator.
        self.assertIsNone(resolve_experiment_validation(config, self.BASELINE))

    def test_unknown_parent_is_rejected_rather_than_silently_skipped(self):
        config = load_experiment_config(self.LOSS001)
        config["post_holdout"]["parent_experiment_id"] = "something-unregistered"

        # Silently skipping validation is how an unregistered experiment runs.
        with self.assertRaises(ValueError):
            resolve_experiment_validation(config, self.LOSS001)


if __name__ == "__main__":
    unittest.main()
