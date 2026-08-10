"""Configuration contracts for the Phase 9.3 development baseline."""

from pathlib import Path
import tempfile
import unittest

from src.utils.config import (
    baseline_recipe_differences,
    load_experiment_config,
    validate_postholdout_baseline_config,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PATH = ROOT / "configs" / "deep3_canonical.toml"
BASELINE_PATH = ROOT / "configs" / "deep3_postholdout_baseline.toml"


class PostHoldoutBaselineConfigTest(unittest.TestCase):
    def test_baseline_preserves_all_canonical_recipe_values(self):
        canonical = load_experiment_config(CANONICAL_PATH)
        baseline = load_experiment_config(BASELINE_PATH)

        self.assertEqual(
            baseline_recipe_differences(canonical, baseline),
            {
                "post_holdout": {
                    "artifact_namespace": "deep3-postholdout-research-01-baseline",
                    "cv_manifest_path": "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
                    "experiment_id": "deep3-postholdout-research-01-baseline",
                    "parent_experiment_id": "deep3-postholdout-research-01",
                    "split_manifest_path": "configs/splits/deep3-postholdout-research-01.json",
                }
            },
        )
        validated = validate_postholdout_baseline_config(CANONICAL_PATH, BASELINE_PATH)
        self.assertTrue(validated["recipe_equivalent"])

    def test_changed_training_value_is_rejected(self):
        modified = BASELINE_PATH.read_text(encoding="utf-8").replace(
            "batch_size = 64",
            "batch_size = 32",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.toml"
            path.write_text(modified, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recipe"):
                validate_postholdout_baseline_config(CANONICAL_PATH, path)


if __name__ == "__main__":
    unittest.main()
