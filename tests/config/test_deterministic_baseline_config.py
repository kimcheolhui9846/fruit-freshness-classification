"""The deterministic baseline configuration and its registered lineage."""

from pathlib import Path
import tomllib
import unittest

from src.utils.config import (
    DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES,
    DETERMINISTIC_BASELINE_EXPECTED_VALUES,
    load_experiment_config,
    resolve_experiment_validation,
)


ROOT = Path(__file__).resolve().parents[2]
DET_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline_det.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"

EXPECTED_DIFFERENCES = {
    "runtime.cudnn_benchmark",
    "runtime.seed",
    "runtime.determinism_level",
    "post_holdout.experiment_id",
    "post_holdout.parent_experiment_id",
    "post_holdout.artifact_namespace",
}


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class DeterministicBaselineConfigTest(unittest.TestCase):
    def test_recipe_differs_only_in_determinism_and_identity(self):
        baseline = _flatten(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))
        det = _flatten(tomllib.loads(DET_CONFIG.read_text(encoding="utf-8")))

        differing = {
            key for key in set(baseline) | set(det) if baseline.get(key) != det.get(key)
        }
        # A deterministic baseline that also changed the recipe would not be
        # a baseline for the recipe.
        self.assertEqual(differing, EXPECTED_DIFFERENCES)

    def test_full_length_schedule_is_preserved(self):
        det = _flatten(tomllib.loads(DET_CONFIG.read_text(encoding="utf-8")))

        # The Phase 9.7 check config was bounded at 2 epochs. This one is not
        # a smoke test; it must run the real schedule.
        self.assertEqual(det["training.epochs"], 120)
        self.assertEqual(det["fine_tuning.epochs"], 20)
        self.assertEqual(det["training.batch_size"], 64)

    def test_frozen_folds_are_reused(self):
        det = _flatten(tomllib.loads(DET_CONFIG.read_text(encoding="utf-8")))

        self.assertEqual(
            det["post_holdout.split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(
            det["post_holdout.cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )
        self.assertEqual(det["cross_validation.random_state"], 42)
        self.assertEqual(det["cross_validation.n_splits"], 3)

    def test_adopted_determinism_settings_are_carried(self):
        config = load_experiment_config(DET_CONFIG)

        self.assertEqual(config["runtime"]["seed"], 20260815)
        self.assertEqual(config["runtime"]["determinism_level"], "A_STRICT")
        self.assertFalse(config["runtime"]["cudnn_benchmark"])

    def test_lineage_validates_and_names_its_differences(self):
        config = load_experiment_config(DET_CONFIG)
        result = resolve_experiment_validation(config, DET_CONFIG)

        self.assertIsNotNone(result)
        self.assertTrue(result["single_factor_verified"])
        self.assertEqual(set(result["differences"]), EXPECTED_DIFFERENCES)

    def test_lineage_pins_the_adopted_level(self):
        # The Phase 9.7 ladder is finished and A_STRICT is adopted, so unlike
        # the check config there is no descent left to keep available.
        self.assertEqual(
            DETERMINISTIC_BASELINE_EXPECTED_VALUES["runtime.determinism_level"],
            "A_STRICT",
        )
        self.assertIn(
            "runtime.determinism_level", DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES
        )

    def test_unregistered_parent_still_raises(self):
        config = load_experiment_config(DET_CONFIG)
        config["post_holdout"]["parent_experiment_id"] = "deep3-unregistered"

        with self.assertRaises(ValueError):
            resolve_experiment_validation(config, DET_CONFIG)


if __name__ == "__main__":
    unittest.main()
