"""Validation of the optional Phase 9.7 runtime keys and the check lineage."""

import copy
from pathlib import Path
import tempfile
import tomllib
import unittest

from src.utils.config import (
    A_STRICT,
    B_CUDNN,
    C_SEED_ONLY,
    DETERMINISM_CHECK_ALLOWED_DIFFERENCES,
    DETERMINISM_CHECK_EXPECTED_VALUES,
    load_experiment_config,
    resolve_experiment_validation,
)


ROOT = Path(__file__).resolve().parents[2]
CHECK_CONFIG = ROOT / "configs" / "deep3_postholdout_determinism_check.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"


def _minimal_runtime(**overrides) -> dict:
    runtime = {"cudnn_benchmark": True}
    runtime.update(overrides)
    return runtime


def _config_with_runtime(runtime: dict) -> dict:
    config = copy.deepcopy(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))
    config["runtime"] = runtime
    return config


class OptionalRuntimeKeyTest(unittest.TestCase):
    """Driven through a written file, because that is the only path
    production code uses to reach the validator."""

    def _write(self, config: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "candidate.toml"
        lines = []
        for section, values in config.items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                elif isinstance(value, str):
                    rendered = f'"{value}"'
                elif isinstance(value, list):
                    rendered = "[" + ", ".join(str(item) for item in value) + "]"
                else:
                    rendered = repr(value)
                lines.append(f"{key} = {rendered}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_config_without_the_new_keys_still_loads(self):
        path = self._write(_config_with_runtime(_minimal_runtime()))
        config = load_experiment_config(path)

        # Optional means optional. Every frozen config lacks these keys.
        self.assertNotIn("seed", config["runtime"])
        self.assertNotIn("determinism_level", config["runtime"])

    def test_seed_without_a_level_is_rejected(self):
        path = self._write(_config_with_runtime(_minimal_runtime(seed=20260815)))
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_level_without_a_seed_is_rejected(self):
        path = self._write(
            _config_with_runtime(_minimal_runtime(determinism_level=B_CUDNN))
        )
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_unknown_level_name_is_rejected(self):
        path = self._write(
            _config_with_runtime(
                _minimal_runtime(seed=20260815, determinism_level="MOSTLY")
            )
        )
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_negative_seed_is_rejected(self):
        path = self._write(
            _config_with_runtime(_minimal_runtime(seed=-1, determinism_level=B_CUDNN))
        )
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_constrained_levels_reject_the_autotuner(self):
        for level in (A_STRICT, B_CUDNN):
            with self.subTest(level=level):
                path = self._write(
                    _config_with_runtime(
                        {
                            "cudnn_benchmark": True,
                            "seed": 20260815,
                            "determinism_level": level,
                        }
                    )
                )
                # A config that asks for determinism and leaves the autotuner
                # on contradicts itself; the autotuner picks kernels
                # nondeterministically.
                with self.assertRaises(ValueError):
                    load_experiment_config(path)

    def test_seed_only_level_permits_the_autotuner(self):
        path = self._write(
            _config_with_runtime(
                {
                    "cudnn_benchmark": True,
                    "seed": 20260815,
                    "determinism_level": C_SEED_ONLY,
                }
            )
        )
        config = load_experiment_config(path)

        self.assertEqual(config["runtime"]["determinism_level"], C_SEED_ONLY)


class DeterminismCheckConfigTest(unittest.TestCase):
    def test_check_config_uses_the_frozen_development_route(self):
        config = load_experiment_config(CHECK_CONFIG)

        # The canonical route trains on the 4,298 locked-test examples.
        self.assertIn("post_holdout", config)
        self.assertEqual(
            config["post_holdout"]["split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(
            config["post_holdout"]["cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )

    def test_check_config_is_bounded_and_exercises_fine_tuning(self):
        config = load_experiment_config(CHECK_CONFIG)

        # epochs 2 with fine_tuning 1 puts one normal epoch and one
        # fine-tuning epoch in every fold.
        self.assertEqual(config["training"]["epochs"], 2)
        self.assertEqual(config["fine_tuning"]["epochs"], 1)
        self.assertEqual(config["training"]["batch_size"], 64)

    def test_check_config_carries_the_frozen_seed(self):
        config = load_experiment_config(CHECK_CONFIG)

        self.assertEqual(config["runtime"]["seed"], 20260815)
        self.assertFalse(config["runtime"]["cudnn_benchmark"])

    def test_check_lineage_validates_and_names_its_differences(self):
        config = load_experiment_config(CHECK_CONFIG)
        result = resolve_experiment_validation(config, CHECK_CONFIG)

        self.assertIsNotNone(result)
        self.assertTrue(result["single_factor_verified"])
        self.assertEqual(
            set(result["differences"]),
            {
                "runtime.cudnn_benchmark",
                "runtime.seed",
                "runtime.determinism_level",
                "training.epochs",
                "fine_tuning.epochs",
                "post_holdout.experiment_id",
                "post_holdout.parent_experiment_id",
                "post_holdout.artifact_namespace",
            },
        )

    def test_check_lineage_does_not_pin_the_level(self):
        # The frozen ladder descends from A_STRICT to B_CUDNN. The level must
        # be a registered difference, so the check config is allowed to name
        # it, but it must not be a pinned value, or the registered descent
        # could never be executed.
        self.assertIn("runtime.determinism_level", DETERMINISM_CHECK_ALLOWED_DIFFERENCES)
        self.assertNotIn("runtime.determinism_level", DETERMINISM_CHECK_EXPECTED_VALUES)

    def test_unregistered_parent_still_raises(self):
        config = load_experiment_config(CHECK_CONFIG)
        config["post_holdout"]["parent_experiment_id"] = "deep3-unregistered"

        with self.assertRaises(ValueError):
            resolve_experiment_validation(config, CHECK_CONFIG)


if __name__ == "__main__":
    unittest.main()
