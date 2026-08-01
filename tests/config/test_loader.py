import tempfile
import unittest
from pathlib import Path

from src.utils.config import load_experiment_config


CONFIG_PATH = Path("configs/deep3.toml")


class ExperimentConfigLoaderTest(unittest.TestCase):
    def test_loads_the_committed_toml_with_expected_container_types(self):
        config = load_experiment_config(CONFIG_PATH)

        self.assertIsInstance(config, dict)
        self.assertIs(type(config["training"]["epochs"]), int)
        self.assertIs(type(config["loss"]["class_balanced_beta"]), float)
        self.assertIs(type(config["runtime"]["cudnn_benchmark"]), bool)
        self.assertIs(type(config["reporting"]["figure_size"]), list)
        self.assertEqual(config["reporting"]["figure_size"], [10, 4])

    def test_accepts_string_and_path_inputs_without_caching(self):
        from_path = load_experiment_config(CONFIG_PATH)
        from_string = load_experiment_config(str(CONFIG_PATH))

        self.assertEqual(from_path, from_string)
        self.assertIsNot(from_path, from_string)

    def test_missing_required_key_fails_explicitly_without_default(self):
        text = CONFIG_PATH.read_text(encoding="utf-8").replace("epochs = 120\n", "", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-key.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(KeyError, r"\[training\]\.epochs"):
                load_experiment_config(path)

    def test_wrong_scalar_type_is_not_coerced(self):
        text = CONFIG_PATH.read_text(encoding="utf-8").replace("batch_size = 192", 'batch_size = "192"', 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-type.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(TypeError, r"\[training\]\.batch_size"):
                load_experiment_config(path)

    def test_absolute_checkpoint_filename_is_rejected(self):
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            'final_model_filename = "last_model_weights.pt"',
            'final_model_filename = "C:/temporary/last_model_weights.pt"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absolute-filename.toml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "portable filename"):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
