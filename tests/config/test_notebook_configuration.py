import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path


CONFIG_PATH = Path("configs/deep3.toml")
NOTEBOOK_PATH = Path("deep3.ipynb")
BASE_NOTEBOOK_COMMIT = "479c36b"


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def find_main_source(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source or "pass")
        if any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
            return source
    raise AssertionError("main() cell was not found")


def config_lookup(node):
    keys = []
    while isinstance(node, ast.Subscript):
        if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str):
            return None
        keys.append(node.slice.value)
        node = node.value
    if not isinstance(node, ast.Name) or node.id != "config":
        return None
    return tuple(reversed(keys))


def main_assignments(main_source):
    main_node = next(
        node for node in ast.parse(main_source).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignments = {}
    for node in main_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node.value
    return assignments


class NotebookConfigurationWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook()
        cls.sources = ["".join(cell["source"]) for cell in cls.notebook["cells"] if cell["cell_type"] == "code"]
        cls.main_source = find_main_source(cls.notebook)
        cls.assignments = main_assignments(cls.main_source)

    def test_notebook_uses_one_explicit_relative_toml_path_and_loader_call(self):
        imports = ast.parse(self.sources[0])
        imported = {
            (node.module, alias.name)
            for node in imports.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn(("src.utils.config", "load_experiment_config"), imported)
        self.assertIn(("pathlib", "Path"), imported)
        self.assertIn('CONFIG_PATH = Path("configs/deep3.toml")', self.sources[0])
        self.assertEqual(self.main_source.count("load_experiment_config(CONFIG_PATH)"), 1)
        self.assertIn("config = load_experiment_config(CONFIG_PATH)", self.main_source)

    def test_existing_notebook_variable_names_receive_config_values(self):
        expected = {
            "beta": ("loss", "class_balanced_beta"),
            "EPOCHS": ("training", "epochs"),
            "FINETUNE_EPOCHS": ("fine_tuning", "epochs"),
            "BATCH_SIZE": ("training", "batch_size"),
            "K": ("cross_validation", "n_splits"),
            "MIXUP_ALPHA": ("mixup", "alpha"),
            "MIXUP_P": ("mixup", "probability"),
            "LR_CNN": ("optimization", "lr_cnn"),
            "LR_TRANS": ("optimization", "lr_trans"),
            "WEIGHT_DECAY": ("optimization", "weight_decay"),
            "EMA_DECAY": ("ema", "decay"),
            "USE_CE_LS": ("loss", "use_ce_label_smoothing"),
            "LABEL_SMOOTHING": ("loss", "label_smoothing"),
        }
        for name, lookup in expected.items():
            self.assertIn(name, self.assignments)
            self.assertEqual(config_lookup(self.assignments[name]), lookup, name)

    def test_config_values_preserve_orchestration_wiring(self):
        anchors = [
            'torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]',
            'shuffle=config["cross_validation"]["shuffle"]',
            'random_state=config["cross_validation"]["random_state"]',
            'gamma=config["loss"]["focal_gamma"]',
            'save_model_state(model, config["checkpoint"]["final_model_filename"])',
            'plt.figure(figsize=tuple(config["reporting"]["figure_size"]))',
            "for epoch in range(1, EPOCHS+1):",
            "is_finetuning = (epoch > EPOCHS - FINETUNE_EPOCHS)",
            "load_fold_models(K, num_classes, device, ckpt_dir)",
        ]
        for anchor in anchors:
            self.assertIn(anchor, self.main_source)

    def test_derived_values_and_original_environment_specific_path_remain_outside_config(self):
        self.assertIn("num_classes = len(final_dataset", self.main_source)
        self.assertIn("alpha = build_class_balanced_alpha(class_counts, beta, num_classes)", self.main_source)
        self.assertIn('ensure_output_directory("C:/Users/user/Desktop/deep/model_data")', self.main_source)
        self.assertNotIn("save_dir = config", self.main_source)

    def test_notebook_and_historical_notebooks_remain_structurally_valid(self):
        self.assertEqual(len(self.sources), 3)
        for source in self.sources:
            compile(source, "deep3.ipynb", "exec")
        for path in ("deep.ipynb", "deep1.ipynb", "deep2.ipynb"):
            baseline = subprocess.check_output(["git", "show", f"{BASE_NOTEBOOK_COMMIT}:{path}"])
            current = Path(path).read_bytes()
            self.assertEqual(
                hashlib.sha256(current.replace(b"\r\n", b"\n")).digest(),
                hashlib.sha256(baseline.replace(b"\r\n", b"\n")).digest(),
                path,
            )


if __name__ == "__main__":
    unittest.main()
