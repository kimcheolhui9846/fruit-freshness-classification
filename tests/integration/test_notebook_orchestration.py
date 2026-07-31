import ast
import hashlib
import json
import subprocess
import unittest
from pathlib import Path


BASE_NOTEBOOK_COMMIT = "e23d601"
NOTEBOOK_PATH = Path("deep3.ipynb")


def load_notebook(path_or_text):
    text = path_or_text.read_text(encoding="utf-8") if isinstance(path_or_text, Path) else path_or_text
    return json.loads(text)


def code_sources(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def find_main_source(sources):
    for source in sources:
        tree = ast.parse(source or "pass")
        if any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
            return source
    raise AssertionError("main() orchestration cell was not found")


def assignment_signature(source):
    main_node = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    names = {
        "beta",
        "EPOCHS",
        "FINETUNE_EPOCHS",
        "BATCH_SIZE",
        "K",
        "MIXUP_ALPHA",
        "MIXUP_P",
        "LR_CNN",
        "LR_TRANS",
        "WEIGHT_DECAY",
        "EMA_DECAY",
        "USE_CE_LS",
        "LABEL_SMOOTHING",
    }
    signature = []
    for node in main_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                signature.append((node.targets[0].id, ast.dump(node.value, include_attributes=False)))
    return signature


class NotebookOrchestrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook(NOTEBOOK_PATH)
        cls.sources = code_sources(cls.notebook)
        cls.main_source = find_main_source(cls.sources)
        baseline_text = subprocess.check_output(
            ["git", "show", f"{BASE_NOTEBOOK_COMMIT}:deep3.ipynb"],
            text=True,
            encoding="utf-8",
        )
        cls.baseline = load_notebook(baseline_text)
        cls.baseline_main_source = find_main_source(code_sources(cls.baseline))

    def test_notebook_is_compilable_and_has_no_empty_or_comment_only_code_cells(self):
        self.assertEqual(len(self.sources), 3)
        for source in self.sources:
            self.assertTrue(source.strip())
            self.assertNotEqual(source.strip(), "# Inference helpers are provided by src.inference.")
            compile(source, "deep3.ipynb", "exec")

    def test_imports_are_explicit_and_only_main_remains_as_a_notebook_definition(self):
        imports = ast.parse(self.sources[0])
        imported = {
            (node.module, alias.name)
            for node in imports.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        expected = {
            ("src.utils.runtime", "resolve_device"),
            ("src.datasets.fruit_freshness", "load_fruit_freshness_dataset"),
            ("src.transforms.classification", "build_train_transform"),
            ("src.models.factory", "build_cmt_classifier"),
            ("src.trainers.loops", "train_one_epoch"),
            ("src.evaluation.metrics", "compute_validation_metrics"),
            ("src.inference.loading", "load_fold_models"),
            ("src.inference.ensemble", "run_ensemble_holdout"),
        }
        self.assertTrue(expected.issubset(imported))
        definitions = []
        for source in self.sources:
            tree = ast.parse(source)
            definitions.extend(
                node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            )
            self.assertFalse(any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)))
        self.assertEqual(definitions, ["main"])

    def test_outer_orchestration_and_hyperparameters_remain_at_the_notebook_layer(self):
        anchors = [
            "for fold, (train_idx, val_idx) in enumerate(",
            "for epoch in range(1, EPOCHS+1):",
            "is_finetuning = (epoch > EPOCHS - FINETUNE_EPOCHS)",
            "train_one_epoch(",
            "validate_one_epoch(",
            "if va_acc > best_acc_fold + 1e-6:",
            "scheduler.step()",
            "load_fold_models(K, num_classes, device, ckpt_dir)",
            "run_ensemble_holdout(models, test_loader, device)",
            'print("Final Holdout Acc:", t_correct / t_total)',
        ]
        for anchor in anchors:
            self.assertIn(anchor, self.main_source)
        self.assertEqual(
            assignment_signature(self.main_source),
            assignment_signature(self.baseline_main_source),
        )

    def test_completed_implementation_duplicates_are_absent(self):
        prohibited = [
            "torch.utils.data.DataLoader",
            "torch.optim.AdamW",
            "torch.optim.lr_scheduler",
            "torch.save(",
            "torch.load(",
            "load_state_dict(",
            "state_dict(",
            "f1_score(",
            "balanced_accuracy_score(",
            "top_k_accuracy_score(",
            "torch.flip(",
            "def load_fold_models",
            "def ensemble_logits",
            "def train_one_epoch",
            "def validate_one_epoch",
            "class ModelEma",
            "class FocalLoss",
            "class FruitHFDataset",
            "class CMTClassifier",
        ]
        notebook_source = "\n".join(self.sources)
        for marker in prohibited:
            self.assertNotIn(marker, notebook_source, marker)

    def test_historical_notebooks_and_gitignore_match_the_phase_baseline(self):
        for path in ("deep.ipynb", "deep1.ipynb", "deep2.ipynb", ".gitignore"):
            baseline = subprocess.check_output(
                ["git", "show", f"{BASE_NOTEBOOK_COMMIT}:{path}"],
            )
            current = Path(path).read_bytes()
            current_normalized = current.replace(b"\r\n", b"\n")
            baseline_normalized = baseline.replace(b"\r\n", b"\n")
            self.assertEqual(
                hashlib.sha256(current_normalized).digest(),
                hashlib.sha256(baseline_normalized).digest(),
                path,
            )
            self.assertEqual(
                subprocess.run(
                    ["git", "diff", "--quiet", BASE_NOTEBOOK_COMMIT, "--", path],
                    check=False,
                ).returncode,
                0,
                path,
            )


if __name__ == "__main__":
    unittest.main()