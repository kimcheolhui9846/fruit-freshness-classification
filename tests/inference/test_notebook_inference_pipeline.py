import json
import subprocess
import unittest
from pathlib import Path


LEGACY_NOTEBOOK_COMMIT = "6e0bcca"
NOTEBOOK_PATH = Path("deep3.ipynb")


def load_cells(notebook_text):
    return ["".join(cell["source"]) for cell in json.loads(notebook_text)["cells"]]


class NotebookInferencePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
        cls.current_cells = load_cells(cls.current_text)
        legacy_text = subprocess.check_output(
            ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:deep3.ipynb"],
            text=True,
            encoding="utf-8",
        )
        cls.legacy_cells = load_cells(legacy_text)

    def test_notebook_is_valid_and_inference_imports_are_explicit(self):
        self.assertEqual(len(self.current_cells), len(self.legacy_cells) - 2)
        for source in self.current_cells:
            compile(source, "deep3.ipynb", "exec")
        imports = self.current_cells[0]
        self.assertIn("from src.inference.loading import load_fold_models", imports)
        self.assertIn("from src.inference.ensemble import run_ensemble_holdout", imports)

    def test_active_final_inference_is_module_backed(self):
        legacy = self.legacy_cells[3] + self.legacy_cells[4]
        current = self.current_cells[2]
        self.assertIn("def load_fold_models", legacy)
        self.assertIn("def ensemble_logits", legacy)
        self.assertIn("def ensemble_logits_tta_hflip", legacy)
        self.assertIn("for x, y in tqdm(test_loader, ncols=100):", legacy)
        self.assertNotIn("def load_fold_models", current)
        self.assertNotIn("def ensemble_logits", current)
        self.assertNotIn("for x, y in tqdm(test_loader, ncols=100):", current)
        self.assertIn("run_ensemble_holdout(models, test_loader, device)", current)

    def test_orchestration_reporting_and_completed_boundaries_are_preserved(self):
        legacy = self.legacy_cells[4]
        current = self.current_cells[2]
        anchors = [
            "ckpt_dir = save_dir",
            "models = load_fold_models(K, num_classes, device, ckpt_dir)",
            'test_ds = FruitHFDataset(final_dataset["test"], transform=val_transform)',
            "test_loader = build_holdout_dataloader(test_ds, BATCH_SIZE)",
            'print("Final Holdout Acc:", t_correct / t_total)',
            "total_time = time.time() - start_time",
            "scheduler.step()",
        ]
        for anchor in anchors:
            self.assertEqual(legacy.count(anchor), current.count(anchor), anchor)
        self.assertIn('print("\\n[최종 평가] Holdout Test Set (Ensemble + TTA)")', current)
        self.assertLess(
            current.index('print("\\n[최종 평가] Holdout Test Set (Ensemble + TTA)")'),
            current.index("run_ensemble_holdout(models, test_loader, device)"),
        )

    def test_completed_trainer_and_evaluation_modules_are_unchanged(self):
        for path in ("src/trainers/loops.py", "src/evaluation/metrics.py"):
            legacy = subprocess.check_output(
                ["git", "show", f"{LEGACY_NOTEBOOK_COMMIT}:{path}"],
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(Path(path).read_text(encoding="utf-8"), legacy, path)


if __name__ == "__main__":
    unittest.main()
