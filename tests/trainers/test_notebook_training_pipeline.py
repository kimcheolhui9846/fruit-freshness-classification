import json
import subprocess
import unittest
from pathlib import Path


LEGACY_NOTEBOOK_COMMIT = "0f89baa"
NOTEBOOK_PATH = Path("deep3.ipynb")


def load_cells(notebook_text):
    return ["".join(cell["source"]) for cell in json.loads(notebook_text)["cells"]]


class NotebookTrainingPipelineTest(unittest.TestCase):
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

    def test_notebook_is_valid_and_trainer_imports_are_explicit(self):
        self.assertEqual(len(self.current_cells), len(self.legacy_cells))
        for source in self.current_cells:
            compile(source, "deep3.ipynb", "exec")
        self.assertIn(
            "from src.trainers.loops import train_one_epoch, validate_one_epoch",
            self.current_cells[0],
        )

    def test_active_single_epoch_loops_are_module_backed(self):
        legacy = self.legacy_cells[4]
        current = self.current_cells[4]
        self.assertIn("for x, y in pbar:", legacy)
        self.assertNotIn("for x, y in pbar:", current)
        self.assertIn("for x, y in tqdm(val_loader", legacy)
        self.assertNotIn("for x, y in tqdm(val_loader", current)
        self.assertNotIn("optimizer.zero_grad(set_to_none=True)", current)
        self.assertIn("train_one_epoch(", current)
        self.assertIn("validate_one_epoch(", current)

    def test_outer_orchestration_boundaries_are_preserved(self):
        legacy = self.legacy_cells[4]
        current = self.current_cells[4]
        anchors = [
            "for fold, (train_idx, val_idx) in enumerate(",
            "for epoch in range(1, EPOCHS+1):",
            "is_finetuning = (epoch > EPOCHS - FINETUNE_EPOCHS)",
            "train_ds.tf = ft_transform",
            "history[\"train_loss\"].append(tr_loss)",
            "if va_acc > best_acc_fold + 1e-6:",
            "save_model_state(ema.module, save_path)",
            "scheduler.step()",
            "models = load_fold_models(K, num_classes, device, ckpt_dir)",
        ]
        for anchor in anchors:
            self.assertEqual(legacy.count(anchor), current.count(anchor), anchor)
        self.assertLess(
            current.index("if va_acc > best_acc_fold + 1e-6:"),
            current.index("scheduler.step()"),
        )
        self.assertIn("for x, y in tqdm(test_loader, ncols=100):", legacy)
        self.assertIn("run_ensemble_holdout(models, test_loader, device)", current)

        self.assertNotIn("scheduler.step()", self.current_cells[0])


if __name__ == "__main__":
    unittest.main()
