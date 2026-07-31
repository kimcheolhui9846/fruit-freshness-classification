import json
import subprocess
import unittest
from pathlib import Path


LEGACY_NOTEBOOK_COMMIT = "fec42a2"
NOTEBOOK_PATH = Path("deep3.ipynb")


def load_cells(notebook_text):
    return ["".join(cell["source"]) for cell in json.loads(notebook_text)["cells"]]


class NotebookEvaluationPipelineTest(unittest.TestCase):
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

    def test_notebook_is_valid_and_evaluation_import_is_explicit(self):
        self.assertEqual(len(self.current_cells), len(self.legacy_cells))
        for source in self.current_cells:
            compile(source, "deep3.ipynb", "exec")
        self.assertIn(
            "from src.evaluation.metrics import compute_validation_metrics",
            self.current_cells[0],
        )

    def test_active_notebook_metrics_are_module_backed(self):
        legacy = self.legacy_cells[4]
        current = self.current_cells[4]
        self.assertIn('va_f1   = f1_score(all_labels, all_preds, average="macro")', legacy)
        self.assertIn("va_bal  = balanced_accuracy_score(all_labels, all_preds)", legacy)
        self.assertIn("va_top2 = top_k_accuracy_score(", legacy)
        self.assertNotIn("f1_score(all_labels, all_preds", current)
        self.assertNotIn("balanced_accuracy_score(all_labels, all_preds)", current)
        self.assertNotIn("top_k_accuracy_score(all_labels, all_logits", current)
        self.assertIn("compute_validation_metrics(", current)
        self.assertIn("all_labels, all_preds, all_logits", current)

    def test_execution_and_orchestration_boundaries_are_preserved(self):
        legacy = self.legacy_cells[4]
        current = self.current_cells[4]
        anchors = [
            "validate_one_epoch(",
            "all_logits = np.concatenate(all_logits, axis=0)",
            "history[\"val_acc\"].append(va_acc)",
            "val_f1_list.append(va_f1)",
            'print(f"Val (EMA)',
            "if va_acc > best_acc_fold + 1e-6:",
            "scheduler.step()",
            "for x, y in tqdm(test_loader, ncols=100):",
            "x = x.to(device)",
            "logits = ensemble_logits_tta_hflip(models, x)",
            "pred = logits.argmax(1)",
        ]
        for anchor in anchors:
            self.assertEqual(legacy.count(anchor), current.count(anchor), anchor)
        self.assertLess(
            current.index("compute_validation_metrics("),
            current.index("val_acc_list.append(va_acc)"),
        )


if __name__ == "__main__":
    unittest.main()
