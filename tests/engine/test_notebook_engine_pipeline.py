import json
import subprocess
import unittest
from pathlib import Path


LEGACY_NOTEBOOK_COMMIT = "c2eebb1"
NOTEBOOK_PATH = Path("deep3.ipynb")


def load_cells(notebook_text):
    return ["".join(cell["source"]) for cell in json.loads(notebook_text)["cells"]]


class NotebookEnginePipelineTest(unittest.TestCase):
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

    def test_notebook_is_valid_and_engine_imports_are_explicit(self):
        self.assertEqual(len(self.current_cells), len(self.legacy_cells))
        for source in self.current_cells:
            compile(source, "deep3.ipynb", "exec")
        imports = self.current_cells[0]
        self.assertIn("from src.engine.checkpoint import load_model_state, save_model_state", imports)
        self.assertIn("from src.engine.ema import ModelEma", imports)
        self.assertIn("from src.engine.optimization import build_optimizer, build_scheduler", imports)

    def test_active_engine_definitions_and_construction_are_module_backed(self):
        self.assertIn("class ModelEma", self.legacy_cells[2])
        self.assertNotIn("class ModelEma", self.current_cells[2])
        self.assertIn("m.load_state_dict(torch.load(path, map_location=device))", self.legacy_cells[3])
        self.assertIn("load_model_state(m, path, map_location=device)", self.current_cells[3])
        self.assertNotIn("m.load_state_dict(torch.load(path, map_location=device))", self.current_cells[3])
        self.assertIn("torch.optim.AdamW", self.legacy_cells[4])
        self.assertNotIn("torch.optim.AdamW", self.current_cells[4])
        self.assertIn("CosineAnnealingLR", self.legacy_cells[4])
        self.assertNotIn("CosineAnnealingLR", self.current_cells[4])
        self.assertIn("build_optimizer(", self.current_cells[4])
        self.assertIn("build_scheduler(optimizer, t_max=EPOCHS)", self.current_cells[4])
        self.assertIn("save_model_state(ema.module, save_path)", self.current_cells[4])
        self.assertIn('save_model_state(model, "last_model_weights.pt")', self.current_cells[4])

    def test_loop_and_decision_boundaries_remain_in_the_notebook(self):
        anchors = [
            "for epoch in range(1, EPOCHS+1):",
            "if va_acc > best_acc_fold + 1e-6:",
            "scheduler.step()",
            "scaler = GradScaler()",
        ]
        for anchor in anchors:
            self.assertEqual(
                self.legacy_cells[4].count(anchor),
                self.current_cells[4].count(anchor),
                anchor,
            )
        self.assertIn("from src.trainers.loops import train_one_epoch, validate_one_epoch", self.current_cells[0])
        self.assertNotIn("for x, y in pbar:", self.current_cells[4])
        self.assertNotIn("for x, y in tqdm(val_loader", self.current_cells[4])
        self.assertNotIn("ema.update(model)", self.current_cells[4])
        self.assertIn("train_one_epoch(", self.current_cells[4])
        self.assertIn("validate_one_epoch(", self.current_cells[4])
        self.assertIn("def load_fold_models", self.current_cells[3])
        self.assertIn("def ensemble_logits", self.current_cells[3])
        self.assertIn("def ensemble_logits_tta_hflip", self.current_cells[3])


if __name__ == "__main__":
    unittest.main()
