import json
from pathlib import Path
import unittest


class NotebookLossPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path("deep3.ipynb").read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    def test_active_loss_implementation_is_module_backed(self):
        self.assertIn(
            "from src.losses.focal import FocalLoss, build_class_balanced_alpha",
            self.source,
        )
        self.assertNotIn("from src.losses.mixup import mixup_criterion", self.source)
        trainer_source = Path("src/trainers/loops.py").read_text(encoding="utf-8")
        self.assertIn("from src.losses.mixup import mixup_criterion", trainer_source)
        self.assertNotIn("class FocalLoss", self.source)

    def test_loss_orchestration_remains_at_the_same_boundary(self):
        self.assertIn(
            "alpha = build_class_balanced_alpha(class_counts, beta, num_classes)",
            self.source,
        )
        self.assertIn(
            "criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)",
            self.source,
        )
        self.assertIn(
            "criterion = FocalLoss(alpha=alpha.to(device), gamma=config[\"loss\"][\"focal_gamma\"]).to(device)",
            self.source,
        )
        trainer_source = Path("src/trainers/loops.py").read_text(encoding="utf-8")
        self.assertIn(
            "loss = mixup_criterion(criterion, out, y_a, y_b, lam)",
            trainer_source,
        )
        self.assertIn("loss = criterion(out, y)", trainer_source)


if __name__ == "__main__":
    unittest.main()
