import json
from pathlib import Path
import unittest


class NotebookModelPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path("deep3.ipynb").read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    def test_active_model_implementation_is_module_backed(self):
        self.assertIn(
            "from src.models.factory import build_cmt_classifier",
            self.source,
        )
        for class_name in (
            "DropPath",
            "DepthwiseConv2d",
            "ConvBNGELU",
            "ConvStage",
            "LPU",
            "MLP",
            "TransformerBlock",
            "CMTClassifier",
        ):
            self.assertNotIn(f"class {class_name}", self.source)

    def test_model_construction_and_ema_boundary_are_preserved(self):
        self.assertIn(
            "m = build_cmt_classifier(num_classes).to(device)",
            self.source,
        )
        self.assertIn(
            "model = build_cmt_classifier(num_classes).to(device)",
            self.source,
        )
        self.assertIn("class ModelEma(nn.Module):", self.source)
        self.assertIn("ema = ModelEma(model, decay=EMA_DECAY, device=device)", self.source)


if __name__ == "__main__":
    unittest.main()
