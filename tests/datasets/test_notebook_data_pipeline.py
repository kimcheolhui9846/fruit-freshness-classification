import json
from pathlib import Path
import unittest


class NotebookDataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(Path("deep3.ipynb").read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    def test_active_dataset_implementation_is_module_backed(self):
        self.assertIn(
            "from src.datasets.fruit_freshness import FruitHFDataset, load_fruit_freshness_dataset",
            self.source,
        )
        self.assertIn(
            "from src.datasets.folds import iter_stratified_folds, select_fold_datasets",
            self.source,
        )
        self.assertIn(
            "from src.datasets.loaders import build_fold_dataloaders, build_holdout_dataloader",
            self.source,
        )
        self.assertNotIn("def prepare_dataset():", self.source)
        self.assertNotIn("class FruitHFDataset(Dataset):", self.source)
        self.assertNotIn("DataLoader(", self.source)
        self.assertNotIn("StratifiedKFold(", self.source)

    def test_transform_assignments_remain_at_the_same_orchestration_boundary(self):
        self.assertIn(
            "train_ds = FruitHFDataset(train_split, transform=train_transform)",
            self.source,
        )
        self.assertIn(
            "val_ds   = FruitHFDataset(val_split,  transform=val_transform)",
            self.source,
        )
        self.assertIn(
            "test_ds = FruitHFDataset(final_dataset[\"test\"], transform=val_transform)",
            self.source,
        )
        self.assertIn("ft_transform = build_finetune_transform()", self.source)

    def test_orchestration_preserves_dataset_and_loader_calls(self):
        self.assertIn("final_dataset = load_fruit_freshness_dataset()", self.source)
        self.assertIn(
            "iter_stratified_folds(final_dataset[\"train\"], n_splits=K, shuffle=config[\"cross_validation\"][\"shuffle\"], random_state=config[\"cross_validation\"][\"random_state\"])",
            self.source,
        )
        self.assertIn(
            "train_split, val_split = select_fold_datasets(final_dataset[\"train\"], train_idx, val_idx)",
            self.source,
        )
        self.assertIn(
            "train_loader, val_loader = build_fold_dataloaders(train_ds, val_ds, BATCH_SIZE)",
            self.source,
        )
        self.assertIn("test_loader = build_holdout_dataloader(test_ds, BATCH_SIZE)", self.source)


if __name__ == "__main__":
    unittest.main()
