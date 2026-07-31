import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset

from src.engine.checkpoint import save_model_state
from src.engine.ema import ModelEma
from src.engine.optimization import build_optimizer, build_scheduler
from src.evaluation.metrics import compute_validation_metrics
from src.inference.ensemble import (
    ensemble_logits,
    ensemble_logits_tta_hflip,
    run_ensemble_holdout,
)
from src.inference.loading import load_fold_models
from src.losses.focal import FocalLoss, build_class_balanced_alpha
from src.trainers.loops import train_one_epoch, validate_one_epoch
from src.transforms.classification import build_validation_transform
from src.utils.paths import build_fold_checkpoint_path


class TinyPipelineModel(nn.Module):
    """Small model exposing the module layout expected by build_optimizer."""

    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=1)
        self.stage1 = nn.Identity()
        self.stage2 = nn.Identity()
        self.stage3 = nn.Identity()
        self.to_embed1 = nn.Conv2d(4, 4, kernel_size=1)
        self.trans1 = nn.Identity()
        self.down_tokens = nn.Identity()
        self.trans2 = nn.Identity()
        self.head_norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, images):
        features = self.stem(images)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.to_embed1(features)
        features = self.trans1(features)
        features = self.down_tokens(features)
        features = self.trans2(features)
        features = self.head_norm(features)
        return self.fc(features.mean(dim=(2, 3)))


def passthrough_tqdm(iterable, *args, **kwargs):
    return iterable


class ModularPipelineIntegrationTest(unittest.TestCase):
    def test_synthetic_pipeline_connects_extracted_layers(self):
        torch.manual_seed(17)
        validation_transform = build_validation_transform()
        transformed = validation_transform(
            Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8))
        )
        self.assertEqual(tuple(transformed.shape), (3, 224, 224))
        self.assertEqual(transformed.dtype, torch.float32)

        images = torch.randn(6, 3, 8, 8)
        labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
        dataloader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
        model = TinyPipelineModel(num_classes=3)
        initial_fc_weight = model.fc.weight.detach().clone()
        alpha = build_class_balanced_alpha([2, 2, 2], beta=0.999, num_classes=3)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        optimizer = build_optimizer(model, lr_cnn=1e-2, lr_trans=2e-2, weight_decay=0.0)
        scheduler = build_scheduler(optimizer, t_max=1)
        ema = ModelEma(model, decay=0.9, device="cpu")
        scaler = GradScaler()

        with patch("src.trainers.loops.tqdm", side_effect=passthrough_tqdm):
            train_acc, train_loss = train_one_epoch(
                model,
                dataloader,
                criterion,
                optimizer,
                "cpu",
                scaler,
                ema,
                is_finetuning=False,
                mixup_probability=0.0,
                mixup_alpha=0.8,
                progress_description="synthetic train",
            )
            val_acc, val_loss, predictions, targets, logits = validate_one_epoch(
                ema.module,
                dataloader,
                criterion,
                "cpu",
                progress_description="synthetic validation",
            )
        scheduler.step()

        self.assertTrue(np.isfinite(train_loss))
        self.assertTrue(np.isfinite(val_loss))
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertFalse(torch.equal(initial_fc_weight, model.fc.weight.detach()))
        self.assertEqual(len(predictions), len(labels))
        self.assertEqual(len(targets), len(labels))
        metrics = compute_validation_metrics(targets, predictions, np.concatenate(logits, axis=0))
        self.assertEqual(len(metrics), 4)
        self.assertTrue(all(value is None or np.isfinite(value) for value in metrics))

        with tempfile.TemporaryDirectory() as directory:
            fold_path = build_fold_checkpoint_path(directory, 1)
            save_model_state(ema.module, fold_path)
            with patch(
                "src.inference.loading.build_cmt_classifier",
                side_effect=lambda classes: TinyPipelineModel(classes),
            ):
                models = load_fold_models(1, 3, "cpu", directory)

            raw_logits = ensemble_logits(models, images)
            tta_logits = ensemble_logits_tta_hflip(models, images)
            self.assertEqual(tuple(raw_logits.shape), (len(images), 3))
            self.assertEqual(tuple(tta_logits.shape), (len(images), 3))
            self.assertFalse(raw_logits.requires_grad)
            self.assertFalse(tta_logits.requires_grad)

            holdout_labels = tta_logits.argmax(1).cpu()
            holdout_loader = DataLoader(
                TensorDataset(images, holdout_labels), batch_size=2, shuffle=False
            )
            with patch("src.inference.ensemble.tqdm", side_effect=passthrough_tqdm):
                correct, total = run_ensemble_holdout(models, holdout_loader, "cpu")
            self.assertEqual((correct, total), (len(images), len(images)))
            self.assertTrue(Path(fold_path).is_file())


if __name__ == "__main__":
    unittest.main()