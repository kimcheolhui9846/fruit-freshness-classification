# Training entry point

## Canonical command

Run training from the repository root with the module entry point:

```powershell
python -m scripts.train --config configs/deep3.toml --output-dir weights
```

`python -m scripts.train` is the supported invocation because it keeps the repository root on Python's module path without a `sys.path` modification. `--config` defaults to `configs/deep3.toml`; `--output-dir` defaults to the repository-relative `weights/` directory. The CLI intentionally has no learning-rate, epoch, batch-size, seed, or arbitrary configuration override.

## Configuration and output policy

The script loads the same committed TOML contract used by `deep3.ipynb` through `src.utils.config.load_experiment_config`. Relative CLI paths are resolved from the repository location, not from an arbitrary caller working directory. A user may explicitly supply an absolute output path, but the default never writes outside the repository.

The notebook retains a legacy machine-specific output-directory literal. The script does not copy it. It creates the selected output directory only after configuration validation and dataset preparation begin. It writes the existing artifact names to that directory:

- `label_names.json`
- `best_model_fold{fold}.pt` for the best EMA validation accuracy in each fold
- `last_model_weights.pt` for the raw model from the final fold, matching the notebook's final-state save choice

Existing overwrite behavior is preserved. There is no timestamped run directory, automatic experiment renaming, resume support, optimizer-state checkpoint, or scheduler-state checkpoint.

## Notebook-to-script parity map

| Notebook responsibility | Script location | Reused API |
| --- | --- | --- |
| Device selection | `run_training()` start | `src.utils.runtime.resolve_device` |
| Config validation | `run_training()` start | `src.utils.config.load_experiment_config` |
| Dataset preparation and label persistence | dataset setup | `load_fruit_freshness_dataset`, `save_label_names` |
| Fold split and loaders | outer fold loop | `iter_stratified_folds`, `select_fold_datasets`, `build_fold_dataloaders` |
| Train/validation transforms | setup and fine-tuning transition | transform builders under `src.transforms.classification` |
| Model and class-balanced loss inputs | per-fold construction | `build_cmt_classifier`, `build_class_balanced_alpha` |
| CE/Focal criterion choice | per-fold construction | PyTorch CE and `FocalLoss` |
| Optimizer and scheduler | per-fold construction | `build_optimizer`, `build_scheduler` |
| EMA and scaler | per-fold construction | `ModelEma`, `torch.amp.GradScaler` |
| Epoch training and validation | epoch loop | `train_one_epoch`, `validate_one_epoch` |
| Metrics and histories | epoch loop | `compute_validation_metrics` |
| Best fold checkpoint | validation decision | `build_fold_checkpoint_path`, `save_model_state` |
| Fine-tuning transition | final configured epochs | existing transform switch and trainer Mixup flag |
| Final model state | after all folds | `save_model_state` |

The notebook has no explicit random-seed initialization, and the script adds none. The script preserves the runtime/device, dataset/fold, model/optimizer/scheduler/scaler/EMA, and fine-tuning construction sequence structurally. Exact production RNG parity remains unverified because the Hugging Face dataset and full CMT training have not been executed in this environment.

## Evaluation boundary

The training script intentionally ends after training and required saves. Saved-fold loading, holdout DataLoader construction, raw-logit ensemble evaluation, and horizontal-flip TTA are provided separately by `scripts/evaluate.py`; notebook-specific Matplotlib history plotting remains in `deep3.ipynb`. See [evaluation.md](evaluation.md) for the evaluation input and checkpoint contract.
## Requirements and verification status

Install the project requirements described in `docs/environment.md`, including Hugging Face `datasets` for production loading. Dataset access may require network access or an existing Hugging Face cache. The entry point uses the existing CUDA-first device helper; CPU execution follows existing PyTorch behavior but has not been validated as a production run.

Phase 5.3 verification covers import safety, `--help`, parser behavior, and synthetic orchestration/checkpoint policy with patched production boundaries. It does not cover a production dataset download, real CMT training, generated production checkpoints, or a clean-environment installation. Run the full test suite before a real training run.
