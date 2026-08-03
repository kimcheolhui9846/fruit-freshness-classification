# Canonical Training Runbook

## Purpose and Preconditions

This runbook describes the exact operator sequence for a future owner-approved canonical run of the frozen `deep3` experiment. It is not an authorization to run it now.

The readiness decision is `BLOCKED` because the configured training batch size 192 is `LIKELY_UNSAFE` on the audited 8 GiB GPU. Do not start the command below until a new explicit owner approval resolves the batch-size or hardware decision, output-directory decision, and interruption-risk acceptance.

No canonical three-fold training was run in Phase 8.1. No canonical checkpoint, weight, result, benchmark, or publication artifact was created.

## Frozen Inputs

Verify all of these before a future run:

| Input | Required identity |
|---|---|
| Base commit | `046760e19e77c7aa0c6cbc065358acfd46aac346` |
| Experiment config | `configs/deep3.toml`, SHA-256 `62c7ae4ee5c33974fa48342b6af1b7b54c2e4938159429cbd1a86524fc7c13f1` |
| Dataset revision | `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Dataset archive SHA-256 | `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e` |
| Environment | Python 3.12.10, torch 2.6.0+cu124, torchvision 0.21.0+cu124, datasets 5.0.1, huggingface-hub 1.26.0 |
| Split | 21,486 training rows, 5,372 holdout rows, 3 stratified folds, random state 42 |

The expected class count is 14. Dataset files remain in the Hugging Face cache and must not be added to Git.

## Required Owner Approval Record

Fill and approve this record in a new explicit Phase before starting a canonical run:

OWNER_CANONICAL_TRAINING_APPROVAL:
PENDING

OWNER_BATCH_SIZE_DECISION:
PENDING

OWNER_OUTPUT_DIRECTORY_APPROVAL:
PENDING

OWNER_INTERRUPTION_RISK_ACCEPTANCE:
PENDING

DATASET_PUBLICATION: NO

WEIGHT_PUBLICATION: NO

CHECKPOINT_PUBLICATION: NO

OTHER_BINARY_ARTIFACT_PUBLICATION: NO

A configuration change is outside this Phase and requires a new explicit approval.

## Preflight Checklist

1. Check out the approved frozen commit or its explicitly approved successor.
2. Create a clean virtual environment and install the exact committed requirements.
3. Confirm `python -m pip check` succeeds.
4. Confirm CUDA is available, record GPU model, driver, free VRAM, temperature, and free disk space.
5. Run the committed dataset loader once and verify repository, revision, archive SHA-256, 30,357 source rows, 26,858 filtered rows, 21,486 training rows, 5,372 holdout rows, and 14 classes.
6. Confirm the requested output directory is fresh, untracked, and empty. Do not reuse a directory containing `best_model_fold*.pt` or `last_model_weights.pt`.
7. Record the approved run identifier in the form `deep3-canonical-YYYYMMDD-<shortsha>`.
8. Confirm the owner has accepted the restart-from-zero interruption consequence.

## Training Command

After every preflight condition and owner approval is satisfied, run the frozen entry point from the repository root:

    python scripts/train.py --config configs/deep3.toml --output-dir weights/<run-id>

For the frozen config, this runs 3 folds for 120 epochs. The final 20 epochs use the existing fine-tuning transform path; the implementation's existing Mixup, EMA, loss selection, optimizer, scheduler, validation, and checkpoint behavior remain unchanged.

Expected files within the fresh output directory are:

- `label_names.json`
- `best_model_fold1.pt`
- `best_model_fold2.pt`
- `best_model_fold3.pt`
- `last_model_weights.pt`

The three best-fold files are the required ensemble inputs. The last-model file is the final raw model state from the last fold; it is not a substitute for a complete best-fold ensemble.

## Monitoring and Stop Conditions

Record at least once per fold:

- start/end time and epoch progress;
- GPU free/reserved memory and temperature;
- disk free space in the output volume;
- dataset revision and archive hash;
- validation accuracy, F1, balanced accuracy, top-2, and top-3 values printed by the existing pipeline;
- output filenames and file hashes after each completed fold.

Stop and report, rather than modifying code in place, if the dataset identity changes, an out-of-memory error occurs, disk capacity becomes unsafe, a required checkpoint is absent, or runtime behavior diverges from the frozen configuration.

## Interruption and Resume Policy

No resume implementation exists in the frozen training entry point. The optimizer, scheduler, AMP scaler, EMA, and RNG states are not checkpointed. If the process is interrupted, the frozen workflow must restart from the beginning; do not manually stitch partially completed folds into a canonical run.

This restart-from-zero consequence is classified as `ACCEPTABLE_WITH_OWNER_RISK` only after an explicit owner decision. It is not silently accepted by this runbook.

## Evaluation Command

After all three best-fold checkpoints exist in the same fresh run directory, evaluate the frozen holdout ensemble:

    python scripts/evaluate.py --config configs/deep3.toml --checkpoint-dir weights/<run-id>

The evaluation entry point requires every `best_model_fold1.pt` through `best_model_fold3.pt` file and uses the existing ensemble plus horizontal-flip TTA path. Treat its metrics as a new canonical result only after the run record, input hashes, and checkpoint identity have been reviewed.

## Storage and Publication Boundaries

The observed raw state dictionary is about 51.7 MiB; four nominal model-state files are about 206.7 MiB before serialization overhead. Plan capacity beyond that estimate for logs and incidental filesystem overhead.

Dataset, trained weights, checkpoints, and other binary artifacts remain uncommitted and unpublished unless a later explicit approval says otherwise. This runbook does not create a release, a tag, a dataset mirror, or a model upload.
