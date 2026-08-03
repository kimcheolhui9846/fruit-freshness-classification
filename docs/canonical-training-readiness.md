# Canonical Training Readiness

## Scope and Decision

Phase 8.1 freezes and audits the existing `deep3` experiment before any canonical run. It does not change source code, configuration, dependencies, model behavior, training behavior, data policy, repository governance, or the published `v0.1.0` prerelease.

**Final readiness classification:** `BLOCKED`

The frozen configuration requests training with batch size 192. A bounded production-path training probe completed at batches 2, 4, 8, and 16 only; its conservative batch-192 reserved-memory projection is 107.2% of the available 8 GiB VRAM. Therefore the current canonical configuration is not safe to start on the audited hardware.

No canonical three-fold training was run in Phase 8.1. No canonical checkpoint, weight, result, benchmark, or publication artifact was created. A configuration change is outside this Phase and requires a new explicit approval.

## Frozen Implementation Identity

The readiness evidence refers to the `main` commit below, before the Phase 8.1 documentation additions.

| Item | Git blob | SHA-256 |
|---|---|---|
| Frozen commit | `046760e19e77c7aa0c6cbc065358acfd46aac346` | Git commit identity |
| `configs/deep3.toml` | `7d88da60f540728aae9259273aae32b4ce0b3bc1` | `62c7ae4ee5c33974fa48342b6af1b7b54c2e4938159429cbd1a86524fc7c13f1` |
| `requirements.txt` | `a0f8ab7af3593786d635f84338a8f53490936147` | `86776c4ccd296dfb828121bd968ecf8cec8fd763b4a2cd600c68a280c6a90919` |
| `requirements-dev.txt` | `89e729557cdee7a20ba8637ce2dd22ba4e2db7ab` | `73e6c3f9d71614711e6e6ac942bba660463a8a80831a609b36a59a41d6e38e4d` |
| `scripts/train.py` | `9a3ef99a5b0595b309a63909f10320ea80e22d16` | `9778a941d18240c3813ee24fcd77e61b0eeef33cd5e45c21c9ba9d0286df06a4` |
| `scripts/evaluate.py` | `8fbffd3aae3a1828783b0691ac9287778f4509a4` | `379b976f196a05f584c39fdef79489f2f5c321d1207c7475f798ea2e6794b6` |

The committed configuration is frozen as follows:

- 120 epochs, with the final 20 epochs using the existing fine-tuning path.
- 3 folds with shuffled stratification and random state 42.
- Training batch size 192; Mixup alpha 0.8 with probability 0.5.
- CNN learning rate 5e-5, transformer learning rate 1e-4, weight decay 1e-4.
- Cross-entropy loss with label smoothing 0.01 is selected by the existing configuration; focal-loss parameters remain part of the frozen fallback path.
- EMA decay 0.999 and cuDNN benchmark enabled.
- The final filename remains `last_model_weights.pt`.

## Environment Evidence

The evidence was collected in a new isolated virtual environment, separate from the repository and removed after the audit. Package installation used the committed exact pins; `pip check` reported no broken requirements.

| Component | Observed value |
|---|---|
| Python | 3.12.10 |
| Operating system | Windows 11 build 26200 |
| PyTorch / CUDA build | torch 2.6.0+cu124 / CUDA 12.4 |
| torchvision | 0.21.0+cu124 |
| datasets / huggingface-hub | datasets 5.0.1 / huggingface-hub 1.26.0 |
| scikit-learn | 1.9.0 |
| matplotlib / Pillow / tqdm | 3.11.1 / 12.3.0 / 4.70.0 |
| CUDA / cuDNN | available / 90100 |
| GPU | NVIDIA GeForce RTX 3070 Ti, 8 GiB |
| GPU driver | 591.86 |

At preflight the device had 6.92 GiB free and a 45 C temperature. A post-probe check observed 6.91 GiB free and 47 C. The free space on the relevant local volume was 556.15 GiB. These values are point-in-time evidence, not an environment guarantee.

## Dataset and Split Contract

The production loader successfully used the pinned Hugging Face dataset without placing dataset files in the repository.

| Property | Verified value |
|---|---|
| Repository | Densu341/Fresh-rotten-fruit |
| Revision | `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Archive | `freshness_fruit.zip` |
| Archive SHA-256 | `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e` |
| Source rows | 30,357 |
| Rows after existing label removal/remapping | 26,858 |
| Train / holdout rows | 21,486 / 5,372 |
| Class count | 14 classes |
| Holdout split | 20%, random state 42 |
| Image sample modes | RGB |

The first pinned-load preparation took 485.916 seconds because the archive was absent from cache; a second load reused the prepared cache with the same counts in 14.111 seconds. Setup time is not part of the training-duration estimate.

| Fold | Train rows | Validation rows | Train batches at 192 | Validation batches at 192 | Index overlap |
|---:|---:|---:|---:|---:|---|
| 1 | 14,324 | 7,162 | 75 | 38 | none |
| 2 | 14,324 | 7,162 | 75 | 38 | none |
| 3 | 14,324 | 7,162 | 75 | 38 | none |

All folds were stratified with the committed helper. They cover the 21,486 training examples and have no train/validation index overlap.

## Output, Storage, and Collision Review

Before and after the bounded audit, `weights/` and `results/` existed but contained no top-level files and had zero recorded file bytes. The audit created no canonical output in either directory.

The frozen training entry point creates its output directory if necessary and uses fixed names:

- `label_names.json`
- `best_model_fold1.pt`, `best_model_fold2.pt`, and `best_model_fold3.pt`
- `last_model_weights.pt`

It does not check that an existing output directory is empty. A future canonical run must therefore use an owner-approved, fresh, untracked run directory, for example `weights/deep3-canonical-YYYYMMDD-046760e`; it must not reuse a prior run directory.

The model has 13,534,894 parameters. Its in-memory state dictionary contains 222 tensors and 54,174,280 bytes (about 51.7 MiB). Four nominal raw model-state files would be about 206.7 MiB before serialization and filesystem overhead. This is a planning estimate only; no checkpoint was written for it.

## Bounded Production-Path Evidence

The bounded probe used actual prepared images, the committed training transform, CMT classifier factory, selected cross-entropy loss, optimizer, scheduler, AMP GradScaler, EMA update, `train_one_epoch`, `validate_one_epoch`, horizontal-flip TTA, and the three-model ensemble TTA path. It deliberately did not run a training step at batch 192.

| Training batch | Result | Peak reserved VRAM | Duration for one batch | Temperature after |
|---:|---|---:|---:|---:|
| 2 | completed | 250 MiB | 0.8201 s | 49 C |
| 4 | completed | 318 MiB | 0.1253 s | 50 C |
| 8 | completed | 430 MiB | 0.1512 s | 50 C |
| 16 | completed | 732 MiB | 0.2015 s | 51 C |

Every listed training probe completed forward pass, backward pass, optimizer step, scheduler construction, AMP scaler update, and EMA update. Based on batch 16, a linear reserved-memory projection for the configured training batch size 192 is 9,210,691,584 bytes, or 107.2% of the available VRAM. Its classification is `LIKELY_UNSAFE`; this is conservative evidence, not an actual batch-192 training attempt.

| Inference path | Batch | Result | Peak reserved VRAM | Duration |
|---|---:|---|---:|---:|
| Validation with horizontal-flip TTA | 16 | completed | 268 MiB | 0.0798 s |
| Validation with horizontal-flip TTA | 192 | completed | 1,792 MiB | 0.7344 s |
| Three-model ensemble with horizontal-flip TTA | 16 | completed | 362 MiB | 0.1573 s |

The batch-192 validation probe was allowed only after the batch-16 inference projection was below the conservative threshold. It is evidence for evaluation capacity only and does not make training batch 192 safe.

## Time, Reproducibility, and Interruption Posture

A counterfactual capacity-mitigation estimate uses the completed batch-16 training throughput and batch-192 validation throughput: approximately 18.0 hours for training batches plus 2.8 hours for validation batches. Including data, metrics, logging, and host variability, a broad 21–31 hour reference range is reasonable. It is not an authorization to change batch size, not a promise that the frozen batch-192 configuration can run, and not a benchmark result.

Reproducibility classification: `REFERENCE_RUN_WITH_RECORDED_ENVIRONMENT`. The identity, data revision, dependencies, and bounded execution conditions are recorded, but the project does not set all global random-number generators or deterministic algorithms. Numeric bit-for-bit reproducibility is not claimed.

Interruption posture: `ACCEPTABLE_WITH_OWNER_RISK` only after explicit owner acceptance. No resume implementation exists in the frozen training entry point; optimizer, scheduler, AMP scaler, EMA, and RNG states are not checkpointed. An interrupted run must restart from the beginning. This is a significant operational cost for a three-fold, 120-epoch run.

## Owner Approval Gate

The following decisions remain unresolved. They are deliberately not inferred from this audit.

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

A future approval must either keep the frozen batch-192 configuration and provide hardware that can run it safely, or explicitly authorize a new configuration Phase. This Phase does not permit either action.

## Governance Boundaries

The `v0.1.0` prerelease, its annotated tag, the protected `main` ruleset, the protected `v0.1.0` tag ruleset, and the local-only backup branch policy remain unchanged. The Phase does not publish data, weights, checkpoints, or other binary artifacts.
