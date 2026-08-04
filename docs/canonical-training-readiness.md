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

## Phase 8.2 — Approved Derived Configuration and Resume Evidence

Phase 8.1 remains the historical audit for the original frozen configuration. Phase 8.2 does not alter that configuration. It applies the owner-approved derived configuration and optional operational resume support without starting canonical three-fold training.

### Approved Decisions

OWNER_APPROVAL_STATUS: APPROVED

APPROVED_CANONICAL_CONFIG_STRATEGY: CREATE_DERIVED_CANONICAL_CONFIG

APPROVED_DERIVED_CONFIG_PATH: configs/deep3_canonical.toml

APPROVED_TARGET_BATCH_SIZE: 64

APPROVED_LEARNING_RATE_POLICY: KEEP_EXISTING_UNSCALED

APPROVED_RESUME_POLICY: IMPLEMENT_EPOCH_BOUNDARY_RESUME

APPROVED_OUTPUT_DIRECTORY: weights/deep3-canonical-reference-01

APPROVED_OUTPUT_COLLISION_POLICY: FAIL_IF_NOT_EMPTY

APPROVED_LOG_FILE: results/deep3-canonical-reference-01.log

APPROVED_GLOBAL_SEED_CHANGE: NO

APPROVED_CUDNN_POLICY_CHANGE: NO

DATASET_PUBLICATION: NO

WEIGHT_PUBLICATION: NO

CHECKPOINT_PUBLICATION: NO

OTHER_BINARY_ARTIFACT_PUBLICATION: NO

### Configuration Classification

| Configuration | Classification | Reason |
|---|---|---|
| `configs/deep3.toml` | BLOCKED on RTX 3070 Ti 8 GiB | Phase 8.1’s batch-192 probe projected 107.2% reserved VRAM. |
| `configs/deep3_canonical.toml` | READY_FOR_OWNER_APPROVAL | The approved batch-64 bounded CUDA and resume probes passed under the recorded conditions. |

configs/deep3.toml:
BLOCKED on RTX 3070 Ti 8 GiB

configs/deep3_canonical.toml differs from `configs/deep3.toml` at exactly one parsed key: `training.batch_size`, from 192 to 64. The original SHA-256 remains `62c7ae4ee5c33974fa48342b6af1b7b54c2e4938159429cbd1a86524fc7c13f1`. Epochs (120), fine-tuning epochs (20), folds (3), random state (42), Mixup, optimizer parameters, learning rates, weight decay, EMA, checkpoint name, global-seed posture, and cuDNN policy are unchanged. Learning rates use `KEEP_EXISTING_UNSCALED`; no learning-rate scaling was applied.

The derived batch changes the optimization trajectory because its mini-batch composition and update sequence differ. It is therefore a separately identified canonical candidate, not a performance-equivalent continuation of the original batch-192 configuration.

### Bounded CUDA Evidence

The external, exact-pinned environment used Python 3.12.10, torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, and the pinned dataset route. It ran the actual pinned dataset loader, existing training transform, CMT factory, selected cross-entropy criterion, optimizer, scheduler, AMP GradScaler, EMA, and `train_one_epoch`. It did not run a full epoch, validation epoch, fold, or canonical training job.

At preflight, NVIDIA GeForce RTX 3070 Ti had 7,031 MiB free of 8,192 MiB, was 44 C, and used driver 591.86. Display processes reported by the driver had no numeric compute-memory use; total pre-probe occupancy was below the documented 20% material-use guard.

| Batch | Limited optimizer step | Peak allocated | Peak reserved | Reserved VRAM | Temperature | Duration | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 1 | 2,032 MiB | 2,716 MiB | 33.2% | 47 C | 0.8763 s | forward, backward, finite gradients, optimizer, EMA, and scheduler completed |
| 64 | 2 | 2,108 MiB | 2,720 MiB | 33.2% | 49 C | 0.4458 s | forward, backward, finite gradients, optimizer, EMA, and scheduler completed |

The AMP GradScaler remained at 65,536 after both completed steps; no repeated overflow occurred. The peak-reserved threshold was below 70% of total VRAM, no CUDA OOM occurred, and temperature remained below the conservative 80 C guard. Observed loss and accuracy values are diagnostic only and are not model-quality, benchmark, or performance claims.

### Resume Interoperability Evidence

The external probe saved a trusted local `training_state.pt` after a controlled completed sequence, destroyed the CMT training objects, rebuilt model/EMA/criterion/optimizer/scheduler/GradScaler, loaded and validated the state, restored RNG last, and completed a further batch-64 optimizer step.

The resumed step used 2,254 MiB peak allocated and 2,924 MiB peak reserved (35.7% of total VRAM), completed at 49 C, and retained a GradScaler scale of 65,536. Model, EMA, optimizer, scheduler, scaler, metadata, histories, and RNG restoration all completed. Optimizer state tensors were restored to CUDA; scheduler state advanced from 1 before resume to 2 after continuation. The controlled sequence recorded no repeated fold or epoch.

The production implementation saves an epoch-boundary state only after training, validation, metric calculation, best-checkpoint decision, history update, and `scheduler.step()`. `RUNNING` records the fully completed epoch and next epoch. `FOLD_COMPLETE` records completed fold histories and starts the next fold at epoch 1. After the final raw-model save, `COMPLETED` is written and is rejected for normal resume.

The schema includes `model_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `grad_scaler_state_dict`, `python_rng_state`, `numpy_rng_state`, `torch_cpu_rng_state`, and `torch_cuda_rng_states`, plus immutable run, dataset, label, fold-index, and configuration identity. State is written atomically through a unique same-directory temporary file and replacement. A state may be loaded only as a trusted local file generated by this project; operators must not load a downloaded or untrusted state file.

This is same-run continuation evidence, not a claim of bit-for-bit reproducibility from scratch. The project still does not introduce global initial seeding or a changed cuDNN policy.

### Current Readiness and Boundaries

ORIGINAL_CONFIG_READINESS:
BLOCKED

DERIVED_CANONICAL_CONFIG_READINESS:
READY_FOR_OWNER_APPROVAL

No full canonical three-fold training was run in Phase 8.2. No canonical weights, checkpoints, result files, benchmark result, release, or publication artifact was created. No benchmark result is claimed in Phase 8.2. The approved output and external log paths remain absent from the repository after bounded validation.

Phase 8.3 remains owner-gated. Passing Phase 8.2 does not authorize a full canonical run, publication, or release.

## Phase 8.3 — Canonical Training Completion

Phase 8.3 executed the owner-approved `deep3-canonical-reference-01` run on the frozen commit `0c669d58852082785c79699231e09b5ae26757cc`. It used `configs/deep3_canonical.toml` with batch size 64, completed three folds and 120 epochs per fold, and reached trusted state `COMPLETED`. Training artifacts remain local-only and ignored; no holdout evaluation, numeric metric publication, binary publication, Release, or tag was created.

| Configuration | Current classification |
|---|---|
| `configs/deep3.toml` | BLOCKED on RTX 3070 Ti 8 GiB |
| `configs/deep3_canonical.toml` | CANONICAL TRAINING COMPLETED |

configs/deep3_canonical.toml:
CANONICAL TRAINING COMPLETED

TRAINED_CHECKPOINT_HOLDOUT_EVALUATION:
PENDING