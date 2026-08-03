# Canonical Training Unblock Record — Phase 8.2

## Decision Record

Phase 8.2 implements the owner-approved resource-safe configuration and optional epoch-boundary resume support. The original `configs/deep3.toml` is unchanged and remains blocked on NVIDIA GeForce RTX 3070 Ti, 8 GiB. The approved derived candidate is `configs/deep3_canonical.toml` with batch size 64, output `weights/deep3-canonical-reference-01`, and external log `results/deep3-canonical-reference-01.log`.

| Approved boundary | Decision |
|---|---|
| Config strategy | `CREATE_DERIVED_CANONICAL_CONFIG` |
| Batch | 64 |
| Learning-rate policy | `KEEP_EXISTING_UNSCALED` |
| Resume | `IMPLEMENT_EPOCH_BOUNDARY_RESUME` |
| State / manifest | `training_state.pt` / `run_manifest.json` |
| Collision policy | `FAIL_IF_NOT_EMPTY` |
| Seed / cuDNN change | NO / NO |
| Dataset, weights, checkpoints, other binaries | NO publication |
| Release creation | NO |

## Derived Configuration

The derived TOML is a complete copy of the original and differs at exactly `training.batch_size`: 192 to 64. No learning-rate scaling occurred. It keeps 120 epochs, 20 fine-tuning epochs, three shuffled stratified folds with random state 42, Mixup, loss settings, optimizer parameters, EMA, and the final checkpoint filename. Batch 64 creates a different optimization trajectory, so Phase 8.2 does not equate it with the original batch-192 training result.

## Bounded Batch-64 Validation

An external exact-pinned CUDA environment used the production pinned dataset loader, training transform, CMT model, criterion, optimizer, scheduler, AMP GradScaler, EMA, and `train_one_epoch`. It ran two limited batch-64 optimizer steps, not a full epoch or fold.

| Step | Peak allocated | Peak reserved | Reserved of 8 GiB | Temperature | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 2,032 MiB | 2,716 MiB | 33.2% | 47 C | finite gradients; optimizer, EMA, and scheduler completed |
| 2 | 2,108 MiB | 2,720 MiB | 33.2% | 49 C | finite gradients; optimizer, EMA, and scheduler completed |

The 70% VRAM threshold and conservative 80 C temperature guard passed. No CUDA OOM occurred. The observed loss and accuracy are not model-quality or benchmark results.

## Resume Design and Validation

The versioned epoch-boundary state includes `model_state_dict`, `ema_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `grad_scaler_state_dict`, `python_rng_state`, `numpy_rng_state`, `torch_cpu_rng_state`, `torch_cuda_rng_states`, histories, accuracies, best accuracy, immutable run metadata, and fold-index hashes.

State is saved atomically only after training, validation, metrics, best-checkpoint decision, history update, and `scheduler.step()`. State saving occurs after scheduler.step(). `RUNNING`, `FOLD_COMPLETE`, and `COMPLETED` record epoch, fold, and terminal boundaries; normal resume rejects `COMPLETED`. Resume validates metadata before applying state, rebuilds runtime objects, restores model/EMA/optimizer/scheduler/scaler and histories, and restores RNG last. Trusted local loading is explicit; operators must not load a downloaded or untrusted state file.

The bounded CUDA resume probe saved state, destroyed training objects, reconstructed them, loaded state, restored RNG, and completed a further batch-64 optimizer step. It verified model and EMA loading, CUDA optimizer tensors, scheduler advance from 1 to 2, continued GradScaler state, continued history, valid state metadata, and no repeated controlled fold or epoch. This proves same-run continuation interoperability only; it does not prove bit-for-bit from-scratch reproducibility.

## Output and Logging Safety

Fresh canonical mode requires an absent or empty output directory and rejects a non-empty output directory before dataset loading. The manifest and expected canonical artifact allowlist protect resume against unexpected files. Never run two processes against the same directory, delete partial artifacts during incident review, overwrite identity fields, or publish artifacts automatically. The approved log is external to the checkpoint directory.

## Tests and Remaining Limitation

CPU-only unit tests cover schema, atomic replacement, simulated save failure, trusted-local loading, metadata and index mismatch, completed-state rejection, and CPU/CUDA RNG restoration after CUDA map-location loading. Synthetic integration tests cover uninterrupted versus interrupted/resumed orchestration, fold boundary, fine-tuning boundary, collision protection, and legacy stateless behavior.

No full canonical three-fold training was run in Phase 8.2. No canonical weights, checkpoints, result files, benchmark result, release, or publication artifact was created. No benchmark result is claimed in Phase 8.2.

DERIVED_CANONICAL_CONFIG_READINESS:
READY_FOR_OWNER_APPROVAL

## Phase 8.3 Owner Approval Gate

APPROVED_CANONICAL_TRAINING_ACTION:
<RUN_DERIVED_CONFIG | DEFER | BLOCKED>

APPROVED_CONFIG:
configs/deep3_canonical.toml

APPROVED_BATCH_SIZE:
64

APPROVED_DEVICE:
NVIDIA GeForce RTX 3070 Ti, 8 GiB

APPROVED_RUN_ID:
deep3-canonical-reference-01

APPROVED_OUTPUT_DIRECTORY:
weights/deep3-canonical-reference-01

APPROVED_LOG_FILE:
results/deep3-canonical-reference-01.log

APPROVED_RESUME_POLICY:
USE_EPOCH_BOUNDARY_RESUME

APPROVED_INTERRUPTION_POLICY:
RESUME_FROM_LAST_COMPLETED_EPOCH

APPROVED_CHECKPOINT_RETENTION:
KEEP_ALL_FOLD_BEST_FINAL_AND_RESUME_STATE

APPROVED_WEIGHT_PUBLICATION:
NO

APPROVED_DATASET_PUBLICATION:
NO

APPROVED_RELEASE_CREATION:
NO

Phase 8.2 completion does not infer an approval. Do not begin Phase 8.3 without a new explicit owner decision.
