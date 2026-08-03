# Canonical Training Runbook

## Status and Scope

This runbook describes the owner-approved derived configuration prepared in Phase 8.2. It is not authorization to execute a full canonical three-fold training run. `configs/deep3.toml` remains historically blocked on the audited RTX 3070 Ti 8 GiB because its batch-192 projection was unsafe. The candidate for a future owner action is `configs/deep3_canonical.toml` with batch size 64.

No full canonical three-fold training was run in Phase 8.2. No benchmark result is claimed in Phase 8.2. Dataset, weights, checkpoints, and other binary artifacts remain unpublished.

## Frozen and Derived Inputs

- Original config: `configs/deep3.toml`, SHA-256 `62c7ae4ee5c33974fa48342b6af1b7b54c2e4938159429cbd1a86524fc7c13f1`; unchanged and blocked on the audited hardware.
- Derived config: `configs/deep3_canonical.toml`; only `training.batch_size` changes, from 192 to 64.
- Learning-rate policy: `KEEP_EXISTING_UNSCALED`.
- Dataset: `Densu341/Fresh-rotten-fruit` at revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`; archive SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`.
- Target device: NVIDIA GeForce RTX 3070 Ti, 8 GiB.
- Approved run ID: `deep3-canonical-reference-01`.

The derived configuration is a different optimization trajectory because its mini-batch update sequence differs from batch 192. It is not a result-comparable substitution until an explicitly approved full run is executed and reviewed.

## Fresh Canonical Run

Only after a separate Phase 8.3 owner approval, run from the repository root:

```powershell
python -m scripts.train `
  --config configs/deep3_canonical.toml `
  --output-dir weights/deep3-canonical-reference-01 `
  --save-training-state `
  --require-empty-output-dir `
  --run-id deep3-canonical-reference-01
```

The fresh command rejects a non-empty output directory, including hidden entries, before production dataset preparation or model construction. An absent output directory may be created. Do not pre-create files in the checkpoint directory.

The approved external log is outside the checkpoint directory, so it does not violate the empty-output rule:

```powershell
<approved command> 2>&1 | Tee-Object `
  -FilePath results/deep3-canonical-reference-01.log
```

Do not use the angle-bracket placeholder literally. Substitute the approved fresh or resume command as one command line before piping. Do not start a second process against the same output directory.

Expected canonical artifacts are `run_manifest.json`, `training_state.pt`, `label_names.json`, `best_model_fold1.pt`, `best_model_fold2.pt`, `best_model_fold3.pt`, and `last_model_weights.pt`. Best-fold files preserve the EMA state-dict checkpoint policy; the final file preserves the raw final-fold state-dict policy. Evaluation does not depend on `training_state.pt`.

## Resume a Single Interrupted Run

Use only a state generated locally by this project for this same run:

```powershell
python -m scripts.train `
  --config configs/deep3_canonical.toml `
  --output-dir weights/deep3-canonical-reference-01 `
  --resume-state weights/deep3-canonical-reference-01/training_state.pt `
  --save-training-state `
  --run-id deep3-canonical-reference-01
```

The fresh command and resume command are different. Never use `--require-empty-output-dir` on resume. The existing output directory, `run_manifest.json`, and `training_state.pt` must exist; config hash, run ID, repository identity, dataset identity, labels, fold indices, folds, epochs, and batch size must validate. A `COMPLETED` state cannot be resumed normally. Remote URLs, downloaded state files, and untrusted state files are prohibited.

State saving is epoch-boundary only. It happens after successful training, validation, metrics, best-model decision, history update, and `scheduler.step()`. The state is atomic and includes model, EMA, optimizer, scheduler, GradScaler, Python/NumPy/Torch CPU/Torch CUDA RNG, histories, and fold-index hashes. RNG is restored after runtime reconstruction. This supports continuation of the same interrupted run; it does not promise deterministic reproduction from a new run.

If an incident occurs, stop the process, retain partial artifacts for review, and do not delete partial artifacts during incident review. Do not manually stitch folds, overwrite the run manifest, or automatically publish any output.

## Preflight and Monitoring

1. Verify the explicitly approved Phase 8.3 action, config, batch, run ID, output directory, log path, and no-publication policy.
2. Install the committed pinned requirements in an isolated environment and confirm `python -m pip check`.
3. Verify CUDA device identity, free VRAM, temperature, driver, and disk space.
4. Verify the pinned dataset revision, archive hash, expected 14 classes, and existing split contract.
5. Confirm the output directory is absent or empty for a fresh run; do not reuse an existing canonical directory.
6. Record per-fold progress, memory, temperature, metrics, checkpoints, and file hashes.

Stop and report if dataset identity or configuration hash differs, an OOM occurs, temperature or disk capacity becomes unsafe, the expected state/manifest is invalid, or runtime behavior diverges. Do not modify model, transforms, loss, optimizer, scheduler, epochs, folds, global seeding, or cuDNN policy during a run.

## Legacy Invocation

When none of the optional stateful controls are provided, the existing command remains supported and does not create a resume state:

```powershell
python -m scripts.train --config configs/deep3.toml --output-dir weights
```

This legacy invocation preserves prior behavior. It is not the approved canonical command for the audited RTX 3070 Ti 8 GiB.

## Phase 8.3 Owner Approval Block

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

Do not infer this approval from Phase 8.2 completion. Do not begin Phase 8.3 without an explicit owner decision.
