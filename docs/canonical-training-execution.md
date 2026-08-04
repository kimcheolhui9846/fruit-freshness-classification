# Canonical Training Execution

## Scope

Full derived canonical training completed with `configs/deep3_canonical.toml` at batch size: 64. The run completed three folds with 120 epochs per fold; fine-tuning began at epoch 101. Holdout evaluation: No. Benchmark claim: No.

## Frozen identity

| Field | Value |
|---|---|
| Training commit | `0c669d58852082785c79699231e09b5ae26757cc` |
| Run ID | `deep3-canonical-reference-01` |
| Configuration | `configs/deep3_canonical.toml` |
| Config SHA-256 | `8d40ed34ddcb0eeaea4ca9e03754c579c983e71d1e3b4ae121c512d1fc073c42` |
| Dataset | `Densu341/Fresh-rotten-fruit` at revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Dataset archive SHA-256 | `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e` |
| Device | NVIDIA GeForce RTX 3070 Ti, 8 GiB |
| Environment | Python 3.12.10; torch 2.6.0+cu124; torchvision 0.21.0+cu124; CUDA 12.4 |

## Execution

- Start mode: FRESH
- Fresh invocation count: 1
- Validated resume count: 0
- Interruption reason: none
- Start: `2026-08-03T09:08:07.188011+00:00`
- Completion: `2026-08-03T20:15:26.877171+00:00`
- Elapsed wall-clock duration: approximately 11 hours 7 minutes 20 seconds
- Fold completion: 1, 2, and 3 completed
- Epoch completion: 120 epochs per fold
- Fine-tuning transition: verified at epoch 101
- Final process exit code: 0

## Completion evidence

The trusted local training state has status `COMPLETED`. Its run ID, training commit, config path and hash, dataset identity, batch size, fold count, and epoch count match the run manifest. Completed history lengths are 120, 120, and 120. The output allowlist contains only the expected artifacts; the log has no unresolved traceback, CUDA OOM, runtime error, or non-finite invalidation; and no stale atomic state temporary file remains.

Strict CPU loading of all three fold-best checkpoints and the final raw checkpoint passed. Each checkpoint had strict keys, finite tensors, and a fixed synthetic output shape of `(1, 14)`.

## Limitations and boundaries

configs/deep3.toml remains BLOCKED on the audited RTX 3070 Ti 8 GiB. The derived configuration is a different trajectory from batch 192; exact from-scratch numerical reproduction is not claimed. Numeric training and validation metrics remain in trusted local artifacts and are not published here.

Holdout evaluation: No. Benchmark claim: No. Phase 8.4: PENDING. Local weights, checkpoints, training state, and log are retained only for the next approved evaluation phase; publication occurred: No.

CI checkpoint requirement: No.
CI CUDA requirement: No.
CI production dataset access: No.
Release creation: No.