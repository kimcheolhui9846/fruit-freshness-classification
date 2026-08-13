# Post-Holdout Baseline Runbook

## Status and Scope

This runbook describes how the owner-approved Phase 9.3 baseline is executed. The owner granted the separate explicit execution decision recorded in the approval block at the end of this document on 2026-08-12. Authorization covers baseline training and development-only OOF evaluation; every other boundary in this document stays closed.

```text
RUNBOOK_STATUS:
APPROVED_FOR_EXECUTION
BASELINE_EXECUTION_STATUS:
COMPLETED
PHASE_9_4:
BASELINE_EXECUTION_AUTHORIZED
PHASE_9_4_TRAINING_AUTHORIZATION:
GRANTED
PHASE_9_4_TRAINING_AUTHORIZATION_DATE:
2026-08-12
```

The baseline is a controlled canonical-recipe reproduction on the frozen Phase 9 development pool. It is not a new canonical reference, a locked-test result, an external benchmark, or an improvement claim. See [post-holdout-baseline.md](post-holdout-baseline.md) for the frozen protocol and [post-holdout-research-plan.md](post-holdout-research-plan.md) for the research boundary.

## Frozen Inputs

All file hashes below are **SHA-256 over LF-normalized bytes**. Raw-byte hashes differ between platforms because tracked text is checked out with native line endings; only the LF-normalized value is stable across Windows and Linux.

| Field | Value |
|---|---|
| Experiment ID | `deep3-postholdout-research-01-baseline` |
| Parent experiment ID | `deep3-postholdout-research-01` |
| Config | `configs/deep3_postholdout_baseline.toml` |
| Config LF-normalized SHA-256 | `7cb01e8fe251fd1648ba3a53601e471d9b3693e5d50090f7e7d9c9c5586b11c7` |
| Split manifest | `configs/splits/deep3-postholdout-research-01.json` |
| Split manifest LF-normalized SHA-256 | `cd7182c18d81cfac877fb2dab8573695b6bdd8116aeb23b19c3e4457e36be169` |
| CV manifest | `configs/splits/deep3-postholdout-research-01-baseline-cv.json` |
| CV manifest LF-normalized SHA-256 | `494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f` |
| Dataset | `Densu341/Fresh-rotten-fruit` at revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Development pool | 17,188 examples |
| Locked test pool | 4,298 examples, `FROZEN_UNOBSERVED_BY_MODEL` |
| Historical canonical holdout | 5,372 examples, historical evidence only |
| Cross-validation | 3-fold stratified, random state 42 |
| Fold sizes | 11458/5730, 11459/5729, 11459/5729 |
| Target device | NVIDIA GeForce RTX 3070 Ti, 8 GiB |

The baseline config is recipe-equivalent to `configs/deep3_canonical.toml`. The only permitted differences are experiment identity, parent identity, split manifest path, CV manifest path, and artifact namespace. `validate_postholdout_baseline_config` enforces this before both training and evaluation; a mismatch aborts the run.

## Fresh Baseline Run

Only after a separate owner approval, run from the repository root:

```powershell
python -m scripts.train `
  --config configs/deep3_postholdout_baseline.toml `
  --output-dir weights/deep3-postholdout-research-01-baseline `
  --save-training-state `
  --require-empty-output-dir `
  --run-id deep3-postholdout-research-01-baseline
```

The fresh command rejects a non-empty output directory, including hidden entries, before dataset preparation or model construction. An absent output directory may be created. Do not pre-create files in the checkpoint directory.

The approved external log lives outside the checkpoint directory, so it does not violate the empty-output rule:

```powershell
<approved command> 2>&1 | Tee-Object `
  -FilePath results/deep3-postholdout-research-01-baseline.log
```

Do not use the angle-bracket placeholder literally. Substitute the approved fresh or resume command as one command line before piping. Do not start a second process against the same output directory.

Expected artifacts are `run_manifest.json`, `training_state.pt`, `label_names.json`, `best_model_fold1.pt`, `best_model_fold2.pt`, `best_model_fold3.pt`, and `last_model_weights.pt`. Best-fold files preserve the EMA state-dict checkpoint policy; the final file preserves the raw final-fold state-dict policy. The run manifest additionally records a `post_holdout` block with the experiment identity, both manifest hashes, development and locked-test counts, and `locked_test_model_access: NO`.

## Resume an Interrupted Run

Use only a state generated locally by this project for this same run:

```powershell
python -m scripts.train `
  --config configs/deep3_postholdout_baseline.toml `
  --output-dir weights/deep3-postholdout-research-01-baseline `
  --resume-state weights/deep3-postholdout-research-01-baseline/training_state.pt `
  --save-training-state `
  --run-id deep3-postholdout-research-01-baseline
```

Never use `--require-empty-output-dir` on resume. Epoch-boundary resume semantics, state contents, and validation rules are unchanged from [canonical-training-runbook.md](canonical-training-runbook.md); the frozen CV fold indices are validated by hash on resume in the same way canonical fold indices were.

## Development-Only OOF Evaluation

After training completes, evaluate only the development cross-validation. The locked test and historical canonical holdout stay out of the model-visible pipeline.

```powershell
python -m scripts.evaluate_postholdout_baseline `
  --config configs/deep3_postholdout_baseline.toml `
  --checkpoint-dir weights/deep3-postholdout-research-01-baseline `
  --output-dir results/deep3-postholdout-research-01-baseline
```

The output directory must be absent or empty; the writer refuses any collision rather than replacing an existing result. Each fold's best checkpoint is evaluated on that fold's held-out validation indices, and the per-fold outputs are assembled into one out-of-fold prediction set covering all 17,188 development examples exactly once.

Artifacts, all local-only:

- `development_oof_metrics.json` — aggregate and per-fold metrics plus an integrity block asserting zero locked-test and zero canonical-holdout forward passes
- `development_oof_per_class_metrics.csv`
- `development_oof_confusion_matrix.csv`
- `development_oof_predictions.npz`

The primary selection metric is development Macro F1, with balanced accuracy and Top-1 secondary. Report aggregate OOF metrics and per-fold metrics separately; the mean of fold metrics is not the same quantity as the aggregate OOF metric and must not be presented as if it were. This is a development measurement, not a final claim; a final claim requires the untouched locked test under a separate authorization.

## Resource Expectation

The canonical run trained 21,486 examples at batch size 64 for three folds of 120 epochs in approximately 11 hours 7 minutes on the audited RTX 3070 Ti 8 GiB. The Phase 9 development pool is 17,188 examples, about 0.80 of that volume, so the baseline is expected to take roughly **8.5 to 9.5 hours** under comparable conditions. This is a planning estimate, not a guarantee.

VRAM behavior is expected to match the canonical run because the architecture, batch size, image size, and AMP policy are unchanged. Batch size 64 was previously verified safe on this device; `configs/deep3.toml` at batch 192 remains blocked.

## Preflight

1. Verify the explicitly approved action, config, run ID, output directory, log path, and no-publication policy in the approval block below.
2. Install the committed pinned requirements in an isolated environment and confirm `python -m pip check`.
3. Record the actual runtime environment again rather than assuming the previously verified versions still hold.
4. Verify CUDA device identity, free VRAM, temperature, driver, and free disk space.
5. Verify the pinned dataset revision and archive hash, the expected 14 classes, and the existing split contract.
6. Recompute the LF-normalized SHA-256 of the config, the split manifest, and the CV manifest, and compare against the frozen values above. Stop on any mismatch.
7. Confirm the development pool resolves to 17,188 examples and the three folds resolve to 11458/5730, 11459/5729, and 11459/5729.
8. Confirm the output directory is absent or empty for a fresh run; do not reuse an existing Phase 9 directory.
9. Confirm no locked-test or canonical-holdout index enters the training or validation loaders.

## Monitoring and Stop Conditions

Record per-fold progress, memory, temperature, metrics, checkpoints, and file hashes.

Stop and report if dataset identity or any manifest hash differs, an OOM occurs, temperature or disk capacity becomes unsafe, the expected state or manifest is invalid, a locked-test or canonical-holdout index appears in a model-visible loader, or runtime behavior diverges. Do not modify model, transforms, loss, optimizer, scheduler, epochs, folds, global seeding, or cuDNN policy during a run.

If an incident occurs, stop the process, retain partial artifacts for review, and do not delete partial artifacts during incident review. Do not manually stitch folds, overwrite the run manifest, or automatically publish any output.

## Prohibited During Baseline Execution

- Evaluating the 4,298-example locked test or the 5,372-example historical canonical holdout with any model
- Hyperparameter, loss, augmentation, sampler, optimization, or architecture experimentation
- Checkpoint selection using anything other than the frozen development CV
- Publishing weights, checkpoints, dataset copies, raw logits, raw predictions, Actions artifacts, Release assets, Releases, or tags
- Deleting, relocating, converting, or repackaging existing canonical artifacts

## Owner Approval Block

APPROVED_BASELINE_EXECUTION_ACTION:
RUN_BASELINE

APPROVED_CONFIG:
configs/deep3_postholdout_baseline.toml

APPROVED_RUN_ID:
deep3-postholdout-research-01-baseline

APPROVED_OUTPUT_DIRECTORY:
weights/deep3-postholdout-research-01-baseline

APPROVED_LOG_FILE:
results/deep3-postholdout-research-01-baseline.log

APPROVED_OOF_OUTPUT_DIRECTORY:
results/deep3-postholdout-research-01-baseline

APPROVED_RESUME_POLICY:
USE_EPOCH_BOUNDARY_RESUME

APPROVED_INTERRUPTION_POLICY:
RESUME_FROM_LAST_COMPLETED_EPOCH

APPROVED_CHECKPOINT_RETENTION:
KEEP_ALL_FOLD_BEST_FINAL_AND_RESUME_STATE

APPROVED_LOCKED_TEST_EVALUATION:
NO

APPROVED_CANONICAL_HOLDOUT_REEVALUATION:
NO

APPROVED_WEIGHT_PUBLICATION:
NO

APPROVED_DATASET_PUBLICATION:
NO

APPROVED_RELEASE_CREATION:
NO

This approval was not inferred from Phase 9.3 completion. The owner granted it explicitly on 2026-08-12 after the preflight below was executed and recorded. It authorizes exactly one baseline training run against `weights/deep3-postholdout-research-01-baseline` and the development-only OOF evaluation that follows it. It does not authorize locked-test evaluation, canonical-holdout re-evaluation, weight or dataset publication, release creation, or any second run; each of those requires a further explicit owner decision.

### Recorded Preflight — 2026-08-12

Executed in an isolated virtual environment created for this run.

| Check | Result |
|---|---|
| Pinned requirements installed in isolated env; `python -m pip check` | 9/9 pins matched, no broken requirements |
| `huggingface-hub` | `1.26.0`, matching the audited pinned-archive download API |
| Runtime | Python 3.12.10, `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `datasets==5.0.1` |
| CUDA device | RTX 3070 Ti, 8 GiB, capability 8.6, CUDA 12.4, driver 591.86, 52 °C |
| Dataset archive | 3,053,594,823 bytes, SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`, revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Config, split, and CV LF-normalized SHA-256 | All three matched the frozen values above |
| Development pool and folds | 17,188; 11458/5730, 11459/5729, 11459/5729 |
| Output directories | Both absent before the run |
| Leakage | development ∩ locked test = 0; per fold train ∩ val = 0 and train/val ∩ locked test = 0; validation union covered all 17,188 development examples exactly once |
| Repository suite | 260 tests passed; `compileall` clean |

### Recorded Execution — 2026-08-12 to 2026-08-13

The run started 2026-08-12 19:10 local under repository commit
`5757d0efb3fe0f4b5f6399e52eb745c4d59cd008`. It was interrupted at 2026-08-13
05:34:53 during fold 3 epoch 30 by an automatic Windows Update restart (System
event 1074, `MoUsoCoreWorker.exe`), not by a training fault. The epoch-boundary
state written 20 seconds earlier held fold 3 through completed epoch 29, and the
approved resume policy recovered the run without repeating folds 1 and 2. The
resume ran 2026-08-13 10:31 to 12:43 and completed fold 3 in 131.5 minutes.

Folds 1 and 2 were unaffected. Every documented artifact was produced, and the
recorded `repository_commit` still resolves in history.

| Field | Value |
|---|---|
| Interruption cause | Windows Update restart, external to training |
| Resume point | fold 3, completed epoch 29, next epoch 30 |
| Final training state | `status = COMPLETED`, `completed_epoch = 120` |
| Fold best EMA validation accuracy | 0.9529 (fold 1, epoch 103); 0.9592 (fold 2, epoch 74); 0.9578 (fold 3, epoch 110) |
| Artifacts | all seven expected files present |

Development-only OOF evaluation ran 2026-08-13 13:24. Aggregate metrics cover all
17,188 development examples exactly once.

| Aggregate OOF metric | Value |
|---|---|
| Macro F1 (primary) | 0.9012 |
| Balanced accuracy | 0.9007 |
| Top-1 | 0.9566 |
| Top-2 / Top-3 | 0.9769 / 0.9885 |

Per-fold Macro F1 is a separate quantity and is reported separately: 0.8907
(fold 1), 0.9098 (fold 2), 0.9022 (fold 3). Their mean is 0.900945, which is not
the aggregate OOF value of 0.901167 and must not be substituted for it.

The integrity block in `development_oof_metrics.json` records zero locked-test
forward passes, zero locked-test predictions, zero locked-test metrics, zero
canonical-holdout forward passes, and zero new canonical-holdout metrics. Both
evaluation boundaries stayed closed.

Macro F1 is held down by one class. Aggregate `freshpotato` F1 is 0.3682 with
recall 0.2738: of 347 examples only 95 are correct, while 164 are predicted
`rottenpotato`, 43 `rottenbanana`, and 30 `freshbanana`. The reciprocal error
appears as low `rottenpotato` precision (0.6809). `rottencucumber` (0.7932) and
`rottentomato` (0.8765) are the next weakest; the remaining ten classes score
0.929 to 0.999. Because the dominant confusion crosses the fresh/rotten
distinction, this is a substantive finding for the research plan rather than a
tuning detail.

These are development measurements only. A final claim still requires the
untouched 4,298-example locked test under a separate owner authorization.
