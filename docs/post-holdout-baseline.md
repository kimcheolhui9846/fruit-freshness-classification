# Post-Holdout Development Baseline

## Status

```text
OWNER_PHASE_9_3_APPROVAL:
APPROVED
EXPERIMENT_ID:
deep3-postholdout-research-01-baseline
PARENT_EXPERIMENT_ID:
deep3-postholdout-research-01
ROLE:
POST_HOLDOUT_DEVELOPMENT_BASELINE
BASELINE_EXECUTION_STATUS:
NOT_YET_RUN
PHASE_9_4:
NOT STARTED
```

This record defines the first controlled retraining baseline for Phase 9. It is not a new canonical reference, final model, locked-test result, external benchmark, or improvement claim.

## Frozen data boundary

```text
DATA_PROTOCOL:
DEV_PLUS_LOCKED_TEST
DEVELOPMENT_COUNT:
17188
LOCKED_TEST_COUNT:
4298
CANONICAL_HOLDOUT_COUNT:
5372
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
LOCKED_TEST_MODEL_ACCESS:
NO
CANONICAL_HOLDOUT_MODEL_ACCESS:
NO
```

Only the frozen development pool may enter model construction, training, checkpoint selection, validation, or development-CV metrics. The locked test and historical canonical holdout may be read only to validate dataset identity before model construction; neither is returned to the model-visible pipeline.

## Recipe equivalence

```text
CANONICAL_CONFIG:
configs/deep3_canonical.toml
BASELINE_CONFIG:
configs/deep3_postholdout_baseline.toml
BASELINE_RECIPE_EQUIVALENCE:
PASS
ALLOWED_CONFIG_DIFFERENCES:
experiment identity, parent identity, split manifest path, CV manifest path, artifact namespace
```

All runtime, loss, training, fine-tuning, cross-validation, mixup, optimizer, EMA, checkpoint, and reporting values are identical to the canonical config. The baseline retains 120 epochs, batch size 64, three stratified folds, shuffle enabled, and random state 42. No loss, augmentation, sampler, optimization, architecture, ensemble, or hyperparameter experiment is authorized.

## CV identity and artifacts

The deterministic CV identity will be materialized exactly once from the pinned dataset and frozen development manifest before the baseline is launched. Its indices are relative to the 17,188-example development pool and will be tracked as `configs/splits/deep3-postholdout-research-01-baseline-cv.json`.

```text
CV_IDENTITY_STATUS:
NOT_YET_MATERIALIZED
BASELINE_ARTIFACT_PUBLICATION:
LOCAL_ONLY
LOCKED_TEST_EVALUATION:
PROHIBITED
CANONICAL_HOLDOUT_REEVALUATION:
PROHIBITED
```

Any later checkpoints, state, log, local metrics, and prediction artifacts remain local-only. No dataset, weight, checkpoint, raw prediction, raw logit, GitHub Actions artifact, Release asset, Release, or tag is authorized by this baseline record.