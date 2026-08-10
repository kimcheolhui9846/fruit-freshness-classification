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
MATERIALIZED
CV_MANIFEST:
configs/splits/deep3-postholdout-research-01-baseline-cv.json
CV_MANIFEST_LF_NORMALIZED_SHA256:
494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f
CV_DEVELOPMENT_MANIFEST_LF_NORMALIZED_SHA256:
cd7182c18d81cfac877fb2dab8573695b6bdd8116aeb23b19c3e4457e36be169
CV:
3_FOLD_STRATIFIED
CV_RANDOM_STATE:
42
FOLD_1:
train=11458, validation=5730
FOLD_2:
train=11459, validation=5729
FOLD_3:
train=11459, validation=5729
CV_VALIDATION_COVERAGE:
17188_OF_17188_EXACTLY_ONCE
POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:
0
CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:
0
BASELINE_ARTIFACT_PUBLICATION:
LOCAL_ONLY
LOCKED_TEST_EVALUATION:
PROHIBITED
CANONICAL_HOLDOUT_REEVALUATION:
PROHIBITED
```

The data-only audit reconstructed 26,858 filtered examples, 21,486 historical-training examples, and 5,372 historical-canonical-holdout examples. It verified the approved 17,188/4,298 development/locked partition, exact development class counts, all fold hashes, and exhaustive validation coverage. Model construction and model forward passes were both zero.

Any later checkpoints, state, log, local metrics, and prediction artifacts remain local-only. No dataset, weight, checkpoint, raw prediction, raw logit, GitHub Actions artifact, Release asset, Release, or tag is authorized by this baseline record.