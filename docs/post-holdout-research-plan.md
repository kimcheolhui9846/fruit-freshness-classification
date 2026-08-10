# Post-holdout research plan

## Research status

```text
RESEARCH_STAGE:
POST_HOLDOUT
CANONICAL_REFERENCE:
deep3-canonical-reference-01
NEW_EXPERIMENT_ID:
deep3-postholdout-research-01
POST_HOLDOUT_RESEARCH:
YES
```

The canonical reference is closed. Its 5,372-example holdout has already been evaluated, observed, and interpreted; it is `HISTORICAL_EVALUATION_ONLY` and is prohibited for tuning, selection, early stopping, thresholding, ensemble weighting, or untouched final-test claims.

## Primary research question

Can the fruit-freshness classifier improve class-balanced performance and reduce dominant class-specific confusion patterns without sacrificing overall accuracy, under a newly defined post-holdout experimental protocol? This question does not promise improvement.

## Historical motivation

`HISTORICAL_OBSERVATION_ONLY`: canonical Top-1 was 0.955510, Macro F1 0.903737, balanced accuracy 0.899969, Top-2 0.981199, and Top-3 0.992740. Historical weak classes include `freshpotato`, `rottentomato`, and `rottencucumber`; 194 of 239 historical top-1 errors (about 81.17%) were concentrated. These observations motivate hypotheses only, never a new validation target.

## Metrics and success criteria

```text
PRIMARY_SELECTION_METRIC:
development Macro F1
FINAL_CLAIM_REQUIRES_NEW_UNTOUCHED_EVALUATION:
YES
TOP1_GUARDRAIL:
A candidate should not materially degrade development Top-1 accuracy while improving Macro F1.
```

A candidate may advance only if it improves predefined development Macro F1 over the frozen canonical architecture baseline under the same Phase 9 development protocol. Secondary metrics are balanced accuracy and Top-1 accuracy. Diagnostics are per-class precision, recall, F1, confusion matrix, Top-2, and Top-3.

## Data and evaluation boundary

The filtered total is 26,858; historical canonical training is 21,486 and the historical canonical holdout is 5,372. The 5,372 examples cannot become Phase 9 development data.

```text
SOURCE_POOL:
historical canonical training pool only
SIZE:
21486
CANONICAL_HOLDOUT_CHECKPOINT_SELECTION:
PROHIBITED
```

Before model experiments, create a newly locked test subset from the source pool and do not inspect it during development. Future experiments must use a new experiment identity. Split execution is not authorized in Phase 9.1.

| Option | Cleanliness / leakage | Cost / practicality | Claim support |
|---|---|---|---|
| A. Nested CV on 21,486 | Strong selection hygiene; repeated-fold complexity | High compute | Development comparison, limited independent-test claim |
| B. Stratified development + newly locked test, with CV inside development | Clear locked boundary when frozen before experiments | Moderate and practical | Best portfolio comparison; still internal-data generalization limits |
| C. New external evaluation dataset | Strongest generalization evidence | Acquisition and compatibility cost | Best generalization claims if provenance is approved |

**Recommendation:** Option B (`DEV_PLUS_LOCKED_TEST`) is the Phase 9.2 candidate: stratify the 21,486-example source pool into `POST_HOLDOUT_DEVELOPMENT_POOL` and `POST_HOLDOUT_LOCKED_TEST_POOL`, then perform CV only inside development. The ratio, seed, and nested-CV decision remain owner-gated.

## Hypotheses and priority

| Priority | Hypothesis family | Controlled future investigation |
|---:|---|---|
| 1 | Evaluation protocol freeze | Freeze split and baseline before selection |
| 2 | H1 loss / imbalance | Class-balanced focal parameters, class weighting, label smoothing |
| 3 | H2 augmentation | Strength, class-sensitive augmentation, RandAugment/TrivialAugment, Mixup/CutMix |
| 4 | H3 sampling | Weighted or class-aware sampling |
| 5 | H4 optimization | Fine-tuning start, layer-group learning rates, scheduler behavior |
| 6 | H5 architecture | CMT changes or alternative backbones, after controlled training/data hypotheses |
| 7 | H6 error-focused analysis | Historical weak classes only; image-level review needs separate approval |

Change one major factor at a time, then evaluate controlled combinations. Stop when a primary family is exhausted, improvements repeatedly fail on development Macro F1, the Top-1 guardrail repeatedly fails, cost is disproportionate, evidence points to data quality, or external evaluation becomes necessary.

## Baseline, resources, and reproducibility

```text
PHASE_9_BASELINE_MODEL:
canonical CMT implementation
PHASE_9_BASELINE_CONFIG_SOURCE:
configs/deep3_canonical.toml
CANONICAL_CONFIG_MUTATION:
NO
DEFAULT_RESOURCE_REFERENCE:
canonical batch 64 was previously verified safe
```

New Phase 9 configs must be new files. Every future run must record repository/config SHA, experiment and parent IDs, dataset revision, split identity/hash, seeds, runtime/packages/GPU, duration, checkpoint hashes, result hashes, resource use, and advancement decision.

## Comparison and artifact policy

```text
ONE_PRIMARY_CHANGE_PER_INITIAL_EXPERIMENT:
YES
NO_RESULT_CHERRY_PICKING:
YES
```

Every future experiment records parent, hypothesis, exact config, commit SHA, data/split identity, seed, metrics, resource use, success/failure, and decision. All planned experiments stay registered, including unsuccessful ones. Checkpoints are selected only through the Phase 9 development protocol. No Phase 9.1 artifact, checkpoint, split, training, evaluation, download, or publication is authorized.

## Phase 9.2 frozen protocol

```text
PHASE_9_2:
PROTOCOL_FROZEN
APPROVED_PHASE_9_DATA_PROTOCOL:
DEV_PLUS_LOCKED_TEST
APPROVED_PHASE_9_SPLIT_SEED:
20260810
APPROVED_PHASE_9_DEVELOPMENT_METRIC:
MACRO_F1
APPROVED_PHASE_9_INTERNAL_CV:
3_FOLD_STRATIFIED
APPROVED_PHASE_9_INTERNAL_CV_RANDOM_STATE:
42
APPROVED_PHASE_9_BASELINE_EXECUTION:
NO
APPROVED_PHASE_9_FIRST_HYPOTHESIS:
LOSS
APPROVED_SAMPLE_LEVEL_REVIEW:
NO
APPROVED_EXTERNAL_DATA_ACQUISITION:
NO
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
PHASE_9_3_TRAINING_AUTHORIZATION:
NOT GRANTED
```

The source is restricted to the historical canonical training pool. The frozen split record is [`configs/splits/deep3-postholdout-research-01.json`](../configs/splits/deep3-postholdout-research-01.json); the original canonical holdout remains historical evidence only. No Phase 9.2 model training, model evaluation, checkpoint creation, or locked-test model evaluation occurred.