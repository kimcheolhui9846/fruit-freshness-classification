# Canonical run closure

## Closure record

```text
RUN_ID:
deep3-canonical-reference-01
CANONICAL_RUN_STATUS:
CLOSED_REFERENCE
TRAINING:
COMPLETED
LOCKED_HOLDOUT_EVALUATION:
COMPLETED
POST_HOLDOUT_TUNING:
NO
```

The canonical reference run is closed as a completed training and locked internal-holdout evidence record. This closure does not publish or transfer a binary artifact.

## Frozen identity and integrity references

| Field | Frozen value |
|---|---|
| Phase 8.6 start main | `9fc7477bb1da43eeefbae5e497a2ca76310871f5` |
| Training commit | `0c669d58852082785c79699231e09b5ae26757cc` |
| Evaluation commit | `4b3808efb3abaf4682e1150ce69ddcdb6585e451` |
| Configuration | `configs/deep3_canonical.toml` |
| Configuration SHA-256 | `8d40ed34ddcb0eeaea4ca9e03754c579c983e71d1e3b4ae121c512d1fc073c42` |
| Dataset repository and revision | `Densu341/Fresh-rotten-fruit` at `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Dataset archive SHA-256 | `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e` |
| Final raw checkpoint SHA-256 | `3fb0e5575ddc4c6ca2bceb955d17a85fd5965bc325ff5b261dded5dab5cbb29f` |
| Evaluation JSON SHA-256 | `592b88a506d946fcb3b4108f3dacfcd0fe15202b8adeda009f61aeaa29446443` |
| Classification CSV SHA-256 | `8c8422311120ca75459ad33a9ecd4541415c2011deb5b708ebf7525d4c2b8213` |
| Confusion CSV SHA-256 | `64bbdbc156da4061ccf093a0e51ab6b74706aca441eb34c2debe483797a5d444` |
| Predictions NPZ SHA-256 | `f36783f2be1d09bbd7178b734ba70023d54860f018a067d9f9cb1b3794331e0c` |

No private filesystem path is part of this public record.

## Locked internal-holdout summary

The already-completed locked evaluation recorded 5,133 correct top-1 predictions out of 5,372 examples: top-1 `0.955510`, macro F1 `0.903737`, balanced accuracy `0.899969`, top-2 `0.981199`, and top-3 `0.992740`. These frozen aggregates are not newly derived values and are not an external benchmark or production claim. The holdout shares byte-identical images with the training pool — 1,618 of its 5,372 rows — so this figure is higher than it would be on distinct images; on rows without such a copy Top-1 is 0.9414. The figure is unrevised. See [the dataset duplication audit](dataset-duplication-audit.md).

## Publication and retention boundary

```text
DOCUMENTATION: PUBLIC
MODEL_WEIGHTS: LOCAL_ONLY
FOLD_CHECKPOINTS: LOCAL_ONLY
FINAL_RAW_CHECKPOINT: LOCAL_ONLY
TRAINING_STATE: LOCAL_ONLY
TRAINING_LOG: LOCAL_ONLY
EVALUATION_LOG: LOCAL_ONLY
RAW_LOGITS: LOCAL_ONLY
RAW_PREDICTIONS: LOCAL_ONLY
DATASET: NOT_REDISTRIBUTED
DATASET_LICENSE_CLEARANCE:
NOT_CONFIRMED
BINARY_RETENTION:
KEEP_LOCAL_ONLY
RETENTION_DURATION:
UNTIL_EXPLICIT_OWNER_CHANGE
DELETION_AUTHORIZED:
NO
RELOCATION_AUTHORIZED:
NO
CONVERSION_AUTHORIZED:
NO
PACKAGING_AUTHORIZED:
NO
REMOTE_UPLOAD_AUTHORIZED:
NO
```

The MIT license applies only to repository software and project-authored documentation. The external dataset has separate terms; clearance is `NOT_CONFIRMED`, and local-only binaries do not establish distribution rights.

## Holdout boundary for future work

The fixed holdout has already been observed and interpreted. No post-holdout tuning, reevaluation, alternate-checkpoint evaluation, or sample-level image review was performed. Later model development is post-holdout research. Future experiments must use a new experiment identity. After tuning begins, the same holdout is not untouched evidence after tuning and must not be presented as an independent untouched validation result.

## CI boundary

```text
CI_LOCAL_BINARY_ARTIFACT_REQUIREMENT:
NO
CI_PRODUCTION_DATASET_ACCESS:
NO
CI_CUDA_REQUIREMENT:
NO
```

Normal repository CI is offline and does not require retained binaries, production dataset access, CUDA, canonical training, or real holdout evaluation.
