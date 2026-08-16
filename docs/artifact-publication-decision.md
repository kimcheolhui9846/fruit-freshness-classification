# Artifact Publication Decision

## Scope

This is the canonical Phase 8.5 artifact-publication governance record for `deep3-canonical-reference-01`. It applies the approved documentation-only action without copying, moving, uploading, or publishing a dataset, checkpoint, weight, training state, log, raw logit, raw prediction, or other binary artifact.

## Current Decision

CURRENT_PUBLICATION_ACTION:
PUBLISH_DOCUMENTATION_ONLY

MODEL_CARD_PUBLICATION:
YES

AGGREGATED_RESULTS_PUBLICATION:
YES

PER_CLASS_METRICS_PUBLICATION:
YES

AGGREGATED_CONFUSION_MATRIX_PUBLICATION:
YES

MODEL_WEIGHT_PUBLICATION:
BLOCKED_PENDING_LICENSE_AND_PROVENANCE_CLEARANCE

FOLD_CHECKPOINT_PUBLICATION:
BLOCKED_PENDING_LICENSE_AND_PROVENANCE_CLEARANCE

FINAL_RAW_CHECKPOINT_PUBLICATION:
NO_CURRENT_USE_CASE

TRAINING_STATE_PUBLICATION:
NO

TRAINING_LOG_PUBLICATION:
NO

EVALUATION_LOG_PUBLICATION:
NO

RAW_LOGIT_PUBLICATION:
NO

RAW_PREDICTION_PUBLICATION:
NO

DATASET_PUBLICATION:
NO

GITHUB_ACTIONS_ARTIFACT_UPLOAD:
NO

RELEASE_ASSET_UPLOAD:
NO

NEW_RELEASE:
NO

NEW_TAG:
NO

LOCAL_ARTIFACT_RETENTION:
YES

## Recommendation

PRIMARY_RECOMMENDATION:
PUBLISH_DOCUMENTATION_ONLY

SECONDARY_RECOMMENDATION:
KEEP_ALL_BINARY_ARTIFACTS_LOCAL_ONLY

BINARY_PUBLICATION_GATE:
BLOCKED

The repository MIT terms cover repository software and project-authored documentation. External dataset rights remain separate. Public accessibility does not itself establish redistribution rights. The surfaced `openrail` metadata is not treated as sufficient trained-weight clearance, and trained-weight publication remains blocked until a separate rights/provenance review is completed. This is an operational governance decision, not legal advice.

Normal CI does not require local artifacts, CUDA, or production dataset access. It does not upload GitHub Actions artifacts, rerun production evaluation, or depend on a trained checkpoint.

## Retention and Phase Boundary

All binary artifacts remain local-only through Phase 8.6. This record does not authorize their deletion, relocation, conversion, bundling, release attachment, or remote upload. Documentation-only publication does not create a new Release or tag.

## Phase 8.6 Owner Gate

These fields were left intentionally unresolved when this document was written. The owner resolved them on 2026-08-16, after Phase 9 closed and `v0.2.0` was published. The values below are the decision; the unfilled form is preserved in git history.

The governing choice is that trained weights are not published at all. The earlier stance deferred the question behind a rights and provenance review; the owner instead judged the weights an asset that should not be distributed, which closes the question rather than completing that review.

APPROVED_NEXT_ACTION:
KEEP_ALL_BINARY_ARTIFACTS_LOCAL_ONLY

APPROVED_MODEL_WEIGHT_PUBLICATION:
NO

APPROVED_CHECKPOINT_SET:
NONE

APPROVED_ARTIFACT_FORMAT:
NONE

APPROVED_HOSTING_DESTINATION:
NONE

APPROVED_DATASET_LICENSE_CLEARANCE:
DEFER

APPROVED_MODEL_CARD_PUBLICATION:
YES

APPROVED_BINARY_RETENTION:
KEEP_LOCAL_ONLY

APPROVED_RELEASE_CREATION:
YES

APPROVED_TAG_CREATION:
YES

Notes on the values that do not follow directly from the weight decision:

- `APPROVED_DATASET_LICENSE_CLEARANCE` stays `DEFER`. The owner did not resolve the dataset rights question and did not need to: with nothing distributed, the clearance is moot rather than settled. It would have to be answered before any future reversal, and recording it as confirmed would be a claim nobody has verified.
- `APPROVED_MODEL_CARD_PUBLICATION` is `YES` because it records an accomplished fact. `model-card.md` is documentation in this public repository and already published.
- `APPROVED_RELEASE_CREATION` and `APPROVED_TAG_CREATION` are `YES` for the same reason. `v0.1.0` and `v0.2.0` are both published, both source-only, both with zero assets, so neither release contradicts the weight decision.

Reversing `APPROVED_MODEL_WEIGHT_PUBLICATION` requires a new explicit owner approval and would reopen the dataset rights and provenance question this decision made moot.