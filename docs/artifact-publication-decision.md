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

The following fields are intentionally unresolved. They require a new explicit owner approval before Phase 8.6 may begin.

APPROVED_NEXT_ACTION:
<PUBLISH_DOCUMENTATION_ONLY |
 KEEP_ALL_BINARY_ARTIFACTS_LOCAL_ONLY |
 PREPARE_SAFE_MODEL_PACKAGE_AFTER_CLEARANCE |
 REQUEST_DATASET_RIGHTS_CLARIFICATION |
 DEFER>

APPROVED_MODEL_WEIGHT_PUBLICATION:
<YES_AFTER_CLEARANCE | NO | DEFER>

APPROVED_CHECKPOINT_SET:
<FOLD_BEST_ENSEMBLE | NONE | DEFER>

APPROVED_ARTIFACT_FORMAT:
<PYTORCH_STATE_DICT |
 SAFETENSORS_AFTER_VERIFIED_CONVERSION |
 NONE |
 DEFER>

APPROVED_HOSTING_DESTINATION:
<GITHUB_RELEASE |
 HUGGING_FACE_MODEL_REPOSITORY |
 OWNER_CONTROLLED_PRIVATE_STORAGE |
 NONE |
 DEFER>

APPROVED_DATASET_LICENSE_CLEARANCE:
<CONFIRMED | NOT_CONFIRMED | DEFER>

APPROVED_MODEL_CARD_PUBLICATION:
<YES | NO | DEFER>

APPROVED_BINARY_RETENTION:
<KEEP_LOCAL_ONLY |
 CREATE_OWNER_CONTROLLED_BACKUP |
 DEFER>

APPROVED_RELEASE_CREATION:
<YES | NO | DEFER>

APPROVED_TAG_CREATION:
<YES | NO | DEFER>

Do not resolve these fields or begin Phase 8.6 without a new owner approval.