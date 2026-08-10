# Model Card: deep3-canonical-reference-01

## Model Status

`deep3-canonical-reference-01` is a frozen research reference run. It completed one three-fold training execution and one locked internal holdout evaluation. The model binary is not distributed: checkpoints, weights, training state, execution logs, raw logits, and raw predictions remain local-only through Phase 8.6.

## Model Details

| Item | Value |
| --- | --- |
| Task | 14-class fresh/rotten fruit image classification |
| Architecture | CMT classifier |
| Training configuration | `configs/deep3_canonical.toml` |
| Training batch size | 64 |
| Ensemble | Three fold-best EMA checkpoints with equal raw-logit averaging |
| Test-time augmentation | Equal original and horizontal-flip views |
| Validation preprocessing | Resize 256, center crop 224, ImageNet normalization |
| Output | Ordered 14-class logits; top-1 uses `argmax` |

The frozen label order is `freshapples`, `freshbanana`, `freshcapsicum`, `freshcucumber`, `freshoranges`, `freshpotato`, `freshtomato`, `rottenapples`, `rottenbanana`, `rottencapsicum`, `rottencucumber`, `rottenoranges`, `rottenpotato`, and `rottentomato`.

## Intended Use

This model is documented for research reproduction, experiment comparison, and study of the recorded internal holdout result. The repository provides labeled-holdout evaluation only. It does not provide a supported generic unlabeled-image inference interface, a clinical or food-safety decision workflow, or a production deployment contract.

## Training and Evaluation Data

Training and evaluation used the external `Densu341/Fresh-rotten-fruit` Hugging Face dataset at the pinned revision and archive identity described in [dataset.md](dataset.md). The project preserves its existing label filtering, RGB conversion, and seed-42 80/20 split: 26,858 filtered examples, 21,486 training examples, and 5,372 fixed holdout examples.

Dataset files are not redistributed through this repository. The dataset composition, source quality, labeling, and class balance limit any interpretation beyond the documented internal holdout.

## Evaluation

The locked internal holdout result is 5,133 / 5,372 top-1 correct predictions (0.955510), macro F1 of 0.903737, and balanced accuracy of 0.899969. The result used the fold-best EMA ensemble only; the final raw checkpoint was excluded. No post-holdout tuning occurred.

The complete frozen protocol, per-class metrics, and aggregated confusion matrix are in [canonical-holdout-evaluation.md](canonical-holdout-evaluation.md). The interpretation of lower-performing classes, including `freshpotato`, `rottencucumber`, and `rottentomato`, is in [canonical-results.md](canonical-results.md).

## Limitations and Risks

- This is one internal fixed holdout evaluation, not an external benchmark, production-validation, or generalization claim.
- `freshpotato` has materially lower recall and F1 than the strongest classes; aggregate accuracy alone is insufficient for class-specific decisions.
- No alternate checkpoint evaluation, sample-level review, post-holdout tuning, or independent-machine reproduction was performed.
- Dataset and label limitations may affect performance, coverage, and bias; no fairness or safety assessment is claimed.
- The derived batch-64 configuration is a different trajectory from the original batch-192 configuration.

## Artifact Availability

No checkpoint, trained weight, training state, execution log, raw logit, raw prediction, dataset, or other binary artifact is downloadable from this repository, GitHub Actions, or a Release. A separate owner-approved artifact and provenance review is required before any publication change.

## License and Provenance

The MIT License applies to the repository software and project-authored documentation. The external `Densu341/Fresh-rotten-fruit` dataset is governed separately by its original-source terms and is not redistributed here. The public metadata observed during the governance audit was labeled `openrail`, but the surfaced dataset-card README was empty; this model card does not infer redistribution permission from that metadata.

Trained weights are not distributed and require a separate review before publication. See [governance-decisions.md](governance-decisions.md) for the repository license, citation, dataset, and trained-weight boundaries.