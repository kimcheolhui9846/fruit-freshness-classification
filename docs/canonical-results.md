# Canonical Result Interpretation and Publication Decision

## Scope

This Phase 8.5 record interprets the frozen result for `deep3-canonical-reference-01` and states the owner-approved publication boundary. It publishes aggregate metrics, per-class metrics, and aggregated confusion-matrix documentation only. The evaluated set is the fixed 5,372-example internal holdout described in [canonical-holdout-evaluation.md](canonical-holdout-evaluation.md).

No post-holdout tuning occurred. No holdout reevaluation, alternate-checkpoint evaluation, sample-level image review, or inference rerun occurred in this Phase. This is an internal fixed holdout result, not an external benchmark, leaderboard, production-validation, or generalization claim.

## Locked Result Summary

| Metric | Value |
| --- | ---: |
| Correct / total | 5,133 / 5,372 |
| Top-1 accuracy | 0.955510 |
| Wilson 95% descriptive interval | [0.949663, 0.960706] |
| Macro F1 | 0.903737 |
| Balanced accuracy | 0.899969 |
| Top-2 accuracy | 0.981199 |
| Top-3 accuracy | 0.992740 |

The locked protocol used three fold-best EMA checkpoints, equal raw-logit averaging, equal original/horizontal-flip TTA averaging, deterministic validation preprocessing, and `argmax` top-1 selection. The final raw checkpoint was excluded. The CLI and detailed read-only metric pass matched exactly within `1e-12`.

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| freshapples | 0.984871 | 1.000000 | 0.992378 | 651 |
| freshbanana | 0.986994 | 1.000000 | 0.993455 | 683 |
| freshcapsicum | 0.985915 | 1.000000 | 0.992908 | 210 |
| freshcucumber | 0.975758 | 0.964072 | 0.969880 | 167 |
| freshoranges | 0.997041 | 1.000000 | 0.998519 | 337 |
| freshpotato | 0.659574 | 0.300971 | 0.413333 | 103 |
| freshtomato | 0.877030 | 0.945000 | 0.909747 | 400 |
| rottenapples | 0.992941 | 1.000000 | 0.996458 | 844 |
| rottenbanana | 0.943890 | 1.000000 | 0.971135 | 757 |
| rottencapsicum | 0.994709 | 1.000000 | 0.997347 | 188 |
| rottencucumber | 0.860870 | 0.750000 | 0.801619 | 132 |
| rottenoranges | 1.000000 | 1.000000 | 1.000000 | 390 |
| rottenpotato | 0.679426 | 0.893082 | 0.771739 | 159 |
| rottentomato | 0.970370 | 0.746439 | 0.843800 | 351 |

## Aggregated Confusion-Matrix Interpretation

The complete true-row / predicted-column aggregated matrix is published in [canonical-holdout-evaluation.md](canonical-holdout-evaluation.md). It contains 5,372 counts and a 5,133-count diagonal. The largest off-diagonal counts are summarized below; these are aggregate counts, not sample-level predictions or images.

| True class | Predicted class | Count |
| --- | --- | ---: |
| rottentomato | freshtomato | 51 |
| freshpotato | rottenpotato | 44 |
| freshpotato | rottenbanana | 17 |
| rottentomato | freshpotato | 15 |
| rottentomato | rottencucumber | 15 |
| rottenpotato | rottenbanana | 14 |
| rottencucumber | rottenbanana | 13 |
| rottencucumber | rottenpotato | 13 |

The lower `freshpotato` recall (0.300971) and its 44 predictions as `rottenpotato` are the clearest class-specific limitation. `rottencucumber` and `rottentomato` also have lower recall than the strongest classes. These observations describe the locked aggregate outcome only; they do not establish a causal explanation without a separately approved analysis.

## Interpretation Boundary

The high aggregate top-1 result coexists with uneven class behavior. Macro F1 and balanced accuracy are reported alongside top-1 accuracy because the holdout supports differ by class. The result is limited by this fixed source, the project cleaning and split policy, the absence of external validation, and the absence of post-holdout analysis. It must not be used to claim state-of-the-art performance, production readiness, fairness, safety, or performance on another dataset.

## Approved Publication Decision

Aggregate metrics, the per-class table, and aggregated confusion-matrix documentation are tracked Markdown. They are derived summaries, not releases of the underlying evaluation outputs.

Raw logits and predictions published: No

Checkpoint publication: No

Weight publication: No

Dataset publication: No

Training-state publication: No

Execution-log publication: No

GitHub Actions artifact upload: No

Release asset upload: No

Release creation: No

Tag creation: No

Retained locally through Phase 8.6.

The local CLI log, evaluation JSON, classification CSV, confusion-matrix CSV, prediction archive, checkpoints, weights, and training state are neither copied nor moved into this worktree. Their exact local-only inventory remains in [canonical-holdout-artifacts.md](canonical-holdout-artifacts.md).

## License and Provenance Audit

The MIT License applies to repository source code and project-authored documentation, including this result interpretation and the model card. It does not transfer rights to the external dataset or to future trained-weight distribution.

The evaluation source is `Densu341/Fresh-rotten-fruit` at the pinned revision recorded in the holdout evaluation. The public dataset metadata observed during the earlier governance audit was labeled `openrail`, while the surfaced dataset-card README was empty. This record does not infer additional permission, ownership, attribution completeness, or redistribution rights from that metadata. Dataset images are not redistributed through this repository.

Trained checkpoints and weights remain unpublished and require a separate review before publication. The approved aggregate result documentation contains no dataset image, checkpoint tensor, training state, raw logit, raw prediction, sample identifier, or local path. This is an operational governance record, not legal advice.