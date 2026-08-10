# Canonical Holdout Evaluation

## Scope

The trained `deep3-canonical-reference-01` fold ensemble was evaluated once against the fixed internal holdout. This is the first trained holdout evaluation for this run. The holdout contains 5,372 examples, the evaluation protocol was locked before any metric was read, and no post-holdout tuning occurred.

This is an internal fixed holdout result, not an external benchmark, leaderboard result, production validation, or state-of-the-art result.

## Frozen Identity

| Item | Frozen value |
| --- | --- |
| Training run | `deep3-canonical-reference-01` |
| Training commit | `0c669d58852082785c79699231e09b5ae26757cc` |
| Evaluation commit | `4b3808efb3abaf4682e1150ce69ddcdb6585e451` |
| Configuration | `configs/deep3_canonical.toml` |
| Configuration SHA-256 | `8d40ed34ddcb0eeaea4ca9e03754c579c983e71d1e3b4ae121c512d1fc073c42` |
| Dataset | `Densu341/Fresh-rotten-fruit` at revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` |
| Dataset archive SHA-256 | `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e` |
| Label SHA-256 | `c0c229be2509141e1ca3ddf994192b05de69bdec61ea0f97ed554d905eacaae9` |
| Fold 1 checkpoint SHA-256 | `e89a9f7b6128f1c6a8fbd4885d86ee81ca5c0eac4c6601c35fbabcfae5822a24` |
| Fold 2 checkpoint SHA-256 | `19eb264786f339fb738c218b891283ac17a6fa449a6f16ae25b18b33c93299ff` |
| Fold 3 checkpoint SHA-256 | `85a3a2ac5cb373906c81de7a69e223b7c353dedfd51b19ef975340040bd27068` |
| Environment | Python 3.12.10; torch 2.6.0+cu124; torchvision 0.21.0+cu124; CUDA 12.4 |
| Device | NVIDIA GeForce RTX 3070 Ti, 8 GiB; driver 591.86 |

The preserved label order is: `freshapples`, `freshbanana`, `freshcapsicum`, `freshcucumber`, `freshoranges`, `freshpotato`, `freshtomato`, `rottenapples`, `rottenbanana`, `rottencapsicum`, `rottencucumber`, `rottenoranges`, `rottenpotato`, `rottentomato`.

## Protocol

- Fold-best EMA checkpoints were loaded in order: `best_model_fold1.pt`, `best_model_fold2.pt`, and `best_model_fold3.pt`.
- The final raw checkpoint policy is explicit: last_model_weights.pt is excluded from the canonical holdout ensemble.
- The ensemble uses an equal raw-logit ensemble across the three folds.
- Inference averages the original view and horizontal-flip TTA view equally.
- The committed deterministic validation transform was used: resize 256, center crop 224, tensor conversion, and ImageNet normalization.
- Training/evaluation batch size: 64.
- Top-1 uses `argmax`; no thresholding, fold weighting, or checkpoint substitution was used. No threshold tuning was performed.

## Primary Result

| Metric | Value |
| --- | ---: |
| Correct / total | 5,133 / 5,372 |
| Top-1 accuracy | 0.955510 |
| Wilson 95% descriptive interval | [0.949663, 0.960706] |

## Secondary Results

| Metric | Value |
| --- | ---: |
| Macro F1 | 0.903737 |
| Balanced accuracy | 0.899969 |
| Top-2 accuracy | 0.981199 |
| Top-3 accuracy | 0.992740 |

## Per-Class Metrics

Per-class precision, recall, and F1 use zero-division-safe reporting (`zero_division=0`). Values are rounded to six decimal places only for this document.

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

## Confusion Matrix

Rows are true classes; columns are predicted classes. The labels below retain the frozen order from the identity section. The matrix sum is 5,372 and its diagonal sum is 5,133.

```text
true\pred,freshapples,freshbanana,freshcapsicum,freshcucumber,freshoranges,freshpotato,freshtomato,rottenapples,rottenbanana,rottencapsicum,rottencucumber,rottenoranges,rottenpotato,rottentomato
freshapples,651,0,0,0,0,0,0,0,0,0,0,0,0,0
freshbanana,0,683,0,0,0,0,0,0,0,0,0,0,0,0
freshcapsicum,0,0,210,0,0,0,0,0,0,0,0,0,0,0
freshcucumber,0,0,1,161,0,0,0,0,1,0,1,0,3,0
freshoranges,0,0,0,0,337,0,0,0,0,0,0,0,0,0
freshpotato,0,7,0,0,0,31,1,3,17,0,0,0,44,0
freshtomato,10,0,0,0,1,0,378,2,0,1,0,0,0,8
rottenapples,0,0,0,0,0,0,0,844,0,0,0,0,0,0
rottenbanana,0,0,0,0,0,0,0,0,757,0,0,0,0,0
rottencapsicum,0,0,0,0,0,0,0,0,0,188,0,0,0,0
rottencucumber,0,1,2,3,0,0,1,0,13,0,99,0,13,0
rottenoranges,0,0,0,0,0,0,0,0,0,0,0,390,0,0
rottenpotato,0,1,0,0,0,1,0,1,14,0,0,0,142,0
rottentomato,0,0,0,1,0,15,51,0,0,0,15,0,7,262
```

## Crosscheck

The canonical CLI pass exited with code 0 and reported top-1 accuracy `0.9555100521221147`. The detailed read-only pass, using the same committed data, validation transform, fold order, equal raw-logit ensemble, CUDA autocast behavior, and horizontal-flip TTA, reported `0.9555100521221147` from 5,133 / 5,372. `cli_accuracy` and `detailed_accuracy` matched exactly within `1e-12` (`MATCH_EXACT_WITHIN_1E-12`).

The initial external detailed runner encountered a UTF-16 CLI-log decoding fault after inference and before any detailed artifact was written. The owner explicitly approved one replacement detailed pass. It retained the locked identity and protocol, produced the local-only detailed artifacts, and is the completed crosscheck recorded here. No result was selected from repeated alternatives.

## Limitations and Interpretation Boundary

- This is an internal fixed holdout; no external validation was performed.
- No state-of-the-art claim is made.
- This is not a production-quality claim or a generalization result.
- No post-holdout tuning occurred.
- The final raw checkpoint was not evaluated as a separate candidate or ensemble member.
- batch 64 is a different training trajectory from batch 192; this result does not compare their quality.
- Exact numerical rerun identity is not claimed because global seeding and deterministic-algorithm controls were not changed.
- The Wilson interval is descriptive only; it does not remove dataset-dependence assumptions.
- Dataset composition, labels, and source quality remain limitations.

## Phase 8.5 documentation status

Phase 8.5 documentation and publication decision is recorded. It adds only aggregate result interpretation, per-class metrics, aggregated confusion-matrix documentation, a model card, and a concise README summary. It does not rerun the holdout, tune after the holdout, evaluate alternate checkpoints, inspect sample images, or publish a dataset, checkpoint, weight, training state, execution log, raw logit, or raw prediction.

See [canonical-results.md](canonical-results.md) for interpretation and the approved publication boundary, and [model-card.md](model-card.md) for intended use and limitations. The binary artifact retention decision continues through Phase 8.6.