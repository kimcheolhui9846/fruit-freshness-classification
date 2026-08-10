# Post-holdout split freeze

## Scope and frozen boundary

This document records the Phase 9.2 data-protocol freeze for `deep3-postholdout-research-01`. It is a reproducibility record, not a model-result report.

```text
PROTOCOL:
DEV_PLUS_LOCKED_TEST
SOURCE_POOL:
HISTORICAL_CANONICAL_TRAIN_ONLY
SOURCE_POOL_SIZE:
21486
CANONICAL_HOLDOUT_SIZE:
5372
LOCKED_TEST_FRACTION:
0.20
SPLIT_SEED:
20260810
STRATIFIED:
YES
PRIMARY_SELECTION_METRIC:
Macro F1
INTERNAL_CV:
3-fold stratified, shuffle=true, random_state=42
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
```

The source is exactly the reconstructed historical canonical training pool. The 5,372-example canonical holdout is `HISTORICAL_EVIDENCE_ONLY` and is excluded from both Phase 9 pools. The frozen record is [`deep3-postholdout-research-01.json`](../configs/splits/deep3-postholdout-research-01.json).

## Exact split result

| Pool | Count |
|---|---:|
| Source pool | 21,486 |
| Post-holdout development pool | 17,188 |
| Post-holdout locked test pool | 4,298 |
| Historical canonical holdout (excluded) | 5,372 |

Indices are zero-based positions relative to the reconstructed 21,486-example source pool. Hashes use SHA-256 over the listed signed 64-bit little-endian integer sequence.

| Identity | SHA-256 |
|---|---|
| Source label sequence | `69dd64bc924bc70c13eb80f9c728635e7a60e2e813b0a636d54f964dc2fd0460` |
| Development indices | `329086d616fbf72e79bb65f00966259d6788cd8ff85daf4aff444688e06dfc19` |
| Locked-test indices | `386498e238b2e5b905b599c63f633b3482f08370f111fcf5bca08dea2b9166c2` |
| Repository SHA at materialization | `b81ab1aedb19b35beef4db215c04746bb50c030c` |

## Stratification record

The following are label counts only. No samples, paths, images, predictions, logits, checkpoints, or other binary artifacts are included.

| Class | Source | Development | Locked test |
|---|---:|---:|---:|
| freshapples | 2,564 | 2,051 | 513 |
| freshbanana | 2,677 | 2,142 | 535 |
| freshcapsicum | 780 | 624 | 156 |
| freshcucumber | 608 | 486 | 122 |
| freshoranges | 1,517 | 1,214 | 303 |
| freshpotato | 433 | 347 | 86 |
| freshtomato | 1,458 | 1,166 | 292 |
| rottenapples | 3,392 | 2,714 | 678 |
| rottenbanana | 3,075 | 2,460 | 615 |
| rottencapsicum | 713 | 570 | 143 |
| rottencucumber | 544 | 435 | 109 |
| rottenoranges | 1,608 | 1,286 | 322 |
| rottenpotato | 643 | 514 | 129 |
| rottentomato | 1,474 | 1,179 | 295 |
| **Total** | **21,486** | **17,188** | **4,298** |

## Leakage and immutability audit

- Development/locked-test overlap: `0`
- Development plus locked-test coverage: `21,486 / 21,486`
- Duplicate development indices: `0`
- Duplicate locked-test indices: `0`
- Canonical holdout overlap: `0`
- Dataset: `Densu341/Fresh-rotten-fruit`
- Dataset revision: `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`
- Unsafe overwrite policy: the freeze script refuses an existing manifest.

The label sequence and relative index lists are immutable evidence. Regenerating or replacing this manifest requires a separately approved governance Phase.

## Access policy

Labels were used only to construct and verify stratification. The locked test remains unobserved by every model.

- Model predictions: NO
- Model metrics: NO
- Candidate selection using locked test: NO
- Checkpoint selection using locked test: NO
- Augmentation, loss, architecture, sampler, or optimizer decisions using locked test: NO
- Model training in Phase 9.2: NO
- Model evaluation in Phase 9.2: NO
- External dataset acquisition in Phase 9.2: NO

Future Phase 9 development may use only `POST_HOLDOUT_DEVELOPMENT_POOL`. Evaluation of the locked test requires an explicit future final-evaluation gate; this document grants no training or evaluation authority.