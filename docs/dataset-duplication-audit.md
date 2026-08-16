# Dataset Duplication Audit

## Status

```text
AUDIT_DATE:
2026-08-16
METHOD:
SHA-256 over image file bytes
GPU_HOURS:
0
MODEL_RUNS:
0
FINDING:
DUPLICATE_IMAGES_CROSS_EVERY_SPLIT_BOUNDARY
RECORDED_METRICS_REVISED:
NO
LOCKED_TEST_MODEL_FORWARD_PASSES:
0
```

The source dataset contains byte-identical copies of the same image, and the project's row-wise split scatters those copies across both sides of every boundary derived from it. Recorded results are therefore measured partly on images the model had already seen.

**No recorded metric is revised.** Every figure this repository reports remains what it always was: the value measured on the split that produced it. What this document adds is the size of the effect, so a reader can weigh those figures rather than discover the problem later.

## How the duplication arises

The source stores images under both a `Train` and a `Test` directory. `load_fruit_freshness_dataset` concatenates them into one table, filters eight class directories, and then calls `train_test_split(test_size=0.2, seed=42)` over **rows**. Nothing at any stage groups copies of one image together, so a random row split puts one copy on each side whenever a duplicate exists.

```text
SOURCE_ROWS:
30357
SOURCE_CLASSES:
22
FILTERED_ROWS:
26858
UNIQUE_IMAGES_AFTER_FILTER:
21413
DUPLICATE_GROUPS:
3781
EXTRA_COPIES:
5445
CROSS_CLASS_DUPLICATE_GROUPS:
0
GROUPS_SPANNING_TRAIN_AND_TEST_DIRECTORIES:
2198
```

**No duplicate group spans two class directories**, so no image carries contradictory labels. The redundancy is within classes only, which makes this a leakage problem rather than a labelling one.

## The class filter is correct

Four of the eight removed class directories are misspellings: `freshpatato`, `rottenpatato`, `freshtamto`, and `rottentamto`. All 1,248 of their images are byte-identical to images in the correctly spelled classes that are kept.

| Removed directory | Images | Byte-identical to a kept image |
|---|---:|---:|
| `freshpatato` | 270 | 270 |
| `rottenpatato` | 370 | 370 |
| `freshtamto` | 255 | 255 |
| `rottentamto` | 353 | 353 |

The filter is deduplicating, not discarding. The other four removed directories — `freshbittergroud`, `rottenbittergroud`, `freshokra`, `rottenokra` — are different vegetables and fall outside the fourteen-class problem.

This matters because those directories look, by name and count, like unused data that could enlarge the smallest class. They are not.

## Where the copies land

```text
CANONICAL_TRAIN_POOL:
21486
CANONICAL_HOLDOUT:
5372
HOLDOUT_ROWS_DUPLICATING_A_TRAIN_ROW:
1618
HOLDOUT_CONTAMINATED_FRACTION:
0.301
DISTINCT_IMAGES_ON_BOTH_SIDES:
1465
```

```text
POST_HOLDOUT_DEVELOPMENT:
17188
POST_HOLDOUT_LOCKED_TEST:
4298
LOCKED_ROWS_DUPLICATING_A_DEVELOPMENT_ROW:
1140
LOCKED_CONTAMINATED_FRACTION:
0.265
DEVELOPMENT_ROWS_WHOSE_COPY_SITS_IN_ANOTHER_CV_FOLD:
3312
```

The locked test has still never been evaluated and still has zero model forward passes. The contamination described here is a property of how it was constructed, not evidence that it was used.

## How much the recorded figures are inflated

Computed from prediction files already on disk. No model was run.

### Canonical holdout

| Subset | Rows | Top-1 | Macro F1 |
|---|---:|---:|---:|
| All rows, as reported | 5372 | 0.9555 | 0.9037 |
| Rows duplicating a training image | 1618 | 0.9883 | 0.7590 |
| Rows not duplicating a training image | 3754 | 0.9414 | 0.8957 |

### Development OOF, deterministic baseline

| Subset | Rows | Top-1 | Macro F1 |
|---|---:|---:|---:|
| All rows, as reported | 17188 | 0.9543 | 0.9019 |
| Rows whose copy sits in another fold | 3312 | 0.9885 | 0.7056 |
| Rows with no copy in another fold | 13876 | 0.9462 | 0.8969 |

**Top-1 is the comparison to read.** On the contaminated rows it is 0.988 in both tables, which is what memorisation looks like. Removing those rows moves canonical Top-1 from 0.9555 to 0.9414, a difference of 0.0141, and development Top-1 from 0.9543 to 0.9462, a difference of 0.0081.

**The macro F1 column of the subset rows is not comparable to the whole.** Macro F1 averages over classes, and the duplicated rows are concentrated in a few classes, so a subset's macro F1 reflects its class composition as much as its accuracy. It is shown for completeness, not as a corrected score.

### Why this size matters

The macro F1 difference between the full development set and its uncontaminated part is 0.0050. The effect Phase 9.6 attempted to measure was 0.0052 against the three-replicate mean. **The leak is the same size as the signal that phase was chasing.**

## What is affected and what is not

Extra copies by class:

| Class | Extra copies |
|---|---:|
| `freshbanana` | 1398 |
| `rottenapples` | 1293 |
| `freshapples` | 1127 |
| `rottenbanana` | 1078 |
| `freshcucumber` | 279 |
| `rottencucumber` | 255 |
| `rottentomato` | 14 |
| `freshtomato` | 1 |

**`freshpotato` and `rottenpotato` contain no duplicates at all**, nor do the capsicum or orange classes. Every Phase 9 finding about `freshpotato` therefore stands unchanged: the label audit, the per-class variance decomposition, the stability grouping, and the coverage observation are all computed on a class with no redundancy.

The measurement floor is also unaffected. It describes how far a result moves between runs of the same recipe on the same data, which duplication does not change.

What is affected is the level of the headline numbers: the canonical holdout result and the development baselines are higher than they would be on distinct images.

## Reproducing this

```powershell
.venv/Scripts/python.exe -m scripts.audit_dataset_duplication `
  --root <image-folder dataset root> `
  --classes freshapples freshbanana freshcapsicum freshcucumber freshoranges `
            freshpotato freshtomato rottenapples rottenbanana rottencapsicum `
            rottencucumber rottenoranges rottenpotato rottentomato `
  --output results/dataset-duplication-audit.json
```

The script hashes file bytes rather than decoded pixels, so every match it reports is exact. Two files that decode to the same picture through different encodings would not be counted, which makes these figures a lower bound on the true redundancy.

## What was decided

```text
DECISION:
RECORD_WITHOUT_REVISING
RECORDED_METRICS_CHANGED:
NONE
RESPLIT_ON_UNIQUE_IMAGES:
NOT_AUTHORIZED
```

The owner chose on 2026-08-16 to record the finding and leave the recorded numbers as they stand.

Two alternatives were considered and declined. Promoting the uncontaminated-subset figures to headline results would substitute a subset chosen after seeing the problem for the split the protocol froze beforehand, which is the kind of after-the-fact selection this project's governance exists to prevent. Re-splitting on unique images and retraining would be the scientifically clean repair, but it invalidates every frozen artifact — the canonical result, the split and CV manifests and their hashes, the baselines, the noise floor, and the deterministic baseline — and amounts to restarting the project from the split freeze.

Neither is foreclosed. Both remain available if the owner later decides the corrected level matters more than the frozen record.
