# Post-Holdout Stability Measurement Limit

## Status

```text
PHASE:
9.9
ROLE:
NEGATIVE_METHODOLOGICAL_RESULT
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
COMPLETED
TRAINING_RUN_COUNT:
0
GPU_HOURS:
0
OUTCOME:
PER_IMAGE_TESTING_DOES_NOT_ESCAPE_THE_MEASUREMENT_FLOOR
LOCKED_TEST_MODEL_ACCESS:
NO
ARTIFACT_PUBLICATION:
LOCAL_ONLY
```

Phase 9.8 registered Phase 9.9 as `FRESHPOTATO_STABILITY` and designed it nowhere. This document records what designing it produced: a per-image test that looked well powered, a calibration against runs that share an identical configuration, and the demonstration that the test does not work. No training run was executed and none is proposed.

## What was attempted

Phase 9.8 left `freshpotato` characterised and unexplained. Of 347 development images, 183 are misclassified in all three unseeded replicates, 102 flip between runs, and 62 are correct in all three. The class carries 90.56% of Macro F1's run-to-run variance, and its own two-sigma is 0.147833 — so wide that no single-run comparison of its F1 can resolve anything.

The proposed escape was to change the unit of analysis. Rather than compare one class-level F1 against another, compare the two runs image by image on the same 347 images and test whether the disagreement is asymmetric. McNemar's exact test on discordant pairs has a sample size of hundreds rather than three, and on that reasoning it appeared to be adequately powered by a single run per arm.

**That reasoning is wrong, and the error is instructive enough to record rather than quietly drop.**

## The calibration that refutes it

A test's false positive rate can be measured directly here, because this project has four runs of the *same* configuration. Every pair of them is a case where the null hypothesis is true by construction: the two runs differ only by seed and, for one of them, by the deterministic kernel policy adopted in Phase 9.7.

| Pair, identical configuration | b | c | discordant | p, exact |
|---|---:|---:|---:|---:|
| `baseline` vs `rep002` | 19 | 57 | 76 | 0.000 |
| `baseline` vs `rep003` | 24 | 30 | 54 | 0.497 |
| `baseline` vs `det` | 21 | 40 | 61 | 0.020 |
| `rep002` vs `rep003` | 53 | 21 | 74 | 0.000 |
| `rep002` vs `det` | 48 | 29 | 77 | 0.040 |
| `rep003` vs `det` | 27 | 40 | 67 | 0.142 |

```text
NULL_PAIRS_TESTED:
6
NULL_PAIRS_SIGNIFICANT_AT_0.05:
4
OBSERVED_FALSE_POSITIVE_RATE:
about 0.67 against a nominal 0.05
```

**Four of six comparisons between identical configurations reject the null.** A test that fires two thirds of the time when nothing has changed cannot be used to decide whether something has changed.

Two limits on that rate. Six pairs is a small calibration set, and the pairs are not independent of one another: they come from four runs, so each run appears in three of them, and a single aberrant run inflates several rows at once. The figure 0.67 should be read as "far above nominal", not as a precise false positive rate. The conclusion does not rest on the precision: one pair of identical configurations returning `p = 0.000` is already sufficient to show the test is miscalibrated for this use, and three separate pairs do.

### Why it fails

McNemar tests marginal homogeneity, and the marginals genuinely differ between runs. The number of `freshpotato` images each run gets wrong is 252, 214, 246, and 233 — a swing of 38 across four runs of the same recipe.

At the median discordant count of 70, the exact test calls a result significant once the net change `c - b` reaches 18. **The swing produced by changing nothing is roughly twice the change the test calls significant.** The test is detecting seed noise faithfully; seed noise is simply not what it was asked about.

The deeper point is about effective sample size. Splitting one pair of runs into 347 image-level comparisons does not create 347 independent observations. The quantity that varies between runs is a property of the run — which decision boundary the training landed on — and it moves all 347 images together. **The effective sample size is the number of runs, not the number of images, and no re-slicing of a single pair changes that.**

## The same wall, three times

This is the third design in this project defeated by the same fact, and the pattern is worth stating plainly because a fourth attempt would otherwise be natural.

| Phase | Attempt | Why it failed |
|---|---|---|
| 9.6 | Measure the intervention on aggregate Macro F1 | The acceptance margin of 0.010 sat below the 0.012177 noise floor |
| 9.8 | Move to the per-class F1 of the target class | `freshpotato` is the noisiest class in the set; its two-sigma is 0.147833 |
| 9.9 | Move to per-image tests within one pair of runs | The effective sample size is runs, not images |

Each attempt moved to a finer-grained measurement hoping to gain power. Fineness of the measurement is not the constraint. **The constraint is that the thing being measured varies at the level of the run, and only more runs buy information about it.**

## What would actually be needed

A valid test needs the run as the unit of analysis, which means several runs per arm. Using the recorded per-class two-sigma of 0.147833 and the measured 9.13-hour run duration:

```text
MINIMUM_DESIGN:
three runs per arm
RUN_COUNT:
6
GPU_HOURS:
55
POWER_AT_THAT_SIZE:
low; three samples per arm estimate a variance with two degrees of freedom
```

Fifty-five hours buys a comparison that is still underpowered for an effect smaller than the class's own spread. This is recorded so that the cost is visible before anyone proposes it, not as a recommendation.

## Descriptive: the failure is shared, not divergent

One question could be answered from artifacts already on disk, at no GPU cost: are the runs failing on *different* images, in a way that combining them would repair?

| Quantity, `freshpotato`, n = 347 | Correct |
|---|---:|
| `baseline` | 95 |
| `rep002` | 133 |
| `rep003` | 101 |
| `det` | 114 |
| Logit-average ensemble of the three unseeded runs | 100 |
| Correct in every unseeded run | 62 |
| Correct in at least one unseeded run | 164 |

**The ensemble is worse than the best single run**, at 100 against 133, and worse than the mean of its members. Averaging does not repair the failure because the runs are not making independent errors: they share a core of 183 images they are all confidently wrong about, with mean probability on the true class of 0.06 and a margin of −0.47 against `rottenpotato`. Averaging confident agreement preserves it.

Even the oracle bound — picking, per image, whichever run happened to be right — reaches only 164 of 347. **More than half the class is wrong in every run there is.**

```text
DESCRIPTIVE_STATUS:
EXPLORATORY_DESCRIPTIVE
MAY_ADVANCE_A_CANDIDATE:
NO
MAY_SUPPORT_A_CLAIM:
NO
```

No rule about what these numbers mean was frozen before they were computed. They describe; they decide nothing.

## What this closes and what it does not

```text
FRESHPOTATO_STABILITY_AS_A_TESTABLE_PHASE:
CLOSED_NOT_MEASURABLE_AT_THIS_SCALE
PER_IMAGE_TESTING:
CALIBRATED_AND_REJECTED
LOSS001_VERDICT:
NOT_ADVANCED, unchanged and not re-scored
H1_STATUS:
CLOSED_BELOW_RESOLUTION, unchanged
DETERMINISTIC_BASELINE:
0.901891, unchanged
```

**Closed:** the plan to test a `freshpotato` intervention with one run per arm, by any statistic computed from that pair. The calibration above rules out the per-image route specifically, and the Phase 9.8 floor rules out the class-F1 route.

**Not closed:** the underlying question. Whether `freshpotato` can be learned stably remains open and unanswered. Nothing here says the class cannot be improved; it says this project cannot tell whether a given change improved it.

**Not claimed:** that the augmentation, the loss, the architecture, or anything else is responsible. A hypothesis was examined during design — that `ColorJitter` perturbs the surface-appearance cue by more than the cue itself — and the supporting measurement is recorded in the next section as a hypothesis for anyone with the budget to test it. It was never tested here.

## Recorded hypothesis, untested

While selecting an intervention, mean image brightness was compared between the 183 images wrong in every run and the 62 right in every run, before and after applying the training pipeline's `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02)`.

| | Always wrong | Always right | Separation, Cohen's d |
|---|---:|---:|---:|
| Raw images | 0.8170 | 0.8349 | −0.41 |
| After `ColorJitter` | 0.7807 | 0.7960 | −0.21 |

The jitter adds a within-group spread of 0.0566 against a between-group signal of 0.0178, roughly three times larger, and halves the separation by this measure.

This is suggestive and no more. Mean brightness is a proxy; a convolutional network does not classify on it, and a feature the network actually uses may be untouched by the same augmentation. The measurement is recorded because it was made, and because discarding a hypothesis that was examined would leave the design history incomplete. It is not evidence that removing `ColorJitter` would help, and this document authorizes no test of it.

## Boundaries

Not authorized by this document:

- any training run, of any configuration, of any length
- adding an `[augmentation]` configuration section or changing any transform
- re-scoring loss-001, the noise floor, Phase 9.7, or the Phase 9.8 baseline
- evaluating or inspecting the 4,298-example locked test, or re-evaluating the canonical holdout
- publishing weights, checkpoints, dataset copies, predictions, Releases, or tags

```text
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:
0
CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:
0
BINARY_PUBLICATION:
NO
```

## Owner approval block

```text
APPROVED_NEGATIVE_RESULT_RECORD:
YES
APPROVED_DESCRIPTIVE_ENSEMBLE_ANALYSIS:
YES
APPROVED_TRAINING_RUN_COUNT:
0
APPROVED_INTERVENTION_TEST:
NO
APPROVED_LOCKED_TEST_EVALUATION:
NO
APPROVED_WEIGHT_PUBLICATION:
NO
```

The owner approved recording the negative methodological result and the zero-GPU descriptive analysis on 2026-08-16, after being shown the calibration table that refuted the proposed design and declining the 55-hour alternative.
