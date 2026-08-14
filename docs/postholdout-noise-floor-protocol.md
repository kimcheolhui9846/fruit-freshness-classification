# Post-Holdout Run-to-Run Noise Floor Protocol

## Status

```text
PHASE:
9.6a
ROLE:
MEASUREMENT_NOT_HYPOTHESIS_TEST
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
IN_PROGRESS
REPLICATE_COUNT:
2
TOTAL_SAMPLE_SIZE:
3
LOCKED_TEST_MODEL_ACCESS:
NO
ARTIFACT_PUBLICATION:
LOCAL_ONLY
```

This measures how much the aggregate development OOF Macro F1 moves between identical runs. It tests no hypothesis and can advance no candidate. The interpretation rule is fixed here, before the replicates run.

## Why this is needed, and what it revealed

The Phase 9.6 loss-001 experiment returned `NOT_ADVANCED` at Macro F1 0.9102 against a threshold of 0.9112 — short by 0.0010, with a measured improvement of +0.0090. The loss-001 protocol had already recorded that the run-to-run noise floor was unmeasured and that results within roughly ±0.005 of the threshold should be read with that in mind. +0.0090 sits inside that band, so the recorded verdict cannot be distinguished from noise without a measurement.

Designing this measurement surfaced something larger. **The training pipeline sets no random seed at all.** There is no `torch.manual_seed`, `np.random.seed`, or `random.seed` anywhere in `src/` or `scripts/`; `torch.use_deterministic_algorithms` is never called; `cudnn_benchmark` is `true`, which selects kernels nondeterministically; and the training `DataLoader` uses `shuffle=True` with no `generator`.

Two consequences follow.

**Measuring the noise floor needs no seed change**, because there is no seed to change. Every run already draws fresh weight initialisation, batch order, mixup sampling, and augmentation randomness from OS entropy. Rerunning the baseline config unchanged *is* the replicate.

**The repository's training runs are not reproducible.** Dataset revision, archive hash, config hash, split indices, and CV fold indices are all frozen and verifiable; the training that consumes them is not. The same command on the same commit produces different weights and different metrics. That gap is recorded here and addressed in the follow-up described at the end — it is not fixed inside this measurement, because changing the training code mid-measurement would mean the replicates no longer measure the same pipeline that produced the baseline and loss-001 figures.

## What is measured

Three samples of aggregate development OOF Macro F1 produced by the identical baseline recipe:

| Sample | Source | Status |
|---|---|---|
| 1 | `deep3-postholdout-research-01-baseline` | already recorded, 0.9012 |
| 2 | `deep3-postholdout-research-01-baseline-rep002` | to run |
| 3 | `deep3-postholdout-research-01-baseline-rep003` | to run |

| Field | Value |
|---|---|
| Replicate configs | `configs/deep3_postholdout_baseline_rep002.toml`, `configs/deep3_postholdout_baseline_rep003.toml` |
| Permitted differences from the baseline config | `post_holdout.experiment_id` and `post_holdout.artifact_namespace` only |
| CV manifest | `configs/splits/deep3-postholdout-research-01-baseline-cv.json`, LF SHA-256 `494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f` |
| Split manifest | `configs/splits/deep3-postholdout-research-01.json` |

The folds are the baseline's own frozen folds. Changing them would measure split variation instead of training variation, which is a different quantity and not the one that governs the loss-001 comparison.

The replicate configs keep `parent_experiment_id = "deep3-postholdout-research-01"`, so they route through the existing canonical recipe check rather than the loss-experiment validator. No validator allowance is added for this measurement.

## Frozen interpretation rule

Let `s` be the sample standard deviation of the three Macro F1 values, computed with Bessel's correction, and let `d = 0.0090` be the loss-001 improvement already recorded.

```text
NOISE_STATISTIC:
sample standard deviation of the three baseline Macro F1 values
COMPARISON:
d = 0.0090 against 2s
INCONCLUSIVE:
d <= 2s
H1_EXHAUSTED_STANDS:
d > 2s
```

**If `d <= 2s`** — the loss-001 improvement is inside two standard deviations of ordinary run-to-run variation. The experiment was underpowered to detect an effect of that size, so "H1 exhausted" is not a sound conclusion from it. Phase 9.6 is recorded as `INCONCLUSIVE` and the next phase becomes the owner's decision.

**If `d > 2s`** — the improvement exceeds ordinary variation but still fell short of the pre-registered threshold. `H1 exhausted` stands and Phase 9.7 is H2 augmentation, as the loss-001 protocol fixed.

The range, `max - min`, is reported alongside for transparency. If the range and `2s` point to different conclusions, that disagreement is recorded rather than resolved in favour of whichever is convenient.

### What this measurement cannot do

**It does not reverse the loss-001 verdict.** `NOT_ADVANCED` was computed against a threshold frozen before that run and is recorded permanently. What is at stake is only whether the *next-phase selection* that followed from it was sound. A measurement that arrives after a result may inform what to do next; it may never be used to re-score the result itself.

**Three samples estimate `s` poorly.** With two degrees of freedom the estimate is wide, and a small `s` from three runs is weak evidence that variation is genuinely small. The rule is applied as written regardless, because choosing the sample size after seeing the spread would reintroduce exactly the freedom this protocol removes. The imprecision is reported with the result.

## Boundaries

Not authorized by this document:

- executing the replicates; each run requires the owner decision recorded in the approval block
- any change to the training recipe, including adding seeding, before both replicates finish
- re-scoring loss-001 or any earlier result against a new threshold
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

## Follow-up: determinism, after the measurement

Once both replicates are recorded, the training pipeline should gain explicit seeding — `torch.manual_seed`, NumPy and Python seeds, a `DataLoader` generator, and a documented decision on `cudnn_benchmark` and `torch.use_deterministic_algorithms`, since determinism and the benchmark autotuner trade against each other.

That work is deliberately sequenced after this measurement. Introducing seeding first would change the pipeline whose variation is being measured, and the resulting figure would not describe the pipeline that produced the baseline and loss-001 results. It also needs its own decision about whether existing documentation overstates reproducibility: what this repository has frozen and verified is the data and the configuration, not the training outcome.

## Owner approval block

```text
APPROVED_MEASUREMENT:
YES
APPROVED_REPLICATE_COUNT:
2
APPROVED_INTERPRETATION_RULE:
TWO_SIGMA
APPROVED_EXECUTION:
GRANTED
APPROVED_EXECUTION_DATE:
2026-08-14
APPROVED_SEEDING_CHANGE:
DEFERRED_UNTIL_AFTER_MEASUREMENT
APPROVED_LOCKED_TEST_EVALUATION:
NO
```

The owner approved the replicate count and the two-sigma interpretation rule on 2026-08-14, before either replicate ran, and granted execution the same day.

Authorization covers exactly the two replicate runs named above and their development-only OOF evaluations. It does not authorize a third replicate, any change to the training recipe including seeding, re-scoring loss-001, locked-test evaluation, or publication.
