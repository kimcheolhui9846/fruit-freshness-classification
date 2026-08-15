# Post-Holdout Measurement Floor Protocol

## Status

```text
PHASE:
9.8
ROLE:
BASELINE_REESTABLISHMENT_AND_MEASUREMENT_FRAMEWORK
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
IN_PROGRESS
TRAINING_RUN_COUNT:
1
SEED:
20260815
DETERMINISM_LEVEL:
A_STRICT
LOCKED_TEST_MODEL_ACCESS:
NO
ARTIFACT_PUBLICATION:
LOCAL_ONLY
```

This phase re-establishes the baseline under the pipeline Phase 9.7 adopted, freezes a minimum detectable effect derived from measured noise, closes H1 on a recorded calculation, and moves the research question. It advances no candidate and re-scores nothing.

## Why the baseline must be re-established

Phase 9.7 adopted `A_STRICT`: seeded, `cudnn.deterministic` true, `cudnn.benchmark` false, `torch.use_deterministic_algorithms` true. Two bounded runs were bit-exact and four checkpoint files matched under `sha256sum`.

The recorded 0.901167 came from the unseeded pipeline with the cuDNN autotuner selecting kernels. It is not a valid comparison basis for a run of the adopted pipeline, so every future comparison needs a baseline produced under `A_STRICT`. That run is required regardless of which candidate follows, which is why it is not charged against any one experiment.

## The measurement floor, and why determinism does not lower it

Phase 9.6a ran the identical baseline recipe three times on the identical frozen folds. Those three runs are the only direct evidence this project has about how much its own measurements move.

| Run | Macro F1 | Top-1 |
|---|---:|---:|
| `deep3-postholdout-research-01-baseline` | 0.901167 | 0.956598 |
| `...-rep002` | 0.912041 | 0.957936 |
| `...-rep003` | 0.901858 | 0.956016 |

```text
MACRO_F1_MEAN:
0.905022
MACRO_F1_SAMPLE_STDEV:
0.006089
MACRO_F1_TWO_SIGMA:
0.012177
TOP1_MEAN:
0.956850
TOP1_SAMPLE_STDEV:
0.000984
TOP1_TWO_SIGMA:
0.001969
```

**Seeding makes a run reproducible. It does not make the outcome less variable across seeds.** The variation those three runs exhibit came from weight initialisation, batch order, and augmentation sampling drawn from operating-system entropy. Fixing a seed pins one draw from that same distribution; it does not narrow the distribution. A second seed would land elsewhere in the same spread.

The three replicates are therefore the correct estimate of how far a result would move at a different seed, even though they were produced before seeding existed. This is the central reason Phase 9.7 could not, by itself, make small effects measurable.

### Where the noise actually comes from

Macro F1 is the unweighted mean of fourteen per-class F1 scores. Decomposing its variance across the three replicates:

| Class | Share of Macro F1 variance | Per-class stdev |
|---|---:|---:|
| `freshpotato` | 90.56% | 0.073917 |
| `rottenpotato` | 5.43% | 0.018099 |
| remaining twelve | 4.01% | 0.009580 and below |

**One class accounts for nine tenths of the metric's instability, and it is the class the research was trying to improve.** Had `freshpotato` been as stable as `rottenpotato`, the Macro F1 standard deviation would fall from 0.006089 to roughly 0.0021, and the floor below with it.

This is why Phase 9.6 could not have succeeded as designed. It measured an intervention aimed at `freshpotato` with an instrument whose noise is `freshpotato`.

## Frozen minimum detectable effect

```text
MDE_MACRO_F1:
0.012177
MDE_TOP1:
0.001969
MDE_FRESHPOTATO_F1:
0.147833
MDE_BASIS:
two sample standard deviations of the three Phase 9.6a replicates
MDE_APPLIES_TO:
single-run candidate comparisons on the development OOF metrics
```

A single-run candidate may be recorded `ADVANCED` only if its improvement on the pre-registered primary metric is at least that metric's MDE. A candidate whose improvement falls below its MDE is recorded `BELOW_RESOLUTION` — neither a success nor a failure, but a result this project cannot distinguish from the seed it happened to draw.

Three consequences are recorded now rather than argued later.

**The MDE binds regardless of determinism.** A deterministic run has no measurement noise, so the same command reproduces the number exactly. That reproducibility says nothing about whether a different seed would reproduce the *effect*.

**A paired comparison is better but not quantified.** Running a candidate and a baseline at the same seed shares initialisation and batch order, so the difference has lower variance than two independent runs. How much lower depends on a correlation this project has not measured. Until it is measured, paired comparisons are held to the same MDE, which is conservative and may be substantially too strict.

**An MDE is not a target.** Choosing interventions because they might clear 0.012177 rather than because they address a diagnosed failure would replace one bad selection rule with another.

### What Phase 9.6 got wrong

The loss-001 threshold was Macro F1 at least 0.9112, being the baseline's 0.901167 plus a reasoned margin of 0.010. That margin is **below** the 0.012177 floor measured afterwards. A criterion narrower than the noise cannot separate signal from noise, and the protocol recorded at the time that the floor was unmeasured and the margin was a reasoned rather than a measured choice.

That verdict is not revised. It was computed against a threshold frozen before the run and stands as recorded.

## H1 closed below resolution

Two effect sizes can be read from the loss-001 result, and they differ enough to matter:

| Reference | loss-001 Macro F1 | Difference |
|---|---:|---:|
| Single baseline, 0.901167, as the protocol pre-registered | 0.910182 | +0.009015 |
| Three-run mean, 0.905022, a better estimate of expected value | 0.910182 | +0.005160 |

Taking the paired difference's standard deviation at its conservative bound of `0.006089 × √2 = 0.008611` — the zero-correlation case — the number of paired seeds needed for `2 × 0.008611 / √K < d` is:

| Effect size | Paired seeds | Runs | GPU hours at 8.85 per run |
|---|---:|---:|---:|
| d = 0.009015 | 4 | 8 | 71 |
| d = 0.005160 | 12 | 24 | 212 |

The per-run figure is measured, not estimated. `deep3-postholdout-research-01-baseline-rep003` is the only replicate that ran to completion without an interruption, and it took 530.98 minutes across three folds of 176.75, 177.07, and 177.15 minutes. The other two runs were stopped and resumed, so their recorded totals cover only the resumed portion and must not be used here.

```text
H1_STATUS:
CLOSED_BELOW_RESOLUTION
H1_CLOSURE_BASIS:
71 to 212 GPU hours required to resolve the observed effect
LOSS001_VERDICT:
NOT_ADVANCED, unchanged and not re-scored
```

This is neither "H1 is exhausted", which the evidence does not support, nor "inconclusive, keep trying", which the arithmetic prices out of reach. The effect, if real, is smaller than this project can measure at a cost this project can pay. If the paired correlation is high the true cost is lower, but it is unmeasured, so the conservative bound governs.

### One observation recorded with its limits

loss-001 moved `rottenbanana` F1 by −0.008153 against a per-class two-sigma of 0.000988 — roughly eight times that class's own noise. `freshpotato` moved +0.086105 against a two-sigma of 0.147833, well inside its noise.

Read literally, the harm to a large stable class is detectable while the benefit to the target class is not. That reading is offered with its limits attached: each per-class standard deviation rests on three samples and is imprecise, and fourteen classes were examined without any multiplicity control. It is a reason to be curious, not a finding.

## Deterministic baseline run

One run of `configs/deep3_postholdout_baseline_det.toml`, which differs from `configs/deep3_postholdout_baseline.toml` only in the three `[runtime]` determinism keys and the `post_holdout` identity fields. The frozen split and CV manifests are unchanged, so the folds are the baseline's own.

| Field | Value |
|---|---|
| Split manifest | `configs/splits/deep3-postholdout-research-01.json` |
| CV manifest | `configs/splits/deep3-postholdout-research-01-baseline-cv.json` |
| Seed | 20260815 |
| Determinism level | `A_STRICT` |
| Expected duration | about 8.85 hours, the measured duration of the one uninterrupted replicate |

### Pre-registered validity check

`A_STRICT` forces deterministic kernel selection, which substitutes algorithms the autotuner would otherwise choose. Whether that substitution changes results materially is an empirical question, and it is asked here before the answer exists.

```text
VALIDITY_ENVELOPE:
0.892845 to 0.917199
ENVELOPE_BASIS:
three-replicate mean 0.905022 plus and minus two sigma 0.012177
INSIDE_ENVELOPE:
consistent with A_STRICT not having changed the pipeline materially; adopt as the deterministic baseline
OUTSIDE_ENVELOPE:
investigate before adopting; do not record the value as a baseline until the cause is understood
```

Falling inside the envelope is consistent with the substitution being immaterial. It is not proof of it: an envelope built from three samples is wide, and a real shift smaller than the envelope would pass unnoticed.

## Exploratory instability diagnostic

The three replicates each wrote `development_oof_predictions.npz`. Comparing them costs no GPU time and asks one question:

**Are the `freshpotato` images the model gets wrong the same images in every run, or different ones?**

A stable error set would mean the class has a hard core and the instability lives in a small boundary population. A shifting error set would mean the decision boundary lands somewhere different each run, which is a different problem with different remedies.

```text
DIAGNOSTIC_STATUS:
EXPLORATORY_DESCRIPTIVE
DIAGNOSTIC_MAY_ADVANCE_A_CANDIDATE:
NO
DIAGNOSTIC_MAY_SUPPORT_A_CLAIM:
NO
DIAGNOSTIC_PURPOSE:
generate hypotheses for Phase 9.9
```

This is exploratory by construction: no rule about what the result means is frozen here, because none was chosen before looking. Its output is a description and a list of candidate explanations, and any of those explanations that Phase 9.9 pursues must be pre-registered there on its own terms.

## The research question moves

The question was "can `freshpotato` F1 be raised". Phase 9.5 established the labels are sound, Phase 9.6 could not measure its intervention, and the variance decomposition above shows why.

The question becomes: **why does `freshpotato` F1 move by roughly 0.15 between runs of an identical configuration, and can it be stabilised?**

A class whose F1 ranges across a 0.15 band under a fixed recipe has not been learned to a stable decision boundary; it is landing somewhere different each run. That is a description of the failure, not a restatement of it.

```text
PHASE_9_9:
FRESHPOTATO_STABILITY
PHASE_9_9_STATUS:
REGISTERED_NOT_DESIGNED
PHASE_9_9_AUTHORIZED:
NO
```

Stabilisation is worth having on its own. It would also sharpen every later experiment: at `rottenpotato`-level stability the Macro F1 MDE falls from 0.012177 to roughly 0.0042, so effects this project currently cannot see would come into range.

Phase 9.9 is registered here and designed nowhere. Nothing in this document authorizes it.

## Boundaries

Not authorized by this document:

- executing the deterministic baseline run; execution requires the owner decision recorded in the approval block
- re-running loss-001, or any other candidate
- re-scoring loss-001, the noise floor, the Phase 9.7 outcome, or any earlier result
- designing or executing Phase 9.9
- modifying any frozen configuration, protocol threshold, seed, denominator, or split
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
APPROVED_BASELINE_REESTABLISHMENT:
YES
APPROVED_TRAINING_RUN_COUNT:
1
APPROVED_MDE_FRAMEWORK:
YES
APPROVED_H1_CLOSURE:
CLOSED_BELOW_RESOLUTION
APPROVED_EXPLORATORY_DIAGNOSTIC:
YES
APPROVED_LOSS001_RERUN:
NO
APPROVED_EXECUTION:
GRANTED
APPROVED_EXECUTION_DATE:
2026-08-16
APPROVED_LOCKED_TEST_EVALUATION:
NO
APPROVED_WEIGHT_PUBLICATION:
NO
APPROVED_PHASE_9_9_EXECUTION:
NO
```

The owner approved the phase scope, the MDE framework, the H1 closure basis, and the research question move on 2026-08-15, before the deterministic baseline ran. Execution of that run was granted separately on 2026-08-16, after the implementation was merged and with the validity envelope already frozen.

Authorization covers exactly one run of `configs/deep3_postholdout_baseline_det.toml` and its development-only OOF evaluation. It does not authorize a second run, a candidate experiment, a change to the seed or the envelope, locked-test evaluation, or publication. Windows Update was verified paused until 2026-08-22 before launch, against an expected finish around 10:12 on 2026-08-16.
