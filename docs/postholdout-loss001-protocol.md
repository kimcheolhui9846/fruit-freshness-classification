# Post-Holdout H1 Loss Experiment Protocol — loss-001

## Status

```text
PHASE:
9.6
EXPERIMENT_ID:
deep3-postholdout-research-01-loss-001
PARENT_EXPERIMENT_ID:
deep3-postholdout-research-01-baseline
ROLE:
H1_LOSS_AND_CLASS_IMBALANCE
HYPOTHESIS_FAMILY:
H1
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
COMPLETED
OUTCOME:
NOT_ADVANCED
LOCKED_TEST_MODEL_ACCESS:
NO
CANONICAL_HOLDOUT_MODEL_ACCESS:
NO
ARTIFACT_PUBLICATION:
LOCAL_ONLY
```

This document freezes the experiment before it runs. The single changed factor, its value, the comparison basis, the acceptance threshold, and what happens on failure are all fixed here so none of them can be chosen after seeing the result.

Freezing this document does not authorize training. Execution requires a separate explicit owner decision, as Phase 9.4 did for the baseline.

## Why this experiment, and what it is not

The Phase 9.5 audit returned `DEFECT_NOT_CONFIRMED`: two reviewers read the `freshpotato` images as fresh, so the labels are sound and the model is wrong about them. That returned the binding constraint to H1, where the pre-registered plan had it.

H1 here is **not** "introduce class balancing". The baseline already trains with class-balanced focal loss. What this experiment changes is how hard that existing mechanism pushes.

Two facts about the baseline point in opposite directions, and this run is designed to settle which one governs.

`freshpotato` already carries the largest weight of all fourteen classes, and it still collapses to recall 0.274:

| Class | Development n | Alpha at β=0.999 | Alpha at β=0.9999 | Inverse frequency |
|---|---:|---:|---:|---:|
| `freshpotato` | 347 | 1.841 | 2.244 | 2.295 |
| `rottenpotato` | 514 | 1.343 | 1.528 | 1.550 |
| `rottenapples` | 2714 | 0.578 | 0.322 | 0.293 |

But the correction is only partly applied. At β=0.999 the `freshpotato`-to-`rottenapples` weight ratio is 3.18 against a frequency ratio of 7.82 — roughly 40 percent of full inverse-frequency reweighting. At β=0.9999 that ratio becomes 6.97, essentially inverse frequency.

So this run asks one question: **with the reweighting effectively taken to its limit, does the minority class recover?** A negative answer is informative, which is why the failure branch is fixed below rather than left open.

## The single changed factor

```text
CHANGED_PARAMETER:
loss.class_balanced_beta
BASELINE_VALUE:
0.999
EXPERIMENT_VALUE:
0.9999
```

Nothing else moves. `focal_gamma` stays 2.0, `label_smoothing` stays 0.01, `use_ce_label_smoothing` stays true, mixup stays 0.8 / 0.5, batch size stays 64, epochs stay 120 with 20 fine-tuning epochs, and the optimizer, scheduler, EMA, and cuDNN policy are unchanged.

| Field | Value |
|---|---|
| Config | `configs/deep3_postholdout_loss001.toml` |
| Config LF-normalized SHA-256 | `6ced28e530a4bfef44b0bb22edc24641c68404d552ddc3bfd4c2287888b247ec` |
| Baseline config LF-normalized SHA-256 | `7cb01e8fe251fd1648ba3a53601e471d9b3693e5d50090f7e7d9c9c5586b11c7` |

The only permitted differences from the baseline config are `loss.class_balanced_beta`, `post_holdout.experiment_id`, `post_holdout.parent_experiment_id`, and `post_holdout.artifact_namespace`.

**That list is enforced in code, not by discipline.** "One factor at a time" is the plan's central rule, and a rule that depends on nobody mistyping a TOML key is not a rule.

Enforcement comes in two places, and only the first exists today:

- `tests/repository/test_loss001_protocol_contract.py` compares the two configs offline and fails if the differing key set is anything other than the four above. This is active now and runs in CI.
- A runtime validator, in the shape of `validate_postholdout_baseline_config`, must abort before dataset preparation or model construction on any difference outside the list. **This does not exist yet and is a prerequisite for execution**, not an optional extra: the offline test protects the committed files, while the runtime check protects the run from a config edited after CI passed.

## Comparison basis

The comparison is only meaningful against identical folds, so this run reuses the baseline's frozen cross-validation identity rather than generating its own.

| Field | Value |
|---|---|
| Split manifest | `configs/splits/deep3-postholdout-research-01.json` |
| Split manifest LF-normalized SHA-256 | `cd7182c18d81cfac877fb2dab8573695b6bdd8116aeb23b19c3e4457e36be169` |
| CV manifest | `configs/splits/deep3-postholdout-research-01-baseline-cv.json` |
| CV manifest LF-normalized SHA-256 | `494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f` |
| Cross-validation | 3-fold stratified, random state 42 |
| Development pool | 17,188 examples |
| Locked test | 4,298 examples, `FROZEN_UNOBSERVED_BY_MODEL` |

Evaluation follows the same development-only out-of-fold route the baseline used. Each fold's best checkpoint predicts that fold's held-out indices, and the per-fold outputs assemble into one prediction set covering all 17,188 development examples exactly once.

## Baseline figures this run is measured against

```text
BASELINE_DEVELOPMENT_OOF_MACRO_F1:
0.9012
BASELINE_DEVELOPMENT_OOF_BALANCED_ACCURACY:
0.9007
BASELINE_DEVELOPMENT_OOF_TOP1:
0.9566
BASELINE_FRESHPOTATO_F1:
0.3682
BASELINE_FRESHPOTATO_RECALL:
0.2738
```

## Frozen decision rule

```text
PRIMARY_METRIC:
aggregate development OOF Macro F1
ADVANCE_THRESHOLD:
Macro F1 >= 0.9112
TOP1_GUARDRAIL:
Top-1 >= 0.9466
RULE:
ADVANCE only if both hold; otherwise NOT_ADVANCED
```

The thresholds are the pre-registered `+0.010` improvement and `-0.010` guardrail applied to the baseline figures above.

For scale: `freshpotato` F1 rising from 0.368 to about 0.51 moves the fourteen-class Macro F1 by roughly +0.010, and rising to 0.75 moves it by roughly +0.027. A change below +0.005 would not be separable from seed variation.

**Known limitation.** This project has no repeat-seed measurement, so the run-to-run noise floor is unmeasured. Establishing it would require rerunning the baseline under a different training seed, another nine to twelve GPU hours, which was considered and declined. The +0.010 threshold is therefore a reasoned choice, not a measured one, and any result within roughly ±0.005 of it should be read with that in mind. This is recorded here rather than discovered later.

## Failure branch, fixed in advance

**If `ADVANCE`** — H1 is supported. The next phase is the owner's decision, informed by the result.

**If `NOT_ADVANCED`** — H1 is recorded as **exhausted**, and Phase 9.7 is H2 augmentation. No further loss variant is run.

Fixing this now is the point. β=0.9999 puts the weight ratio at 6.97 against an inverse-frequency ceiling of 7.82, so a null result here is close to the strongest form of "reweighting does not fix this class". Leaving the branch open would invite trying γ, then label smoothing, then a combination, exploring the H1 family without pre-registration until something clears the bar by chance. The research plan's own stop condition — "stop when a primary family is exhausted" — is what this encodes.

## Reported as diagnostics only

These are reported and never feed the decision rule:

- `freshpotato` F1, precision, and recall against the baseline's 0.3682 and 0.2738
- per-class precision, recall, and F1 for all fourteen classes
- the aggregate confusion matrix, and the `freshpotato` row in particular
- per-fold Macro F1, reported separately from the aggregate, which is a different quantity
- balanced accuracy, Top-2, Top-3
- the confidence distribution on `freshpotato`, to see whether the inverted pattern from Phase 9.4 persists

One outcome deserves naming ahead of time: **`freshpotato` may improve while Macro F1 does not**, if the added weight costs other classes more than it gains. That is a substantive finding about the reweighting trade-off, not a partial success, and it does not clear the threshold.

## Boundaries

Not authorized by this document:

- executing this run; training requires a separate explicit owner decision, and the runtime config validator described above must exist and pass first
- evaluating, inspecting, or relabeling the 4,298-example locked test
- re-evaluating the 5,372-example historical canonical holdout
- any second run, any additional candidate, or any change to the frozen parameter after a result is seen
- tuning any other hyperparameter, loss term, augmentation, sampler, or architecture
- publishing weights, checkpoints, dataset copies, raw logits, raw predictions, Actions artifacts, Release assets, Releases, or tags

```text
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:
0
CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:
0
BINARY_PUBLICATION:
NO
RELEASE_OR_TAG_CREATION:
NO
```

A final claim of any kind still requires the untouched locked test under its own separate authorization. This experiment measures development data only.

## Owner approval block

```text
APPROVED_PROTOCOL_FREEZE:
YES
APPROVED_EXECUTION:
GRANTED
APPROVED_EXECUTION_DATE:
2026-08-14
APPROVED_CANDIDATE_COUNT:
1
APPROVED_LOCKED_TEST_EVALUATION:
NO
APPROVED_WEIGHT_PUBLICATION:
NO
APPROVED_RELEASE_CREATION:
NO
```

The owner approved the single changed factor, the acceptance threshold, the candidate count, and the failure branch on 2026-08-14, before any training.

The owner then granted execution on 2026-08-14, after the runtime single-factor validator this protocol named as a prerequisite was implemented and exercised: a config with a second changed factor, one with an unfrozen value for the allowed factor, and one naming an unregistered parent were each rejected. Authorization covers exactly one training run against `weights/deep3-postholdout-research-01-loss-001` and the development-only OOF evaluation that follows it, plus applying the frozen decision rule to the result. It does not authorize a second run, locked-test evaluation, canonical-holdout re-evaluation, weight or dataset publication, release creation, or any change to the frozen parameter or thresholds.


## Recorded Execution — 2026-08-14

The run started 02:55 and completed 11:44 local under repository commit `00bbdec`, 527.9 minutes across three folds with no interruption and no error. The single-factor validation printed before dataset preparation and reported exactly the four registered differences. Final state was `status = COMPLETED`, `completed_epoch = 120`, with all seven expected artifacts present.

### Outcome

`scripts.apply_loss001_decision` computed the verdict from the two OOF metric files:

```text
OUTCOME:
NOT_ADVANCED
MACRO_F1:
0.9102 (delta +0.0090, threshold 0.9112)
TOP1:
0.9561 (delta -0.0005, guardrail 0.9466)
```

The Top-1 guardrail passed. Macro F1 fell short by 0.0010.

| Aggregate metric | Baseline | loss-001 | Delta |
|---|---:|---:|---:|
| Macro F1 | 0.9012 | 0.9102 | +0.0090 |
| Balanced accuracy | 0.9007 | 0.9047 | +0.0040 |
| Top-1 | 0.9566 | 0.9561 | −0.0005 |

Per-fold Macro F1 is a separate quantity and is reported separately: 0.8907 to 0.9003, 0.9098 to 0.9076, and 0.9023 to 0.9221. Fold 2 declined while fold 3 rose sharply; the spread across folds, 0.022, exceeds the aggregate improvement of 0.009.

### The intervention worked where it was aimed

| Class | Baseline F1 | loss-001 F1 | Delta | Recall delta |
|---|---:|---:|---:|---:|
| `freshpotato` | 0.3682 | 0.5140 | **+0.1457** | +0.1239 |
| `rottenpotato` | 0.7741 | 0.7883 | +0.0142 | −0.0564 |
| `rottencucumber` | 0.7932 | 0.8057 | +0.0125 | +0.0276 |
| `freshcucumber` | 0.9358 | 0.9236 | −0.0122 | −0.0247 |
| Nine well-learned classes | | | −0.009 to +0.001 | |

Reweighting did what it was supposed to: `freshpotato` F1 rose 0.146 and its recall 0.124. It also cost the classes that were already working, and the sum of those small losses absorbed much of the gain. `rottenpotato` gained F1 while losing 0.056 of recall — the reciprocal of predictions moving toward `freshpotato`.

This is the partial realisation of the outcome this protocol named in advance: `freshpotato` improving without Macro F1 clearing the bar.

### Why the failure branch was not applied immediately

This protocol recorded that the run-to-run noise floor was unmeasured and that results within roughly ±0.005 of the threshold should be read with that in mind. The observed +0.0090 sits inside that band, so whether this is a real shortfall or ordinary variation cannot be determined from this run alone.

The threshold was not reinterpreted. The recorded verdict is `NOT_ADVANCED`, computed against the value frozen before the run. What was deferred is only the *next-phase selection* that would follow from it, pending the measurement frozen in [postholdout-noise-floor-protocol.md](postholdout-noise-floor-protocol.md). That measurement can determine whether "H1 exhausted" is a sound conclusion; it cannot re-score this result.

Designing it surfaced a larger finding: the training pipeline sets no random seed anywhere, so every run is already an independent draw and the repository's training results are not reproducible. Details and the deferred remediation are in the noise floor protocol.
