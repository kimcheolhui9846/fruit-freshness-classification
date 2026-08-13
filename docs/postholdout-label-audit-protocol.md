# Post-Holdout Label Quality Audit Protocol

## Status

```text
PHASE:
9.5
EXPERIMENT_ID:
deep3-postholdout-research-01-label-audit
PARENT_EXPERIMENT_ID:
deep3-postholdout-research-01-baseline
ROLE:
DEVELOPMENT_LABEL_QUALITY_AUDIT
AUDIT_PROTOCOL_STATUS:
FROZEN
AUDIT_EXECUTION_STATUS:
COMPLETED
AUDIT_OUTCOME:
DEFECT_NOT_CONFIRMED
MODEL_TRAINING:
NO
MODEL_INFERENCE:
NO
LOCKED_TEST_INSPECTION:
NO
DATASET_DOWNLOAD:
NO
IMAGE_PUBLICATION:
NO
```

This document freezes the audit method before any image is reviewed. Freezing first is the point: judgment criteria, review-set composition, blinding, and the decision rule are all fixed here so that none of them can be adjusted after seeing results.

Nothing in this document authorizes training, model inference, locked-test inspection, or publication of image content.

## Why this phase exists

The pre-registered hypothesis order in [post-holdout-research-plan.md](post-holdout-research-plan.md) put H1 loss / class imbalance next. The completed Phase 9.4 baseline produced evidence that the binding constraint is not class frequency, so this phase is inserted before H1. The reordering and its evidence are recorded here rather than applied silently.

Aggregate development OOF Macro F1 is 0.9012. Removing `freshpotato` alone raises the 13-class figure to 0.9533, so one class accounts for roughly 5.2 points.

Class frequency does not explain the failure:

| Class | Support | F1 |
|---|---:|---:|
| `freshpotato` | 347 | 0.3682 |
| `rottencucumber` | 435 | 0.7932 |
| `freshcucumber` | 486 | 0.9358 |
| `rottenpotato` | 514 | 0.7741 |
| `rottencapsicum` | 570 | 0.9965 |
| `freshcapsicum` | 624 | 0.9889 |
| `rottenapples` | 2714 | 0.9960 |

The development imbalance ratio is 7.8:1, which is moderate, and the support-to-F1 correlation is 0.500. `rottenpotato` (514) and `rottencapsicum` (570) have nearly the same support and F1 scores of 0.7741 and 0.9965. Failures group by produce type, not by frequency: decomposing each prediction into produce identity and freshness state gives 0.8420 and 0.7364 for potato against 1.0000 and 1.0000 for capsicum.

Three independent signals point to a data property rather than a training outcome:

1. **Inverted confidence.** On `freshpotato` the model averages 0.608 confidence when correct and 0.745 when wrong. The global pattern is the opposite, 0.963 against 0.676. Of `freshpotato` errors, 27 percent exceed 0.9 confidence.
2. **One-way confusion.** `freshpotato` is predicted `rottenpotato` 164 times; `rottenpotato` is predicted `freshpotato` 5 times. Mutual visual similarity would produce a roughly symmetric exchange.
3. **Cross-fold reproduction.** Per-fold `freshpotato` recall is 0.216, 0.383, and 0.224, and the reverse direction is 0.000, 0.023, and 0.006. Three models trained on different data and validated on disjoint sets reproduce the same asymmetry.

A further 73 of 347 `freshpotato` examples are predicted `freshbanana` or `rottenbanana`, which is not a confusion a working potato class should produce.

This evidence is consistent with label noise and inconsistent with pure imbalance. It does not prove label noise. Confirming or refuting it requires looking at the images, which is what this phase does.

## Frozen judgment criteria

These are operational definitions for this audit. A reviewer assigns exactly one category per image from the image alone.

**`FRESH`** — firm, intact skin with uniform coloring; no mold; no soft, sunken, or weeping areas; surface blemishes limited to soil, superficial scuffing, or shallow eyes without shoots.

**`ROTTEN`** — any of: visible mold growth; dark sunken lesions; wet, weeping, or collapsed tissue; extensive shriveling or wrinkling; widespread black or brown decay on the surface; pronounced sprouting accompanied by shriveling.

**`NOT_A_POTATO`** — the dominant subject is a different produce type, the frame contains a mixture with no dominant potato, or the content is not produce.

**`UNDECIDABLE`** — potato is the subject but the category cannot be determined: image quality prevents assessment, the visible surface is too occluded, or the indicators conflict. Green skin alone, isolated shallow sprouting without shriveling, and soil that cannot be distinguished from mold all resolve to `UNDECIDABLE` rather than being forced into `FRESH` or `ROTTEN`.

A reviewer who is unsure chooses `UNDECIDABLE`. Forcing borderline images into a decision would manufacture the disagreement this audit is meant to measure.

## Review set

Development pool only. Total 497 images.

| Group | Source class | Count | Purpose |
|---|---|---:|---|
| Subject | `freshpotato` | 347 | All development examples of the suspect class |
| Control | `rottenpotato` | 150 | Baseline reviewer error rate |

The control group is what makes the subject error rate interpretable. A reviewer told only that a class is suspect will find defects in it; the control measures that tendency on a class the model already learns well (`rottenpotato` recall 0.897, confidence 0.907 when correct).

Images are identified by their zero-based index into the reconstructed historical canonical training pool, the same index identity used by `configs/splits/deep3-postholdout-research-01.json`.

Construction is deterministic and fixed here so the control sample cannot be reselected later:

```text
CONTROL_SAMPLING:
uniform without replacement from development rottenpotato indices
CONTROL_SAMPLE_SEED:
20260813
PRESENTATION_ORDER_SEED:
20260813
SUBJECT_COUNT:
347
CONTROL_COUNT:
150
REVIEW_SET_COUNT:
497
```

Materializing the review set is the first execution step under this frozen specification. It records the selected indices and their LF-normalized SHA-256 so the set is verifiable after the fact. The source archive is already present locally; no download occurs.

## Blinding

Each reviewer sees the image and nothing else. The original class label, the baseline model's prediction, its confidence, the group membership, and the other reviewer's judgments are all withheld until every judgment is recorded.

Presentation order is a single shuffle of the combined 497 under `PRESENTATION_ORDER_SEED`, so subject and control images interleave and neither group is identifiable by position.

## Independent dual review

Two reviewers judge the full set independently: the repository owner and the assistant. Neither sees the other's judgments until both are complete.

A single reviewer would produce a finding that reduces to one person's opinion. Two independent passes yield inter-rater agreement, and the disagreements are themselves evidence: images that two reviewers categorize differently are demonstrably ambiguous, which bears directly on whether the class is mislabeled or merely hard.

## Outputs

After unblinding:

- per-reviewer category counts for subject and control groups
- `freshpotato` label error rate, meaning the share judged `ROTTEN` or `NOT_A_POTATO`
- `rottenpotato` control error rate, meaning the share judged `FRESH` or `NOT_A_POTATO`
- inter-rater agreement over all 497, reported as raw agreement and Cohen's kappa
- the disagreement list, with indices
- agreement between reviewer judgments and the baseline model's predictions on the subject group

The judgment record is metadata keyed by index. It contains no image content and is tracked. Image files are not published.

## Frozen decision rule

Error rates are computed per reviewer, over that reviewer's full group, with `UNDECIDABLE` counted in the denominator and not as an error:

```text
SUBJECT_ERROR_RATE:
count(ROTTEN or NOT_A_POTATO) / 347
CONTROL_ERROR_RATE:
count(FRESH or NOT_A_POTATO) / 150
MATERIAL_DIFFERENCE_THRESHOLD:
15 percentage points
RULE_EVALUATION:
each reviewer independently
```

Keeping `UNDECIDABLE` in the denominator is deliberate. Dropping it would let a reviewer inflate the subject error rate by abstaining on the images they found clean.

**Defect confirmed** — both reviewers independently produce a subject error rate at least 15 percentage points above their own control error rate. Phase 9.6 becomes the remediation decision among relabel, exclude, or retain, which requires its own owner authorization. The `UNDECIDABLE` share is reported alongside, since a large ambiguous fraction argues for exclusion over relabeling.

**Defect not confirmed** — neither reviewer clears the threshold. The labels are sound and potato freshness is genuinely hard to discriminate. The pre-registered order resumes and Phase 9.6 is H1, loss and class imbalance, as originally planned.

**Split outcome** — exactly one reviewer clears the threshold. No phase is selected automatically. The disagreement list and both rate pairs go to the owner, who decides whether to extend the audit or proceed to H1. Naming this case in advance keeps it from being resolved by whichever reading is convenient at the time.

Requiring each reviewer to clear the threshold against their own control, rather than pooling the judgments, is what makes the control do its work: a reviewer who is systematically harsh raises both of their own rates, so the comparison stays internal to that reviewer.

Either outcome determines the next phase. Neither leaves room to select a favorable reading afterwards.

## Locked test

The locked test contains 86 `freshpotato` examples. That count comes from the split manifest, which has always been public; no image is inspected to obtain it.

Those 86 images are not reviewed in this phase. The frozen protocol says: inspecting them during development would violate the Phase 9.2 boundary, and their labels carry the same suspected defect, so a final claim scored against them would understate potato performance if the defect is real.

The resolution defers rather than chooses:

```text
LOCKED_TEST_LABEL_AUDIT:
DEFERRED_TO_FINAL_EVALUATION
LOCKED_TEST_AUDIT_CRITERIA:
THIS_DOCUMENT_UNCHANGED
```

The criteria frozen above are applied to the 86 locked-test images only at final-evaluation time, under the separate owner authorization that final evaluation already requires. Because the criteria are fixed now, from development evidence alone, they cannot later be shaped to flatter the test result.

## Boundaries

Not authorized by this document:

- training any model, or running inference with any model
- inspecting, evaluating, or relabeling the 4,298-example locked test
- re-evaluating the 5,372-example historical canonical holdout
- relabeling, excluding, or otherwise modifying any development label; this phase produces a finding, and remediation is a separate decision
- downloading or acquiring any dataset
- publishing images, image derivatives, checkpoints, weights, or any binary artifact
- creating a release or tag

```text
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
LOCKED_TEST_MODEL_ACCESS:
NO
CANONICAL_HOLDOUT_MODEL_ACCESS:
NO
POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:
0
CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:
0
AUDIT_ARTIFACT_PUBLICATION:
JUDGMENT_RECORD_ONLY
```

## Owner approval block

```text
APPROVED_AUDIT_SCOPE:
DEVELOPMENT_FRESHPOTATO_AND_ROTTENPOTATO_CONTROL
APPROVED_REVIEW_SET_COUNT:
497
APPROVED_REVIEWERS:
OWNER_AND_ASSISTANT_INDEPENDENT
APPROVED_LOCKED_TEST_INSPECTION:
NO
APPROVED_RELABELING:
NO
APPROVED_IMAGE_PUBLICATION:
NO
```

The owner approved the audit scope and the dual-review design on 2026-08-13, after the Phase 9.4 diagnostic was presented. That approval covers freezing this protocol and executing the review. It does not authorize remediation, locked-test inspection, or any publication; each requires a further explicit decision.

## Recorded Execution — 2026-08-13

The review set was materialized on 2026-08-13 with `presentation_indices_sha256`
`db5b29e766ec77555c1600f891470fa92c50ecb532180875f247d34255153baf`, recorded
before any image was opened and verified again at unblinding. Group sizes
resolved to 347 subject and 150 control.

### Outcome

Neither reviewer's subject error rate reaches 15 percentage points above their
own control rate, so the frozen decision rule returns `DEFECT_NOT_CONFIRMED`
and Phase 9.6 is H1, loss and class imbalance, as originally pre-registered.

| Reviewer | Scope | Subject error | Control error | Difference | Clears 15 pt |
|---|---|---:|---:|---:|---|
| Assistant | all 497 | 0.0259 | 0.0800 | −0.0541 | no |
| Owner | 100-image subsample | 0.1324 | 0.1250 | +0.0074 | no |

The assistant's subject error rate is *lower* than its control rate: images
labeled `freshpotato` were judged fresh 97.4 percent of the time. Over the
100-image overlap both reviewers overwhelmingly agreed the subject images look
fresh — 65 of 68 and 55 of 68 respectively.

Inter-rater agreement over the overlap is 0.80 raw, Cohen's kappa 0.6259. The
largest divergence is category use rather than freshness: the owner assigned
`NOT_A_POTATO` 9 times and `UNDECIDABLE` 8 times against the assistant's 0 and
3, so the assistant resolves doubtful produce identity toward "potato" where the
owner does not. That difference does not move the outcome — the subject and
control rates are near-equal under either reviewer.

### What this refutes

The Phase 9.4 diagnostic proposed label noise in `freshpotato` on three
signals: the model was more confident when wrong (0.745) than right (0.608), the
confusion with `rottenpotato` ran 164 to 5 in one direction, and all three folds
reproduced it. Every one of those observations holds. The inference drawn from
them does not: the labels are sound, and the model is simply wrong about images
two humans read as fresh.

That returns the binding constraint to the pre-registered H1. `freshpotato` is
the smallest class at 347 examples and is visually adjacent to `rottenpotato` at
514; a minority class collapsing into a visually similar majority class is a
characteristic imbalance failure, and the argument that a 7.8:1 ratio is too
mild did not account for that adjacency. The audit was built to be able to
falsify its own motivating hypothesis, and it did.

### Recorded deviations

1. The owner reviewed a seeded random 100-image subsample (seed 20260814,
   `positions_sha256` `dd476643391ba6bf129d0519a889d56b9a9f0bd1c8b57d0de8c0b66648e5e6c9`)
   rather than all 497. Their control group is therefore 32 examples, so their
   control rate carries a wide interval.
2. The assistant's judgments were visible in the session transcript before the
   owner judged. The two reviews are not fully independent, and the reported
   agreement is an upper bound on what independent review would have produced.
3. `scripts.analyze_label_audit` was not run. It requires two complete 497-row
   judgment files by design, and that guard was not weakened to accommodate a
   partial second review; the reported figures were computed separately under
   the same frozen definitions.

Because of deviation 1 the two-reviewer standard written into this protocol was
not met. The outcome is reported as an evidenced finding rather than as a
mechanically satisfied decision, and selecting Phase 9.6 was the owner's call.
No label was modified, no locked-test image was inspected, and no model was run.
