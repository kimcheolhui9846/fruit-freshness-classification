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
NOT_YET_RUN
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
