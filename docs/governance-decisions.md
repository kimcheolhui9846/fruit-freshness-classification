# Governance decisions

This document records the Phase 6.3 audit history and the owner-approved governance decisions applied in Phase 6.4. It is informational only and is not legal advice.

## Resolved software license

MIT was explicitly selected during Phase 6.4. The repository did not contain a software license before this explicit decision; this is a new licensing decision, not a restoration claim.

| Item | Approved value |
|---|---|
| Software license | MIT |
| SPDX identifier | MIT |
| Copyright year | 2025 |
| Copyright holder | 김철희 |
| Canonical file | [`LICENSE`](../LICENSE) |

The repository software and project-authored documentation are licensed under the MIT License. The canonical file uses the approved copyright line `Copyright (c) 2025 김철희` and unmodified MIT terms. This software license does not determine the terms of the external dataset or any future trained-weight distribution.

## Dataset and trained-weight governance

| Item | Current boundary |
|---|---|
| Dataset | [`Densu341/Fresh-rotten-fruit`](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit) |
| Dataset owner | `Densu341` |
| Public page metadata | Labeled `openrail` in the Hugging Face page/search metadata reviewed during the audit |
| Dataset-card content | The surfaced dataset-card README was empty |
| Dataset terms | Governed by the terms supplied by the original external source |
| Dataset redistribution | Dataset contents are not redistributed through this repository |
| Weight distribution | Trained weights are not currently distributed and require a separate review before publication |

The MIT License applies to repository software and project-authored documentation only. The repository does not claim that the external dataset metadata is compatible with MIT, permission to redistribute external images, or automatic permission to distribute future trained weights. See the [dataset documentation](dataset.md) for dataset identity, source revision, and project data-handling boundaries.

## Resolved repository-only citation

Repository-only citation was explicitly selected during Phase 6.4. [`CITATION.cff`](../CITATION.cff) contains truthful software metadata without claiming a paper, DOI, release, or artifact distribution.

| Citation input | Approved value |
|---|---|
| Project title | `Fruit Freshness Classification` |
| Citation policy | Repository-only citation |
| Citation author | Choelhui Kim |
| Given names | `Choelhui` |
| Family names | `Kim` |
| Author email | Omitted by owner decision |
| Author affiliation | Omitted |
| Author ORCID | Omitted |
| Additional authors | None |
| Repository URL | `https://github.com/kimcheolhui9846/fruit-freshness-classification` |
| Citation license | MIT |
| Citation version and release date | Deferred to Phase 6.5 |
| Paper and DOI | Unavailable |

No version, release date, DOI, paper citation, affiliation, ORCID, additional author, or email field was added to `CITATION.cff`. A future versioned release may update citation metadata only with separate Phase 6.5 authorization.

## Historical Phase 6.4 non-actions

No Git tag, GitHub Release, repository setting, branch protection, ruleset, DOI, dataset copy, or trained weight was created in Phase 6.4. This is a historical record for that Phase; later owner-approved Phases established the current release, ruleset, and repository-metadata state below.

## Remaining release and repository decisions

| Decision | Current state | Owner action needed |
|---|---|---|
| Version tag | `v0.1.0` engineering-milestone tag published | A new tag requires separate explicit approval |
| GitHub Release | `v0.1.0` prerelease published with zero assets | A new Release requires separate explicit approval |
| Release date | `2026-08-02` for the engineering milestone | A new release date requires a new release action |
| Dataset redistribution | Pending source-terms review | Confirm attribution and redistribution boundary |
| Trained-weight distribution | Pending separate review | Confirm applicable terms before publication |
| Branch protection | `Protect main` and `Protect v0.1.0` rulesets active | Change only through a separate governance Phase |
| Repository metadata | Approved description and topics applied in Phase 7.3 | Change only through a separate metadata Phase |
| Canonical training and result documentation | One local-only run and one internal fixed-holdout evaluation documented | Decide artifact retention or publication only through a new explicit Phase |
| Phase 8.5 artifact publication | [Documentation-only publication decision](artifact-publication-decision.md); binary publication blocked | Resolve only through the explicit Phase 8.6 owner gate |

No Git tag, GitHub Release, repository setting, branch-protection rule, dataset copy, trained weight, checkpoint, or binary artifact was created in Phase 8.5.
## Phase 8.6 — Canonical Run Closure

The canonical reference run is CLOSED_REFERENCE; documentation remains public while all canonical binaries remain local-only. Dataset license clearance remains NOT_CONFIRMED. Weight and checkpoint publication is not authorized, and no artifact deletion, relocation, conversion, or packaging is authorized. No Release or tag was created.

Future binary publication requires a separate explicit owner-approved governance Phase. Future post-holdout research requires a new experiment identity. The already-observed canonical holdout must not be treated as untouched test evidence for tuned successors.

## Phase 9.1 — Post-Holdout Research Planning

Phase 8 remains closed and its holdout is historical evidence only. Phase 9 is explicitly post-holdout research: no training, new split, or metric result was authorized in Phase 9.1. Phase 9.2 requires explicit owner approval and a new experiment identity; the observed canonical holdout may not be treated as untouched evidence for tuned successors.

## Phase 9.2 — Post-Holdout Protocol Freeze

Phase 9.1 is complete. The owner approved `DEV_PLUS_LOCKED_TEST` with a pre-registered split seed of `20260810`. The Phase 9 source is restricted to the 21,486-example historical canonical training pool; the 5,372-example historical canonical holdout is excluded and remains `HISTORICAL_EVIDENCE_ONLY`.

The new locked test is frozen as `FROZEN_UNOBSERVED_BY_MODEL`. No model training, model evaluation, canonical-holdout evaluation, new locked-test model evaluation, sample-level image review, external dataset acquisition, checkpoint creation, binary publication, Release creation, or tag creation occurred in Phase 9.2. Future model development requires a separate Phase authorization and may use only the post-holdout development pool.
## Phase 9.3 — Development Baseline Authorization

The owner approved one canonical-recipe reproduction: `deep3-postholdout-research-01-baseline`. It is limited to the 17,188-example frozen development pool, 3-fold stratified CV with random state 42, batch size 64, and the unmodified canonical recipe. The 4,298-example locked test is `FROZEN_UNOBSERVED_BY_MODEL`; the 5,372-example historical canonical holdout remains historical evidence only.

```text
OWNER_PHASE_9_3_APPROVAL:
APPROVED
BASELINE_ARTIFACT_PUBLICATION:
LOCAL_ONLY
BASELINE_EXECUTION_STATUS:
COMPLETED
PHASE_9_4:
BASELINE_EXECUTED
```

The deterministic development-CV identity is materialized once as a tracked index record; it is not a dataset or prediction artifact. No hyperparameter tuning, loss/augmentation/sampler/optimization/architecture experiment, external data acquisition, binary publication, Release, or tag is approved by this decision.

## Phase 9.4 — Baseline Execution

Phase 9.4 began as runbook preparation only. [postholdout-baseline-runbook.md](postholdout-baseline-runbook.md) recorded the frozen inputs, the fresh and resume commands, the development-only OOF evaluation command, preflight and stop conditions, a resource expectation, and an initially unresolved owner approval block.

The owner resolved that block on 2026-08-12 after the recorded preflight, authorizing exactly one baseline training run and the development-only OOF evaluation that follows it. The run executed 2026-08-12 to 2026-08-13 and completed. Training and evaluation are therefore `YES` for this phase; every other boundary stayed closed.

```text
PHASE_9_4_SCOPE:
BASELINE_EXECUTION
BASELINE_EXECUTION_STATUS:
COMPLETED
MODEL_TRAINING:
YES
MODEL_EVALUATION:
DEVELOPMENT_OOF_ONLY
LOCKED_TEST_EVALUATION:
NO
CANONICAL_HOLDOUT_REEVALUATION:
NO
POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:
0
CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:
0
BINARY_PUBLICATION:
NO
RELEASE_OR_TAG_CREATION:
NO
PHASE_9_5:
LABEL_AUDIT_PROTOCOL_FROZEN
```

The run was interrupted once at 2026-08-13 05:34:53 during fold 3 epoch 30 by an automatic Windows Update restart, not by a training fault or a stop condition. The approved epoch-boundary resume policy recovered it from completed epoch 29 without repeating folds 1 and 2 and without altering the recipe. The integrity block written by the evaluator records zero locked-test and zero canonical-holdout forward passes.

A second baseline run, locked-test evaluation, canonical-holdout re-evaluation, weight or dataset publication, and Release or tag creation each still require a further explicit owner decision.

Phase numbering is resolved as follows: Phase 9.4 covers baseline execution. An earlier external handoff draft numbered the loss experiment as 9.4; the repository documents and their contract tests are the authoritative numbering. The loss/class-imbalance experiment was Phase 9.5 until the Phase 9.5 decision below reordered it to Phase 9.6.

## Phase 9.5 — Development Label Quality Audit

The owner approved a development label quality audit on 2026-08-13 and deferred the loss/class-imbalance experiment to Phase 9.6. The audit method is frozen in [postholdout-label-audit-protocol.md](postholdout-label-audit-protocol.md) before any image is reviewed.

```text
PHASE_9_5_SCOPE:
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
APPROVED_RELABELING:
NO
LABELS_MODIFIED:
0
IMAGE_PUBLICATION:
NO
PHASE_9_6:
H1_LOSS_AND_CLASS_IMBALANCE
```

This reorders the pre-registered hypothesis queue, so the evidence for doing so is recorded rather than applied silently. The Phase 9.4 baseline showed that the dominant error is not explained by class frequency: the development imbalance ratio is 7.8:1, the support-to-F1 correlation is 0.500, and `rottenpotato` at 514 examples scores 0.7741 while `rottencapsicum` at 570 scores 0.9965. On `freshpotato` the model is more confident when wrong (0.745) than when right (0.608), the confusion with `rottenpotato` runs 164 to 5 in one direction, and all three folds reproduce the asymmetry. A frequency-based reweighting experiment would target something other than the binding constraint.

The audit is falsifiable by construction. A `rottenpotato` control group measures reviewer error on a class the model already learns well, two reviewers judge blind and independently, and a decision rule fixed in advance selects Phase 9.6 from the outcome. Confirming a defect leads to a remediation decision, which is not authorized here; comparable error rates return to the pre-registered H1 loss experiment.

The locked test is not inspected. Its 86 `freshpotato` images carry the same suspected defect, so the frozen criteria are applied to them only at final-evaluation time under the authorization that final evaluation already requires. Fixing the criteria now, from development evidence alone, prevents them from being shaped to flatter the final result.

### Recorded integration deviation — PR #5

PR #5 was integrated into `main` with GitHub "Rebase and merge" rather than the fast-forward-only integration this project prefers. The rebase rewrote commit identifiers: PR head `48a61eb63d57604351088ea72bbe69d22fe50a39` became merge commit `15eb552e7ae1f698e99c5d3bac3e9516180f7053`.

Verified consequences: the resulting tree hash `4275acce956419989dfcb6a3bb1158538eafe9d1` is byte-identical to the PR head tree, linear history is preserved, the `Protect main` ruleset accepted the integration, final-main CI passed on both required jobs with zero Actions artifacts, and the source branch `experiment/phase-9.3-development-baseline` is retained at its original SHA. No content was lost or altered.

This deviation is recorded rather than reverted, because reverting would require a force push to protected `main` and a ruleset bypass, which carries more risk than the deviation itself. Future phases use fast-forward-only integration.
### Phase 9.5 outcome — hypothesis refuted, pre-registered order restored

The audit ran on 2026-08-13 and returned `DEFECT_NOT_CONFIRMED`. Neither reviewer's subject error rate reached the pre-committed 15-point margin over their own control: the assistant scored 0.0259 against 0.0800 across all 497 images, the owner 0.1324 against 0.1250 across a 100-image subsample. The assistant's subject error rate is *lower* than its control rate.

The labels are sound and the reordering recorded above was wrong. The three signals that motivated it all hold — inverted confidence, one-way confusion, cross-fold reproduction — but they indicate a model failure, not a data failure: two reviewers read the `freshpotato` images as fresh, agreeing 65 of 68 and 55 of 68 over the shared subsample. The binding constraint returns to H1, where the pre-registered plan put it. `freshpotato` is the smallest class at 347 and visually adjacent to `rottenpotato` at 514, and a minority class collapsing into a visually similar majority is a characteristic imbalance failure; the argument that a 7.8:1 ratio was too mild did not weigh that adjacency.

This is the mechanism working as designed. The threshold, the control group, and the decision rule were fixed before any image was opened, so the audit could falsify the hypothesis that produced it, and it did.

Three deviations are recorded in the protocol and bear on how much the result carries. The owner reviewed a seeded 100-image subsample rather than all 497, leaving their control group at 32 examples. The assistant's judgments were visible in the session transcript before the owner judged, so the two reviews are not fully independent and the reported agreement of 0.80 raw, kappa 0.6259, is an upper bound. And `scripts.analyze_label_audit` was not run, because it requires two complete 497-row judgment files by design and that guard was not weakened to accept a partial second review.

The two-reviewer standard was therefore not met. Phase 9.6 was selected by the owner on the strength of the evidence rather than by a mechanically satisfied rule. Executing H1 remains a separate authorization, as does any locked-test evaluation.

## Phase 9.6 — H1 Loss Experiment Protocol Freeze

The owner approved the single changed factor, the acceptance threshold, the candidate count, and the failure branch on 2026-08-14, before any training. The method is frozen in [postholdout-loss001-protocol.md](postholdout-loss001-protocol.md).

```text
PHASE_9_6_SCOPE:
H1_LOSS_AND_CLASS_IMBALANCE
EXPERIMENT_ID:
deep3-postholdout-research-01-loss-001
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
NOT_YET_RUN
APPROVED_EXECUTION:
NOT_YET_GRANTED
CANDIDATE_COUNT:
1
LOCKED_TEST_MODEL_ACCESS:
NO
ARTIFACT_PUBLICATION:
LOCAL_ONLY
PHASE_9_6_OUTCOME:
NOT_ADVANCED_BUT_INCONCLUSIVE
PHASE_9_7:
DETERMINISM_THEN_RETEST
```

Freezing this protocol does not authorize the run. Training requires a separate explicit decision, as Phase 9.4 did for the baseline.

The experiment changes one parameter, `loss.class_balanced_beta`, from 0.999 to 0.9999. The baseline already trains with class-balanced focal loss, so this is a question about strength rather than about introducing the mechanism: `freshpotato` already carries the largest alpha of the fourteen classes and still collapses to recall 0.274, yet the correction is only partly applied — the weight ratio against `rottenapples` is 3.18 where the frequency ratio is 7.82. At 0.9999 that ratio becomes 6.97, essentially inverse frequency. The permitted config differences are enforced by a validator before dataset preparation, so "one factor at a time" is checked mechanically rather than trusted.

The acceptance threshold is fixed at Macro F1 at least 0.9112 with Top-1 at least 0.9466, being the pre-registered +0.010 improvement and −0.010 guardrail against the baseline's 0.9012 and 0.9566. The protocol records that this project has no repeat-seed measurement, so the run-to-run noise floor is unmeasured and the threshold is a reasoned rather than a measured choice; establishing it would have cost another nine to twelve GPU hours and was declined.

The failure branch is fixed in advance: a result below the threshold retires H1 as exhausted and makes Phase 9.7 H2 augmentation. Leaving it open would invite trying gamma, then label smoothing, then a combination, searching the H1 family without pre-registration until something cleared the bar by chance. The research plan's own stop condition, "stop when a primary family is exhausted", is what this encodes.

The 4,298-example locked test remains `FROZEN_UNOBSERVED_BY_MODEL`. No weight, checkpoint, dataset copy, Release, or tag is authorized.


## Phase 9.6a — Noise Floor Measured, Phase 9.6 Inconclusive

The measurement frozen before it ran returned `INCONCLUSIVE`. Three runs of the identical baseline recipe on identical folds gave Macro F1 0.901167, 0.912041, and 0.901858, so the sample standard deviation is 0.006089 and 2s is 0.012177. The loss-001 improvement of 0.0090 falls at or below that, and the range of 0.010874 agrees.

The decisive observation is rep002. Rerunning the baseline without changing a character of configuration produced 0.912041, above the 0.9112 threshold that loss-001 missed at 0.9102. An intervention that did nothing would have been accepted on that draw. A single run cannot resolve an effect of roughly one point of Macro F1 on this pipeline.

Phase 9.6 is therefore recorded as `INCONCLUSIVE` rather than as H1 exhausted: the experiment was underpowered to support that conclusion. The loss-001 verdict of `NOT_ADVANCED` is unchanged, having been computed against a threshold frozen before the run; this measurement re-scores nothing and bears only on the inference drawn afterwards.

Three samples give `s` two degrees of freedom, so the estimate is imprecise. The rule was applied as written, because selecting a sample size after seeing the spread would restore the freedom the protocol removed.

The owner chose determinism before re-testing over raising the effect-size bar or adopting a multi-seed protocol costing roughly 27 hours per candidate. Phase 9.7 introduces explicit seeding — `torch`, NumPy, and Python seeds, a `DataLoader` generator, and a documented decision on `cudnn_benchmark` against `torch.use_deterministic_algorithms` — and must also decide whether existing documentation overstates reproducibility. What this repository froze and verified is the data and the configuration; the training outcome was never reproducible. Whether loss-001 is re-run afterwards is a separate decision.

## Recorded history rewrite — 2026-08-15

A co-authorship trailer inconsistent with this repository's sole-authorship policy remained on the Phase 9.4 authorization commit. The history was rewritten to conform it to that policy.

Only commit messages changed. The tree hash at the branch head is identical before and after the rewrite, so no tracked file content was altered, and `git diff` between the pre-rewrite backup and the rewritten head is empty.

```text
REWRITE_DATE:
2026-08-15
REASON:
SOLE_AUTHORSHIP_POLICY_CONFORMANCE
COMMITS_REWRITTEN:
26
FILE_CONTENT_CHANGED:
NONE
TREE_HASH_BEFORE_AND_AFTER:
d4561b7b0c97df34f28ed1db3cb1ff7412f2e3ce
```

### Sequencing

The rewrite waited until both noise-floor replicates finished. `run_manifest.json` and `training_state.pt` record the repository commit a run started from, and `scripts/train.py` aborts a resume when the current `git rev-parse HEAD` does not match it. Rewriting while a run was in flight would have orphaned that SHA and made epoch-boundary resume impossible. The same constraint had already deferred this work once, on 2026-08-13, while the baseline run was executing.

### Run manifests are not edited

The `run_manifest.json` files under `weights/` record the commit each run actually started from, and remain untouched. Editing them would replace a record of what happened with a record of what is convenient. The table below is what carries provenance across the rewrite.

### Commit correspondence

| Before | After | Subject |
|---|---|---|
| `5757d0e` | `ec6eaea` | feat: authorize phase 9.4 baseline execution |
| `2f8a5ac` | `d13866a` | feat: record phase 9.4 baseline execution and development OOF result |
| `53c7dd0` | `0a6fc67` | feat: freeze phase 9.5 label quality audit protocol |
| `47a980f` | `28c0a61` | docs: add phase 9.5 label audit execution runbook |
| `a4e7347` | `93eb555` | docs: share one canonical pool reconstruction in the audit runbook |
| `8786cbb` | `0196879` | feat: add deterministic label audit review-set selection |
| `504278c` | `f4bfeaf` | feat: add label audit scoring and frozen decision rule |
| `04988d7` | `48f5e9b` | docs: make the audit runbook's threshold boundary test discriminating |
| `eafc7bc` | `0213e67` | fix: strengthen boundary test to discriminate >= vs > |
| `d4469d6` | `ba02e40` | feat: add blinded label audit review-set materialization |
| `e386d27` | `a515f90` | feat: add label audit unblinding and findings analysis |
| `21602b5` | `18d118c` | docs: cover the two-reviewer guard and main() in the audit runbook |
| `0252e7f` | `d062244` | test: cover the two-reviewer guard and the decision/model isolation in analyze_label_audit |
| `5016d3b` | `8b325a0` | test: pin the frozen decision threshold and surface per-group judgment counts |
| `4df6376` | `b4b3f51` | fix: harden the label audit CLIs against tampering, duplicate reviewers, and overwrite |
| `8c35126` | `d7c7189` | refactor: consolidate the frozen audit seeds and bump the findings schema |
| `578c58c` | `92fd673` | fix: derive review-set groups from the seeds instead of trusting the sealed key |
| `1ceb2ae` | `a214643` | feat: record phase 9.5 label audit outcome and restore the pre-registered order |
| `3d1e809` | `131b9c0` | feat: freeze the phase 9.6 H1 loss experiment protocol |
| `f6607aa` | `47bdaf3` | docs: add the phase 9.6 loss-001 execution runbook |
| `0702afd` | `3e88fd9` | feat: add key-level single-factor validation for loss experiments |
| `ed71060` | `7d7f5cb` | feat: enforce single-factor validation by experiment lineage |
| `574f1d2` | `0a1e074` | feat: compute the frozen loss-001 decision rule |
| `00bbdec` | `9c84a79` | feat: authorize phase 9.6 loss-001 execution |
| `5e1847f` | `635f0b0` | feat: record loss-001 outcome and freeze the noise floor measurement |
| `bf44372` | `c257838` | feat: record phase 9.6 outcome and the measured noise floor |

Runs affected, by the commit each recorded at launch:

| Run | Recorded in manifest | Equivalent after rewrite |
|---|---|---|
| `deep3-postholdout-research-01-baseline` | `5757d0e` | `ec6eaea` |
| `deep3-postholdout-research-01-loss-001` | `00bbdec` | `9c84a79` |
| `deep3-postholdout-research-01-baseline-rep002` | `5e1847f` | `635f0b0` |
| `deep3-postholdout-research-01-baseline-rep003` | `5e1847f` | `635f0b0` |

Merged pull request bodies quote pre-rewrite identifiers. Those are historical records of what was reviewed at the time and are left as written; this table resolves them.

## Phase 9.7 — determinism introduced and the reproducibility record corrected, 2026-08-15

Phase 9.6a measured a run-to-run noise floor of `2s = 0.012177` against a loss-001 improvement of `0.0090` and returned `INCONCLUSIVE`. The cause was that the training pipeline set no random seed. Phase 9.7 introduces explicit seeding under the frozen protocol in [postholdout-determinism-protocol.md](postholdout-determinism-protocol.md), which fixes the seed at 20260815 and a six-branch adoption ladder before any verification run executes.

Every recorded result produced before this phase came from the unseeded pipeline: `deep3-canonical-reference-01`, `deep3-postholdout-research-01-baseline`, `deep3-postholdout-research-01-loss-001`, `deep3-postholdout-research-01-baseline-rep002`, and `deep3-postholdout-research-01-baseline-rep003`. Their recorded metrics are unchanged and remain accurate measurements of the runs that produced them. None is re-scored.

The documentation failure was one of placement, not of truthfulness. `training.md`, `canonical-training-readiness.md`, `canonical-holdout-evaluation.md`, and `canonical-training-unblock.md` each stated that bit-for-bit reproducibility was not claimed and that global seeding was not introduced. The top-level documents a reader reaches first — the README opening sentence, the README reproducibility status table, and the "Remaining limitations" list in `reproducibility.md` — carried no such qualification, and the research plan required every run to record a seed that did not exist. Those four locations are corrected here. The individual result documents are not annotated; this entry carries the traceability.

Two design findings are recorded because both could have been silently lost. The `DataLoader` generator proposed by the Phase 9.6a follow-up is rejected: at `num_workers = 0` the sampler draws from the global torch generator, which `training_state.pt` already persists, so a separate generator would have broken working epoch-boundary resume determinism. And determinism removes measurement noise but not seed-to-seed variation, so a deterministic baseline-candidate pair is a paired comparison under common random numbers that estimates the effect for one seed draw only. Phase 9.8 must argue its claim strength against that limit rather than inherit it unexamined.
