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
NOT_YET_RUN
PHASE_9_4:
RUNBOOK_PREPARED
```

The deterministic development-CV identity is materialized once as a tracked index record; it is not a dataset or prediction artifact. No hyperparameter tuning, loss/augmentation/sampler/optimization/architecture experiment, external data acquisition, binary publication, Release, or tag is approved by this decision.

## Phase 9.4 — Baseline Execution Runbook Preparation

Phase 9.4 documents how the approved baseline would be executed and does not execute it. [postholdout-baseline-runbook.md](postholdout-baseline-runbook.md) records the frozen inputs, the fresh and resume commands, the development-only OOF evaluation command, preflight and stop conditions, a resource expectation, and an unresolved owner approval block.

```text
PHASE_9_4_SCOPE:
DOCUMENTATION_ONLY
BASELINE_EXECUTION_STATUS:
NOT_YET_RUN
MODEL_TRAINING:
NO
MODEL_EVALUATION:
NO
PHASE_9_5:
NOT STARTED
```

Phase numbering is resolved as follows: Phase 9.4 covers baseline execution, and the first loss/class-imbalance experiment is Phase 9.5. An earlier external handoff draft numbered the loss experiment as 9.4; the repository documents and their contract tests are the authoritative numbering.

### Recorded integration deviation — PR #5

PR #5 was integrated into `main` with GitHub "Rebase and merge" rather than the fast-forward-only integration this project prefers. The rebase rewrote commit identifiers: PR head `48a61eb63d57604351088ea72bbe69d22fe50a39` became merge commit `15eb552e7ae1f698e99c5d3bac3e9516180f7053`.

Verified consequences: the resulting tree hash `4275acce956419989dfcb6a3bb1158538eafe9d1` is byte-identical to the PR head tree, linear history is preserved, the `Protect main` ruleset accepted the integration, final-main CI passed on both required jobs with zero Actions artifacts, and the source branch `experiment/phase-9.3-development-baseline` is retained at its original SHA. No content was lost or altered.

This deviation is recorded rather than reverted, because reverting would require a force push to protected `main` and a ruleset bypass, which carries more risk than the deviation itself. Future phases use fast-forward-only integration.