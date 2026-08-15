# Repository Metadata and Portfolio Discoverability

## Purpose

Repository metadata helps portfolio reviewers and research users understand this project before opening the README. It must remain concise, truthful, version-independent, free of performance exaggeration, and consistent with the README and release limitations.

## Approved Description

> Reproducible PyTorch fruit freshness classification pipeline with CMT, config-driven training/evaluation, cross-platform CI, and documented engineering workflows.

This description is accurate because PyTorch is the implementation framework, the task is fruit freshness classification, CMT is the active model architecture, and training and evaluation are config-driven. The repository CI runs on Windows and Ubuntu, and the engineering workflows are documented.

It does not claim production deployment, trained-model availability, benchmark superiority, an end-user inference service, or state-of-the-art performance.

## Approved Topics

The exact approved topic set is:

- `pytorch` - Identifies the primary deep-learning framework used by the project.
- `computer-vision` - Describes the visual-learning problem domain.
- `image-classification` - Names the supervised classification task.
- `deep-learning` - Locates the project within neural-network research practice.
- `machine-learning` - Supports broader machine-learning discovery.
- `reproducibility` - Reflects the documented environment and verification workflow.
- `mlops` - Reflects the configuration, CI, and engineering practices in scope.
- `huggingface-datasets` - Identifies the external dataset-loading integration.
- `research-software` - Signals that the repository is maintained as reusable research software.
- `fruit-freshness` - Names the project-specific application domain.

The list intentionally avoids redundant project-suffix variants and does not add unsupported production, performance, deployment, personal, or version-specific labels.

## Homepage Policy

The repository homepage remains empty because no stable public project site exists. The GitHub repository and README remain the canonical project entry point. A homepage may be added only after an actual stable project site is deployed.

## Social Preview Policy

Custom social preview remains deferred. No graphic is required for repository correctness, and no generated image is added in this Phase. Any future preview must be explicitly designed and approved.

## Profile Discoverability

Profile pinning is a recommendation only: the owner may pin this repository near other strong portfolio projects and may use a profile README for context when one exists. This Phase makes no profile-level change and records no pinning action.

## Mutation Boundaries

Only description and topics are authorized live mutations in Phase 7.3. Phase 7.3 does not change the homepage, visibility, default branch, merge settings, repository features, rulesets, tags, releases, or source files.

Repository visibility remains public and unchanged. The default branch remains `main`. Both GitHub rulesets remain unchanged. The published `v0.1.0` tag remains unchanged.

## Rollback Policy

Record all original metadata before mutation. If a partial API mutation fails, restore only the Phase 7.3 metadata that changed, verify rollback through live readback, and do not alter unrelated repository fields. A partially applied metadata state must not be left undocumented.

## Live Execution Record

- Starting `main` SHA: `ec5b3d8af8d1ab2e72c64f23d0dc8d6144344412`.
- Phase branch: `docs/phase-7.3-repository-metadata`.
- Original metadata: description `For my data science studies`; empty homepage; no topics.
- Implementation commit: `2f75ea400dc85d5eda563969d5b9318184cc2c64` (`docs: define repository metadata and discoverability`). GitHub Actions run `30746176046` passed both required jobs before any metadata mutation.
- Approved API operations: `PATCH /repos/{owner}/{repo}` changed only the description, then `PUT /repos/{owner}/{repo}/topics` replaced the empty topic list with the exact approved ten-topic set documented above.
- Live readback confirms the approved description, an empty homepage, public visibility, default branch `main`, an unchanged deferred custom social-preview state, and all ten approved topics. GitHub may return topic names in a normalized order; set equality and lowercase uniqueness were verified.
- No rollback was required. The two external temporary JSON payloads contained no credentials and were removed after readback.
- `Protect main` (ID `20229405`) and `Protect v0.1.0` (ID `20232130`) remain active and unchanged. The protected tag still peels to `b38ebd36f4fa4f1fe012b957095db6dcbce20832`; the GitHub prerelease remains published, non-draft, and has zero assets.
- The final handoff commit is intended for a protected fast-forward-only update of `main` after exact-SHA CI verification. A known externally managed temporary-ref issue can make `git fetch origin` fail; it is not modified, and local/remote/GitHub SHA comparison is the safe fallback.
