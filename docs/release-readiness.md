# Release readiness

## Decision boundary

This document began as a Phase 6.3 release audit for the repository state that began at commit `be6e347328f80c423d2358c291257640a8147fd4`. Phase 6.4 applied the explicitly approved MIT software license and repository-only citation metadata. Phase 6.5 records the owner-authorized first engineering release preparation; this document is not itself a Git tag, GitHub Release, DOI, or repository-settings change.

The release is an engineering and reproducibility milestone, not a trained-model benchmark release. No trained weights or benchmark-quality metrics are included.

## Phase 6.5 release authorization

| Approval | Value |
|---|---|
| Authorized release target | `v0.1.0` |
| Approved release type | `PRERELEASE` |
| Approved release date | `2026-08-02` |
| Approved release title | `Fruit Freshness Classification v0.1.0 — Engineering Milestone` |
| Release-note source | Phase 6.3 draft finalized as [`docs/releases/v0.1.0.md`](releases/v0.1.0.md) |
| Tag type | Annotated |
| Artifact attachment policy | No dataset, trained weights, checkpoints, or other binary artifacts |
| Branch protection | Deferred |
| Canonical training | Deferred |

Authorized release target: `v0.1.0`.
Approved release type: `PRERELEASE`.
Approved release date: `2026-08-02`.

At this document-edit time, the annotated tag and GitHub Release are authorized but have not yet been created. The release will be published only after the release branch and merged `main` pass their required Windows and Ubuntu CI checks.

## Candidate milestone

The truthful candidate is an **engineering and reproducibility milestone**: a modular PyTorch research repository with a committed experiment configuration, production dataset compatibility, thin training and holdout-evaluation CLIs, reproducibility evidence, and cross-platform repository CI.

It is **not a trained-model benchmark release**. No trained weights or benchmark-quality metrics are included.

### Verified engineering capabilities

- Modular `src/` packages for data, transforms, model, losses, engine, trainers, evaluation, inference, and utilities.
- A committed TOML experiment contract shared by `deep3.ipynb`, `python -m scripts.train`, and `python -m scripts.evaluate`.
- Fixed-revision Hugging Face archive loading with safe managed extraction and explicit ImageFolder handling.
- Clean-environment installation evidence, real-data CUDA CMT smoke coverage, checkpoint interoperability, and full-holdout evaluation-path execution with untrained compatibility fixtures.
- Offline Windows and Ubuntu CPU CI, currently passing 148 tests per runner with five intentional CUDA-only skips.
- Portfolio-oriented [README](../README.md) and detailed usage documentation.

### Unverified or unavailable capabilities

- Canonical three-fold training has not been run.
- Trained checkpoint performance and trained-checkpoint holdout evaluation are not available.
- Benchmark reproduction, full `deep3.ipynb` execution, independent-machine reproduction, and generic unlabeled-image inference are not verified or implemented.
- The repository does not distribute datasets, trained checkpoints, or model weights.

## Release-readiness matrix

| Area | Status | Evidence | Boundary |
|---|---|---|---|
| Source architecture | Verified | Modular packages and parity/integration tests | None for an engineering milestone |
| Configuration | Verified | Committed `configs/deep3.toml` and loader tests | None |
| Dependencies | Verified boundary | Exact direct pins and clean-environment evidence | Other Python versions are not independently verified |
| Dataset loading | Verified | Fixed archive, safe extraction, counts, and loader tests | Dataset governance remains separate |
| Training CLI | Available | Parser, orchestration, and checkpoint-policy tests | Canonical training not run |
| Evaluation CLI | Available | Complete-fold validation and real holdout-path evidence | Only untrained compatibility fixtures were evaluated |
| Tests | Verified | Local full suite and repository contracts | Re-run at the final release commits |
| CI | Verified | Windows/Ubuntu CPU run `30733985992` | CPU/offline health gate only |
| Reproducibility | Partially verified | Clean environment and bounded real-data evidence | No independent-machine or canonical-run proof |
| README and detailed docs | Verified | Public commands, limits, and links are contract-tested | None |
| Software license | Resolved | Canonical MIT `LICENSE` | Software/dataset boundary remains separate |
| Citation | Resolved repository metadata | `CITATION.cff` identifies Choelhui Kim | No DOI, paper, or release metadata claim |
| Dataset attribution | Incomplete | Source is linked; public metadata labels it `openrail` | Terms, attribution, and redistribution review remain pending |
| Trained artifacts | Unavailable | Git excludes weights and checkpoints | Distribution remains subject to separate review and canonical-training authorization |
| Benchmark evidence | Unavailable | No trained benchmark artifact exists | Canonical training and evaluation remain deferred |
| Release notes | Finalized | [`docs/releases/v0.1.0.md`](releases/v0.1.0.md) | Must remain truthful at publication |
| Versioning | Authorized | Owner-approved `v0.1.0` prerelease | Tag and GitHub Release not yet created at document-edit time |
| Branch protection | Deferred | `main` is unprotected; rulesets are empty | Separate owner decision |
| Repository metadata | Unchanged | Public repository, `main`, no homepage or topics | Separate owner decision |

## Approved release target

`v0.1.0` is the first authorized public engineering milestone. It uses the approved prerelease type and release date above. The finalized release notes preserve the Phase 6.3 draft's scope: modular research code, shared configuration, training and labeled-holdout evaluation commands, fixed-revision dataset compatibility, bounded reproducibility evidence, Windows/Ubuntu CPU CI, MIT licensing, and repository-only citation metadata.

There is no prior versioned release, so no upgrade path or previous-version comparison is claimed. The external dataset is not redistributed; trained weights remain unavailable and excluded. Canonical training, trained-checkpoint evaluation, benchmark reproduction, full notebook execution, and independent-machine verification remain incomplete.

## Branch-protection recommendations

Recommendations only; no branch protection or ruleset changes were made.

1. Require pull requests before merging once the repository moves away from the current solo-maintainer, locally verified fast-forward workflow.
2. Require the `Repository CI` workflow and both current Windows and Ubuntu job results; confirm the exact required-check names at configuration time.
3. Prevent force pushes and branch deletion on `main`.
4. Require branches to be current before merge if PR protection is enabled.

For a solo developer, this improves auditability but adds a PR step to the present direct-main fast-forward process. Retained phase branches remain useful whether this policy is adopted or deferred.

## Repository metadata recommendations

Recommendations only; metadata was not changed.

- Replace the current generic description, `For my data science studies`, with a concise scope statement about modular PyTorch fruit-freshness classification and reproducibility.
- Add a project homepage only when a durable portfolio or documentation page exists.
- Consider the topics `pytorch`, `computer-vision`, `image-classification`, `machine-learning`, `reproducibility`, `mlops`, and `huggingface-datasets`.
- Keep the repository public, unarchived, and on `main` if that remains the owner's intended collaboration model.

## Artifact and secret audit

The current tree and normal local/origin branch history contain no tracked dataset archives, extracted images, virtual environments, checkpoints, weights, Hugging Face caches, generated outputs, or large blobs above 5 MiB. Secret-pattern matches occur only inside tests that assert secrets are absent; no credential value was found.

The retained notebooks and continuation log contain historical machine-specific path literals. The portable CLI entry points do not use those literals, but the notebook path is a reproducibility risk and should not be presented as a portable notebook workflow until a separately authorized behavior-preserving migration is completed. No history rewrite is recommended.

## Release decision

The owner explicitly authorized `v0.1.0` as a prerelease engineering milestone dated `2026-08-02`. This authorization applies to source code and documentation only. Dataset redistribution, trained-weight distribution, branch protection, canonical training, a model-performance release, and a DOI remain separate decisions.
