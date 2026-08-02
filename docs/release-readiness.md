# Release readiness

## Decision boundary

This document began as a Phase 6.3 release audit for the repository state that began at commit `be6e347328f80c423d2358c291257640a8147fd4`. Phase 6.4 applies the explicitly approved MIT software license and repository-only citation metadata. It remains a readiness record, not a release, tag, GitHub Release, DOI, or repository-settings change.

## Candidate milestone

The truthful candidate is an **engineering and reproducibility milestone**: a modular PyTorch research repository with a committed experiment configuration, production dataset compatibility, thin training and holdout-evaluation CLIs, reproducibility evidence, and cross-platform repository CI.

It is **not a trained-model benchmark release**. No trained weights or benchmark-quality metrics are included.

### Verified engineering capabilities

- Modular `src/` packages for data, transforms, model, losses, engine, trainers, evaluation, inference, and utilities.
- A committed TOML experiment contract shared by `deep3.ipynb`, `python -m scripts.train`, and `python -m scripts.evaluate`.
- Fixed-revision Hugging Face archive loading with safe managed extraction and explicit ImageFolder handling.
- Clean-environment installation evidence, real-data CUDA CMT smoke coverage, checkpoint interoperability, and full-holdout evaluation-path execution with untrained compatibility fixtures.
- Offline Windows and Ubuntu CPU CI, currently passing 136 tests per runner with five intentional CUDA-only skips.
- Portfolio-oriented [README](../README.md) and detailed usage documentation.

### Unverified or unavailable capabilities

- Canonical three-fold training has not been run.
- Trained checkpoint performance and trained-checkpoint holdout evaluation are not available.
- Benchmark reproduction, full `deep3.ipynb` execution, independent-machine reproduction, and generic unlabeled-image inference are not verified or implemented.
- The repository does not distribute datasets, trained checkpoints, or model weights.

## Release-readiness matrix

| Area | Status | Evidence | Blocker or required decision |
|---|---|---|---|
| Source architecture | Verified | Modular packages and parity/integration tests | None for an engineering milestone |
| Configuration | Verified | Committed `configs/deep3.toml` and loader tests | None |
| Dependencies | Verified boundary | Exact direct pins and clean-environment evidence | Other Python versions are not independently verified |
| Dataset loading | Verified | Fixed archive, safe extraction, counts, and loader tests | Dataset governance review remains required |
| Training CLI | Available | Parser, orchestration, and checkpoint-policy tests | Canonical training not run |
| Evaluation CLI | Available | Complete-fold validation and real holdout-path evidence | Only untrained compatibility fixtures were evaluated |
| Tests | Verified | Local full suite and repository contracts | Re-run before any release action |
| CI | Verified | Windows/Ubuntu CPU run `30696266143` | CPU/offline health gate only |
| Reproducibility | Partially verified | Clean environment and bounded real-data evidence | No independent-machine or canonical-run proof |
| README | Verified | Public commands, limits, and links are contract-tested | None |
| Detailed documentation | Verified | Environment through CI documents | CI checkout wording corrected in this phase |
| Software license | Resolved | Canonical MIT `LICENSE` added in Phase 6.4 | Software/dataset boundary remains documented separately |
| Citation | Resolved repository metadata | `CITATION.cff` identifies Choelhui Kim for repository-only citation | Version, release date, paper, and DOI remain unavailable |
| Dataset attribution | Incomplete | Source is linked; Hub metadata labels it `openrail` | Terms, attribution, and redistribution review remain pending |
| Trained artifacts | Unavailable | Git excludes weights and checkpoints | Distribution remains subject to separate terms review and canonical-training authorization |
| Benchmark evidence | Unavailable | No trained benchmark artifact exists | Authorize canonical training and evaluation |
| Release notes | Draft ready | This document and [CHANGELOG](../CHANGELOG.md) | Owner review required |
| Versioning | Pending | No Git tags or GitHub Releases exist | Phase 6.5 approval is required before any first tag |
| Branch protection | Not configured | `main` is unprotected; rulesets are empty | Owner must choose workflow policy |
| Repository metadata | Needs review | Public repository, `main`, no homepage or topics | Owner approval required for metadata edits |

## Version recommendation

There are no existing Git tags or GitHub Releases. The MIT License and repository-only citation metadata resolve repository governance, but they do not authorize a release. The appropriate first public engineering milestone remains a possible **`v0.1.0` prerelease** only after separate Phase 6.5 approval.

A version tag and GitHub Release remain pending. A release date, final release notes, dataset-redistribution review, trained-weight-distribution review, and model-evidence decisions remain separate blockers. The repository is not ready for a trained-model or benchmark-performance release.
## Draft milestone release notes

> Draft only. No Git tag or GitHub Release has been created.

### Summary

This milestone packages the repository as a reproducible engineering baseline for fresh/rotten fruit classification research.

### Included

- Modular research code and a shared TOML experiment configuration.
- Training and labeled-holdout evaluation commands:

```powershell
python -m scripts.train --config configs/deep3.toml --output-dir weights
python -m scripts.evaluate --config configs/deep3.toml --checkpoint-dir weights
```

- Fixed-revision Hugging Face dataset compatibility, clean-environment evidence, real-data CUDA smoke coverage, checkpoint interoperability, and Windows/Ubuntu CPU CI.
- Installation instructions in [environment documentation](environment.md) and operational details in [training](training.md) and [evaluation](evaluation.md).

### Verification and limits

The repository has passed its documented cross-platform health checks. No trained weights or benchmark-quality metrics are included. Canonical training, trained-checkpoint evaluation, benchmark reproduction, full notebook execution, and independent-machine verification remain outside this milestone.

### Governance and upgrade notes

The MIT License and repository-only citation metadata are now committed. Dataset-governance review, version tag, GitHub Release, release date, DOI, and branch-protection policy remain owner decisions. There are no upgrade notes because no prior versioned release exists.

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

## Recommendation

**Eligible for Phase 6.5 engineering-milestone approval review.** MIT licensing and repository citation identity are resolved, but a version tag, release policy/date, final notes, dataset redistribution review, and trained-weight distribution review remain pending. It is not ready for a trained-model or benchmark-performance release.
