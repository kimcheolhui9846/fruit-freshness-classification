# Release checklist

This is a decision checklist, not evidence that a tag or GitHub Release exists. `[x]` means the Phase 6.3 audit found evidence; `[ ]` requires a later owner decision or release-time verification.

## Required for an engineering milestone

### Repository state

- [x] Phase branches are retained for audit.
- [x] Generated datasets, caches, checkpoints, weights, and virtual environments are excluded from Git.
- [x] Protected nested repository content is not tracked by this repository.
- [ ] Reconfirm a clean, synchronized `main` immediately before any tag or GitHub Release action.
- [ ] Reconfirm local, `origin/main`, and GitHub `main` resolve to the same release candidate SHA.

### Code and tests

- [x] Full local `unittest` validation was completed for the audited commit.
- [x] Windows CPU CI passed.
- [x] Ubuntu CPU CI passed.
- [x] `compileall` and both CLI help paths passed.
- [x] CI repository-cleanliness checks passed.
- [ ] Re-run the full suite and CI at the final proposed release SHA.

### Documentation

- [x] README documents setup, configuration, training, evaluation, limitations, and CI.
- [x] Environment, dataset, configuration, training, evaluation, reproducibility, and CI documents exist.
- [x] Release-readiness, governance, and changelog documents exist.
- [ ] Owner reviews the draft release notes and changelog before publication.

### Governance

- [ ] Select and approve a software license.
- [ ] Review dataset attribution, terms, and redistribution boundaries.
- [ ] Approve citation identity and author metadata.
- [ ] Create `CITATION.cff` only after the identity and license decisions are approved.

### Release artifacts

- [ ] Approve or defer the proposed `v0.1.0` prerelease.
- [ ] Finalize the changelog entry and release notes.
- [ ] Authorize a Git tag.
- [ ] Authorize a GitHub Release.

## Required only for a model-performance release

- [ ] Complete the canonical three-fold training run.
- [ ] Produce traceable trained fold checkpoints outside Git or through an approved distribution channel.
- [ ] Reproduce trained-checkpoint holdout evaluation.
- [ ] Validate a benchmark-quality result with experiment provenance.
- [ ] Decide how weights, results, and model cards will be distributed and attributed.

## Explicit non-actions in Phase 6.3

- [x] No software license was selected or added.
- [x] No `CITATION.cff` was created.
- [x] No Git tag or GitHub Release was created.
- [x] No branch-protection, ruleset, or repository-metadata setting was changed.
