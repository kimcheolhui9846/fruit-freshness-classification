# Release checklist

This checklist records verified repository state through Phase 6.4. `[x]` means committed evidence exists; `[ ]` requires a later owner decision or release-time verification. It does not mean that a tag or GitHub Release exists.

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
- [x] Release-readiness, governance, citation, and changelog documents exist.
- [ ] Owner reviews the draft release notes and changelog before publication.

### Governance

- [x] Software license selected: MIT.
- [x] Canonical `LICENSE` added.
- [x] Copyright identity approved: 2025 김철희.
- [x] Citation identity approved: Choelhui Kim.
- [x] `CITATION.cff` added with repository-only metadata.
- [x] Software, dataset, and trained-weight licensing boundaries documented.
- [ ] Dataset attribution, terms, and redistribution approved.
- [ ] Trained-weight redistribution approved.

### Release artifacts

- [ ] Approve or defer the proposed `v0.1.0` prerelease.
- [ ] Approve a release date.
- [ ] Finalize the changelog entry and release notes.
- [ ] Authorize a Git tag.
- [ ] Create a Git tag.
- [ ] Authorize a GitHub prerelease or Release.
- [ ] Create a GitHub prerelease or Release.
- [ ] Create a DOI.

## Required only for a model-performance release

- [ ] Complete the canonical three-fold training run.
- [ ] Produce traceable trained fold checkpoints outside Git or through an approved distribution channel.
- [ ] Reproduce trained-checkpoint holdout evaluation.
- [ ] Validate a benchmark-quality result with experiment provenance.
- [ ] Decide how weights, results, and model cards will be distributed and attributed.

## Explicit non-actions in Phase 6.4

- [x] No Git tag, GitHub prerelease, GitHub Release, release date, or DOI was created.
- [x] No dataset contents or trained weights were distributed.
- [x] No branch-protection, ruleset, or repository-metadata setting was changed.
