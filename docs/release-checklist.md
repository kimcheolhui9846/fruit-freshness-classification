# Release checklist

This checklist records the authorized Phase 6.5 engineering-milestone preparation state. `[x]` means the documented prerequisite or approval exists; `[ ]` marks a release action or model-performance requirement that is still incomplete at this document-edit time.

## Approved release identity

- [x] Approved version tag: `v0.1.0`.
- [x] Approved release type: `PRERELEASE`.
- [x] Approved release date: `2026-08-02`.
- [x] Approved title: `Fruit Freshness Classification v0.1.0 — Engineering Milestone`.
- [x] Approved annotated tag message: `Fruit Freshness Classification v0.1.0 engineering milestone`.
- [x] Approved final notes use the Phase 6.3 draft as their source.
- [x] Engineering-milestone scope is distinguished from a model-performance release.

## Repository state

- [x] Phase branches are retained for audit.
- [x] Generated datasets, caches, checkpoints, weights, and virtual environments are excluded from Git.
- [x] Protected nested repository content is not tracked by this repository.
- [x] Starting local `main`, `origin/main`, and GitHub `main` were verified at the Phase 6.4 SHA.
- [x] Windows CPU CI passed for the starting `main` candidate.
- [x] Ubuntu CPU CI passed for the starting `main` candidate.
- [x] README, changelog, release-readiness audit, governance documentation, and release checklist were reviewed.
- [x] Canonical `LICENSE` added.
- [x] `CITATION.cff` added with repository-only metadata.
- [x] MIT `LICENSE` and repository-only `CITATION.cff` are present.
- [x] Release documentation and offline release-publication contracts are prepared.

## Artifact policy

- [x] No dataset attachment is authorized.
- [x] No trained-weight attachment is authorized.
- [x] No checkpoint attachment is authorized.
- [x] No cache, environment, log, or other binary artifact attachment is authorized.
- [x] Dataset redistribution remains excluded.
- [x] Trained-weight distribution remains excluded.

## External publication actions

- [ ] Final release-branch CI passes on Windows and Ubuntu.
- [ ] Annotated Git tag created.
- [ ] Tag pushed.
- [ ] GitHub prerelease created.
- [ ] GitHub prerelease verified.

## Required only for a model-performance release

- [ ] Canonical training completed.
- [ ] Trained checkpoints produced.
- [ ] Trained evaluation reproduced.
- [ ] Benchmark result validated.
- [ ] Weight, result, and model-card distribution governance approved.

## Explicit non-actions in Phase 6.5 preparation

- [x] No tag or GitHub Release has been created at this document-edit time.
- [x] No dataset contents, trained weights, checkpoints, or binary artifacts were attached.
- [x] No branch-protection, ruleset, or repository-metadata setting was changed. No DOI was created.
