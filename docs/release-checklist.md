# Release checklist

This checklist records the published Phase 6.5 engineering milestone. `[x]` means a documented prerequisite or external publication action is complete; `[ ]` marks a model-performance requirement that remains incomplete.

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

- [x] Final release-branch CI passed on Windows and Ubuntu.
- [x] Annotated Git tag created.
- [x] Tag pushed.
- [x] GitHub prerelease created.
- [x] GitHub prerelease verified.
- [x] Tagged main CI run 30738724706 passed on Windows and Ubuntu.

## Required only for a model-performance release

- [ ] Canonical training completed.
- [ ] Trained checkpoints produced.
- [ ] Trained evaluation reproduced.
- [ ] Benchmark result validated.
- [ ] Weight, result, and model-card distribution governance approved.

## Publication evidence and non-actions

- [x] Annotated tag `v0.1.0` was pushed and peels to the CI-verified release commit.
- [x] GitHub prerelease was published from the committed release notes and verified with no uploaded assets.
- [x] Historical pre-publication state is retained for audit: before external actions, `[ ] Annotated Git tag created.` and `[ ] GitHub prerelease created.` were intentionally pending.
- [x] No dataset contents, trained weights, checkpoints, caches, environments, logs, or manually uploaded binary artifacts were attached.
- [x] GitHub source archives, when offered by the platform, are platform-generated defaults rather than manual uploads.
- [x] No branch-protection, ruleset, or repository-metadata setting was changed. No DOI was created.
