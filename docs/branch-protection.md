# Main branch protection

## Purpose

Phase 7.1 protects the default branch, `main`, while preserving the repository's established, SHA-preserving Phase workflow. The approved configuration requires the two `Repository CI` jobs before an update to `main`, requires linear history, blocks force pushes, and blocks branch deletion.

This document describes the approved configuration and operating procedure. The live ruleset identifier and API readback evidence are added only after the approved ruleset is created and verified.

## Required workflow

1. Create a Phase branch from synchronized `main`.
2. Implement and validate changes on the Phase branch.
3. Push the Phase branch.
4. Wait for both required CI jobs on the exact final Phase-branch commit.
5. Verify the local Phase branch equals the GitHub Phase branch.
6. Merge locally using `git merge --ff-only`.
7. Push `main` normally.
8. Verify the new `main` CI run.

The workflow preserves original commit SHAs. A local `git merge --ff-only` creates no merge commit, and the protected normal push proves that the exact tested commit is accepted on `main`.

## Required checks

The ruleset requires these exact GitHub Actions check contexts:

- `ubuntu-latest / Python 3.12`
- `windows-latest / Python 3.12`

The contexts are emitted by the `Repository CI` workflow. Their GitHub Actions provenance was verified during Phase 7.1. `integration_id` is intentionally omitted: exact context names are sufficient, and no integration-ID constraint is needed for the approved design.

## Applied rules

The approved branch ruleset has the name `Protect main`, targets the repository default branch, and uses active enforcement.

- Required status checks: enabled for the two contexts above.
- Strict up-to-date policy: disabled.
- Linear history: required.
- Force pushes: prohibited through the non-fast-forward rule.
- Branch deletion: prohibited.
- Pull requests: explicitly not required in Phase 7.1.
- Approvals: not required.
- Signed commits: explicitly deferred and not required.
- Merge queue: not required.
- Deployments, code scanning, and code quality gates: not required.
- Push-actor restrictions: not configured.
- Bypass actors: absent.
- Tag protection: deferred; no tag ruleset is created.

## Why pull requests are not required

This repository's established workflow validates the exact Phase-branch commit before a local fast-forward-only merge and normal push to `main`. Requiring a GitHub pull request would impose a different merge strategy rather than preserving this SHA-stable workflow. This does not mean pull requests are undesirable in general; PR-based governance can be considered separately when the repository workflow changes.

## Recovery policy

Do not force push, do not bypass failed CI, and do not disable protection merely to merge a failing change. Correct the Phase branch, rerun the required checks, and retry only a normal fast-forward update. A ruleset may be changed only for a proven configuration mistake, while preserving required status checks, linear history, force-push prevention, and deletion prevention. Any administrator recovery must be recorded in `SESSION_HANDOFF.md`.

## Release-tag boundary

The published annotated tag `v0.1.0` is unaffected by this branch ruleset. Tag protection remains deferred, and `v0.1.0` must not be moved, recreated, deleted, or force-updated. The source-only engineering milestone remains distinct from a model-performance release: canonical training, trained-checkpoint evaluation, and benchmark reproduction remain incomplete.

## Repository-setting scope

The approved `Protect main` branch ruleset is the only repository-setting mutation in Phase 7.1. This Phase does not create a classic branch-protection rule, change repository metadata, alter the default branch, create tag rules, require pull requests, require signed commits, or change the GitHub Release.

## Live ruleset record

- **Ruleset:** GitHub repository ruleset `Protect main` (ID `20229405`), source type `Repository`, target `branch`, active enforcement.
- **Target condition:** includes only `~DEFAULT_BRANCH` and has no exclusions; on this repository, that is `main`.
- **Creation evidence:** Created through GitHub's repository-ruleset REST endpoint on `2026-08-02T18:40:33.840+09:00`, after the Phase branch commit `4e655a39c1f58f6c05c6551144009757b7b54a0f` passed `Repository CI` run `30742002870` on both required jobs.
- **API readback:** The ruleset contains exactly `deletion`, `non_fast_forward`, `required_linear_history`, and `required_status_checks`. The status-check rule has `do_not_enforce_on_create: false`, `strict_required_status_checks_policy: false`, and exactly the two documented contexts. `bypass_actors` is an empty list.
- **Effective-rule evidence:** GitHub's active-rules endpoint for `main` returned exactly those same four rule types after creation. No integration-ID constraint was added; the two exact contexts remain the approved check selectors.
- **Scope confirmation:** This live ruleset does not require pull requests, reviews, signed commits, merge queue, deployments, code scanning, push-actor restrictions, or tag protection. It does not alter the published `v0.1.0` annotated tag or its source-only prerelease.
- **Operational check pending at this record:** The documentation handoff commit must pass both required checks on its Phase branch before the normal fast-forward-only update of `main` is attempted. No force push, bypass, direct deletion test, tag mutation, or branch-protection weakening is authorized.
