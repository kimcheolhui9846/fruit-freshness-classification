# Phase Branch Retention and Cleanup Policy

## Purpose

Branches are reviewed separately from commit history because a branch is a named development or audit reference, not merely a duplicate of its commits. Deleting a fully merged branch does not remove commits already reachable from `main`, but deletion still requires reachability evidence, operational checks, and explicit owner approval. Fully merged never means that branch deletion is automatically harmless.

## Canonical References

- `main` is the canonical current repository state and must always be retained.
- Annotated tags are canonical published-release references; `v0.1.0` is a protected tag, not a branch.
- `SESSION_HANDOFF.md` is the phase audit trail.
- Branches are temporary development or audit references unless their classification requires retention.

## Classification System

| Classification | Meaning |
|---|---|
| `MANDATORY_RETAIN` | The default branch or another proven operationally mandatory branch. |
| `RETAIN_RELEASE_AUDIT` | A release-preparation or publication branch with current audit value. |
| `RETAIN_UNIQUE_COMMITS` | A branch with commits not reachable from `main`. |
| `RETAIN_ACTIVE_OR_RECENT` | The current or recently active governance branch. |
| `TEMPORARY_RETAIN` | A fully merged branch retained through the conservative retention window. |
| `SAFE_DELETE_CANDIDATE` | A branch that satisfies every documented safe-deletion gate; this remains non-authorizing. |
| `REVIEW_REQUIRED` | A branch whose purpose or retained evidence needs an owner decision. |
| `BLOCKED_UNVERIFIED` | A branch whose state cannot be established reliably. |

## Retention Periods

- Retain the current Phase branch through Phase completion and review.
- Retain merged Phase branches for at least 90 days after the latest published milestone or until the next milestone is published, whichever is later.
- Retain a release branch at least until the next release milestone is published and verified.
- Retain branches with unique commits until they are explicitly resolved.
- Retain failed or blocked branches when they contain unique diagnostic evidence.
- Retain remote-only unknown branches until their state is verified.

The latest milestone is the `v0.1.0` engineering prerelease published on 2026-08-02. This is a conservative recommendation, not a deletion schedule or immediate authorization.

## Cleanup Authorization

Phase 7.4 does not authorize deletion. Phase 7.5 requires exact owner-approved lists. Local and remote deletion approvals must be separate. Any future approved cleanup uses non-force branch-ref deletion only; protected refs and published tags are never included.

A `SAFE_DELETE_CANDIDATE` requires every gate to be proven: it is not `main`, is not checked out or protected, is fully reachable from `main`, has zero unique commits, has understood local and remote state, has no open or draft PR, has no workflow or active-release dependency, is not required by the current Phase, has no unexplained remote-only state or incident evidence, and cannot affect a tag or Release. A branch with unique commits, an open or draft PR, or branch protection cannot be a safe-delete candidate.

## History Policy

Never rewrite `main`, squash or recreate published history for cleanup, or force push. Never remove historical notebooks for branch cleanup. Never modify externally managed refs. Never use garbage collection as a branch-cleanup substitute.

## Recovery

Stop if an unexpected unique commit appears; do not delete first and investigate later. Before a future approved deletion, create a new preservation branch when necessary. Document every deletion result in `SESSION_HANDOFF.md`. Do not recreate a branch at a different SHA under the same historical name without explicit approval.

## Review Cadence

Review branches after a published milestone, before a major new project phase, when branch count becomes operationally confusing, and before enabling automatic deletion settings. No automatic review job is scheduled.

## Project Boundaries

This policy changes no source behavior, configuration, dependency, model, dataset loader, script, notebook, CI workflow, repository metadata, ruleset, tag, or GitHub Release. Canonical training remains unverified, trained weights remain undistributed, and model-performance claims remain unavailable.
