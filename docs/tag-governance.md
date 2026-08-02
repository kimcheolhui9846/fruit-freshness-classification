# Published tag governance

## Purpose

The published `v0.1.0` release tag is part of the repository's public release identity. It is an annotated tag for a source-only prerelease engineering milestone and must remain immutable. This policy protects the already-published tag without representing canonical training, trained-checkpoint evaluation, benchmark reproduction, or generic inference as complete.

## Protected tag

The only protected tag condition is `refs/tags/v0.1.0`. Its immutable peeled release commit is `b38ebd36f4fa4f1fe012b957095db6dcbce20832`.

## Approved rules

The active tag ruleset blocks tag deletion and non-fast-forward updates. It has no bypass actors and applies to this exact tag only.

- Tag creation is not restricted because `v0.1.0` already exists.
- Required status checks are absent for this tag.
- No signed-tag rule or signed-commit rule is configured.
- Future tags are explicitly unaffected.
- The separate `Protect main` branch ruleset remains unchanged.

## Why the rule is exact

Broad version wildcard patterns, such as `v*`, are explicitly rejected because they could lock an accidentally created future tag before its release has been audited. Every future release tag must first complete its own release process. A later exact tag rule may be added through a separate audited ruleset update or a separate exact ruleset; `v0.1.0` protection must not be generalized automatically.

## Normal future release workflow

1. Prepare a new version on a release branch.
2. Pass branch CI.
3. Fast-forward the verified commit into protected `main`.
4. Pass `main` CI.
5. Create and verify the new annotated tag.
6. Publish and verify the GitHub Release.
7. Add exact tag protection only after publication.
8. Record the new tag ruleset or audited ruleset update.

## Recovery policy

Do not delete the ruleset merely to move a published release tag, do not force-update a published tag, and do not reuse an existing version number. Publish a corrective release under a new approved version, such as `v0.1.1` or `v0.2.0`, when appropriate. A ruleset repair is allowed only for a proven configuration error, and every administrative exception must be recorded in `SESSION_HANDOFF.md`.

## Non-destructive verification policy

Do not delete `v0.1.0`, attempt a force update, or move the tag temporarily. Verify enforcement through ruleset API readback and immutable ref comparison; do not use a destructive enforcement test.

## Main protection boundary

`Protect main` is a separate branch ruleset and must not be changed by Phase 7.2. This Phase does not add pull-request requirements, signed commits, broad future-tag protection, or tag-creation restrictions.

## Release evidence boundary

The protected release remains a prerelease engineering milestone. Canonical training remains unverified, and trained weights remain undistributed. Dataset and trained-weight governance remain separate decisions.

## Live ruleset record

The exact live ruleset identifier, API readback, tag/ref integrity evidence, and protected-main fast-forward evidence are added after the approved ruleset is created and verified.
