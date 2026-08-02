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

## Live ruleset evidence

- **Phase context:** Started from `main` SHA `5ba51ecc58176e8a97f0668ee0127823256563ad` on `chore/phase-7.2-release-tag-governance`. Implementation commit `9568b2b377de4946bfcf9e17201684da6fa5bf00` passed `Repository CI` run `30745036512` on both required jobs.
- **Ruleset:** `Protect v0.1.0` (ID `20232130`, node ID `RRS_lACqUmVwb3NpdG9yec48FxsWzgE0t8I`), source type `Repository`, target `tag`, active enforcement.
- **Exact scope:** The include condition is only `refs/tags/v0.1.0`; exclusions are empty. The complete rule list is `deletion` and `non_fast_forward`. `bypass_actors` is an empty list, and no creation, update, status-check, signature, or future-tag rule exists.
- **Creation and readback:** Created at `2026-08-02T20:11:35.508+09:00` and read back at `2026-08-02T20:11:35.524+09:00`. The ruleset detail and tag-target ruleset list match the approved design. The temporary payload was stored outside the repository, contained no credentials, and was removed after readback. No ruleset configuration correction was required.
- **Immutable-tag evidence:** The tag object remains `1044e6523a501fe82f5b59667c320ee2ec59eb89`; its peeled commit remains `b38ebd36f4fa4f1fe012b957095db6dcbce20832`; its annotation remains `Fruit Freshness Classification v0.1.0 engineering milestone`.
- **Release evidence:** The associated GitHub Release remains the published source-only prerelease engineering milestone, is not a draft, and has zero uploaded assets.
- **Boundary verification:** `Protect main` remains the separate active branch ruleset with its Phase 7.1 rule set unchanged. No classic branch protection, broad tag rule, future-tag rule, tag movement, tag deletion, force update, destructive enforcement test, or Release mutation was performed.
- **Next protected update:** The evidence handoff commit must pass both required checks on the Phase branch before it is fast-forwarded into protected `main` through a normal push. The known external Codex temporary-ref limitation remains untouched; GitHub SHA comparison is the fallback if a fetch fails.
