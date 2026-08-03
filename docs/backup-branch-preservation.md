# Backup Branch Preservation Decision

## Current Risk

`backup/before-fruit-freshness-switch-20260729` is local-only and is the sole retained branch reference for 15 unique commits. Its SHA is `a9a6d1d28e35a4cc587860ae09534f5c827e43da`. Loss of this local clone could make the disconnected history difficult to recover.

The audit also found personal commit metadata, local-path signals in every notebook, keyword-only service or session references, unresolved ownership, unresolved dataset provenance, notebook outputs, and one unreviewed filesystem-removal pattern. These findings make public disclosure inappropriate.

## Primary Recommendation

```text
PRIMARY_RECOMMENDATION:
REVIEW_REQUIRED
```

A preservation action must remain owner-gated. The history is related research rather than a proven earlier version of the current project, and its public-publication gate has failures and unknowns. No evidence authorizes a public push, merge, tag, bundle, rewrite, or deletion.

## Secondary Recommendation

```text
SECONDARY_RECOMMENDATION:
KEEP_LOCAL_ONLY
```

Keep the existing local-only branch unchanged while the owner decides whether a privacy-reviewed offline preservation copy is needed. This avoids disclosure and preserves commit context, but it does not remove the single-machine loss risk.

## Preservation Options

| Option | Suitable now | Risk | Recommendation |
|---|---|---|---|
| Keep local only | yes, as an interim state | single-machine loss and limited recoverability | secondary recommendation |
| Create offline Git bundle | not yet | privacy, encryption, storage, and integrity policy unresolved | only after explicit owner approval and full content review |
| Push to current public repository | no | public disclosure, privacy, licensing, and portfolio contamination | prohibited |
| Push to separate private repository | not yet | access-control and ownership decision required | possible future option after approval |
| Extract selected non-sensitive content | not yet | can lose context and needs attribution review | possible future option after approval |

## Prohibited Actions Until Approval

- Backup branch deletion is prohibited.
- Public push is prohibited without owner approval.
- Merge into `main` is prohibited.
- Cherry-pick from this history is prohibited.
- Tag creation that points to this history is prohibited.
- History rewrite or sanitization is prohibited.
- Unreviewed bundle creation is prohibited.
- Any branch rename, reset, publication, extraction, or new repository creation is prohibited.

## Proposed Owner Approval Block

All fields remain unresolved and must be supplied by the repository owner.

```text
APPROVED_BACKUP_ACTION:
[owner input required]

APPROVED_PUBLIC_DISCLOSURE:
[owner input required]

APPROVED_ARCHIVE_LOCATION_POLICY:
[owner input required]

APPROVED_ENCRYPTION_POLICY:
[owner input required]

APPROVED_RETENTION_PERIOD:
[owner input required]

APPROVED_BRANCH_DELETION_AFTER_PRESERVATION:
[owner input required]
```

Acceptable action choices are `KEEP_LOCAL_ONLY`, `CREATE_OFFLINE_GIT_BUNDLE`, `PUSH_TO_CURRENT_PUBLIC_REPOSITORY`, `PUSH_TO_SEPARATE_PRIVATE_REPOSITORY`, `EXTRACT_SELECTED_NON_SENSITIVE_CONTENT`, or `DEFER`. Public disclosure must not be approved unless every failed or unknown publication gate is independently resolved.

## Governance Boundary

`Protect main` remains unchanged. `v0.1.0` protection remains unchanged. The annotated release tag and its prerelease remain unchanged. This decision changes no source behavior, dataset loader, configuration, dependency, script, notebook, model, result, or model-performance claim.

## Next Execution Phase

The repository is ready only for **Phase 7.6 — Apply the Approved Backup Preservation Action** after explicit owner approval. Phase 7.6 must use the exact approved values above and must not infer them from this audit.
