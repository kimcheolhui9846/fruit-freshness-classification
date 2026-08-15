# Backup Branch Preservation Decision

## Approved Phase 7.6 Decision

Phase 7.6 applies the repository owner's explicit preservation decision as a documentation-only record. The Phase 7.5 audit findings remain unchanged: `backup/before-fruit-freshness-switch-20260729` is `RELATED_RESEARCH_HISTORY`, has unresolved provenance and ownership concerns, and must not be publicly disclosed.

```text
APPROVED_BACKUP_ACTION:
KEEP_LOCAL_ONLY

APPROVED_PUBLIC_DISCLOSURE:
NO

APPROVED_ARCHIVE_LOCATION_POLICY:
NOT_APPLICABLE

APPROVED_ENCRYPTION_POLICY:
NOT_APPLICABLE

APPROVED_RETENTION_PERIOD:
PERMANENT

APPROVED_BRANCH_DELETION_AFTER_PRESERVATION:
NO

APPROVED_REMOTE_PUBLICATION:
NO

APPROVED_PRIVATE_REPOSITORY_CREATION:
NO

APPROVED_GIT_BUNDLE_CREATION:
NO

APPROVED_SELECTED_CONTENT_EXTRACTION:
NO

APPROVED_HISTORY_REWRITE:
NO
```

## Retained Local Backup

The local-only branch remains the sole preservation reference for the disconnected history:

- Branch: `backup/before-fruit-freshness-switch-20260729`
- Required SHA: `a9a6d1d28e35a4cc587860ae09534f5c827e43da`
- Total reachable commits: 15
- Unique commits relative to `main`: 15
- Disconnected from `main`: yes
- Remote backup branch: absent
- Tag pointing to the backup history: absent

The branch is retained permanently, is not renamed, reset, deleted, merged, cherry-picked, tagged, or pushed. No archive, encrypted archive, plaintext Git bundle, selected-content extraction, public copy, or private repository is created in this Phase.

## Risk Acceptance and Reconsideration

The owner accepts the remaining single-machine loss risk of `KEEP_LOCAL_ONLY`. No archive destination, encryption tool, encryption mode, passphrase, password hint, recovery key, or storage path is required or recorded because archive creation is not approved.

This decision may be reconsidered only through a new explicit owner-approved preservation Phase. A later approval must independently define its permitted action and safeguards; it must not infer permission from this record.

## Continuing Prohibitions

- Public disclosure and remote publication remain prohibited.
- Private-repository creation remains prohibited.
- Git bundle and archive creation remain prohibited.
- Backup branch deletion, rename, reset, merge, cherry-pick, tag creation, and history rewrite remain prohibited.
- Historical notebooks, outputs, paths, and blobs must not be copied into `main`.
- The Phase 7.5 audit findings, the protected nested repository, and externally managed references remain untouched.

## Governance Boundary

`Protect main` remains unchanged. `v0.1.0` protection remains unchanged. The annotated release tag and its prerelease remain unchanged. This decision makes no source or model-performance claims or changes, and changes no dataset loader, configuration, dependency, script, notebook, model, result, or model-performance claim.
