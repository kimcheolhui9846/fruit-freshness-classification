# Backup Branch Audit

## Scope

- **Audited branch:** `backup/before-fruit-freshness-switch-20260729`
- **Branch SHA:** `a9a6d1d28e35a4cc587860ae09534f5c827e43da`
- **Merge base with `main`:** none; the histories are disconnected.
- **Unique commits relative to `main`:** 15
- **Current-main-only commits:** 63
- **Audit date:** 2026-08-03
- **Method:** Git-object inspection only. The backup branch was not checked out, executed, merged, copied, published, tagged, bundled, renamed, reset, or deleted.
- **Commit metadata:** one display-name author; personal email detected: yes, `[redacted]`.

The branch is local-only. No matching origin or GitHub branch exists. Each of its 15 commits is reachable from this backup branch only among local branches and tags examined during the audit.

## Commit Summary

Every row has sensitive metadata present because the author email is personal and has been redacted. No credential value is reproduced.

| Commit | Date | Subject | High-level purpose | File impact | Sensitive metadata present |
|---|---|---|---|---|---|
| `2ea6f36` | 2026-05-11 | first commit | Initial notebook baseline | +1 / ~0 / -0 | yes — email redacted |
| `52b7d55` | 2026-05-12 | Add files via upload | Added a second notebook copy | +1 / ~0 / -0 | yes — email redacted |
| `30a3072` | 2026-06-30 | Update notebook | Updated notebook and added an extensionless text artifact | +1 / ~1 / -0 | yes — email redacted |
| `9489954` | 2026-06-30 | Merge remote-tracking branch | Two-parent merge; no selected-parent file delta | merge commit | yes — email redacted |
| `577035a` | 2026-06-30 | feature: add code files and exclude model weights | Added notebooks and historical ignore policy | +5 / ~0 / -0 | yes — email redacted |
| `3a20ef1` | 2026-07-03 | feature: revise CMT model code | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `18a4863` | 2026-07-03 | CMT model revision 2 | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `5647934` | 2026-07-08 | CMT model revision 3 | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `040b6c9` | 2026-07-08 | CMT model revision 3 | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `b1c2a9e` | 2026-07-08 | CMT model revision 3 | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `6abf3e6` | 2026-07-13 | CMT model revision 4 | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `5b30475` | 2026-07-13 | CMT model revision 5 | CMT notebook iteration | +0 / ~1 / -0 | yes — email redacted |
| `ee4f412` | 2026-07-20 | new model | Added convolutional-model notebook | +1 / ~0 / -0 | yes — email redacted |
| `ab75386` | 2026-07-23 | second revision | Revised convolutional-model notebook | +0 / ~1 / -0 | yes — email redacted |
| `a9a6d1d` | 2026-07-28 | data inspection | Added data-inspection notebook | +1 / ~0 / -0 | yes — email redacted |

There is one merge commit (`9489954`) and no commit deletes a tracked path. Commit subjects describe notebook-oriented experimentation rather than the current modular repository structure.

## Tree Summary

The backup tip contains 10 tracked paths, 8 distinct blobs, and one root-level directory layout. It contains 8 notebooks, 1 configuration file, and 1 unclassified extensionless text file. There are no tracked source modules, datasets, images, weights, checkpoints, archives, executables, or binary files by extension.

| Path | Blob | Bytes | Extension | Classification | Exists in current `main` | Same blob | Review requirement |
|---|---|---:|---|---|---|---|---|
| `.gitignore` | `c015f01` | 249 | `.gitignore` | configuration | yes | no | historical policy review |
| `33.ipynb` | `1a80344` | 27,656 | `.ipynb` | notebook | no | no | privacy review |
| `Untitled-1.ipynb` | `3cbda41` | 2,839,288 | `.ipynb` | notebook | no | no | privacy and size review |
| `Untitled-2.ipynb` | `3cbda41` | 2,839,288 | `.ipynb` | notebook | no | no | privacy and size review |
| `Untitled-2_backup.ipynb` | `3cbda41` | 2,839,288 | `.ipynb` | notebook | no | no | privacy and size review |
| `cmt 모델.ipynb` | `4675eb1` | 91,014 | `.ipynb` | notebook | no | no | privacy review |
| `conv.ipynb` | `d11143e` | 34,714 | `.ipynb` | notebook | no | no | privacy review |
| `da` | `f47c4ff` | 1,434 | none | unknown text-like artifact | no | no | manual content review |
| `데이터 정제 파이프라인.ipynb` | `6b194ad` | 25,758 | `.ipynb` | notebook | no | no | privacy and destructive-operation review |
| `크롤링(노이즈제거모델)-1.ipynb` | `db7d9f8` | 2,132,783 | `.ipynb` | notebook | no | no | privacy and size review |

The `da` file has no NUL byte and does not structurally resemble a script or JSON document, but its purpose is unresolved. It must not be assumed safe for redistribution.

## Notebook Findings

Notebook inspection was JSON-structural only. **Execution performed: No.** No notebook output was cleared or rewritten.

| Notebook | Cells | Code cells | Cells with outputs | Executed cells | Embedded image MIME entries | Local-path signal | Keyword-only credential signal |
|---|---:|---:|---:|---:|---:|---|---|
| `33.ipynb` | 2 | 2 | 2 | 1 | 0 | yes | service-token keyword |
| `Untitled-1.ipynb` | 14 | 14 | 14 | 14 | 0 | yes | service-token and session keyword |
| `Untitled-2.ipynb` | 14 | 14 | 14 | 14 | 0 | yes | service-token and session keyword |
| `Untitled-2_backup.ipynb` | 14 | 14 | 14 | 14 | 0 | yes | service-token and session keyword |
| `cmt 모델.ipynb` | 5 | 5 | 1 | 0 | 0 | yes | no |
| `conv.ipynb` | 2 | 2 | 0 | 0 | 0 | yes | no |
| `데이터 정제 파이프라인.ipynb` | 1 | 1 | 1 | 1 | 0 | yes | no |
| `크롤링(노이즈제거모델)-1.ipynb` | 5 | 5 | 5 | 4 | 0 | yes | no |

No embedded data URI, content email, or telephone-number pattern was found in these tip notebooks. Local-path signals occur in every notebook. Notebook outputs remain a privacy and reproducibility concern even though no embedded image MIME entry was found.

## Difference from Main

**Project relevance classification: `RELATED_RESEARCH_HISTORY`.**

The backup and current `main` have no common ancestor. A direct tree comparison reports 9 backup-only paths, one changed `.gitignore`, and 122 paths that exist only in current `main`. No backup-tip blob is shared with current `main` at the same path.

Evidence of related research exists: fruit-related terms occur in three historical notebooks, one notebook references Hugging Face dataset loading, one is CMT-focused, and three are crawler or noise-related. However, there is no exact fruit-freshness phrase, no shared Git ancestry, no shared project layout, and no shared source artifact. The history is therefore related experimentation, not a proven earlier version of the current modular project. No file is approved for selective recovery.

## Security Findings

- **Confirmed secrets:** 0. No recognized credential-format pattern, private-key marker, credential-like tracked filename, or credential file was found across the 15 commit snapshots.
- **Possible secrets:** 38 commit-and-path keyword matches. They are limited to service-token or session-related keywords in historical notebooks. Values are not reproduced, were not tested, and require manual review before any external preservation or publication.
- **Personal information:** personal email detected in commit metadata for all 15 commits and redacted here. No content email or telephone pattern was found.
- **Local paths:** 76 commit-and-path matches across historical notebook snapshots; every tip notebook has a local-path signal. Values are redacted.
- **Sensitive filenames:** none detected.
- **Executable safety:** no executable-mode file, shell script, PowerShell script, compiled binary, archive, or history-rewrite signal was found. One filesystem-removal pattern appears in `데이터 정제 파이프라인.ipynb`; it was not executed and requires manual review.

The remediation is to keep the branch private and unchanged, do not publish it, and require owner review before any external copy, bundle, extraction, or migration.

## Large Objects

The unique history contains 17 blob objects. Two exceed 1 MiB; none exceeds 10 MiB. Neither crosses GitHub's normal file-size warning threshold, but both are large notebook blobs with privacy and repository-bloat risk.

| Rank | Blob | Size | Path | Commit introduction | Category | Publication risk |
|---:|---|---:|---|---|---|---|
| 1 | `3cbda41` | 2,839,288 B | `Untitled-1.ipynb` | `52b7d55` | notebook | output bloat and privacy review |
| 2 | `db7d9f8` | 2,132,783 B | `크롤링(노이즈제거모델)-1.ipynb` | `577035a` | notebook | output bloat and privacy review |
| 3 | `a0a9352` | 94,123 B | `cmt 모델.ipynb` | `6abf3e6` | notebook | low size risk |
| 4 | `4675eb1` | 91,014 B | `cmt 모델.ipynb` | `5b30475` | notebook | low size risk |
| 5 | `e67078d` | 70,524 B | `cmt 모델.ipynb` | `b1c2a9e` | notebook | low size risk |
| 6 | `5bd0d14` | 67,466 B | `cmt 모델.ipynb` | `5647934` | notebook | low size risk |
| 7 | `e62bc24` | 67,466 B | `cmt 모델.ipynb` | `040b6c9` | notebook | low size risk |
| 8 | `307661e` | 50,158 B | `cmt 모델.ipynb` | `18a4863` | notebook | low size risk |
| 9 | `d11143e` | 34,714 B | `conv.ipynb` | `ab75386` | notebook | low size risk |
| 10 | `f7fdd0d` | 32,728 B | `cmt 모델.ipynb` | `3a20ef1` | notebook | low size risk |
| 11 | `1a80344` | 27,656 B | `33.ipynb` | `a9a6d1d` | notebook | low size risk |
| 12 | `6b194ad` | 25,758 B | `데이터 정제 파이프라인.ipynb` | `577035a` | notebook | low size risk |
| 13 | `852d73e` | 17,380 B | `cmt 모델.ipynb` | `577035a` | notebook | low size risk |
| 14 | `4d5408f` | 13,544 B | `conv.ipynb` | `ee4f412` | notebook | low size risk |
| 15 | `2002b6e` | 7,456 B | `Untitled-1.ipynb` | `2ea6f36` | notebook | low size risk |
| 16 | `f47c4ff` | 1,434 B | `da` | `30a3072` | unknown | manual review |
| 17 | `c015f01` | 249 B | `.gitignore` | `577035a` | configuration | low size risk |

## Data and Artifact Findings

No raw dataset, image collection, weight, checkpoint, serialized model, archive, cache, or generated result file is tracked at the backup tip. No archive or binary is present by extension.

Notebook code references model or checkpoint-style filenames in four notebooks and a dataset-style filename in one notebook. The historical ignore policy also mentions data or weight-style paths. These are references only, not tracked artifacts. Dataset provenance and redistribution terms are incomplete; crawler or noise-related notebook content creates an additional disclosure review requirement. Public redistribution is therefore unknown and unapproved.

## License Findings

No license, notice, copyright text, attribution statement, dependency lockfile, or environment descriptor exists in the backup tip. There is no evidence that the current repository's MIT license applies retroactively to this disconnected history. Ownership, third-party attribution, and dataset terms are unresolved; compatibility with the current MIT project is **UNKNOWN**.

## Public Publication Gate

| Gate | Result | Evidence |
|---|---|---|
| No confirmed credential value | PASS | No credential-format or private-key marker found. |
| No private personal information | FAIL | Personal commit email and local-path signals require redaction outside this audit. |
| No restricted dataset content | UNKNOWN | No raw dataset is tracked, but provenance and crawler-related terms are unresolved. |
| No unapproved weights or checkpoints | PASS | No model artifact is tracked. |
| No problematic large blob | PASS | No blob exceeds 10 MiB; two notebooks exceed 1 MiB and remain review items. |
| Clear copyright ownership | UNKNOWN | No license, notice, or attribution evidence. |
| License compatibility understood | UNKNOWN | Disconnected historical content has no applicable documented license. |
| Relevant to current repository | FAIL | Classified `RELATED_RESEARCH_HISTORY`, not a proven current-project predecessor. |
| No unrelated-project contamination | FAIL | Disconnected layout and history contain separate experimental material. |
| Commit metadata acceptable for publication | FAIL | Personal email metadata would be disclosed by public push. |
| Owner explicitly approves public disclosure | OWNER_APPROVAL_REQUIRED | No owner approval is supplied in this phase. |

## Audit Limitations

This is a read-only Git-native audit. It does not prove the absence of every secret, establish dataset licenses, test credentials, contact any service, OCR images, execute notebooks, execute scripts, extract archives, or sanitize historical content. No external secret scanner was installed. These limitations require owner review before any preservation action beyond retaining the local branch.

`Protect main` remains unchanged. `v0.1.0` protection remains unchanged. This audit makes no source or model-performance claims or changes.
