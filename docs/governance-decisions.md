# Governance decisions

This document records the Phase 6.3 audit history and the owner-approved governance decisions applied in Phase 6.4. It is informational only and is not legal advice.

## Resolved software license

MIT was explicitly selected during Phase 6.4. The repository did not contain a software license before this explicit decision; this is a new licensing decision, not a restoration claim.

| Item | Approved value |
|---|---|
| Software license | MIT |
| SPDX identifier | MIT |
| Copyright year | 2025 |
| Copyright holder | 김철희 |
| Canonical file | [`LICENSE`](../LICENSE) |

The repository software and project-authored documentation are licensed under the MIT License. The canonical file uses the approved copyright line `Copyright (c) 2025 김철희` and unmodified MIT terms. This software license does not determine the terms of the external dataset or any future trained-weight distribution.

## Dataset and trained-weight governance

| Item | Current boundary |
|---|---|
| Dataset | [`Densu341/Fresh-rotten-fruit`](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit) |
| Dataset owner | `Densu341` |
| Public page metadata | Labeled `openrail` in the Hugging Face page/search metadata reviewed during the audit |
| Dataset-card content | The surfaced dataset-card README was empty |
| Dataset terms | Governed by the terms supplied by the original external source |
| Dataset redistribution | Dataset contents are not redistributed through this repository |
| Weight distribution | Trained weights are not currently distributed and require a separate review before publication |

The MIT License applies to repository software and project-authored documentation only. The repository does not claim that the external dataset metadata is compatible with MIT, permission to redistribute external images, or automatic permission to distribute future trained weights. See the [dataset documentation](dataset.md) for dataset identity, source revision, and project data-handling boundaries.

## Resolved repository-only citation

Repository-only citation was explicitly selected during Phase 6.4. [`CITATION.cff`](../CITATION.cff) contains truthful software metadata without claiming a paper, DOI, release, or artifact distribution.

| Citation input | Approved value |
|---|---|
| Project title | `Fruit Freshness Classification` |
| Citation policy | Repository-only citation |
| Citation author | Choelhui Kim |
| Given names | `Choelhui` |
| Family names | `Kim` |
| Author email | Omitted by owner decision |
| Author affiliation | Omitted |
| Author ORCID | Omitted |
| Additional authors | None |
| Repository URL | `https://github.com/kimcheolhui9846/fruit-freshness-classification` |
| Citation license | MIT |
| Citation version and release date | Deferred to Phase 6.5 |
| Paper and DOI | Unavailable |

No version, release date, DOI, paper citation, affiliation, ORCID, additional author, or email field was added to `CITATION.cff`. A future versioned release may update citation metadata only with separate Phase 6.5 authorization.

## Remaining release and repository decisions

| Decision | Current state | Owner action needed |
|---|---|---|
| Version tag | Pending; no tag exists | Approve or defer the proposed `v0.1.0` engineering milestone |
| GitHub Release | Pending; no Release exists | Approve prerelease or normal-release policy and final notes |
| Release date | Pending | Approve only with a release action |
| Dataset redistribution | Pending source-terms review | Confirm attribution and redistribution boundary |
| Trained-weight distribution | Pending separate review | Confirm applicable terms before publication |
| Branch protection | Not configured | Choose the solo-workflow tradeoff separately |
| Repository metadata | Unchanged | Approve exact wording/topics separately |
| Canonical training | Not run | Authorize hardware, time, and artifact plan separately |

No Git tag, GitHub Release, repository setting, branch protection, ruleset, DOI, dataset copy, or trained weight was created in Phase 6.4.
