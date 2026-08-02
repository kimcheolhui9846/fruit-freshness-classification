# Governance decisions

This package records owner decisions that are still required. It is informational only and is not legal advice. No repository license has been selected or added in Phase 6.3. No citation file has been created.

## Software license decision

No `LICENSE` file, repository license metadata, source-file license header, notebook license statement, or package license declaration was found in the audit.

| Option | High-level fit | Main tradeoff |
|---|---|---|
| MIT | Permissive reuse with minimal conditions; preserves attribution and license notice | No explicit patent grant |
| Apache-2.0 | Permissive reuse with an explicit patent grant and notice requirements | More notice and license text requirements |
| GPL-3.0 | Reciprocal open-source distribution terms | Derivative distribution has stronger copyleft obligations |

The owner should decide:

1. Should commercial reuse be allowed?
2. Should derivative works remain open source?
3. Is an explicit patent grant desired?
4. Are there third-party code obligations?
5. Is the dataset license compatible with the selected project license?

A license choice must be reviewed against the project dependencies, any future code contributions, the dataset source, and planned model-weight distribution. This document does not determine that compatibility.

## Dataset governance and attribution

| Item | Audit result |
|---|---|
| Dataset | [`Densu341/Fresh-rotten-fruit`](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit) |
| Dataset owner | `Densu341` |
| Public page metadata | Labeled `openrail` in the Hugging Face page/search metadata reviewed during the audit |
| Dataset-card content | The surfaced dataset-card README was empty |
| Access | Publicly accessible through the linked Hugging Face repository at audit time |
| Attribution instructions | Not surfaced in the reviewed card content |
| Redistribution permission | Not determined from the surfaced card content |
| Repository data policy | Dataset archives, extracted images, and caches are not committed |
| Weight implications | Future trained-weight distribution needs a separate terms review |

The Hugging Face page itself should remain the attribution and access reference. Because detailed license text, attribution instructions, and redistribution terms were not available in the card content surfaced during this audit, the repository must not claim project-license compatibility or redistribution permission. Link to the source, keep data out of Git, and obtain owner review before distributing data-derived artifacts.

## Citation decision

| Citation input | Audit result |
|---|---|
| Preferred project title | `Fruit Freshness Classification` |
| Canonical repository URL | `https://github.com/kimcheolhui9846/fruit-freshness-classification` |
| Paper | None found |
| DOI | None found |
| Approved author list | Not supplied |
| Institution or affiliation | Not supplied |
| Versioned release | None; no tag or GitHub Release exists |
| Release date | Not applicable until a release is approved |
| Software license | Not selected |

`CITATION.cff` remains pending owner-approved author and licensing information.

### Citation paths to consider later

- **Repository citation only:** suitable before a paper or DOI exists. It will need an approved title, authors, repository URL, version, release date, and license.
- **Paper citation:** use only when an actual paper exists.
- **Zenodo DOI integration:** optional later archival work after a release and citation identity are approved.

Do not invent an author, affiliation, ORCID, paper, DOI, version, or release date.

## Release and branch-governance decisions

| Decision | Current recommendation | Owner action needed |
|---|---|---|
| First version | Delay a tag; later consider `v0.1.0` prerelease for the engineering milestone | Approve or defer |
| GitHub Release | Do not publish yet | Approve only after governance review |
| Branch protection | Recommend PRs, current CI, no force pushes, no deletion, and up-to-date branches | Choose tradeoff for solo workflow |
| Repository metadata | Improve description and consider topics | Approve exact wording/topics |
| Canonical training | Keep outside release scope unless separately authorized | Approve hardware/time/artifact plan |

No branch protection, ruleset, repository metadata, tag, or GitHub Release was changed in Phase 6.3. See [release readiness](release-readiness.md) and the [release checklist](release-checklist.md) for the operational decision boundaries.
