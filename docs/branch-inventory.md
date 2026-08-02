# Phase 7.4 Branch Inventory

## Snapshot Information

- Audit date: 2026-08-03.
- Starting `main` SHA: `d205f2995b6b44345fba1efffa2bb9a6db44fb10`.
- Repository: `kimcheolhui9846/fruit-freshness-classification`.
- Default branch: `main`.
- Initial audit snapshot after local Phase-branch creation and before its first push: 24 local branches, 21 origin-tracking branches excluding `origin/HEAD`, and 21 live GitHub branches.
- Post-implementation CI recheck: 24 local branches, 22 origin-tracking branches, and 22 live GitHub branches. The only remote addition was this Phase branch. Every pre-existing remote branch retained its original SHA; `main` remained at the starting SHA.
- Open pull requests: 0. No historical pull requests were returned by GitHub.
- Active rulesets: `Protect main` (ID `20229405`) for the default branch and `Protect v0.1.0` (ID `20232130`) for the exact protected tag.
- Known limitation: the externally managed Codex temporary ref can make `git fetch origin` fail. GitHub API and `git ls-remote` were used as remote sources of truth; no Codex ref was changed.
- Remote consistency: GitHub branch API and `git ls-remote --heads origin` both returned the same 21 remote branches and SHAs. No stale or remote-only branch was observed.
- Workflow dependency: `.github/workflows/ci.yml` triggers on `main` and wildcard pushes, but names no Phase or release branch as a required deployment, environment, or reusable-workflow source.

## Complete Branch Classification

| Branch | Location | SHA | Main relationship | Unique commits | PR | References | Classification | Recommended action |
|---|---|---|---|---:|---|---|---|---|
| `backup/before-fruit-freshness-switch-20260729` | local only | `a9a6d1d28e35a4cc587860ae09534f5c827e43da` | diverged from main | 15 | no PR | historical: `SESSION_HANDOFF.md` | `RETAIN_UNIQUE_COMMITS` | Retain; unique historical CMT and notebook work is reachable only here. |
| `chore/phase-5.1-environment` | local + remote | `479c36babdcc3ed388fd56e269845095cfb63e2a` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `chore/phase-7.1-main-protection` | local + remote | `5ba51ecc58176e8a97f0668ee0127823256563ad` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md`; release-tag ancestor | `TEMPORARY_RETAIN` | Retain as recent governance evidence. |
| `chore/phase-7.2-release-tag-governance` | local + remote | `ec5b3d8af8d1ab2e72c64f23d0dc8d6144344412` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md`, `docs/tag-governance.md`; release-tag ancestor | `TEMPORARY_RETAIN` | Retain as recent governance evidence. |
| `ci/phase-6.1-repository-health` | local + remote | `c54c0b7bec97d6be001215abf582f9a657bdae4a` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `docs/phase-6.2-readme-usage` | local + remote | `be6e347328f80c423d2358c291257640a8147fd4` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `docs/phase-6.3-release-audit` | local + remote | `243b7ea66d66a3cfd6621ee54ab21e05f9dd557b` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `docs/phase-6.4-license-citation` | local + remote | `ff0b3f7bfcde95ff22893b142062f9746abf319b` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `docs/phase-7.3-repository-metadata` | local + remote | `d205f2995b6b44345fba1efffa2bb9a6db44fb10` | identical to main | 0 | no PR | historical: `docs/repository-metadata.md`; release-tag ancestor | `TEMPORARY_RETAIN` | Retain as recent governance evidence. |
| `docs/phase-7.4-branch-retention-policy` | local + remote after recheck; local only initially | `4aff4a7dea4a5d5f6c0b6349a5f02ac6918fb87b` | ahead of main | 1 | no PR | historical: this inventory after recheck | `RETAIN_ACTIVE_OR_RECENT` | Retain as the current audit branch with one expected policy commit. |
| `feat/phase-5.3-training-entrypoint` | local + remote | `c4e48a746e6117e044f9ff3a2f234551cedbd104` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `feat/phase-5.4-evaluation-inference` | local + remote | `5537a8d45e17e8c727200ae33ac4b8f1188f5d58` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `fix/phase-5.5-dataset-loader-compatibility` | local + remote | `6e6a6198598625945f98cf2b642de02f46b610c5` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `main` | local + remote | `d205f2995b6b44345fba1efffa2bb9a6db44fb10` | identical to main | 0 | no PR | operational: workflow and 58 tracked references; protected | `MANDATORY_RETAIN` | Retain permanently. |
| `refactor/phase-4.10-notebook-orchestration` | local + remote | `4afed76ad7ac1394470bb8033138fbd6edf53569` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `refactor/phase-4.5-losses` | local + remote | `c2eebb18aa4ac49d45f914bdaad172b5c1e2b8e1` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `refactor/phase-4.6-engine-foundations` | local + remote | `0f89baadf8f693c6d0a58e7ba2f064f3894d46c5` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `refactor/phase-4.7-training-loops` | local + remote | `fec42a20faa78faebe47046ed4cabd760ed512b4` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `refactor/phase-4.8-evaluation-metrics` | local + remote | `6e0bccafa7e899106d31ef99eaa147646e083637` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `refactor/phase-4.9-inference-ensemble` | local + remote | `e23d6019fb6fbf0ba6ba82e1f626fec6d81575d8` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `refactor/phase-5.2-experiment-config` | local + remote | `570e91edaccef5e6c62d054a31c52f201e33e78b` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |
| `release/phase-6.5-v0.1.0` | local + remote | `4116da7ac34b703f8412abde8b432c5820382794` | ancestor of main | 0 | no PR | release-related: `SESSION_HANDOFF.md`; contains the release commit | `RETAIN_RELEASE_AUDIT` | Retain until a later milestone is published and verified. |
| `test/phase-5.5-reproducibility` | local only | `5537a8d45e17e8c727200ae33ac4b8f1188f5d58` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain; the failed test name has audit context despite no unique commit. |
| `test/phase-5.5-reproducibility-rerun` | local + remote | `f2fd443186a1d1217ac278590a1b2857b4268e2c` | ancestor of main | 0 | no PR | historical: `SESSION_HANDOFF.md` | `TEMPORARY_RETAIN` | Retain through the policy window. |

Every row has exactly one classification. No remote-only branch, stale origin-tracking mismatch, open or draft PR, workflow dependency, or Phase-branch ruleset dependency was observed.

## Unique-Commit Appendix

Two branches have commits not reachable from `main`. `backup/before-fruit-freshness-switch-20260729` has 15 historical commits reachable only from that local backup branch; no tag, other local branch, or other ref contains them. The current Phase branch has one expected documentation commit, reachable only from its local and matching origin-tracking refs while the Phase remains unmerged.

| Unique commit | Date | Subject | Reachable elsewhere | Recommendation |
|---|---|---|---|---|
| `4aff4a7dea4a5d5f6c0b6349a5f02ac6918fb87b` | 2026-08-03 | docs: define branch retention and cleanup policy | Local and origin Phase refs only | Retain as the active Phase branch. |

The historical backup commits follow.

| Unique commit | Date | Subject | Reachable elsewhere | Recommendation |
|---|---|---|---|---|
| `a9a6d1d28e35a4cc587860ae09534f5c827e43da` | 2026-07-28 | 데이터 검사 | No | Retain backup branch. |
| `ab75386e7e61f28fdd1ddbbc6eac8fe5b69d0da1` | 2026-07-23 | 2차 수정 | No | Retain backup branch. |
| `ee4f412259beaacb8518741fe5e14215dd07a509` | 2026-07-20 | 새로운 모델 | No | Retain backup branch. |
| `5b30475078700bb84a34c1bce8e0d5332044d448` | 2026-07-13 | cmt 모델 수정 5 | No | Retain backup branch. |
| `6abf3e6b05a30a2fab21b30d1a44f232c4085453` | 2026-07-13 | cmt 모델 수정4 | No | Retain backup branch. |
| `b1c2a9e73f50d0b10354e0771c58a94fea362d87` | 2026-07-08 | cmt 모델 수정3 | No | Retain backup branch. |
| `040b6c9e0a632c515c519112a43e0286c12ba631` | 2026-07-08 | cmt모델 수정3 | No | Retain backup branch. |
| `5647934b0642ab61f5eb5e343ace454aa0854062` | 2026-07-08 | cmt 모델 수정3 | No | Retain backup branch. |
| `18a48633af431a1b8389d16f5e9ba60bee7f31a9` | 2026-07-03 | cmt 모델 수정 2 | No | Retain backup branch. |
| `3a20ef1fd28a44136eb32e98e0a654f9a2ee6f2c` | 2026-07-03 | feat: cmt 모델 코드 수정 | No | Retain backup branch. |
| `577035a9c554459b4da5dc1b956ed87714ca763d` | 2026-06-30 | feat: 코드 파일 추가 및 모델 가중치 제외 | No | Retain backup branch. |
| `948995420085208eeea1872efde4973685b7c4d0` | 2026-06-30 | Merge remote-tracking branch `origin/main` | No | Retain backup branch. |
| `30a307205805af80c3275f9a58b2f9d44099a857` | 2026-06-30 | Update notebook | No | Retain backup branch. |
| `52b7d557d48e5bb528f27dbd4dd71378e0bc70a0` | 2026-05-12 | Add files via upload | No | Retain backup branch. |
| `2ea6f365be7b0af99c5e1d5e4882407a8d0699ca` | 2026-05-11 | first commit | No | Retain backup branch. |

## Release and Tag Relationships

The protected annotated `v0.1.0` tag is the canonical release identifier and peels to `b38ebd36f4fa4f1fe012b957095db6dcbce20832`. It is not a branch. `release/phase-6.5-v0.1.0` contains the release commit and is retained for release-audit value even though it has zero unique commits. `main`, the Phase 7.1 through Phase 7.4 governance branches, and the release branch contain the release commit; earlier modernization branches do not. Historical handoff references preserve the release-branch context.

## Safe-Delete Candidates

No deletion is authorized by this document.

No branch is a `SAFE_DELETE_CANDIDATE` in this snapshot. All fully merged Phase branches remain inside the recommended retention window following the 2026-08-02 milestone, the release branch retains current audit value, the backup branch has unique commits, and the current Phase branch is active.

## Retain List

- `main`: mandatory protected default branch.
- `backup/before-fruit-freshness-switch-20260729`: 15 unique commits reachable only from this local backup branch.
- `release/phase-6.5-v0.1.0`: current release-preparation and publication audit context.
- `docs/phase-7.4-branch-retention-policy`: current Phase branch with one expected unmerged policy commit.
- All remaining fully merged modernization and governance branches: temporary retention through the conservative policy window.

## Review and Blocker List

- `REVIEW_REQUIRED`: NONE.
- `BLOCKED_UNVERIFIED`: NONE.
- The known Codex fetch limitation is documented, but independent GitHub API and `git ls-remote` evidence establishes the branch inventory; it does not block this classification.

## Recommended Phase 7.5 Input

```text
LOCAL_BRANCH_DELETE_CANDIDATES:
NONE

REMOTE_BRANCH_DELETE_CANDIDATES:
NONE

MANDATORY_RETAIN:
- main

TEMPORARY_RETAIN:
- chore/phase-5.1-environment
- chore/phase-7.1-main-protection
- chore/phase-7.2-release-tag-governance
- ci/phase-6.1-repository-health
- docs/phase-6.2-readme-usage
- docs/phase-6.3-release-audit
- docs/phase-6.4-license-citation
- docs/phase-7.3-repository-metadata
- feat/phase-5.3-training-entrypoint
- feat/phase-5.4-evaluation-inference
- fix/phase-5.5-dataset-loader-compatibility
- refactor/phase-4.10-notebook-orchestration
- refactor/phase-4.5-losses
- refactor/phase-4.6-engine-foundations
- refactor/phase-4.7-training-loops
- refactor/phase-4.8-evaluation-metrics
- refactor/phase-4.9-inference-ensemble
- refactor/phase-5.2-experiment-config
- test/phase-5.5-reproducibility
- test/phase-5.5-reproducibility-rerun

REVIEW_REQUIRED:
NONE
```

The Phase 7.5 input is a decision package only. It does not authorize a local or remote branch change.
