# Canonical Holdout Evaluation Artifacts

The Phase 8.4 outputs below are retained locally through Phase 8.6. Exact local paths are intentionally omitted. Every artifact is ignored by Git, unstaged, untracked, uncommitted, and unpublished.

| Artifact | Purpose | Size | SHA-256 | Local-only | Ignored | Tracked | Published |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `holdout-cli.log` | Canonical CLI stdout/stderr capture | 18,390 B | `cb4a7ee621fa32a20edc64bd0ad8f7e79f483f7e3d70bcf07336c139edca5e9d` | Yes | Yes | No | No |
| `holdout-evaluation.json` | Full-precision evaluation identity and aggregate metrics | 7,254 B | `592b88a506d946fcb3b4108f3dacfcd0fe15202b8adeda009f61aeaa29446443` | Yes | Yes | No | No |
| `holdout-classification-report.csv` | Full-precision per-class metric table | 932 B | `8c8422311120ca75459ad33a9ecd4541415c2011deb5b708ebf7525d4c2b8213` | Yes | Yes | No | No |
| `holdout-confusion-matrix.csv` | Aggregated true-by-predicted class counts | 823 B | `64bbdbc156da4061ccf093a0e51ab6b74706aca441eb34c2debe483797a5d444` | Yes | Yes | No | No |
| `holdout-predictions.npz` | Local labels, predictions, and raw averaged TTA logits | 178,468 B | `f36783f2be1d09bbd7178b734ba70023d54860f018a067d9f9cb1b3794331e0c` | Yes | Yes | No | No |

Local-only: Yes

Ignored: Yes

Tracked: No

Published: No

Raw logits and predictions published: No

Checkpoint publication: No

Weight publication: No

Dataset publication: No

Training-state publication: No

Execution-log publication: No

GitHub Actions artifact upload: No

Release asset upload: No

Release creation: No

Tag creation: No

The prediction archive is never copied into tracked Markdown, GitHub releases, or CI artifacts. Deletion, rename, relocation, or any publication of these artifacts requires a later explicit artifact-governance approval. Retained locally through Phase 8.6.

The committed documentation is an offline contract only. CI checkpoint requirement: No. CI CUDA requirement: No. CI production dataset access: No. CI local evaluation output requirement: No. CI neither loads the local ensemble nor reruns production evaluation.

Phase 8.5 documentation and publication decision is recorded. Phase 8.6 remains owner-gated for any new artifact-retention or publication decision.