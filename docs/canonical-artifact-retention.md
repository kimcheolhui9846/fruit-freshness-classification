# Canonical artifact retention

All listed artifacts remain local-only; paths are deliberately omitted.

```text
LOCAL_ONLY: YES
REMOTE_BACKUP: NO
PUBLIC_BACKUP: NO
RELEASE_ASSET: NO
ACTIONS_ARTIFACT: NO
```

| Artifact | Class | Retain | Publication | Mutation | Size | SHA-256 |
|---|---|---|---|---|---:|---|
| `run_manifest.json` | metadata | Yes | No | No | 1,379 B | `5906977c6998e3cac8df07356229c7dd68a0551789637e7f7cca45d2b479ba4c` |
| `training_state.pt` | state | Yes | No | No | 217,014,152 B | `00a92f51394bc9a2a2dbb18fa84acb1cb756a0e01e845f64f5817090e7fc03c3` |
| `label_names.json` | metadata | Yes | No | No | 227 B | `c0c229be2509141e1ca3ddf994192b05de69bdec61ea0f97ed554d905eacaae9` |
| `best_model_fold1.pt` | checkpoint | Yes | No | No | 54,253,480 B | `e89a9f7b6128f1c6a8fbd4885d86ee81ca5c0eac4c6601c35fbabcfae5822a24` |
| `best_model_fold2.pt` | checkpoint | Yes | No | No | 54,253,480 B | `19eb264786f339fb738c218b891283ac17a6fa449a6f16ae25b18b33c93299ff` |
| `best_model_fold3.pt` | checkpoint | Yes | No | No | 54,253,480 B | `85a3a2ac5cb373906c81de7a69e223b7c353dedfd51b19ef975340040bd27068` |
| `last_model_weights.pt` | raw checkpoint | Yes | No | No | 54,253,932 B | `3fb0e5575ddc4c6ca2bceb955d17a85fd5965bc325ff5b261dded5dab5cbb29f` |
| `deep3-canonical-reference-01.log` | training log | Yes | No | No | 25,098,708 B | `3c969b7be56b12a267287f3dc9e504275adbae09062d789478b6e252215227d4` |
| `deep3-canonical-reference-01-holdout-cli.log` | evaluation log | Yes | No | No | 18,390 B | `cb4a7ee621fa32a20edc64bd0ad8f7e79f483f7e3d70bcf07336c139edca5e9d` |
| `deep3-canonical-reference-01-holdout-evaluation.json` | summary | Yes | No | No | 7,254 B | `592b88a506d946fcb3b4108f3dacfcd0fe15202b8adeda009f61aeaa29446443` |
| `deep3-canonical-reference-01-holdout-classification-report.csv` | metrics | Yes | No | No | 932 B | `8c8422311120ca75459ad33a9ecd4541415c2011deb5b708ebf7525d4c2b8213` |
| `deep3-canonical-reference-01-holdout-confusion-matrix.csv` | confusion matrix | Yes | No | No | 823 B | `64bbdbc156da4061ccf093a0e51ab6b74706aca441eb34c2debe483797a5d444` |
| `deep3-canonical-reference-01-holdout-predictions.npz` | raw predictions | Yes | No | No | 178,468 B | `f36783f2be1d09bbd7178b734ba70023d54860f018a067d9f9cb1b3794331e0c` |

Retention is `KEEP_LOCAL_ONLY` until an explicit owner change. No deletion, relocation, conversion, packaging, remote backup, or publication is authorized.
