# Canonical Training Artifacts

The following completed-run artifacts are retained locally. Their exact local storage path is intentionally omitted.

| Artifact | Purpose | Size | SHA-256 | Tracked | Published |
|---|---|---:|---|---|---|
| `run_manifest.json` | Portable immutable run identity | 1,379 B | `5906977c6998e3cac8df07356229c7dd68a0551789637e7f7cca45d2b479ba4c` | No | No |
| `training_state.pt` | Completed trusted epoch-boundary operational state | 217,014,152 B | `00a92f51394bc9a2a2dbb18fa84acb1cb756a0e01e845f64f5817090e7fc03c3` | No | No |
| `label_names.json` | Preserved ordered 14-class label contract | 227 B | `c0c229be2509141e1ca3ddf994192b05de69bdec61ea0f97ed554d905eacaae9` | No | No |
| `best_model_fold1.pt` | EMA best checkpoint for fold 1 | 54,253,480 B | `e89a9f7b6128f1c6a8fbd4885d86ee81ca5c0eac4c6601c35fbabcfae5822a24` | No | No |
| `best_model_fold2.pt` | EMA best checkpoint for fold 2 | 54,253,480 B | `19eb264786f339fb738c218b891283ac17a6fa449a6f16ae25b18b33c93299ff` | No | No |
| `best_model_fold3.pt` | EMA best checkpoint for fold 3 | 54,253,480 B | `85a3a2ac5cb373906c81de7a69e223b7c353dedfd51b19ef975340040bd27068` | No | No |
| `last_model_weights.pt` | Final raw model checkpoint | 54,253,932 B | `3fb0e5575ddc4c6ca2bceb955d17a85fd5965bc325ff5b261dded5dab5cbb29f` | No | No |
| `deep3-canonical-reference-01.log` | External execution log | 25,098,708 B | `3c969b7be56b12a267287f3dc9e504275adbae09062d789478b6e252215227d4` | No | No |

Local-only: Yes.
Tracked: No.
Published: No.

All artifacts are ignored, unstaged, uncommitted, not uploaded, and not attached to a Release. Checkpoints passed strict CPU loading with finite tensors and the fixed synthetic output contract. Holdout evaluation is pending. Deletion or renaming is prohibited until Phase 8.4 and a separate artifact-governance review.

CI checkpoint requirement: No.
CI CUDA requirement: No.
CI production dataset access: No.
Release creation: No.