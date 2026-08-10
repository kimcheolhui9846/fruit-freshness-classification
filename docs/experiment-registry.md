# Experiment registry

| Experiment ID | Phase | Status | Parent | Holdout boundary |
|---|---:|---|---|---|
| `deep3-canonical-reference-01` | 8 | `CLOSED_REFERENCE` | None | `OBSERVED`; historical evidence only |
| `deep3-postholdout-research-01` | 9.1 | `PLANNING` | `deep3-canonical-reference-01` | New untouched evaluation required |

Future child runs use: `deep3-postholdout-research-01-baseline`, `deep3-postholdout-research-01-loss-001`, `deep3-postholdout-research-01-aug-001`, `deep3-postholdout-research-01-sampler-001`, `deep3-postholdout-research-01-opt-001`, and `deep3-postholdout-research-01-arch-001`.

No training artifact is created by registration. Each future child remains in this registry regardless of outcome under `NO_RESULT_CHERRY_PICKING: YES`.