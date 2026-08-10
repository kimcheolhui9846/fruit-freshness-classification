# Experiment registry

| Experiment ID | Phase | Status | Parent | Holdout boundary |
|---|---:|---|---|---|
| `deep3-canonical-reference-01` | 8 | `CLOSED_REFERENCE` | None | `OBSERVED`; historical evidence only |
| `deep3-postholdout-research-01` | 9.2 | `PROTOCOL_FROZEN` | `deep3-canonical-reference-01` | Historical canonical holdout excluded; new locked test is frozen and unobserved by a model |

Future child runs use: `deep3-postholdout-research-01-baseline`, `deep3-postholdout-research-01-loss-001`, `deep3-postholdout-research-01-aug-001`, `deep3-postholdout-research-01-sampler-001`, `deep3-postholdout-research-01-opt-001`, and `deep3-postholdout-research-01-arch-001`.

No training artifact is created by registration. Each future child remains in this registry regardless of outcome under `NO_RESULT_CHERRY_PICKING: YES`.
## Phase 9.2 frozen state

- baseline: `NOT RUN`
- model experiments: `NOT RUN`
- locked test model evaluation: `NOT RUN`
- historical canonical holdout use: `HISTORICAL_EVIDENCE_ONLY`

The Phase 9 split manifest freezes source-relative indices and hashes before any model-development feedback. Future child experiments remain registered but require a separate explicit Phase authorization before training.