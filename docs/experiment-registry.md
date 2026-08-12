# Experiment registry

| Experiment ID | Phase | Status | Parent | Holdout boundary |
|---|---:|---|---|---|
| `deep3-canonical-reference-01` | 8 | `CLOSED_REFERENCE` | None | `OBSERVED`; historical evidence only |
| `deep3-postholdout-research-01` | 9.2 | `PROTOCOL_FROZEN` | `deep3-canonical-reference-01` | Historical canonical holdout excluded; new locked test is frozen and unobserved by a model |
| `deep3-postholdout-research-01-baseline` | 9.3 | `PRE_TRAINING_CV_FROZEN` | `deep3-postholdout-research-01` | Development CV only; locked test and canonical holdout are model-inaccessible |

Future child runs use: `deep3-postholdout-research-01-baseline`, `deep3-postholdout-research-01-loss-001`, `deep3-postholdout-research-01-aug-001`, `deep3-postholdout-research-01-sampler-001`, `deep3-postholdout-research-01-opt-001`, and `deep3-postholdout-research-01-arch-001`.

No training artifact is created by registration. Each future child remains in this registry regardless of outcome under `NO_RESULT_CHERRY_PICKING: YES`.
## Phase 9.2 frozen state

- baseline: `NOT RUN`
- model experiments: `NOT RUN`
- locked test model evaluation: `NOT RUN`
- historical canonical holdout use: `HISTORICAL_EVIDENCE_ONLY`

The Phase 9 split manifest freezes source-relative indices and hashes before any model-development feedback. Future child experiments remain registered but require a separate explicit Phase authorization before training.
## Phase 9.3 baseline authorization

```text
OWNER_PHASE_9_3_APPROVAL:
APPROVED
EXPERIMENT_ID:
deep3-postholdout-research-01-baseline
ROLE:
POST_HOLDOUT_DEVELOPMENT_BASELINE
BASELINE_EXECUTION_STATUS:
NOT_YET_RUN
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
LOCKED_TEST_MODEL_ACCESS:
NO
CANONICAL_HOLDOUT_MODEL_ACCESS:
NO
```

The baseline is a controlled parent for later Phase 9 candidates, not a result-selected run. Any baseline artifacts are local-only. Phase 9 research remains active.

## Phase 9.4 runbook preparation

The baseline execution procedure is documented in [postholdout-baseline-runbook.md](postholdout-baseline-runbook.md). The runbook does not authorize execution.

```text
PHASE_9_4:
RUNBOOK_PREPARED
BASELINE_EXECUTION_STATUS:
NOT_YET_RUN
PHASE_9_5:
NOT STARTED
```

Phase 9.5 is the first loss/class-imbalance experiment. It remains unstarted and unauthorized.