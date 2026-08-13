# Experiment registry

| Experiment ID | Phase | Status | Parent | Holdout boundary |
|---|---:|---|---|---|
| `deep3-canonical-reference-01` | 8 | `CLOSED_REFERENCE` | None | `OBSERVED`; historical evidence only |
| `deep3-postholdout-research-01` | 9.2 | `PROTOCOL_FROZEN` | `deep3-canonical-reference-01` | Historical canonical holdout excluded; new locked test is frozen and unobserved by a model |
| `deep3-postholdout-research-01-baseline` | 9.3 | `COMPLETED_DEVELOPMENT_BASELINE` | `deep3-postholdout-research-01` | Development CV only; locked test and canonical holdout are model-inaccessible |

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
COMPLETED
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
LOCKED_TEST_MODEL_ACCESS:
NO
CANONICAL_HOLDOUT_MODEL_ACCESS:
NO
```

The baseline is a controlled parent for later Phase 9 candidates, not a result-selected run. Any baseline artifacts are local-only. Phase 9 research remains active.

## Phase 9.4 baseline execution

The baseline execution procedure is documented in [postholdout-baseline-runbook.md](postholdout-baseline-runbook.md). The owner resolved that runbook's approval block on 2026-08-12, and the run and its development-only OOF evaluation completed on 2026-08-13.

```text
PHASE_9_4:
BASELINE_EXECUTED
BASELINE_EXECUTION_STATUS:
COMPLETED
DEVELOPMENT_OOF_MACRO_F1:
0.9012
LOCKED_TEST_MODEL_ACCESS:
NO
CANONICAL_HOLDOUT_MODEL_ACCESS:
NO
BASELINE_ARTIFACT_PUBLICATION:
LOCAL_ONLY
PHASE_9_5:
NOT STARTED
```

The baseline result is a development measurement, not a final claim. It selects nothing on the locked test, which stays `FROZEN_UNOBSERVED_BY_MODEL` with zero model forward passes.

Phase 9.5 is the first loss/class-imbalance experiment. It remains unstarted and unauthorized. The completed baseline gives it a concrete target: aggregate `freshpotato` F1 of 0.3682 against 0.929 to 0.999 for ten of the fourteen classes.