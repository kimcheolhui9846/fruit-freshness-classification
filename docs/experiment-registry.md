# Experiment registry

| Experiment ID | Phase | Status | Parent | Holdout boundary |
|---|---:|---|---|---|
| `deep3-canonical-reference-01` | 8 | `CLOSED_REFERENCE` | None | `OBSERVED`; historical evidence only |
| `deep3-postholdout-research-01` | 9.2 | `PROTOCOL_FROZEN` | `deep3-canonical-reference-01` | Historical canonical holdout excluded; new locked test is frozen and unobserved by a model |
| `deep3-postholdout-research-01-baseline` | 9.3 | `COMPLETED_DEVELOPMENT_BASELINE` | `deep3-postholdout-research-01` | Development CV only; locked test and canonical holdout are model-inaccessible |
| `deep3-postholdout-research-01-label-audit` | 9.5 | `COMPLETED_DEFECT_NOT_CONFIRMED` | `deep3-postholdout-research-01-baseline` | Development images only; locked test is not inspected |
| `deep3-postholdout-research-01-loss-001` | 9.6 | `PROTOCOL_FROZEN` | `deep3-postholdout-research-01-baseline` | Development CV only; locked test and canonical holdout are model-inaccessible |
| `deep3-postholdout-determinism-check-01` | 9.7 | `COMPLETED_A_ADOPTED` | `deep3-postholdout-determinism-check` | Development route only; locked test is never trained on |
| `deep3-postholdout-research-01-baseline-det` | 9.8 | `REGISTERED_NOT_YET_RUN` | `deep3-postholdout-deterministic-baseline` | Development CV only; locked test and canonical holdout are model-inaccessible |

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
LABEL_AUDIT_PROTOCOL_FROZEN
```

The baseline result is a development measurement, not a final claim. It selects nothing on the locked test, which stays `FROZEN_UNOBSERVED_BY_MODEL` with zero model forward passes.

## Phase 9.5 label quality audit

The baseline diagnostic reordered the hypothesis queue. Phase 9.5 is a development label quality audit, and the loss/class-imbalance experiment moves to Phase 9.6. The method is frozen in [postholdout-label-audit-protocol.md](postholdout-label-audit-protocol.md) before any image is reviewed.

```text
EXPERIMENT_ID:
deep3-postholdout-research-01-label-audit
ROLE:
DEVELOPMENT_LABEL_QUALITY_AUDIT
AUDIT_PROTOCOL_STATUS:
FROZEN
AUDIT_EXECUTION_STATUS:
COMPLETED
AUDIT_OUTCOME:
DEFECT_NOT_CONFIRMED
MODEL_TRAINING:
NO
MODEL_INFERENCE:
NO
LOCKED_TEST_INSPECTION:
NO
LABELS_MODIFIED:
0
PHASE_9_6:
H1_LOSS_AND_CLASS_IMBALANCE
```

The audit reviewed 347 development `freshpotato` images against a 150-image `rottenpotato` control. Neither reviewer's subject error rate reached 15 percentage points above their own control rate — the assistant's was 0.0259 against 0.0800 over all 497, the owner's 0.1324 against 0.1250 over a 100-image subsample — so the frozen rule returns `DEFECT_NOT_CONFIRMED` and Phase 9.6 is H1, as pre-registered.

The labels are sound. The Phase 9.4 inference that `freshpotato` was mislabeled is refuted: two reviewers read those images as fresh, so the model is wrong rather than the data. Details and the recorded deviations, including that the owner reviewed a subsample rather than the full set, are in [postholdout-label-audit-protocol.md](postholdout-label-audit-protocol.md).
## Phase 9.6 H1 loss experiment

The first loss / class-imbalance candidate is frozen in [postholdout-loss001-protocol.md](postholdout-loss001-protocol.md) before it runs. Freezing does not authorize training.

```text
EXPERIMENT_ID:
deep3-postholdout-research-01-loss-001
HYPOTHESIS_FAMILY:
H1
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
NOT_YET_RUN
CHANGED_PARAMETER:
loss.class_balanced_beta 0.999 -> 0.9999
ADVANCE_THRESHOLD:
Macro F1 >= 0.9112 and Top-1 >= 0.9466
CANDIDATE_COUNT:
1
LOCKED_TEST_MODEL_ACCESS:
NO
PHASE_9_6_OUTCOME:
NOT_ADVANCED_BUT_INCONCLUSIVE
PHASE_9_7:
DETERMINISM_THEN_RETEST
```

The baseline already trains with class-balanced focal loss, so this run changes how hard that mechanism pushes rather than introducing it. At the baseline's beta the `freshpotato`-to-`rottenapples` weight ratio is 3.18 against a frequency ratio of 7.82; at 0.9999 it is 6.97, essentially inverse frequency. A null result is therefore close to the strongest available form of "reweighting does not fix this class", which is why the failure branch retires H1 rather than trying another loss variant.

## Phase 9.6a run-to-run noise floor

Three runs of the identical baseline recipe on the identical frozen folds gave Macro F1 0.901167, 0.912041, and 0.901858: mean 0.905022, sample standard deviation 0.006089, range 0.010874. The frozen two-sigma rule compares the loss-001 improvement of 0.0090 against 2s = 0.012177 and returns `INCONCLUSIVE`.

```text
NOISE_FLOOR_SAMPLE_SIZE:
3
SAMPLE_STDEV:
0.006089
TWO_SIGMA:
0.012177
PHASE_9_6_STATUS:
INCONCLUSIVE
PHASE_9_7:
DETERMINISM_THEN_RETEST
```

Rerunning the baseline unchanged produced 0.912041, above the 0.9112 acceptance threshold loss-001 missed. A null intervention would have cleared the bar on that draw, so a single run cannot resolve an effect of that size under this pipeline. The cause is that training sets no random seed; Phase 9.7 introduces determinism before any further candidate is judged. The loss-001 verdict of `NOT_ADVANCED` is unchanged — the measurement bears only on the inference drawn from it.

## Phase 9.7 training determinism

The pipeline gained explicit seeding under [postholdout-determinism-protocol.md](postholdout-determinism-protocol.md), whose six-branch adoption ladder was frozen before either verification run executed. Two bounded runs of the same configuration produced identical weights, identical EMA weights, and identical fold histories, so the ladder's first branch was reached directly.

```text
CHECK_EXPERIMENT_ID:
deep3-postholdout-determinism-check-01
SEED:
20260815
LEVEL_ATTEMPTED:
A_STRICT
NONDETERMINISTIC_OPERATION_ERROR:
NONE
BIT_EXACT:
YES
OUTCOME:
A_ADOPTED
LOCKED_TEST_MODEL_ACCESS:
NO
PHASE_9_8:
DETERMINISTIC_BASELINE_THEN_RETEST_DECISION
```

This check advances no candidate and is not an experiment in the H-family sense; it is registered so the runs that consumed GPU time are accounted for. Because `torch.use_deterministic_algorithms(True)` raises rather than degrading silently, completing the run is itself the evidence that no nondeterministic operation is reachable in this model's training path.

Phase 9.8 needs a new baseline under the adopted pipeline: the recorded 0.9012 came from the unseeded pipeline and is not a valid comparison basis for a deterministic run.

## Phase 9.8 measurement floor

The baseline is re-established under the pipeline Phase 9.7 adopted, because the recorded 0.901167 came from the unseeded pipeline and is not a valid comparison basis for a deterministic run. The frozen protocol is [postholdout-measurement-floor-protocol.md](postholdout-measurement-floor-protocol.md).

```text
DETERMINISTIC_BASELINE_ID:
deep3-postholdout-research-01-baseline-det
SEED:
20260815
DETERMINISM_LEVEL:
A_STRICT
MDE_MACRO_F1:
0.012177
VALIDITY_ENVELOPE:
0.892845 to 0.917199
H1_STATUS:
CLOSED_BELOW_RESOLUTION
LOCKED_TEST_MODEL_ACCESS:
NO
PHASE_9_9:
FRESHPOTATO_STABILITY
```

The measurement floor is derived from the three Phase 9.6a replicates and binds regardless of determinism: fixing a seed pins one draw from the same distribution rather than narrowing it. `freshpotato` alone accounts for 90.56% of Macro F1's run-to-run variance, so the instrument's noise is the class the research was trying to improve.
