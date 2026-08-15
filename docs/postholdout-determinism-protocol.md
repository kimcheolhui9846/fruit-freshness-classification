# Post-Holdout Training Determinism Protocol

## Status

```text
PHASE:
9.7
ROLE:
PIPELINE_CHANGE_NOT_HYPOTHESIS_TEST
PROTOCOL_STATUS:
FROZEN
EXECUTION_STATUS:
NOT_YET_RUN
SEED:
20260815
VERIFICATION_RUN_COUNT:
2
LOCKED_TEST_MODEL_ACCESS:
NO
ARTIFACT_PUBLICATION:
LOCAL_ONLY
```

This phase changes the training pipeline. It tests no hypothesis, evaluates no candidate, and advances nothing. The adoption rule is fixed here, before the verification runs execute.

## Why this is needed

Phase 9.6a measured the run-to-run noise floor and returned `INCONCLUSIVE`. Three executions of the identical baseline recipe produced aggregate development OOF Macro F1 of 0.901167, 0.912041, and 0.901858, giving `s = 0.006089` and `2s = 0.012177` against the loss-001 improvement of `d = 0.0090`. The clearest single fact is that rep002 — the baseline rerun with no configuration change whatsoever — scored 0.912041, above the 0.9112 threshold that loss-001 had missed.

The cause is that **the training pipeline sets no random seed**. There is no `torch.manual_seed`, `np.random.seed`, or `random.seed` anywhere in `src/` or `scripts/`; `torch.use_deterministic_algorithms` is never called; and `cudnn_benchmark` is `true`, which lets the autotuner select kernels nondeterministically. Every run draws weight initialisation, batch order, mixup sampling, DropPath sampling, and augmentation randomness from operating-system entropy.

Until this is fixed, no single-run experiment on this pipeline can resolve an effect smaller than roughly 0.012 Macro F1. The owner chose determinism followed by re-testing over the alternatives of raising the effect-size bar or adopting a multi-seed protocol costing roughly 27 GPU hours per candidate.

## What already works, and what does not

`src/engine/training_state.py` already captures and restores `python_rng_state`, `numpy_rng_state`, `torch_cpu_rng_state`, and `torch_cuda_rng_states` at every epoch boundary. The resume path therefore already preserves RNG continuity. The gap is confined to **run start**, where nothing is seeded.

Every random draw in training reaches a global generator: weight initialisation and `DropPath` through the torch CPU generator, `DataLoader` shuffling through `RandomSampler`'s draw from the torch CPU generator, and mixup's `np.random.beta`, `np.random.rand`, and `torch.randperm` through the NumPy and torch generators. Seeding `random`, `numpy`, and `torch` at run start therefore covers all of them. Every `DataLoader` in the pipeline uses `num_workers = 0`, so no worker seeding is required.

### The DataLoader generator is deliberately not added

The Phase 9.6a follow-up section proposed adding a `DataLoader` generator. **That proposal is rejected here, and the reason is recorded rather than left implicit.**

With `num_workers = 0` and `shuffle = True`, `RandomSampler` draws a fresh seed from the **global** torch generator on each `__iter__`. That global state is exactly what `capture_rng_state` and `restore_rng_state` already persist. Introducing a separate `torch.Generator` would place the sampler's state outside `training_state.pt`, so a resumed run would reseed that generator and replay the first epoch's ordering instead of continuing the sequence. Adding the generator would therefore **break the epoch-boundary resume determinism that currently works**. The global generator is the correct mechanism here, and it is already covered.

## What changes

| Component | Change |
|---|---|
| `src/utils/determinism.py` | New. `seed_everything(seed)` and `apply_determinism(level, *, seed)`, returning the applied policy record. |
| `src/utils/config.py` | Optional `[runtime].seed` and `[runtime].determinism_level` validation. |
| `scripts/train.py` | Applies the policy before dataset and model construction when the keys are present. |
| `scripts/evaluate.py`, `scripts/evaluate_postholdout_baseline.py` | Apply the same policy when the keys are present. |
| `scripts/verify_determinism.py` | New. Compares two completed output directories; runs no training. |
| `configs/deep3_postholdout_determinism_check.toml` | New. Bounded verification config on the frozen development route. |

### The new configuration keys are optional

`configs/deep3_postholdout_baseline.toml` **must not change.** Its LF-normalized SHA-256 is recorded in `postholdout-loss001-protocol.md` and asserted by `tests/repository/test_loss001_protocol_contract.py`. The same applies to the canonical, loss-001, and replicate configs.

`[runtime].seed` and `[runtime].determinism_level` are therefore validated only when present. A configuration without them retains the current behaviour exactly, which keeps every frozen configuration's hash and meaning intact and keeps the historical runs interpretable against the pipeline that produced them.

Two consistency rules are enforced, because a configuration that contradicts itself is worse than one that omits the setting:

- the two keys must both be present or both absent
- when `determinism_level` is `A_STRICT` or `B_CUDNN`, `cudnn_benchmark` must be `false`

`CUBLAS_WORKSPACE_CONFIG` takes effect only before the cuBLAS handle is created. If `A_STRICT` is requested after CUDA has already been initialised, `apply_determinism` raises rather than applying a setting that would be silently ignored.

### Determinism levels

```text
A_STRICT:
seeds, cudnn.deterministic true, cudnn.benchmark false, use_deterministic_algorithms true, CUBLAS_WORKSPACE_CONFIG set
B_CUDNN:
seeds, cudnn.deterministic true, cudnn.benchmark false
C_SEED_ONLY:
seeds only
```

`C_SEED_ONLY` is defined so the level vocabulary is complete and testable. It is not a candidate for adoption in this phase.

## Verification runs

Two executions of `configs/deep3_postholdout_determinism_check.toml` into separate output directories, compared by `scripts/verify_determinism.py`.

| Field | Value |
|---|---|
| Route | post-holdout development, never the canonical route |
| Split manifest | `configs/splits/deep3-postholdout-research-01.json` |
| CV manifest | `configs/splits/deep3-postholdout-research-01-baseline-cv.json` |
| Epochs | 2, with `fine_tuning.epochs` 1 |
| Batch size | 64, unchanged from the baseline |
| Estimated cost | roughly 10 to 15 minutes per run including dataset preparation |

Two epochs against one fine-tuning epoch exercises both a normal epoch and a fine-tuning epoch in every fold, so the bounded run covers both branches of the `epoch > epochs - fine_tuning_epochs` condition rather than only the first.

**The verification must not use the canonical training route.** That route trains on the full 21,486-example historical pool, which contains the 4,298 locked-test examples. Training on them is a model forward pass over locked-test data regardless of whether any metric is computed, and it would break the invariant below. The check configuration therefore carries a `post_holdout` section pointing at the baseline's frozen manifests.

The check configuration's lineage is registered in `resolve_experiment_validation` with an explicit allowlist. It may differ from the baseline in `runtime`, `training.epochs`, `fine_tuning.epochs`, and the `post_holdout` identity fields, and in nothing else. An unregistered parent still raises.

## Frozen adoption ladder

Let a comparison be **bit-exact** when the two runs produce identical SHA-256 digests over the sorted `model_state_dict`, the sorted `ema_state_dict`, and the recorded per-fold metric histories.

```text
LEVEL_ORDER:
A_STRICT then B_CUDNN
BIT_EXACT_CRITERION:
identical SHA-256 over model_state_dict, ema_state_dict, and fold metric histories
A_ADOPTED:
A completes both runs and the digests match
A_DEGRADED:
A completes both runs and the digests differ
B_ADOPTED:
A raises a nondeterministic-operation error, B completes both runs, digests match
B_DEGRADED:
A raises a nondeterministic-operation error, B completes both runs, digests differ
A_FAILED_OTHER:
A does not complete for any reason other than a nondeterministic-operation error
BLOCKED:
B does not complete
```

Every combination of level, completion, and bit-exactness has a named outcome. A ladder with an unhandled branch would return discretion to whoever reads the result first, which is the freedom this protocol exists to remove.

`A_FAILED_OTHER` is separated from the descent to B on purpose. A resource failure such as an out-of-memory error caused by the cuBLAS workspace setting is not evidence about determinism, and treating it as one would silently convert an environment problem into a scientific conclusion. That branch stops and reports.

**`A_ADOPTED`** — the pipeline is bit-exact and no nondeterministic operation is reachable. This is the strongest available outcome.

**`A_DEGRADED`** — level A is retained, because it is strictly stronger than B, and the measured residual variation is recorded. The claim becomes "seeded, with residual variation of the recorded size" rather than "bit-exact".

**`B_ADOPTED`** and **`B_DEGRADED`** — the operation that raised under level A is recorded by name. Level B carries no static guarantee that a nondeterministic operation is never dispatched, so `B_ADOPTED` rests on the empirical bit-exactness of the bounded run and says so.

**`BLOCKED`** — the phase stops and the direction returns to the owner.

The adopted level and the measured outcome are recorded in this document. They may not be renegotiated after the fact.

## What determinism does not buy

**Determinism removes measurement noise. It does not remove seed-to-seed variation.**

Running a baseline and a candidate under the same seed makes them share weight initialisation and batch ordering, and their RNG streams stay synchronised because the number of draws per epoch is identical. That is a paired comparison under common random numbers, and the variance of the paired difference is genuinely smaller than the variance of the unpaired difference Phase 9.6 relied on. That is the real gain.

It is still one draw. The measured difference estimates the effect **for that seed**, not the expected effect across seeds. A deterministic single-pair comparison cannot establish that an effect generalises.

This is recorded now, before any result exists, because the failure mode is specific and foreseeable: Phase 9.8 could replace the overclaim this phase was created to retire — "H1 is exhausted" — with a different one of the same kind — "the effect is exactly `d`". Whatever claim strength Phase 9.8 adopts must be argued in its own protocol against this limitation.

## Documentation corrections

The repository's run-level documents were accurate. `training.md`, `canonical-training-readiness.md`, `canonical-holdout-evaluation.md`, and `canonical-training-unblock.md` each state that bit-for-bit reproducibility is not claimed and that global seeding was not introduced.

The failure was at the top level, where a reader arrives first:

| Location | Problem |
|---|---|
| `README.md` opening line | describes the pipeline as "reproducible" without qualification |
| `README.md` reproducibility status table | eleven rows, none of them training-run reproducibility |
| `docs/reproducibility.md` remaining limitations | omits the largest limitation |
| `docs/post-holdout-research-plan.md` | requires every run to record "seeds" that did not exist |

Each is corrected, and `governance-decisions.md` records the correction naming the affected runs: `deep3-canonical-reference-01`, `deep3-postholdout-research-01-baseline`, `deep3-postholdout-research-01-loss-001`, `deep3-postholdout-research-01-baseline-rep002`, and `deep3-postholdout-research-01-baseline-rep003`. The record states the distinction accurately — the disclaimers existed in the run-level documents and were missing from the top-level ones — rather than reporting a false claim that was not made.

The recorded metrics do not change. What changes is that a reader can now see the conditions under which they were produced without reading five documents to assemble it.

## Boundaries

Not authorized by this document:

- executing the verification runs; execution requires the owner decision recorded in the approval block
- re-establishing a deterministic baseline or re-running loss-001; both belong to Phase 9.8
- re-scoring loss-001, the noise floor, or any earlier result against a new threshold
- modifying any frozen configuration, protocol threshold, seed, denominator, or split
- evaluating or inspecting the 4,298-example locked test, or re-evaluating the canonical holdout
- publishing weights, checkpoints, dataset copies, predictions, Releases, or tags

```text
POST_HOLDOUT_LOCKED_TEST_STATUS:
FROZEN_UNOBSERVED_BY_MODEL
POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:
0
CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:
0
BINARY_PUBLICATION:
NO
```

## Follow-up: Phase 9.8

Once a level is adopted, re-testing loss-001 requires a **new baseline under the adopted pipeline**. The recorded 0.9012 was produced by the pre-9.7 pipeline and is not a valid comparison basis for a deterministic run. Phase 9.8 therefore costs roughly two full runs, and `cudnn.benchmark = false` is expected to make each somewhat slower than the nine hours the pre-9.7 runs took.

Phase 9.8 needs its own frozen protocol, including its threshold, its claim strength against the seed-conditional limitation above, and an explicit decision on whether loss-001 is re-run at all. None of that is authorized here.

## Owner approval block

```text
APPROVED_SEEDING_INTRODUCTION:
YES
APPROVED_SEED:
20260815
APPROVED_DETERMINISM_LADDER:
YES
APPROVED_VERIFICATION_RUN_COUNT:
2
APPROVED_DOCUMENTATION_CORRECTION_SCOPE:
TOP_LEVEL_PLUS_GOVERNANCE_LEDGER
APPROVED_EXECUTION:
NOT_YET_GRANTED
APPROVED_LOCKED_TEST_EVALUATION:
NO
APPROVED_LOSS001_RERUN:
DEFERRED_TO_PHASE_9_8
APPROVED_WEIGHT_PUBLICATION:
NO
```

The owner approved the phase scope, the determinism ladder, and the documentation correction scope on 2026-08-15, before any implementation existed. Execution of the verification runs is a separate decision and is not yet granted.
