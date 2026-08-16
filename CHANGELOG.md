# Changelog

## [Unreleased]

### Added
- Recorded Phase 9.9 as a negative methodological result at zero GPU cost. A per-image McNemar test on the 347 `freshpotato` images was proposed to escape the measurement floor, then calibrated against run pairs that share an identical configuration: four of six such pairs reject the null at 0.05. The runs differ in their marginals — 252, 214, 246, and 233 errors on the same images — a swing of 38 where the test calls a net change of 18 significant. Splitting one pair of runs into 347 image-level comparisons does not create 347 independent observations; the effective sample size is the run count. `freshpotato` stability is closed as not measurable at this scale, the underlying question stays open, and the three-runs-per-arm alternative at about 55 GPU hours was declined as still underpowered.
- Recorded a zero-GPU descriptive finding alongside it: the logit-average ensemble of the three unseeded runs classifies 100 of 347 `freshpotato` images correctly, worse than the best single run at 133 and worse than the mean of its members. The runs share 183 images they are all confidently wrong about, at mean probability 0.06 on the true class, so averaging preserves the agreement. The oracle bound — picking whichever run happened to be right, per image — reaches only 164.
- Executed the Phase 9.8 deterministic baseline and adopted development OOF Macro F1 0.901891 as the comparison basis for every later candidate. The pre-registered validity envelope of 0.892845 to 0.917199 returns `INSIDE_ENVELOPE`, with the result 0.26 of an MDE below the three-replicate mean. `cudnn_benchmark = false` cost about 3% of wall time, 547.85 minutes against 530.98.
- Recorded that Top-1 landed 1.28 times its own MDE below the replicate mean in that run, with three per-class F1 values outside their own two-sigma bands. The outcome is unchanged, because the envelope was frozen on Macro F1 before the run and adding a gate afterwards would be choosing a criterion after seeing the result. The observation is kept with the three reasons it is weaker than the ratios suggest.
- Froze the Phase 9.8 measurement floor. `MDE_MACRO_F1` is 0.012177, `MDE_TOP1` 0.001969, and `MDE_FRESHPOTATO_F1` 0.147833, each two sample standard deviations of the three Phase 9.6a replicates. A candidate whose improvement falls below its metric's MDE is recorded `BELOW_RESOLUTION`, neither a success nor a failure. `src/utils/measurement_floor.py` implements the rule with three verdicts, because a candidate worse than the baseline by more than the floor is a measured regression rather than an unresolvable result.
- Recorded that `freshpotato` alone accounts for 90.56% of Macro F1's run-to-run variance, against 5.43% for `rottenpotato` and 4.01% for the remaining twelve classes. The instrument's noise was the class the research was trying to improve, which is why the Phase 9.6 acceptance margin of 0.010 — below the 0.012177 floor measured afterwards — could not have separated signal from noise.
- Closed H1 as `CLOSED_BELOW_RESOLUTION`. Resolving the observed effect needs 71 to 212 GPU hours under a conservative zero-correlation bound on the paired difference, using the measured 8.85-hour run duration of the one uninterrupted replicate. This is neither "H1 is exhausted", which the evidence does not support, nor "inconclusive, keep trying", which the arithmetic prices out of reach. The loss-001 verdict is unchanged and is not re-scored.
- Added `scripts/diagnose_freshpotato_instability.py`, which reads recorded prediction files and describes error-set movement without loading a model or consuming GPU. Of 347 development `freshpotato` images, 183 are misclassified in all three unseeded replicates, 102 flip between runs, and 62 are correct in all three.
- Adopted `A_STRICT` determinism in Phase 9.7 and verified it. Two bounded runs of the same configuration produced identical digests over model weights, EMA weights, and fold histories, and `sha256sum` over the four checkpoint files each run wrote confirmed the match independently of the project's own comparison script. The frozen six-branch ladder reached its first branch directly; no descent to `B_CUDNN` was needed.
- Added `src/utils/determinism.py` with `seed_everything`, `apply_determinism`, and `resolve_policy`. `A_STRICT` refuses to run once CUDA is initialised, because `CUBLAS_WORKSPACE_CONFIG` is read when the cuBLAS handle is created and is silently ignored afterwards, so without the guard a run could record strict determinism it never had. `torch.use_deterministic_algorithms(True)` raised on nothing in this model's training path, so completing the run is itself the evidence that no nondeterministic operation is reachable.
- Added `scripts/verify_determinism.py`, which compares two completed output directories by digest and exits non-zero when they differ.
- Executed the Phase 9.6 H1 loss experiment. `scripts.apply_loss001_decision` computed `NOT_ADVANCED`: development OOF Macro F1 0.9102 against the frozen threshold of 0.9112, short by 0.0010, with the Top-1 guardrail passed. `freshpotato` F1 rose from 0.3682 to 0.5140 and its recall from 0.2738 to 0.3977, while the nine already-working classes each lost a little and absorbed much of the gain.
- Measured the run-to-run noise floor under a rule frozen before the replicates ran. Three executions of the identical baseline recipe on identical folds gave Macro F1 0.901167, 0.912041, and 0.901858: sample standard deviation 0.006089, 2s = 0.012177, range 0.010874. The two-sigma rule returns `INCONCLUSIVE` against the loss-001 improvement of 0.0090, and the range agrees. Rerunning the baseline unchanged produced 0.912041 — above the threshold loss-001 missed.
- Recorded that Phase 9.6 is `INCONCLUSIVE` rather than H1 exhausted, because it was underpowered to support that conclusion. The loss-001 verdict is unchanged; the measurement bears only on the inference drawn afterwards and re-scores nothing.
- Recorded that the training pipeline sets no random seed, so training results are not reproducible even though the data, configuration, and fold indices are frozen and verifiable. No single-run experiment on this pipeline can resolve an effect below roughly 0.012 Macro F1. Phase 9.7 introduces determinism before any further candidate is judged.
- Froze the Phase 9.6 H1 loss experiment `deep3-postholdout-research-01-loss-001` before running it. It changes one parameter, `loss.class_balanced_beta`, from 0.999 to 0.9999, reuses the baseline's frozen folds so the comparison is meaningful, and fixes acceptance at development Macro F1 at least 0.9112 with Top-1 at least 0.9466. A result below that retires H1 as exhausted and makes Phase 9.7 H2 augmentation, so the family cannot be searched without pre-registration. Freezing does not authorize training.
- Recorded that the baseline already trains with class-balanced focal loss, so H1 is a question of reweighting strength rather than of introducing the mechanism: `freshpotato` carries the largest alpha of the fourteen classes and still collapses to recall 0.274, while only about 40 percent of full inverse-frequency correction is applied.
- Recorded that the run-to-run noise floor is unmeasured, so the +0.010 acceptance margin is a reasoned rather than a measured choice.
- Executed the Phase 9.5 label quality audit. It returned `DEFECT_NOT_CONFIRMED`: neither reviewer's `freshpotato` subject error rate reached 15 percentage points above their own `rottenpotato` control rate (0.0259 against 0.0800 over all 497 images; 0.1324 against 0.1250 over a 100-image subsample). Inter-rater agreement over the shared subsample was 0.80 raw, Cohen's kappa 0.6259. No label was modified, no locked-test image was inspected, and no model was run.
- Recorded that the Phase 9.4 label-noise inference is refuted. The three signals behind it hold, but two reviewers read the `freshpotato` images as fresh, so the model is wrong rather than the data. Phase 9.6 is H1 loss and class imbalance, as originally pre-registered.
- Recorded three deviations that limit the result: the owner reviewed a seeded 100-image subsample rather than all 497, the assistant's judgments were visible before the owner judged so the reviews are not fully independent, and `scripts.analyze_label_audit` was not run because it requires two complete 497-row files by design and that guard was not weakened.
- Froze the Phase 9.5 development label quality audit protocol before reviewing any image: judgment criteria, a 347-image `freshpotato` subject group against a 150-image `rottenpotato` control, seeded review-set construction, blind independent dual review, and a decision rule with a pre-committed 15-percentage-point threshold that selects Phase 9.6 from the outcome. No image was reviewed, no label was changed, and the locked test was not inspected.
- Executed the owner-authorized post-holdout development baseline `deep3-postholdout-research-01-baseline` and its development-only out-of-fold evaluation. Aggregate development OOF Macro F1 is 0.9012 (balanced accuracy 0.9007, Top-1 0.9566) over all 17,188 development examples, each predicted exactly once. Per-fold Macro F1 is reported separately as 0.8907, 0.9098, and 0.9022; their mean is a different quantity from the aggregate and is not substituted for it.
- Recorded that aggregate `freshpotato` F1 is 0.3682 at recall 0.2738, with 164 of 347 examples predicted `rottenpotato`, identifying the fresh/rotten confusion that Phase 9.5 targets.
- Prepared the post-holdout baseline execution runbook, including frozen inputs, fresh and resume commands, development-only OOF evaluation, preflight and stop conditions, and an unresolved owner approval block. No training was started.
- Extended CI to verify the `scripts.evaluate_postholdout_baseline` command-line interface.
- Materialized the deterministic Phase 9.3 development-CV identity without training, model construction, or model inference.
- Authorized the Phase 9.3 post-holdout development baseline without starting training or publishing artifacts.
- Froze the Phase 9 post-holdout development and locked-test protocol.
- Added a reproducible stratified split derived only from the historical canonical training pool.
- Preserved both the historical canonical holdout and the newly locked Phase 9 test pool outside model-development feedback.

- Canonical internal-holdout result interpretation, per-class metrics, aggregated confusion-matrix documentation, a model card, and a documentation-only [artifact publication decision](docs/artifact-publication-decision.md) for `deep3-canonical-reference-01`.

- Closed the canonical reference run after completed training, locked holdout evaluation, and result interpretation.
- Recorded local-only retention for canonical binary artifacts until an explicit future owner decision.
- No model weights, checkpoints, training state, logs, raw predictions, raw logits, dataset content, GitHub Actions artifacts, Release assets, Release, or tag were published.

- Started post-holdout research planning under a new experiment identity.
- Defined the boundary between the closed canonical holdout and future development/evaluation.
- Added experiment-registration and pre-registration rules before any Phase 9 training.

### Fixed
- Corrected the experiment registry, which recorded `deep3-postholdout-research-01-loss-001` as `PROTOCOL_FROZEN` and its section as `EXECUTION_STATUS: NOT_YET_RUN`. It ran on 2026-08-14 and returned `NOT_ADVANCED`, and the outcome sat two lines below the stale status.
- Corrected README and the model card, which both stated that no sample-level image review was performed. The Phase 9.5 label audit reviewed 497 images. Both now record what was reviewed and that it was development images only, never a canonical-holdout image, which is the distinction the original sentence existed to protect.
- Corrected the top-level reproducibility record. The run-level documents already disclaimed bit-for-bit reproducibility, but the README opening described the pipeline as "reproducible" without qualification, the README status table omitted training-run reproducibility entirely, `docs/reproducibility.md` omitted it from its limitations list, and the research plan required every run to record a seed that did not exist.
- Normalized tracked post-holdout JSON identity hashes to LF before SHA-256 calculation, preventing cross-platform CI mismatches.

### Changed
- Applied the determinism policy once at training and evaluation start-up, before the device is resolved, because `CUBLAS_WORKSPACE_CONFIG` is read when the cuBLAS handle is created and is ignored afterwards. The per-fold `cudnn.benchmark` assignment was deleted in favour of that single call.
- Moved the run manifest to `schema_version` 2 with a `determinism` block. An unseeded run records its seed and level as null rather than omitting the fields, so "this run had no seed" is a recorded fact rather than a missing one.
- Made `[runtime].seed` and `[runtime].determinism_level` optional, so every configuration frozen before Phase 9.7 keeps its recorded SHA-256 and its original meaning. The two must appear together, and `A_STRICT` or `B_CUDNN` with `cudnn_benchmark` left true is refused as self-contradictory.
- Recorded that the `DataLoader` generator proposed by the Phase 9.6a follow-up is deliberately not added. At `num_workers = 0` the sampler draws from the global torch generator, which `training_state.pt` already persists, so a separate generator would sit outside that state and break epoch-boundary resume determinism that currently works.
- Added a Phase 9 section to the README. The front page previously showed only the canonical holdout result, leaving the post-holdout research invisible to anyone who did not open `docs/`.
- Recorded in the model card that `freshpotato` is unstable and not merely weak: F1 0.428 with a standard deviation of 0.074 against 0.018 for the next-noisiest class, 183 of 347 wrong in every run and 62 right in every run. The entry states that these come from the development split and do not restate the holdout numbers the card reports.

- Reordered the pre-registered hypothesis queue: H6 error-focused analysis becomes Phase 9.5 and H1 loss / class imbalance moves to Phase 9.6. The baseline evidence for the change is recorded in the research plan, governance decisions, and registry rather than applied silently — class frequency does not explain the dominant error, since the imbalance ratio is 7.8:1, the support-to-F1 correlation is 0.500, and `rottenpotato` (514 examples, F1 0.7741) sits beside `rottencapsicum` (570 examples, F1 0.9965).
- Phase 9.4 moved from runbook preparation to recorded baseline execution across the runbook, experiment registry, governance decisions, research plan, and baseline record. The `deep3-postholdout-research-01-baseline` registry status is now `COMPLETED_DEVELOPMENT_BASELINE`.
- Narrowed the post-holdout baseline contract test rather than dropping it: it now asserts the executed state and additionally fails if the spent training approval ever widens into locked-test access, canonical-holdout re-evaluation, binary publication, or release creation.
- README and reproducibility status now distinguish the completed local canonical run and locked internal holdout from the historical untrained compatibility evidence.

### Artifact policy

- Aggregate metrics and documentation are public; dataset content, checkpoints, weights, training state, logs, raw logits, raw predictions, and all binary artifacts remain local-only through Phase 8.6.
## [0.1.0] - 2026-08-02

### Added
- Materialized the deterministic Phase 9.3 development-CV identity without training, model construction, or model inference.
- Froze the Phase 9 post-holdout development and locked-test protocol.
- Added a reproducible stratified split derived only from the historical canonical training pool.
- Preserved both the historical canonical holdout and the newly locked Phase 9 test pool outside model-development feedback.

- Canonical MIT software license and repository-only `CITATION.cff` metadata.
- Modular `src/` architecture for the dataset, transforms, model, losses, training engine, evaluation, inference, and utilities.
- Version-controlled `configs/deep3.toml` experiment configuration.
- Training and labeled-holdout evaluation CLI entry points.
- Offline repository contract tests and Windows/Ubuntu CPU GitHub Actions CI.
- Portfolio-oriented README, detailed operation documents, release-readiness audit, governance decision package, and release checklist.

### Changed

- The active `deep3.ipynb` notebook delegates reusable implementation to modular source APIs while retaining orchestration and presentation.
- CI checks out the complete repository history so the existing historical architecture-parity test can access its fixed baseline.
- Governance documentation now distinguishes the resolved repository software/citation decisions from the separate external-dataset and trained-weight boundaries.

### Fixed

- Hugging Face dataset loading now uses the pinned source archive, safe managed extraction, and an explicit ImageFolder content root.

### Verified

- Clean-environment installation, fixed-revision dataset loading, real-data CUDA CMT smoke coverage, checkpoint interoperability, and the labeled holdout evaluation path with untrained compatibility fixtures.
- Windows and Ubuntu CPU CI health checks, including repository cleanliness.

### Artifact policy

- This release distributes source code and documentation only; it does not redistribute the external dataset, trained weights, checkpoints, caches, environments, logs, or other binary artifacts.

### Known limitations

- Canonical three-fold training, trained-checkpoint evaluation, benchmark reproduction, full notebook execution, and independent-machine reproduction have not been completed.
- No trained weights or benchmark-quality metrics are distributed.
- Dataset attribution and redistribution remain subject to the original external source terms, and trained-weight distribution requires a separate review.
