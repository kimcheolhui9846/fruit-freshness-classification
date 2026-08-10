# Changelog

## [Unreleased]

### Added
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

### Changed

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
