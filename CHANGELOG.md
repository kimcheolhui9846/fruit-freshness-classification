# Changelog

## Unreleased

### Added

- Canonical MIT software license and repository-only citation metadata.

- Modular `src/` architecture for the dataset, transforms, model, losses, training engine, evaluation, inference, and utilities.
- Version-controlled `configs/deep3.toml` experiment configuration.
- Training and labeled-holdout evaluation CLI entry points.
- Offline repository contract tests and Windows/Ubuntu CPU GitHub Actions CI.
- Portfolio-oriented README, detailed operation documents, release-readiness audit, governance decision package, and release checklist.

### Changed

- Governance documentation now records approved MIT software-license and repository-only citation decisions; release readiness distinguishes these resolved repository decisions from pending release authorization.

- The active `deep3.ipynb` notebook delegates reusable implementation to modular source APIs while retaining orchestration and presentation.
- CI now checks out the complete repository history so the existing historical architecture-parity test can access its fixed baseline.

### Fixed

- Hugging Face dataset loading now uses the pinned source archive, safe managed extraction, and an explicit ImageFolder content root.

### Verified

- Clean-environment installation, fixed-revision dataset loading, real-data CUDA CMT smoke coverage, checkpoint interoperability, and the labeled holdout evaluation path with untrained compatibility fixtures.
- Windows and Ubuntu CPU CI health checks, including repository cleanliness.

### Known limitations

- Canonical three-fold training, trained-checkpoint evaluation, benchmark reproduction, full notebook execution, and independent-machine reproduction have not been completed.
- No trained weights or benchmark-quality metrics are distributed.
- Dataset attribution and redistribution, trained-weight distribution, release authorization, release date, and DOI decisions remain pending.
- No tag or GitHub Release has been created.
