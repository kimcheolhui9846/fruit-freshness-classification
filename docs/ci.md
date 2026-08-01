# Continuous Integration

## Purpose

`Repository CI` verifies that the repository remains installable, import-safe, testable, and clean without training a model or downloading a dataset. It is a repository health gate, not an experiment runner or benchmark.

## Triggers

The workflow runs for pushes to `main` and all other branches, pull requests targeting `main`, and manual dispatches. No path filters are used, so documentation, workflow, dependency, and source changes all receive the same health checks.

## Runner Matrix

The workflow uses Python 3.12 on both `windows-latest` and `ubuntu-latest`. It installs the pinned PyTorch and TorchVision versions from the official CPU wheel index, asserts that CUDA is unavailable, and does not provision GPU runners.

Existing CUDA-only parity tests are intentionally skipped on these CPU runners. Their skip behavior is part of the test suite's contract; CPU CI must still pass every non-CUDA test.

## Validation Sequence

Each runner performs the following checks:

1. Installs CPU-only PyTorch/TorchVision, then the committed development dependencies.
2. Runs `python -m pip check` and reports Python, PyTorch, TorchVision, datasets, JupyterLab, and ipykernel versions.
3. Imports runtime dependencies and representative reusable project APIs without invoking the dataset loader.
4. Loads and validates `configs/deep3.toml`.
5. Compiles `src/`, `scripts/`, and `tests/`.
6. Runs the complete `unittest` discovery suite, including the static CI contract test.
7. Runs `python -m scripts.train --help` and `python -m scripts.evaluate --help` only.
8. Requires a clean Git worktree after the checks.

The workflow never trains, evaluates a holdout set, writes checkpoints, starts Jupyter, uploads artifacts, or invokes a production dataset-loading command.

## Dataset and Display Safety

Validation steps set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `MPLBACKEND=Agg`. These variables are scoped to validation steps rather than package installation so dependency installation is not unintentionally constrained. The CI contract also prohibits a production dataset-loader invocation in the workflow.

## Supply-Chain and Permission Policy

Workflow permissions are restricted to `contents: read`. `actions/checkout` and `actions/setup-python` are official Actions pinned to immutable full commit SHAs, with their reviewed release tags recorded inline. Checkout fetches a depth of 32: enough for the committed historical parity baselines (currently at most 29 commits behind the Phase 6.1 head) plus the planned handoff commit, while avoiding an unnecessary full-history fetch. It disables credential persistence and submodules. Concurrency cancels an older in-progress run for the same workflow/ref pair, and each job has a 30-minute timeout.

## Interpreting Results

A green matrix means the pinned dependency set, import surface, configuration, source syntax, tests, CLI help paths, and repository cleanliness passed on both supported hosted operating systems. It is not evidence of a completed training run, a model-quality result, a dataset-download test, or GPU/CUDA parity.

A failure should be triaged by its named CI step. Dependency failures should be resolved through the committed dependency specifications; import/config/test failures should preserve the documented runtime behavior; cleanliness failures should identify the generated tracked or untracked file before changing policy.

## Repository Settings Boundary

This phase does not change GitHub branch protection, required status checks, repository rulesets, badges, releases, or Actions permissions outside this workflow. Those repository-level policies require a separate explicit decision after the workflow has demonstrated stable passing runs.