# Fruit Freshness Classification

[![Repository CI](https://github.com/kimcheolhui9846/fruit-freshness-classification/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kimcheolhui9846/fruit-freshness-classification/actions/workflows/ci.yml)

A modular PyTorch research pipeline for fresh/rotten fruit classification, with a reproducible environment, dataset identity, and configuration. See [Reproducibility status](#reproducibility-status) for what training-run reproducibility does and does not cover. The active experiment combines a CMT-based classifier, Hugging Face dataset integration, stratified K-Fold training, Mixup, configurable CE/Focal loss, EMA, checkpointed fold models, horizontal-flip TTA, and raw-logit ensemble evaluation. The active notebook and CLI entry points use the same committed TOML configuration and reusable `src/` APIs.

## Overview

This repository is designed as a research-oriented computer-vision project: reusable implementation lives in `src/`, `deep3.ipynb` remains the active orchestration and presentation notebook, and `scripts/` exposes thin training and labeled-holdout evaluation entry points. Reproducibility evidence and CI validate repository mechanics. A frozen canonical run and one trained-checkpoint internal holdout evaluation are documented separately; neither substitutes for external validation, deployment validation, or a benchmark claim.

## Verified capabilities

- Modular datasets, transforms, models, losses, training loops, checkpointing, evaluation, inference, and utility packages.
- CMT classifier with stratified three-fold training, AMP, Mixup, configurable CE-with-label-smoothing or Focal Loss, class-balanced alpha, and EMA.
- Pinned Hugging Face archive loading with safe cache extraction and explicit ImageFolder handling.
- Deterministic holdout preprocessing, ordered fold checkpoint loading, horizontal-flip TTA, and raw-logit ensemble evaluation.
- Committed experiment configuration shared by the notebook, training CLI, and evaluation CLI.
- Clean-environment evidence plus offline CPU repository CI on Windows and Ubuntu.

## Dataset

The project uses [`Densu341/Fresh-rotten-fruit`](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit). The loader pins a Hub revision and `freshness_fruit.zip`, safely extracts it into the Hugging Face cache, and loads the explicit ImageFolder content root with `datasets==5.0.1`.

| Dataset stage | Images |
|---|---:|
| Source archive | 30,357 |
| After label filtering | 26,858 |
| Training split | 21,486 |
| Holdout split | 5,372 |
| Final labels | 14 |

Non-RGB images are converted to RGB. The 80/20 project split uses seed 42. Dataset contents, archives, and caches are intentionally not committed. See [dataset documentation](docs/dataset.md) for the fixed revision, archive, label policy, and dataset terms.

## Quick start

From PowerShell, clone or enter the repository, create an environment, and install the matching PyTorch build before the remaining development requirements.

```powershell
git clone https://github.com/kimcheolhui9846/fruit-freshness-classification.git
Set-Location fruit-freshness-classification
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the appropriate CPU or CUDA PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the repository requirements and run the local health checks.

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall src scripts tests
python -m scripts.train --help
python -m scripts.evaluate --help
```

The documented CPU and CUDA wheel commands, supported environment, and package rationale are in [environment documentation](docs/environment.md). `.venv` is local-only and remains ignored by Git.

## Configuration

[`configs/deep3.toml`](configs/deep3.toml) is the active experiment contract. It controls training epochs, batch size, fold count and seed, Mixup, learning rates, loss settings, EMA, checkpoint naming, and reporting figure size. Derived runtime state (such as fold indices, models, loaders, and histories) is not stored in the file.

The notebook, training CLI, and evaluation CLI load the same committed config. There is no environment-variable or CLI hyperparameter override hierarchy: create a clearly named copy under `configs/` or intentionally edit the experiment file before a run. See [configuration documentation](docs/configuration.md) for the section-level contract.

## Training

The default command uses `configs/deep3.toml` and writes to the repository-relative `weights/` directory.

```powershell
python -m scripts.train
```

Use explicit paths when desired:

```powershell
python -m scripts.train `
  --config configs/deep3.toml `
  --output-dir weights
```

Training requires production dataset access and creates the selected output directory during execution. With the current three-fold configuration, expected artifacts are `label_names.json`, `best_model_fold1.pt`, `best_model_fold2.pt`, `best_model_fold3.pt`, and `last_model_weights.pt`. There is no resume interface; the default configuration is computationally expensive, and training-history plots remain notebook-specific. See [training documentation](docs/training.md).

## Evaluation

Evaluate the complete labeled holdout ensemble with compatible fold checkpoints:

```powershell
python -m scripts.evaluate `
  --config configs/deep3.toml `
  --checkpoint-dir weights
```

`--checkpoint-dir` is required. Every configured fold checkpoint must exist; partial ensembles are rejected, fold order is deterministic, and no result file is written by default. Evaluation applies horizontal-flip TTA and averages raw logits on the production holdout split. Generic unlabeled image inference is not implemented. See [evaluation documentation](docs/evaluation.md).
## Canonical Result (Internal Holdout)

The frozen `deep3-canonical-reference-01` three-fold EMA ensemble was evaluated once on the fixed 5,372-example internal holdout. It achieved 5,133 / 5,372 top-1 correct predictions (0.955510), macro F1 of 0.903737, and balanced accuracy of 0.899969. The protocol used equal three-fold raw-logit averaging plus equal original/horizontal-flip TTA; no post-holdout tuning, alternate checkpoint evaluation, or rerun was performed.

This is an internal fixed holdout result, not an external benchmark, production-validation result, or generalization claim. The complete per-class metrics and aggregated confusion matrix are in the [canonical result interpretation](docs/canonical-results.md); model scope and limitations are in the [model card](docs/model-card.md); and the documentation-only artifact boundary is in the [artifact publication decision](docs/artifact-publication-decision.md). Checkpoints, weights, raw logits, predictions, logs, and dataset contents are not published.

## Post-holdout research (Phase 9)

A separate research programme ran after the canonical result was closed. It never touched the canonical holdout. The 21,486-example historical training pool was split once into 17,188 development and 4,298 locked-test examples, and the locked test has had **zero model forward passes** since.

Every phase froze its protocol and its numeric decision rule before execution, which is why several of them were able to refute the hypothesis that motivated them.

| Phase | Question | Outcome |
|---|---|---|
| 9.5 | Are the `freshpotato` labels wrong? | `DEFECT_NOT_CONFIRMED` — a blind 497-image audit found the labels sound. The model is wrong, not the data. |
| 9.6 | Does stronger class-balanced reweighting help? | `NOT_ADVANCED` at Macro F1 0.9102 against a pre-registered 0.9112. |
| 9.6a | Is that difference bigger than noise? | `INCONCLUSIVE`. Three identical runs gave 0.9012, 0.9120, 0.9019, so 2σ = 0.0122 against an effect of 0.0090. |
| 9.7 | Can the pipeline be made deterministic? | `A_ADOPTED`. It set no random seed at all; with seeding and strict determinism two runs are now bit-exact. |
| 9.8 | What can this project actually measure? | Minimum detectable effect frozen at 0.0122 Macro F1. H1 closed as `CLOSED_BELOW_RESOLUTION`. |
| 9.9 | Can per-image testing get around that? | No. The test fires on four of six run pairs that share an identical configuration. |

Three findings are worth stating on their own.

**The noise floor was larger than the acceptance margin.** Phase 9.6 accepted at baseline plus 0.010 while run-to-run 2σ, measured afterwards, was 0.0122. A criterion narrower than the noise cannot separate signal from noise, and the rerun of the *unchanged* baseline scored higher than the intervention did.

**One class produces 90.56% of the metric's instability, and it is the class the research was trying to fix.** `freshpotato` has a run-to-run standard deviation of 0.0739 against 0.0181 for the next-noisiest class. The instrument's noise was the object of study.

**Determinism removes measurement noise but not seed-to-seed variation.** Fixing a seed pins one draw from the same distribution rather than narrowing it, so bit-exactness did not lower the measurement floor.

Full protocols, decision rules, and outcomes are in [the post-holdout research plan](docs/post-holdout-research-plan.md), [the experiment registry](docs/experiment-registry.md), and [governance decisions](docs/governance-decisions.md). No Phase 9 weights, checkpoints, or predictions are published.

## Canonical run closure

- Canonical reference run: Closed
- Training completed; locked internal-holdout evaluation completed.
- Binaries are not published; no post-holdout tuning was performed.

See [canonical run closure](docs/canonical-run-closure.md), [canonical artifact retention](docs/canonical-artifact-retention.md), the prior [artifact publication decision](docs/artifact-publication-decision.md), and [Phase 8.6 governance resolution](docs/phase-8.6-governance-resolution.md). These documents do not provide binary downloads.
## Notebook usage

`deep3.ipynb` is the active orchestration notebook. It loads the same configuration and calls the modular source APIs while retaining plotting and exploratory presentation. `deep.ipynb`, `deep1.ipynb`, and `deep2.ipynb` are historical experiment notebooks, not current entry points. Full `deep3.ipynb` execution has not been verified.

## Architecture

```text
Config
  -> Dataset + Transforms
  -> Stratified Folds + DataLoaders
  -> CMT + Loss + Optimizer + EMA
  -> Train / Validate
  -> Fold Checkpoints
  -> Holdout TTA Raw-Logit Ensemble Evaluation
```

## Repository structure

```text
configs/                 Active TOML experiment inputs
docs/                    Detailed environment, dataset, usage, reproducibility, and CI notes
scripts/                 Training and labeled-holdout evaluation entry points
src/
  datasets/              Hugging Face preparation, folds, and loaders
  models/                CMT classifier and construction
  losses/                Focal Loss and Mixup criteria
  trainers/              Train and validation loops
  engine/                EMA, optimization, and checkpoint helpers
  evaluation/            Validation metrics
  transforms/            Classification and Mixup transforms
  utils/                 Configuration, paths, labels, and runtime helpers
  inference/             Fold loading and TTA ensemble utilities
tests/                   Standard-library unittest coverage and repository contracts
weights/                 Local generated checkpoints; ignored by Git
deep3.ipynb              Active orchestration notebook
requirements.txt         Runtime dependencies
requirements-dev.txt     Runtime plus notebook-development dependencies
```

## Testing

Run the standard-library test suite and compilation check from the repository root:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall src scripts tests
```

No `pytest` dependency is required. Normal tests are offline and do not run the production dataset download, GPU training, long-running canonical training, or real holdout evaluation. At the Phase 6.1 milestone, both CPU CI runners completed 130 tests with 5 intentional CUDA-only skips; a local CUDA-capable environment can run those parity tests instead of skipping them.

## Continuous integration

`Repository CI` runs on Windows and Ubuntu with Python 3.12 and CPU-only PyTorch. It uses offline Hugging Face mode and headless Matplotlib, checks dependencies and imports, validates the config, compiles source, runs the full unittest suite and CLI help paths, and requires a clean worktree. The workflow uses immutable official Action SHAs and read-only `contents` permission.

CI does not download the production dataset, run CUDA, train the model, perform real holdout evaluation, or configure branch protection/required checks. See [CI documentation](docs/ci.md).

## Reproducibility status

| Capability | Status |
|---|---|
| Clean Python environment install | Verified on the documented Windows host |
| Windows CPU CI | Verified |
| Ubuntu CPU CI | Verified |
| Production dataset load | Verified |
| CUDA CMT smoke | Verified |
| Checkpoint interoperability | Verified |
| Evaluation CLI on the full holdout | Verified with untrained compatibility checkpoints |
| Full canonical three-fold training | Completed once with the derived batch-64 configuration; checkpoints remain local-only |
| Trained-checkpoint evaluation | Completed once on the locked internal holdout |
| Training-run reproducibility | Not verified before Phase 9.7; the pipeline set no random seed, so identical commands produced different weights and metrics |
| Benchmark reproduction | Not verified |
| Independent-machine reproduction | Not verified |

The temporary compatibility-checkpoint holdout measurement remains an interoperability check, not model performance. The documented canonical result is a separate locked internal holdout assessment; it is not a benchmark or production claim. See [reproducibility documentation](docs/reproducibility.md) and the [canonical result interpretation](docs/canonical-results.md).

Every result recorded before Phase 9.7 was produced by a pipeline that set no random seed. Three executions of one unchanged configuration gave development OOF Macro F1 of 0.901167, 0.912041, and 0.901858. The recorded metrics are accurate measurements of the runs that produced them, and any single one of them would land elsewhere on a rerun. See [the determinism protocol](docs/postholdout-determinism-protocol.md).
## Limitations

- Full canonical three-fold training completed once. Its checkpoints remain local-only; no trained binary is committed or published.
- The reported model result is one internal fixed holdout, not an external benchmark, production-validation result, or generalization claim.
- The canonical holdout was evaluated once. It was never reevaluated, no alternate checkpoint was scored against it, and no tuning was performed after seeing it.
- Sample-level image review was performed once, in the Phase 9.5 label audit, on 497 **development** images drawn from the separate post-holdout split. No canonical-holdout image was reviewed.
- Full notebook execution is not verified.
- Generic unlabeled image inference is not implemented.
- A same-machine clean environment is verified; independent-machine reproduction is not.
- Dataset contents and checkpoints are intentionally excluded from Git. Production dataset access requires network or an existing cache outside CI.
- CI is CPU-only and offline; it does not test CUDA, production loading, training, or real holdout evaluation.
- The derived batch-64 run is a different training trajectory from the original batch-192 configuration.
## License

The repository software and project-authored documentation are licensed under the MIT [LICENSE](LICENSE). The external Hugging Face dataset is governed separately by its original source terms; see the [dataset documentation](docs/dataset.md). Dataset files are not included in this repository.

## Citation

Repository-only citation metadata is available in [CITATION.cff](CITATION.cff). The approved repository author is Choelhui Kim. No DOI, paper citation, or released software version is claimed.

## Documentation

| Document | Purpose |
|---|---|
| [Environment](docs/environment.md) | Python, PyTorch, CUDA, and installation |
| [Dataset](docs/dataset.md) | Dataset identity and compatibility loader |
| [Configuration](docs/configuration.md) | Experiment configuration |
| [Training](docs/training.md) | Training CLI and checkpoint policy |
| [Evaluation](docs/evaluation.md) | Holdout evaluation CLI |
| [Canonical results](docs/canonical-results.md) | Frozen internal-holdout interpretation and publication boundary |
| [Model card](docs/model-card.md) | Model scope, performance, limitations, and provenance boundary |
| [Artifact publication decision](docs/artifact-publication-decision.md) | Documentation-only publication boundary and unresolved Phase 8.6 owner gate |
| [Reproducibility](docs/reproducibility.md) | Verified reproducibility evidence and boundaries |
| [Continuous Integration](docs/ci.md) | GitHub Actions repository health |
| [Release readiness](docs/release-readiness.md) | Engineering-milestone audit and release-note draft |
| [Governance decisions](docs/governance-decisions.md) | License, citation, dataset, and repository-policy decisions |
| [Changelog](CHANGELOG.md) | Unreleased capability and limitation summary |

## Contribution boundary

Use feature branches, run the full unittest suite, and keep production artifacts out of Git. Repository software and project-authored documentation are licensed under the MIT [LICENSE](LICENSE). Repository citation metadata is available in [CITATION.cff](CITATION.cff); dataset terms remain available through the [dataset documentation](docs/dataset.md).
