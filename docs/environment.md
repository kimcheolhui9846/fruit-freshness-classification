# Environment and dependency specification

This document specifies the minimal environment for the modular fruit-freshness classification project. It records the environment used for Phase 5.1 validation and distinguishes it from clean-machine reproduction, which has not yet been performed.

## Scope and Python version

The verified interpreter is **Python 3.12.10 (64-bit)** on Windows 11. The current dependency specification is validated only with that exact Python version; other Python versions, including later 3.12 patch releases, have not been separately validated. This repository intentionally does not add `.python-version`, Conda, Poetry, uv, Pipenv, PDM, Hatch, or a lock file.

`requirements.txt` contains direct runtime dependencies. `requirements-dev.txt` adds only the tooling needed to launch and edit the active `deep3.ipynb` notebook. The test suite uses the standard-library `unittest` runner, so no third-party test runner is required.

## Verified environment

| Component | Verified value | Status |
| --- | --- | --- |
| Operating system | Windows 11, 64-bit | validated host |
| Python | 3.12.10, 64-bit | interpreter used for validation |
| PyTorch | 2.6.0+cu124 | validated host |
| Torchvision | 0.21.0+cu124 | validated host |
| CUDA runtime reported by PyTorch | 12.4 | available |
| GPU | NVIDIA GeForce RTX 3070 Ti | CUDA available |
| NumPy | 2.5.1 | validated by current tests |
| scikit-learn | 1.9.0 | validated by metric tests |
| Pillow | 12.3.0 | validated by current tests |
| tqdm | 4.70.0 | validated by imports and tests |
| Matplotlib | 3.11.1 | imported by `deep3.ipynb` |
| ipykernel | 7.2.0 | installed, but notebook-server execution was not run |
| Hugging Face `datasets` | not installed | production dataset execution not validated |
| JupyterLab | not installed | development-only installation remains unvalidated |

The active source imports `datasets.ClassLabel`, `datasets.DatasetDict`, and `datasets.load_dataset`, then uses dataset split, map, select, and feature APIs. `datasets>=3.0` is therefore a direct runtime requirement. Its lower bound preserves these active APIs while avoiding an unsupported exact-version claim: the package was not present on the validation host, so downloading `Densu341/Fresh-rotten-fruit` was not performed in this phase. Hugging Face documents `load_dataset()` as the supported Hub-loading API, and the current package metadata supports Python 3.10 or newer. See the [Hugging Face loading guide](https://huggingface.co/docs/datasets/en/loading) and the [datasets PyPI project](https://pypi.org/project/datasets/).

## Dependency policy

| Package | Requirement | Policy | Reason |
| --- | --- | --- | --- |
| `numpy` | `==2.5.1` | exact pin | Directly imported by data, transforms, training, and evaluation; this version passed the current suite. |
| `torch` | `==2.6.0` | exact base pin | Model, engine, loss, trainer, and inference behavior are PyTorch-sensitive. CUDA wheel selection is documented separately. |
| `torchvision` | `==0.21.0` | exact base pin | Direct transform dependency; paired with the verified PyTorch release. |
| `datasets` | `>=3.0` | minimum bound | Direct Hugging Face dataset API dependency; production execution is not locally verified, so no unsubstantiated exact pin is used. |
| `scikit-learn` | `==1.9.0` | exact pin | Direct metric and stratified-fold dependency validated by tests. Its installed metadata accepts NumPy 2.5.1. |
| `matplotlib` | `==3.11.1` | exact pin | Direct notebook plotting import on the verified host. |
| `Pillow` | `==12.3.0` | exact pin | Image handling dependency; tests import it through `PIL`, and the production data path consumes PIL images decoded by Hugging Face Datasets. |
| `tqdm` | `==4.70.0` | exact pin | Direct trainer and inference progress dependency. |
| `jupyterlab` | unpinned | development-only direct requirement | Needed to open and execute the active notebook, but no JupyterLab distribution was installed on the validation host. |
| `ipykernel` | `==7.2.0` | exact development pin | Installed kernel package for notebook execution; the notebook server itself was not run in this phase. |

`requirements.txt` deliberately uses base PyTorch versions, not a machine-specific local version such as `torch==2.6.0+cu124`. This keeps the file valid for CPU and CUDA users while requiring explicit wheel-index selection below.

## Create an environment

The supported workflow is Python `venv` on the verified Windows host:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose **one** PyTorch wheel path before installing the general requirements. Do not install both CPU and CUDA wheels in the same environment.

### CPU-only

For a CPU-only environment, use the official PyTorch CPU wheel index:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### NVIDIA CUDA

For an NVIDIA system, use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to select the operating system, package manager, Python version, and supported CUDA build. The verified host used CUDA 12.4 and the following PyTorch 2.6.0 command, which is listed in the [official previous-versions guide](https://pytorch.org/get-started/previous-versions/):

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The Python wheel supplies its compatible CUDA runtime components; it does **not** install or replace the NVIDIA system driver. Use a driver supported by the selected PyTorch build. CUDA availability depends on the operating system, driver, hardware, and selected wheel, so the CUDA command above is a tested reference rather than a universal instruction.

### Notebook development tools

Install development-only tooling when opening or executing `deep3.ipynb` interactively:

```powershell
python -m pip install -r requirements-dev.txt
jupyter lab deep3.ipynb
```

`matplotlib` belongs to the runtime file because notebook code imports it to render plots. JupyterLab and IPython kernel tooling belong to the development file because they provide the interactive server and kernel rather than the classification pipeline itself. `pytest`, `nbformat`, linters, formatters, and documentation frameworks are intentionally omitted: they are neither imported nor configured by this repository.

## Verification

After installation, run the following from the repository root:

```powershell
python -c "import torch, torchvision, numpy, sklearn, PIL, tqdm, matplotlib; print('core imports ok')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "from src.models.factory import build_cmt_classifier; from src.losses.focal import FocalLoss; from src.trainers.loops import train_one_epoch; print('project imports ok')"
python -m unittest discover -s tests -p "test_*.py" -v
```

For the production dataset path, also verify the intentional Hugging Face dependency after installing `datasets`:

```powershell
python -c "from datasets import ClassLabel, DatasetDict, load_dataset; print('datasets imports ok')"
```

## Known limitations and next verification

- The production dataset workflow requires the Hugging Face `datasets` package and may require network access or an existing Hugging Face cache. It was not executed in Phase 5.1 because `datasets` is absent from the verified environment.
- The current unit suite validates modular behavior with synthetic data; no production training run, dataset download, checkpoint load, or full notebook execution was run during this phase.
- Checkpoints, model weights, datasets, caches, and generated results are intentionally not stored in Git.
- A clean-machine or clean-virtual-environment installation has not yet been performed. Phase 5.5 should perform that end-to-end reproducibility check before broader compatibility is claimed.
- A known Codex-managed temporary Git reference may cause `git fetch origin` to fail. It is local tool metadata, not a project dependency or environment problem, and must not be modified.
