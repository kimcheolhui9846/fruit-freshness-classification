# Environment and dependency specification

This document specifies the minimal environment for the modular fruit-freshness classification project. It records the Phase 5.1 baseline and the Phase 5.5A clean-virtual-environment dataset-loader validation.

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
| ipykernel | 7.2.0 | installed; bounded JupyterLab server startup passed, but notebook execution was not run |
| Hugging Face `datasets` | 5.0.1 | clean-environment loader validation |
| `huggingface-hub` | 1.26.0 | pinned source-download API |
| JupyterLab | 4.6.2 | installed; bounded server startup and notebook execution remain separate checks |

The active source imports `datasets.ClassLabel`, `datasets.DatasetDict`, `datasets.load_dataset`, and `huggingface_hub.hf_hub_download`. It uses a fixed Hub revision and an explicit local ImageFolder content root because the automatic Hub zip route fails during `ClassLabel` encoding in the verified package. `datasets==5.0.1` and `huggingface-hub==1.26.0` are therefore direct, exact runtime requirements. The pinned path was validated against `Densu341/Fresh-rotten-fruit` in a new virtual environment and empty cache. See [the Hugging Face loading guide](https://huggingface.co/docs/datasets/en/loading) and [dataset.md](dataset.md) for the source contract.

## Dependency policy

| Package | Requirement | Policy | Reason |
| --- | --- | --- | --- |
| `numpy` | `==2.5.1` | exact pin | Directly imported by data, transforms, training, and evaluation; this version passed the current suite. |
| `torch` | `==2.6.0` | exact base pin | Model, engine, loss, trainer, and inference behavior are PyTorch-sensitive. CUDA wheel selection is documented separately. |
| `torchvision` | `==0.21.0` | exact base pin | Direct transform dependency; paired with the verified PyTorch release. |
| `datasets` | `==5.0.1` | exact pin | Verified ImageFolder, DatasetDict, map, split, and feature API behavior. |
| `huggingface-hub` | `==1.26.0` | exact pin | Direct pinned-archive download API used by the dataset loader. |
| `scikit-learn` | `==1.9.0` | exact pin | Direct metric and stratified-fold dependency validated by tests. Its installed metadata accepts NumPy 2.5.1. |
| `matplotlib` | `==3.11.1` | exact pin | Direct notebook plotting import on the verified host. |
| `Pillow` | `==12.3.0` | exact pin | Image handling dependency; tests import it through `PIL`, and the production data path consumes PIL images decoded by Hugging Face Datasets. |
| `tqdm` | `==4.70.0` | exact pin | Direct trainer and inference progress dependency. |
| `jupyterlab` | `==4.6.2` | exact development pin | Needed to open the active notebook and to reproduce the bounded JupyterLab server verification. |
| `ipykernel` | `==7.2.0` | exact development pin | Installed kernel package; bounded JupyterLab server startup passed, but notebook execution was not run. |

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

For the production dataset path, confirm the pinned Hugging Face dependencies after installation:

```powershell
python -c "from datasets import ClassLabel, DatasetDict, load_dataset; print('datasets imports ok')"
```

## Known limitations and next verification

- The production dataset workflow requires network access for its first download or an existing Hugging Face cache. Phase 5.5A established the cold-cache loader proof; the later Phase 5.5 rerun separately verified bounded CMT execution, temporary checkpoint interoperability, and real-holdout evaluation. See [reproducibility.md](reproducibility.md) for the precise evidence boundary.
- The unit suite primarily validates modular behavior with synthetic data. Phase 5.5 rerun added real-data bounded CMT, temporary-checkpoint, and holdout-evaluation evidence; it did not run canonical training, trained-checkpoint evaluation, benchmark reproduction, or full notebook execution.
- Checkpoints, model weights, datasets, caches, and generated results are intentionally not stored in Git.
- A clean virtual-environment installation, production loader, bounded CMT runtime, checkpoint path, and real holdout evaluation passed in Phase 5.5. Full canonical training reproducibility remains unverified.
- A known Codex-managed temporary Git reference may cause `git fetch origin` to fail. It is local tool metadata, not a project dependency or environment problem, and must not be modified.
