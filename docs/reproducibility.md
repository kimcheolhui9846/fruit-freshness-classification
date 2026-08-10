# Reproducibility verification

## Scope and verification boundary

Phase 5.5 rerun verified the committed project in newly created, isolated Windows virtual environments on **2026-08-01**. It verifies installation, the production dataset path, bounded real-data CMT execution, temporary checkpoint interoperability, and the labeled holdout evaluation path. The historical Phase 5.5 evidence did not claim a completed canonical training experiment or a benchmark-quality result.

## Current canonical experiment status (2026-08-10)

A later owner-approved canonical run, `deep3-canonical-reference-01`, completed once with the derived batch-64 configuration. Its fold-best EMA checkpoints were then assessed once against the locked 5,372-example internal holdout. The result is 5,133 / 5,372 top-1 correct predictions (0.955510), macro F1 of 0.903737, and balanced accuracy of 0.899969. The full identity, protocol, metrics, and confusion matrix are recorded in [canonical-holdout-evaluation.md](canonical-holdout-evaluation.md); interpretation and publication limits are in [canonical-results.md](canonical-results.md).

This is not a benchmark, production, external-validation, or independent-machine reproduction claim. Checkpoints, weights, training state, execution logs, raw logits, raw predictions, dataset contents, and other binary artifacts remain local-only and are retained through Phase 8.6.

| Component | Verified value |
| --- | --- |
| Operating system | Windows 11, 64-bit |
| Python | 3.12.10, 64-bit |
| GPU | NVIDIA GeForce RTX 3070 Ti (8 GiB) |
| PyTorch CUDA build / runtime | 2.6.0+cu124 / 12.4 |
| Torchvision | 0.21.0+cu124 |
| NumPy | 2.5.1 |
| Hugging Face `datasets` | 5.0.1 |
| `huggingface-hub` | 1.26.0 |
| scikit-learn | 1.9.0 |
| Pillow | 12.3.0 |
| tqdm | 4.70.0 |
| Matplotlib | 3.11.1 |
| JupyterLab | 4.6.2 |
| ipykernel package | 7.2.0 |

JupyterLab was observed as an unpinned direct dependency in the first clean environment. Its resolved version, 4.6.2, is now explicitly pinned in `requirements-dev.txt`; a second new virtual environment installed that pinned specification successfully. No transitive dependency was pinned.

## Clean environment

Create the environment outside the repository and invoke its Python executable explicitly. The following PowerShell pattern is the tested CUDA workflow:

```powershell
$ReproRoot = Join-Path $env:TEMP ("fruit-freshness-repro-" + [guid]::NewGuid().ToString("N"))
python -m venv (Join-Path $ReproRoot "venv")
$ReproPython = Join-Path $ReproRoot "venv\Scripts\python.exe"

& $ReproPython -m pip install --upgrade pip
& $ReproPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
& $ReproPython -m pip install -r requirements-dev.txt
& $ReproPython -m pip check

& $ReproPython -m unittest discover -s tests -p "test_*.py" -v
& $ReproPython -m compileall src scripts tests
& $ReproPython -m scripts.train --help
& $ReproPython -m scripts.evaluate --help
```

The final verification environment was isolated (`sys.prefix != sys.base_prefix`), used Python 3.12.10, resolved every direct dependency above, and passed `pip check`. The full clean-environment suite passed **119 tests**, with **0 failures** and **0 skips**, before the durable contract tests below were added.

`python -m jupyterlab --version` reported 4.6.2. A temporary JupyterLab server bound only to `127.0.0.1` started successfully with browser launch disabled; a temporary kernel specification referenced the isolated environment's Python executable. No notebook was executed.

## Dataset evidence

The production loader remains `src.datasets.fruit_freshness.load_fruit_freshness_dataset()` and uses:

- Dataset: [`Densu341/Fresh-rotten-fruit`](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit)
- Fixed revision: `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`
- Source archive: `freshness_fruit.zip`
- Archive SHA-256: `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`
- Runtime package: `datasets==5.0.1`

Phase 5.5A established the cold-cache proof: a new environment with an empty external Hugging Face cache downloaded the fixed revision, safely extracted the archive, and built the explicit ImageFolder content root. This rerun used that same revision-specific **external warm cache** after validating the archive size (3,053,594,823 bytes), hash, and managed extraction root. It does not claim a second cold-cache download.

The rerun reproduced the following project contract through the committed API:

| Check | Result |
| --- | ---: |
| Source image rows | 30,357 |
| Filtered rows | 26,858 |
| Train rows | 21,486 |
| Holdout test rows | 5,372 |
| Remapped classes | 14 |
| Split seed | 42 |
| Validation tensor batch | `(192, 3, 224, 224)` `float32` |
| Label tensor dtype | `int64` |

The preserved label order is `freshapples`, `freshbanana`, `freshcapsicum`, `freshcucumber`, `freshoranges`, `freshpotato`, `freshtomato`, `rottenapples`, `rottenbanana`, `rottencapsicum`, `rottencucumber`, `rottenoranges`, `rottenpotato`, and `rottentomato`. Both a train and test image decoded as RGB; a second API call reused the prepared cache. Dataset archives, extracted images, and caches stay outside Git.

## Verification matrix

| Level | Status | Evidence |
| --- | --- | --- |
| Clean environment | Passed | Two new isolated venvs; final venv installed the corrected JupyterLab pin. |
| Dependency installation | Passed | Official cu124 wheels, requirements installation, and `pip check` passed. |
| Project imports | Passed | Runtime and project imports completed without dataset, model, or checkpoint work on import. |
| Full unittest suite | Passed | 119 passed, 0 failed, 0 skipped in the final clean environment before these new offline tests. |
| JupyterLab startup | Passed | JupyterLab 4.6.2 started temporarily on localhost; kernel used the final venv. |
| Production dataset loader | Passed | Fixed revision, safe managed extraction route, counts, labels, RGB, and warm-cache reuse matched. |
| Production transforms | Passed | Train, validation, and fine-tuning transforms produced `float32` `(3, 224, 224)` tensors. |
| Production DataLoaders | Passed | Three stratified folds, 192-sized train/validation/holdout loaders, and a final incomplete batch were checked. |
| Real-data CMT training batch | Passed | Two real samples used the actual factory, CE-with-label-smoothing branch, optimizer, scaler, EMA, scheduler, and `train_one_epoch`. |
| Real-data CMT validation batch | Passed | Two real samples used EMA validation, hflip TTA, `validate_one_epoch`, and metrics. |
| Checkpoint engine round trip | Passed | Actual CMT state keys/order/tensors and fixed-input logits matched after strict map-location load. |
| Inference fold loader | Passed | Three configured temporary fold files loaded in order, in eval mode, with matching logits. |
| Evaluation CLI on real holdout | Passed | `scripts.evaluate` processed all 5,372 examples with three-fold raw-logit hflip TTA. |
| Bounded training CLI | Not run | Not resource-safe on the verified 8 GiB GPU; it is optional for this phase. |
| Full canonical training | Not run | The 3-fold configuration has 120 epochs with the final 20 fine-tuning epochs; it was not authorized. |
| Trained-checkpoint evaluation | Not run | Only temporary compatibility fixtures were created. |
| Benchmark reproduction | Not run | No trained benchmark artifact was produced. |

For the CMT smoke, the first automatic mixed-precision attempt detected overflow and GradScaler reduced its scale from 65,536 to 32,768 without an optimizer step. A second bounded call through the same `train_one_epoch` path then had finite gradients, created optimizer state, updated EMA, and stepped the scheduler. The verification therefore records normal GradScaler scale-back behavior rather than treating the initial skipped step as a model-quality result. The CMT smoke used CUDA, a `(2, 3, 224, 224)` input, a `(2, 14)` validation-logit contract, and a peak allocated CUDA value of 508,085,248 bytes.

The representative 192-example evaluation batch used three temporary CMT fold models with hflip TTA, produced `(192, 14)` logits, took 0.7761 seconds for model work, and peaked at 1,943,769,600 allocated CUDA bytes. The full evaluation command was:

```powershell
python -m scripts.evaluate --config configs/deep3.toml --checkpoint-dir <temporary-checkpoint-directory>
```

It completed the 5,372-example holdout in 28 batches and printed `Final Holdout Acc: 0.09735666418466121` (523/5,372). The checkpoint files were unchanged and no repository result file was created.

## Artifact interpretation

The temporary fold files are **untrained CMT compatibility fixtures**. Their holdout accuracy is only evidence that the model, checkpoint, inference loader, ensemble arithmetic, TTA, labels, and evaluation CLI interoperate. It is not model-quality, benchmark, or trained-checkpoint evidence.

## Remaining limitations

- **Same-machine fresh environment:** verified on the stated Windows/CUDA host.
- **Independent clean machine:** not verified.
- **Cold-cache download:** verified in Phase 5.5A; this rerun was a warm-cache integration check.
- **CPU execution:** not verified for CMT training or evaluation.
- **Full notebook execution:** not run; only bounded JupyterLab server startup was verified.
- **Canonical training:** not run. The configured three folds use 21,486 examples, 192 batch size, 120 epochs, and a final 20-epoch fine-tuning portion. A 192-example backward pass was not attempted because the 8 GiB device did not provide a conservative safety margin from the observed two-sample training-memory behavior.
- **Trained-checkpoint evaluation and benchmark reproduction:** not run.

All temporary virtual environments, Jupyter runtime files, diagnostic scripts, pip-freeze audit output, checkpoints, and data caches are external to the repository and are not committed.