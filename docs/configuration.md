# Experiment configuration

`configs/deep3.toml` is the single active experiment configuration for `deep3.ipynb`. It is parsed by `src.utils.config.load_experiment_config()` with Python 3.12's standard-library `tomllib`; no configuration framework or additional dependency is used.

## Structure

| Section | Purpose |
| --- | --- |
| `runtime` | Existing cuDNN benchmark setting. |
| `loss` | Class-balanced beta, CE/Focal choice, label smoothing, and Focal gamma. |
| `training` / `fine_tuning` | Epoch counts and training batch size. |
| `cross_validation` | Existing fold count, shuffle flag, and split seed. |
| `mixup` | Existing Mixup alpha and probability. |
| `optimization` | CNN/transformer learning rates and weight decay. |
| `ema` | Existing EMA decay. |
| `checkpoint` | Portable final-weight filename only. |
| `reporting` | Plot figure size. TOML loads it as a list; the notebook restores the original tuple at the Matplotlib boundary. |

The notebook loads this file once in `main()` and assigns the existing notebook-facing variable names (`EPOCHS`, `BATCH_SIZE`, `K`, `MIXUP_ALPHA`, and others) before calling completed modules. No completed module receives a configuration object.

## Deliberately derived or fixed values

`num_classes`, class counts, class-balanced alpha, fold indices, checkpoint paths, models, DataLoaders, optimizers, schedulers, EMA objects, metrics, histories, and current epoch/fold remain derived or runtime state. They are not serialized in TOML.

The Hugging Face dataset identifier, excluded labels, split construction, image transforms, CMT architecture values, and fold checkpoint filename template remain encapsulated in previously completed modules because they are not notebook-level inputs. The existing notebook output-directory literal is machine-specific and therefore intentionally remains outside the committed configuration; this document does not repeat it.

## Creating a future experiment configuration

1. Copy `configs/deep3.toml` to one clearly named TOML file under `configs/`.
2. Preserve every required section, key, and scalar type; TOML arrays remain lists unless the notebook explicitly restores a tuple.
3. Intentionally update `CONFIG_PATH` in `deep3.ipynb` to select that one file.
4. Run the configuration tests and full unittest discovery before training.

There is currently no override hierarchy, CLI override, environment-variable override, profile, or interpolation syntax. A future Phase 5.3 training entry point should consume this same explicit configuration contract rather than adding a parallel one.
