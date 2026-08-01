# Dataset pipeline

## Source and reproducibility boundary

The project uses the public [Densu341/Fresh-rotten-fruit](https://huggingface.co/datasets/Densu341/Fresh-rotten-fruit) Hugging Face dataset repository. The loader fixes the source to revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` and downloads only `freshness_fruit.zip` through `huggingface_hub.hf_hub_download`.

The verified archive is 3,053,594,823 bytes and has SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`. It contains 30,357 image files under a single `dataset/` content root:

| Archive directory | Images | Classes |
| --- | ---: | ---: |
| `dataset/Train/` | 23,619 | 18 |
| `dataset/Test/` | 6,738 | 14 |
| Combined content root | 30,357 | 22 |

The source directories are intentionally combined into one ImageFolder `train` split before the project's existing cleanup and deterministic holdout split. This preserves the prior loader contract, which consumed one 30,357-example Hugging Face `train` split rather than the archive directory names as separate project splits.

## Loader compatibility

`datasets==5.0.1` correctly discovers the 22 class names inside the Hub zip, but its automatic Hub ImageFolder route passes the zip's parent (`Fresh-rotten-fruit@<revision>`) as every generated example label. That string is not one of the class names, so `ClassLabel` encoding fails before project preprocessing begins.

`src.datasets.fruit_freshness.load_fruit_freshness_dataset()` resolves this by:

1. Reusing a complete managed extraction cache when available.
2. Downloading the pinned source archive when needed.
3. Safely extracting it under the Hugging Face Datasets cache.
4. Calling `load_dataset("imagefolder", data_dir=<extracted dataset root>)` so each image's parent class directory supplies its label.

No images are copied into the repository. The extraction cache lives under the configured Hugging Face Datasets cache and remains ignored by Git. The loader also rejects zip members that would escape that managed cache directory.

## Cleaning, split, and labels

After ImageFolder construction, the original notebook behavior is unchanged:

- Removed original label IDs: `18`, `20`, `16`, `13`, `2`, `5`, `7`, and `9`.
- Remaining examples: 26,858.
- Holdout split: `test_size=0.2`, `seed=42`.
- Resulting project splits: 21,486 `train` examples and 5,372 `test` examples.
- Image conversion: non-RGB images are converted to RGB before the PyTorch wrapper applies transforms.

The retained label order is:

```text
freshapples, freshbanana, freshcapsicum, freshcucumber, freshoranges,
freshpotato, freshtomato, rottenapples, rottenbanana, rottencapsicum,
rottencucumber, rottenoranges, rottenpotato, rottentomato
```

The dataset wrapper returns `(image, label)` where `label` is a `torch.long` tensor. The verified validation-transform/DataLoader smoke batch has shape `(2, 3, 224, 224)`, image dtype `torch.float32`, and label dtype `torch.int64`.

## Verified clean-environment result

Phase 5.5A validated the loader in a new Windows Python 3.12.10 virtual environment with `datasets==5.0.1`, `huggingface-hub==1.26.0`, `torch==2.6.0+cu124`, and `torchvision==0.21.0+cu124`. The Hugging Face cache was empty before the run. It loaded the pinned archive, preserved all 30,357 source image-to-class assignments, completed the existing filter/remap/RGB pipeline, and produced the split and batch contract above.

The validation did not run CMT construction, checkpoint handling, holdout evaluation, or training.