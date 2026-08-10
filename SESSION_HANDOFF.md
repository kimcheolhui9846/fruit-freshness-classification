# Session Handoff — Fruit Freshness Classification

## Purpose

This file preserves the project state, completed analysis, architecture decisions, and operating constraints so a later Codex session can resume safely.

## Current Repository State

- Active repository: `kimcheolhui9846/fruit-freshness-classification`
- Remote: `https://github.com/kimcheolhui9846/fruit-freshness-classification`
- Active branch: `main`
- Current target commit: `f02433e9ca082d71f75a54577978fe8a9e4b9ca8` (`Add files via upload`)
- Intended local path after rename: `C:\Users\00\grad\fruit-freshness-classification`
- A local safety branch, `backup/before-fruit-freshness-switch-20260729`, preserves the unrelated previous `graduate_project` state. It must not be pushed to this repository.

### Important Local Workspace Anomaly

The repository root currently has an untracked nested directory named `fruit-freshness-classification/`. It is a byte-for-byte duplicate of the outer repository's five tracked files and contains its own `.git` directory. The nested repository is owned by the desktop user and Git reports dubious ownership when inspected from the sandbox.

This nested repository is not part of the desired architecture and must not be committed. It was discovered during the path-rename workflow. Do not remove it without an explicit cleanup instruction and a verified backup.

## Current Tracked Files

```text
README.md
deep.ipynb
deep1.ipynb
deep2.ipynb
deep3.ipynb
```

There are no Python modules, configs, docs, tests, requirements files, `.gitignore`, tracked data, tracked model weights, or deployment files.

## Phase 0 — Completed Audit

### Project Summary

The project is a PyTorch multiclass fruit-freshness classifier. It downloads `Densu341/Fresh-rotten-fruit` using Hugging Face Datasets. The active candidate is `deep3.ipynb`, which implements a custom CMT-like CNN+Transformer model with 3-Fold stratified cross-validation, Mixup, EMA, horizontal-flip TTA, and a final Fold ensemble on a holdout split.

### Dataset Behavior in the Current Notebook

- Loads `dataset["train"]` from `Densu341/Fresh-rotten-fruit`.
- Removes label IDs `[18, 20, 16, 13, 2, 5, 7, 9]`.
- Splits retained data into 80% train/CV and 20% holdout with seed 42.
- Remaps retained labels to contiguous IDs and persists label names as JSON.
- Converts images to RGB.
- Uses ImageNet normalization.

### Current Candidate: `deep3.ipynb`

- Custom CMT-like architecture: CNN stem/stages followed by 256-dimensional and 384-dimensional Transformer stages.
- Transformer blocks contain LPU, multi-head attention, MLP, residual paths, and optional DropPath.
- Train config: 120 epochs, batch size 192, 3 folds, AdamW, CosineAnnealingLR, CE with label smoothing 0.01 by default, Mixup alpha 0.8 at probability 0.5, EMA decay 0.999.
- Final 20 epochs use weaker augmentation and disable Mixup.
- Saves Fold-best EMA state dicts as `best_model_fold{fold}.pt` to a hardcoded Windows path.
- Loads Fold checkpoints and evaluates holdout accuracy using Fold-logit averaging plus horizontal-flip TTA.

### Notebook Version Assessment

| Current file | Intended meaningful version | Status |
|---|---|---|
| `deep.ipynb` | `v0_cmt_focal_baseline` | Legacy; malformed notebook JSON; Fold indices are computed but ignored by DataLoaders |
| `deep1.ipynb` | `v1_cmt_discriminative_lr` | Legacy; malformed notebook JSON; same invalid Fold usage |
| `deep2.ipynb` | `v2_cmt_mixup_label_smoothing_ensemble` | Legacy; malformed notebook JSON; Fold loader may miss checkpoints because it uses global best accuracy |
| `deep3.ipynb` | `v3_cmt_ema_finetune_tta` | Structurally valid and active candidate; uses Fold-local best accuracy |

### Key Audit Findings

- `deep.ipynb`, `deep1.ipynb`, and `deep2.ipynb` fail standard notebook JSON parsing due to malformed quote escaping. Preserve them as historical artifacts; do not treat them as active executables.
- All notebooks have six code cells, zero Markdown cells, and no saved output records. No result values can be asserted from the repository.
- No standalone inference API, deployment code, ONNX export, requirements, tests, docs, or `.gitignore` exist.
- `deep3.ipynb` imports unused items including `torchvision.models`, `load_from_disk`, `Variable`, typing aliases, `classification_report`, and `confusion_matrix`.
- `deep3.ipynb` still uses hardcoded absolute output paths and unconditional CUDA AMP APIs, so portability is weak.
- Model checkpoints contain state dicts only; resolved config and complete checkpoint metadata are not persisted.
- The final raw `last_model_weights.pt` differs semantically from the Fold-best EMA checkpoint artifacts.

## Phase 1 — Approved Architecture Design

The target repository structure is:

```text
configs/    docs/       images/    notebooks/  results/
scripts/    weights/    src/       tests/      reports/
```

`src/` is organized into:

```text
datasets/ models/ losses/ trainers/ engine/
evaluation/ transforms/ utils/ inference/
```

Key policies:

- Datasets, cache, generated weights, and large generated results are never committed.
- `scripts/` are thin CLI entry points; reusable logic belongs in `src/`.
- `notebooks/` are for EDA, audit, error analysis, and thin reproduction. They are not the source of truth for production/research logic.
- `notebooks/archive/` preserves original legacy notebooks.
- Config uses composable YAML: `base`, `data`, `model`, `train`, and `experiment` layers.
- Runs use immutable IDs: `<model-version>__<YYYYMMDD-HHMMSS>__seed-<seed>`.

## Phase 2 — Approved Module Mapping

### Core Source Mapping

| Notebook responsibility | Target module |
|---|---|
| HF download, label removal/remapping, RGB conversion, PyTorch wrapper | `src/datasets/hf_fruit_freshness.py` |
| Train/eval/fine-tuning transforms | `src/transforms/classification.py` |
| Batch Mixup | `src/transforms/mixup.py` |
| CMT layers and classifier | `src/models/cmt_classifier.py` |
| EMA model state | `src/models/ema.py` |
| Model construction | `src/models/factory.py` |
| Focal Loss and class-balanced alpha | `src/losses/focal.py` |
| CE/Focal selection | `src/losses/registry.py` |
| Optimizer and scheduler construction | `src/trainers/optimization.py` |
| Experiment/Fold orchestration | `src/trainers/classification_trainer.py` |
| Batch training loop | `src/engine/train_epoch.py` |
| Validation loop | `src/engine/validate.py` |
| Checkpoint save/load | `src/engine/checkpoint.py` |
| Classification metrics | `src/evaluation/metrics.py` |
| Fold ensemble and holdout evaluation | `src/evaluation/ensemble.py` |
| Horizontal-flip TTA | `src/evaluation/tta.py` |
| History plots | `src/evaluation/reporting.py` |
| Paths, label JSON, runtime and config loading | `src/utils/` |
| Train/evaluate/infer/export entry points | `scripts/` |

### Important Design Decisions

- Do not create a shared `imports.py` or giant `utils.py`.
- Keep CMT primitive layers and `CMTClassifier` together because they are tightly coupled.
- Keep EMA separate from model architecture because it is training state.
- Keep trainer orchestration separate from batch-level engine loops.
- Keep metrics, TTA, ensemble, and plotting separate within `evaluation/`.
- During the first refactor pass, preserve v3 behavior and hyperparameters exactly; do not improve algorithms.

## Status and Next-Step Constraint

Phases 0, 1, and 2 are planning/audit only and are complete. No project refactor has started.

Do not move, rename, delete, or refactor existing project files until the user explicitly authorizes the next implementation phase. The next likely phase is a controlled directory-creation and legacy-preservation plan, followed by behavior-preserving extraction from `deep3.ipynb`.

When resuming, first read this file and check `git status --short --branch`. Confirm the nested duplicate repository is still untracked and avoid committing it accidentally.

## Ongoing Work Agreement

For all future implementation work:

1. Work proceeds in one small, explicitly scoped unit at a time.
2. A unit is considered complete only after proportionate verification has finished.
3. After every completed unit, update this file before starting the next unit.
4. Each update records the changed files, intent, verification performed, Git status implications, unresolved risks, and the exact next recommended unit.
5. Do not silently broaden scope or advance to a new phase without the user's instruction.

## Latest Completed Unit

- **Scope:** Record the agreed incremental-work and handoff-update workflow.
- **Changed file:** `SESSION_HANDOFF.md` only.
- **Verification:** Confirmed the added section is present by reading the document tail.
- **Git impact:** The handoff remains an intentional untracked file; no tracked project code changed.
- **Unresolved risk:** The untracked nested `fruit-freshness-classification/` duplicate repository remains and must not be committed.
- **Next recommended unit:** Await explicit authorization for the first implementation unit; do not begin Phase 3 automatically.

## Completed Unit - Codex MCP Foundation (2026-07-29)

- **Scope:** Configure the local Codex harness for Context7 and Playwright MCP.
- **Modified paths:** `C:/Users/00/.codex/config.toml` and `C:/Users/00/.codex/tools/node-v24.18.0-win-x64/`; this handoff file records the work.
- **MCP configuration:** Context7 uses `https://mcp.context7.com/mcp` with `CONTEXT7_API_KEY`; Playwright uses the verified local Node runtime without changing the global PATH.
- **Verification:** Node.js `v24.18.0`, npm/npx `v11.16.0`, valid TOML parsing, both MCP entries present, and Playwright MCP `--help` succeeded.
- **Credentials:** `CONTEXT7_API_KEY` is absent from process, user, and machine scopes. No blank or fabricated secret was stored.
- **Git impact:** No tracked repository file changed. `SESSION_HANDOFF.md` and the nested duplicate repository remain untracked.
- **Next recommended unit:** Obtain a Context7 API key from the Context7 dashboard, store it in the user-level `CONTEXT7_API_KEY` environment variable, then restart Codex to load both MCP servers.

## Completed Unit - Context7 Credential (2026-07-29)

- **Scope:** Persist the user-supplied Context7 API credential for the configured MCP server.
- **Changed location:** User-level `CONTEXT7_API_KEY` environment variable only; no secret was written to the repository or this document.
- **Verification:** Read the user-scope variable and confirmed a non-empty persisted value without printing it.
- **Security:** The API key must not be committed, added to `.env` in this repository, or pasted into future session notes.
- **Git impact:** No tracked repository file changed; this handoff remains an intentional untracked file.
- **Next recommended unit:** Restart Codex, then confirm that Context7 and Playwright are both available as MCP servers.

## Completed Unit - GitHub Integration Audit (2026-07-29)

- **Scope:** Diagnose GitHub Connect, GitHub CLI, and Git remote authentication without changing accounts or repository state.
- **Codex GitHub Connect:** Not connected. The GitHub app connector returned an explicit not-connected status when querying the target repository.
- **Local GitHub CLI:** `gh` is installed, but its active token is invalid and requires re-authentication.
- **Git remote:** `origin` is the intended repository; host-level `git ls-remote` and non-mutating `git push --dry-run` both succeeded through Git Credential Manager.
- **Network:** GitHub access fails only inside the sandbox; the host network is healthy.
- **Conclusion:** The failed app connector is independent of the working Git credential path. Reconnecting the Codex/ChatGPT GitHub app is required for connector tools.
- **Git impact:** No repository or account state changed; this handoff remains intentionally untracked.
- **Next recommended unit:** Reconnect GitHub Connect in the Codex/ChatGPT UI and authorize the target repository, then re-run the connector query.

## Completed Unit - GitHub CLI Re-authentication (2026-07-29)

- **Scope:** Complete and verify local GitHub CLI web OAuth re-authentication.
- **Changed location:** GitHub CLI credential stored in the local OS keyring; no token was printed or written into the repository.
- **Verification:** Host-level `gh auth status` is authenticated for the active account with `repo` and `workflow` scopes; GitHub API user lookup succeeded.
- **Repository access:** `gh repo view` confirmed `kimcheolhui9846/fruit-freshness-classification`, default branch `main`, and `ADMIN` viewer permission.
- **Sandbox caveat:** The restricted sandbox cannot reliably validate the Windows keyring-backed token; host-level verification is authoritative.
- **Git impact:** No repository file changed; this handoff file remains intentionally untracked.
- **Next recommended unit:** Reconnect the separate Codex/ChatGPT GitHub Connect app, then re-run the connector check.

## Blocked Unit - GitHub Connect Reconnection Attempt (2026-07-29)

- **Scope:** Begin the separate Codex/ChatGPT GitHub Connect reauthorization flow.
- **Attempt:** Confirmed that the current session has no callable browser-control runtime for opening and approving the account-linking UI.
- **Blocker:** GitHub Connect requires an interactive account approval in the Codex/ChatGPT UI; it cannot be repaired through local Git or `gh` credentials.
- **Environment note:** The current session was started before the updated local MCP/browser setup, so a restart is required before those controls can load.
- **Git impact:** No repository or account state changed during this attempt.
- **Next recommended unit:** Restart Codex, reconnect GitHub Connect in the UI, authorize the target repository, and then ask for a connector re-check.

## Completed Unit - GitHub Connect Verification (2026-07-29)

- **Scope:** Verify the reconnected Codex/ChatGPT GitHub Connect app against the target repository.
- **Verification:** The GitHub connector successfully returned the installed repository record without an authentication error.
- **Repository:** `kimcheolhui9846/fruit-freshness-classification`, public, active, default branch `main`.
- **Permissions:** Connector reports admin, maintain, push, pull, and triage permissions.
- **Git impact:** No repository file or GitHub resource was modified; this handoff remains intentionally untracked.
- **Next recommended unit:** GitHub Connect, local Git, and GitHub CLI are ready; await the next authorized repository task.

## Completed Unit - Phase 3 Directory and Package Structure (2026-07-31)

- **Scope:** Create only the approved empty repository directories, Python package markers, and Git ignore policy. No notebook migration or implementation extraction was performed.
- **Created directories:** `assets`, `configs`, `docs`, `notebooks`, `results`, `scripts`, `tests`, `weights`, `src`, and the nine approved `src` subdirectories.
- **Created files:** `.gitignore` plus ten marker-only `__init__.py` files under `src/` and its approved subpackages.
- **Git policy:** `.gitignore` now ignores datasets/caches, Python artifacts, model checkpoints and label metadata, generated results/logs, local environments, and editor/system files. No `.gitkeep` was added.
- **Preservation:** SHA-256 verification confirmed all four legacy notebooks are unchanged. No tracked file was moved, renamed, deleted, or modified.
- **Known deviation:** Earlier Phase 1 notes mention `images/` and `reports/`; the newer explicit Phase 3 approval specifies `assets/` and omits those directories, so only `assets/` was created.
- **Nested repository:** The untracked nested `fruit-freshness-classification/.git` repository remains preserved and must not be committed.
- **Verification:** `git diff --stat`, `git diff`, and tracked-file name-status are empty because all new repository artifacts are untracked; no implementation code exists.
- **Next recommended unit:** The repository is structurally ready for the approved Phase 4 reusable-module extraction, subject to a separate explicit instruction and preservation of `deep3.ipynb` behavior.

## Completed Unit - Post-Phase-3 Git Ignore Audit (2026-07-31)

- **Scope:** Run the requested `git check-ignore -v` checks against source packages and generated-artifact directories.
- **Finding:** `.gitignore` rule `datasets/` incorrectly ignores `src/datasets/__init__.py` because the pattern is not root-anchored.
- **Passing checks:** `src/models/__init__.py`, `results/`, and `weights/` are not ignored; `results/*` and `weights/*` correctly ignore generated contents.
- **Impact:** The current Git policy violates the approved rule that source code must not be ignored.
- **Changes made:** None. This was a read-only audit; no source, notebook, or Git policy file was modified.
- **Recommended correction:** Replace `datasets/` with `/datasets/` so only a repository-root dataset directory is ignored. Consider similarly anchoring other root-only data/cache rules during the correction review.
- **Phase status:** Phase 3 needs this small `.gitignore` correction before its Git policy can be considered fully compliant.

- **Next recommended unit:** Await explicit authorization to correct `.gitignore`, then re-run the ignore verification.

## Completed Unit - Phase 3.6 Git Ignore Source Package Fix (2026-07-31)

- **Scope:** Apply the approved minimal correction for the source-package ignore conflict.
- **Exact change:** Replaced `datasets/` with `/datasets/` on line 3 of `.gitignore`; no other ignore rule changed.
- **Source verification:** `src/datasets/__init__.py` and `src/models/__init__.py` are no longer ignored and are trackable.
- **Artifact verification:** `datasets/example.bin`, `results/example.json`, and `weights/example.pth` remain ignored by the approved policy.
- **Safety verification:** All four notebooks remain unchanged; no Python implementation, move, rename, deletion, commit, or push occurred.
- **Git impact:** `.gitignore` remains untracked pending a future intentional commit; existing handoff, source-package, and nested-repository untracked state remains preserved.
- **Next recommended unit:** The Git ignore blocker is resolved. Do not begin Phase 4 until separately instructed.

## Completed Unit - Phase 4.1 Runtime, Path, and Label Utilities (2026-07-31)

- **Scope:** Extract only CUDA-first device resolution, output-directory preparation, and label-name JSON persistence from `deep3.ipynb`. No training, dataset download, or future Phase 4 work was performed.
- **Created files:** `src/utils/runtime.py` (`resolve_device`), `src/utils/paths.py` (`ensure_output_directory`, `build_fold_checkpoint_path`), and `src/utils/labels.py` (`save_label_names`). `src/utils/__init__.py` remains a package marker so notebook imports are explicit.
- **Notebook migration:** `deep3.ipynb` cells 0, 1, 4, and 5 now explicitly import and call the utilities. The CUDA-versus-CPU condition, device printing, `C:/Users/user/Desktop/deep/model_data` path, `exist_ok=True`, Fold checkpoint filename, `label_names.json` filename, UTF-8 encoding, and `ensure_ascii=False` are unchanged. Checkpoint serialization remains in the notebook; only its shared filename construction is delegated to `utils`.
- **Verification:** Host Python import smoke test and runtime parity passed (`torch.device`, CUDA on the verification host). Temporary-directory checks passed for string path return and directory creation, and for ordered Unicode label JSON round-trip. The notebook remains valid JSON with six code cells and unchanged outputs; cells 2-3 match the baseline exactly, and cells 4-5 differ only in the approved utility calls. SHA-256 hashes confirm `deep.ipynb`, `deep1.ipynb`, and `deep2.ipynb` remain unchanged. New utility files are trackable; `.gitignore` has no diff.
- **Git impact:** `deep3.ipynb` is modified; the three utility modules are new under the existing untracked `src/` tree. No file was moved or deleted, no commit or push was made, and the nested duplicate repository remains clean and untouched.
- **Next recommended unit:** Await explicit authorization for Phase 4.2 - Extract Classification Transforms and Mixup. Do not begin it automatically.

## Completed Unit - Phase 4.2 Classification Transforms and Mixup (2026-07-31)

- **Scope:** Extract only the active training, validation, and fine-tuning `torchvision.transforms` pipelines plus the existing Mixup helper from `deep3.ipynb`. No dataset, DataLoader, model, loss, optimizer, scheduler, engine, evaluation, or inference code changed.
- **Created files:** `src/transforms/classification.py`, `src/transforms/mixup.py`, `tests/transforms/test_classification.py`, and `tests/transforms/test_mixup.py`. The existing `src/transforms/__init__.py` remains a package marker; notebook imports are explicit.
- **Notebook migration:** Cells 0, 1, 4, and 5 import and call `build_train_transform`, `build_validation_transform`, `build_finetune_transform`, and `mixup_data`. Inline active transform and Mixup definitions were removed. Transform ordering, parameters, defaults, NumPy Beta sampling, PyTorch permutation, device handling, and return order are preserved.
- **Verification:** Import smoke test passed. Five CPU-compatible `unittest` parity tests passed, covering transform structure, output shape/dtype, deterministic validation output, seeded train/fine-tune output parity, Mixup alpha <= 0 behavior, and seeded positive-alpha tensor/target/lambda parity. `deep3.ipynb` remains valid with six cells and preserved outputs; cells 2-3 match the Phase 4.1 baseline exactly. SHA-256 checks confirm all historical notebooks are unchanged. Source files are trackable and `.gitignore` is unchanged.
- **Implementation commit:** `b96863e` - `refactor: extract classification transforms and mixup`.
- **Implementation push:** `origin/main` advanced from `c8e3ea9` to `b96863e`; local HEAD matched `origin/main` after the push.
- **Remaining items:** The preserved nested `fruit-freshness-classification/` repository remains untracked and untouched. Local Python `__pycache__` directories remain ignored and uncommitted.
- **Next recommended unit:** Await explicit authorization for Phase 4.3 - Extract Dataset Preparation and DataLoader Pipeline. Do not begin it automatically.


## Completed Unit - Phase 4.3 Dataset Preparation and DataLoader Pipeline (2026-07-31)

- **Scope:** Extract only the active Hugging Face dataset preparation, label filtering/remapping, RGB normalization, stratified fold selection, subset construction, and DataLoader construction from `deep3.ipynb`. Model, loss, optimizer, scheduler, training, validation, evaluation, inference, transforms, and Mixup logic were not changed.
- **Created files:** `src/datasets/fruit_freshness.py`, `src/datasets/folds.py`, `src/datasets/loaders.py`, `tests/datasets/test_fruit_freshness.py`, `tests/datasets/test_folds.py`, `tests/datasets/test_loaders.py`, and `tests/datasets/test_notebook_data_pipeline.py`.
- **Modified file:** `deep3.ipynb` only among notebooks. The active `prepare_dataset` function and `FruitHFDataset` class were removed from cells 1-2; cells 0 and 5 now use explicit `src.datasets` imports and calls.
- **Public APIs:** `load_fruit_freshness_dataset`, `FruitHFDataset`, `iter_stratified_folds`, `select_fold_datasets`, `build_fold_dataloaders`, and `build_holdout_dataloader`.
- **Behavior preserved:** Dataset `Densu341/Fresh-rotten-fruit`; removed-label list; 80/20 split with seed 42; label-name order and remapping; ClassLabel metadata; RGB conversion; item tuple and long target dtype; K-fold shuffle/seed; subset index order; train/validation/holdout loader batch, shuffle, worker, pin-memory, and default loader behavior. No new random seeds, samplers, caching, or data sources were added.
- **Verification:** Eight focused tests passed (dataset/remapping/item contract, fold parameters/index order, loader parameters/batches, and notebook data/transform orchestration). Dataset module syntax, notebook JSON, notebook source compilation, and scope comparison passed. SHA-256 checks confirm `deep.ipynb`, `deep1.ipynb`, and `deep2.ipynb` remain unchanged. Source modules are trackable; `.gitignore` and Phase 4.2 transform code have no diff. The protected nested repository remains clean and untracked.
- **Production integration status:** Not executed. The local Python environment lacks the `datasets` and `scikit-learn` packages, so no production dataset was downloaded and the direct production import smoke test is environment-blocked. Dependency-isolated synthetic tests and structural parity checks passed; this is not an implementation workaround.
- **Implementation commit:** `9f64a3a` - `refactor: extract dataset and dataloader pipeline`.
- **Implementation push:** Succeeded to `origin/main` (`470e704..9f64a3a`).
- **Known Git limitation:** A Codex-managed temporary ref under `.git/refs/codex/turn-diffs/` may make `git fetch origin` fail. Do not modify it; use the documented local/remote/GitHub CLI SHA fallback if needed.
- **Remaining items:** Ignored Python caches and the protected untracked nested repository remain local only. No dataset, Hugging Face cache, checkpoint, weight, output, or log was staged.
- **Next recommended unit:** Phase 4.3 documentation commit and synchronization verification only; do not begin Phase 4.4 automatically.


## Completed Unit - Phase 4.4 Model Architecture and Construction (2026-07-31)

- **Scope:** Extract only the active custom CMT architecture, its primitive layers, and direct model construction from `deep3.ipynb`. EMA remains in the notebook as training state; loss, optimizer, scheduler, checkpoint implementation, training, validation, evaluation, TTA, ensemble, inference, datasets, transforms, and Mixup were not changed.
- **Architecture:** `CMTClassifier` is the notebook's custom CNN-plus-transformer classifier with `DropPath`, depthwise LPU, multi-head attention, MLP, and the original classifier head. It does not use timm, pretrained weights, parameter freezing, or a model registry.
- **Created files:** `src/models/cmt_classifier.py`, `src/models/factory.py`, `tests/models/test_architecture.py`, and `tests/models/test_notebook_model_pipeline.py`. `src/models/__init__.py` remains a package marker so imports stay explicit.
- **Modified file:** `deep3.ipynb` only among notebooks. Cell 0 imports `build_cmt_classifier`; cell 2 retains `ModelEma` but removes active CMT definitions; cells 3 and 4 retain their checkpoint/inference and training orchestration while delegating only model construction to the factory.
- **Public APIs:** `CMTClassifier` and `build_cmt_classifier(num_classes)`.
- **Compatibility verification:** Legacy code is loaded read-only from the Phase 4.3 notebook commit `7eb6e2a`. Constructor signature, child-module names/order, state-dict keys/order/shapes/dtypes/values, named-parameter order, named-buffer order, total and trainable parameter counts, requires-grad flags, CPU initialization RNG consumption, evaluation forward output, seeded training forward output, and bidirectional `strict=True` in-memory state-dict loading all match exactly.
- **Checkpoint format:** The notebook saves only state dictionaries: per-fold `ema.module.state_dict()` and final `model.state_dict()`. It loads fold state dictionaries with the default strict behavior. No full-model pickle is used, no on-disk checkpoint is locally available, and no checkpoint was generated or committed.
- **Verification:** Model import smoke test, AST syntax validation, nine focused unittest parity tests, notebook JSON/source comparison, and historical notebook SHA-256 preservation all passed. `.gitignore`, completed datasets, transforms, Mixup, and utilities have no diff; source files are trackable and the nested repository is clean and untouched.
- **Pretrained and CUDA status:** Pretrained-weight integration was not applicable because the active constructor has no pretrained path. CUDA is available, but constructor initialization does not create CUDA tensors or consume CUDA RNG; CPU construction RNG parity is the applicable check.
- **Implementation commit:** `40a354b` - `refactor: extract model architecture and construction`.
- **Implementation push:** Succeeded to `origin/main` (`7eb6e2a..40a354b`); local implementation HEAD matched `origin/main` after push.
- **Known Git limitation:** A Codex-managed temporary ref under `.git/refs/codex/turn-diffs/` may make `git fetch origin` fail. Do not modify it; use the documented local/remote/GitHub CLI SHA fallback if needed.
- **Remaining items:** Ignored Python caches and the protected untracked nested repository remain local only. No weight, checkpoint, output, cache, or dataset was staged.
- **Next recommended unit:** Phase 4.4 documentation commit and synchronization verification only; do not begin Phase 4.5 automatically.

## Completed Unit - Phase 4.5 Loss Functions and Loss-Related Helpers (2026-07-31)

- **Scope:** Extract only the active Focal Loss, class-balanced alpha construction, and Mixup loss combination from `deep3.ipynb`. Dataset/DataLoader, transforms and `mixup_data`, model, EMA, optimizer, scheduler, checkpoint implementation, training, validation, evaluation, inference, and hyperparameters were not changed.
- **Created files:** `src/losses/focal.py` (`FocalLoss`, `build_class_balanced_alpha`), `src/losses/mixup.py` (`mixup_criterion`), and three focused tests under `tests/losses/`.
- **Notebook migration:** `deep3.ipynb` now imports the extracted helpers, delegates only the class-balanced alpha formula and Mixup loss expression, and removes only the inline Focal Loss class. Cross-entropy construction, Focal Loss constructor arguments, non-Mixup loss path, data counting, and the existing training orchestration remain in the notebook.
- **Behavior preserved:** The Focal Loss formula and operation order, `alpha` buffer behavior, gamma/reduction/epsilon semantics, class-balanced-alpha formula, CrossEntropy label smoothing path, and `lam * criterion(out, y_a) + (1-lam) * criterion(out, y_b)` behavior are preserved exactly.
- **Verification:** 11 loss-focused parity tests passed, including values, gradients, reductions, alpha variants, invalid-index/broadcast behavior, CPU/CUDA parity, class-alpha construction, CrossEntropy/Focal Mixup parity, and notebook orchestration. Existing regressions also passed: 9 model tests, 5 transform tests, and 8 dataset tests. Notebook JSON/source validation, historical notebook SHA-256 preservation, source tracking, and staged whitespace checks passed.
- **Environment note:** Loss tests ran successfully. The existing production dataset integration limitation remains: local `datasets` and `scikit-learn` packages are not installed, so no Hugging Face download was attempted.
- **Implementation commit:** `531fc19` - `refactor: extract loss functions and helpers`, pushed to `origin/refactor/phase-4.5-losses` and verified against GitHub.
- **Safety:** No dataset, weight, checkpoint, output, cache, notebook checkpoint, or nested repository content was staged. The protected nested `fruit-freshness-classification/` repository remains clean and untracked. The known Codex-managed temporary-ref `git fetch` limitation remains untouched.
- **Next recommended unit:** After this phase is fully synchronized, await separate authorization for Phase 4.6 (checkpointing, EMA, and training-engine foundations). Do not begin it automatically.

## Completed Unit - Phase 4.6 Checkpointing, EMA, and Training Engine Foundations (2026-07-31)

- **Scope:** Extract only the active EMA object, state-dict checkpoint primitives, AdamW construction, and CosineAnnealingLR construction from `deep3.ipynb`. Training/validation loops, fold and epoch orchestration, metrics, evaluation, inference, TTA, ensemble, best-model policy, save conditions, update/step timing, and hyperparameters remain in the notebook.
- **Created files:** `src/engine/ema.py`, `src/engine/checkpoint.py`, `src/engine/optimization.py`, and focused tests `tests/engine/test_ema.py`, `test_checkpoint.py`, `test_optimization.py`, and `test_notebook_engine_pipeline.py`.
- **Modified files:** `src/engine/__init__.py` now exposes explicit engine APIs; `deep3.ipynb` imports and calls the engine helpers; `tests/models/test_notebook_model_pipeline.py` updates only its obsolete expectation that EMA must remain notebook-local, while retaining the model-construction assertions.
- **Public APIs:** `ModelEma`, `save_model_state(model, path)`, `load_model_state(model, path, *, map_location=None)`, `build_optimizer(model, lr_cnn, lr_trans, weight_decay)`, and `build_scheduler(optimizer, t_max)`.
- **EMA compatibility:** Constructor/interface, `module` path, deepcopy timing, eval mode, decay/device attributes, state-dict keys and ordering, parameters, buffers, initial copied values, CPU/CUDA placement, no-grad update behavior, first/repeated `update`, and `set` behavior match the Phase 4.5 legacy notebook exactly. This includes floating and integer buffer treatment.
- **Checkpoint compatibility:** Active checkpoints remain raw model state dictionaries only: best Fold `ema.module.state_dict()` and final `model.state_dict()`. Save/load round trips work in both legacy-to-helper and helper-to-legacy directions; default strict loading and map-location behavior remain unchanged. There is no active payload/resume/optimizer/scheduler checkpoint format, and no checkpoint was generated or committed.
- **Optimization compatibility:** The helper preserves the two original AdamW parameter groups and their module/parameter order (`stem` through `stage3`, then token/transformer/head modules), learning rates, and weight decay. Optimizer class, defaults, groups, state before/after a synthetic step, and CMT group ordering match.
- **Scheduler compatibility:** The helper constructs the original `CosineAnnealingLR(optimizer, T_max=EPOCHS)` only. `scheduler.step()` remains at its original end-of-epoch notebook location; controlled LR sequences and scheduler state match.
- **GradScaler boundary:** `GradScaler()` remains directly in the notebook because its useful boundary is the AMP training loop; no autocast, scale, step, update, or clipping orchestration was extracted.
- **Verification:** 15 focused engine tests passed, including CUDA EMA parity. Completed regressions passed: 11 loss, 9 model, 5 transform, and 8 dataset tests. Engine imports, Python syntax, notebook JSON/source compilation, exact approved cell-boundary comparison, historical notebook SHA-256 preservation, source tracking, whitespace, protected-module/.gitignore diff, and nested-repository checks passed.
- **Environment note:** No packages were installed and no full training or production dataset download was run. The pre-existing local `datasets` and `scikit-learn` absence continues to block only production dataset integration, not engine-module testing.
- **Implementation commit:** `700f2ad` - `refactor: extract training engine foundations`, pushed to `origin/refactor/phase-4.6-engine-foundations` and verified against GitHub.
- **Merge plan:** Keep the Phase branch for audit, then fast-forward only into `main` after verifying both local and GitHub `main` remain at `c2eebb1`.
- **Safety:** No dataset, weight, generated checkpoint, output, cache, notebook checkpoint, `.gitignore`, historical notebook, nested repository, or Codex-managed temporary ref was staged or modified.
- **Next recommended unit:** After Phase 4.6 synchronization, await separate authorization for Phase 4.7 - Extract Training and Validation Loops. Do not begin it automatically.

## Completed Unit - Phase 4.7 Training and Validation Loops (2026-07-31)

- **Scope:** Extract only the active reusable single-epoch training and EMA-validation loops from `deep3.ipynb`. Fold/epoch orchestration, fine-tuning transition, model/criterion/optimizer/scheduler/scaler construction, history handling, validation summaries, best-score policy, checkpoint decisions, scheduler stepping, ensemble loading, holdout evaluation, TTA orchestration, and reporting remain in the notebook.
- **Created files:** `src/trainers/loops.py` and focused tests `tests/trainers/test_train_epoch.py`, `test_validate_epoch.py`, and `test_notebook_training_pipeline.py`.
- **Modified files:** `src/trainers/__init__.py` exposes explicit loop APIs; `deep3.ipynb` imports and calls them; the existing engine/loss notebook-boundary tests update only obsolete assumptions that active batch loops and Mixup criterion calls must remain notebook-local.
- **Public APIs:** `train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, ema, is_finetuning, mixup_probability, mixup_alpha, progress_description)` returns `(tr_acc, tr_loss)` and preserves the existing train summary print. `validate_one_epoch(model, dataloader, criterion, device, progress_description)` returns `(va_acc, va_loss, all_preds, all_labels, all_logits)`; notebook-local sklearn metric calculations remain after its call.
- **Training behavior:** The exact order remains zero-grad with `set_to_none=True`, CUDA autocast, forward/loss, scaler scale/backward, scaler step, scaler update, EMA update, batch-size-weighted loss accumulation, prediction accumulation, and progress output. Fine-tuning still disables Mixup before the same batch loop; normal training still uses NumPy `np.random.rand()` before `mixup_data` with the original alpha.
- **Validation behavior:** The notebook still selects `ema.module`; the trainer remains EMA-agnostic, sets the supplied model to eval, uses `torch.inference_mode()` and the original CUDA autocast/TTA expression, then returns loss/accuracy plus ordered prediction, target, and logit collections. F1, balanced accuracy, top-k, history, and print behavior remain notebook orchestration.
- **Boundaries:** `scheduler.step()` remains at the original end-of-epoch location after checkpoint policy. Trainers neither save/load checkpoints nor compare best scores, choose filenames, transition fine-tuning, or perform final evaluation.
- **Parity:** Test-only legacy loop references are dynamically loaded from notebook commit `0f89baa`. Identical synthetic epochs preserve return values, model/optimizer/scaler/EMA states, criterion target-call order, optimizer/scaler/EMA operation order, progress description, print output, Mixup behavior, final incomplete batch weighting, and Python/NumPy/PyTorch/CUDA RNG ending states. Validation preserves return ordering, logits/predictions/targets, model state, eval mode, no-grad behavior, progress, and RNG state.
- **Verification:** Seven trainer-focused tests passed, including actual CUDA AMP/GradScaler parity. Completed regressions passed: 15 engine, 11 loss, 9 model, 5 transform, and 8 dataset tests. Trainer import/syntax checks, notebook JSON/source compilation, exact approved cell-boundary comparison, historical notebook SHA-256 preservation, source tracking, whitespace, protected source/.gitignore diff, and nested-repository checks passed.
- **Execution status:** No package was installed, no production dataset download, full CMT training, or end-to-end notebook execution was run. The current environment lacks `datasets` and `scikit-learn`; synthetic dependency-isolated single-epoch parity is complete, while production-data integration remains environment-blocked.
- **Implementation commit:** `207a931` - `refactor: extract training and validation loops`, pushed to `origin/refactor/phase-4.7-training-loops` and verified against GitHub.
- **Merge plan:** Retain the Phase branch for audit and fast-forward only into `main` after verifying local and GitHub `main` remain at `0f89baa`.
- **Safety:** No dataset, weight, checkpoint, output, cache, notebook checkpoint, `.gitignore`, historical notebook, nested repository, or Codex-managed temporary ref was staged or modified.
- **Next recommended unit:** After Phase 4.7 synchronization, await separate authorization for Phase 4.8 - Extract Evaluation and Metric Utilities. Do not begin it automatically.

## Completed Unit - Phase 4.8 Evaluation and Metric Utilities (2026-07-31)

- **Scope:** Extract only the active notebook-level validation metric calculations from `deep3.ipynb`: macro F1, balanced accuracy, and logits-based top-2/top-3 accuracy. Model forward execution, DataLoader iteration, validation TTA, EMA selection, checkpoint loading and policy, fold/epoch orchestration, holdout evaluation, ensemble execution, and reporting remain notebook-local.
- **Created files:** `src/evaluation/metrics.py`, `tests/evaluation/test_metrics.py`, and `tests/evaluation/test_notebook_evaluation_pipeline.py`.
- **Modified files:** `src/evaluation/__init__.py` exports the explicit metric API, and `deep3.ipynb` imports and calls it. Only notebook Cell 0 (one import boundary) and Cell 4 (the pure metric block after logits concatenation) changed; no outputs, cell IDs, historical notebooks, or unrelated cells changed.
- **Public API:** `compute_validation_metrics(labels, predictions, logits)` returns `(va_f1, va_bal, va_top2, va_top3)` without type coercion or rounding.
- **Metric contract:** Preserves exact scikit-learn calls and order: `f1_score(labels, predictions, average="macro")`, `balanced_accuracy_score(labels, predictions)`, then `top_k_accuracy_score` for `k=2` and `k=3`. The original bare `except` remains behaviorally equivalent: a top-k error sets both top-k results to `None`.
- **Logits and label order:** The notebook still performs `all_logits = np.concatenate(all_logits, axis=0)` before the call. Logits remain unnormalized class scores; top-k still receives `labels=np.arange(logits.shape[1])`, preserving the dataset integer label order, missing-class behavior, tie behavior, scalar types, warnings, and exception path. No softmax, prediction extraction, device transfer, or NumPy/list conversion was moved.
- **Inactive candidates:** `classification_report`, `confusion_matrix`, precision, recall, accuracy_score, fold-summary aggregation, and report formatting have no active notebook calculation in this version, so no report, matrix, or aggregation helper was created.
- **Boundaries:** Validation-loop accuracy/argmax and TTA remain in `src/trainers/loops.py` under the existing Phase 4.7 API. The final holdout `argmax`, ensemble model iteration, horizontal-flip TTA, and `Final Holdout Acc` print remain in the notebook.
- **Parity:** Test-only legacy references reproduce the pre-Phase 4.8 Cell 4 metric region. Perfect, all-incorrect, imbalanced, missing-prediction-class, missing-label-class, all-class, single-sample, top-k success, top-k warning, and two-class top-k failure cases preserve values, scalar types, warnings, and `None` fallback behavior. The notebook boundary test verifies execution, checkpoint, scheduler, holdout, ensemble, and print anchors remain in the original order.
- **Verification:** Six evaluation tests passed. Regressions passed: 7 trainer, 15 engine, 11 loss, 9 model, 5 transform, and 8 dataset tests. Evaluation import smoke, Python syntax compilation, notebook JSON/source compilation, exact allowed Cell 0/4 boundary comparison, historical notebook SHA-256 preservation, source tracking, protected-path/.gitignore diff, whitespace, and nested-repository checks passed.
- **Dependency status:** User explicitly authorized local installation of `scikit-learn`; the active Python environment now uses `scikit-learn 1.9.0`. No repository dependency declaration changed. `datasets` remains unavailable in the active environment, but evaluation tests do not import dataset loading.
- **Execution status:** No production dataset download, full notebook run, fold training, checkpoint loading, or holdout/ensemble evaluation was performed. Numerical metric parity is complete with actual scikit-learn; production evaluation remains data/environment dependent.
- **Implementation commit:** `6f1159f` - `refactor: extract evaluation metrics`, pushed and verified at `origin/refactor/phase-4.8-evaluation-metrics` / GitHub SHA `6f1159ff4e4e9f30c7e22f345609510441ba5df7`.
- **Merge plan:** Retain the Phase branch for audit and fast-forward only into `main` after verifying local and GitHub `main` remain at Phase 4.7 SHA `fec42a2`.
- **Known limitation:** `git fetch origin` may fail because of the external Codex-managed `.git/refs/codex/turn-diffs/` bad-object reference. Do not repair, prune, pack, delete, or otherwise modify it; use GitHub API SHA verification as fallback.
- **Next recommended unit:** After Phase 4.8 synchronization, await separate authorization for Phase 4.9 - Extract Inference, TTA, and Ensemble Pipeline. Do not begin it automatically.

## Completed Unit - Phase 4.9 Inference, TTA, and Ensemble Pipeline (2026-07-31)

- **Scope:** Extract only active post-training fold-checkpoint loading, raw-logit ensemble inference, horizontal-flip TTA, and the final holdout counting loop from `deep3.ipynb`. Dataset creation, DataLoader construction, transforms, model architecture/factory, losses, EMA, checkpoint primitives/policy, optimizer/scheduler, fold/epoch orchestration, training, validation, and `src/evaluation/metrics.py` were not changed. The Phase 4.7 validation TTA remains intentionally duplicated in `src/trainers/loops.py` to avoid changing its established behavior.
- **Created files:** `src/inference/loading.py`, `src/inference/ensemble.py`, and four focused tests under `tests/inference/`. `src/inference/__init__.py` now exposes explicit inference APIs.
- **Notebook migration:** Only Cells 0, 3, and 4 changed. Cell 0 imports `load_fold_models` and `run_ensemble_holdout`; Cell 3 delegates the former inline inference helpers to `src.inference`; Cell 4 retains checkpoint directory assignment, model loading call, holdout dataset/loader creation, final evaluation header/comment, and `Final Holdout Acc` reporting while delegating the exact counting loop.
- **Public APIs:** `load_fold_models(num_folds, num_classes, device, ckpt_dir)`, `ensemble_logits(models, x)`, `ensemble_logits_tta_hflip(models, x)`, and `run_ensemble_holdout(models, dataloader, device) -> (t_correct, t_total)`.
- **Checkpoint-loading contract:** Folds are processed in ascending `range(1, num_folds + 1)` order. Each model is built with `build_cmt_classifier(num_classes).to(device)`, loaded from the pre-existing `best_model_fold{fold}.pt` path via `load_model_state(..., map_location=device)` (default strict state-dict behavior), set to eval, and appended without missing-checkpoint recovery or reordering.
- **Inference contract:** The ensemble averages raw logits, starting with `logits_sum = 0`, in supplied list order, with no softmax, weighting, or probability conversion. TTA applies `torch.flip(x, dims=[3])`, evaluates original then flipped inputs, and returns `(logits + logits_flip) / 2`; both helpers retain `torch.inference_mode()`.
- **Final holdout contract:** The reusable loop preserves DataLoader iteration order, `x.to(device)` / `y.to(device)`, CUDA autocast placement, `argmax(1)`, correct-count and total-count updates, and the notebook’s final `t_correct / t_total` print. It does not collect labels, predictions, probabilities, or extra metrics.
- **Verification:** The Phase 4.9 focused synthetic CPU/CUDA parity suite previously passed (12 tests covering loading, strict/missing checkpoint behavior, raw-logit order, hflip dimension/formula, final incomplete batch/counts, no-grad, RNG, notebook boundaries) together with prior module regressions: evaluation 6, trainers 7, engine 15, losses 11, models 9, transforms 5, and datasets 8. After a final non-functional Cell 3 comment-newline normalization, notebook JSON/source compilation, inference source compilation, exact intended Cell 0/3/4 boundary validation, historical notebook preservation, protected-module comparison, source tracking, whitespace, and nested-repository checks passed. A new full `pytest` rerun could not start because the currently discoverable Python executables contain no `pytest`; no package was installed or modified. No production dataset, checkpoint, full notebook, or holdout evaluation was executed.
- **Implementation commit:** `089f433a9cdd123ca9e865ae11c1e89099932041` - `refactor: extract inference and checkpoint ensemble pipeline`, pushed to `origin/refactor/phase-4.9-inference-ensemble`; local and remote phase-branch HEAD matched immediately after push.
- **Merge plan:** Retain the Phase branch for audit. Fast-forward only after documenting this entry and verifying both local and GitHub `main` remain at `6e0bcca`.
- **Safety:** No dataset, Hugging Face cache, weight, generated checkpoint, output, log, notebook checkpoint, `.gitignore`, historical notebook, nested repository content, or Codex-managed temporary ref was staged or modified. The untracked nested `fruit-freshness-classification/` repository remains clean and untouched.
- **Known limitation:** `git fetch origin` may fail because of the external Codex-managed `.git/refs/codex/turn-diffs/` bad-object reference. Do not repair, prune, pack, delete, or otherwise modify it; use GitHub API SHA verification as fallback.
- **Next recommended unit:** This Phase 4.9 handoff documentation commit and fast-forward synchronization only. Do not begin a later refactoring phase automatically; await explicit authorization for any subsequent orchestration/configuration work.

## Completed Unit - Phase 4.10 Notebook Orchestration Cleanup and Integration Verification (2026-07-31)

- **Scope:** Clean only `deep3.ipynb` orchestration boundaries and add integration/architecture coverage. No model, transform, dataset, label mapping, loss, EMA, checkpoint, optimizer/scheduler, training, validation, evaluation, inference/TTA/ensemble, hyperparameter, configuration, script, `.gitignore`, historical-notebook, nested-repository, or Codex-ref behavior changed.
- **Notebook responsibility audit:** Cell 0 is explicit imports; Cell 1 creates the retained setup globals (`device`, train/validation transforms); Cell 2 retains the pre-existing `main()` experiment orchestration. The main cell still owns dataset acquisition, hyperparameters, fold/epoch loops, criterion/model/optimizer/scheduler/scaler creation, history, fine-tuning transition, checkpoint policy, inference setup, and final plots/reporting. There are no active duplicate module classes/functions or inline DataLoader/optimizer/checkpoint/metric/TTA/ensemble implementations.
- **Notebook cleanup:** Removed one empty output-free code cell and one output-free inference comment-only code cell. Removed only unused direct imports from Cell 0, including imports now owned and used inside extracted modules. No stored output or executable orchestration source was altered; notebook code-cell count is now three.
- **Tests:** Added `tests/integration/test_modular_pipeline.py` (synthetic transform contract, optimizer/scheduler, EMA, train/validation, metrics, temporary fold checkpoint loading, raw ensemble, hflip TTA, and holdout counting) and `test_notebook_orchestration.py` (JSON/AST wiring, duplicate absence, outer orchestration, hyperparameter signature, historical notebook and `.gitignore` preservation). Added test-package markers under all test subdirectories because Python 3.12 `unittest discover -s tests` otherwise discovered zero tests; markers have no runtime side effects. Updated only notebook-boundary tests to reflect the two removed cells and removed unused direct import expectations.
- **Import graph and API audit:** Static analysis found no circular `src` imports and no source-level top-level calls. Dependency directions remain valid: trainers depend only on losses/transforms; inference depends on engine/models/utils; evaluation does not execute models. Notebook imports explicit submodule APIs; existing `__init__.py` exports are limited to engine/evaluation/trainers/inference contracts and have consumers. No speculative export needed removal.
- **Verification:** Focused integration/wiring discovery passed (6 tests). Authoritative `python -m unittest discover -s tests -p "test_*.py" -v` passed 79 tests in 1.718 seconds: datasets 8, engine 15, evaluation 6, inference 12, integration 6, losses 11, models 9, trainers 7, transforms 5. `compileall src tests`, notebook JSON parsing and per-cell structural compilation, source tracking, duplicate scan, historical notebook SHA-256 recording, `.gitignore` baseline comparison, whitespace, and nested-repository checks passed.
- **Optional dependencies:** torch 2.6.0+cu124, torchvision 0.21.0+cu124, numpy 2.5.1, scikit-learn 1.9.0, Pillow 12.3.0, and tqdm 4.70.0 are available. `datasets` is unavailable, so only `src.datasets.fruit_freshness` import and production dataset/notebook execution remain environment-blocked; no package was installed and no data was downloaded.
- **Implementation commit:** `0de234386246acde7b889b9b2ca83e974ff78de5` - `refactor: finalize notebook orchestration and integration`, pushed to and verified against GitHub `origin/refactor/phase-4.10-notebook-orchestration`.
- **Merge plan:** Retain the Phase branch for audit and fast-forward only into `main` after this documentation commit is pushed and local/GitHub `main` are again confirmed at `e23d601`.
- **Known limitation:** `git fetch origin` may fail solely due to the external Codex-managed `.git/refs/codex/turn-diffs/` bad-object reference. Do not repair, prune, pack, delete, or otherwise modify it; use GitHub API SHA fallback.
- **Next recommended phase:** Phase 5 may begin only with explicit authorization. The exact readiness blocker is reproducibility specification: no dependency declaration exists and `datasets` is not installed in the verified environment. Phase 5.1 should formalize dependencies/environment before configuration or entry-point work; do not begin it automatically.

## Phase 5.1 Dependency and Environment Specification (2026-08-01)

- **Scope:** Added a minimal, source-driven dependency specification and reproducibility environment guide only. No source module, test, notebook, `.gitignore`, dataset, weight, cache, or nested-repository content changed.
- **Created files:** `requirements.txt`, `requirements-dev.txt`, and `docs/environment.md`. The repository intentionally does not add Conda, `.python-version`, a package-manager migration, a lock file, or environment artifacts.
- **Runtime dependencies and policy:** Exact tested pins are `numpy==2.5.1`, `torch==2.6.0`, `torchvision==0.21.0`, `scikit-learn==1.9.0`, `matplotlib==3.11.1`, `Pillow==12.3.0`, and `tqdm==4.70.0`. Hugging Face `datasets>=3.0` is a direct requirement because the source uses `ClassLabel`, `DatasetDict`, and `load_dataset`; it has a minimum bound rather than an unverified exact pin.
- **Development dependencies:** `requirements-dev.txt` references runtime requirements and adds unpinned `jupyterlab` plus `ipykernel==7.2.0` for editing/executing the active `deep3.ipynb` notebook. The suite uses standard-library `unittest`; pytest, nbformat, linters, formatters, and other tooling were not added.
- **Verified environment:** Windows 11 64-bit, Python 3.12.10 64-bit, `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`, CUDA runtime 12.4, and NVIDIA GeForce RTX 3070 Ti with CUDA available. The guide records separate official CPU and CUDA 12.4 PyTorch wheel-index commands and explains that the wheel does not install the NVIDIA driver.
- **Compatibility status:** Current NumPy 2.5.1 and scikit-learn 1.9.0 are compatible in the verified environment; no active deprecated NumPy aliases were found. `datasets` and JupyterLab were absent, so no production dataset download, full notebook execution, or clean-environment installation was claimed. Importing `src.datasets.fruit_freshness` is blocked only by the documented missing `datasets` dependency.
- **Validation:** Requirement syntax/encoding/duplicate/path/URL/marker checks passed; security and portability scan found no secret, absolute user path, private index, or local wheel path; core and project import smoke tests passed; `pip check` reported no broken requirements; full `python -m unittest discover -s tests -p "test_*.py" -v` passed 79 tests with 0 failures and 0 skips. A pip dry-run was intentionally not run because it would contact package indexes; no package installation occurred.
- **Implementation commit:** `5469690d225545d9dba544342af603b186d6b740` — `chore: specify project dependencies and environment`. It was pushed successfully to `origin/chore/phase-5.1-environment` and verified against the GitHub phase-branch SHA.
- **Merge plan:** Commit this handoff entry separately as `docs: record phase 5.1 handoff`, push and verify the retained phase branch, then fast-forward only into `main` after confirming GitHub `main` is still at `4afed76`. No pull request, force push, or history rewrite is part of this Phase policy.
- **Known limitation:** `git fetch origin` can fail because of an external Codex-managed temporary ref under `.git/refs/codex/turn-diffs/`; do not repair or modify it. Use local/remote/GitHub SHA comparison as the safe fallback.
- **Next phase:** Phase 5.2 — Experiment Configuration Extraction may be considered only after this Phase has been fast-forwarded and synchronized. Remaining reproducibility blockers are the unverified Hugging Face dataset path and the untested clean-environment installation.

## Phase 5.2 Experiment Configuration Extraction (2026-08-01)

- **Scope:** Extracted active, notebook-level experiment inputs from `deep3.ipynb` into one committed TOML file while retaining the notebook as the orchestration entry point. No completed module API, model, dataset behavior, transform behavior, loss formula, optimizer/scheduler implementation, EMA behavior, checkpoint policy, training/validation/evaluation/inference behavior, dependency specification, `.gitignore`, historical notebook, or nested-repository content changed.
- **Selected format and files:** Added `configs/deep3.toml`, parsed with Python 3.12 standard-library `tomllib` through the new `src.utils.config.load_experiment_config(path: str | Path) -> dict`. Added `docs/configuration.md`, `tests/config/` (14 focused tests), and minimal notebook-boundary test updates. No dependency, framework, override system, environment variable, CLI, cache, global singleton, path discovery, or hidden default was added.
- **Configuration schema:** `runtime.cudnn_benchmark`; `loss.class_balanced_beta`, `use_ce_label_smoothing`, `label_smoothing`, and `focal_gamma`; `training.epochs` and `batch_size`; `fine_tuning.epochs`; `cross_validation.n_splits`, `shuffle`, and `random_state`; `mixup.alpha` and `probability`; `optimization.lr_cnn`, `lr_trans`, and `weight_decay`; `ema.decay`; `checkpoint.final_model_filename`; and `reporting.figure_size`.
- **Value and type parity:** TOML values exactly preserve the pre-wiring notebook baseline: beta `0.999`; epochs `120`/`20`; batch size `192`; folds `3`; CV shuffle `True` and seed `42`; Mixup `0.8`/`0.5`; learning rates `5e-5`/`1e-4`; weight decay `1e-4`; EMA `0.999`; CE toggle `True`; label smoothing `0.01`; Focal gamma `2.0`; cuDNN benchmark `True`; final filename `last_model_weights.pt`; and figure size `(10, 4)`. TOML loads figure size as `[10, 4]`, and `deep3.ipynb` explicitly restores `tuple(config["reporting"]["figure_size"])` at the Matplotlib call.
- **Values retained outside configuration:** Dataset identifiers/label cleanup/split logic and transform/model/checkpoint-template internals remain encapsulated in existing modules. `num_classes`, class counts, class-balanced alpha, fold indices, runtime devices, model/optimizer/scheduler/scaler/EMA objects, histories, metrics, checkpoint paths, and current fold/epoch remain derived or runtime state. The original notebook output-directory literal remains outside config because it is machine-specific; it was not moved or altered.
- **Notebook wiring:** Cell 0 adds explicit `Path` and loader imports plus repository-relative `CONFIG_PATH = Path("configs/deep3.toml")`. Cell 2 calls the loader exactly once and assigns the existing notebook names before their original consumers. It passes the same values to folds, cuDNN benchmark, loss construction, optimizer/scheduler, EMA, train/validation calls, final filename, and plotting. No outputs were cleared; only Cells 0 and 2 changed.
- **Validation:** All 14 focused config tests passed (TOML parse, required keys, exact value/type parity, no derived state, secret/path audit, loader string/Path inputs, no hidden defaults, invalid type/missing key/path failures, loader wiring, and historical notebook preservation). Full `python -m unittest discover -s tests -p "test_*.py" -v` passed 93 tests with 0 failures and 0 skips. `python -m compileall src tests`, TOML parse, isolated config-path wiring, notebook JSON/code-cell compilation, historical-notebook SHA checks, protected dependency-file and `.gitignore` checks, and nested-repository checks passed. Production dataset download, training, checkpoint generation, and full notebook execution were not run.
- **Implementation commit:** `27ccc2d6e23ceab0ea79713ace2e3c992ade3ca1` — `refactor: extract experiment configuration`. It was pushed successfully to `origin/refactor/phase-5.2-experiment-config` and verified against the GitHub phase-branch SHA.
- **Merge plan:** Commit this handoff entry separately as `docs: record phase 5.2 handoff`, push and verify the retained Phase branch, then fast-forward only into `main` after confirming local and GitHub `main` remain at `479c36b`. No pull request, force push, or history rewrite is part of this Phase policy.
- **Known limitation:** `git fetch origin` can fail because of an external Codex-managed temporary ref under `.git/refs/codex/turn-diffs/`; do not repair or modify it. Use local/remote/GitHub SHA comparison as the safe fallback.
- **Phase 5.3 readiness:** The configuration contract is ready for a future explicit training-entry-point phase. Remaining environment blockers are unchanged: `datasets` is not installed, production Hugging Face dataset execution is unverified, and a clean virtual-environment installation has not been run. Do not begin Phase 5.3 automatically.

## Phase 5.3 Training Script Entry Point (2026-08-01)

- **Scope:** Added a script-level orchestration entry point only. It reuses completed configuration, runtime, transform, dataset, model, loss, engine, trainer, evaluation, label, and path APIs without modifying their algorithmic behavior. `deep3.ipynb`, historical notebooks, config values, dependency files, `.gitignore`, existing `src/` modules, inference code, data, weights, checkpoints, the nested repository, and Codex-managed refs remain unchanged.
- **Created files:** `scripts/__init__.py`, `scripts/train.py`, `docs/training.md`, and `tests/scripts/` with CLI, import-safety, synthetic orchestration, checkpoint-policy, and output-directory failure coverage.
- **Public entry-point contract:** `scripts.train.build_parser()`, `run_training(args)`, and `main(argv=None) -> int`. Canonical invocation is `python -m scripts.train --config configs/deep3.toml --output-dir weights`. The CLI exposes only `--config` and `--output-dir`; no hyperparameter, seed, key-value, resume, dry-run, device, or data-cache override was introduced. Importing the module and running `--help` do not parse training arguments at import, load production data, initialize CUDA, create directories, print training output, or create checkpoints.
- **Portable output policy:** Relative paths are resolved from the repository root. The default output directory is repository-relative `weights/`; an absolute path is accepted only when explicitly selected by the caller. Directory creation occurs during execution after config validation and dataset preparation begin. The notebook's legacy machine-specific output literal was not copied. Existing artifact names and overwrite behavior remain: `label_names.json`, `best_model_fold{fold}.pt`, and `last_model_weights.pt`; the final raw-model state is now placed inside the selected output directory so the script never silently writes elsewhere.
- **Notebook-to-script parity:** `run_training()` has exactly one orchestration counterpart for device/config setup; transform construction; dataset loading, names, class counts, and label persistence; stratified folds and DataLoaders; per-fold model/EMA/criterion/optimizer/scheduler/scaler construction; class-balanced alpha; initial and fine-tuning epoch transitions; trainer/validator calls; metric and history collection; per-epoch scheduler stepping; best EMA fold checkpoint policy; and final raw-model save. `docs/training.md` contains the detailed responsibility/API table. The comparison remains structural: no explicit notebook seed exists and the script adds none, but exact production RNG parity has not been claimed without production execution.
- **Deferred to Phase 5.4:** Fold-checkpoint loading, holdout dataset/DataLoader construction, ensemble inference, horizontal-flip TTA, final holdout accuracy, and interactive Matplotlib plotting are intentionally absent from `scripts/train.py`. They remain notebook presentation or future evaluation/inference entry-point work.
- **Validation:** `python -m compileall src scripts tests`, `python -m scripts.train --help`, import safety from a temporary working directory without `datasets`, parser/default/custom/unknown/missing-config checks, synthetic fold/epoch/fine-tuning orchestration, exact checkpoint improvement/non-improvement tests, and output-directory failure propagation all passed. Full `python -m unittest discover -s tests -p "test_*.py" -v` passed **103 tests**, 0 failures, 0 skips. Security/portability and duplicate-implementation scans were clear; protected config/dependency/ignore/notebook baseline checks and nested-repository checks passed.
- **Production boundary:** No package was installed. The Hugging Face dataset download, production dataset processing, full CMT training, production checkpoint generation, post-training ensemble evaluation, and clean-environment installation remain unexecuted because `datasets` is unavailable in the validated environment.
- **Implementation commit:** `5d28353d272a37e7d2b61cce6fb96da835df2a1f` — `feat: add reproducible training entry point`. It was pushed successfully to `origin/feat/phase-5.3-training-entrypoint` and matched the GitHub Phase-branch SHA.
- **Merge plan:** Commit this handoff entry separately as `docs: record phase 5.3 handoff`, push and verify the retained Phase branch, then fast-forward only into `main` after confirming local, remote, and GitHub `main` remain at `570e91e`. Do not open a pull request for this Phase, force push, rebase, or rewrite history.
- **Known limitation:** `git fetch origin` may fail due to the external Codex-managed bad-object reference under `.git/refs/codex/turn-diffs/`. Do not repair, prune, pack, delete, or otherwise modify it; use local/remote/GitHub SHA comparison as the safe fallback.
- **Phase 5.4 readiness:** The repository is ready for explicit evaluation/inference-entry-point design, but end-to-end claims remain blocked by unavailable Hugging Face dataset execution, no production checkpoints, no real training run, and no clean-environment verification. Phase 5.4 must define an explicit checkpoint/input path rather than infer the notebook's legacy absolute directory, verify script/notebook checkpoint interoperability with real artifacts, and preserve the separation between training and inference. Do not begin Phase 5.4 automatically.

## Phase 5.4 Evaluation and Inference Entry Points (2026-08-01)

- **Scope:** Added the smallest script-level wrapper for the active labeled holdout evaluation path in `deep3.ipynb`. No existing source module, training script, notebook, configuration value, dependency file, `.gitignore` rule, dataset, checkpoint, output, cache, nested repository, or Codex-managed ref was changed.
- **Created implementation:** `scripts/evaluate.py` exposes `build_parser()`, `resolve_fold_checkpoint_paths()`, `run_evaluation(args)`, and `main(argv=None) -> int`. The canonical invocation is `python -m scripts.evaluate --config configs/deep3.toml --checkpoint-dir weights`. Relative config/checkpoint paths resolve from the repository root; absolute paths are accepted only when explicitly supplied.
- **CLI contract:** `--config` defaults to `configs/deep3.toml`; `--checkpoint-dir` is required. There are no device, seed, fold, batch-size, TTA, result-export, resume, or hyperparameter overrides. Importing the module and using `--help` do not import the Hugging Face dataset module, download data, initialize CUDA work, create directories, write artifacts, or start evaluation.
- **Behavioral parity:** The entry point uses `load_experiment_config`, `resolve_device`, the configured cuDNN benchmark value, `load_fruit_freshness_dataset`, `build_validation_transform`, `FruitHFDataset`, `build_holdout_dataloader`, `load_fold_models`, and `run_ensemble_holdout`. It derives class count from the processed training split, evaluates the existing `test` holdout split, preserves fold order, raw-logit ensemble behavior, horizontal-flip TTA, correct/total counting, and prints `Final Holdout Acc: <correct / total>`. It deliberately creates no plot, history presentation, result file, prediction file, cache, checkpoint, or output directory.
- **Checkpoint input policy:** The caller must explicitly supply a directory containing the existing path-helper names `best_model_fold1.pt` through `best_model_foldK.pt`, where `K` comes from `cross_validation.n_splits` (three in `configs/deep3.toml`). Missing directories, file paths, empty directories, and incomplete fold sets fail before dataset loading or model loading. There is no legacy absolute-path fallback, automatic `weights/` fallback, checkpoint discovery, or partial ensemble. The filename template is existing code rather than a user-configurable template, so no malformed-template input is exposed.
- **Inference decision:** The active notebook has no distinct unlabeled image input/output, submission, or prediction-artifact contract. `scripts/infer.py` was intentionally not created; `docs/evaluation.md` documents this deferment so a future generic prediction workflow is designed explicitly rather than inferred from the evaluation code.
- **Tests and documentation:** Added `docs/evaluation.md` plus four script test modules. Coverage includes parser/default/unknown-argument behavior, import and help safety without `datasets`, missing/file/empty/incomplete/complete checkpoint input behavior, synthetic holdout orchestration, and a temporary state-dict integration that loads deterministic fold checkpoints through the existing loader and verifies ascending model order and eval mode at the ensemble boundary.
- **Verification:** `python -m compileall src scripts tests`, `python -m scripts.evaluate --help`, `git diff --check`, and `python -m unittest discover -s tests -p "test_*.py" -v` passed. The full suite ran 116 tests with 0 failures and 0 skips. No production dataset download, CMT training, production checkpoint loading, or real holdout evaluation was run: `datasets` is unavailable and no production-compatible checkpoints exist in the verified environment.
- **Implementation commit:** `3ffdd47e48f93977bc0a56d6c34a31e4115f9795` - `feat: add holdout evaluation entry point`. It was pushed to `origin/feat/phase-5.4-evaluation-inference` and the GitHub Phase-branch SHA matched.
- **Synchronization plan:** Commit this handoff entry as `docs: record phase 5.4 handoff`, push and verify the retained Phase branch, then fast-forward it into `main` from the verified `c4e48a7` base. Do not open a pull request, force push, rebase, rewrite history, or modify the known external Codex temporary ref.
- **Phase 5.5 readiness:** The code structure is ready, but end-to-end reproducibility remains blocked by clean-environment installation, compatible `datasets` installation, Hugging Face access, production training feasibility, real fold checkpoints, script/notebook interoperability against real artifacts, storage/runtime confirmation, and the external Codex temporary-ref fetch limitation. Do not begin Phase 5.5 automatically.

## Phase 5.5A Hugging Face Dataset Loader Compatibility (2026-08-01)

- **Scope:** Diagnosed and restored only the Hugging Face dataset loader compatibility required to unblock Phase 5.5. The blocked `test/phase-5.5-reproducibility` branch remains unchanged at `5537a8d`; this work is isolated on `fix/phase-5.5-dataset-loader-compatibility` from the same main base.
- **Root cause:** At Hub revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`, the repository contains only `freshness_fruit.zip`. On `datasets==5.0.1`, automatic `load_dataset("Densu341/Fresh-rotten-fruit")` correctly infers 22 labels but emits the archive parent `Fresh-rotten-fruit@<revision>` as each example label, causing `DatasetGenerationError` with `Invalid string class label`. Explicit revision, direct archive, and archive-root ImageFolder routes reproduce the same failure; streaming exposes the same invalid label when iterated. The generated Parquet service is diagnostic-only and was not adopted because it is a derived `refs/convert/parquet` source rather than the pinned source revision.
- **Fix:** `src.datasets.fruit_freshness` now pins the Hub repository, revision, and archive filename; safely extracts the archive below `HF_DATASETS_CACHE`; and passes the extracted `dataset/` content root to `load_dataset("imagefolder", data_dir=...)`. It reuses a complete extraction cache without Hub access. Existing label removal, seed-42 80/20 split, remapping, RGB conversion, wrapper, and downstream APIs remain unchanged.
- **Source identity:** Archive SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`; 30,357 image members; exact image-to-class assignment comparison against the archive passed. The cleaned result is 26,858 examples: train 21,486 and test 5,372 with the existing 14-label order.
- **Dependency and tests:** Runtime pins are `datasets==5.0.1` and `huggingface-hub==1.26.0`. New non-network tests cover pinned download arguments, managed extraction, complete-cache reuse without Hub access, and archive path traversal rejection; the existing data cleanup/remapping/wrapper test now asserts the explicit ImageFolder boundary.
- **Clean-environment integration:** A newly created Python 3.12.10 virtual environment installed the project dependencies with `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `datasets==5.0.1`, and `huggingface-hub==1.26.0`; `pip check` passed. With a separate empty Hugging Face cache, the real loader completed and a validation-transform/DataLoader smoke batch yielded `(2, 3, 224, 224)` `torch.float32` images and `torch.int64` labels. No CMT construction, checkpointing, evaluation, training, or notebook execution was run.
- **Harness note:** A first inline Windows diagnostic was time-limited because `datasets.map(num_proc>1)` launched from `python -` is not a safe multiprocessing entry point. Its exact external child processes were stopped after identification; no repository file or dataset source was deleted. The final file-based `if __name__ == "__main__"` harness completed successfully in a distinct empty cache.
- **Documentation:** Added `docs/dataset.md` and updated environment/evaluation status to record the exact source lineage, cache behavior, clean-environment result, and remaining boundaries.
- **Next boundary:** Phase 5.5 remains incomplete. Only its dataset-loader blocker is resolved; do not begin CMT, checkpoint, evaluation, or bounded-training work without explicit Phase 5.5 authorization.

## Phase 5.5 Rerun - Clean Environment and Bounded End-to-End Verification (2026-08-01)

- **Prerequisite and branch safety:** Phase 5.5A loader compatibility fix `6e6a6198598625945f98cf2b642de02f46b610c5` was present on local `main`, `origin/main`, and GitHub `main` before this work. The original blocked branch `test/phase-5.5-reproducibility` remains untouched at `5537a8d45e17e8c727200ae33ac4b8f1188f5d58` with no commits beyond current main and no configured remote tracking branch. The rerun branch `test/phase-5.5-reproducibility-rerun` was created from `6e6a619` and is the only branch used for this phase.
- **Clean environments and installation:** Two new isolated external Python 3.12.10 64-bit venvs were created under unique temporary roots, without system site packages. The first exposed the unpinned direct JupyterLab dependency; its resolved `4.6.2` version was evidence for the minimal `requirements-dev.txt` pin. The second new venv installed the final specification from scratch with the official `torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` command followed by `requirements-dev.txt`. `pip check` passed. Direct resolved versions: torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, numpy `2.5.1`, datasets `5.0.1`, huggingface-hub `1.26.0`, scikit-learn `1.9.0`, Pillow `12.3.0`, tqdm `4.70.0`, matplotlib `3.11.1`, JupyterLab `4.6.2`, and ipykernel package `7.2.0`.
- **Clean and original regression:** The final clean venv and the original Python 3.12.10 environment each passed `python -m unittest discover -s tests -p "test_*.py"` with **125 tests**, 0 failures, and 0 skips. Both also passed `python -m compileall src scripts tests`, `python -m scripts.train --help`, and `python -m scripts.evaluate --help`. Runtime and project imports passed; invalid evaluation checkpoint input failed before dataset loading or model construction.
- **JupyterLab:** JupyterLab 4.6.2 and ipykernel package 7.2.0 were available. A unique temporary server started on `127.0.0.1` with browser launch disabled, and a temporary kernel specification pointed to the isolated venv Python. Only that temporary server process was stopped. No notebook was executed.
- **Dataset evidence:** Phase 5.5A remains the cold-cache proof. This rerun is explicitly a warm-cache integration run using only its external, revision-specific cache after verifying `freshness_fruit.zip` size `3,053,594,823` bytes and SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`. The committed loader reproduced repository `Densu341/Fresh-rotten-fruit`, revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`, 30,357 source images, 26,858 filtered images, 21,486 train rows, 5,372 test rows, 14 labels, seed 42, preserved label order, RGB conversion, safe extraction root, and second-load prepared-cache reuse.
- **Data pipeline:** Existing train, validation, and fine-tuning transforms returned `float32` `(3, 224, 224)` tensors with `int64` labels. Existing three stratified folds produced first-fold sizes 14,324/7,162. Existing loaders preserved batch size 192, train shuffle/validation and holdout ordering, zero workers, expected pin-memory settings, a normal `(192, 3, 224, 224)` batch, and a final incomplete batch without mutating the source split.
- **CUDA CMT smoke:** On the verified NVIDIA GeForce RTX 3070 Ti (8 GiB), CUDA 12.4, actual two-sample production-data CMT training/validation used the current factory, CE-with-label-smoothing selection, optimizer, scheduler, GradScaler, EMA, `train_one_epoch`, `validate_one_epoch`, and metric helper. The first automatic mixed-precision call correctly detected overflow and reduced GradScaler from 65,536 to 32,768 without an optimizer step; the second bounded call had finite gradients, created optimizer state, updated EMA, stepped the scheduler, and validated a `(2, 14)` logits contract. Peak allocated CUDA memory was 508,085,248 bytes. This is functional execution evidence, not an experiment-quality result.
- **Checkpoint and inference compatibility:** Actual CMT state dictionaries saved and loaded through engine helpers with strict map-location behavior; key order, every tensor, and fixed-input logits matched. Configured temporary `best_model_fold1.pt` through `best_model_fold3.pt` files then loaded through `load_fold_models` in ascending order, eval mode, and fixed-input logit parity. The temporary checkpoint directory was deleted after evidence capture and was never staged.
- **Real holdout evaluation:** A representative 192-example, three-model hflip-TTA batch took 0.7761 seconds and peaked at 1,943,769,600 allocated CUDA bytes. `python -m scripts.evaluate --config configs/deep3.toml --checkpoint-dir <temporary-checkpoint-directory>` completed all 5,372 holdout examples in 28 batches; it printed `Final Holdout Acc: 0.09735666418466121` (523/5,372), created no result file, and did not mutate the temporary checkpoint hashes. The three fold files were untrained compatibility fixtures; this accuracy is not model-quality, trained-checkpoint, or benchmark evidence.
- **Training boundary:** The optional bounded `scripts.train` CLI smoke was not run because a 192-sample backward pass was not conservatively safe on the 8 GiB device from the observed real two-sample memory behavior. Canonical three-fold 120-epoch training (including the final 20 fine-tuning epochs), trained-checkpoint evaluation, benchmark reproduction, full notebook execution, CPU model execution, and independent-clean-machine validation remain unverified.
- **Committed implementation:** `3f2dfda3ffd6650cd46e0f1feca476c8ffc4f299` - `test: verify and finalize reproducible execution`. It adds `docs/reproducibility.md`, offline reproducibility contract tests, the evidence-based `jupyterlab==4.6.2` direct pin, and only necessary environment/evaluation status documentation. The commit is pushed to GitHub on `origin/test/phase-5.5-reproducibility-rerun` and the remote SHA matches. No model, dataset, transform, loss, optimizer/scheduler, EMA, trainer, checkpoint format, TTA/ensemble, script behavior, config, historical notebook, `.gitignore`, nested repository, dataset, cache, weight, checkpoint, virtual environment, pip-freeze, or secret was committed.
- **Artifacts and Git limitation:** Rerun checkpoints were removed; no training output existed. The final isolated venv, its temporary diagnostics/Jupyter runtime, and its external pip audit are retained outside Git; the prior Phase 5.5A external dataset cache is shared evidence and was retained untouched. The protected untracked nested repository remains clean. `git fetch origin` may fail because of the known Codex-managed temporary ref; do not modify it. Use local `origin/*` refs plus `gh api` SHA comparison as the fallback.
- **Merge intent and next readiness:** After this handoff is separately committed and pushed, fast-forward only `test/phase-5.5-reproducibility-rerun` into `main` if GitHub main remains at `6e6a619`. Phase 5.5 is ready to close once that synchronization succeeds. Potential next work is Phase 6.1 CI/repository health, Phase 6.2 README/usage consolidation, Phase 6.3 release checklist, or separately authorized canonical training; do not start any automatically.
## Phase 6.1 - Repository CI and Health Checks (2026-08-01)

- **Scope:** Added repository-health CI only. No runtime dependency, configuration, dataset-loader, transform, model, loss, optimizer/scheduler, EMA, trainer, checkpoint, inference, script, notebook, `.gitignore`, dataset, cache, weight, or nested-repository content was changed.
- **Starting point and branch:** The dedicated `ci/phase-6.1-repository-health` branch was created from the verified Phase 5.5 `main` SHA `f2fd443186a1d1217ac278590a1b2857b4268e2c`. At this handoff point, local `main`, `origin/main`, and GitHub `main` still equal that SHA. The protected untracked nested `fruit-freshness-classification/` repository remains untouched and uncommitted.
- **Created implementation:** `.github/workflows/ci.yml`, `docs/ci.md`, `tests/repository/__init__.py`, and `tests/repository/test_ci_contract.py`. The static test guards the CI triggers, CPU-only Windows/Ubuntu Python 3.12 matrix, immutable official Action SHAs, read-only permissions, no secrets/artifact upload, offline validation environment, CLI-help-only behavior, test/compile/pip checks, and final worktree cleanliness.
- **Workflow policy:** `Repository CI` runs on pushes to `main` and all branch families, pull requests targeting `main`, and manual dispatch. It uses `contents: read`, per-ref cancellation concurrency, a 30-minute job timeout, CPU-only PyTorch installation from `https://download.pytorch.org/whl/cpu`, pip caching through `actions/setup-python`, `python -m pip check`, runtime/version/import/config/compile/test/CLI/cleanliness checks, and no production training or evaluation.
- **Action supply-chain policy:** `actions/checkout` is pinned to `3d3c42e5aac5ba805825da76410c181273ba90b1` (reviewed `v7.0.1`); `actions/setup-python` is pinned to `5fda3b95a4ea91299a34e894583c3862153e4b97` (reviewed `v7.0.0`). Checkout disables credential persistence and submodules. It uses depth 32, the smallest documented bounded history that covers the active historical-parity baselines and this handoff while avoiding a full-history fetch.
- **Offline and platform policy:** Validation steps set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `MPLBACKEND=Agg`; the unit-test step additionally sets `PYTHONIOENCODING=utf-8` for Windows console compatibility. The workflow imports project APIs without invoking the production dataset loader, does not execute notebooks, and creates no checkpoints or uploaded artifacts. CPU runners assert that CUDA is unavailable; the existing five CUDA-only parity tests skip by design.
- **Local verification:** YAML parsing, the five repository-CI contract tests, `python -m unittest discover -s tests -p "test_*.py" -v` (**130 passed, 0 failed, 0 skipped** on the local CUDA-capable environment), `python -m compileall src scripts tests`, `python -m scripts.train --help`, `python -m scripts.evaluate --help`, and `git diff --check` passed. No package was installed during this Phase.
- **Branch CI evidence:** Successful workflow run `30693987711` tested head `95b955c27da563f02a403a843ffc8b50b050ccc2`. `windows-latest / Python 3.12` passed in 3m27s with torch `2.6.0+cpu`, torchvision `0.21.0+cpu`, CUDA `False`, and **130 tests, 5 skips**. `ubuntu-latest / Python 3.12` passed in 1m46s with **130 tests, 5 skips**. Both runners passed dependency consistency, runtime/project imports, config smoke, compileall, CLI help, and repository cleanliness. Dataset access, production training/evaluation, notebook execution, checkpoint creation, and artifact upload did not occur.
- **Follow-up CI fixes:** `f29e969` moved the configuration smoke command into a YAML block scalar after the first workflow registration failure. `e3c9809` removed a locally detected malformed command prefix and simplified that smoke output. The next CI run then exposed pre-existing cross-platform test requirements: depth-1 checkout could not read historical notebook baselines, and Windows CP1252 could not print the existing `▶` symbol. `72dd487` introduced bounded history depth 30 plus test-step UTF-8 output; `95b955c` adjusted the bounded depth to 32 so the forthcoming documentation commit remains within the historical-test window. No test assertion was weakened and no project runtime code was changed.
- **GitHub health state:** This Phase did not change repository settings. Before and after implementation, no workflow existed previously; Dependabot and CODEOWNERS were absent; branch protection and required checks were absent; and rulesets were empty. Branch protection remains intentionally unconfigured pending a separate explicit decision.
- **Git limitation:** `git fetch origin` may still fail due to the externally managed Codex temporary ref under `.git/refs/codex/turn-diffs/`. Do not repair or modify that ref. Use local/remote/GitHub SHA comparison as the safe synchronization fallback.
- **Merge intent:** Commit this handoff entry separately as `docs: record phase 6.1 handoff`, push it on the retained CI branch, wait for its new CI run to pass on both runners, then fast-forward only into unchanged `main`. Do not force-push, rebase, squash, or delete the Phase branch.
- **Phase 6.2 readiness and remaining boundary:** After the final documentation-run and main-run CI verification, the repository is ready for a separately authorized README/portfolio documentation consolidation. CI is a repository-health gate only; canonical long-running training, trained-checkpoint evaluation, GPU CI, full notebook execution, and branch protection policy remain intentionally out of scope.

## Phase 6.2 - README and Usage Consolidation (2026-08-01)

- **Scope and branch:** Public-facing README and usage consolidation was implemented on `docs/phase-6.2-readme-usage` from the verified Phase 6.1 base `c54c0b7bec97d6be001215abf582f9a657bdae4a`. The prior README contained only the `data_project_kim` study placeholder; it is now the concise project entry point. The protected untracked nested `fruit-freshness-classification/` repository remains untouched.
- **README structure and badge:** The final README covers Overview, verified capabilities, dataset, Quick Start, configuration, training, evaluation, notebook usage, architecture, repository tree, testing, CI, reproducibility status, limitations, detailed documentation, and contribution boundary. A verified public `Repository CI` badge targets this repository's `ci.yml` workflow on `main`.
- **Usage and data facts:** It documents the shared `configs/deep3.toml` workflow, `python -m scripts.train`, and the required-checkpoint `python -m scripts.evaluate` command. It records `Densu341/Fresh-rotten-fruit`, its fixed revision/archive and safe ImageFolder cache route, 30,357 source images, 26,858 filtered images, 21,486/5,372 deterministic seed-42 train/holdout rows, and 14 final labels. Dataset contents and weights remain outside Git.
- **Truthful evidence and limits:** The README identifies verified clean-environment installation, production loader, CUDA CMT smoke, checkpoint interoperability, and evaluation CLI execution with untrained compatibility checkpoints. It explicitly does not present the temporary 523/5,372 result as model performance and preserves the boundaries: canonical training, trained-checkpoint evaluation, benchmark reproduction, full notebook execution, and independent-machine reproduction are not verified. CI remains offline, CPU-only, and does not load data, train, run CUDA, or perform real holdout evaluation.
- **Detailed-document consistency:** `docs/configuration.md` now correctly states that `deep3.ipynb`, `scripts.train`, and `scripts.evaluate` share the committed configuration. `docs/training.md` now links to the implemented evaluation entry point instead of referring to a deferred Phase 5.4 task. `tests/repository/test_readme_contract.py` adds offline checks for durable README claims, links, commands, badge accuracy, and limitations.
- **CI follow-up correction:** Initial documentation head `fd7cabfb835bb14d0d631877ea25adbdfd0cb03a` ran as GitHub Actions `30695742238` and failed only because the existing architecture-parity test reads fixed historical commit `7eb6e2a`, while the Phase 6.1 shallow checkout of 32 commits could no longer contain that baseline (32 commits lay between it and the new head). Follow-up commit `48cf569b4c8b132f1002e91b43d5dc5bf4f8bc87` changes only `ci.yml` checkout depth to `0` and updates the matching CI contract to require full history; it preserves read-only permissions, no persisted credentials, CPU/offline policy, action SHA pins, and all project behavior. GitHub Actions `30695968854` passed on Windows and Ubuntu (136 tests with 5 intentional CUDA-only skips on CPU CI).
- **Local verification:** The repository-contract suite passed 11 tests; the full suite passed 136 tests with 0 failures and 0 skips on the local CUDA-capable environment. `python -m compileall src scripts tests`, `python -m scripts.train --help`, `python -m scripts.evaluate --help`, UTF-8/Markdown/link/PowerShell checks, and `git diff --check` passed. No production dataset access, training, evaluation, generated output, or package installation occurred in this phase.
- **Files:** Created `tests/repository/test_readme_contract.py`; modified `README.md`, `docs/configuration.md`, `docs/training.md`, `.github/workflows/ci.yml`, and `tests/repository/test_ci_contract.py`. No `src/`, config, train/evaluate script, dependency, notebook, dataset-loader, model, loss, engine, trainer, inference, or `.gitignore` file changed. The CI checkout-depth exception is documented because it was necessary to retain the existing historical parity check.
- **Commits and push:** `fd7cabf` (`docs: consolidate project readme and usage`) and `48cf569` (`fix: correct repository CI workflow`) are pushed to `origin/docs/phase-6.2-readme-usage`. No history was rewritten, no repository setting was changed, and no PR was created. The known externally managed Codex temporary-ref issue may still make `git fetch origin` fail; do not alter it and use the local/remote/GitHub SHA fallback.
- **Next synchronization:** Commit this handoff as `docs: record phase 6.2 handoff`, push it, verify its new Windows/Ubuntu branch CI, then fast-forward only into unchanged `main` if local `main`, `origin/main`, and GitHub `main` still equal `c54c0b7`. Retain all branches; do not create a release tag or start Phase 6.3.
- **Phase 6.3 readiness and blockers:** Documentation and CI are ready after the required final branch/main synchronization. Governance remains unresolved because no `LICENSE` or `CITATION.cff` exists. Canonical training, trained checkpoints, benchmark results, and independent-machine reproduction also remain unverified and must not be represented as completed.

## Phase 6.3 - Release Audit and Governance Decision Package (2026-08-02)

- **Scope and base:** Audited release readiness and prepared governance decision material on `docs/phase-6.3-release-audit`, created from `main` at `be6e347328f80c423d2358c291257640a8147fd4`. This phase changes release documentation, a changelog, and repository contract coverage only; it does not make a software-license, citation, release, tag, metadata, or settings decision.
- **Initial release state:** The public GitHub repository had no tags, GitHub Releases, repository license metadata, `LICENSE`, `CITATION.cff`, branch protection, or rulesets. Its default branch was `main`, it was not archived, its description was the generic "For my data science studies", and homepage/topics were unset.
- **Candidate and version recommendation:** The candidate is an engineering and reproducibility milestone, not a trained-model or benchmark release. No canonical three-fold training, trained checkpoint performance, benchmark reproduction, full notebook execution, independent-machine reproduction, or generic unlabeled inference was claimed. Do not tag yet; after owner governance approval, consider an explicitly approved `v0.1.0` prerelease. No prior tag supports a different version lineage.
- **Governance audit:** No software license was invented or added. `docs/governance-decisions.md` records high-level MIT, Apache-2.0, and GPL-3.0 tradeoffs, without legal advice, plus the owner decisions still required. `CITATION.cff` remains pending owner-approved author identity, preferred citation form, and a compatible license decision; no paper, DOI, author metadata, institution, release date, or citation metadata was discovered.
- **Dataset governance:** The source dataset remains `Densu341/Fresh-rotten-fruit` on Hugging Face. Its publicly surfaced metadata identifies owner `Densu341` and license `openrail`; the surfaced dataset card README was empty, and attribution and redistribution conditions were not established. The repository contains no dataset copy. Any checkpoint distribution needs a separate dataset-license and artifact review.
- **Release package:** Created `docs/release-readiness.md`, `docs/release-checklist.md`, `docs/governance-decisions.md`, `CHANGELOG.md`, and `tests/repository/test_release_contract.py`. The readiness document includes the release-note draft and full evidence/blocker matrix. The checklist separates engineering-milestone requirements from model-performance-release requirements. The changelog is intentionally `Unreleased`, contains no date/version, and states the lack of trained weights and benchmark-quality metrics. `README.md` gains only index links to these documents.
- **Documentation consistency correction:** `docs/ci.md` now accurately describes the existing workflow's `fetch-depth: 0` historical-parity requirement. The CI workflow itself was not changed.
- **Protection and metadata recommendations:** Current protection/ruleset state is unprotected/none. The documents recommend PR review, the Repository CI workflow with both Windows and Ubuntu jobs, up-to-date branches, and prevention of force pushes/deletion; no protection or ruleset changed. They also recommend an AI-project-specific description, optional homepage, and topics such as `pytorch`, `computer-vision`, `image-classification`, `machine-learning`, `reproducibility`, `mlops`, and `huggingface-datasets`; no repository metadata changed.
- **Artifact and secret audit:** Normal reachable branch history and the current tree contain no committed datasets, downloaded archives, extracted images, virtual environments, checkpoints, weights, Hugging Face cache, generated outputs, large blobs over 5 MiB, or credential/token matches. Legacy notebooks and historical handoff material contain machine-specific paths, recorded as a portability risk; this phase did not alter them or rewrite history. The external Codex-managed temporary ref remains a known `git fetch origin` limitation and was not touched.
- **Validation:** Local release-contract discovery passed 18 tests; the full suite passed 143 tests with no failures (the CUDA-capable local run had no skips). `python -m compileall src scripts tests`, `python -m scripts.train --help`, and `python -m scripts.evaluate --help` passed. Markdown UTF-8, fences, links, headings, trailing whitespace, unsupported-release-claim, and release-document local-path checks passed. No package installation, production dataset access, training, evaluation, notebook execution, tag, GitHub Release, or settings update was performed.
- **Implementation commit and CI:** `9c43db11df110a94a2ff819e791a4e9755fe9a18` (`docs: prepare release audit and governance package`) was pushed to `origin/docs/phase-6.3-release-audit`. GitHub Actions run `30731595501` passed both Ubuntu and Windows jobs.
- **Files modified by implementation:** `README.md` and `docs/ci.md`; the five release-package files above were created. Source, configuration, dependencies, dataset loader, training/evaluation scripts, notebooks, CI behavior, `.gitignore`, nested repository, and Codex temporary refs remain unchanged.
- **Repository state and next actions:** No tag, GitHub Release, license, citation file, branch protection, ruleset, or metadata change was created. After this handoff commit and its branch CI pass, fast-forward merge only into unchanged `main` is permitted. Later work requires explicit owner approval: Phase 6.4 for license/citation metadata, Phase 6.5 for an approved tag/release, optional canonical training and checkpoint evaluation, and optional protection configuration. Remaining blockers are the license choice, citation identity, dataset attribution/redistribution clarity, version/release authorization, and model-evidence decisions.

## Phase 6.4 - Approved MIT License and Repository Citation Metadata (2026-08-02)

- **Scope and base:** Applied only the owner-approved MIT software license and repository-only citation metadata on `docs/phase-6.4-license-citation`, created from synchronized Phase 6.3 `main` SHA `243b7ea66d66a3cfd6621ee54ab21e05f9dd557b`. No model, dataset-loader, configuration, dependency, script, notebook, CI-workflow, repository-setting, or release operation changed.
- **Approved license decision:** MIT was explicitly selected in Phase 6.4; it is a new decision, not a claim that the repository was previously licensed. Added canonical UTF-8 `LICENSE` with unmodified MIT terms and the exact approved line `Copyright (c) 2025 김철희`. SPDX identifier is `MIT`.
- **Approved citation decision:** Added repository-only `CITATION.cff` for `Fruit Freshness Classification`, with one author: given names `Choelhui`, family names `Kim` (Choelhui Kim), canonical repository-code URL, and `license: MIT`. Author email is omitted by explicit owner request; affiliation and ORCID are omitted; additional authors are none. Citation version, release date, DOI, paper citation, identifiers, artifact URL, and preferred citation are intentionally omitted and remain unavailable or deferred to Phase 6.5.
- **Software, dataset, and weight boundary:** Repository software and project-authored documentation are MIT licensed. The external `Densu341/Fresh-rotten-fruit` Hugging Face dataset remains governed by its original-source terms and is not redistributed through this repository. The repository does not claim compatibility of the source metadata with MIT or permission to redistribute external images. Trained weights are not distributed and require a separate terms review before publication.
- **Documentation:** Updated README License/Citation sections and contribution boundary; updated governance decisions, release readiness, release checklist, and the Unreleased changelog. Governance is resolved only for repository license/citation identity; tag, GitHub Release, release date, DOI, dataset redistribution, trained-weight distribution, branch protection, and canonical training remain pending.
- **Contracts and validation:** Added `tests/repository/test_governance_metadata.py` and updated the prior Phase 6.3 release-contract expectation from pending to resolved-but-release-neutral governance. The new offline checks validate the MIT heading/copyright, CFF structure, exact author and repository URL, absence of unapproved identity/release fields, documentation boundaries, portability, and secret-like content. Repository contract discovery passed 23 tests; the complete suite passed 148 tests with no failures. `python -m compileall src scripts tests`, `python -m scripts.train --help`, and `python -m scripts.evaluate --help` passed. No external CFF validator was installed; structural tests and manual UTF-8/key/indentation validation are the evidence.
- **Implementation commit and CI:** `4216a376c0ebea121723487509450707e8c39c41` (`docs: add approved MIT license and citation metadata`) was pushed to `origin/docs/phase-6.4-license-citation`. GitHub Actions run `30733704946` passed both Ubuntu and Windows jobs. GitHub API verification confirmed that branch copies of `LICENSE` and `CITATION.cff` are visible, the citation title is correct, the author is Choelhui Kim, and no email/version/date/DOI field is present.
- **Files:** Created `LICENSE`, `CITATION.cff`, and `tests/repository/test_governance_metadata.py`; modified `README.md`, `CHANGELOG.md`, `docs/governance-decisions.md`, `docs/release-readiness.md`, `docs/release-checklist.md`, and `tests/repository/test_release_contract.py`. No source, config, requirements, workflow, notebook, or `.gitignore` file changed.
- **Safety and remaining work:** No tag, GitHub prerelease, GitHub Release, DOI, repository metadata, branch protection, ruleset, required check, dataset copy, trained weight, or repository setting was created. The protected untracked nested repository remains untouched. The external Codex-managed temporary ref remains a known `git fetch origin` limitation and was not modified. After this handoff commit and its branch CI pass, only a fast-forward merge into unchanged `main` is authorized. Phase 6.5 still requires explicit approval for `v0.1.0`, prerelease versus normal release, release date, final notes, confirmation that no dataset/weights will attach, and final default-branch metadata/CI verification.

## Phase 6.5 - v0.1.0 Engineering Milestone Release Preparation (2026-08-02)

- **Authorized scope:** Owner approved `v0.1.0` as a `PRERELEASE` engineering and reproducibility milestone, dated `2026-08-02`, with the title `Fruit Freshness Classification v0.1.0 — Engineering Milestone`. The approved release notes use the Phase 6.3 draft as their source and are finalized at [`docs/releases/v0.1.0.md`](docs/releases/v0.1.0.md).
- **Starting state and branch:** The Phase began from synchronized Phase 6.4 `main` SHA `ff0b3f7bfcde95ff22893b142062f9746abf319b`. The retained branch `release/phase-6.5-v0.1.0` was created at that SHA and is pushed to its matching origin branch. Local `main`, `origin/main`, and GitHub `main` still resolve to the same starting SHA at this pre-publication record.
- **Release identity:** The authorized tag is annotated `v0.1.0`, with annotation message `Fruit Freshness Classification v0.1.0 engineering milestone`. The intended GitHub object is a prerelease, not a trained-model or benchmark-performance release.
- **Preparation contents:** `CHANGELOG.md` now retains an empty `[Unreleased]` section and records `[0.1.0] - 2026-08-02`. Release readiness and checklist record the approved identity, source-only scope, no-attachment policy, and the external publication actions that remain incomplete. Offline contracts cover release-note presence, identity, portability, privacy, artifact exclusions, model-performance boundaries, links, and incomplete performance-release items.
- **Artifact policy:** Dataset attachment, trained-weight attachment, checkpoint attachment, and every other binary artifact attachment are explicitly `No`. Dataset redistribution and trained-weight distribution remain excluded. No cache, environment, log, or generated artifact is included.
- **Evidence and limitations:** Local validation passed 27 repository contract tests and 152 full tests with 0 failures and 0 skips on the local CUDA-capable environment, plus `compileall` and both CLI help paths. The Phase does not run a production dataset download, canonical training, real evaluation, or notebook execution. Canonical three-fold training, trained-checkpoint evaluation, benchmark reproduction, full notebook execution, independent-machine reproduction, and generic unlabeled inference remain incomplete or unavailable.
- **Preparation commit and CI:** Commit `0bd0bf3ccd909734954c11b66cbb11a04a0fdec3` (`docs: prepare v0.1.0 engineering milestone release`) is pushed. GitHub Actions run `30738387903` passed on this exact SHA: Ubuntu Python 3.12 CPU passed in 1m47s and Windows Python 3.12 CPU passed in 3m27s. No follow-up implementation commit was required.
- **Intended next release actions:** Commit this handoff separately, verify its branch CI, then fast-forward only into unchanged `main`. After a matching successful `main` CI at the merged preparation SHA, create only the approved annotated tag, push only that tag, verify its peeled target, and create the approved GitHub prerelease from `docs/releases/v0.1.0.md` with no uploaded assets.
- **Non-actions and safety:** No repository metadata, branch protection, ruleset, workflow, source, configuration, dependency, script, notebook, license, citation, dataset-loader, or nested repository change was made. The protected untracked nested `fruit-freshness-classification/` directory remains untouched. The external Codex temporary-ref issue may still make `git fetch origin` fail; do not modify it and use local/origin/GitHub SHA comparison as the fallback.
- **Publication state at this record:** The tag and GitHub Release have not yet been created.
## Phase 6.5 - v0.1.0 Engineering Milestone Publication Record (2026-08-02)

- **Published milestone:** The owner-approved `v0.1.0` `PRERELEASE`, titled `Fruit Freshness Classification v0.1.0 — Engineering Milestone`, was published from the CI-verified release commit `b38ebd36f4fa4f1fe012b957095db6dcbce20832`.
- **Annotated tag evidence:** Local and remote verification confirm annotated tag object `1044e6523a501fe82f5b59667c320ee2ec59eb89` named `v0.1.0`, with approved message `Fruit Freshness Classification v0.1.0 engineering milestone`, peeling to `b38ebd36f4fa4f1fe012b957095db6dcbce20832`. Only this tag was pushed; no force update was used.
- **Tagged main CI:** GitHub Actions run `30738724706` completed successfully on the tagged commit. Ubuntu Python 3.12 CPU passed in 1m52s and Windows Python 3.12 CPU passed in 3m22s. Release-publication, governance, readiness, README, CI, and repository-cleanliness contracts passed in the same workflow.
- **GitHub Release:** [https://github.com/kimcheolhui9846/fruit-freshness-classification/releases/tag/v0.1.0](https://github.com/kimcheolhui9846/fruit-freshness-classification/releases/tag/v0.1.0) was published at `2026-08-02T08:00:52Z`. It is not a draft, is a prerelease, uses the approved title and exact committed notes from `docs/releases/v0.1.0.md`, and has zero uploaded assets.
- **Artifact interpretation:** No dataset, trained weights, checkpoints, caches, environments, logs, or manually uploaded binaries are attached. GitHub-generated source-code archive links are platform defaults, not uploaded artifacts.
- **Remaining limitations:** This is a source-only engineering milestone, not a model-performance release. Canonical training, trained checkpoints, trained evaluation, benchmark reproduction, full notebook execution, independent-machine reproduction, generic inference, dataset/weight redistribution review, and branch protection remain incomplete or deferred.
- **Intentional history relationship:** The next commit records publication evidence only. After it is fast-forwarded, `main` will be one documentation-only commit ahead of `v0.1.0`; the immutable tag remains on `b38ebd36f4fa4f1fe012b957095db6dcbce20832`.
- **Safety:** No source, configuration, dependency, dataset-loader, script, notebook, workflow, license, citation, repository-setting, or nested-repository change was made during publication. The known externally managed Codex temporary ref remains untouched.

## Phase 7.1 - Main Branch Protection (2026-08-02)

- **Authorized scope:** Protect only the default branch using one active GitHub repository ruleset named `Protect main`; preserve the existing Phase-branch, exact-SHA CI, local `git merge --ff-only`, and normal `git push origin main` workflow. Pull requests, reviews, signed commits, merge queue, deployments, code scanning, push restrictions, tag protection, repository metadata changes, and classic branch protection are outside this authorization.
- **Implementation branch:** `chore/phase-7.1-main-protection`, created from synchronized `main` SHA `4116da7ac34b703f8412abde8b432c5820382794`. Commit `4e655a39c1f58f6c05c6551144009757b7b54a0f` (`docs: define protected main workflow`) adds only `docs/branch-protection.md` and `tests/repository/test_branch_protection_contract.py`.
- **Local validation:** Repository-contract discovery passed 31 tests and the full suite passed 156 tests. `compileall`, training/evaluation CLI help, source-diff, release-tag, and release-asset checks passed. No production dataset download, training, evaluation, notebook execution, package installation, source-code change, or artifact generation occurred.
- **Phase-branch CI:** GitHub Actions `Repository CI` run `30742002870` passed on that exact implementation SHA. Both `ubuntu-latest / Python 3.12` and `windows-latest / Python 3.12` succeeded.
- **Live ruleset:** Repository ruleset ID `20229405` is active for `~DEFAULT_BRANCH`, with no exclusions and no bypass actors. It contains exactly `deletion`, `non_fast_forward`, `required_linear_history`, and `required_status_checks`. The required contexts are `ubuntu-latest / Python 3.12` and `windows-latest / Python 3.12`; strict up-to-date policy and creation exemption are both false. GitHub effective-rule readback for `main` matched those four rules. No integration-ID constraint was configured.
- **Release boundary:** The annotated `v0.1.0` tag remains at `b38ebd36f4fa4f1fe012b957095db6dcbce20832`; the published GitHub prerelease remains source-only with zero uploaded assets. No tag, release, dataset, checkpoint, weight, cache, nested repository, or external Codex temporary reference was changed.
- **Next required steps:** Commit this evidence as `docs: record phase 7.1 protection handoff`, push and verify both required CI jobs on that exact handoff SHA, then perform only a normal fast-forward update of `main` and verify its CI/readback. Do not use force push, bypass actors, classic protection, deletion testing, or protection weakening. Retain the Phase branch after completion.

## Phase 7.2 - Published v0.1.0 Tag Protection (2026-08-02)

- **Authorized scope:** Protect only the already-published `v0.1.0` tag with one active exact tag ruleset. No broad future-tag pattern, tag creation restriction, tag update restriction, status check, signature rule, bypass actor, main-ruleset modification, classic protection, repository metadata change, tag movement, or GitHub Release mutation is authorized.
- **Starting state:** `main`, `origin/main`, and GitHub `main` all resolved to `5ba51ecc58176e8a97f0668ee0127823256563ad`. The annotated local, remote, and GitHub tag object remained `1044e6523a501fe82f5b59667c320ee2ec59eb89`, peeling to `b38ebd36f4fa4f1fe012b957095db6dcbce20832` with its approved annotation. The GitHub Release remained a published prerelease with zero uploaded assets. `Protect main` was active and no tag-target ruleset existed.
- **Implementation branch and validation:** `chore/phase-7.2-release-tag-governance` was created from the synchronized starting main SHA. Commit `9568b2b377de4946bfcf9e17201684da6fa5bf00` (`docs: define published tag governance`) adds only `docs/tag-governance.md` and `tests/repository/test_tag_governance_contract.py`. Local repository contracts passed 35 tests and the full suite passed 160 tests; compilation and both CLI help commands passed with offline dataset flags.
- **Implementation CI:** GitHub Actions `Repository CI` run `30745036512` passed on the exact implementation SHA. Both `ubuntu-latest / Python 3.12` and `windows-latest / Python 3.12` succeeded.
- **Live tag ruleset:** Repository ruleset ID `20232130`, node ID `RRS_lACqUmVwb3NpdG9yec48FxsWzgE0t8I`, named `Protect v0.1.0`, is active for target `tag`. It includes only `refs/tags/v0.1.0`, has no exclusions or bypass actors, and contains exactly `deletion` and `non_fast_forward`. Creation and update timestamps are `2026-08-02T20:11:35.508+09:00` and `2026-08-02T20:11:35.524+09:00`.
- **Readback and integrity:** The repository now has exactly `Protect main` and `Protect v0.1.0`. Exact tag-target and detail readback match approval. The local/remote/GitHub tag object, peeled target, annotation, and prerelease state remain unchanged; no destructive tag test was performed. `Protect main` remains active with its four Phase 7.1 rules and two required check contexts unchanged.
- **Payload and safety:** The external temporary payload contained only the approved tag ruleset JSON and no credentials; it was removed after creation. A PowerShell JSON-array interpretation check stopped one post-create list assertion, but direct raw-list and detail readback confirmed the approved configuration, so no live ruleset correction was needed. No dataset, weight, checkpoint, release asset, source, configuration, dependency, script, notebook, workflow, license, citation, changelog, release note, nested repository, or external Codex reference was changed.
- **Next required steps:** Commit this evidence as `docs: record phase 7.2 tag protection handoff`, push and verify both required CI jobs on that exact handoff SHA, then use only a normal fast-forward update of `main` and verify its CI. Do not force push, use bypass actors, disable or delete either ruleset, push tags, move the tag, or alter the GitHub Release. Retain the Phase branch after completion.
## Phase 7.3 - Repository Metadata and Portfolio Discoverability (2026-08-02)

- **Authorized scope:** Updated only the approved public repository description and the exact approved ten-topic set. No homepage, social-preview image, profile setting, visibility, default branch, merge setting, repository feature, ruleset, tag, GitHub Release, source, configuration, dependency, script, notebook, workflow, README, license, citation, changelog, or release-note change is authorized or made.
- **Starting state:** `main`, `origin/main`, and GitHub `main` resolved to `ec5b3d8af8d1ab2e72c64f23d0dc8d6144344412`. The working tree had no tracked change; only the protected nested `fruit-freshness-classification/` repository remained untracked and untouched. Original live metadata was description `For my data science studies`, empty homepage, and no topics. The repository was public, not archived, not a fork, defaulted to `main`, and recognized the MIT License. The custom social-preview state was not enabled.
- **Governance and release preconditions:** `Protect main` (ID `20229405`) remained active for the default branch with deletion, non-fast-forward, linear-history, and the two required Windows/Ubuntu CI contexts; it has no bypass actors. `Protect v0.1.0` (ID `20232130`) remained active for only the published tag with deletion and non-fast-forward rules; it has no bypass actors. No classic branch protection exists. The annotated `v0.1.0` tag still peels to `b38ebd36f4fa4f1fe012b957095db6dcbce20832`; the GitHub Release remains a published prerelease, non-draft, with zero assets.
- **Implementation:** Created `docs/repository-metadata.md` and `tests/repository/test_repository_metadata_contract.py`. The offline contract checks exact approved description/topic documentation, lowercase unique topic syntax, homepage/social-preview/profile boundaries, release-neutral claims, public `main`, unchanged rulesets/tag, and privacy/portability boundaries. Commit `2f75ea400dc85d5eda563969d5b9318184cc2c64` (`docs: define repository metadata and discoverability`) contains only those two files.
- **Validation and implementation CI:** Local repository-contract discovery passed 39 tests; the full suite passed 164 tests with no failures. Compilation and both CLI help commands passed without production dataset access, training, evaluation, or output artifacts. GitHub Actions run `30746176046` passed on the exact implementation SHA: Ubuntu completed in 1m56s and Windows in 4m00s, including repository-contract, full-test, CLI, and cleanliness steps.
- **Live metadata mutation:** After exact-SHA CI passed, one description-only `PATCH /repos/{owner}/{repo}` changed the approved description and one topics-only `PUT /repos/{owner}/{repo}/topics` replaced the empty list with exactly `pytorch`, `computer-vision`, `image-classification`, `deep-learning`, `machine-learning`, `reproducibility`, `mlops`, `huggingface-datasets`, `research-software`, and `fruit-freshness`. Both payloads were valid UTF-8, contained no credentials, and were removed after readback.
- **Live readback:** The description exactly matches approval; the homepage remains empty; the exact ten-topic set is present, lowercase, and unique; visibility remains public; default branch remains `main`; archive/fork/MIT state remains unchanged; and custom social preview remains unchanged. Feature and merge settings also remained unchanged. No rollback was required.
- **Integrity after mutation:** Both rulesets, the published tag target, prerelease/draft state, release title, published time, release notes, and zero-asset state remained unchanged. No destructive tag test occurred. No profile pinning or other profile setting was performed.
- **Next required steps:** Commit this evidence separately as `docs: record phase 7.3 metadata handoff`, push and verify both required CI jobs on that exact handoff SHA, then fast-forward only into unchanged protected `main` and verify its CI and final metadata readback. Retain the Phase branch. Do not force push, bypass protection, alter rulesets/tags/releases, add a homepage, create a social-preview image, or start a new Phase automatically.
- **Known limitation:** `git fetch origin` may fail only because of the externally managed Codex temporary ref under `.git/refs/codex/turn-diffs/`. Do not repair, prune, pack, remove, rename, or otherwise modify that ref; compare local, origin-tracking, and GitHub API SHAs as the safe synchronization fallback.
## Phase 7.4 - Branch Retention and Cleanup Decision Package (2026-08-03)

- **Scope and starting state:** This audit and policy Phase began from synchronized `main`, `origin/main`, and GitHub `main` SHA `d205f2995b6b44345fba1efffa2bb9a6db44fb10`. It created only the retained Phase branch `docs/phase-7.4-branch-retention-policy`; no existing branch was deleted, renamed, moved, reset, recreated, or force-updated. The protected nested repository remained untracked and untouched.
- **Inventory sources:** Local refs, origin-tracking refs excluding `origin/HEAD`, GitHub branch API pagination, `git ls-remote --heads origin`, pull-request history, merge-base reachability, unique-commit counts, release ancestry, tracked-document references, workflow definitions, and both active rulesets were audited. The initial audit had 24 local branches, 21 origin-tracking branches, and 21 live GitHub branches. No pull request, open or historical, was returned.
- **Remote consistency and later observation:** GitHub API and `git ls-remote` agreed on every remote branch and SHA. After the implementation branch was pushed, the required recheck observed 24 local branches, 22 origin-tracking branches, and 22 GitHub branches. The only added remote branch was the current Phase branch at `4aff4a7dea4a5d5f6c0b6349a5f02ac6918fb87b`; every pre-existing remote branch and `main` remained unchanged.
- **Unique-commit findings:** `backup/before-fruit-freshness-switch-20260729` is local-only, diverges from `main`, and has 15 unique historical commits reachable from no other branch, tag, or ref; it is classified `RETAIN_UNIQUE_COMMITS`. The current Phase branch has its expected one unmerged policy commit and remains `RETAIN_ACTIVE_OR_RECENT`. No other branch has unique commits.
- **Classifications:** `main` is the sole `MANDATORY_RETAIN` branch. `release/phase-6.5-v0.1.0` is `RETAIN_RELEASE_AUDIT`. Twenty fully merged modernization or governance branches are `TEMPORARY_RETAIN` under the conservative 90-day-or-next-milestone policy. There are no `SAFE_DELETE_CANDIDATE`, `REVIEW_REQUIRED`, or `BLOCKED_UNVERIFIED` branches.
- **Decision package:** Both local and remote deletion candidate lists are `NONE`. No deletion is authorized in Phase 7.4. Phase 7.5 requires exact, separately approved local and remote branch lists plus any retention exceptions; silence is not approval.
- **Dependencies and release evidence:** The CI workflow names `main` and wildcard pushes but no required Phase or release branch. `Protect main` remains active only for the default branch; `Protect v0.1.0` remains active only for the exact tag. The protected annotated tag remains at its original object and peeled release commit. The GitHub prerelease remains non-draft with zero assets. Repository metadata and settings were not changed.
- **Implementation and validation:** Commit `4aff4a7dea4a5d5f6c0b6349a5f02ac6918fb87b` (`docs: define branch retention and cleanup policy`) adds only the policy, inventory, and offline contract test. Local repository-contract discovery passed 44 tests and the full suite passed 169 tests. Compilation and both CLI help commands passed without production dataset access, training, or evaluation. GitHub Actions run `30757113760` passed on the exact implementation SHA: Ubuntu in 1m53s and Windows in 3m26s.
- **Next required step:** Commit this audit evidence, including the mandated post-CI inventory observation, then verify its exact branch CI before a protected fast-forward-only update of unchanged `main`. Do not begin Phase 7.5 or apply any candidate list automatically.
- **Known limitation:** `git fetch origin` can fail because of the externally managed Codex temporary ref. It was not modified; GitHub API and `git ls-remote` are the remote branch sources of truth.

## Phase 7.5 - Backup Branch Audit and Preservation Decision (2026-08-03)

- **Scope and starting state:** Audited only the local-only `backup/before-fruit-freshness-switch-20260729` history from synchronized Phase 7.4 `main` SHA `0951f4ab3536b492fe116314fc8ca72fc933ebf8`. The audit branch is `docs/phase-7.5-backup-branch-audit`; the protected nested repository remained untracked and untouched.
- **Backup identity:** The backup remains at `a9a6d1d28e35a4cc587860ae09534f5c827e43da`, has no common ancestor with current `main`, and has 15 unique commits. Current `main` has 63 commits not in the backup. The backup remains local-only: no origin or GitHub ref, tag, or other branch contains its unique commits.
- **Methodology and relevance:** Used read-only Git-object, tree, metadata, pattern, blob-size, and notebook-JSON structural inspection. The backup was not checked out, executed, merged, cherry-picked, tagged, pushed, bundled, archived, renamed, reset, deleted, or rewritten. It is classified `RELATED_RESEARCH_HISTORY`: related ML and fruit or CMT experimentation, but not a proven earlier version of the current modular project.
- **Security and privacy:** No recognized credential-format pattern, private-key marker, credential-like tracked file, content email, or telephone pattern was found. Personal email is present in commit metadata and is redacted from audit documents. Local-path signals occur in every tip notebook; 38 historical keyword-only service or session matches require manual review. No credential was tested or exposed.
- **Artifacts and license:** The tip has 10 paths, including 8 notebooks, no tracked raw dataset, weights, checkpoints, archives, executable files, or binary artifacts. Two notebook blobs exceed 1 MiB; none exceeds 10 MiB. Dataset provenance, crawler-related redistribution terms, ownership, third-party attribution, and license compatibility are unresolved. One extensionless artifact and one non-executed filesystem-removal pattern require manual review.
- **Publication and preservation decision:** The public-push gate fails or remains unknown because of personal metadata, local paths, disconnected-history relevance, ownership, license, and provenance concerns. Primary recommendation: `REVIEW_REQUIRED`. Secondary recommendation: `KEEP_LOCAL_ONLY`. Owner approvals for disclosure, archive location, encryption, retention, and any future deletion remain unresolved.
- **Implementation and validation:** Commit `ece0dec83e40ef389e2082f3f7c2dba58c30d8bf` (`docs: audit unique backup branch history`) adds only `docs/backup-branch-audit.md`, `docs/backup-branch-preservation.md`, and `tests/repository/test_backup_branch_audit_contract.py`. Local repository contracts passed 51 tests and the full suite passed 176 tests; compilation and both CLI help commands passed without dataset access, training, evaluation, or backup-history execution. GitHub Actions run `30783345098` passed on the exact commit: Ubuntu in 1m27s and Windows in 3m29s.
- **Post-CI recheck and governance:** Backup SHA, unique count, remote absence, tag absence, and other-branch absence were rechecked after CI. No bundle was created; no branch was deleted or moved; no tag was created; no GitHub setting changed. `Protect main`, `Protect v0.1.0`, the annotated release tag, and the zero-asset prerelease remain unchanged.
- **Next required action:** Commit this handoff, verify exact-SHA Phase-branch CI, then use only a protected fast-forward update of unchanged `main`. Phase 7.6 requires explicit owner approval using the unresolved fields in `docs/backup-branch-preservation.md`; do not infer or begin an action.
- **Known limitation:** `git fetch origin` may fail only because of the externally managed Codex temporary reference. Do not modify that reference; use local, origin-tracking, GitHub API, and `git ls-remote` readback as synchronization evidence.

## Phase 7.6 - Local-Only Backup Preservation (2026-08-03)

- **Authorized scope:** Applied the owner-approved `KEEP_LOCAL_ONLY` decision as a documentation-only preservation record. No Git bundle, encrypted archive, plaintext archive, destination directory, selected-content extraction, remote backup publication, private repository, tag, merge, cherry-pick, rewrite, branch rename, reset, or deletion was authorized or performed.
- **Starting state:** The Phase branch `docs/phase-7.6-local-only-preservation` was created only from synchronized `main` SHA `e10ac49439502d1f34ee41a9c7dc67d429a60683`. The tracked working tree was clean; the protected nested `fruit-freshness-classification/` repository remained the sole untracked path and was untouched.
- **Approved decision:** `APPROVED_BACKUP_ACTION=KEEP_LOCAL_ONLY`; public disclosure, remote publication, private-repository creation, Git bundle creation, selected-content extraction, and history rewrite are `NO`; archive location and encryption are `NOT_APPLICABLE`; retention is `PERMANENT`; and backup-branch deletion is `NO`. The owner accepts the remaining single-machine loss risk. Reconsideration requires a new explicit owner-approved preservation Phase.
- **Backup integrity:** `backup/before-fruit-freshness-switch-20260729` remains locally available at `a9a6d1d28e35a4cc587860ae09534f5c827e43da`, with 15 total and 15 `main`-unique commits and no common ancestor with `main`. It remains absent from origin and GitHub, and no tag points to its history. No archive or bundle exists because creation was not approved.
- **Implementation:** Commit `2e9d273cbebdce25d95e012b1444d52c6012bf78` (`docs: record local-only backup preservation decision`) changes only `docs/backup-branch-preservation.md` and `tests/repository/test_backup_branch_audit_contract.py`. The document replaces the Phase 7.5 unresolved approval block with the exact approved values and prohibits unapproved preservation actions; its offline contract test checks those values, privacy, risk acceptance, and the future-approval boundary.
- **Local validation:** Repository contracts passed 52 tests and the full suite passed 177 tests. `python -m compileall src scripts tests`, `python -m scripts.train --help`, and `python -m scripts.evaluate --help` passed. No dataset access, training, evaluation, notebook execution, backup-history execution, package installation, source change, or artifact generation occurred.
- **Implementation CI:** GitHub Actions `Repository CI` run `30786674546` passed on the exact implementation SHA. `ubuntu-latest / Python 3.12` completed in 1m40s and `windows-latest / Python 3.12` completed in 4m05s; contract, full-test, CLI, and repository-cleanliness steps all passed.
- **Governance and release integrity:** `Protect main` (ID `20229405`, 4 rules) and `Protect v0.1.0` (ID `20232130`, 2 rules) remained active and unchanged. The annotated `v0.1.0` tag remains unchanged, and its GitHub Release remains a published non-draft prerelease with zero assets. No GitHub setting or repository metadata changed.
- **Next required steps:** Commit this handoff separately as `docs: record phase 7.6 preservation handoff`, push and verify CI on that exact handoff SHA, then fast-forward only the unchanged protected `main` and verify final-main CI and remote readback. Do not begin another Phase automatically.
- **Known limitation:** `git fetch origin` can fail because of an externally managed Codex temporary ref. Do not repair, prune, pack, remove, rename, or otherwise modify it; use local, origin-tracking, GitHub API, and `git ls-remote` SHA checks as the safe synchronization fallback.


## Phase 8.1 - Canonical Training Readiness and Experiment Freeze (2026-08-03)

- **Authorized scope:** Audited and froze canonical-training readiness only. No source, model, notebook, configuration, dependency, script, CI, README, license, release, tag, ruleset, dataset policy, output policy implementation, or nested-repository content changed. No canonical training, checkpoint, weight, result, benchmark, evaluation, or publication artifact was created.
- **Starting point and implementation:** The Phase branch `docs/phase-8.1-canonical-training-readiness` started from synchronized `main` commit `046760e19e77c7aa0c6cbc065358acfd46aac346`. Commit `03a5a48978304f7237b67e7ff0aad75e4df7e6b3` (`docs: assess canonical training readiness`) adds only `docs/canonical-training-readiness.md`, `docs/canonical-training-runbook.md`, and `tests/repository/test_canonical_training_readiness_contract.py`.
- **Frozen identity:** The readiness record pins the Git blobs and SHA-256 values for `configs/deep3.toml`, `requirements.txt`, `requirements-dev.txt`, `scripts/train.py`, and `scripts/evaluate.py`. The unchanged config is 120 epochs, final 20 fine-tuning epochs, 3 shuffled stratified folds with random state 42, training batch 192, Mixup 0.8/0.5, existing optimizer rates/weight decay, EMA 0.999, and the existing label-smoothed cross-entropy selection.
- **Environment and dataset evidence:** A new isolated environment with the committed exact pins passed `pip check`: Python 3.12.10, torch 2.6.0+cu124, torchvision 0.21.0+cu124, datasets 5.0.1, and huggingface-hub 1.26.0. The committed production loader verified `Densu341/Fresh-rotten-fruit` revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c` and archive SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`; counts are 30,357 source, 26,858 filtered, 21,486 training, 5,372 holdout, and 14 classes. All three folds have 14,324/7,162 train/validation rows with no index overlap.
- **Bounded GPU evidence:** On the audited NVIDIA GeForce RTX 3070 Ti (8 GiB), actual production-path train probes completed at batches 2, 4, 8, and 16 using transforms, CMT factory, selected loss, optimizer, scheduler, AMP scaler, EMA, and `train_one_epoch`. Batch 16 reserved 732 MiB; the conservative projection for training batch 192 is 9,210,691,584 bytes (107.2% of VRAM), classified `LIKELY_UNSAFE`. Training batch 192 was not attempted. Validation horizontal-flip TTA completed at batches 16 and 192, and three-model ensemble horizontal-flip TTA completed at batch 16; this evaluation evidence does not make training batch 192 safe.
- **Readiness result and operator controls:** Final classification is `BLOCKED`. Output directories were empty before/after the audit; the runbook documents fixed checkpoint filenames and requires a new owner-approved empty untracked directory for any future run. Reproducibility is `REFERENCE_RUN_WITH_RECORDED_ENVIRONMENT`, not a bit-for-bit claim. Interruptions require restart from the beginning because optimizer, scheduler, AMP scaler, EMA, and RNG states are not checkpointed; this is `ACCEPTABLE_WITH_OWNER_RISK` only after explicit owner acceptance. Batch/hardware, output-directory, and interruption-risk decisions remain pending. Dataset, weights, checkpoints, and other binary publication remain `NO`.
- **Validation and implementation CI:** The new contract followed a red-to-green sequence (missing documents failed; completed documents passed). Fresh local CI-equivalent verification passed compilation, 184 unit/contract tests, both CLI help commands, and whitespace checks. GitHub Actions `Repository CI` run `30790370134` passed on the exact implementation SHA: Ubuntu in 1m46s and Windows in 3m01s, including dependency consistency, imports, config, compilation, full tests, CLI, and repository cleanliness.
- **Temporary evidence handling:** The isolated external virtual environment, external probe script, and external JSON evidence were removed after their verified values were recorded. No repository artifact was removed. The Hugging Face dataset cache remains outside the repository and uncommitted.
- **Governance and safety:** `Protect main`, `Protect v0.1.0`, the annotated `v0.1.0` tag, its zero-asset prerelease, and the local-only backup branch remain unchanged. The protected nested `fruit-freshness-classification/` repository remains the sole untracked path and was not read, staged, changed, or removed.
- **Next required steps:** Commit this handoff as `docs: record phase 8.1 training readiness handoff`, push and verify both required CI jobs on that exact SHA, then fast-forward only into unchanged protected `main` and verify final-main CI and remote readback. Do not begin Phase 8.2. No canonical training can begin without a new explicit owner-approved Phase resolving the `BLOCKED` gate.
- **Known limitation:** `git fetch origin` may fail because of an external Codex-managed temporary reference under `.git/refs/codex/turn-diffs/`. Do not repair, prune, pack, remove, rename, or otherwise modify it; use local, origin-tracking, GitHub API, and `git ls-remote` SHA readback as the safe synchronization fallback.

## Phase 8.2 - Resource-Safe Canonical Configuration and Epoch Resume (2026-08-03)

- **Authorized scope and branch:** The owner approved `CREATE_DERIVED_CANONICAL_CONFIG`, batch 64 on NVIDIA GeForce RTX 3070 Ti (8 GiB), `KEEP_EXISTING_UNSCALED`, optional `IMPLEMENT_EPOCH_BOUNDARY_RESUME`, output `weights/deep3-canonical-reference-01`, external log `results/deep3-canonical-reference-01.log`, `FAIL_IF_NOT_EMPTY`, and no seed or cuDNN policy change. All dataset, weight, checkpoint, other-binary, and release publication values remain `NO`. Work is on `feat/phase-8.2-safe-resumable-canonical-training`, created from synchronized `main` SHA `14acf24f71fa1c42615bad78d61c08e86d95ab2a`.
- **Derived config:** `configs/deep3_canonical.toml` is a complete copy of `configs/deep3.toml` with exactly one parsed difference: `training.batch_size` changes from 192 to 64. The original config remains SHA-256 `62c7ae4ee5c33974fa48342b6af1b7b54c2e4938159429cbd1a86524fc7c13f1`; epochs, folds, fine-tuning boundary, random state, Mixup, learning rates, weight decay, EMA, checkpoint naming, model, transforms, losses, and evaluation are unchanged. Batch 64 is documented as a separate optimization trajectory, not a performance-equivalent batch-192 result.
- **Resume implementation:** `scripts.train` preserves legacy behavior when no new flags are supplied and adds opt-in `--resume-state`, `--save-training-state`, `--require-empty-output-dir`, and `--run-id`. Fresh stateful runs reject every non-empty output entry before dataset/model construction. Resume accepts only the approved local `training_state.pt` in the selected output directory with `run_manifest.json`, validates immutable identity and fold-index hashes, rejects unexpected files and `COMPLETED` state, and does not accept remote/untrusted state.
- **State schema and cadence:** `src/engine/training_state.py` uses a versioned schema with model, EMA, optimizer, scheduler, GradScaler, Python/NumPy/Torch CPU/Torch CUDA RNG states, histories, best accuracy, fold indices, config/run/dataset/label identity, and `RUNNING`, `FOLD_COMPLETE`, and `COMPLETED` statuses. State is atomically replaced from a unique same-directory temporary file only after training, validation, metrics, checkpoint decision, history update, and scheduler step. CPU and CUDA RNG byte tensors are moved back to CPU before restoration after CUDA map-location loading; this production-path defect was discovered by the bounded CUDA probe and fixed with CPU-only regression coverage.
- **Tests:** New engine tests cover schema, atomic replacement, simulated failure preservation, trusted-local loading, mismatches, terminal-state rejection, and CPU/CUDA RNG map-location restore. Synthetic integration tests cover interrupted versus uninterrupted runs, EMA/optimizer/scheduler parity, fold boundary, fine-tuning epochs 100/101/119, collision protection, and legacy stateless behavior. Repository contracts protect the exact derived-config diff, privacy, publication boundary, documentation, and Phase 8.3 gate. Final local verification passed 205 tests, compilation, both CLI help commands, config parity, and staged whitespace checks.
- **Bounded production CUDA evidence:** An external exact-pinned environment used the actual pinned dataset loader, training transform, CMT classifier, selected criterion, optimizer, scheduler, AMP GradScaler, EMA, and `train_one_epoch`. Two limited batch-64 optimizer steps completed with finite gradients and EMA updates. Peak reserved memory was 2,716 MiB then 2,720 MiB (33.2% of 8 GiB), below the 70% guard; temperature was 47 C then 49 C; no CUDA OOM or repeated scaler overflow occurred. These loss/accuracy observations are diagnostic only, not benchmark or model-quality evidence.
- **Bounded CUDA resume evidence:** A trusted local state was saved after a controlled limited sequence, objects were destroyed and rebuilt, state/metadata/EMA/optimizer/scheduler/scaler/RNG were restored, and another batch-64 optimizer step completed. Optimizer tensors were CUDA-resident; scheduler advanced from 1 to 2; resumed peak reserved memory was 2,924 MiB (35.7%); no controlled fold or epoch repeated. This is same-run continuation interoperability evidence, not bit-for-bit from-scratch reproducibility.
- **Artifacts and publication:** No full canonical three-fold training, canonical weight, checkpoint, result, benchmark, release, or publication artifact was created. The approved repository output directory and log file remain absent. The external virtual environment, scripts, states, and JSON evidence were removed after their non-sensitive values were documented. The dataset cache remains outside Git.
- **Implementation commit and CI:** `4fa8485ea595fec3eb8e4f0e206c2b9b0008a9bb` (`feat: add safe canonical config and epoch resume`) contains the approved implementation, documentation, and tests. GitHub Actions run `30796466284` passed on that exact SHA: both Ubuntu and Windows completed dependency, import, config, compile, unit/contract, CLI, and repository-cleanliness checks.
- **Post-CI governance recheck:** Local `main`, `origin/main`, and GitHub `main` remain at `14acf24f71fa1c42615bad78d61c08e86d95ab2a`. `Protect main` and `Protect v0.1.0` remain active. The annotated `v0.1.0` tag still peels to `b38ebd36f4fa4f1fe012b957095db6dcbce20832`; its GitHub Release remains published prerelease, non-draft, with zero assets. Local-only `backup/before-fruit-freshness-switch-20260729` remains at `a9a6d1d28e35a4cc587860ae09534f5c827e43da` with 15 commits, no remote branch, and no containing tag. The protected nested repository remains the sole untracked path and was untouched.
- **Next required sequence:** Commit this handoff as `docs: record phase 8.2 canonical training unblock`, push it, verify its exact branch CI, then fast-forward only into unchanged protected `main` and verify final-main CI and remote readback. `DERIVED_CANONICAL_CONFIG_READINESS` is `READY_FOR_OWNER_APPROVAL`, while the original config remains blocked. Do not begin Phase 8.3 without a new explicit owner decision for `APPROVED_CANONICAL_TRAINING_ACTION` (`RUN_DERIVED_CONFIG | DEFER | BLOCKED`).
- **Known limitation:** `git fetch origin` may fail because of the externally managed Codex temporary reference. Do not repair, prune, pack, delete, rename, or modify that reference; use local, origin-tracking, GitHub API, and `git ls-remote` SHA readback as the synchronization fallback.

## Phase 8.3 — Canonical Training Completion

- Owner approval: `APPROVED`; action: `RUN_DERIVED_CONFIG`.
- Frozen training commit and starting `main`: `0c669d58852082785c79699231e09b5ae26757cc`.
- Phase branch: `experiment/phase-8.3-canonical-training-run`.
- Run identity: `deep3-canonical-reference-01`; config: `configs/deep3_canonical.toml`; batch: 64; device: NVIDIA GeForce RTX 3070 Ti, 8 GiB.
- Environment retained for Phase 8.4: Python 3.12.10, torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, CUDA 12.4.
- Dataset identity: `Densu341/Fresh-rotten-fruit` at revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`; archive SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`.
- Execution: one fresh invocation; zero validated resumes; no interruption; exit code 0; completed at `2026-08-03T20:15:26.877171+00:00` after approximately 11 hours 7 minutes 20 seconds.
- Completion: all three folds and 120 epochs per fold completed; fine-tuning boundary at epoch 101; trusted state is `COMPLETED`; no holdout evaluation and no numeric metric publication.
- Artifact integrity: state, manifest, labels, all three fold-best checkpoints, final raw checkpoint, and external log were rehashed after documentation CI; hashes and sizes match `docs/canonical-training-artifacts.md`. Strict CPU loading, finite tensors, and fixed synthetic `(1, 14)` outputs passed for every checkpoint.
- Local-only artifacts: `run_manifest.json` (1,379 B, `5906977c6998e3cac8df07356229c7dd68a0551789637e7f7cca45d2b479ba4c`); `training_state.pt` (217,014,152 B, `00a92f51394bc9a2a2dbb18fa84acb1cb756a0e01e845f64f5817090e7fc03c3`); `label_names.json` (227 B, `c0c229be2509141e1ca3ddf994192b05de69bdec61ea0f97ed554d905eacaae9`); `best_model_fold1.pt` (54,253,480 B, `e89a9f7b6128f1c6a8fbd4885d86ee81ca5c0eac4c6601c35fbabcfae5822a24`); `best_model_fold2.pt` (54,253,480 B, `19eb264786f339fb738c218b891283ac17a6fa449a6f16ae25b18b33c93299ff`); `best_model_fold3.pt` (54,253,480 B, `85a3a2ac5cb373906c81de7a69e223b7c353dedfd51b19ef975340040bd27068`); `last_model_weights.pt` (54,253,932 B, `3fb0e5575ddc4c6ca2bceb955d17a85fd5965bc325ff5b261dded5dab5cbb29f`); external log (25,098,708 B, `3c969b7be56b12a267287f3dc9e504275adbae09062d789478b6e252215227d4`).
- Publication: no dataset, weights, checkpoints, state, log, metrics, GitHub Actions artifact, Release asset, Release, or tag was published or created.
- Documentation commit: `a9675882351b305a2ad98fb5a6e4ecb4f3a144ec` (`docs: record canonical training execution`); exact CI run `30868638224` passed on Ubuntu and Windows.
- Next gate: Phase 8.4 trained-checkpoint holdout evaluation is pending. Do not delete, rename, publish, or relocate canonical artifacts or the external environment before that approved Phase.
- Synchronization note: do not modify Codex-managed temporary refs; `git fetch origin` may fail on those externally managed refs, so use GitHub API and tracking-ref fallback when needed.

## Phase 8.4 - Locked Trained-Checkpoint Holdout Evaluation (2026-08-04)

- **Owner approval and boundary:** The owner approved `RUN_LOCKED_TRAINED_HOLDOUT_EVALUATION` for `deep3-canonical-reference-01`. The action is one locked internal holdout assessment with the canonical CLI raw-logit ensemble plus horizontal-flip TTA and an identical-API detailed metric crosscheck. Dataset, weights, checkpoints, training state, execution log, raw logits, raw predictions, and local evaluation artifacts remain unpublished; Release and tag creation are `NO`; post-holdout tuning is `NO`.
- **Starting state and branch:** The Phase started from synchronized local, origin-tracking, and GitHub `main` SHA `4b3808efb3abaf4682e1150ce69ddcdb6585e451`. The Phase branch is `experiment/phase-8.4-trained-holdout-evaluation`; it was pushed at the same SHA before evaluation. The protected nested `fruit-freshness-classification/` directory remains the only untracked repository path and was untouched. No matching canonical training process was active.
- **Frozen training and evaluation identity:** Training run `deep3-canonical-reference-01` was produced by frozen commit `0c669d58852082785c79699231e09b5ae26757cc`. Evaluation used `configs/deep3_canonical.toml` (SHA-256 `8d40ed34ddcb0eeaea4ca9e03754c579c983e71d1e3b4ae121c512d1fc073c42`), batch 64, three folds, and evaluation commit `4b3808efb3abaf4682e1150ce69ddcdb6585e451`. The evaluation-script SHA-256 is `379b976f196a05f584c39fdef79489f2f5c321d1207c7475f798ea2a1e6794b6`; the ensemble SHA-256 is `e2851d9b60f347e4d85deb1a727a4105e31d878052160e790fa4086d9cae2573`.
- **Environment and dataset:** Retained canonical environment: Python 3.12.10, torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, CUDA 12.4, and NVIDIA GeForce RTX 3070 Ti (8 GiB; driver 591.86). `pip check`, compilation, both CLI help commands, repository contracts, and the full local suite passed before and after documentation. The committed loader verified `Densu341/Fresh-rotten-fruit` revision `2077850adc575aa1e8d6029e6cd6cefe9e403a1c`, archive SHA-256 `a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e`, 30,357 source rows, 26,858 filtered rows, 21,486 training rows, 5,372 holdout rows, RGB decoding, and the preserved 14-label order.
- **Checkpoint policy and integrity:** The canonical ensemble loaded only fold-best EMA checkpoints in order: fold 1 `e89a9f7b6128f1c6a8fbd4885d86ee81ca5c0eac4c6601c35fbabcfae5822a24`, fold 2 `19eb264786f339fb738c218b891283ac17a6fa449a6f16ae25b18b33c93299ff`, and fold 3 `85a3a2ac5cb373906c81de7a69e223b7c353dedfd51b19ef975340040bd27068`. The label artifact SHA-256 is `c0c229be2509141e1ca3ddf994192b05de69bdec61ea0f97ed554d905eacaae9`. `last_model_weights.pt` was explicitly excluded. Every Phase 8.3 artifact was rehashed before and after evaluation; all sizes, hashes, and trusted `COMPLETED` state remained unchanged.
- **Execution and result:** The evaluation lock started at `2026-08-04T03:12:35.5618705Z`. The canonical CLI completed with exit code 0. The completed local evaluation record ended at `2026-08-04T03:19:51.705037+00:00` (436.143167 seconds from the original lock). Correct/total is 5,133 / 5,372; top-1 accuracy is `0.9555100521221147`; the Wilson 95% descriptive interval is `[0.9496634053530901, 0.9607057038344422]`; macro F1 is `0.9037370610237657`; balanced accuracy is `0.8999688026799192`; top-2 is `0.981198808637379`; and top-3 is `0.9927401340282949`.
- **Protocol and crosscheck:** The holdout used deterministic validation preprocessing, equal raw-logit averaging over the three fold-best EMA models, equal original/horizontal-flip TTA averaging, CUDA autocast parity, and argmax top-1 without thresholds or fold weighting. The detailed pass produced labels `(5372,)`, predictions `(5372,)`, and logits `(5372, 14)`; all logits were finite. CLI and detailed top-1 both equal `0.9555100521221147`; confusion-matrix sum is 5,372 and diagonal sum is 5,133; status is `MATCH_EXACT_WITHIN_1E-12`. The initial external detailed runner reached its post-inference UTF-16 CLI-log parser fault before writing detailed artifacts. The owner explicitly approved one replacement detailed pass using the unchanged locked protocol; no competing result was selected.
- **Local-only evaluation artifacts:** CLI log: 18,390 B, `cb4a7ee621fa32a20edc64bd0ad8f7e79f483f7e3d70bcf07336c139edca5e9d`; evaluation JSON: 7,254 B, `592b88a506d946fcb3b4108f3dacfcd0fe15202b8adeda009f61aeaa29446443`; classification report CSV: 932 B, `8c8422311120ca75459ad33a9ecd4541415c2011deb5b708ebf7525d4c2b8213`; confusion matrix CSV: 823 B, `64bbdbc156da4061ccf093a0e51ab6b74706aca441eb34c2debe483797a5d444`; predictions NPZ: 178,468 B, `f36783f2be1d09bbd7178b734ba70023d54860f018a067d9f9cb1b3794331e0c`. All remain ignored, unstaged, untracked, and retained through Phase 8.5.
- **Documentation and CI:** Documentation commit `ac154c39f4f5ae218126a88607c83604961997e8` (`docs: record canonical holdout evaluation`) created the evaluation record, local artifact inventory, readiness/evaluation-guide updates, and offline contract. Exact GitHub Actions run `30874848617` passed on Ubuntu and Windows; it uploaded zero Actions artifacts. The full local suite passed 211 tests; the repository-contract suite passed 72 tests; no local checkpoint, CUDA, dataset, or evaluation output is required by CI.
- **Governance and next gate:** No source, configuration, script, dependency, workflow, notebook, README, license, citation, release, tag, ruleset, metadata, dataset, weight, checkpoint, or binary artifact changed or was published. `v0.1.0`, its zero-asset prerelease, both rulesets, and the local-only backup branch remain unchanged. Phase 8.5 is owner-gated; do not tune, delete, rename, relocate, or publish the retained artifacts before a new explicit decision.
- **Known limitation:** Do not modify Codex-managed temporary refs. `git fetch origin` may be unsafe or uninformative because of the external temporary-ref limitation; use origin-tracking refs, GitHub API, and `git ls-remote` for synchronization verification.
## Phase 8.5 - Canonical Result Interpretation and Publication Decision (2026-08-10)

- **Owner-approved scope:** `INTERPRET_RESULTS_AND_PREPARE_PUBLICATION_DECISION`; result documentation, model-card creation, README result summary, per-class metric documentation, aggregated confusion-matrix documentation, and license/provenance audit are `YES`. Post-holdout tuning, holdout reevaluation, alternate-checkpoint evaluation, and sample-level image review are `NO`.
- **Locked result:** The one frozen `deep3-canonical-reference-01` internal holdout result remains 5,133 / 5,372 top-1 correct (0.955510), macro F1 0.903737, balanced accuracy 0.899969, top-2 0.981199, and top-3 0.992740. The evaluation protocol, checkpoints, dataset identity, and crosscheck are unchanged from Phase 8.4.
- **Documentation:** Added `docs/canonical-results.md` for aggregate interpretation, per-class metrics, aggregate confusion observations, explicit limitations, and publication decision. Added `docs/model-card.md` for scope, intended use, metrics, limitations, artifact availability, and provenance. README, evaluation, reproducibility, governance, changelog, and the existing holdout-artifact record now distinguish the canonical internal result from historical untrained compatibility evidence.
- **Interpretation:** Aggregate accuracy does not describe all classes equally. `freshpotato` has 0.300971 recall and 0.413333 F1; `rottencucumber` and `rottentomato` also have lower recall than the strongest classes. These are aggregate observations only, not causal conclusions.
- **Publication boundary:** Aggregate documentation is tracked. Dataset, checkpoints, weights, training state, execution logs, raw logits, raw predictions, other binaries, GitHub Actions artifacts, Release assets, Releases, and tags remain unpublished. Existing local artifacts are retained unchanged through Phase 8.6; no artifact was copied, moved, renamed, deleted, or added to this worktree.
- **License and provenance:** MIT applies to repository software and project-authored documentation only. The external `Densu341/Fresh-rotten-fruit` source remains governed separately; its surfaced metadata was labeled `openrail`, its surfaced dataset-card README was empty, and no redistribution or trained-weight permission is inferred. This is an operational record, not legal advice.
- **Governance recheck:** GitHub `main` remains `fb009ed66bbdca2e70e195e7e8b5e26bf7178b04`; the final Phase 8.4 CI run remains successful; `v0.1.0` remains a zero-asset prerelease; and `Protect main` plus `Protect v0.1.0` rulesets remain active. No GitHub setting, Release, tag, or artifact operation was performed in this Phase.
- **Local verification:** New result contract, existing holdout contract, and README contract passed (13 tests). The full retained-environment suite passed 214 tests; `compileall src scripts tests`, `python -m scripts.train --help`, and `python -m scripts.evaluate --help` passed. Expected existing test-generated argparse and sklearn warnings remain non-failing.
- **Branch and PR status:** Worktree branch is `docs/phase-8.5-results-publication`, created from `main` at `fb009ed66bbdca2e70e195e7e8b5e26bf7178b04`. Draft PR [#1](https://github.com/kimcheolhui9846/fruit-freshness-classification/pull/1) is open to `main`. Do not merge, release, publish artifacts, or begin Phase 8.6 without a new owner decision.
- **Branch CI verification:** Commit `895ff379fdcf9899c3be13a19240b8af5eeb3b4f` was pushed to `origin/docs/phase-8.5-results-publication`. GitHub Actions [Repository CI run 31355226439](https://github.com/kimcheolhui9846/fruit-freshness-classification/actions/runs/31355226439) completed successfully on that exact SHA: Ubuntu Python 3.12 and Windows Python 3.12 both passed dependency, import, configuration, compilation, unit/contract, CLI, and repository-cleanliness checks. The workflow uploaded no artifacts. A draft PR to `main` remains the next approved integration action; no merge has been performed.
### Phase 8.5 PR Correction — Frozen Evidence and Publication Governance (2026-08-10)

- **Correction basis:** Review found that Phase 8.5 had added status and forward links to four immutable Phase 8.4 execution-evidence files. Those files are restored byte-for-byte from frozen base `fb009ed66bbdca2e70e195e7e8b5e26bf7178b04`: `docs/canonical-holdout-evaluation.md`, `docs/canonical-holdout-artifacts.md`, `docs/evaluation.md`, and `tests/repository/test_canonical_holdout_evaluation_contract.py`.
- **Completed record:** `docs/canonical-results.md` now records the frozen 239 top-1 errors, 194 concentrated errors (approximately 81.17%), 5,271 top-2 correct, 5,333 top-3 correct, 138 rank-2 recoveries, 62 additional rank-3 recoveries, and 39 outside top-3. It explicitly separates observed facts, supported inferences, and unverified hypotheses; no causal claim, image inspection, retraining, evaluation rerun, or post-holdout tuning occurred.
- **Publication governance:** `docs/artifact-publication-decision.md` is the canonical documentation-only publication decision. Aggregate documentation and the model card are public; trained weights and fold checkpoints are blocked pending separate license/provenance clearance; all local binary artifacts remain local-only through Phase 8.6. Dataset, states, logs, raw logits, raw predictions, Actions artifacts, Release assets, Releases, and tags remain unpublished.
- **Safety and next gate:** The model card now excludes food-safety, health, regulatory, laboratory-replacement, autonomous disposal/rejection, and unvalidated-domain use. The Phase 8.6 owner-gate fields remain deliberately unresolved. Do not merge the draft PR, mark it ready for review, publish binaries, or begin Phase 8.6 without a new explicit owner approval.
## Phase 8.6 — Canonical Run Closure and Local-Only Artifact Preservation

```text
PHASE:
8.6 — Canonical Run Closure and Local-Only Artifact Preservation
OWNER_APPROVAL_STATUS:
APPROVED
STARTING_MAIN:
9fc7477bb1da43eeefbae5e497a2ca76310871f5
PHASE_BRANCH:
docs/phase-8.6-canonical-run-closure
RUN_ID:
deep3-canonical-reference-01
CANONICAL_RUN_STATUS:
CLOSED_REFERENCE
IMPLEMENTATION_COMMIT:
f62156d3c38116f80a92d9d86a77c95099325226

Closure contract:
passed
Repository contracts:
79 passed
Full suite:
218 passed
compileall:
passed
scripts.train --help:
passed
scripts.evaluate --help:
passed

Push CI:
31364353731
PR CI:
31364357148
Ubuntu:
success
Windows:
success
Actions artifacts:
0

APPROVED_NEXT_ACTION:
KEEP_ALL_BINARY_ARTIFACTS_LOCAL_ONLY
APPROVED_DATASET_LICENSE_CLEARANCE:
NOT_CONFIRMED
BINARY_PUBLICATION_GATE:
CLOSED_WITHOUT_PUBLICATION
MODEL_WEIGHT_PUBLICATION:
NO
CHECKPOINT_PUBLICATION:
NO
DATASET_PUBLICATION:
NO
BINARY_RETENTION:
KEEP_LOCAL_ONLY
RETENTION_DURATION:
UNTIL_EXPLICIT_OWNER_CHANGE
DELETION:
NOT AUTHORIZED
RELOCATION:
NOT AUTHORIZED
CONVERSION:
NOT AUTHORIZED
PACKAGING:
NOT AUTHORIZED
REMOTE UPLOAD:
NOT AUTHORIZED

retained canonical artifacts:
13
size verification:
passed
SHA-256 verification:
passed
mutation:
none
publication:
none

PR:
#2
state:
Draft
base:
main
implementation head:
f62156d3c38116f80a92d9d86a77c95099325226
READY_FOR_OWNER_PR_REVIEW:
PENDING_HANDOFF_CI
OWNER_PR_MERGE_APPROVAL:
NOT YET REQUESTED
PHASE_9:
NOT STARTED
```

No training, holdout reevaluation, alternate-checkpoint evaluation, or post-holdout tuning occurred in Phase 8.6. The canonical holdout is already observed; future model development requires a new experiment identity, and the observed holdout may not be presented as untouched evidence after future tuning. The Phase worktree remains isolated; private artifact paths and binary contents are not recorded here.