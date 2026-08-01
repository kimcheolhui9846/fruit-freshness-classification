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
