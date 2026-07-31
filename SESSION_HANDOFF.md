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
