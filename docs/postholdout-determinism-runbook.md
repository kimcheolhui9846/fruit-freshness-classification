# Phase 9.7 Determinism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the training and evaluation pipeline explicit seeding and a selectable backend determinism policy, verify it with a bounded two-run comparison, and correct the top-level reproducibility documentation.

**Architecture:** One pure policy module applies seeding and backend settings and returns a record of what it actually applied. The configuration loader validates two new optional `[runtime]` keys, which keeps every frozen configuration's SHA-256 and meaning intact. Training and both evaluation entry points call the policy once at start-up, before any CUDA work, and record the applied policy in their output. A separate comparison script reads two completed output directories and reports whether they are bit-exact; it runs no training.

**Tech Stack:** Python 3.12, PyTorch 2.6.0+cu124, NumPy 2.5.1, `unittest` (this repository does not use pytest), TOML via `tomllib`.

## Global Constraints

- The frozen protocol is `docs/postholdout-determinism-protocol.md`. Values below are copied from it and may not be changed during implementation.
- Seed is `20260815`. Determinism levels are exactly `A_STRICT`, `B_CUDNN`, `C_SEED_ONLY`.
- `configs/deep3_postholdout_baseline.toml`, `configs/deep3_canonical.toml`, `configs/deep3_postholdout_loss001.toml`, `configs/deep3_postholdout_baseline_rep002.toml`, and `configs/deep3_postholdout_baseline_rep003.toml` **must not change by a single byte**. Their hashes are recorded in frozen protocol documents and asserted by contract tests.
- The two new `[runtime]` keys are **optional**. A configuration without them must behave exactly as before.
- No `DataLoader` generator is added. See Task 3 for the reason; it is a deliberate rejection of the Phase 9.6a follow-up text.
- The verification configuration uses the post-holdout development route only. The canonical route trains on the 4,298 locked-test examples and would break `POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES: 0`.
- No task in this plan executes a training run. Execution is a separate owner decision recorded as `APPROVED_EXECUTION: NOT_YET_GRANTED`.
- Tests run with `.venv/Scripts/python.exe -m unittest`. The full suite is currently **340 tests, 0 failures**.
- Commit messages name no tool or assistant, and carry no `Co-Authored-By` trailer. The repository owner is sole author.

## File Structure

| File | Responsibility |
|---|---|
| `src/utils/config.py` (modify) | Owns the level name constants and validates the optional keys. Stays standard-library only. Registers the verification lineage. |
| `src/utils/determinism.py` (create) | Applies seeding and the backend policy; returns the applied record. Knows nothing about files or the CLI. |
| `scripts/train.py` (modify) | Applies the policy once before device resolution; records it in the run manifest. |
| `scripts/evaluate.py` (modify) | Applies the policy in place of the bare `cudnn_benchmark` assignment. |
| `scripts/evaluate_postholdout_baseline.py` (modify) | Same, and records the policy in the OOF integrity block. |
| `scripts/verify_determinism.py` (create) | Compares two completed output directories by digest. Runs no training. |
| `configs/deep3_postholdout_determinism_check.toml` (create) | Bounded verification configuration. |
| `tests/utils/test_determinism.py` (create) | Policy module behaviour. |
| `tests/config/test_determinism_config.py` (create) | Optional-key validation and frozen-hash preservation. |
| `tests/scripts/test_verify_determinism.py` (create) | Digest behaviour and the CLI. |
| `tests/repository/test_determinism_protocol_contract.py` (create) | Pins the frozen protocol. |

### Why the level constants live in `src/utils/config.py`

`src/utils/config.py` is documented as a standard-library loader and imports only `hashlib`, `pathlib`, and `tomllib`. It must validate the level name, so it needs the vocabulary. Putting the constants in `determinism.py` and importing them into `config.py` would pull `torch` and `numpy` into configuration loading. The dependency therefore runs the other way: `config.py` declares the names, `determinism.py` imports them.

### Why `validate_loss_experiment_config` is not renamed

Task 2 registers a second lineage through this function, so its name no longer matches every caller. It is **not** renamed. `docs/postholdout-loss001-runbook.md` records the Phase 9.6 implementation and names the function nine times; renaming it would make a completed phase's record wrong. Its docstring is extended to state that it is lineage-generic and that the name records its first caller.

---

### Task 1: Determinism policy module

**Files:**
- Modify: `src/utils/config.py` (add constants near the top, after `_REQUIRED_TYPES`)
- Create: `src/utils/determinism.py`
- Create: `tests/utils/__init__.py`
- Test: `tests/utils/test_determinism.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `A_STRICT`, `B_CUDNN`, `C_SEED_ONLY`, `DETERMINISM_LEVELS` from `src.utils.config`; `seed_everything(seed: int) -> None`, `apply_determinism(level: str, *, seed: int) -> dict[str, object]`, and `resolve_policy(config: dict) -> dict[str, object]` from `src.utils.determinism`. The returned record always has exactly the keys `seed`, `level`, `cudnn_benchmark`, `cudnn_deterministic`, `use_deterministic_algorithms`, `cublas_workspace_config`.

- [ ] **Step 1: Add the level constants to `src/utils/config.py`**

Insert immediately after the `_REQUIRED_TYPES` dictionary (which currently ends at line 34):

```python
# Phase 9.7 determinism level names. They live here, not in
# src/utils/determinism.py, because this module is a standard-library-only
# loader and must validate the name without importing torch or numpy.
A_STRICT = "A_STRICT"
B_CUDNN = "B_CUDNN"
C_SEED_ONLY = "C_SEED_ONLY"
DETERMINISM_LEVELS = (A_STRICT, B_CUDNN, C_SEED_ONLY)
_CUDNN_CONSTRAINED_LEVELS = (A_STRICT, B_CUDNN)
```

- [ ] **Step 2: Create `tests/utils/__init__.py`**

Every test subdirectory in this repository has one. Create it empty:

```bash
printf '' > tests/utils/__init__.py
```

- [ ] **Step 3: Write the failing tests**

Create `tests/utils/test_determinism.py`:

```python
"""Behaviour of the Phase 9.7 seeding and backend determinism policy."""

import os
import random
import unittest

import numpy as np
import torch

from src.utils.config import A_STRICT, B_CUDNN, C_SEED_ONLY, DETERMINISM_LEVELS
from src.utils.determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    apply_determinism,
    resolve_policy,
    seed_everything,
)


RECORD_KEYS = {
    "seed",
    "level",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "use_deterministic_algorithms",
    "cublas_workspace_config",
}


def _draw() -> tuple:
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


class SeedEverythingTest(unittest.TestCase):
    def test_same_seed_reproduces_every_generator(self):
        seed_everything(20260815)
        first = _draw()
        seed_everything(20260815)
        second = _draw()
        # All three generators feed the training pipeline. Seeding two of
        # three would leave the run nondeterministic while looking seeded.
        self.assertEqual(first, second)

    def test_different_seeds_produce_different_draws(self):
        seed_everything(20260815)
        first = _draw()
        seed_everything(20260816)
        second = _draw()
        self.assertNotEqual(first, second)

    def test_negative_and_non_integer_seeds_are_rejected(self):
        for value in (-1, 1.0, "20260815", True):
            with self.assertRaises(ValueError):
                seed_everything(value)


class ApplyDeterminismTest(unittest.TestCase):
    def setUp(self):
        self.original_benchmark = torch.backends.cudnn.benchmark
        self.original_deterministic = torch.backends.cudnn.deterministic
        self.original_strict = torch.are_deterministic_algorithms_enabled()
        self.original_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    def tearDown(self):
        torch.backends.cudnn.benchmark = self.original_benchmark
        torch.backends.cudnn.deterministic = self.original_deterministic
        torch.use_deterministic_algorithms(self.original_strict)
        if self.original_env is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.original_env

    def test_unknown_level_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_determinism("MOSTLY_DETERMINISTIC", seed=20260815)

    def test_b_cudnn_disables_the_autotuner_without_strict_algorithms(self):
        record = apply_determinism(B_CUDNN, seed=20260815)

        self.assertFalse(record["cudnn_benchmark"])
        self.assertTrue(record["cudnn_deterministic"])
        self.assertFalse(record["use_deterministic_algorithms"])
        self.assertIsNone(record["cublas_workspace_config"])

    def test_a_strict_enables_strict_algorithms_and_the_cublas_workspace(self):
        record = apply_determinism(A_STRICT, seed=20260815)

        self.assertFalse(record["cudnn_benchmark"])
        self.assertTrue(record["cudnn_deterministic"])
        self.assertTrue(record["use_deterministic_algorithms"])
        self.assertEqual(record["cublas_workspace_config"], CUBLAS_WORKSPACE_CONFIG)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_WORKSPACE_CONFIG)

    def test_c_seed_only_leaves_the_backend_untouched(self):
        torch.backends.cudnn.benchmark = True
        record = apply_determinism(C_SEED_ONLY, seed=20260815)

        # C exists so the vocabulary is complete and testable. It must not
        # quietly acquire B's behaviour.
        self.assertTrue(record["cudnn_benchmark"])
        self.assertFalse(record["use_deterministic_algorithms"])

    def test_record_reports_the_applied_state_not_the_request(self):
        record = apply_determinism(B_CUDNN, seed=20260815)

        self.assertEqual(set(record), RECORD_KEYS)
        self.assertEqual(record["seed"], 20260815)
        self.assertEqual(record["level"], B_CUDNN)
        # Read back from torch, so a setting that silently failed to apply
        # cannot be recorded as applied.
        self.assertEqual(record["cudnn_benchmark"], torch.backends.cudnn.benchmark)
        self.assertEqual(record["cudnn_deterministic"], torch.backends.cudnn.deterministic)

    def test_every_declared_level_is_applicable(self):
        for level in DETERMINISM_LEVELS:
            with self.subTest(level=level):
                record = apply_determinism(level, seed=20260815)
                self.assertEqual(record["level"], level)


class ResolvePolicyTest(unittest.TestCase):
    def setUp(self):
        self.original_benchmark = torch.backends.cudnn.benchmark
        self.original_strict = torch.are_deterministic_algorithms_enabled()

    def tearDown(self):
        torch.backends.cudnn.benchmark = self.original_benchmark
        torch.use_deterministic_algorithms(self.original_strict)

    def test_config_without_the_keys_keeps_the_legacy_behaviour(self):
        record = resolve_policy({"runtime": {"cudnn_benchmark": True}})

        # The pre-9.7 pipeline set only cudnn.benchmark and seeded nothing.
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertIsNone(record["seed"])
        self.assertIsNone(record["level"])
        self.assertFalse(record["use_deterministic_algorithms"])

    def test_legacy_record_has_the_same_keys_as_an_applied_record(self):
        legacy = resolve_policy({"runtime": {"cudnn_benchmark": True}})

        # "This run was unseeded" must be a recorded fact, not a missing field.
        self.assertEqual(set(legacy), RECORD_KEYS)

    def test_config_with_the_keys_applies_the_named_level(self):
        record = resolve_policy(
            {
                "runtime": {
                    "cudnn_benchmark": False,
                    "seed": 20260815,
                    "determinism_level": B_CUDNN,
                }
            }
        )

        self.assertEqual(record["seed"], 20260815)
        self.assertEqual(record["level"], B_CUDNN)
        self.assertFalse(torch.backends.cudnn.benchmark)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.utils.test_determinism -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.determinism'`

- [ ] **Step 5: Create `src/utils/determinism.py`**

```python
"""Explicit seeding and backend determinism policy.

Before Phase 9.7 this pipeline set no random seed anywhere, which left a
run-to-run aggregate Macro F1 noise floor of 2s = 0.012177 -- wide enough to
swallow the effect Phase 9.6 was trying to measure. The frozen protocol is
docs/postholdout-determinism-protocol.md.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.config import A_STRICT, C_SEED_ONLY, DETERMINISM_LEVELS


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_CUBLAS_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"


def seed_everything(seed: int) -> None:
    """Seed every global generator the pipeline draws from.

    Weight initialisation and DropPath draw from the torch CPU generator;
    DataLoader shuffling draws from it too, because RandomSampler takes its
    seed from the global generator when no generator is supplied; mixup draws
    from both NumPy and torch. Seeding a subset would leave the run
    nondeterministic while appearing seeded.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _policy_record(seed: int | None, level: str | None) -> dict[str, object]:
    """Read the applied state back from torch rather than echoing the request."""
    return {
        "seed": seed,
        "level": level,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get(_CUBLAS_ENV_VAR),
    }


def apply_determinism(level: str, *, seed: int) -> dict[str, object]:
    """Seed every generator and apply the level's backend policy."""
    if level not in DETERMINISM_LEVELS:
        expected = ", ".join(DETERMINISM_LEVELS)
        raise ValueError(f"Unknown determinism level {level!r}; expected one of {expected}.")
    if level == A_STRICT and torch.cuda.is_initialized():
        raise RuntimeError(
            f"{A_STRICT} must be applied before CUDA is initialized: "
            f"{_CUBLAS_ENV_VAR} is read when the cuBLAS handle is created, and "
            "setting it afterwards is silently ignored rather than refused."
        )

    seed_everything(seed)

    strict = level == A_STRICT
    if level != C_SEED_ONLY:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if strict:
        os.environ[_CUBLAS_ENV_VAR] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(strict)

    return _policy_record(seed, level)


def resolve_policy(config: dict) -> dict[str, object]:
    """Apply the configuration's determinism policy and return what was applied.

    A configuration without the optional keys keeps the pre-9.7 behaviour
    exactly: nothing is seeded, and only cudnn.benchmark is set. The record is
    still returned in full, so an unseeded run records that it was unseeded
    instead of omitting the field.
    """
    runtime = config["runtime"]
    level = runtime.get("determinism_level")
    if level is None:
        torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
        return _policy_record(None, None)
    return apply_determinism(level, seed=runtime["seed"])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.utils.test_determinism -v`
Expected: PASS, 12 tests

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 352 tests` (340 + 12), `OK`

- [ ] **Step 8: Commit**

```bash
git add src/utils/config.py src/utils/determinism.py tests/utils/__init__.py tests/utils/test_determinism.py
git commit -m "feat: add the phase 9.7 seeding and determinism policy module"
```

---

### Task 2: Optional configuration keys and the verification lineage

**Files:**
- Modify: `src/utils/config.py` (`_validate_config` at line 46, and the lineage block at lines 264-301)
- Create: `configs/deep3_postholdout_determinism_check.toml`
- Test: `tests/config/test_determinism_config.py`

**Interfaces:**
- Consumes: `A_STRICT`, `B_CUDNN`, `DETERMINISM_LEVELS`, `_CUDNN_CONSTRAINED_LEVELS` from Task 1; the existing `validate_loss_experiment_config(baseline_path, experiment_path, *, allowed_differences: frozenset[str], expected_values: dict[str, object]) -> dict` and `resolve_experiment_validation(config: dict, config_path: str | Path) -> dict | None`.
- Produces: `DETERMINISM_CHECK_PARENT_ID`, `DETERMINISM_CHECK_ALLOWED_DIFFERENCES`, `DETERMINISM_CHECK_EXPECTED_VALUES` in `src.utils.config`; the file `configs/deep3_postholdout_determinism_check.toml`.

- [ ] **Step 1: Write the failing tests**

Create `tests/config/test_determinism_config.py`:

```python
"""Validation of the optional Phase 9.7 runtime keys and the check lineage."""

import copy
from pathlib import Path
import tempfile
import tomllib
import unittest

from src.utils.config import (
    A_STRICT,
    B_CUDNN,
    C_SEED_ONLY,
    DETERMINISM_CHECK_ALLOWED_DIFFERENCES,
    DETERMINISM_CHECK_EXPECTED_VALUES,
    load_experiment_config,
    resolve_experiment_validation,
)


ROOT = Path(__file__).resolve().parents[2]
CHECK_CONFIG = ROOT / "configs" / "deep3_postholdout_determinism_check.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"


def _minimal_runtime(**overrides) -> dict:
    runtime = {"cudnn_benchmark": True}
    runtime.update(overrides)
    return runtime


def _config_with_runtime(runtime: dict) -> dict:
    base = tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(base)
    config["runtime"] = runtime
    return config


class OptionalRuntimeKeyTest(unittest.TestCase):
    """These drive the validator through a written file, because that is the
    only path production code uses."""

    def _write(self, config: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "candidate.toml"
        lines = []
        for section, values in config.items():
            lines.append(f"[{section}]")
            for key, value in values.items():
                if isinstance(value, bool):
                    rendered = "true" if value else "false"
                elif isinstance(value, str):
                    rendered = f'"{value}"'
                elif isinstance(value, list):
                    rendered = "[" + ", ".join(str(item) for item in value) + "]"
                else:
                    rendered = repr(value)
                lines.append(f"{key} = {rendered}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_config_without_the_new_keys_still_loads(self):
        path = self._write(_config_with_runtime(_minimal_runtime()))
        config = load_experiment_config(path)

        # Optional means optional. Every frozen config lacks these keys.
        self.assertNotIn("seed", config["runtime"])
        self.assertNotIn("determinism_level", config["runtime"])

    def test_seed_without_a_level_is_rejected(self):
        path = self._write(_config_with_runtime(_minimal_runtime(seed=20260815)))
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_level_without_a_seed_is_rejected(self):
        path = self._write(
            _config_with_runtime(_minimal_runtime(determinism_level=B_CUDNN))
        )
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_unknown_level_name_is_rejected(self):
        path = self._write(
            _config_with_runtime(
                _minimal_runtime(seed=20260815, determinism_level="MOSTLY")
            )
        )
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_negative_seed_is_rejected(self):
        path = self._write(
            _config_with_runtime(
                _minimal_runtime(seed=-1, determinism_level=B_CUDNN)
            )
        )
        with self.assertRaises(ValueError):
            load_experiment_config(path)

    def test_constrained_levels_reject_the_autotuner(self):
        for level in (A_STRICT, B_CUDNN):
            with self.subTest(level=level):
                path = self._write(
                    _config_with_runtime(
                        {
                            "cudnn_benchmark": True,
                            "seed": 20260815,
                            "determinism_level": level,
                        }
                    )
                )
                # A config that asks for determinism and leaves the autotuner
                # on contradicts itself; the autotuner picks kernels
                # nondeterministically.
                with self.assertRaises(ValueError):
                    load_experiment_config(path)

    def test_seed_only_level_permits_the_autotuner(self):
        path = self._write(
            _config_with_runtime(
                {
                    "cudnn_benchmark": True,
                    "seed": 20260815,
                    "determinism_level": C_SEED_ONLY,
                }
            )
        )
        config = load_experiment_config(path)

        self.assertEqual(config["runtime"]["determinism_level"], C_SEED_ONLY)


class DeterminismCheckConfigTest(unittest.TestCase):
    def test_check_config_uses_the_frozen_development_route(self):
        config = load_experiment_config(CHECK_CONFIG)

        # The canonical route trains on the 4,298 locked-test examples.
        self.assertIn("post_holdout", config)
        self.assertEqual(
            config["post_holdout"]["split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(
            config["post_holdout"]["cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )

    def test_check_config_is_bounded_and_exercises_fine_tuning(self):
        config = load_experiment_config(CHECK_CONFIG)

        # epochs 2 with fine_tuning 1 puts one normal epoch and one
        # fine-tuning epoch in every fold.
        self.assertEqual(config["training"]["epochs"], 2)
        self.assertEqual(config["fine_tuning"]["epochs"], 1)
        self.assertEqual(config["training"]["batch_size"], 64)

    def test_check_config_carries_the_frozen_seed(self):
        config = load_experiment_config(CHECK_CONFIG)

        self.assertEqual(config["runtime"]["seed"], 20260815)
        self.assertFalse(config["runtime"]["cudnn_benchmark"])

    def test_check_lineage_validates_and_names_its_differences(self):
        config = load_experiment_config(CHECK_CONFIG)
        result = resolve_experiment_validation(config, CHECK_CONFIG)

        self.assertIsNotNone(result)
        self.assertTrue(result["single_factor_verified"])
        self.assertEqual(
            set(result["differences"]),
            {
                "runtime.cudnn_benchmark",
                "runtime.seed",
                "runtime.determinism_level",
                "training.epochs",
                "fine_tuning.epochs",
                "post_holdout.experiment_id",
                "post_holdout.parent_experiment_id",
                "post_holdout.artifact_namespace",
            },
        )

    def test_check_lineage_does_not_pin_the_level(self):
        # The frozen ladder descends from A_STRICT to B_CUDNN. The level must
        # be a registered difference, so the check config is allowed to name
        # it, but it must not be a pinned value, or the registered descent
        # could never be executed.
        self.assertIn(
            "runtime.determinism_level", DETERMINISM_CHECK_ALLOWED_DIFFERENCES
        )
        self.assertNotIn(
            "runtime.determinism_level", DETERMINISM_CHECK_EXPECTED_VALUES
        )

    def test_unregistered_parent_still_raises(self):
        config = load_experiment_config(CHECK_CONFIG)
        config["post_holdout"]["parent_experiment_id"] = "deep3-unregistered"

        with self.assertRaises(ValueError):
            resolve_experiment_validation(config, CHECK_CONFIG)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.config.test_determinism_config -v`
Expected: FAIL — `configs/deep3_postholdout_determinism_check.toml` does not exist, and the optional-key tests do not raise.

- [ ] **Step 3: Add the validation to `src/utils/config.py`**

In `_validate_config`, add one line immediately before the existing `if "post_holdout" in config:` block (currently line 73):

```python
    _validate_runtime_determinism(config["runtime"])
```

Add this function immediately after `_validate_figure_size` (which currently ends at line 117):

```python
def _validate_runtime_determinism(runtime: dict) -> None:
    """Validate the optional Phase 9.7 determinism keys.

    Both keys are optional so that every configuration frozen before Phase 9.7
    keeps its recorded SHA-256 and its original meaning. They must appear
    together: a seed with no level, or a level with no seed, describes a
    behaviour nobody can read off the file.
    """
    has_seed = "seed" in runtime
    has_level = "determinism_level" in runtime
    if has_seed != has_level:
        raise ValueError(
            "[runtime].seed and [runtime].determinism_level must be set together."
        )
    if not has_seed:
        return

    seed = runtime["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("[runtime].seed must be a non-negative integer.")

    level = runtime["determinism_level"]
    if type(level) is not str or level not in DETERMINISM_LEVELS:
        expected = ", ".join(DETERMINISM_LEVELS)
        raise ValueError(f"[runtime].determinism_level must be one of {expected}.")

    if level in _CUDNN_CONSTRAINED_LEVELS and runtime["cudnn_benchmark"]:
        raise ValueError(
            f"[runtime].determinism_level {level} requires cudnn_benchmark = false; "
            "the cuDNN autotuner selects kernels nondeterministically."
        )
```

- [ ] **Step 4: Register the verification lineage in `src/utils/config.py`**

Add these constants immediately after `LOSS001_EXPECTED_VALUES` (currently line 269):

```python
DETERMINISM_CHECK_PARENT_ID = "deep3-postholdout-determinism-check"
DETERMINISM_CHECK_ALLOWED_DIFFERENCES = frozenset(
    {
        "runtime.cudnn_benchmark",
        "runtime.seed",
        "runtime.determinism_level",
        "training.epochs",
        "fine_tuning.epochs",
        "post_holdout.experiment_id",
        "post_holdout.parent_experiment_id",
        "post_holdout.artifact_namespace",
    }
)
# The level is deliberately absent: the frozen ladder descends from A_STRICT to
# B_CUDNN, and pinning it here would make the registered descent unexecutable.
DETERMINISM_CHECK_EXPECTED_VALUES = {
    "runtime.seed": 20260815,
    "runtime.cudnn_benchmark": False,
    "training.epochs": 2,
    "fine_tuning.epochs": 1,
    "post_holdout.split_manifest_path": (
        "configs/splits/deep3-postholdout-research-01.json"
    ),
    "post_holdout.cv_manifest_path": (
        "configs/splits/deep3-postholdout-research-01-baseline-cv.json"
    ),
}
```

In `resolve_experiment_validation`, add this branch immediately before the final `raise ValueError` (currently line 298):

```python
    if parent == DETERMINISM_CHECK_PARENT_ID:
        return validate_loss_experiment_config(
            BASELINE_CONFIG_PATH,
            config_path,
            allowed_differences=DETERMINISM_CHECK_ALLOWED_DIFFERENCES,
            expected_values=DETERMINISM_CHECK_EXPECTED_VALUES,
        )
```

Extend the `validate_loss_experiment_config` docstring (currently line 223) to:

```python
    """Verify an experiment config changes exactly its registered factor.

    The name records the first caller, the Phase 9.6 loss experiment. The
    function is lineage-generic and is also used by the Phase 9.7 determinism
    check. It is not renamed because docs/postholdout-loss001-runbook.md
    records the Phase 9.6 implementation and names it throughout.
    """
```

- [ ] **Step 5: Create `configs/deep3_postholdout_determinism_check.toml`**

Every value not listed in `DETERMINISM_CHECK_ALLOWED_DIFFERENCES` is copied verbatim from `configs/deep3_postholdout_baseline.toml`.

```toml
[runtime]
cudnn_benchmark = false
seed = 20260815
determinism_level = "A_STRICT"

[loss]
class_balanced_beta = 0.999
use_ce_label_smoothing = true
label_smoothing = 0.01
focal_gamma = 2.0

[training]
epochs = 2
batch_size = 64

[fine_tuning]
epochs = 1

[cross_validation]
n_splits = 3
shuffle = true
random_state = 42

[mixup]
alpha = 0.8
probability = 0.5

[optimization]
lr_cnn = 5e-5
lr_trans = 1e-4
weight_decay = 1e-4

[ema]
decay = 0.999

[checkpoint]
final_model_filename = "last_model_weights.pt"

[reporting]
figure_size = [10, 4]

[post_holdout]
experiment_id = "deep3-postholdout-determinism-check-01"
parent_experiment_id = "deep3-postholdout-determinism-check"
split_manifest_path = "configs/splits/deep3-postholdout-research-01.json"
cv_manifest_path = "configs/splits/deep3-postholdout-research-01-baseline-cv.json"
artifact_namespace = "deep3-postholdout-determinism-check-01"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.config.test_determinism_config -v`
Expected: PASS, 13 tests

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 365 tests`, `OK`. If any existing config test fails, the new validation is rejecting a configuration that must stay valid — fix the validation, never the frozen configuration.

- [ ] **Step 8: Commit**

```bash
git add src/utils/config.py configs/deep3_postholdout_determinism_check.toml tests/config/test_determinism_config.py
git commit -m "feat: validate optional determinism keys and register the check lineage"
```

---

### Task 3: Apply the policy in training and record it

**Files:**
- Modify: `scripts/train.py:33` (`RUN_MANIFEST_SCHEMA_VERSION`), `scripts/train.py:232-277` (`_build_run_manifest`), `scripts/train.py:533-552` (`run_training` start-up), `scripts/train.py:680` (delete)
- Test: `tests/scripts/test_train_orchestration.py` (extend)

**Interfaces:**
- Consumes: `resolve_policy(config: dict) -> dict[str, object]` from Task 1.
- Produces: run manifests at `schema_version` 2 carrying a `determinism` block with the six keys `seed`, `level`, `cudnn_benchmark`, `cudnn_deterministic`, `use_deterministic_algorithms`, `cublas_workspace_config`.

**Why no `DataLoader` generator is added.** `docs/postholdout-noise-floor-protocol.md` proposed one. It must not be added. With `num_workers = 0` and `shuffle = True`, `RandomSampler` draws its seed from the **global** torch generator on each `__iter__`, and `src/engine/training_state.py` already persists and restores that global state at every epoch boundary. A separate `torch.Generator` would live outside `training_state.pt`, so a resumed run would reseed it and replay the first epoch's ordering. Adding it would break resume determinism that currently works.

- [ ] **Step 1: Write the failing tests**

Append to `tests/scripts/test_train_orchestration.py`. Follow the existing module's loading idiom — it already imports `scripts/train.py` through `importlib.util.spec_from_file_location`, because `tests/scripts` shadows `scripts` on the import path. Reuse the module object that file already builds.

```python
class DeterminismManifestTest(unittest.TestCase):
    def test_manifest_schema_version_is_two(self):
        # Adding a field without bumping the version would let a v1 manifest
        # and a v2 manifest both claim v1.
        self.assertEqual(train_module.RUN_MANIFEST_SCHEMA_VERSION, 2)

    def test_manifest_records_an_unseeded_run_explicitly(self):
        manifest = train_module._build_run_manifest(
            metadata=_manifest_metadata(),
            config=_manifest_config(),
            device=torch.device("cpu"),
            resume_enabled=True,
            determinism={
                "seed": None,
                "level": None,
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
                "use_deterministic_algorithms": False,
                "cublas_workspace_config": None,
            },
        )

        # "This run was unseeded" is the fact Phase 9.7 exists to make
        # visible. It must be recorded, not omitted.
        self.assertIn("determinism", manifest)
        self.assertIsNone(manifest["determinism"]["seed"])
        self.assertIsNone(manifest["determinism"]["level"])

    def test_manifest_records_an_applied_policy(self):
        manifest = train_module._build_run_manifest(
            metadata=_manifest_metadata(),
            config=_manifest_config(),
            device=torch.device("cpu"),
            resume_enabled=True,
            determinism={
                "seed": 20260815,
                "level": "A_STRICT",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "use_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
            },
        )

        self.assertEqual(manifest["determinism"]["seed"], 20260815)
        self.assertEqual(manifest["determinism"]["level"], "A_STRICT")
        self.assertEqual(manifest["schema_version"], 2)

    def test_manifest_determinism_block_is_a_copy(self):
        record = {
            "seed": 20260815,
            "level": "B_CUDNN",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "use_deterministic_algorithms": False,
            "cublas_workspace_config": None,
        }
        manifest = train_module._build_run_manifest(
            metadata=_manifest_metadata(),
            config=_manifest_config(),
            device=torch.device("cpu"),
            resume_enabled=True,
            determinism=record,
        )
        record["seed"] = 999

        # A manifest records what happened; a later mutation of the caller's
        # dict must not rewrite it.
        self.assertEqual(manifest["determinism"]["seed"], 20260815)
```

Define the two helpers at module scope in the same test file, above the class:

```python
def _manifest_metadata() -> dict:
    return {
        "run_id": "determinism-check-a",
        "repository_commit": "0" * 40,
        "config_path": "configs/deep3_postholdout_determinism_check.toml",
        "config_sha256": "1" * 64,
        "dataset_repository": "Densu341/Fresh-rotten-fruit",
        "dataset_revision": "2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
        "dataset_archive_sha256": "2" * 64,
        "num_classes": 14,
        "num_folds": 3,
        "epochs": 2,
        "fine_tuning_epochs": 1,
        "batch_size": 64,
    }


def _manifest_config() -> dict:
    return {
        "optimization": {"lr_cnn": 5e-5, "lr_trans": 1e-4, "weight_decay": 1e-4},
        "mixup": {"alpha": 0.8, "probability": 0.5},
        "ema": {"decay": 0.999},
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_train_orchestration -v`
Expected: FAIL — `RUN_MANIFEST_SCHEMA_VERSION` is 1, and `_build_run_manifest()` got an unexpected keyword argument `determinism`.

- [ ] **Step 3: Bump the schema version**

`scripts/train.py:33`:

```python
RUN_MANIFEST_SCHEMA_VERSION = 2
```

The bump is safe: `_load_and_validate_manifest` requires an exact match, but it runs only on resume, and no run is in flight. Every recorded run has status `COMPLETED`, which normal resume rejects regardless.

- [ ] **Step 4: Record the policy in the manifest**

Add the parameter to `_build_run_manifest`'s signature (currently lines 232-238):

```python
def _build_run_manifest(
    *,
    metadata: dict[str, object],
    config: dict,
    device: torch.device,
    resume_enabled: bool,
    determinism: dict[str, object],
) -> dict[str, object]:
```

Add the entry inside the `manifest` dictionary, immediately after the `"cuda_version"` line (currently line 264):

```python
        "determinism": copy.deepcopy(determinism),
```

- [ ] **Step 5: Apply the policy at start-up**

Replace `run_training`'s opening block (currently lines 533-541) with:

```python
def run_training(args: argparse.Namespace) -> dict:
    """Run the notebook-equivalent training flow with optional stateful resume."""
    config_path = _resolve_repository_path(args.config)
    output_directory = _resolve_repository_path(args.output_dir)
    options = _stateful_options(args)

    config = load_experiment_config(config_path)
    # Before resolve_device, because A_STRICT sets CUBLAS_WORKSPACE_CONFIG and
    # that variable is read when the cuBLAS handle is created. Applying it
    # after any CUDA work would be ignored rather than refused.
    determinism = resolve_policy(config)
    print("determinism:", determinism["level"], "seed:", determinism["seed"])

    device = resolve_device()
    print("device:", device)
```

Delete the now-duplicated `config = load_experiment_config(config_path)` line that followed `print("device:", device)`.

Add the import beside the existing config import (currently line 28):

```python
from src.utils.determinism import resolve_policy
```

Pass the record at the `_build_run_manifest` call site (currently lines 581-586):

```python
                _build_run_manifest(
                    metadata=metadata,
                    config=config,
                    device=device,
                    resume_enabled=True,
                    determinism=determinism,
                ),
```

- [ ] **Step 6: Delete the per-fold cuDNN assignment**

Delete `scripts/train.py:680` entirely:

```python
        torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]
```

`resolve_policy` now sets it once at start-up. For a configuration without the new keys the legacy branch sets exactly the same value from exactly the same config key, so behaviour is unchanged; setting it once rather than once per fold has no effect on training. Leaving the line would also silently re-enable the autotuner if a future level ever permitted it.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_train_orchestration -v`
Expected: PASS

- [ ] **Step 8: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`. Any test asserting `schema_version == 1` must be updated to 2; that is the intended change. Any test calling `_build_run_manifest` must pass the new argument.

- [ ] **Step 9: Commit**

```bash
git add scripts/train.py tests/scripts/test_train_orchestration.py
git commit -m "feat: apply the determinism policy at training start-up and record it"
```

---

### Task 4: Apply the policy in both evaluation entry points

**Files:**
- Modify: `scripts/evaluate.py:93-96`
- Modify: `scripts/evaluate_postholdout_baseline.py:400`, and the `payload` construction at line 415
- Test: `tests/scripts/test_evaluate_orchestration.py` (extend)

**Interfaces:**
- Consumes: `resolve_policy(config: dict) -> dict[str, object]` from Task 1.
- Produces: an `integrity["determinism"]` entry in the development OOF payload.

Evaluation is where the Macro F1 that every decision rests on is computed. A nondeterministic evaluation would mean two evaluations of the same checkpoints could disagree, which would leave the phase's purpose half-served.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_evaluate_orchestration.py`:

```python
class EvaluationDeterminismTest(unittest.TestCase):
    def test_evaluation_module_applies_the_shared_policy(self):
        source = (
            Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
        ).read_text(encoding="utf-8")

        # The bare assignment seeds nothing and leaves the autotuner in
        # whatever state the config names.
        self.assertIn("resolve_policy(config)", source)
        self.assertNotIn(
            'torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]',
            source,
        )

    def test_postholdout_evaluation_records_the_policy(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "evaluate_postholdout_baseline.py"
        ).read_text(encoding="utf-8")

        self.assertIn("resolve_policy(config)", source)
        self.assertIn('"determinism": determinism', source)
        self.assertNotIn(
            'torch.backends.cudnn.benchmark = config["runtime"]["cudnn_benchmark"]',
            source,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_evaluate_orchestration -v`
Expected: FAIL — `resolve_policy(config)` is not present in either file.

- [ ] **Step 3: Update `scripts/evaluate.py`**

Replace lines 93-96:

```python
    config = load_experiment_config(config_path)
    determinism = resolve_policy(config)
    print("determinism:", determinism["level"], "seed:", determinism["seed"])

    device = resolve_device()
    print("device:", device)
```

Add beside the existing config import:

```python
from src.utils.determinism import resolve_policy
```

- [ ] **Step 4: Update `scripts/evaluate_postholdout_baseline.py`**

Replace line 400:

```python
    determinism = resolve_policy(config)
```

Add the same import beside the existing config import. Then add the record to the `integrity` entry of the `payload` built at line 415. The payload already carries an `integrity` mapping that `write_development_oof_artifacts` passes through unchanged; add one key to it:

```python
        "determinism": determinism,
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_evaluate_orchestration -v`
Expected: PASS

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate.py scripts/evaluate_postholdout_baseline.py tests/scripts/test_evaluate_orchestration.py
git commit -m "feat: apply the determinism policy in both evaluation entry points"
```

---

### Task 5: The two-run comparison script

**Files:**
- Create: `scripts/verify_determinism.py`
- Test: `tests/scripts/test_verify_determinism.py`

**Interfaces:**
- Consumes: `load_training_state(path, *, trusted_local, map_location, allow_completed)` from `src.engine.training_state`.
- Produces: `digest_tensor_mapping(mapping: dict) -> str`, `digest_history(histories: list) -> str`, `compare_runs(first_directory: Path, second_directory: Path) -> dict`, `main(argv: list[str] | None = None) -> int`. `compare_runs` returns keys `bit_exact` (bool), `model_digests` (list of two hex strings), `ema_digests` (list of two), `history_digests` (list of two), and `first_mismatch` (str or None).

This script runs no training and loads no model. It reads two completed output directories and reports whether they are identical.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_verify_determinism.py`:

```python
"""Digest and comparison behaviour of the determinism verification CLI."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_determinism.py"

# tests/scripts shadows scripts on the import path, so the CLI is loaded by
# file location rather than by module name.
_spec = importlib.util.spec_from_file_location("verify_determinism_cli", SCRIPT)
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


def _state(scale: float = 1.0) -> dict:
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "model_state_dict": {
            "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * scale,
            "layer.bias": torch.tensor([0.5, 0.25]) * scale,
        },
        "ema_state_dict": {"layer.weight": torch.tensor([[1.0, 2.0]]) * scale},
        "completed_fold_histories": [{"val_acc": [0.9 * scale, 0.95 * scale]}],
    }


def _write_run(directory: Path, scale: float = 1.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(_state(scale), directory / "training_state.pt")
    return directory


class DigestTest(unittest.TestCase):
    def test_digest_is_stable_across_insertion_order(self):
        first = {"b": torch.tensor([1.0]), "a": torch.tensor([2.0])}
        second = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0])}

        # Two identical runs may produce dicts built in a different order;
        # that must not read as a determinism failure.
        self.assertEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )

    def test_digest_detects_a_single_changed_element(self):
        first = {"a": torch.tensor([1.0, 2.0])}
        second = {"a": torch.tensor([1.0, 2.0000001])}

        self.assertNotEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )

    def test_digest_separates_shape_from_contents(self):
        first = {"a": torch.tensor([[1.0, 2.0]])}
        second = {"a": torch.tensor([[1.0], [2.0]])}

        # Same bytes, different shape. Hashing only the buffer would call
        # these equal.
        self.assertNotEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )

    def test_digest_distinguishes_keys_from_values(self):
        first = {"ab": torch.tensor([1.0]), "c": torch.tensor([2.0])}
        second = {"a": torch.tensor([1.0]), "bc": torch.tensor([2.0])}

        self.assertNotEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )


class CompareRunsTest(unittest.TestCase):
    def test_identical_runs_are_bit_exact(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = _write_run(Path(root) / "b")
            result = verify.compare_runs(first, second)

        self.assertTrue(result["bit_exact"])
        self.assertIsNone(result["first_mismatch"])

    def test_differing_runs_report_the_first_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a", scale=1.0)
            second = _write_run(Path(root) / "b", scale=1.5)
            result = verify.compare_runs(first, second)

        self.assertFalse(result["bit_exact"])
        self.assertIsNotNone(result["first_mismatch"])

    def test_history_difference_alone_breaks_bit_exactness(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = Path(root) / "b"
            second.mkdir()
            state = _state()
            state["completed_fold_histories"] = [{"val_acc": [0.9, 0.94]}]
            torch.save(state, second / "training_state.pt")
            result = verify.compare_runs(first, second)

        # Identical weights with different recorded metrics would still mean
        # the runs diverged.
        self.assertFalse(result["bit_exact"])
        self.assertEqual(result["first_mismatch"], "completed_fold_histories")

    def test_missing_state_file_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = Path(root) / "b"
            second.mkdir()
            with self.assertRaises(FileNotFoundError):
                verify.compare_runs(first, second)


class CliTest(unittest.TestCase):
    def test_main_writes_a_record_and_returns_zero_when_bit_exact(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = _write_run(Path(root) / "b")
            report = Path(root) / "record.json"
            code = verify.main(
                ["--first", str(first), "--second", str(second), "--output", str(report)]
            )

        self.assertEqual(code, 0)

    def test_main_returns_nonzero_when_not_bit_exact(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a", scale=1.0)
            second = _write_run(Path(root) / "b", scale=2.0)
            report = Path(root) / "record.json"
            code = verify.main(
                ["--first", str(first), "--second", str(second), "--output", str(report)]
            )
            payload = json.loads(report.read_text(encoding="utf-8"))

        # A verification tool that exits 0 on failure is a tool nobody checks.
        self.assertNotEqual(code, 0)
        self.assertFalse(payload["bit_exact"])

    def test_main_refuses_to_overwrite_an_existing_record(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = _write_run(Path(root) / "b")
            report = Path(root) / "record.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                verify.main(
                    [
                        "--first",
                        str(first),
                        "--second",
                        str(second),
                        "--output",
                        str(report),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_verify_determinism -v`
Expected: FAIL with `FileNotFoundError` for `scripts/verify_determinism.py`

- [ ] **Step 3: Create `scripts/verify_determinism.py`**

```python
"""Compare two completed training output directories for bit-exactness.

Phase 9.7 verification. This script trains nothing, constructs no model, and
touches no dataset. It reads the trusted local training state each run wrote
and reports whether the two runs are identical. The frozen adoption ladder is
in docs/postholdout-determinism-protocol.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


TRAINING_STATE_FILENAME = "training_state.pt"
_COMPARED_FIELDS = ("model_state_dict", "ema_state_dict", "completed_fold_histories")


def digest_tensor_mapping(mapping: dict) -> str:
    """Hash a state dict in sorted key order, including dtype and shape.

    Key length is hashed alongside the key so that {"ab": x, "c": y} cannot
    collide with {"a": x, "bc": y}, and shape is hashed separately from the
    buffer so that a reshaped tensor is not read as unchanged.
    """
    hasher = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        hasher.update(str(len(key)).encode("utf-8"))
        hasher.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            hasher.update(str(value.dtype).encode("utf-8"))
            hasher.update(str(tuple(value.shape)).encode("utf-8"))
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))
    return hasher.hexdigest()


def digest_history(histories) -> str:
    """Hash recorded per-fold metric histories with full float precision."""
    encoded = json.dumps(histories, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_state(directory: Path) -> dict:
    path = Path(directory) / TRAINING_STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No training state in {directory}: {path.name} is absent.")
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_runs(first_directory, second_directory) -> dict:
    """Compare two completed runs field by field, in a fixed order."""
    first = _load_state(Path(first_directory))
    second = _load_state(Path(second_directory))

    digests = {}
    first_mismatch = None
    for field in _COMPARED_FIELDS:
        if field == "completed_fold_histories":
            pair = [digest_history(first.get(field)), digest_history(second.get(field))]
        else:
            pair = [
                digest_tensor_mapping(first.get(field, {})),
                digest_tensor_mapping(second.get(field, {})),
            ]
        digests[field] = pair
        if first_mismatch is None and pair[0] != pair[1]:
            first_mismatch = field

    return {
        "bit_exact": first_mismatch is None,
        "first_directory": str(first_directory),
        "second_directory": str(second_directory),
        "model_digests": digests["model_state_dict"],
        "ema_digests": digests["ema_state_dict"],
        "history_digests": digests["completed_fold_histories"],
        "first_mismatch": first_mismatch,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two completed training runs for bit-exactness.",
    )
    parser.add_argument("--first", required=True, help="First run output directory.")
    parser.add_argument("--second", required=True, help="Second run output directory.")
    parser.add_argument("--output", required=True, help="Verification record JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Verification record already exists: {output_path}. "
            "A verification result is evidence and is not overwritten."
        )

    result = compare_runs(Path(args.first), Path(args.second))
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if result["bit_exact"]:
        print("BIT_EXACT: the two runs are identical.")
        return 0
    print(f"NOT_BIT_EXACT: first mismatching field is {result['first_mismatch']}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_verify_determinism -v`
Expected: PASS, 11 tests

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_determinism.py tests/scripts/test_verify_determinism.py
git commit -m "feat: add the two-run bit-exactness verification script"
```

---

### Task 6: Repository contract for the frozen protocol

**Files:**
- Test: `tests/repository/test_determinism_protocol_contract.py` (create)

**Interfaces:**
- Consumes: `configs/deep3_postholdout_determinism_check.toml` from Task 2; `docs/postholdout-determinism-protocol.md`.
- Produces: nothing consumed by later tasks.

This mirrors `tests/repository/test_noise_floor_protocol_contract.py`. Its job is to make a silent edit to a frozen document fail the suite.

- [ ] **Step 1: Write the contract test**

Create `tests/repository/test_determinism_protocol_contract.py`:

```python
"""Offline contract for the frozen Phase 9.7 determinism protocol."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-determinism-protocol.md"
CHECK_CONFIG = ROOT / "configs" / "deep3_postholdout_determinism_check.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"

FROZEN_CONFIGS = (
    "deep3_postholdout_baseline.toml",
    "deep3_postholdout_loss001.toml",
    "deep3_postholdout_baseline_rep002.toml",
    "deep3_postholdout_baseline_rep003.toml",
)


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class DeterminismProtocolContractTest(unittest.TestCase):
    def test_protocol_is_frozen(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "SEED:\n20260815",
            "VERIFICATION_RUN_COUNT:\n2",
            "APPROVED_SEED:\n20260815",
            "APPROVED_DETERMINISM_LADDER:\nYES",
            "APPROVED_VERIFICATION_RUN_COUNT:\n2",
        ):
            self.assertIn(token, document)

    def test_execution_status_is_exactly_one_known_state(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        states = [
            state
            for state in ("NOT_YET_RUN", "IN_PROGRESS", "COMPLETED", "STOPPED")
            if f"EXECUTION_STATUS:\n{state}" in document
        ]
        self.assertEqual(len(states), 1, f"expected exactly one execution status, got {states}")

    def test_every_ladder_branch_is_named(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A ladder with an unhandled branch returns discretion to whoever
        # reads the result first.
        for token in (
            "A_ADOPTED:",
            "A_DEGRADED:",
            "B_ADOPTED:",
            "B_DEGRADED:",
            "A_FAILED_OTHER:",
            "BLOCKED:",
            "LEVEL_ORDER:\nA_STRICT then B_CUDNN",
        ):
            self.assertIn(token, document)

    def test_seed_conditional_limitation_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Recorded before any result exists, so Phase 9.8 cannot replace one
        # overclaim with another.
        self.assertIn("It does not remove seed-to-seed variation", document)
        self.assertIn("common random numbers", document)

    def test_dataloader_generator_rejection_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("The DataLoader generator is deliberately not added", document)

    def test_check_config_changes_only_registered_fields(self):
        baseline = _flatten(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))
        check = _flatten(tomllib.loads(CHECK_CONFIG.read_text(encoding="utf-8")))

        differing = {
            key for key in set(baseline) | set(check) if baseline.get(key) != check.get(key)
        }
        self.assertEqual(
            differing,
            {
                "runtime.cudnn_benchmark",
                "runtime.seed",
                "runtime.determinism_level",
                "training.epochs",
                "fine_tuning.epochs",
                "post_holdout.experiment_id",
                "post_holdout.parent_experiment_id",
                "post_holdout.artifact_namespace",
            },
        )

    def test_check_config_never_uses_the_canonical_route(self):
        check = _flatten(tomllib.loads(CHECK_CONFIG.read_text(encoding="utf-8")))

        # The canonical route trains on the 4,298 locked-test examples.
        self.assertEqual(
            check["post_holdout.split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(
            check["post_holdout.cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )

    def test_frozen_configs_gained_no_determinism_keys(self):
        for name in FROZEN_CONFIGS:
            with self.subTest(name=name):
                config = tomllib.loads(
                    (ROOT / "configs" / name).read_text(encoding="utf-8")
                )
                # Adding required keys would have changed hashes recorded in
                # two frozen protocol documents.
                self.assertNotIn("seed", config["runtime"])
                self.assertNotIn("determinism_level", config["runtime"])

    def test_phase_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_LOSS001_RERUN:\nDEFERRED_TO_PHASE_9_8",
        ):
            self.assertIn(token, document)

        for forbidden in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "APPROVED_WEIGHT_PUBLICATION:\nYES",
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract test**

Run: `.venv/Scripts/python.exe -m unittest tests.repository.test_determinism_protocol_contract -v`
Expected: PASS, 9 tests. If a token assertion fails, the protocol document is the authority — fix the test's expected token to match the document, never the document to match the test.

- [ ] **Step 3: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add tests/repository/test_determinism_protocol_contract.py
git commit -m "test: pin the frozen phase 9.7 determinism protocol"
```

---

### Task 7: Top-level documentation corrections

**Files:**
- Modify: `README.md:5`, `README.md:164-178`
- Modify: `docs/reproducibility.md` (the "Remaining limitations" list, currently lines 119-127)
- Modify: `docs/post-holdout-research-plan.md:101`
- Modify: `docs/training.md:44`
- Modify: `docs/governance-decisions.md` (append a new section)

**Interfaces:**
- Consumes: nothing. This task is documentation only.
- Produces: nothing consumed by later tasks.

The correction scope was approved as top-level documents plus the governance ledger. Do **not** annotate individual result documents; the ledger entry carries the traceability.

The run-level documents were already accurate. `docs/training.md`, `docs/canonical-training-readiness.md`, `docs/canonical-holdout-evaluation.md`, and `docs/canonical-training-unblock.md` each state that bit-for-bit reproducibility is not claimed. Record that distinction; do not report a false claim that was not made.

- [ ] **Step 1: Qualify the README opening sentence**

`README.md:5` currently begins "A modular, reproducible PyTorch research pipeline for fresh/rotten fruit classification." Replace that opening clause with:

```markdown
A modular PyTorch research pipeline for fresh/rotten fruit classification, with a reproducible environment, dataset identity, and configuration. See [Reproducibility status](#reproducibility-status) for what training-run reproducibility does and does not cover.
```

- [ ] **Step 2: Add the missing row to the README status table**

Insert into the table at `README.md:164-176`, immediately before the `| Benchmark reproduction |` row:

```markdown
| Training-run reproducibility | Not verified before Phase 9.7; the pipeline set no random seed, so identical commands produced different weights and metrics |
```

Then extend the paragraph at line 178 with:

```markdown
Every result recorded before Phase 9.7 was produced by a pipeline that set no random seed. Three executions of one unchanged configuration gave development OOF Macro F1 of 0.901167, 0.912041, and 0.901858. The recorded metrics are accurate measurements of the runs that produced them, and any single one of them would land elsewhere on a rerun. See [the determinism protocol](docs/postholdout-determinism-protocol.md).
```

- [ ] **Step 3: Add the omitted limitation to `docs/reproducibility.md`**

This document's "Remaining limitations" list is where a reader looks for exactly this, and it was absent. Insert as the **first** bullet of that list:

```markdown
- **Training-run reproducibility:** not verified before Phase 9.7. The pipeline set no `torch`, NumPy, or Python seed, never called `torch.use_deterministic_algorithms`, and ran with `cudnn_benchmark = true`. The same command on the same commit produced different weights and different metrics. What this repository froze and verified is the environment, the dataset identity, and the configuration — not the training outcome. Phase 9.7 introduces explicit seeding; see [the determinism protocol](postholdout-determinism-protocol.md).
```

- [ ] **Step 4: Correct the unmet seed-recording requirement**

`docs/post-holdout-research-plan.md:101` requires every future run to record "repository/config SHA, experiment and parent IDs, dataset revision, split identity/hash, seeds, runtime/packages/GPU, duration, checkpoint hashes, result hashes, resource use, and advancement decision."

Append this sentence to that paragraph:

```markdown
The seed requirement went unmet from Phase 9.3 through Phase 9.6a: the pipeline set no seed, so runs in that range have no seed to record. Their manifests record the absence explicitly from Phase 9.7 onward, and the determinism protocol supplies the seed the requirement always assumed.
```

- [ ] **Step 5: Update the resume semantics paragraph**

`docs/training.md:44` currently ends "It does not add global initial seeding or change the configured cuDNN behavior." Replace that final sentence with:

```markdown
Before Phase 9.7 the mechanism added no global initial seeding and did not change the configured cuDNN behavior, so a run was reproducible only as a continuation of itself. Phase 9.7 adds start-up seeding through the optional `[runtime].seed` and `[runtime].determinism_level` keys. The resume mechanism itself is unchanged and needs no change: it already captures and restores the global Python, NumPy, and torch generator states, which is exactly what `DataLoader` shuffling draws from at `num_workers = 0`.
```

- [ ] **Step 6: Record the correction in the governance ledger**

Append to `docs/governance-decisions.md`:

```markdown
## Phase 9.7 — determinism introduced and the reproducibility record corrected, 2026-08-15

Phase 9.6a measured a run-to-run noise floor of `2s = 0.012177` against a loss-001 improvement of `0.0090` and returned `INCONCLUSIVE`. The cause was that the training pipeline set no random seed. Phase 9.7 introduces explicit seeding under the frozen protocol in [postholdout-determinism-protocol.md](postholdout-determinism-protocol.md), which fixes the seed at 20260815 and a six-branch adoption ladder before any verification run executes.

Every recorded result produced before this phase came from the unseeded pipeline: `deep3-canonical-reference-01`, `deep3-postholdout-research-01-baseline`, `deep3-postholdout-research-01-loss-001`, `deep3-postholdout-research-01-baseline-rep002`, and `deep3-postholdout-research-01-baseline-rep003`. Their recorded metrics are unchanged and remain accurate measurements of the runs that produced them. None is re-scored.

The documentation failure was one of placement, not of truthfulness. `training.md`, `canonical-training-readiness.md`, `canonical-holdout-evaluation.md`, and `canonical-training-unblock.md` each stated that bit-for-bit reproducibility was not claimed and that global seeding was not introduced. The top-level documents a reader reaches first — the README opening sentence, the README reproducibility status table, and the "Remaining limitations" list in `reproducibility.md` — carried no such qualification, and the research plan required every run to record a seed that did not exist. Those four locations are corrected here. The individual result documents are not annotated; this entry carries the traceability.

Two design findings are recorded because both could have been silently lost. The `DataLoader` generator proposed by the Phase 9.6a follow-up is rejected: at `num_workers = 0` the sampler draws from the global torch generator, which `training_state.pt` already persists, so a separate generator would have broken working epoch-boundary resume determinism. And determinism removes measurement noise but not seed-to-seed variation, so a deterministic baseline-candidate pair is a paired comparison under common random numbers that estimates the effect for one seed draw only. Phase 9.8 must argue its claim strength against that limit rather than inherit it unexamined.
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `OK`. `tests/repository/test_readme_contract.py` and `tests/reproducibility/test_reproducibility_contract.py` both read these files; if either fails, the assertion pins wording this task deliberately changed — update the assertion to the new wording.

- [ ] **Step 8: Commit**

```bash
git add README.md docs/reproducibility.md docs/post-holdout-research-plan.md docs/training.md docs/governance-decisions.md
git commit -m "docs: correct the top-level reproducibility record"
```

---

## After the plan

Execution of the two verification runs is **not** authorized by this plan. When the owner grants it, the sequence is:

```powershell
.venv/Scripts/python.exe -m scripts.train --config configs/deep3_postholdout_determinism_check.toml --output-dir weights/deep3-postholdout-determinism-check-a --save-training-state --require-empty-output-dir --run-id deep3-postholdout-determinism-check-a

.venv/Scripts/python.exe -m scripts.train --config configs/deep3_postholdout_determinism_check.toml --output-dir weights/deep3-postholdout-determinism-check-b --save-training-state --require-empty-output-dir --run-id deep3-postholdout-determinism-check-b

.venv/Scripts/python.exe -m scripts.verify_determinism --first weights/deep3-postholdout-determinism-check-a --second weights/deep3-postholdout-determinism-check-b --output results/determinism-check.json
```

If the first run raises a nondeterministic-operation error, record the operation named in the traceback, change `determinism_level` to `"B_CUDNN"` in the check configuration, delete both output directories, and start the pair again. That descent is the frozen ladder's `B_ADOPTED` path and requires no new decision. Any other failure is `A_FAILED_OTHER`: stop and report.

Do not change the seed, the epochs, the folds, the manifests, or the ladder during execution. They are frozen, and changing them after seeing a result voids the verification.
