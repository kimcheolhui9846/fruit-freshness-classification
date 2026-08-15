# Phase 9.8 Measurement Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen measurement floor executable, register the deterministic baseline configuration, and build the zero-GPU diagnostic that Phase 9.9 will be designed from.

**Architecture:** One pure module holds the measured constants and the two rules derived from them — classify a difference against the minimum detectable effect, and test the deterministic baseline against its validity envelope. A new configuration registers the deterministic baseline as its own lineage. A separate script reads the three already-recorded prediction files and describes how much the `freshpotato` error set moves between runs; it loads no model and consumes no GPU.

**Tech Stack:** Python 3.12, NumPy 2.5.1, `unittest` (this repository does not use pytest), TOML via `tomllib`.

## Global Constraints

- The frozen protocol is `docs/postholdout-measurement-floor-protocol.md`. Every value below is copied from it and may not be changed during implementation.
- `MDE_MACRO_F1 = 0.012177`, `MDE_TOP1 = 0.001969`, `MDE_FRESHPOTATO_F1 = 0.147833`.
- `VALIDITY_ENVELOPE = (0.892845, 0.917199)`.
- Seed is `20260815`; determinism level is `A_STRICT`.
- No configuration frozen before Phase 9.8 may change by a single byte: the canonical, baseline, loss-001, both replicates, and the determinism check. Their hashes are recorded in protocol documents and asserted by contract tests.
- **This plan executes no training run.** The deterministic baseline is authorized separately; `APPROVED_EXECUTION: NOT_YET_GRANTED`.
- The diagnostic is `EXPLORATORY_DESCRIPTIVE`. It may not advance a candidate or support a claim.
- Tests run with `.venv/Scripts/python.exe -m unittest`. The suite currently stands at **397 tests, 0 failures**.
- Commit messages carry no co-authorship trailer, under the repository's sole-authorship policy recorded in `docs/governance-decisions.md`.

## File Structure

| File | Responsibility |
|---|---|
| `src/utils/measurement_floor.py` (create) | The measured constants and the two rules derived from them. No I/O, no CLI. |
| `src/utils/config.py` (modify) | Registers the deterministic-baseline lineage. |
| `configs/deep3_postholdout_baseline_det.toml` (create) | The deterministic baseline configuration. |
| `scripts/diagnose_freshpotato_instability.py` (create) | Reads recorded prediction files; describes error-set movement. |
| `tests/utils/test_measurement_floor.py` (create) | Constants and rule behaviour. |
| `tests/config/test_deterministic_baseline_config.py` (create) | Lineage and permitted differences. |
| `tests/scripts/test_diagnose_freshpotato_instability.py` (create) | Set arithmetic and the CLI. |
| `tests/repository/test_measurement_floor_contract.py` (create) | Pins the frozen protocol. |

### Recorded facts the implementer needs

The development OOF prediction files are NumPy `.npz` archives with four arrays over 17,188 development examples: `labels` (int64), `predictions` (int64), `logits` (float32, 17188×14), and `fold_assignments` (int64). The frozen label order puts **`freshpotato` at index 5** and `rottenpotato` at index 12, across 14 classes.

Three of them already exist and are the diagnostic's only input:

```
results/deep3-postholdout-research-01-baseline/development_oof_predictions.npz
results/deep3-postholdout-research-01-baseline-rep002/development_oof_predictions.npz
results/deep3-postholdout-research-01-baseline-rep003/development_oof_predictions.npz
```

---

### Task 1: Measurement floor module

**Files:**
- Create: `src/utils/measurement_floor.py`
- Test: `tests/utils/test_measurement_floor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: constants `MDE_MACRO_F1`, `MDE_TOP1`, `MDE_FRESHPOTATO_F1`, `MACRO_F1_MEAN`, `MACRO_F1_STDEV`, `VALIDITY_ENVELOPE`, `REPLICATE_MACRO_F1`, `REPLICATE_TOP1`, and verdict names `ADVANCED`, `BELOW_RESOLUTION`, `REGRESSED`; functions `classify_effect(candidate: float, baseline: float, *, mde: float) -> dict` returning keys `verdict`, `difference`, `mde`, `candidate`, `baseline`, and `within_validity_envelope(macro_f1: float) -> bool`.

**Why three verdicts and not two.** A candidate that is *worse* than the baseline by more than the MDE is a detectable regression, not an unresolvable result. Collapsing it into `BELOW_RESOLUTION` would describe a measured decline as unmeasurable. The three verdicts partition the real line completely, with no gap for a reader to fill in.

- [ ] **Step 1: Write the failing tests**

Create `tests/utils/test_measurement_floor.py`:

```python
"""The frozen measurement floor and the rules derived from it."""

import statistics as st
import unittest

from src.utils.measurement_floor import (
    ADVANCED,
    BELOW_RESOLUTION,
    MACRO_F1_MEAN,
    MACRO_F1_STDEV,
    MDE_FRESHPOTATO_F1,
    MDE_MACRO_F1,
    MDE_TOP1,
    REGRESSED,
    REPLICATE_MACRO_F1,
    REPLICATE_TOP1,
    VALIDITY_ENVELOPE,
    classify_effect,
    within_validity_envelope,
)


class FrozenConstantTest(unittest.TestCase):
    def test_constants_match_the_frozen_protocol(self):
        # Pinned directly. A document can be edited without the suite
        # noticing unless the numbers are also asserted in code.
        self.assertEqual(MDE_MACRO_F1, 0.012177)
        self.assertEqual(MDE_TOP1, 0.001969)
        self.assertEqual(MDE_FRESHPOTATO_F1, 0.147833)
        self.assertEqual(VALIDITY_ENVELOPE, (0.892845, 0.917199))

    def test_constants_are_derived_from_the_recorded_replicates(self):
        # These are measured, not chosen. If someone edits a constant, the
        # arithmetic that produced it must still hold.
        self.assertEqual(len(REPLICATE_MACRO_F1), 3)
        self.assertAlmostEqual(st.mean(REPLICATE_MACRO_F1), MACRO_F1_MEAN, places=6)
        self.assertAlmostEqual(st.stdev(REPLICATE_MACRO_F1), MACRO_F1_STDEV, places=6)
        self.assertAlmostEqual(2 * MACRO_F1_STDEV, MDE_MACRO_F1, places=6)
        self.assertAlmostEqual(2 * st.stdev(REPLICATE_TOP1), MDE_TOP1, places=6)

    def test_validity_envelope_is_the_mean_plus_and_minus_the_mde(self):
        low, high = VALIDITY_ENVELOPE
        self.assertAlmostEqual(low, MACRO_F1_MEAN - MDE_MACRO_F1, places=6)
        self.assertAlmostEqual(high, MACRO_F1_MEAN + MDE_MACRO_F1, places=6)


class ClassifyEffectTest(unittest.TestCase):
    def test_improvement_at_or_above_the_mde_advances(self):
        result = classify_effect(0.0 + MDE_MACRO_F1, 0.0, mde=MDE_MACRO_F1)

        # Constructed so the difference is exactly the MDE, with no float
        # slack that would let the boundary pass under either comparison.
        self.assertEqual(result["difference"], MDE_MACRO_F1)
        self.assertEqual(result["verdict"], ADVANCED)

    def test_improvement_just_below_the_mde_is_below_resolution(self):
        result = classify_effect(0.9102, 0.9012, mde=MDE_MACRO_F1)

        # The recorded loss-001 comparison: +0.0090 against a floor of
        # 0.012177.
        self.assertEqual(result["verdict"], BELOW_RESOLUTION)

    def test_decline_at_or_beyond_the_mde_is_a_regression(self):
        result = classify_effect(0.0 - MDE_MACRO_F1, 0.0, mde=MDE_MACRO_F1)

        # A measured decline is not an unmeasurable result.
        self.assertEqual(result["verdict"], REGRESSED)

    def test_small_decline_is_below_resolution_not_a_regression(self):
        result = classify_effect(0.9000, 0.9012, mde=MDE_MACRO_F1)

        self.assertEqual(result["verdict"], BELOW_RESOLUTION)

    def test_no_difference_is_below_resolution(self):
        result = classify_effect(0.9012, 0.9012, mde=MDE_MACRO_F1)

        self.assertEqual(result["verdict"], BELOW_RESOLUTION)

    def test_every_verdict_is_reachable_and_they_are_distinct(self):
        verdicts = {
            classify_effect(1.0, 0.0, mde=MDE_MACRO_F1)["verdict"],
            classify_effect(0.0, 0.0, mde=MDE_MACRO_F1)["verdict"],
            classify_effect(0.0, 1.0, mde=MDE_MACRO_F1)["verdict"],
        }
        # A partition with an unreachable branch is not a partition.
        self.assertEqual(verdicts, {ADVANCED, BELOW_RESOLUTION, REGRESSED})

    def test_result_reports_its_own_inputs(self):
        result = classify_effect(0.92, 0.90, mde=MDE_MACRO_F1)

        # A verdict without its inputs cannot be audited later.
        self.assertEqual(set(result), {"verdict", "difference", "mde", "candidate", "baseline"})
        self.assertEqual(result["candidate"], 0.92)
        self.assertEqual(result["baseline"], 0.90)
        self.assertEqual(result["mde"], MDE_MACRO_F1)

    def test_non_positive_mde_is_rejected(self):
        for bad in (0.0, -0.01):
            with self.subTest(mde=bad):
                # A zero floor would advance every non-negative difference.
                with self.assertRaises(ValueError):
                    classify_effect(0.92, 0.90, mde=bad)


class ValidityEnvelopeTest(unittest.TestCase):
    def test_every_recorded_replicate_falls_inside(self):
        for value in REPLICATE_MACRO_F1:
            with self.subTest(value=value):
                # The envelope is built from these; if one fell outside, the
                # envelope would be describing something else.
                self.assertTrue(within_validity_envelope(value))

    def test_both_boundaries_are_inclusive(self):
        low, high = VALIDITY_ENVELOPE
        self.assertTrue(within_validity_envelope(low))
        self.assertTrue(within_validity_envelope(high))

    def test_values_outside_are_rejected(self):
        low, high = VALIDITY_ENVELOPE
        self.assertFalse(within_validity_envelope(low - 0.001))
        self.assertFalse(within_validity_envelope(high + 0.001))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.utils.test_measurement_floor -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.measurement_floor'`

- [ ] **Step 3: Create `src/utils/measurement_floor.py`**

```python
"""The frozen measurement floor for post-holdout development comparisons.

Every constant here is measured rather than chosen. They come from the three
Phase 9.6a replicates of the identical baseline recipe on the identical
frozen folds, which are the only direct evidence this project has about how
much its own measurements move between runs.

Seeding does not lower these numbers. Fixing a seed pins one draw from the
same distribution; it does not narrow the distribution. The replicates
therefore remain the right estimate of how far a result would move at a
different seed, even though they predate seeding.

The protocol is docs/postholdout-measurement-floor-protocol.md.
"""

from __future__ import annotations


REPLICATE_MACRO_F1 = (0.901167, 0.912041, 0.901858)
REPLICATE_TOP1 = (0.956598, 0.957936, 0.956016)

MACRO_F1_MEAN = 0.905022
MACRO_F1_STDEV = 0.006089

MDE_MACRO_F1 = 0.012177
MDE_TOP1 = 0.001969
MDE_FRESHPOTATO_F1 = 0.147833

VALIDITY_ENVELOPE = (0.892845, 0.917199)

ADVANCED = "ADVANCED"
BELOW_RESOLUTION = "BELOW_RESOLUTION"
REGRESSED = "REGRESSED"


def classify_effect(candidate: float, baseline: float, *, mde: float) -> dict:
    """Classify a single-run difference against a frozen minimum detectable effect.

    Three verdicts, because two would not cover the line. An improvement at
    or above the floor advances; a decline at or beyond it is a measured
    regression; anything between is a result this project cannot separate
    from the seed it happened to draw.
    """
    if mde <= 0:
        raise ValueError("mde must be positive; a zero floor advances every gain.")

    difference = candidate - baseline
    if difference >= mde:
        verdict = ADVANCED
    elif difference <= -mde:
        verdict = REGRESSED
    else:
        verdict = BELOW_RESOLUTION

    return {
        "verdict": verdict,
        "difference": difference,
        "mde": mde,
        "candidate": candidate,
        "baseline": baseline,
    }


def within_validity_envelope(macro_f1: float) -> bool:
    """Test a deterministic baseline against the pre-registered envelope.

    Falling inside is consistent with A_STRICT's deterministic kernel
    selection not having changed results materially. It is not proof: an
    envelope built from three samples is wide, and a real shift smaller than
    the envelope would pass unnoticed.
    """
    low, high = VALIDITY_ENVELOPE
    return low <= macro_f1 <= high
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.utils.test_measurement_floor -v`
Expected: PASS, 14 tests

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 411 tests`, `OK`

- [ ] **Step 6: Commit**

```bash
git add src/utils/measurement_floor.py tests/utils/test_measurement_floor.py
git commit -m "feat: add the frozen measurement floor and its decision rule"
```

---

### Task 2: Deterministic baseline configuration and lineage

**Files:**
- Create: `configs/deep3_postholdout_baseline_det.toml`
- Modify: `src/utils/config.py` (constants beside `DETERMINISM_CHECK_EXPECTED_VALUES`; a branch in `resolve_experiment_validation` before its final `raise`)
- Test: `tests/config/test_deterministic_baseline_config.py`

**Interfaces:**
- Consumes: the existing `validate_loss_experiment_config(baseline_path, experiment_path, *, allowed_differences: frozenset[str], expected_values: dict[str, object]) -> dict` and `resolve_experiment_validation(config: dict, config_path: str | Path) -> dict | None`, plus `BASELINE_CONFIG_PATH`.
- Produces: `DETERMINISTIC_BASELINE_PARENT_ID`, `DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES`, `DETERMINISTIC_BASELINE_EXPECTED_VALUES` in `src.utils.config`; the file `configs/deep3_postholdout_baseline_det.toml`.

Unlike the Phase 9.7 check configuration, the level **is** pinned here. That ladder is finished and `A_STRICT` is the adopted level; there is no descent left to keep available.

- [ ] **Step 1: Write the failing tests**

Create `tests/config/test_deterministic_baseline_config.py`:

```python
"""The deterministic baseline configuration and its registered lineage."""

from pathlib import Path
import tomllib
import unittest

from src.utils.config import (
    DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES,
    DETERMINISTIC_BASELINE_EXPECTED_VALUES,
    load_experiment_config,
    resolve_experiment_validation,
)


ROOT = Path(__file__).resolve().parents[2]
DET_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline_det.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"

EXPECTED_DIFFERENCES = {
    "runtime.cudnn_benchmark",
    "runtime.seed",
    "runtime.determinism_level",
    "post_holdout.experiment_id",
    "post_holdout.parent_experiment_id",
    "post_holdout.artifact_namespace",
}


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class DeterministicBaselineConfigTest(unittest.TestCase):
    def test_recipe_differs_only_in_determinism_and_identity(self):
        baseline = _flatten(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))
        det = _flatten(tomllib.loads(DET_CONFIG.read_text(encoding="utf-8")))

        differing = {
            key for key in set(baseline) | set(det) if baseline.get(key) != det.get(key)
        }
        # A deterministic baseline that also changed the recipe would not be
        # a baseline for the recipe.
        self.assertEqual(differing, EXPECTED_DIFFERENCES)

    def test_full_length_schedule_is_preserved(self):
        det = _flatten(tomllib.loads(DET_CONFIG.read_text(encoding="utf-8")))

        # The Phase 9.7 check config was bounded at 2 epochs. This one is not
        # a smoke test; it must run the real schedule.
        self.assertEqual(det["training.epochs"], 120)
        self.assertEqual(det["fine_tuning.epochs"], 20)
        self.assertEqual(det["training.batch_size"], 64)

    def test_frozen_folds_are_reused(self):
        det = _flatten(tomllib.loads(DET_CONFIG.read_text(encoding="utf-8")))

        self.assertEqual(
            det["post_holdout.split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(
            det["post_holdout.cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )
        self.assertEqual(det["cross_validation.random_state"], 42)
        self.assertEqual(det["cross_validation.n_splits"], 3)

    def test_adopted_determinism_settings_are_carried(self):
        config = load_experiment_config(DET_CONFIG)

        self.assertEqual(config["runtime"]["seed"], 20260815)
        self.assertEqual(config["runtime"]["determinism_level"], "A_STRICT")
        self.assertFalse(config["runtime"]["cudnn_benchmark"])

    def test_lineage_validates_and_names_its_differences(self):
        config = load_experiment_config(DET_CONFIG)
        result = resolve_experiment_validation(config, DET_CONFIG)

        self.assertIsNotNone(result)
        self.assertTrue(result["single_factor_verified"])
        self.assertEqual(set(result["differences"]), EXPECTED_DIFFERENCES)

    def test_lineage_pins_the_adopted_level(self):
        # The Phase 9.7 ladder is finished and A_STRICT is adopted, so unlike
        # the check config there is no descent left to keep available.
        self.assertEqual(
            DETERMINISTIC_BASELINE_EXPECTED_VALUES["runtime.determinism_level"],
            "A_STRICT",
        )
        self.assertIn("runtime.determinism_level", DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES)

    def test_unregistered_parent_still_raises(self):
        config = load_experiment_config(DET_CONFIG)
        config["post_holdout"]["parent_experiment_id"] = "deep3-unregistered"

        with self.assertRaises(ValueError):
            resolve_experiment_validation(config, DET_CONFIG)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.config.test_deterministic_baseline_config -v`
Expected: FAIL — `ImportError` for the new constants, and the configuration file does not exist.

- [ ] **Step 3: Register the lineage in `src/utils/config.py`**

Add immediately after `DETERMINISM_CHECK_EXPECTED_VALUES`:

```python
DETERMINISTIC_BASELINE_PARENT_ID = "deep3-postholdout-deterministic-baseline"
DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES = frozenset(
    {
        "runtime.cudnn_benchmark",
        "runtime.seed",
        "runtime.determinism_level",
        "post_holdout.experiment_id",
        "post_holdout.parent_experiment_id",
        "post_holdout.artifact_namespace",
    }
)
# The level is pinned here, unlike the Phase 9.7 check lineage. That ladder
# is finished and A_STRICT is the adopted level, so there is no registered
# descent left to keep available.
DETERMINISTIC_BASELINE_EXPECTED_VALUES = {
    "runtime.seed": 20260815,
    "runtime.determinism_level": "A_STRICT",
    "runtime.cudnn_benchmark": False,
    "training.epochs": 120,
    "fine_tuning.epochs": 20,
    "post_holdout.split_manifest_path": (
        "configs/splits/deep3-postholdout-research-01.json"
    ),
    "post_holdout.cv_manifest_path": (
        "configs/splits/deep3-postholdout-research-01-baseline-cv.json"
    ),
}
```

Add this branch in `resolve_experiment_validation`, immediately before its final `raise ValueError`:

```python
    if parent == DETERMINISTIC_BASELINE_PARENT_ID:
        return validate_loss_experiment_config(
            BASELINE_CONFIG_PATH,
            config_path,
            allowed_differences=DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES,
            expected_values=DETERMINISTIC_BASELINE_EXPECTED_VALUES,
        )
```

- [ ] **Step 4: Create `configs/deep3_postholdout_baseline_det.toml`**

Every value not listed in `DETERMINISTIC_BASELINE_ALLOWED_DIFFERENCES` is copied verbatim from `configs/deep3_postholdout_baseline.toml`.

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
epochs = 120
batch_size = 64

[fine_tuning]
epochs = 20

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
experiment_id = "deep3-postholdout-research-01-baseline-det"
parent_experiment_id = "deep3-postholdout-deterministic-baseline"
split_manifest_path = "configs/splits/deep3-postholdout-research-01.json"
cv_manifest_path = "configs/splits/deep3-postholdout-research-01-baseline-cv.json"
artifact_namespace = "deep3-postholdout-research-01-baseline-det"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.config.test_deterministic_baseline_config -v`
Expected: PASS, 7 tests

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 418 tests`, `OK`. If a frozen-config test fails, a frozen file was edited — revert it; never adjust the frozen file to satisfy a new test.

- [ ] **Step 7: Commit**

```bash
git add src/utils/config.py configs/deep3_postholdout_baseline_det.toml tests/config/test_deterministic_baseline_config.py
git commit -m "feat: register the deterministic baseline configuration and lineage"
```

---

### Task 3: Exploratory instability diagnostic

**Files:**
- Create: `scripts/diagnose_freshpotato_instability.py`
- Test: `tests/scripts/test_diagnose_freshpotato_instability.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `FRESHPOTATO_INDEX = 5`; `error_positions(labels, predictions, class_index) -> numpy.ndarray`; `compare_error_sets(runs: dict[str, tuple], class_index: int = FRESHPOTATO_INDEX) -> dict` where each value is a `(labels, predictions)` pair of arrays, returning keys `class_index`, `class_support`, `run_names`, `per_run_error_counts`, `stable_error_count`, `union_error_count`, `jaccard`, and `error_frequency_histogram`; `main(argv: list[str] | None = None) -> int`.

This script loads no model, touches no dataset, and consumes no GPU. It reads prediction files that already exist.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_diagnose_freshpotato_instability.py`:

```python
"""Set arithmetic and CLI behaviour of the instability diagnostic."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnose_freshpotato_instability.py"

# tests/scripts shadows scripts on the import path, so the CLI is loaded by
# file location rather than by module name.
_spec = importlib.util.spec_from_file_location("diagnose_instability_cli", SCRIPT)
diagnose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnose)


LABELS = np.array([5, 5, 5, 5, 12, 12], dtype=np.int64)


def _write_run(root: Path, run_name: str, predictions) -> Path:
    """Mirror the real layout: results/<run-id>/development_oof_predictions.npz.

    The CLI derives each run's name from its parent directory, so fixtures
    written side by side in one directory would all collide on the same name.
    """
    directory = root / run_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "development_oof_predictions.npz"
    np.savez(
        path,
        labels=LABELS,
        predictions=np.asarray(predictions, dtype=np.int64),
        logits=np.zeros((LABELS.size, 14), dtype=np.float32),
        fold_assignments=np.ones(LABELS.size, dtype=np.int64),
    )
    return path


class ErrorPositionTest(unittest.TestCase):
    def test_only_the_named_class_counts_as_an_error(self):
        predictions = np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)
        positions = diagnose.error_positions(LABELS, predictions, 5)

        # Position 4 is a rottenpotato predicted freshpotato. That is an
        # error, but not an error *of* freshpotato.
        np.testing.assert_array_equal(positions, np.array([1, 3]))

    def test_a_perfect_class_has_no_error_positions(self):
        predictions = np.array([5, 5, 5, 5, 12, 12], dtype=np.int64)
        positions = diagnose.error_positions(LABELS, predictions, 5)

        self.assertEqual(positions.size, 0)


class CompareErrorSetsTest(unittest.TestCase):
    def test_identical_runs_are_fully_stable(self):
        runs = {
            "a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
            "b": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        self.assertEqual(result["stable_error_count"], 2)
        self.assertEqual(result["union_error_count"], 2)
        self.assertEqual(result["jaccard"], 1.0)

    def test_disjoint_error_sets_have_zero_overlap(self):
        runs = {
            "a": (LABELS, np.array([12, 5, 5, 5, 12, 12], dtype=np.int64)),
            "b": (LABELS, np.array([5, 12, 5, 5, 12, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        # Same error count, no shared example: the boundary moved.
        self.assertEqual(result["per_run_error_counts"], {"a": 1, "b": 1})
        self.assertEqual(result["stable_error_count"], 0)
        self.assertEqual(result["union_error_count"], 2)
        self.assertEqual(result["jaccard"], 0.0)

    def test_frequency_histogram_counts_examples_by_how_often_they_fail(self):
        runs = {
            "a": (LABELS, np.array([12, 12, 5, 5, 12, 12], dtype=np.int64)),
            "b": (LABELS, np.array([12, 5, 5, 5, 12, 12], dtype=np.int64)),
            "c": (LABELS, np.array([12, 5, 12, 5, 12, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        # Position 0 fails in all three, position 1 in one, position 2 in one.
        self.assertEqual(result["error_frequency_histogram"], {"1": 2, "2": 0, "3": 1})

    def test_no_errors_anywhere_gives_a_defined_jaccard(self):
        perfect = np.array([5, 5, 5, 5, 12, 12], dtype=np.int64)
        runs = {"a": (LABELS, perfect), "b": (LABELS, perfect)}
        result = diagnose.compare_error_sets(runs)

        # An empty union must not divide by zero.
        self.assertEqual(result["union_error_count"], 0)
        self.assertEqual(result["jaccard"], 1.0)

    def test_disagreeing_label_arrays_are_refused(self):
        other = np.array([5, 5, 5, 12, 12, 12], dtype=np.int64)
        runs = {
            "a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
            "b": (other, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
        }
        # Runs that disagree about which examples belong to the class are not
        # comparable, and averaging over them would be meaningless.
        with self.assertRaises(ValueError):
            diagnose.compare_error_sets(runs)

    def test_fewer_than_two_runs_is_refused(self):
        runs = {"a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64))}
        with self.assertRaises(ValueError):
            diagnose.compare_error_sets(runs)

    def test_class_support_is_reported(self):
        runs = {
            "a": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
            "b": (LABELS, np.array([5, 12, 5, 12, 5, 12], dtype=np.int64)),
        }
        result = diagnose.compare_error_sets(runs)

        self.assertEqual(result["class_support"], 4)
        self.assertEqual(result["class_index"], 5)


class CliTest(unittest.TestCase):
    def test_main_writes_a_record_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            second = _write_run(Path(root), "b", [12, 12, 5, 5, 5, 12])
            output = Path(root) / "record.json"
            code = diagnose.main(
                ["--predictions", str(first), str(second), "--output", str(output)]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["class_index"], 5)
        self.assertEqual(payload["status"], "EXPLORATORY_DESCRIPTIVE")

    def test_main_refuses_a_single_predictions_file(self):
        with tempfile.TemporaryDirectory() as root:
            only = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            output = Path(root) / "record.json"
            with self.assertRaises(ValueError):
                diagnose.main(["--predictions", str(only), "--output", str(output)])

    def test_main_refuses_to_overwrite_an_existing_record(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            second = _write_run(Path(root), "b", [12, 12, 5, 5, 5, 12])
            output = Path(root) / "record.json"
            output.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                diagnose.main(
                    ["--predictions", str(first), str(second), "--output", str(output)]
                )

    def test_record_carries_its_exploratory_status(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root), "a", [5, 12, 5, 12, 5, 12])
            second = _write_run(Path(root), "b", [12, 12, 5, 5, 5, 12])
            output = Path(root) / "record.json"
            diagnose.main(
                ["--predictions", str(first), str(second), "--output", str(output)]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        # A descriptive record that does not say it is descriptive will be
        # read as a result.
        self.assertEqual(payload["status"], "EXPLORATORY_DESCRIPTIVE")
        self.assertFalse(payload["may_advance_a_candidate"])
        self.assertFalse(payload["may_support_a_claim"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_diagnose_freshpotato_instability -v`
Expected: FAIL with `FileNotFoundError` for `scripts/diagnose_freshpotato_instability.py`

- [ ] **Step 3: Create `scripts/diagnose_freshpotato_instability.py`**

```python
"""Describe how much the freshpotato error set moves between identical runs.

Phase 9.8, exploratory and descriptive. The variance decomposition in
docs/postholdout-measurement-floor-protocol.md shows that this one class
accounts for 90.56% of Macro F1's run-to-run variance. This script asks the
next question: are the images it gets wrong the same images each run, or
different ones?

It loads no model, touches no dataset, and consumes no GPU. It reads
development OOF prediction files that already exist. No rule about what its
output means is frozen anywhere, because none was chosen before looking, so
it can support no claim and advance no candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


FRESHPOTATO_INDEX = 5


def error_positions(labels, predictions, class_index: int) -> np.ndarray:
    """Development positions where this class's examples were misclassified.

    Examples of *other* classes predicted as this one are excluded: they are
    errors, but not errors of this class, and counting them would conflate
    recall failures with precision failures.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")
    return np.flatnonzero((labels == class_index) & (predictions != class_index))


def compare_error_sets(runs: dict, class_index: int = FRESHPOTATO_INDEX) -> dict:
    """Describe how far the error set moves across runs of the same recipe."""
    if len(runs) < 2:
        raise ValueError("Comparing error sets needs at least two runs.")

    names = list(runs)
    reference_labels = np.asarray(runs[names[0]][0])
    for name in names[1:]:
        if not np.array_equal(np.asarray(runs[name][0]), reference_labels):
            raise ValueError(
                f"Run {name!r} has different labels from {names[0]!r}; "
                "runs that disagree about which examples belong to the class "
                "are not comparable."
            )

    sets = {
        name: set(error_positions(labels, predictions, class_index).tolist())
        for name, (labels, predictions) in runs.items()
    }
    union = set().union(*sets.values())
    intersection = set(sets[names[0]])
    for name in names[1:]:
        intersection &= sets[name]

    counts = {position: 0 for position in union}
    for positions in sets.values():
        for position in positions:
            counts[position] += 1
    histogram = {
        str(k): sum(1 for value in counts.values() if value == k)
        for k in range(1, len(names) + 1)
    }

    return {
        "class_index": class_index,
        "class_support": int(np.count_nonzero(reference_labels == class_index)),
        "run_names": names,
        "per_run_error_counts": {name: len(positions) for name, positions in sets.items()},
        "stable_error_count": len(intersection),
        "union_error_count": len(union),
        # An empty union means no run erred at all, which is total agreement.
        "jaccard": 1.0 if not union else len(intersection) / len(union),
        "error_frequency_histogram": histogram,
    }


def load_run(path: Path) -> tuple:
    archive = np.load(Path(path))
    return archive["labels"], archive["predictions"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe error-set movement for one class across runs.",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Two or more development OOF prediction .npz files.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=FRESHPOTATO_INDEX,
        help="Frozen label index; freshpotato is 5.",
    )
    parser.add_argument("--output", required=True, help="Descriptive record JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Record already exists: {output_path}.")

    runs = {Path(path).parent.name or Path(path).stem: load_run(Path(path)) for path in args.predictions}
    if len(runs) != len(args.predictions):
        raise ValueError("Prediction files must have distinguishable run names.")

    result = compare_error_sets(runs, class_index=args.class_index)
    result.update(
        {
            "status": "EXPLORATORY_DESCRIPTIVE",
            "may_advance_a_candidate": False,
            "may_support_a_claim": False,
        }
    )
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"class {result['class_index']}: support {result['class_support']}, "
        f"errors {result['per_run_error_counts']}, "
        f"stable {result['stable_error_count']}, union {result['union_error_count']}, "
        f"jaccard {result['jaccard']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_diagnose_freshpotato_instability -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Run the diagnostic on the three recorded replicates**

This consumes no GPU and reads only files that already exist.

```bash
.venv/Scripts/python.exe -m scripts.diagnose_freshpotato_instability \
  --predictions \
    results/deep3-postholdout-research-01-baseline/development_oof_predictions.npz \
    results/deep3-postholdout-research-01-baseline-rep002/development_oof_predictions.npz \
    results/deep3-postholdout-research-01-baseline-rep003/development_oof_predictions.npz \
  --output results/freshpotato-instability.json
```

Expected: exit 0, and `class_support` is **347**, matching the frozen subject count from Phase 9.5. If it is not 347, stop: the prediction files do not describe the development pool this protocol assumes.

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 431 tests`, `OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/diagnose_freshpotato_instability.py tests/scripts/test_diagnose_freshpotato_instability.py
git commit -m "feat: add the exploratory freshpotato instability diagnostic"
```

---

### Task 4: Repository contract for the frozen protocol

**Files:**
- Create: `tests/repository/test_measurement_floor_contract.py`

**Interfaces:**
- Consumes: `src.utils.measurement_floor` constants from Task 1; `configs/deep3_postholdout_baseline_det.toml` from Task 2; `docs/postholdout-measurement-floor-protocol.md`.
- Produces: nothing consumed by later tasks.

This mirrors `tests/repository/test_determinism_protocol_contract.py`. Its job is to make a silent edit to a frozen document fail the suite, and to catch the document and the code drifting apart.

- [ ] **Step 1: Write the contract test**

Create `tests/repository/test_measurement_floor_contract.py`:

```python
"""Offline contract for the frozen Phase 9.8 measurement floor protocol."""

from pathlib import Path
import unittest

from src.utils.measurement_floor import (
    MDE_FRESHPOTATO_F1,
    MDE_MACRO_F1,
    MDE_TOP1,
    VALIDITY_ENVELOPE,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-measurement-floor-protocol.md"


class MeasurementFloorContractTest(unittest.TestCase):
    def test_protocol_is_frozen(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "SEED:\n20260815",
            "DETERMINISM_LEVEL:\nA_STRICT",
            "TRAINING_RUN_COUNT:\n1",
            "APPROVED_TRAINING_RUN_COUNT:\n1",
            "APPROVED_MDE_FRAMEWORK:\nYES",
        ):
            self.assertIn(token, document)

    def test_execution_status_is_exactly_one_known_state(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        states = [
            state
            for state in ("NOT_YET_RUN", "IN_PROGRESS", "COMPLETED", "STOPPED")
            if f"EXECUTION_STATUS:\n{state}" in document
        ]
        self.assertEqual(
            len(states), 1, f"expected exactly one execution status, got {states}"
        )

    def test_code_constants_match_the_document(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A number pinned in prose and a number used in code can drift apart
        # silently. Assert they are the same number.
        self.assertIn(f"MDE_MACRO_F1:\n{MDE_MACRO_F1}", document)
        self.assertIn(f"MDE_TOP1:\n{MDE_TOP1}", document)
        self.assertIn(f"MDE_FRESHPOTATO_F1:\n{MDE_FRESHPOTATO_F1}", document)
        low, high = VALIDITY_ENVELOPE
        self.assertIn(f"VALIDITY_ENVELOPE:\n{low} to {high}", document)

    def test_h1_closure_and_its_basis_are_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "H1_STATUS:\nCLOSED_BELOW_RESOLUTION",
            "H1_CLOSURE_BASIS:\n71 to 212 GPU hours required to resolve the observed effect",
            "LOSS001_VERDICT:\nNOT_ADVANCED, unchanged and not re-scored",
        ):
            self.assertIn(token, document)

        # "Exhausted" is the claim the evidence does not support and the one
        # most likely to creep back in.
        self.assertNotIn("H1_STATUS:\nEXHAUSTED", document)

    def test_variance_decomposition_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The finding that reorganized the phase.
        self.assertIn("90.56%", document)
        self.assertIn("the class the research was trying to improve", document)

    def test_seeding_does_not_lower_the_floor_is_stated(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn(
            "Seeding makes a run reproducible. It does not make the outcome "
            "less variable across seeds.",
            document,
        )

    def test_run_duration_basis_is_measured_not_assumed(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The cost table drives the H1 closure, so its per-run figure must
        # name the run it came from.
        self.assertIn("530.98 minutes", document)
        self.assertIn("8.85", document)

    def test_diagnostic_cannot_be_promoted_to_a_result(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "DIAGNOSTIC_STATUS:\nEXPLORATORY_DESCRIPTIVE",
            "DIAGNOSTIC_MAY_ADVANCE_A_CANDIDATE:\nNO",
            "DIAGNOSTIC_MAY_SUPPORT_A_CLAIM:\nNO",
        ):
            self.assertIn(token, document)

    def test_phase_9_9_is_registered_but_not_authorized(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("PHASE_9_9_STATUS:\nREGISTERED_NOT_DESIGNED", document)
        self.assertIn("PHASE_9_9_AUTHORIZED:\nNO", document)
        self.assertNotIn("PHASE_9_9_AUTHORIZED:\nYES", document)

    def test_phase_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_LOSS001_RERUN:\nNO",
        ):
            self.assertIn(token, document)

        for forbidden in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "APPROVED_WEIGHT_PUBLICATION:\nYES",
            "APPROVED_LOSS001_RERUN:\nYES",
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the contract test**

Run: `.venv/Scripts/python.exe -m unittest tests.repository.test_measurement_floor_contract -v`
Expected: PASS, 10 tests. If a token assertion fails, the protocol document is the authority — correct the test's expected token to match the document, never the document to match the test.

- [ ] **Step 3: Prove the contract can fail**

A contract test that passes on its first run has not shown it detects anything. Temporarily change `H1_STATUS:\nCLOSED_BELOW_RESOLUTION` to `H1_STATUS:\nEXHAUSTED` in a scratch copy of the protocol, confirm the suite fails, then restore the file and confirm `git diff` is empty.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 441 tests`, `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/repository/test_measurement_floor_contract.py
git commit -m "test: pin the frozen phase 9.8 measurement floor protocol"
```

---

### Task 5: Registry and governance ledger entries

**Files:**
- Modify: `docs/experiment-registry.md` (the table near the top, and a new section appended)
- Modify: `docs/governance-decisions.md` (a new section appended)

**Interfaces:**
- Consumes: nothing. Task 3's diagnostic record is referenced by name, not quoted, so no number needs copying between tasks.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add the registry table row**

In `docs/experiment-registry.md`, add immediately after the `deep3-postholdout-determinism-check-01` row:

```markdown
| `deep3-postholdout-research-01-baseline-det` | 9.8 | `REGISTERED_NOT_YET_RUN` | `deep3-postholdout-deterministic-baseline` | Development CV only; locked test and canonical holdout are model-inaccessible |
```

- [ ] **Step 2: Append the registry section**

Append to `docs/experiment-registry.md`:

```markdown
## Phase 9.8 measurement floor

The baseline is re-established under the pipeline Phase 9.7 adopted, because the recorded 0.901167 came from the unseeded pipeline and is not a valid comparison basis for a deterministic run. The frozen protocol is [postholdout-measurement-floor-protocol.md](postholdout-measurement-floor-protocol.md).

```text
DETERMINISTIC_BASELINE_ID:
deep3-postholdout-research-01-baseline-det
SEED:
20260815
DETERMINISM_LEVEL:
A_STRICT
MDE_MACRO_F1:
0.012177
VALIDITY_ENVELOPE:
0.892845 to 0.917199
H1_STATUS:
CLOSED_BELOW_RESOLUTION
LOCKED_TEST_MODEL_ACCESS:
NO
PHASE_9_9:
FRESHPOTATO_STABILITY
```

The measurement floor is derived from the three Phase 9.6a replicates and binds regardless of determinism: fixing a seed pins one draw from the same distribution rather than narrowing it. `freshpotato` alone accounts for 90.56% of Macro F1's run-to-run variance, so the instrument's noise is the class the research was trying to improve.
```

- [ ] **Step 3: Append the governance ledger entry**

Append to `docs/governance-decisions.md`:

```markdown
## Phase 9.8 — measurement floor frozen and H1 closed below resolution, 2026-08-16

Phase 9.7 made the pipeline bit-exact, and Phase 9.8 records what that does and does not buy. Determinism removes measurement noise; it does not narrow the distribution a seed is drawn from. The three Phase 9.6a replicates therefore remain the right estimate of how far a result would move at a different seed, and they set a minimum detectable effect of 0.012177 on Macro F1, 0.001969 on Top-1, and 0.147833 on `freshpotato` F1.

Decomposing that variance produced the finding that reorganized the phase: `freshpotato` alone accounts for 90.56% of Macro F1's run-to-run variance, against 5.43% for `rottenpotato` and 4.01% for the remaining twelve classes. The instrument's noise is the class the research was trying to improve, which is why the Phase 9.6 acceptance margin of 0.010 — set below the 0.012177 floor measured afterwards — could not have separated signal from noise.

H1 is recorded `CLOSED_BELOW_RESOLUTION`. Resolving the observed effect needs between 71 and 212 GPU hours under a conservative zero-correlation bound on the paired difference, using the measured 8.85-hour run duration. That is neither "H1 is exhausted", which the evidence does not support, nor "inconclusive, keep trying", which the arithmetic prices out of reach. The loss-001 verdict of `NOT_ADVANCED` is unchanged and is not re-scored.

The research question moves from raising `freshpotato` F1 to asking why it moves by roughly 0.15 between runs of an identical configuration. A zero-GPU diagnostic over the three recorded prediction files describes whether the misclassified images are the same each run; it is `EXPLORATORY_DESCRIPTIVE` and can advance nothing. Phase 9.9 is registered and designed nowhere.
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`
Expected: `Ran 441 tests`, `OK`

- [ ] **Step 5: Commit**

```bash
git add docs/experiment-registry.md docs/governance-decisions.md
git commit -m "docs: register the phase 9.8 measurement floor and H1 closure"
```

---

## After the plan

The deterministic baseline run is **not** authorized by this plan. When the owner grants it, record the grant in the protocol's approval block first, then:

```powershell
.venv/Scripts/python.exe -m scripts.train --config configs/deep3_postholdout_baseline_det.toml --output-dir weights/deep3-postholdout-research-01-baseline-det --save-training-state --require-empty-output-dir --run-id deep3-postholdout-research-01-baseline-det
```

Expect about 8.85 hours. Then evaluate the development OOF metrics and apply the pre-registered validity check: a Macro F1 inside `0.892845` to `0.917199` is consistent with `A_STRICT` not having changed results materially and the value is adopted as the deterministic baseline; a value outside must be investigated before it is recorded as a baseline.

Do not change the seed, the envelope, the MDE constants, the folds, or the manifests during execution. They are frozen, and changing them after seeing a result voids the phase.
