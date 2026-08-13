# Post-Holdout loss-001 Runbook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this runbook task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Phase 9.6 loss experiment runnable under a guard that mechanically enforces its single permitted factor, then execute it and apply the frozen decision rule.

**Architecture:** One key-level config comparison in `src/utils/config.py` replaces the existing section-level check for experiment configs, because the permitted difference now lives *inside* a section. Both the training and evaluation entry points dispatch on experiment lineage: a config parented to `deep3-postholdout-research-01` is compared against the canonical config as before, and one parented to `deep3-postholdout-research-01-baseline` is compared against the baseline config under the loss-001 allowance. A separate script applies the frozen decision rule to two OOF metric files so the verdict is computed, not narrated.

**Tech Stack:** Python 3.12, `tomllib`, NumPy, `unittest`. No new dependency.

## Global Constraints

Copied verbatim from [postholdout-loss001-protocol.md](postholdout-loss001-protocol.md). None may be recomputed or adjusted.

- The only permitted differences from the baseline config are `loss.class_balanced_beta`, `post_holdout.experiment_id`, `post_holdout.parent_experiment_id`, and `post_holdout.artifact_namespace`.
- `loss.class_balanced_beta` is `0.999` in the baseline and `0.9999` in the experiment.
- Experiment config LF-normalized SHA-256 is `6ced28e530a4bfef44b0bb22edc24641c68404d552ddc3bfd4c2287888b247ec`; the baseline's is `7cb01e8fe251fd1648ba3a53601e471d9b3693e5d50090f7e7d9c9c5586b11c7`.
- The run reuses the baseline CV manifest `configs/splits/deep3-postholdout-research-01-baseline-cv.json` (LF SHA-256 `494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f`). Different folds would void the comparison.
- `ADVANCE` requires aggregate development OOF Macro F1 ≥ `0.9112` **and** Top-1 ≥ `0.9466`. Anything else is `NOT_ADVANCED`, which retires H1 and makes Phase 9.7 H2 augmentation.
- Baseline reference figures: Macro F1 `0.9012`, Top-1 `0.9566`, `freshpotato` F1 `0.3682`, recall `0.2738`.
- Exactly one candidate. No second run, no other parameter, no threshold change after a result.
- No locked-test or canonical-holdout index enters any model-visible loader. No model touches either pool.
- All artifacts are local-only. No weight, checkpoint, dataset copy, Release, or tag is published.
- Executing the run requires a separate owner decision that has not been made. Tasks 1 through 4 are implementation and are not gated on it; Task 5 is.

## File Structure

| File | Responsibility |
|---|---|
| `src/utils/config.py` | Key-level config flattening and the experiment-lineage validator. Pure; no I/O beyond reading the two configs it is given. |
| `scripts/train.py` | Aborts before dataset preparation when the config's declared lineage fails validation. |
| `scripts/evaluate_postholdout_baseline.py` | Same dispatch, so evaluation cannot run against a config training would have rejected. |
| `scripts/apply_loss001_decision.py` | Reads two OOF metric files, applies the frozen rule, writes the verdict. |
| `tests/config/test_loss_experiment_validation.py` | Unit tests for flattening, the allowance, and rejection. |
| `tests/scripts/test_loss001_decision_cli.py` | CLI contract tests for the decision script. |

---

### Task 1: Key-level experiment config validation

**Files:**
- Modify: `src/utils/config.py`
- Test: `tests/config/test_loss_experiment_validation.py`

**Interfaces:**
- Consumes: `load_experiment_config(path) -> dict` and `_sha256_file(path) -> str`, both already in `src/utils/config.py`.
- Produces:
  - `LOSS001_ALLOWED_DIFFERENCES: frozenset[str]`
  - `flatten_experiment_config(config: dict, prefix: str = "") -> dict[str, object]`
  - `validate_loss_experiment_config(baseline_path, experiment_path, *, allowed_differences: frozenset[str], expected_values: dict[str, object]) -> dict` with keys `single_factor_verified` (bool, always True on success), `baseline_config_sha256`, `experiment_config_sha256`, `differences` (dict of key to `{"baseline": ..., "experiment": ...}`).

The existing `baseline_recipe_differences` compares whole sections. That is sufficient when the only permitted change is the presence of a `post_holdout` section, but loss-001 changes one key *inside* `[loss]`, and a section-level check would accept any other change in the same section. Flattening to dotted keys is what makes the guard exact.

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_loss_experiment_validation.py
import unittest

from src.utils.config import (
    LOSS001_ALLOWED_DIFFERENCES,
    flatten_experiment_config,
    validate_loss_experiment_config,
)

REPOSITORY_ROOT_BASELINE = "configs/deep3_postholdout_baseline.toml"
REPOSITORY_ROOT_EXPERIMENT = "configs/deep3_postholdout_loss001.toml"
EXPECTED = {"loss.class_balanced_beta": 0.9999}


class FlattenExperimentConfigTest(unittest.TestCase):
    def test_nested_sections_become_dotted_keys(self):
        flat = flatten_experiment_config({"loss": {"beta": 0.999}, "training": {"epochs": 120}})

        self.assertEqual(flat, {"loss.beta": 0.999, "training.epochs": 120})

    def test_scalar_at_top_level_is_preserved(self):
        self.assertEqual(flatten_experiment_config({"seed": 42}), {"seed": 42})


class ValidateLossExperimentConfigTest(unittest.TestCase):
    def test_frozen_pair_passes_with_exactly_the_allowed_differences(self):
        result = validate_loss_experiment_config(
            REPOSITORY_ROOT_BASELINE,
            REPOSITORY_ROOT_EXPERIMENT,
            allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
            expected_values=EXPECTED,
        )

        self.assertTrue(result["single_factor_verified"])
        self.assertEqual(set(result["differences"]), set(LOSS001_ALLOWED_DIFFERENCES))
        self.assertEqual(
            result["differences"]["loss.class_balanced_beta"],
            {"baseline": 0.999, "experiment": 0.9999},
        )

    def test_an_extra_changed_key_is_rejected(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "tampered.toml"
            original = Path(REPOSITORY_ROOT_EXPERIMENT).read_text(encoding="utf-8")
            # A second factor is exactly what "one factor at a time" forbids.
            tampered.write_text(original.replace("focal_gamma = 2.0", "focal_gamma = 3.0"), encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                validate_loss_experiment_config(
                    REPOSITORY_ROOT_BASELINE,
                    tampered,
                    allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
                    expected_values=EXPECTED,
                )
        self.assertIn("focal_gamma", str(caught.exception))

    def test_the_experimental_value_itself_is_pinned(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "tampered.toml"
            original = Path(REPOSITORY_ROOT_EXPERIMENT).read_text(encoding="utf-8")
            # The allowed key with an unfrozen value is still an unregistered experiment.
            tampered.write_text(
                original.replace("class_balanced_beta = 0.9999", "class_balanced_beta = 0.99999"),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as caught:
                validate_loss_experiment_config(
                    REPOSITORY_ROOT_BASELINE,
                    tampered,
                    allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
                    expected_values=EXPECTED,
                )
        self.assertIn("0.9999", str(caught.exception))

    def test_an_identical_config_is_rejected_as_no_experiment(self):
        with self.assertRaises(ValueError):
            validate_loss_experiment_config(
                REPOSITORY_ROOT_BASELINE,
                REPOSITORY_ROOT_BASELINE,
                allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
                expected_values=EXPECTED,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.config.test_loss_experiment_validation -v`
Expected: FAIL with `ImportError: cannot import name 'LOSS001_ALLOWED_DIFFERENCES'`

- [ ] **Step 3: Write the implementation**

Append to `src/utils/config.py`:

```python
LOSS001_ALLOWED_DIFFERENCES = frozenset(
    {
        "loss.class_balanced_beta",
        "post_holdout.experiment_id",
        "post_holdout.parent_experiment_id",
        "post_holdout.artifact_namespace",
    }
)


def flatten_experiment_config(config: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a config to dotted keys so a single key can be compared exactly.

    Section-level comparison would accept any other change inside a section
    whose name is allowed to differ, which is precisely what a single-factor
    experiment must not permit.
    """
    flat: dict[str, object] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_experiment_config(value, path))
        else:
            flat[path] = value
    return flat


def validate_loss_experiment_config(
    baseline_path: str | Path,
    experiment_path: str | Path,
    *,
    allowed_differences: frozenset[str],
    expected_values: dict[str, object],
) -> dict[str, object]:
    """Verify an experiment config changes exactly its registered factor."""
    baseline_resolved = Path(baseline_path)
    experiment_resolved = Path(experiment_path)
    baseline = flatten_experiment_config(load_experiment_config(baseline_resolved))
    experiment = flatten_experiment_config(load_experiment_config(experiment_resolved))

    differing = {
        key
        for key in set(baseline) | set(experiment)
        if baseline.get(key) != experiment.get(key)
    }
    unexpected = sorted(differing - allowed_differences)
    if unexpected:
        raise ValueError(
            "Experiment config changes fields outside its registered factor: "
            f"{', '.join(unexpected)}."
        )
    missing = sorted(allowed_differences - differing)
    if missing:
        raise ValueError(
            "Experiment config does not differ from the baseline where it must: "
            f"{', '.join(missing)}."
        )
    for key, expected in expected_values.items():
        if experiment.get(key) != expected:
            raise ValueError(
                f"Experiment config {key} is {experiment.get(key)!r}, "
                f"but the frozen protocol pins it to {expected!r}."
            )

    return {
        "single_factor_verified": True,
        "baseline_config_sha256": _sha256_file(baseline_resolved),
        "experiment_config_sha256": _sha256_file(experiment_resolved),
        "differences": {
            key: {"baseline": baseline.get(key), "experiment": experiment.get(key)}
            for key in sorted(differing)
        },
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.config.test_loss_experiment_validation -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py tests/config/test_loss_experiment_validation.py
git commit -m "feat: add key-level single-factor validation for loss experiments"
```

---

### Task 2: Enforce the guard at both entry points

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/evaluate_postholdout_baseline.py`
- Test: `tests/scripts/test_loss001_lineage_dispatch.py`

**Interfaces:**
- Consumes: `validate_loss_experiment_config` and `LOSS001_ALLOWED_DIFFERENCES` from Task 1.
- Produces: `resolve_experiment_validation(config: dict, config_path: str | Path) -> dict | None`, importable from `src/utils/config.py`, returning the validation result for a baseline-parented config and `None` for a canonical-parented one. The path is a separate argument because `load_experiment_config` does not record where it read from.

`scripts/evaluate_postholdout_baseline.py:384` currently validates every config against `CANONICAL_CONFIG_PATH`. loss-001 differs from canonical inside `[loss]`, so that call rejects it. Dispatching on the declared parent keeps the existing baseline guard exactly as it is while adding the experiment guard beside it — do not relax the canonical comparison to accommodate the new config.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_loss001_lineage_dispatch.py
import unittest

from src.utils.config import load_experiment_config, resolve_experiment_validation


class ResolveExperimentValidationTest(unittest.TestCase):
    LOSS001 = "configs/deep3_postholdout_loss001.toml"
    BASELINE = "configs/deep3_postholdout_baseline.toml"

    def test_baseline_parented_config_is_validated_against_the_baseline(self):
        config = load_experiment_config(self.LOSS001)

        result = resolve_experiment_validation(config, self.LOSS001)

        self.assertIsNotNone(result)
        self.assertTrue(result["single_factor_verified"])
        self.assertIn("loss.class_balanced_beta", result["differences"])

    def test_canonical_parented_config_keeps_the_existing_path(self):
        config = load_experiment_config(self.BASELINE)

        # The baseline is parented to the research identity, not to itself, so
        # it must not be routed through the experiment validator.
        self.assertIsNone(resolve_experiment_validation(config, self.BASELINE))

    def test_unknown_parent_is_rejected_rather_than_silently_skipped(self):
        config = load_experiment_config(self.LOSS001)
        config["post_holdout"]["parent_experiment_id"] = "something-unregistered"

        with self.assertRaises(ValueError):
            resolve_experiment_validation(config, self.LOSS001)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_loss001_lineage_dispatch -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_experiment_validation'`

- [ ] **Step 3: Add the dispatcher**

Append to `src/utils/config.py`:

```python
BASELINE_EXPERIMENT_ID = "deep3-postholdout-research-01-baseline"
RESEARCH_PARENT_ID = "deep3-postholdout-research-01"
BASELINE_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "deep3_postholdout_baseline.toml"
LOSS001_EXPECTED_VALUES = {"loss.class_balanced_beta": 0.9999}


def resolve_experiment_validation(config: dict, config_path: str | Path) -> dict | None:
    """Validate a post-holdout config against the right ancestor, by lineage.

    Returns None for configs parented to the research identity, which keep the
    existing canonical comparison at their call sites. An unrecognized parent
    raises rather than falling through unchecked: a config that names no known
    ancestor has no registered factor to be held to, and silently skipping the
    check is how an unregistered experiment would get to run.
    """
    post_holdout = config.get("post_holdout")
    if post_holdout is None:
        return None
    parent = post_holdout.get("parent_experiment_id")
    if parent == RESEARCH_PARENT_ID:
        return None
    if parent == BASELINE_EXPERIMENT_ID:
        return validate_loss_experiment_config(
            BASELINE_CONFIG_PATH,
            config_path,
            allowed_differences=LOSS001_ALLOWED_DIFFERENCES,
            expected_values=LOSS001_EXPECTED_VALUES,
        )
    raise ValueError(
        f"Config declares an unregistered parent experiment {parent!r}; "
        "every experiment must name the ancestor its single factor is measured against."
    )
```

`config_path` is a separate argument because `load_experiment_config` returns the parsed mapping without recording where it read from, and the validator needs the file itself to hash it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_loss001_lineage_dispatch -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Wire it into both scripts**

In `scripts/train.py`, inside the function that reads `post_holdout` at line 491, before any dataset preparation:

```python
    experiment_validation = resolve_experiment_validation(config, config_path)
    if experiment_validation is not None:
        print(
            "Single-factor validation passed: "
            f"{sorted(experiment_validation['differences'])}"
        )
```

In `scripts/evaluate_postholdout_baseline.py`, replace the unconditional canonical validation at line 384 with:

```python
    config = load_experiment_config(config_path)
    experiment_validation = resolve_experiment_validation(config, config_path)
    if experiment_validation is None:
        validation = validate_postholdout_baseline_config(CANONICAL_CONFIG_PATH, config_path)
        if not validation["recipe_equivalent"]:
            raise RuntimeError("Baseline recipe equivalence validation failed before OOF evaluation.")
```

- [ ] **Step 6: Run the full suite**

```
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
.venv/Scripts/python.exe -m compileall -q src scripts tests
```

Expected: OK. The pre-existing baseline evaluation tests must still pass — they cover the canonical path this change routes around, and a regression there means the dispatch broke the existing guard.

- [ ] **Step 7: Commit**

```bash
git add src/utils/config.py scripts/train.py scripts/evaluate_postholdout_baseline.py tests/scripts/test_loss001_lineage_dispatch.py
git commit -m "feat: enforce single-factor validation by experiment lineage"
```

---

### Task 3: Apply the frozen decision rule in code

**Files:**
- Create: `scripts/apply_loss001_decision.py`
- Test: `tests/scripts/test_loss001_decision_cli.py`

**Interfaces:**
- Consumes: two `development_oof_metrics.json` files produced by `scripts/evaluate_postholdout_baseline.py`, each carrying `metrics.macro_f1` and `metrics.top1_accuracy`.
- Produces: `apply_decision(baseline_metrics: dict, experiment_metrics: dict) -> dict` with keys `outcome` (`"ADVANCE"` or `"NOT_ADVANCED"`), `macro_f1_delta`, `top1_delta`, `clears_macro_f1`, `clears_top1_guardrail`, `next_phase`.

The verdict is computed rather than written by hand for the same reason the Phase 9.5 threshold was: a rule applied by a person after seeing the number is not a pre-committed rule.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_loss001_decision_cli.py
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "phase96_apply_loss001_decision",
    Path(__file__).resolve().parents[2] / "scripts" / "apply_loss001_decision.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _metrics(macro_f1, top1):
    return {"metrics": {"macro_f1": macro_f1, "top1_accuracy": top1}}


class ApplyDecisionTest(unittest.TestCase):
    BASE = _metrics(0.9012, 0.9566)

    def test_clearing_both_thresholds_advances(self):
        result = module.apply_decision(self.BASE, _metrics(0.9112, 0.9466))

        self.assertEqual(result["outcome"], "ADVANCE")
        self.assertTrue(result["clears_macro_f1"])
        self.assertTrue(result["clears_top1_guardrail"])

    def test_macro_f1_just_below_the_threshold_does_not_advance(self):
        # 0.9111 is one ten-thousandth under. The rule is a threshold, not a mood.
        result = module.apply_decision(self.BASE, _metrics(0.9111, 0.9600))

        self.assertEqual(result["outcome"], "NOT_ADVANCED")
        self.assertFalse(result["clears_macro_f1"])

    def test_the_top1_guardrail_can_veto_a_macro_f1_gain(self):
        result = module.apply_decision(self.BASE, _metrics(0.9300, 0.9400))

        self.assertEqual(result["outcome"], "NOT_ADVANCED")
        self.assertTrue(result["clears_macro_f1"])
        self.assertFalse(result["clears_top1_guardrail"])

    def test_not_advancing_names_h2_as_the_next_phase(self):
        result = module.apply_decision(self.BASE, _metrics(0.9012, 0.9566))

        self.assertIn("H2", result["next_phase"])

    def test_parser_requires_both_metric_files(self):
        with self.assertRaises(SystemExit):
            module.build_parser().parse_args(["--baseline-metrics", "a.json"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_loss001_decision_cli -v`
Expected: FAIL with `FileNotFoundError` on `scripts/apply_loss001_decision.py`

- [ ] **Step 3: Write the implementation**

```python
# scripts/apply_loss001_decision.py
"""Apply the frozen Phase 9.6 decision rule to two OOF metric files.

The thresholds are the protocol's, fixed before the run. This script computes
the verdict so it cannot be decided after the number is seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

ADVANCE_MACRO_F1 = 0.9112
TOP1_GUARDRAIL = 0.9466
NEXT_PHASE_ON_ADVANCE = "Owner decision, informed by the result"
NEXT_PHASE_ON_HOLD = "H1 exhausted; Phase 9.7 is H2 augmentation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the frozen loss-001 decision rule."
    )
    parser.add_argument("--baseline-metrics", required=True, help="Baseline development_oof_metrics.json.")
    parser.add_argument("--experiment-metrics", required=True, help="loss-001 development_oof_metrics.json.")
    parser.add_argument("--output", required=True, help="Path for the verdict JSON.")
    return parser


def apply_decision(baseline_metrics: dict, experiment_metrics: dict) -> dict:
    """Compute the verdict. Both conditions must hold; neither is negotiable."""
    base = baseline_metrics["metrics"]
    exp = experiment_metrics["metrics"]
    clears_macro = exp["macro_f1"] >= ADVANCE_MACRO_F1
    clears_top1 = exp["top1_accuracy"] >= TOP1_GUARDRAIL
    advance = clears_macro and clears_top1
    return {
        "outcome": "ADVANCE" if advance else "NOT_ADVANCED",
        "advance_macro_f1_threshold": ADVANCE_MACRO_F1,
        "top1_guardrail": TOP1_GUARDRAIL,
        "baseline_macro_f1": base["macro_f1"],
        "experiment_macro_f1": exp["macro_f1"],
        "macro_f1_delta": exp["macro_f1"] - base["macro_f1"],
        "baseline_top1": base["top1_accuracy"],
        "experiment_top1": exp["top1_accuracy"],
        "top1_delta": exp["top1_accuracy"] - base["top1_accuracy"],
        "clears_macro_f1": clears_macro,
        "clears_top1_guardrail": clears_top1,
        "next_phase": NEXT_PHASE_ON_ADVANCE if advance else NEXT_PHASE_ON_HOLD,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} already exists; refusing to overwrite a recorded verdict.")

    verdict = apply_decision(
        json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8")),
        json.loads(Path(args.experiment_metrics).read_text(encoding="utf-8")),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Outcome: {verdict['outcome']}")
    print(f"Macro F1 {verdict['experiment_macro_f1']:.4f} (delta {verdict['macro_f1_delta']:+.4f})")
    print(f"Top-1    {verdict['experiment_top1']:.4f} (delta {verdict['top1_delta']:+.4f})")
    print(f"Next: {verdict['next_phase']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.scripts.test_loss001_decision_cli -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the full suite and compileall**

```
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"
.venv/Scripts/python.exe -m compileall -q src scripts tests
```

- [ ] **Step 6: Commit**

```bash
git add scripts/apply_loss001_decision.py tests/scripts/test_loss001_decision_cli.py
git commit -m "feat: compute the frozen loss-001 decision rule"
```

---

### Task 4: Execute the run

This task is procedural and is gated on an owner decision that has not been made. Do not begin it without one. Its ordering is what the protocol depends on; do not reorder.

- [ ] **Step 1: Preflight**

Confirm the runtime matches the recorded baseline preflight (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `datasets==5.0.1`, RTX 3070 Ti). Recompute the LF-normalized SHA-256 of `configs/deep3_postholdout_loss001.toml` and compare against `6ced28e530a4bfef44b0bb22edc24641c68404d552ddc3bfd4c2287888b247ec`. Confirm `weights/deep3-postholdout-research-01-loss-001` and `results/deep3-postholdout-research-01-loss-001` are absent. Confirm Windows active hours cover the projected nine to twelve hours.

- [ ] **Step 2: Train**

```powershell
.venv\Scripts\python.exe -m scripts.train `
  --config configs/deep3_postholdout_loss001.toml `
  --output-dir weights/deep3-postholdout-research-01-loss-001 `
  --save-training-state `
  --require-empty-output-dir `
  --run-id deep3-postholdout-research-01-loss-001 2>&1 | tee -a results/deep3-postholdout-research-01-loss-001.log
```

The single-factor validation prints before dataset preparation. If it does not appear, stop — the guard did not run.

Resume after an interruption with `--resume-state weights/deep3-postholdout-research-01-loss-001/training_state.pt` and without `--require-empty-output-dir`.

- [ ] **Step 3: Evaluate**

```powershell
.venv\Scripts\python.exe -m scripts.evaluate_postholdout_baseline `
  --config configs/deep3_postholdout_loss001.toml `
  --checkpoint-dir weights/deep3-postholdout-research-01-loss-001 `
  --output-dir results/deep3-postholdout-research-01-loss-001
```

- [ ] **Step 4: Apply the rule**

```powershell
.venv\Scripts\python.exe -m scripts.apply_loss001_decision `
  --baseline-metrics results/deep3-postholdout-research-01-baseline/development_oof_metrics.json `
  --experiment-metrics results/deep3-postholdout-research-01-loss-001/development_oof_metrics.json `
  --output results/deep3-postholdout-research-01-loss-001/decision.json
```

Report the outcome exactly as computed. If it is `NOT_ADVANCED`, record H1 as exhausted and Phase 9.7 as H2 augmentation. Do not re-read the threshold against the result.

- [ ] **Step 5: Record**

Report aggregate and per-fold Macro F1 separately — they are different quantities. Report the diagnostics the protocol lists, including `freshpotato` F1 and recall against 0.3682 and 0.2738, and name the case where `freshpotato` improves while Macro F1 does not. Update the protocol's `EXECUTION_STATUS`, the registry, governance decisions, `SESSION_HANDOFF.md`, and `CHANGELOG.md`.

## Stop Conditions

Stop and report rather than continuing if the single-factor validation does not print or does not pass, if any config or manifest hash differs from the frozen value, if a locked-test or canonical-holdout index reaches a model-visible loader, if an OOM or thermal or disk limit is hit, or if the CV fold hashes fail on resume.

Do not change the threshold, the guardrail, the changed parameter, or any other hyperparameter during or after the run. Those are frozen, and changing them after seeing a result voids the pre-registration this phase exists to preserve.
