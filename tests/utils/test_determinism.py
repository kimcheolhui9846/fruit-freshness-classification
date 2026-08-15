"""Behaviour of the Phase 9.7 seeding and backend determinism policy."""

import contextlib
import os
import random
import unittest
from unittest import mock

import numpy as np
import torch

from src.utils.config import A_STRICT, B_CUDNN, C_SEED_ONLY, DETERMINISM_LEVELS
from src.utils import determinism as determinism_module
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


@contextlib.contextmanager
def _pretend_cuda_initialized(value: bool):
    """Force the A_STRICT guard's answer through the module's own seam.

    The guard reads process-global CUDA state, so any earlier test in the
    suite that allocates on the GPU changes it, and the A_STRICT behaviour
    tests would otherwise pass alone and fail in the suite for a reason
    unrelated to the behaviour under test. `torch.cuda.is_initialized` is not
    patched directly: `torch.manual_seed` consults it internally, so patching
    it breaks torch's own seeding path.
    """
    with mock.patch.object(
        determinism_module, "_cuda_is_initialized", return_value=value
    ):
        yield


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
            with self.subTest(value=value):
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
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        record = apply_determinism(B_CUDNN, seed=20260815)

        self.assertFalse(record["cudnn_benchmark"])
        self.assertTrue(record["cudnn_deterministic"])
        self.assertFalse(record["use_deterministic_algorithms"])
        self.assertIsNone(record["cublas_workspace_config"])

    def test_a_strict_refuses_after_cuda_is_initialized(self):
        with _pretend_cuda_initialized(True):
            # CUBLAS_WORKSPACE_CONFIG is read when the cuBLAS handle is
            # created. Applying it later is ignored, not refused, so the
            # policy must refuse on our behalf rather than record a setting
            # that never took effect.
            with self.assertRaises(RuntimeError):
                apply_determinism(A_STRICT, seed=20260815)

    def test_lower_levels_are_unaffected_by_cuda_initialization(self):
        with _pretend_cuda_initialized(True):
            for level in (B_CUDNN, C_SEED_ONLY):
                with self.subTest(level=level):
                    record = apply_determinism(level, seed=20260815)
                    self.assertEqual(record["level"], level)

    def test_a_strict_enables_strict_algorithms_and_the_cublas_workspace(self):
        with _pretend_cuda_initialized(False):
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
                with _pretend_cuda_initialized(False):
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
        torch.use_deterministic_algorithms(False)
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
