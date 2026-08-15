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


def _cuda_is_initialized() -> bool:
    """Seam for the A_STRICT guard.

    Tests patch this rather than `torch.cuda.is_initialized`, because
    `torch.manual_seed` consults that predicate internally and patching it
    breaks torch's own seeding path.
    """
    return torch.cuda.is_initialized()


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
    if level == A_STRICT and _cuda_is_initialized():
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
