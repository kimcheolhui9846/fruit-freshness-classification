"""Contracts for the approved Phase 8.2 canonical-training unblock."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class CanonicalTrainingUnblockContractTests(unittest.TestCase):
    """Keep the derived run safe, truthful, private, and owner-gated."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.original_config_path = cls.root / "configs" / "deep3.toml"
        cls.derived_config_path = cls.root / "configs" / "deep3_canonical.toml"
        cls.readiness_path = cls.root / "docs" / "canonical-training-readiness.md"
        cls.runbook_path = cls.root / "docs" / "canonical-training-runbook.md"
        cls.training_path = cls.root / "docs" / "training.md"
        cls.unblock_path = cls.root / "docs" / "canonical-training-unblock.md"
        cls.documents = "\n".join(
            path.read_text(encoding="utf-8") if path.is_file() else ""
            for path in (
                cls.readiness_path,
                cls.runbook_path,
                cls.training_path,
                cls.unblock_path,
            )
        )

    def test_original_config_is_unchanged_and_derived_config_diff_is_exact(self) -> None:
        original = tomllib.loads(self.original_config_path.read_text(encoding="utf-8"))
        derived = tomllib.loads(self.derived_config_path.read_text(encoding="utf-8"))
        self.assertEqual(original["training"]["batch_size"], 192)
        self.assertEqual(derived["training"]["batch_size"], 64)

        flattened_original = {
            f"{section}.{key}": value
            for section, values in original.items()
            for key, value in values.items()
        }
        flattened_derived = {
            f"{section}.{key}": value
            for section, values in derived.items()
            for key, value in values.items()
        }
        differing = {
            key
            for key in flattened_original | flattened_derived
            if flattened_original.get(key) != flattened_derived.get(key)
        }
        self.assertEqual(differing, {"training.batch_size"})

    def test_required_phase_documents_exist(self) -> None:
        for path in (
            self.readiness_path,
            self.runbook_path,
            self.training_path,
            self.unblock_path,
        ):
            self.assertTrue(path.is_file(), path)

    def test_approved_config_and_trajectory_boundaries_are_documented(self) -> None:
        for fragment in (
            "configs/deep3_canonical.toml",
            "batch size 64",
            "KEEP_EXISTING_UNSCALED",
            "different optimization trajectory",
            "configs/deep3.toml:",
            "BLOCKED on RTX 3070 Ti 8 GiB",
        ):
            self.assertIn(fragment, self.documents)

    def test_resume_design_and_safe_loading_policy_are_documented(self) -> None:
        for fragment in (
            "epoch-boundary",
            "model_state_dict",
            "ema_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "grad_scaler_state_dict",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
            "after scheduler.step()",
            "atomic",
            "trusted local",
            "must not load a downloaded or untrusted",
        ):
            self.assertIn(fragment, self.documents)

    def test_output_manifest_and_publication_boundaries_are_documented(self) -> None:
        for fragment in (
            "run_manifest.json",
            "non-empty output directory",
            "--require-empty-output-dir",
            "DATASET_PUBLICATION: NO",
            "WEIGHT_PUBLICATION: NO",
            "CHECKPOINT_PUBLICATION: NO",
            "OTHER_BINARY_ARTIFACT_PUBLICATION: NO",
            "results/deep3-canonical-reference-01.log",
        ):
            self.assertIn(fragment, self.documents)

    def test_phase_is_truthful_and_phase_83_remains_owner_gated(self) -> None:
        for fragment in (
            "No full canonical three-fold training was run in Phase 8.2.",
            "No benchmark result is claimed in Phase 8.2.",
            "DERIVED_CANONICAL_CONFIG_READINESS:",
            "READY_FOR_OWNER_APPROVAL",
            "APPROVED_CANONICAL_TRAINING_ACTION:",
            "<RUN_DERIVED_CONFIG | DEFER | BLOCKED>",
        ):
            self.assertIn(fragment, self.documents)

    def test_documents_are_portable_private_and_non_destructive(self) -> None:
        self.assertNotRegex(
            self.documents,
            r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,}|hf_[a-z0-9]{20,})",
        )
        self.assertNotRegex(self.documents, r"(?im)(?:^|[\s(])[a-z]:[\\/]")
        self.assertNotRegex(self.documents, r"(?im)[\w.+-]+@[\w.-]+\.[a-z]{2,}")
        self.assertNotRegex(
            self.documents,
            r"(?im)^\s*(?:git\s+(?:reset|clean|push\s+.*--force|rebase)|rm\s+-rf)\b",
        )


if __name__ == "__main__":
    unittest.main()
