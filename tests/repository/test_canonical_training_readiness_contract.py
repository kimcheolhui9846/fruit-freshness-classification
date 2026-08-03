"""Offline documentation contracts for canonical-training readiness."""

from __future__ import annotations

import unittest
from pathlib import Path


class CanonicalTrainingReadinessContractTests(unittest.TestCase):
    """Keep the planned canonical run frozen, private, and owner-gated."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.readiness_path = cls.root / "docs" / "canonical-training-readiness.md"
        cls.runbook_path = cls.root / "docs" / "canonical-training-runbook.md"
        cls.readiness = (
            cls.readiness_path.read_text(encoding="utf-8")
            if cls.readiness_path.is_file()
            else ""
        )
        cls.runbook = (
            cls.runbook_path.read_text(encoding="utf-8")
            if cls.runbook_path.is_file()
            else ""
        )
        cls.documents = cls.readiness + "\n" + cls.runbook

    def test_required_documents_exist(self) -> None:
        self.assertTrue(self.readiness_path.is_file())
        self.assertTrue(self.runbook_path.is_file())

    def test_final_readiness_is_explicitly_blocked_without_claiming_execution(self) -> None:
        blocked = "**Final readiness classification:** " + chr(96) + "BLOCKED" + chr(96)
        self.assertIn(blocked, self.readiness)
        for fragment in (
            "No canonical three-fold training was run in Phase 8.1.",
            "No canonical checkpoint, weight, result, benchmark, or publication artifact was created.",
            "A configuration change is outside this Phase and requires a new explicit approval.",
        ):
            self.assertIn(fragment, self.documents)

    def test_frozen_identity_and_dependency_hashes_are_recorded(self) -> None:
        for fragment in (
            "046760e19e77c7aa0c6cbc065358acfd46aac346",
            "7d88da60f540728aae9259273aae32b4ce0b3bc1",
            "62c7ae4ee5c33974fa48342b6af1b7b54c2e4938159429cbd1a86524fc7c13f1",
            "a0f8ab7af3593786d635f84338a8f53490936147",
            "86776c4ccd296dfb828121bd968ecf8cec8fd763b4a2cd600c68a280c6a90919",
            "89e729557cdee7a20ba8637ce2dd22ba4e2db7ab",
            "73e6c3f9d71614711e6e6ac942bba660463a8a80831a609b36a59a41d6e38e4d",
            "9a3ef99a5b0595b309a63909f10320ea80e22d16",
            "9778a941d18240c3813ee24fcd77e61b0eeef33cd5e45c21c9ba9d0286df06a4",
            "8fbffd3aae3a1828783b0691ac9287778f4509a4",
            "379b976f196a05f584c39fdef79489f2f5c321d1207c7475f798ea2e6794b6",
            "batch size 192",
            "120 epochs",
            "3 folds",
        ):
            self.assertIn(fragment, self.documents)

    def test_dataset_fold_and_runtime_evidence_are_recorded(self) -> None:
        for fragment in (
            "Densu341/Fresh-rotten-fruit",
            "2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
            "a34c57ba3354f94d4cc04c4b83939bd6a3105d3708b9a0cd57145b6fc127254e",
            "30,357",
            "26,858",
            "21,486",
            "5,372",
            "14 classes",
            "LIKELY_UNSAFE",
            "107.2%",
            "NVIDIA GeForce RTX 3070 Ti",
            "torch 2.6.0+cu124",
            "datasets 5.0.1",
        ):
            self.assertIn(fragment, self.documents)

    def test_runbook_preserves_exact_cli_and_interruptibility_boundary(self) -> None:
        self.assertIn(
            "python scripts/train.py --config configs/deep3.toml --output-dir weights/<run-id>",
            self.runbook,
        )
        for fragment in (
            "No resume implementation exists in the frozen training entry point.",
            "optimizer, scheduler, AMP scaler, EMA, and RNG states are not checkpointed",
            "restart from the beginning",
            "python scripts/evaluate.py --config configs/deep3.toml --checkpoint-dir weights/<run-id>",
        ):
            self.assertIn(fragment, self.runbook)

    def test_owner_approvals_and_publication_prohibitions_are_unresolved(self) -> None:
        approvals = (
            "OWNER_CANONICAL_TRAINING_APPROVAL:",
            "PENDING",
            "OWNER_BATCH_SIZE_DECISION:",
            "OWNER_OUTPUT_DIRECTORY_APPROVAL:",
            "OWNER_INTERRUPTION_RISK_ACCEPTANCE:",
        )
        for fragment in approvals:
            self.assertIn(fragment, self.documents)
        for fragment in (
            "DATASET_PUBLICATION: NO",
            "WEIGHT_PUBLICATION: NO",
            "CHECKPOINT_PUBLICATION: NO",
            "OTHER_BINARY_ARTIFACT_PUBLICATION: NO",
        ):
            self.assertIn(fragment, self.documents)

    def test_documents_are_private_portable_and_non_destructive(self) -> None:
        self.assertNotRegex(
            self.documents,
            r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,}|hf_[a-z0-9]{20,})",
        )
        self.assertNotRegex(
            self.documents,
            r"(?im)(?:^|[\s(])[a-z]:[\\/]",
        )
        self.assertNotRegex(
            self.documents,
            r"(?im)^\s*(?:git\s+(?:reset|clean|push\s+.*--force|rebase)|rm\s+-rf)\b",
        )


if __name__ == "__main__":
    unittest.main()
