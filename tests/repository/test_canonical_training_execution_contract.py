"""Offline contract for the completed Phase 8.3 canonical training record."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_DOCUMENT = ROOT / "docs" / "canonical-training-execution.md"
ARTIFACT_DOCUMENT = ROOT / "docs" / "canonical-training-artifacts.md"
TRAINING_COMMIT = "0c669d58852082785c79699231e09b5ae26757cc"
ARTIFACTS = (
    "run_manifest.json",
    "training_state.pt",
    "label_names.json",
    "best_model_fold1.pt",
    "best_model_fold2.pt",
    "best_model_fold3.pt",
    "last_model_weights.pt",
    "deep3-canonical-reference-01.log",
)


class CanonicalTrainingExecutionContractTest(unittest.TestCase):
    def test_documents_record_completed_local_only_training_without_metrics(self):
        self.assertTrue(EXECUTION_DOCUMENT.is_file())
        self.assertTrue(ARTIFACT_DOCUMENT.is_file())
        execution = EXECUTION_DOCUMENT.read_text(encoding="utf-8")
        artifacts = ARTIFACT_DOCUMENT.read_text(encoding="utf-8")
        for value in (
            TRAINING_COMMIT,
            "configs/deep3_canonical.toml",
            "batch size: 64",
            "three folds",
            "120 epochs per fold",
            "epoch 101",
            "COMPLETED",
            "Holdout evaluation: No",
            "Benchmark claim: No",
            "configs/deep3.toml remains BLOCKED",
            "different trajectory from batch 192",
            "Phase 8.4: PENDING",
        ):
            self.assertIn(value, execution)
        for artifact in ARTIFACTS:
            self.assertIn(artifact, artifacts)
        self.assertEqual(len(re.findall(r"\b[a-f0-9]{64}\b", artifacts)), 8)
        for value in ("Local-only: Yes", "Tracked: No", "Published: No"):
            self.assertIn(value, artifacts)
        combined = execution + "\n" + artifacts
        self.assertNotRegex(combined, r"[A-Za-z]:[\\/]")
        self.assertNotRegex(combined, r"[\w.+-]+@[\w.-]+")
        self.assertNotRegex(combined, r"(?:ctx7sk-|ghp_|github_pat_)")

    def test_documents_keep_ci_offline_from_cuda_dataset_and_checkpoints(self):
        self.assertTrue(EXECUTION_DOCUMENT.is_file())
        self.assertTrue(ARTIFACT_DOCUMENT.is_file())
        combined = (
            EXECUTION_DOCUMENT.read_text(encoding="utf-8")
            + "\n"
            + ARTIFACT_DOCUMENT.read_text(encoding="utf-8")
        )
        for value in (
            "CI checkpoint requirement: No",
            "CI CUDA requirement: No",
            "CI production dataset access: No",
            "Release creation: No",
        ):
            self.assertIn(value, combined)


if __name__ == "__main__":
    unittest.main()