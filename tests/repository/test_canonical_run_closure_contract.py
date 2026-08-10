"""Offline contract for the Phase 8.6 canonical-run closure."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CanonicalRunClosureContractTests(unittest.TestCase):
    """Protect the public, local-only closure boundary without accessing binaries."""

    def test_closure_documents_enforce_local_only_canonical_run_boundary(self) -> None:
        """Missing closure records or an accidental publication claim must fail CI."""
        document_paths = {
            "closure": REPOSITORY_ROOT / "docs" / "canonical-run-closure.md",
            "retention": REPOSITORY_ROOT / "docs" / "canonical-artifact-retention.md",
            "resolution": REPOSITORY_ROOT / "docs" / "phase-8.6-governance-resolution.md",
        }
        missing = [name for name, path in document_paths.items() if not path.is_file()]
        self.assertEqual([], missing, "Phase 8.6 closure documents must be present")

        closure = document_paths["closure"].read_text(encoding="utf-8")
        retention = document_paths["retention"].read_text(encoding="utf-8")
        resolution = document_paths["resolution"].read_text(encoding="utf-8")
        governance = (REPOSITORY_ROOT / "docs" / "governance-decisions.md").read_text(
            encoding="utf-8"
        )
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        for token in (
            "RUN_ID:\ndeep3-canonical-reference-01",
            "CANONICAL_RUN_STATUS:\nCLOSED_REFERENCE",
            "TRAINING:\nCOMPLETED",
            "LOCKED_HOLDOUT_EVALUATION:\nCOMPLETED",
            "POST_HOLDOUT_TUNING:\nNO",
            "CI_LOCAL_BINARY_ARTIFACT_REQUIREMENT:\nNO",
            "CI_PRODUCTION_DATASET_ACCESS:\nNO",
            "CI_CUDA_REQUIREMENT:\nNO",
            "BINARY_RETENTION:\nKEEP_LOCAL_ONLY",
            "RETENTION_DURATION:\nUNTIL_EXPLICIT_OWNER_CHANGE",
        ):
            self.assertIn(token, closure)

        for token in (
            "DOCUMENTATION: PUBLIC",
            "MODEL_WEIGHTS: LOCAL_ONLY",
            "FOLD_CHECKPOINTS: LOCAL_ONLY",
            "FINAL_RAW_CHECKPOINT: LOCAL_ONLY",
            "TRAINING_STATE: LOCAL_ONLY",
            "TRAINING_LOG: LOCAL_ONLY",
            "EVALUATION_LOG: LOCAL_ONLY",
            "RAW_LOGITS: LOCAL_ONLY",
            "RAW_PREDICTIONS: LOCAL_ONLY",
            "DATASET: NOT_REDISTRIBUTED",
            "DATASET_LICENSE_CLEARANCE:\nNOT_CONFIRMED",
        ):
            self.assertIn(token, closure)

        for token in (
            "LOCAL_ONLY: YES",
            "REMOTE_BACKUP: NO",
            "PUBLIC_BACKUP: NO",
            "RELEASE_ASSET: NO",
            "ACTIONS_ARTIFACT: NO",
            "run_manifest.json",
            "training_state.pt",
            "label_names.json",
            "best_model_fold1.pt",
            "best_model_fold2.pt",
            "best_model_fold3.pt",
            "last_model_weights.pt",
            "deep3-canonical-reference-01.log",
            "deep3-canonical-reference-01-holdout-cli.log",
            "deep3-canonical-reference-01-holdout-evaluation.json",
            "deep3-canonical-reference-01-holdout-classification-report.csv",
            "deep3-canonical-reference-01-holdout-confusion-matrix.csv",
            "deep3-canonical-reference-01-holdout-predictions.npz",
        ):
            self.assertIn(token, retention)

        for token in (
            "OWNER_APPROVAL_STATUS:\nAPPROVED",
            "APPROVED_NEXT_ACTION:\nKEEP_ALL_BINARY_ARTIFACTS_LOCAL_ONLY",
            "APPROVED_MODEL_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_DATASET_LICENSE_CLEARANCE:\nNOT_CONFIRMED",
            "APPROVED_CANONICAL_RUN_STATUS:\nCLOSED_REFERENCE",
            "BINARY_PUBLICATION_GATE:\nCLOSED_WITHOUT_PUBLICATION",
        ):
            self.assertIn(token, resolution)

        combined_public_documents = "\n".join(
            (closure, retention, resolution, governance, readme, changelog)
        )
        self.assertIn("future experiments must use a new experiment identity", closure.lower())
        self.assertIn("same holdout is not untouched evidence after tuning", closure.lower())
        self.assertIn("Canonical reference run: Closed", readme)
        self.assertIn("Binaries are not published", readme)
        self.assertIn("docs/canonical-run-closure.md", readme)
        self.assertIn("docs/canonical-artifact-retention.md", readme)
        self.assertIn("docs/phase-8.6-governance-resolution.md", readme)

        for entry in (
            "- Closed the canonical reference run after completed training, locked holdout evaluation, and result interpretation.",
            "- Recorded local-only retention for canonical binary artifacts until an explicit future owner decision.",
            "- No model weights, checkpoints, training state, logs, raw predictions, raw logits, dataset content, GitHub Actions artifacts, Release assets, Release, or tag were published.",
        ):
            self.assertIn(entry, changelog)

        for forbidden_pattern in (
            r"[A-Za-z]:\\",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"(?:ghp_|github_pat_|ctx7sk-)",
        ):
            self.assertIsNone(re.search(forbidden_pattern, combined_public_documents))


if __name__ == "__main__":
    unittest.main()
