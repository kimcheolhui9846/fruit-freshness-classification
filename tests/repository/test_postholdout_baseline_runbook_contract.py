"""Offline contract for the prepared but unexecuted baseline runbook."""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "docs" / "postholdout-baseline-runbook.md"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"
SPLIT_MANIFEST = ROOT / "configs" / "splits" / "deep3-postholdout-research-01.json"
CV_MANIFEST = ROOT / "configs" / "splits" / "deep3-postholdout-research-01-baseline-cv.json"


def _lf_normalized_sha256(path: Path) -> str:
    """Hash tracked text the way CI must, so Windows checkouts agree with Linux."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class PostHoldoutBaselineRunbookContractTest(unittest.TestCase):
    def test_runbook_documents_a_prepared_but_unauthorized_execution(self):
        document = RUNBOOK.read_text(encoding="utf-8")

        for token in (
            "RUNBOOK_STATUS:\nPREPARED",
            "BASELINE_EXECUTION_STATUS:\nNOT_YET_RUN",
            "PHASE_9_4:\nRUNBOOK_PREPARED",
            "PHASE_9_4_TRAINING_AUTHORIZATION:\nNOT GRANTED",
            "APPROVED_BASELINE_EXECUTION_ACTION:\n<RUN_BASELINE | DEFER>",
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_CANONICAL_HOLDOUT_REEVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_DATASET_PUBLICATION:\nNO",
            "APPROVED_RELEASE_CREATION:\nNO",
        ):
            self.assertIn(token, document)

    def test_runbook_records_the_approved_identity_and_paths(self):
        document = RUNBOOK.read_text(encoding="utf-8")

        for token in (
            "deep3-postholdout-research-01-baseline",
            "configs/deep3_postholdout_baseline.toml",
            "weights/deep3-postholdout-research-01-baseline",
            "results/deep3-postholdout-research-01-baseline.log",
            "python -m scripts.train",
            "python -m scripts.evaluate_postholdout_baseline",
            "--require-empty-output-dir",
            "--save-training-state",
        ):
            self.assertIn(token, document)

    def test_documented_hashes_match_the_frozen_files(self):
        document = RUNBOOK.read_text(encoding="utf-8")

        for path in (BASELINE_CONFIG, SPLIT_MANIFEST, CV_MANIFEST):
            self.assertIn(_lf_normalized_sha256(path), document)

    def test_runbook_keeps_both_evaluation_boundaries_closed(self):
        document = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("FROZEN_UNOBSERVED_BY_MODEL", document)
        self.assertIn("17,188", document)
        self.assertIn("4,298", document)
        self.assertIn("5,372", document)
        self.assertNotIn("APPROVED_BASELINE_EXECUTION_ACTION:\nRUN_BASELINE", document)


if __name__ == "__main__":
    unittest.main()
