"""Offline contract for the completed Phase 8.4 canonical holdout record."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DOCUMENT = ROOT / "docs" / "canonical-holdout-evaluation.md"
ARTIFACT_DOCUMENT = ROOT / "docs" / "canonical-holdout-artifacts.md"
EVALUATION_GUIDE = ROOT / "docs" / "evaluation.md"
READINESS_DOCUMENT = ROOT / "docs" / "canonical-training-readiness.md"


class CanonicalHoldoutEvaluationContractTest(unittest.TestCase):
    """Keep published holdout evidence complete, local-only, and CI-offline."""

    def setUp(self) -> None:
        self.evaluation = (
            EVALUATION_DOCUMENT.read_text(encoding="utf-8")
            if EVALUATION_DOCUMENT.is_file()
            else ""
        )
        self.artifacts = (
            ARTIFACT_DOCUMENT.read_text(encoding="utf-8")
            if ARTIFACT_DOCUMENT.is_file()
            else ""
        )
        self.guide = EVALUATION_GUIDE.read_text(encoding="utf-8")
        self.readiness = READINESS_DOCUMENT.read_text(encoding="utf-8")
        self.documents = "\n".join((self.evaluation, self.artifacts, self.guide, self.readiness))

    def test_evaluation_record_preserves_the_frozen_protocol_and_complete_results(self) -> None:
        self.assertTrue(EVALUATION_DOCUMENT.is_file())
        for value in (
            "deep3-canonical-reference-01",
            "0c669d58852082785c79699231e09b5ae26757cc",
            "4b3808efb3abaf4682e1150ce69ddcdb6585e451",
            "configs/deep3_canonical.toml",
            "batch size: 64",
            "5,372",
            "best_model_fold1.pt",
            "best_model_fold2.pt",
            "best_model_fold3.pt",
            "last_model_weights.pt is excluded",
            "equal raw-logit ensemble",
            "horizontal-flip TTA",
            "argmax",
            "No threshold tuning",
            "5,133 / 5,372",
            "0.955510",
            "0.949663",
            "0.960706",
            "0.903737",
            "0.899969",
            "0.981199",
            "0.992740",
            "MATCH_EXACT_WITHIN_1E-12",
            "Rows are true classes; columns are predicted classes.",
            "internal fixed holdout",
            "No post-holdout tuning occurred.",
            "No state-of-the-art claim is made.",
            "batch 64 is a different training trajectory from batch 192",
            "Phase 8.5 documentation and publication decision is recorded.",
        ):
            self.assertIn(value, self.evaluation)
        for label in (
            "freshapples", "freshbanana", "freshcapsicum", "freshcucumber",
            "freshoranges", "freshpotato", "freshtomato", "rottenapples",
            "rottenbanana", "rottencapsicum", "rottencucumber", "rottenoranges",
            "rottenpotato", "rottentomato",
        ):
            self.assertIn(label, self.evaluation)

    def test_artifact_record_keeps_all_local_results_unpublished_and_portable(self) -> None:
        self.assertTrue(ARTIFACT_DOCUMENT.is_file())
        for artifact in (
            "holdout-cli.log",
            "holdout-evaluation.json",
            "holdout-classification-report.csv",
            "holdout-confusion-matrix.csv",
            "holdout-predictions.npz",
        ):
            self.assertIn(artifact, self.artifacts)
        self.assertGreaterEqual(len(re.findall(r"\b[a-f0-9]{64}\b", self.artifacts)), 5)
        for value in (
            "Local-only: Yes",
            "Ignored: Yes",
            "Tracked: No",
            "Published: No",
            "Raw logits and predictions published: No",
            "Checkpoint publication: No",
            "Weight publication: No",
            "Dataset publication: No",
            "Release creation: No",
            "CI checkpoint requirement: No",
            "CI CUDA requirement: No",
            "CI production dataset access: No",
            "CI local evaluation output requirement: No",
            "Phase 8.5 documentation and publication decision is recorded.",
        ):
            self.assertIn(value, self.artifacts)
        self.assertNotRegex(self.artifacts, r"[A-Za-z]:[\\/]")
        self.assertNotRegex(self.artifacts, r"[\w.+-]+@[\w.-]+")
        self.assertNotRegex(self.artifacts, r"(?:ctx7sk-|ghp_|github_pat_|hf_[a-zA-Z0-9]{20,})")

    def test_existing_evaluation_and_readiness_documents_distinguish_trained_evidence(self) -> None:
        self.assertIn("Canonical trained holdout evaluation", self.guide)
        self.assertIn("canonical-holdout-evaluation.md", self.guide)
        self.assertIn("temporary untrained compatibility", self.guide)
        self.assertIn("TRAINED_CHECKPOINT_HOLDOUT_EVALUATION:\nCOMPLETED", self.readiness)
        self.assertIn("canonical-holdout-evaluation.md", self.readiness)

    def test_committed_contract_stays_offline_and_does_not_require_local_artifacts(self) -> None:
        self.assertTrue(EVALUATION_DOCUMENT.is_file())
        self.assertTrue(ARTIFACT_DOCUMENT.is_file())
        for prohibited in (
            "C:\\Users",
            "load_fruit_freshness_dataset()",
            "torch.cuda",
        ):
            self.assertNotIn(prohibited, self.documents)


if __name__ == "__main__":
    unittest.main()