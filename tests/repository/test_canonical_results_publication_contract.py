"""Offline contract for approved Phase 8.5 aggregate result documentation."""

from __future__ import annotations

from pathlib import Path
import unittest


class CanonicalResultsPublicationContractTest(unittest.TestCase):
    """Keep public aggregate results truthful and binary artifacts local-only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.results_path = cls.root / "docs" / "canonical-results.md"
        cls.model_card_path = cls.root / "docs" / "model-card.md"
        cls.publication_decision_path = (
            cls.root / "docs" / "artifact-publication-decision.md"
        )
        cls.artifacts_path = cls.root / "docs" / "canonical-holdout-artifacts.md"
        cls.readme_path = cls.root / "README.md"
        cls.results = (
            cls.results_path.read_text(encoding="utf-8")
            if cls.results_path.is_file()
            else ""
        )
        cls.model_card = (
            cls.model_card_path.read_text(encoding="utf-8")
            if cls.model_card_path.is_file()
            else ""
        )
        cls.publication_decision = (
            cls.publication_decision_path.read_text(encoding="utf-8")
            if cls.publication_decision_path.is_file()
            else ""
        )
        cls.artifacts = cls.artifacts_path.read_text(encoding="utf-8")
        cls.readme = cls.readme_path.read_text(encoding="utf-8")

    def test_approved_aggregate_results_are_documented(self) -> None:
        self.assertTrue(self.results_path.is_file())
        self.assertTrue(self.model_card_path.is_file())
        published = "\n".join((self.results, self.model_card, self.readme))
        for value in (
            "deep3-canonical-reference-01",
            "5,133 / 5,372",
            "0.955510",
            "0.903737",
            "0.899969",
            "freshpotato",
            "rottencucumber",
            "rottentomato",
            "internal fixed holdout",
            "No post-holdout tuning occurred.",
        ):
            self.assertIn(value, published)

    def test_error_concentration_and_rank_recovery_are_interpreted_without_causal_claims(self) -> None:
        for value in (
            "239 top-1 errors",
            "194 concentrated errors",
            "81.17%",
            "5,271 top-2 correct",
            "5,333 top-3 correct",
            "138 top-1 errors recovered at rank 2",
            "138 rank-2 recovery",
            "62 additional errors recovered at rank 3",
            "62 additional rank-3 recovery",
            "39 outside top-3",
            "44 freshpotato examples were classified as rottenpotato.",
            "Observed fact",
            "Supported inference",
            "Unverified hypothesis",
            "top-k rank recovery does not imply calibrated confidence",
        ):
            self.assertIn(value, self.results)
        for label in (
            "freshapples", "freshbanana", "freshcapsicum", "freshcucumber",
            "freshoranges", "freshpotato", "freshtomato", "rottenapples",
            "rottenbanana", "rottencapsicum", "rottencucumber", "rottenoranges",
            "rottenpotato", "rottentomato",
        ):
            self.assertIn(label, self.results)
        self.assertIn("No state-of-the-art claim is made.", self.results)
        self.assertIn("No post-holdout tuning occurred.", self.results)

    def test_publication_governance_blocks_binary_publication_and_leaves_phase_8_6_owner_gates_unresolved(self) -> None:
        self.assertTrue(self.publication_decision_path.is_file())
        for value in (
            "CURRENT_PUBLICATION_ACTION:\nPUBLISH_DOCUMENTATION_ONLY",
            "MODEL_WEIGHT_PUBLICATION:\nBLOCKED_PENDING_LICENSE_AND_PROVENANCE_CLEARANCE",
            "TRAINING_STATE_PUBLICATION:\nNO",
            "RAW_PREDICTION_PUBLICATION:\nNO",
            "DATASET_PUBLICATION:\nNO",
            "BINARY_PUBLICATION_GATE:\nBLOCKED",
            "PRIMARY_RECOMMENDATION:\nPUBLISH_DOCUMENTATION_ONLY",
            "SECONDARY_RECOMMENDATION:\nKEEP_ALL_BINARY_ARTIFACTS_LOCAL_ONLY",
            "APPROVED_NEXT_ACTION:",
            "<PUBLISH_DOCUMENTATION_ONLY |",
            "APPROVED_MODEL_WEIGHT_PUBLICATION:",
            "APPROVED_CHECKPOINT_SET:",
            "APPROVED_ARTIFACT_FORMAT:",
            "APPROVED_HOSTING_DESTINATION:",
            "APPROVED_DATASET_LICENSE_CLEARANCE:",
            "APPROVED_MODEL_CARD_PUBLICATION:",
            "APPROVED_BINARY_RETENTION:",
            "APPROVED_RELEASE_CREATION:",
            "APPROVED_TAG_CREATION:",
            "Normal CI does not require local artifacts, CUDA, or production dataset access.",
        ):
            self.assertIn(value, self.publication_decision)

    def test_model_card_excludes_food_safety_and_autonomous_operational_use(self) -> None:
        for value in (
            "Visual freshness classification does not establish whether food is safe to eat.",
            "food-safety decisions",
            "pathogen or toxin detection",
            "mold-safety determination",
            "laboratory inspection replacement",
            "health or medical decisions",
            "regulatory decisions",
            "autonomous commercial disposal",
            "autonomous inventory rejection",
            "unvalidated cameras, lighting, fruit varieties, or domains",
            "human-reviewed prototyping",
        ):
            self.assertIn(value, self.model_card)

    def test_publication_boundary_keeps_binaries_and_raw_outputs_local_only(self) -> None:
        published = "\n".join(
            (
                self.results,
                self.model_card,
                self.publication_decision,
                self.artifacts,
                self.readme,
            )
        )
        for value in (
            "Raw logits and predictions published: No",
            "Checkpoint publication: No",
            "Weight publication: No",
            "Dataset publication: No",
            "Training-state publication: No",
            "Execution-log publication: No",
            "All binary artifacts remain local-only through Phase 8.6.",
        ):
            self.assertIn(value, published)
        for value in (
            "GITHUB_ACTIONS_ARTIFACT_UPLOAD:\\nNO",
            "RELEASE_ASSET_UPLOAD:\\nNO",
            "LOCAL_ARTIFACT_RETENTION:\\nYES",
        ):
            self.assertIn(value.replace("\\n", "\n"), self.publication_decision)
        self.assertNotRegex(published, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(published, r"(?i)(?:ctx7sk-|ghp_|github_pat_|hf_[a-zA-Z0-9]{20,})")
        self.assertNotRegex(published, r"[\\w.+-]+@[\\w.-]+")

    def test_model_card_separates_software_data_and_weight_terms(self) -> None:
        self.assertIn("MIT License", self.model_card)
        self.assertIn("Densu341/Fresh-rotten-fruit", self.model_card)
        self.assertIn("not redistributed", self.model_card)
        self.assertIn("separate review before publication", self.model_card)


if __name__ == "__main__":
    unittest.main()