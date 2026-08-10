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

    def test_publication_boundary_keeps_binaries_and_raw_outputs_local_only(self) -> None:
        published = "\n".join((self.results, self.model_card, self.artifacts, self.readme))
        for value in (
            "Raw logits and predictions published: No",
            "Checkpoint publication: No",
            "Weight publication: No",
            "Dataset publication: No",
            "Training-state publication: No",
            "Execution-log publication: No",
            "GitHub Actions artifact upload: No",
            "Release asset upload: No",
            "Retained locally through Phase 8.6",
        ):
            self.assertIn(value, published)
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