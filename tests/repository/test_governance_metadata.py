"""Offline contracts for approved license and repository citation metadata."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class GovernanceMetadataContractTests(unittest.TestCase):
    """Keep Phase 6.4 governance metadata truthful, portable, and release-neutral."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.license_path = cls.root / "LICENSE"
        cls.citation_path = cls.root / "CITATION.cff"
        cls.readme_path = cls.root / "README.md"
        cls.governance_path = cls.root / "docs" / "governance-decisions.md"
        cls.readiness_path = cls.root / "docs" / "release-readiness.md"
        cls.checklist_path = cls.root / "docs" / "release-checklist.md"
        cls.changelog_path = cls.root / "CHANGELOG.md"
        cls.license_text = cls.license_path.read_text(encoding="utf-8")
        cls.citation = cls.citation_path.read_text(encoding="utf-8")
        cls.readme = cls.readme_path.read_text(encoding="utf-8")
        cls.governance = cls.governance_path.read_text(encoding="utf-8")
        cls.readiness = cls.readiness_path.read_text(encoding="utf-8")
        cls.checklist = cls.checklist_path.read_text(encoding="utf-8")
        cls.changelog = cls.changelog_path.read_text(encoding="utf-8")
        cls.governed_content = "\n".join(
            (
                cls.license_text,
                cls.citation,
                cls.readme,
                cls.governance,
                cls.readiness,
                cls.checklist,
                cls.changelog,
            )
        )

    def test_license_is_canonical_mit_with_approved_copyright(self) -> None:
        self.assertTrue(self.license_path.is_file())
        self.assertTrue(self.license_text.startswith("MIT License\n\n"))
        self.assertIn("Copyright (c) 2025 김철희", self.license_text)
        self.assertIn("Permission is hereby granted, free of charge", self.license_text)
        self.assertIn("THE SOFTWARE IS PROVIDED \"AS IS\"", self.license_text)
        self.assertNotRegex(self.license_text, r"(?i)non[- ]commercial")
        self.assertNotIn("dataset", self.license_text.lower())

    def test_citation_has_exactly_the_approved_repository_metadata(self) -> None:
        self.assertTrue(self.citation_path.is_file())
        self.assertNotIn("\t", self.citation)
        self.assertIn("cff-version: 1.2.0", self.citation)
        self.assertIn('message: "If you use this software, please cite it using the metadata in this file."', self.citation)
        self.assertIn('title: "Fruit Freshness Classification"', self.citation)
        self.assertIn("type: software", self.citation)
        self.assertIn('given-names: "Choelhui"', self.citation)
        self.assertIn('family-names: "Kim"', self.citation)
        self.assertIn('repository-code: "https://github.com/kimcheolhui9846/fruit-freshness-classification"', self.citation)
        self.assertIn("license: MIT", self.citation)
        top_level_keys = re.findall(r"(?m)^([a-z][a-z-]*):", self.citation)
        for key in ("cff-version", "message", "title", "type", "authors", "repository-code", "license"):
            self.assertEqual(top_level_keys.count(key), 1, key)
        self.assertEqual(len(re.findall(r"(?m)^  - given-names:", self.citation)), 1)
        self.assertEqual(len(re.findall(r"(?m)^    family-names:", self.citation)), 1)
        self.assertNotRegex(self.citation, r"\{[^}]+\}")
        self.assertNotRegex(self.citation, r"(?i)placeholder")

    def test_citation_omits_unapproved_identity_and_release_fields(self) -> None:
        self.assertNotRegex(
            self.citation,
            r"(?im)^\s*(?:email|affiliation|orcid|doi|version|date-released|identifiers|url|repository-artifact|commit|preferred-citation)\s*:",
        )
        self.assertNotRegex(self.citation, r"(?i)\b(?:journal|conference|paper)\b")
        self.assertNotRegex(self.citation, r"(?im)^\s*version\s*:")
        self.assertNotIn("@", self.citation)

    def test_documentation_separates_software_dataset_and_release_boundaries(self) -> None:
        self.assertIn("[LICENSE](LICENSE)", self.readme)
        self.assertIn("[CITATION.cff](CITATION.cff)", self.readme)
        self.assertIn("external Hugging Face dataset is governed separately", self.readme)
        self.assertIn("[dataset documentation](docs/dataset.md)", self.readme)
        self.assertIn("Dataset files are not included", self.readme)
        self.assertIn("MIT was explicitly selected during Phase 6.4.", self.governance)
        self.assertIn("Choelhui Kim", self.governance)
        self.assertIn("Dataset contents are not redistributed through this repository", self.governance)
        self.assertIn("Trained weights are not currently distributed", self.governance)
        self.assertIn("A version tag and GitHub Release remain pending.", self.readiness)
        self.assertIn("[ ] Create a Git tag.", self.checklist)
        self.assertIn("[ ] Create a GitHub prerelease or Release.", self.checklist)
        self.assertIn("Canonical three-fold training has not been run.", self.readiness)
        self.assertIn("No trained weights or benchmark-quality metrics are distributed.", self.changelog)

    def test_governed_files_are_portable_and_secret_free(self) -> None:
        self.assertNotRegex(self.governed_content, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.governed_content, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(self.governed_content, r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
        self.assertNotRegex(self.governed_content, r"(?m)[ \t]+$")


if __name__ == "__main__":
    unittest.main()
