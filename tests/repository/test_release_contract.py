"""Durable truthfulness contracts for release-audit documentation."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class ReleaseDocumentationContractTests(unittest.TestCase):
    """Keep release materials truthful before a future owner-approved release."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.release_readiness = cls.root / "docs" / "release-readiness.md"
        cls.release_checklist = cls.root / "docs" / "release-checklist.md"
        cls.governance = cls.root / "docs" / "governance-decisions.md"
        cls.changelog = cls.root / "CHANGELOG.md"
        cls.paths = (
            cls.release_readiness,
            cls.release_checklist,
            cls.governance,
            cls.changelog,
        )
        cls.contents = {
            path: path.read_text(encoding="utf-8") for path in cls.paths
        }
        cls.combined = "\n".join(cls.contents.values())

    def test_required_release_documents_exist(self) -> None:
        for path in self.paths:
            self.assertTrue(path.is_file(), path)

    def test_release_scope_preserves_model_evidence_boundaries(self) -> None:
        readiness = self.contents[self.release_readiness]
        self.assertIn("engineering and reproducibility milestone", readiness)
        self.assertIn("not a trained-model benchmark release", readiness)
        self.assertIn("Canonical three-fold training has not been run.", readiness)
        self.assertIn("No trained weights or benchmark-quality metrics are included.", readiness)
        self.assertNotIn("0.097356", self.combined)
        self.assertNotIn("523/5,372", self.combined)

    def test_license_and_citation_remain_pending(self) -> None:
        governance = self.contents[self.governance]
        checklist = self.contents[self.release_checklist]
        self.assertIn("No repository license has been selected or added in Phase 6.3.", governance)
        self.assertRegex(governance, r"`?CITATION\.cff`? remains pending")
        self.assertIn("No `CITATION.cff` was created.", checklist)
        self.assertIn("No software license was selected or added.", checklist)

    def test_no_version_is_presented_as_released(self) -> None:
        changelog = self.contents[self.changelog]
        self.assertIn("## Unreleased", changelog)
        self.assertNotRegex(changelog, r"(?m)^##\s+v?\d+\.\d+\.\d+")
        self.assertIn("Do **not** create a tag yet.", self.contents[self.release_readiness])

    def test_governance_recommendations_are_explicitly_non_operational(self) -> None:
        readiness = self.contents[self.release_readiness]
        governance = self.contents[self.governance]
        self.assertIn("Recommendations only; no branch protection or ruleset changes were made.", readiness)
        self.assertIn("No branch protection, ruleset, repository metadata, tag, or GitHub Release was changed", governance)
        self.assertIn("No Git tag or GitHub Release was created.", self.contents[self.release_checklist])

    def test_release_links_resolve(self) -> None:
        for path, content in self.contents.items():
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
                if "://" in link or link.startswith("#"):
                    continue
                target = link.split("#", maxsplit=1)[0]
                self.assertTrue((path.parent / target).resolve().is_file(), (path, link))

    def test_release_documents_are_portable_and_well_formed(self) -> None:
        self.assertNotRegex(self.combined, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.combined, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertEqual(self.combined.count("```") % 2, 0)
        self.assertNotRegex(self.combined, r"(?m)[ \t]+$")
        for content in self.contents.values():
            headings = re.findall(r"(?m)^(#{1,6})\s+.+$", content)
            self.assertEqual(headings.count("#"), 1)


if __name__ == "__main__":
    unittest.main()
