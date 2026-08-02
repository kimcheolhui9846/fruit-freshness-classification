"""Offline publication contracts for the approved engineering milestone."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class ReleasePublicationContractTests(unittest.TestCase):
    """Keep the v0.1.0 release materials truthful before and after publication."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.notes_path = cls.root / "docs" / "releases" / "v0.1.0.md"
        cls.readiness_path = cls.root / "docs" / "release-readiness.md"
        cls.checklist_path = cls.root / "docs" / "release-checklist.md"
        cls.changelog_path = cls.root / "CHANGELOG.md"
        cls.notes = cls.notes_path.read_text(encoding="utf-8")
        cls.readiness = cls.readiness_path.read_text(encoding="utf-8")
        cls.checklist = cls.checklist_path.read_text(encoding="utf-8")
        cls.changelog = cls.changelog_path.read_text(encoding="utf-8")

    def test_approved_identity_and_versioned_changelog_are_recorded(self) -> None:
        self.assertTrue(self.notes_path.is_file())
        self.assertIn(
            "# Fruit Freshness Classification v0.1.0 — Engineering Milestone",
            self.notes,
        )
        self.assertIn("Approved release target: `v0.1.0`", self.notes)
        self.assertIn("Approved release date: `2026-08-02`", self.notes)
        self.assertIn("Approved release type: `PRERELEASE`", self.readiness)
        self.assertIn("## [Unreleased]", self.changelog)
        self.assertIn("## [0.1.0] - 2026-08-02", self.changelog)

    def test_notes_are_portable_private_and_scope_limited(self) -> None:
        self.assertNotRegex(self.notes, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.notes, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(
            self.notes,
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        )
        self.assertNotRegex(self.notes, r"(?i)\bdoi\b")
        self.assertNotIn("trained weights are attached", self.notes.lower())
        self.assertNotIn("dataset files are attached", self.notes.lower())
        self.assertNotRegex(self.notes, r"(?i)canonical three-fold training (?:is )?completed")
        self.assertNotRegex(self.notes, r"(?i)benchmark (?:result )?(?:has been )?reproduced")
        self.assertNotIn("0.097356", self.notes)
        self.assertNotIn("523/5,372", self.notes)

    def test_notes_link_to_usage_and_separate_release_scopes(self) -> None:
        self.assertIn("[README](../../README.md)", self.notes)
        self.assertIn("engineering and reproducibility milestone", self.readiness)
        self.assertIn("not a trained-model benchmark release", self.readiness)
        for incomplete_item in (
            "[ ] Canonical training completed.",
            "[ ] Trained checkpoints produced.",
            "[ ] Trained evaluation reproduced.",
            "[ ] Benchmark result validated.",
        ):
            self.assertIn(incomplete_item, self.checklist)

    def test_release_materials_are_well_formed(self) -> None:
        combined = "\n".join((self.notes, self.readiness, self.checklist, self.changelog))
        self.assertEqual(self.notes.count("```") % 2, 0)
        self.assertNotRegex(combined, r"(?m)[ \t]+$")
        self.assertNotIn("TODO", combined)
        self.assertNotIn("TBD", combined)


if __name__ == "__main__":
    unittest.main()
