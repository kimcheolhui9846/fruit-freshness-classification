"""Offline publication contract for the v0.2.0 post-holdout research milestone."""

from __future__ import annotations

from pathlib import Path
import unittest


class ReleaseV020ContractTests(unittest.TestCase):
    """Keep the v0.2.0 release materials truthful before and after publication."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.notes_path = cls.root / "docs" / "releases" / "v0.2.0.md"
        cls.notes = cls.notes_path.read_text(encoding="utf-8")
        cls.readiness = (cls.root / "docs" / "release-readiness.md").read_text(
            encoding="utf-8"
        )
        cls.changelog = (cls.root / "CHANGELOG.md").read_text(encoding="utf-8")

    def test_approved_identity_is_recorded(self) -> None:
        self.assertTrue(self.notes_path.is_file())
        self.assertIn(
            "# Fruit Freshness Classification v0.2.0 — Post-Holdout Research Milestone",
            self.notes,
        )
        self.assertIn("Approved release target: `v0.2.0`", self.notes)
        self.assertIn("Approved release type: `PRERELEASE`", self.notes)
        self.assertIn("Approved release date: `2026-08-16`", self.notes)
        self.assertIn("## Approved release target — v0.2.0", self.readiness)
        self.assertIn("### Release decision — v0.2.0", self.readiness)
        self.assertIn("## [0.2.0] - 2026-08-16", self.changelog)

    def test_v010_authorization_survives_unchanged(self) -> None:
        # v0.2.0 supersedes nothing. The earlier record is history and the
        # earlier tag stays immutable.
        self.assertIn("Authorized release target: `v0.1.0`", self.readiness)
        self.assertIn("## [0.1.0] - 2026-08-02", self.changelog)
        self.assertIn("`v0.1.0` stays published, protected, and immutable", self.readiness)

    def test_notes_are_portable_and_private(self) -> None:
        self.assertNotRegex(self.notes, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.notes, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(self.notes, r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
        self.assertNotRegex(self.notes, r"(?i)\bdoi\b")

    def test_notes_claim_no_artifact_distribution(self) -> None:
        lowered = self.notes.lower()
        self.assertNotIn("trained weights are attached", lowered)
        self.assertNotIn("dataset files are attached", lowered)
        # The artifact boundary is the one thing this release must not blur.
        self.assertIn("source-only prerelease", lowered)
        self.assertIn("zero model forward passes", lowered)

    def test_notes_claim_no_benchmark_or_generalization(self) -> None:
        self.assertNotRegex(self.notes, r"(?i)benchmark (?:result )?(?:has been )?reproduced")
        self.assertIn(
            "makes no benchmark, production-validation, or generalization claim",
            self.notes,
        )

    def test_notes_carry_the_constraining_findings(self) -> None:
        # A release note that reports only the flattering half of a research
        # programme is a marketing document.
        self.assertIn("No post-holdout candidate was advanced", self.notes)
        self.assertIn("CLOSED_BELOW_RESOLUTION", self.notes)
        self.assertIn("`freshpotato` is unstable, not merely weak", self.notes)
        self.assertIn(
            "its own measurements",
            self.notes,
        )

    def test_release_materials_are_well_formed(self) -> None:
        combined = "\n".join((self.notes, self.readiness, self.changelog))
        self.assertEqual(self.notes.count("```") % 2, 0)
        self.assertNotRegex(combined, r"(?m)[ \t]+$")
        self.assertNotIn("TODO", combined)
        self.assertNotIn("TBD", combined)


if __name__ == "__main__":
    unittest.main()
