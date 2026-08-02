"""Offline contracts for exact, non-destructive published-tag governance."""

from __future__ import annotations

import unittest
from pathlib import Path


class TagGovernanceContractTests(unittest.TestCase):
    """Keep published-tag governance exact, private, and release-neutral."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.document_path = cls.root / "docs" / "tag-governance.md"
        cls.document = cls.document_path.read_text(encoding="utf-8")

    def test_document_names_only_the_exact_published_tag_and_target(self) -> None:
        self.assertTrue(self.document_path.is_file())
        self.assertIn("The only protected tag condition is `refs/tags/v0.1.0`.", self.document)
        self.assertIn("`b38ebd36f4fa4f1fe012b957095db6dcbce20832`", self.document)
        self.assertIn("`v*`, are explicitly rejected", self.document)
        self.assertNotIn("refs/tags/v*", self.document)

    def test_approved_and_omitted_rule_boundaries_are_explicit(self) -> None:
        for fragment in (
            "blocks tag deletion and non-fast-forward updates",
            "no bypass actors",
            "Tag creation is not restricted",
            "Required status checks are absent",
            "No signed-tag rule or signed-commit rule is configured.",
            "Future tags are explicitly unaffected.",
            "`Protect main` branch ruleset remains unchanged.",
        ):
            self.assertIn(fragment, self.document)

    def test_recovery_and_non_destructive_verification_are_explicit(self) -> None:
        for fragment in (
            "do not force-update a published tag",
            "do not reuse an existing version number",
            "new approved version, such as `v0.1.1` or `v0.2.0`",
            "Do not delete `v0.1.0`",
            "do not use a destructive enforcement test",
        ):
            self.assertIn(fragment, self.document)

    def test_release_boundaries_and_portability_are_preserved(self) -> None:
        self.assertIn("prerelease engineering milestone", self.document)
        self.assertIn("Canonical training remains unverified", self.document)
        self.assertIn("trained weights remain undistributed", self.document)
        self.assertNotRegex(self.document, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.document, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(self.document, r"(?i)authorization\s*:")
        self.assertNotRegex(self.document, r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


if __name__ == "__main__":
    unittest.main()
