"""Offline documentation contracts for the Phase 7.1 main-protection workflow."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class BranchProtectionContractTests(unittest.TestCase):
    """Keep the documented protection model portable and SHA-preserving."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.document_path = cls.root / "docs" / "branch-protection.md"
        cls.document = cls.document_path.read_text(encoding="utf-8")

    def test_document_exists_and_names_the_protected_branch_and_checks(self) -> None:
        self.assertTrue(self.document_path.is_file())
        self.assertIn("default branch, `main`", self.document)
        self.assertIn("`ubuntu-latest / Python 3.12`", self.document)
        self.assertIn("`windows-latest / Python 3.12`", self.document)

    def test_fast_forward_only_workflow_is_explicit(self) -> None:
        self.assertIn("fast-forward-only", self.document)
        self.assertIn("`git merge --ff-only`", self.document)
        self.assertIn("Push `main` normally.", self.document)
        self.assertIn("preserves original commit SHAs", self.document)

    def test_approved_and_deferred_governance_boundaries_are_explicit(self) -> None:
        for fragment in (
            "Linear history: required.",
            "Force pushes: prohibited",
            "Branch deletion: prohibited.",
            "Pull requests: explicitly not required in Phase 7.1.",
            "Signed commits: explicitly deferred and not required.",
            "Bypass actors: absent.",
            "do not bypass failed CI",
            "Tag protection: deferred",
            "`v0.1.0` must not be moved, recreated, deleted, or force-updated.",
            "administrator recovery must be recorded in `SESSION_HANDOFF.md`",
        ):
            self.assertIn(fragment, self.document)

    def test_document_is_portable_private_and_scope_limited(self) -> None:
        self.assertNotRegex(self.document, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.document, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(self.document, r"(?i)authorization\s*:")
        self.assertIn("only repository-setting mutation in Phase 7.1", self.document)
        self.assertIn("does not create a classic branch-protection rule", self.document)
        self.assertIn("canonical training, trained-checkpoint evaluation, and benchmark reproduction remain incomplete", self.document)


if __name__ == "__main__":
    unittest.main()
