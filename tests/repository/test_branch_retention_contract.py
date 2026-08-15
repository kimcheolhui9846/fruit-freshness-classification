"""Offline documentation contracts for Phase 7.4 branch-retention governance."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class BranchRetentionContractTests(unittest.TestCase):
    """Keep branch-cleanup decisions explicit, non-destructive, and private."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.policy_path = cls.root / "docs" / "branch-retention-policy.md"
        cls.inventory_path = cls.root / "docs" / "branch-inventory.md"
        cls.policy = cls.policy_path.read_text(encoding="utf-8")
        cls.inventory = cls.inventory_path.read_text(encoding="utf-8")
        cls.documents = cls.policy + "\n" + cls.inventory

    def test_policy_and_complete_inventory_exist(self) -> None:
        self.assertTrue(self.policy_path.is_file())
        self.assertTrue(self.inventory_path.is_file())
        self.assertIn(
            "| Branch | Location | SHA | Main relationship | Unique commits | PR | References | Classification | Recommended action |",
            self.inventory,
        )
        classification_section = self.inventory.split(
            "## Complete Branch Classification\n", 1
        )[1].split("\n## Unique-Commit Appendix", 1)[0]
        rows = re.findall(r"^\| `[^`]+` \|", classification_section, flags=re.MULTILINE)
        self.assertEqual(len(rows), 24)

    def test_classifications_and_retention_window_are_explicit(self) -> None:
        for classification in (
            "MANDATORY_RETAIN",
            "RETAIN_RELEASE_AUDIT",
            "RETAIN_UNIQUE_COMMITS",
            "RETAIN_ACTIVE_OR_RECENT",
            "TEMPORARY_RETAIN",
            "SAFE_DELETE_CANDIDATE",
            "REVIEW_REQUIRED",
            "BLOCKED_UNVERIFIED",
        ):
            self.assertIn(f"`{classification}`", self.policy)
        self.assertIn("90 days after the latest published milestone or until the next milestone is published, whichever is later", self.policy)
        self.assertIn("`main` | local + remote", self.inventory)
        self.assertIn("`MANDATORY_RETAIN`", self.inventory)
        self.assertIn("`v0.1.0` is a protected tag, not a branch.", self.policy)

    def test_deletion_is_gated_and_non_authorizing(self) -> None:
        for fragment in (
            "Phase 7.4 does not authorize deletion.",
            "Phase 7.5 requires exact owner-approved lists.",
            "Local and remote deletion approvals must be separate.",
            "No deletion is authorized by this document.",
            "LOCAL_BRANCH_DELETE_CANDIDATES:",
            "REMOTE_BRANCH_DELETE_CANDIDATES:",
            "LOCAL_BRANCH_DELETE_CANDIDATES:\nNONE",
            "REMOTE_BRANCH_DELETE_CANDIDATES:\nNONE",
        ):
            self.assertIn(fragment, self.documents)
        self.assertIn("A branch with unique commits, an open or draft PR, or branch protection cannot be a safe-delete candidate.", self.policy)
        self.assertNotRegex(self.documents, r"(?im)^\s*git\s+branch\s+-[dD]\b")
        self.assertNotRegex(self.documents, r"(?im)^\s*git\s+push\s+origin\s+--delete\b")
        self.assertNotRegex(self.documents, r"(?im)^\s*git\s+update-ref\s+-d\b")

    def test_history_safety_and_project_boundaries_are_preserved(self) -> None:
        for fragment in (
            "Never rewrite `main`",
            "force push",
            "Never modify externally managed refs.",
            "Canonical training remains unverified",
            "trained weights remain undistributed",
            "model-performance claims remain unavailable",
        ):
            self.assertIn(fragment, self.policy)

    def test_documents_are_portable_and_private(self) -> None:
        self.assertNotRegex(self.documents, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.documents, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(self.documents, r"(?i)authorization\s*:")
        self.assertNotRegex(
            self.documents,
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        )


if __name__ == "__main__":
    unittest.main()
