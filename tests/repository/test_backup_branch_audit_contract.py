"""Offline documentation contracts for Phase 7.5 backup-branch governance."""

from __future__ import annotations

import unittest
from pathlib import Path


class BackupBranchAuditContractTests(unittest.TestCase):
    """Keep backup-history review private, explicit, and non-destructive."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.audit_path = cls.root / "docs" / "backup-branch-audit.md"
        cls.preservation_path = cls.root / "docs" / "backup-branch-preservation.md"
        cls.audit = cls.audit_path.read_text(encoding="utf-8")
        cls.preservation = cls.preservation_path.read_text(encoding="utf-8")
        cls.documents = cls.audit + "\n" + cls.preservation

    def test_required_documents_and_identity_are_present(self) -> None:
        self.assertTrue(self.audit_path.is_file())
        self.assertTrue(self.preservation_path.is_file())
        self.assertIn(
            "backup/before-fruit-freshness-switch-20260729", self.documents
        )
        self.assertIn("Unique commits relative to `main`:** 15", self.audit)
        self.assertIn("The branch is local-only.", self.audit)

    def test_read_only_methodology_and_safety_prohibitions_are_explicit(self) -> None:
        for fragment in (
            "Git-object inspection only.",
            "not checked out",
            "not executed",
            "Backup branch deletion is prohibited.",
            "Public push is prohibited without owner approval.",
            "Merge into `main` is prohibited.",
            "Cherry-pick from this history is prohibited.",
            "History rewrite or sanitization is prohibited.",
            "Unreviewed bundle creation is prohibited.",
        ):
            self.assertIn(fragment, self.documents)

    def test_required_audit_sections_exist(self) -> None:
        for heading in (
            "## Commit Summary",
            "## Tree Summary",
            "## Notebook Findings",
            "## Difference from Main",
            "## Security Findings",
            "## Large Objects",
            "## Data and Artifact Findings",
            "## License Findings",
            "## Public Publication Gate",
            "## Audit Limitations",
        ):
            self.assertIn(heading, self.audit)
        self.assertIn("`RELATED_RESEARCH_HISTORY`", self.audit)
        self.assertIn("Execution performed: No.", self.audit)

    def test_primary_decision_and_owner_gate_are_explicit(self) -> None:
        self.assertRegex(
            self.preservation,
            r"PRIMARY_RECOMMENDATION:\s*\nREVIEW_REQUIRED",
        )
        self.assertIn("SECONDARY_RECOMMENDATION:", self.preservation)
        for field in (
            "APPROVED_BACKUP_ACTION:",
            "APPROVED_PUBLIC_DISCLOSURE:",
            "APPROVED_ARCHIVE_LOCATION_POLICY:",
            "APPROVED_ENCRYPTION_POLICY:",
            "APPROVED_RETENTION_PERIOD:",
            "APPROVED_BRANCH_DELETION_AFTER_PRESERVATION:",
        ):
            self.assertIn(field, self.preservation)
        self.assertIn("Phase 7.6 — Apply the Approved Backup Preservation Action", self.preservation)

    def test_documents_do_not_expose_private_values_or_machine_paths(self) -> None:
        self.assertNotRegex(
            self.documents,
            r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,}|hf_[a-z0-9]{20,})",
        )
        self.assertNotRegex(
            self.documents,
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        )
        self.assertNotRegex(self.documents, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")

    def test_documents_do_not_authorize_destructive_commands(self) -> None:
        self.assertNotRegex(
            self.documents,
            r"(?im)^\s*git\s+(?:branch\s+-[dD]|push\s+origin\s+--delete|merge\b|cherry-pick\b|reset\b|rebase\b|tag\b|bundle\b)",
        )

    def test_governance_and_project_boundaries_remain_unchanged(self) -> None:
        for fragment in (
            "`Protect main` remains unchanged.",
            "`v0.1.0` protection remains unchanged.",
            "no source or model-performance claims or changes",
        ):
            self.assertIn(fragment, self.documents)


if __name__ == "__main__":
    unittest.main()
