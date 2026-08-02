"""Offline documentation contracts for repository metadata and discoverability."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


APPROVED_DESCRIPTION = (
    "Reproducible PyTorch fruit freshness classification pipeline with CMT, "
    "config-driven training/evaluation, cross-platform CI, and documented "
    "engineering workflows."
)
APPROVED_TOPICS = (
    "pytorch",
    "computer-vision",
    "image-classification",
    "deep-learning",
    "machine-learning",
    "reproducibility",
    "mlops",
    "huggingface-datasets",
    "research-software",
    "fruit-freshness",
)


class RepositoryMetadataContractTests(unittest.TestCase):
    """Keep metadata documentation exact, portable, and release-neutral."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.document_path = cls.root / "docs" / "repository-metadata.md"
        cls.document = cls.document_path.read_text(encoding="utf-8")

    def test_approved_description_and_topics_are_exact_and_unique(self) -> None:
        self.assertTrue(self.document_path.is_file())
        self.assertEqual(self.document.count(APPROVED_DESCRIPTION), 1)

        documented_topics = tuple(
            re.findall(r"^- `([a-z0-9-]+)` - ", self.document, flags=re.MULTILINE)
        )
        self.assertEqual(documented_topics, APPROVED_TOPICS)
        self.assertEqual(len(documented_topics), len(set(documented_topics)))
        self.assertTrue(all(topic == topic.lower() for topic in documented_topics))
        self.assertTrue(all(" " not in topic for topic in documented_topics))

    def test_discoverability_policies_remain_truthful_and_deferred(self) -> None:
        for fragment in (
            "The repository homepage remains empty",
            "Custom social preview remains deferred.",
            "Profile pinning is a recommendation only:",
            "This Phase makes no profile-level change and records no pinning action.",
        ):
            self.assertIn(fragment, self.document)

        self.assertNotRegex(
            self.document,
            r"(?i)(?:profile pinning (?:was|has been)|repository (?:was|has been) pinned)",
        )

    def test_metadata_and_governance_boundaries_are_explicit(self) -> None:
        for fragment in (
            "Only description and topics are authorized live mutations in Phase 7.3.",
            "Repository visibility remains public and unchanged.",
            "The default branch remains `main`.",
            "Both GitHub rulesets remain unchanged.",
            "The published `v0.1.0` tag remains unchanged.",
            "does not change the homepage, visibility, default branch, merge settings, repository features, rulesets, tags, releases, or source files.",
        ):
            self.assertIn(fragment, self.document)

    def test_document_is_portable_private_and_claim_neutral(self) -> None:
        forbidden_claims = (
            r"(?i)trained (?:weights|models?) (?:are|were) (?:available|provided|included|released)",
            r"(?i)canonical training (?:is|was|has been) (?:complete|completed|finished)",
            r"(?i)(?:benchmark results?|accuracy) (?:are|were) (?:available|reported|achieved)",
            r"(?i)(?:production[- ]ready|ready for production)",
            r"(?i)generic inference (?:is|was) (?:available|implemented)",
        )
        for pattern in forbidden_claims:
            self.assertNotRegex(self.document, pattern)

        self.assertNotRegex(self.document, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.document, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotRegex(self.document, r"(?i)authorization\s*:")
        self.assertNotRegex(
            self.document,
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        )


if __name__ == "__main__":
    unittest.main()
