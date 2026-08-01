"""Durable public-documentation contracts for the repository README."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class ReadmeContractTests(unittest.TestCase):
    """Keep README claims, commands, links, and portability boundaries accurate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.readme_path = cls.root / "README.md"
        cls.readme = cls.readme_path.read_text(encoding="utf-8")

    def test_required_sections_and_commands_exist(self) -> None:
        self.assertTrue(self.readme_path.is_file())
        for fragment in (
            "# Fruit Freshness Classification",
            "## Quick start",
            "## Training",
            "## Evaluation",
            "## Testing",
            "## Continuous integration",
            "configs/deep3.toml",
            "Densu341/Fresh-rotten-fruit",
            "python -m scripts.train",
            "python -m scripts.evaluate",
            'python -m unittest discover -s tests -p "test_*.py" -v',
        ):
            self.assertIn(fragment, self.readme)

    def test_repository_relative_documentation_links_resolve(self) -> None:
        expected_documents = {
            "docs/environment.md",
            "docs/dataset.md",
            "docs/configuration.md",
            "docs/training.md",
            "docs/evaluation.md",
            "docs/reproducibility.md",
            "docs/ci.md",
        }
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", self.readme)
        linked_documents = {link for link in links if link.startswith("docs/")}
        self.assertTrue(expected_documents.issubset(linked_documents))
        for link in linked_documents:
            self.assertTrue((self.root / link).is_file(), link)

    def test_badge_targets_the_public_main_workflow(self) -> None:
        badge = (
            "https://github.com/kimcheolhui9846/fruit-freshness-classification/"
            "actions/workflows/ci.yml/badge.svg?branch=main"
        )
        workflow = (
            "https://github.com/kimcheolhui9846/fruit-freshness-classification/"
            "actions/workflows/ci.yml"
        )
        self.assertIn(badge, self.readme)
        self.assertIn(workflow, self.readme)

    def test_portability_and_truthful_limitations_are_explicit(self) -> None:
        self.assertNotRegex(self.readme, r"(?im)(?:^|[\s`(])[a-z]:[\\/]")
        self.assertNotRegex(self.readme, r"(?i)(ghp_|github_pat_|ctx7sk-|sk-[a-z0-9]{16,})")
        self.assertNotIn("0.097356", self.readme)
        self.assertNotIn("523/5,372", self.readme)
        self.assertIn("Full canonical three-fold training has not been run.", self.readme)
        self.assertIn("independent-machine reproduction is not.", self.readme)
        self.assertIn("Generic unlabeled image inference is not implemented.", self.readme)
        self.assertIn("CI is CPU-only and offline", self.readme)

    def test_checkpoint_notebook_tree_and_markdown_contracts(self) -> None:
        self.assertIn("`--checkpoint-dir` is required.", self.readme)
        self.assertIn("partial ensembles are rejected", self.readme)
        self.assertIn("`deep3.ipynb` is the active orchestration notebook.", self.readme)
        self.assertIn("historical experiment notebooks, not current entry points", self.readme)
        for path in ("configs/", "docs/", "scripts/", "src/", "tests/", "weights/"):
            self.assertIn(path, self.readme)
        self.assertEqual(self.readme.count("```") % 2, 0)
        headings = re.findall(r"(?m)^(#{1,6})\s+.+$", self.readme)
        self.assertEqual(headings.count("#"), 1)
        self.assertNotRegex(self.readme, r"(?m)[ \t]+$")

    def test_documentation_fixes_match_current_entry_points(self) -> None:
        configuration = (self.root / "docs" / "configuration.md").read_text(encoding="utf-8")
        training = (self.root / "docs" / "training.md").read_text(encoding="utf-8")
        self.assertIn("`scripts/train.py`", configuration)
        self.assertIn("`scripts/evaluate.py`", configuration)
        self.assertNotIn("future Phase 5.3 training entry point", configuration)
        self.assertIn("scripts/evaluate.py", training)
        self.assertNotIn("Deferred to Phase 5.4", training)


if __name__ == "__main__":
    unittest.main()
