from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ReproducibilityContractTest(unittest.TestCase):
    """Offline checks that preserve the documented Phase 5.5 contract."""

    def test_direct_reproducibility_dependencies_are_pinned(self):
        requirements = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
        development = (REPOSITORY_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
        self.assertIn("datasets==5.0.1", requirements)
        self.assertIn("huggingface-hub==1.26.0", requirements)
        self.assertIn("jupyterlab==4.6.2", development)
        self.assertIn("ipykernel==7.2.0", development)

    def test_dataset_documentation_preserves_fixed_revision_and_safe_route(self):
        documentation = (REPOSITORY_ROOT / "docs" / "dataset.md").read_text(encoding="utf-8")
        self.assertIn("2077850adc575aa1e8d6029e6cd6cefe9e403a1c", documentation)
        self.assertIn("freshness_fruit.zip", documentation)
        self.assertIn("hf_hub_download", documentation)
        self.assertIn("ImageFolder", documentation)
        self.assertIn("No images are copied into the repository", documentation)

    def test_reproducibility_documentation_contains_canonical_commands(self):
        documentation = (REPOSITORY_ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8")
        self.assertIn("python -m venv", documentation)
        self.assertIn("--index-url https://download.pytorch.org/whl/cu124", documentation)
        self.assertIn("-m unittest discover -s tests -p", documentation)
        self.assertIn("python -m scripts.evaluate", documentation)

    def test_reproducibility_documentation_contains_no_machine_specific_path(self):
        documentation = (REPOSITORY_ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8")
        self.assertNotIn("C:\\Users\\", documentation)
        self.assertNotIn("C:/Users/", documentation)
        self.assertNotIn("fruit-freshness-repro-rerun-pinned-", documentation)

    def test_temporary_fixtures_are_not_presented_as_trained_models(self):
        documentation = (REPOSITORY_ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8").lower()
        self.assertIn("untrained cmt compatibility fixtures", documentation)
        self.assertIn("not model-quality", documentation)
        self.assertIn("trained-checkpoint evaluation | not run", documentation)

    def test_git_tracks_no_reproducibility_artifacts_or_freeze_snapshot(self):
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=REPOSITORY_ROOT, text=True, encoding="utf-8"
        ).splitlines()
        disallowed_suffixes = (".pt", ".pth", ".ckpt")
        disallowed_fragments = ("pip-freeze", ".venv/", "/venv/", "hf-cache", "huggingface")
        for path in tracked:
            normalized = path.replace("\\", "/").lower()
            self.assertFalse(normalized.endswith(disallowed_suffixes), path)
            self.assertFalse(any(fragment in normalized for fragment in disallowed_fragments), path)


if __name__ == "__main__":
    unittest.main()