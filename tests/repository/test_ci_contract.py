"""Static safety contract for the repository CI workflow."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class RepositoryCIContractTests(unittest.TestCase):
    """Keep repository CI deterministic, CPU-only, and non-production."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[2]
        cls.workflow_path = cls.repository_root / ".github" / "workflows" / "ci.yml"
        cls.workflow = cls.workflow_path.read_text(encoding="utf-8")

    def test_workflow_exists_and_has_required_triggers(self) -> None:
        self.assertTrue(self.workflow_path.is_file())
        self.assertRegex(self.workflow, r"(?m)^on:\s*$")
        self.assertRegex(self.workflow, r"(?m)^  push:\s*$")
        self.assertRegex(self.workflow, r"(?m)^      - main\s*$")
        self.assertRegex(self.workflow, r"(?m)^  pull_request:\s*$")
        self.assertRegex(self.workflow, r"(?m)^    branches:\s*$")
        self.assertRegex(self.workflow, r"(?m)^  workflow_dispatch:\s*$")
        self.assertNotIn("pull_request_target", self.workflow)
        self.assertNotIn("paths-ignore", self.workflow)

    def test_runner_matrix_is_cpu_only_and_cross_platform(self) -> None:
        self.assertIn("windows-latest", self.workflow)
        self.assertIn("ubuntu-latest", self.workflow)
        self.assertIn('python-version: ["3.12"]', self.workflow)
        self.assertIn("--index-url https://download.pytorch.org/whl/cpu", self.workflow)
        self.assertIn("assert not torch.cuda.is_available()", self.workflow)
        self.assertNotIn("cu124", self.workflow.lower())
        self.assertNotRegex(self.workflow, r"(?i)runs-on:\s*.*gpu")

    def test_security_controls_are_explicit(self) -> None:
        self.assertRegex(self.workflow, r"(?ms)^permissions:\s*\n\s+contents:\s+read\s*$")
        self.assertIn("persist-credentials: false", self.workflow)
        self.assertIn("fetch-depth: 30", self.workflow)
        self.assertNotIn("fetch-depth: 0", self.workflow)
        self.assertIn("timeout-minutes: 30", self.workflow)
        self.assertIn("cancel-in-progress: true", self.workflow)
        self.assertNotRegex(self.workflow, r"(?i)\bsecrets\b")
        self.assertNotRegex(self.workflow, r"(?i)upload-artifact")

    def test_actions_are_official_and_pinned_to_full_shas(self) -> None:
        actions = re.findall(r"(?m)^\s*uses:\s*([^@\s]+)@([^\s#]+)", self.workflow)
        self.assertGreaterEqual(len(actions), 2)
        for action, revision in actions:
            self.assertIn(action, {"actions/checkout", "actions/setup-python"})
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_validation_steps_are_present_and_offline(self) -> None:
        for variable in ("HF_HUB_OFFLINE: \"1\"", "HF_DATASETS_OFFLINE: \"1\"", "MPLBACKEND: Agg", "PYTHONIOENCODING: utf-8"):
            self.assertIn(variable, self.workflow)
        for command in (
            "python -m pip check",
            "python -m compileall src scripts tests",
            'python -m unittest discover -s tests -p "test_*.py" -v',
            "python -m scripts.train --help",
            "python -m scripts.evaluate --help",
            "git diff --exit-code",
            "git status --porcelain --untracked-files=all",
        ):
            self.assertIn(command, self.workflow)
        self.assertNotIn("load_fruit_freshness_dataset", self.workflow)
        self.assertNotIn("upload-artifact", self.workflow.lower())


if __name__ == "__main__":
    unittest.main()
