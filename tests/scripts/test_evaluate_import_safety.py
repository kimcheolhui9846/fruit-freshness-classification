import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class EvaluateImportSafetyTest(unittest.TestCase):
    def test_import_has_no_dataset_requirement_or_filesystem_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(REPOSITORY_ROOT)
                if not existing
                else str(REPOSITORY_ROOT) + os.pathsep + existing
            )
            result = subprocess.run(
                [sys.executable, "-c", "import scripts.evaluate; print('import safe')"],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "import safe\n")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_canonical_help_command_neither_evaluates_nor_creates_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.evaluate", "--help"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        self.assertIn("--checkpoint-dir", result.stdout)
        self.assertNotIn("device:", result.stdout)


if __name__ == "__main__":
    unittest.main()