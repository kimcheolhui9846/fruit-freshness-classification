"""Offline tests for the Phase 9.3 deterministic CV identity entry point."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "freeze_postholdout_cv.py"
SPEC = importlib.util.spec_from_file_location("phase93_freeze_postholdout_cv", SCRIPT_PATH)
freeze_cv = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = freeze_cv
SPEC.loader.exec_module(freeze_cv)


class FreezePostHoldoutCvTest(unittest.TestCase):
    def test_parser_uses_the_tracked_cv_manifest_default(self):
        args = freeze_cv.build_parser().parse_args([])

        self.assertEqual(
            args.output,
            Path("configs/splits/deep3-postholdout-research-01-baseline-cv.json"),
        )

    def test_write_refuses_to_replace_a_frozen_cv_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv.json"
            freeze_cv.write_cv_manifest(path, {"schema_version": 1})
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                freeze_cv.write_cv_manifest(path, {"schema_version": 1})


if __name__ == "__main__":
    unittest.main()
