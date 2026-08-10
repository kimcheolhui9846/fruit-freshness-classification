"""Offline tests for the Phase 9 post-holdout split freeze entry point."""

import contextlib
import hashlib
import io
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

FREEZE_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "freeze_postholdout_split.py"
FREEZE_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "phase92_freeze_postholdout_split",
    FREEZE_SCRIPT_PATH,
)
freeze_postholdout_split = importlib.util.module_from_spec(FREEZE_SCRIPT_SPEC)
sys.modules[FREEZE_SCRIPT_SPEC.name] = freeze_postholdout_split
FREEZE_SCRIPT_SPEC.loader.exec_module(freeze_postholdout_split)


class FreezePostHoldoutSplitTest(unittest.TestCase):
    def _module(self):
        return freeze_postholdout_split

    def test_parser_uses_the_tracked_manifest_default(self):
        module = self._module()

        args = module.build_parser().parse_args([])

        self.assertEqual(
            args.output,
            Path("configs/splits/deep3-postholdout-research-01.json"),
        )

    def test_manifest_records_only_relative_split_identity_and_hashes(self):
        module = self._module()
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        development = np.asarray([0, 1, 3, 4], dtype=np.int64)
        locked_test = np.asarray([2, 5], dtype=np.int64)

        manifest = module.build_split_manifest(
            labels=labels,
            class_names=["fresh", "rotten"],
            development_indices=development,
            locked_test_indices=locked_test,
            repository_sha="a" * 40,
            locked_test_fraction=1 / 3,
            split_seed=20260810,
        )

        expected_development_hash = hashlib.sha256(
            np.asarray(development, dtype="<i8").tobytes()
        ).hexdigest()
        expected_locked_hash = hashlib.sha256(
            np.asarray(locked_test, dtype="<i8").tobytes()
        ).hexdigest()
        self.assertEqual(manifest["experiment_id"], "deep3-postholdout-research-01")
        self.assertEqual(manifest["source_pool_size"], 6)
        self.assertEqual(manifest["development_count"], 4)
        self.assertEqual(manifest["locked_test_count"], 2)
        self.assertEqual(manifest["development_indices_sha256"], expected_development_hash)
        self.assertEqual(manifest["locked_test_indices_sha256"], expected_locked_hash)
        self.assertEqual(manifest["canonical_holdout_overlap"], 0)
        self.assertEqual(manifest["locked_test_status"], "FROZEN_UNOBSERVED_BY_MODEL")
        self.assertNotIn("images", manifest)
        self.assertNotIn("image_paths", manifest)

    def test_class_names_coerce_numpy_label_ids_to_python_integers(self):
        module = self._module()

        class LabelFeature:
            def int2str(self, value):
                if not isinstance(value, int):
                    raise AssertionError("label IDs must be Python integers")
                return {3: "fresh", 8: "rotten"}[value]

        self.assertTrue(
            hasattr(module, "class_names_from_raw_label_ids"),
            "The freeze script must normalize NumPy label IDs before ClassLabel.int2str().",
        )
        names = module.class_names_from_raw_label_ids(
            LabelFeature(),
            np.asarray([3, 8], dtype=np.int64),
        )

        self.assertEqual(names, ["fresh", "rotten"])
    def test_dataset_identity_mismatch_is_reported_as_blocked(self):
        module = self._module()
        self.assertTrue(
            hasattr(module, "DatasetIdentityMismatchError"),
            "The freeze script must classify a dataset identity mismatch explicitly.",
        )
        output = io.StringIO()
        with patch.object(
            module,
            "freeze_postholdout_split",
            side_effect=module.DatasetIdentityMismatchError("identity mismatch"),
        ), contextlib.redirect_stdout(output):
            exit_code = module.main(["--output", "configs/splits/test.json"])

        self.assertEqual(exit_code, 2)
        self.assertIn(
            "PHASE_9_2_SPLIT_STATUS: BLOCKED_DATA_IDENTITY_MISMATCH",
            output.getvalue(),
        )
    def test_write_manifest_refuses_an_existing_frozen_record(self):
        module = self._module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "split.json"
            module.write_split_manifest(output_path, {"schema_version": 1})
            with self.assertRaises(FileExistsError):
                module.write_split_manifest(output_path, {"schema_version": 1})


if __name__ == "__main__":
    unittest.main()