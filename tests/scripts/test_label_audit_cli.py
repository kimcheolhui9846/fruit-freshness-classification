import importlib.util
import sys
import unittest
from pathlib import Path

BUILD_LABEL_AUDIT_SET_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "build_label_audit_set.py"
)
BUILD_LABEL_AUDIT_SET_SPEC = importlib.util.spec_from_file_location(
    "phase95_build_label_audit_set",
    BUILD_LABEL_AUDIT_SET_PATH,
)
build_label_audit_set = importlib.util.module_from_spec(BUILD_LABEL_AUDIT_SET_SPEC)
sys.modules[BUILD_LABEL_AUDIT_SET_SPEC.name] = build_label_audit_set
BUILD_LABEL_AUDIT_SET_SPEC.loader.exec_module(build_label_audit_set)

build_parser = build_label_audit_set.build_parser
partition_outputs = build_label_audit_set.partition_outputs


class BuildLabelAuditSetCliTest(unittest.TestCase):
    def test_parser_requires_an_output_directory(self):
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parser_accepts_the_documented_arguments(self):
        parser = build_parser()

        args = parser.parse_args([
            "--split-manifest", "configs/splits/deep3-postholdout-research-01.json",
            "--output-dir", "results/label-audit",
        ])

        self.assertEqual(args.output_dir, "results/label-audit")

    def test_answer_key_is_written_outside_the_review_directory(self):
        outputs = partition_outputs("results/label-audit")

        # Blinding is enforced by layout: a reviewer opening review/ must not
        # be able to reach the key.
        self.assertNotIn(str(outputs["review_dir"]), str(outputs["key_path"].parent))
        self.assertEqual(outputs["key_path"].name, "review_set_key.json")
        self.assertEqual(outputs["review_dir"].name, "review")


if __name__ == "__main__":
    unittest.main()
