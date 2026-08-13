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

ANALYZE_LABEL_AUDIT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_label_audit.py"
)
ANALYZE_LABEL_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "phase95_analyze_label_audit",
    ANALYZE_LABEL_AUDIT_PATH,
)
analyze_label_audit = importlib.util.module_from_spec(ANALYZE_LABEL_AUDIT_SPEC)
sys.modules[ANALYZE_LABEL_AUDIT_SPEC.name] = analyze_label_audit
ANALYZE_LABEL_AUDIT_SPEC.loader.exec_module(analyze_label_audit)

analyze_parser = analyze_label_audit.build_parser
load_judgments = analyze_label_audit.load_judgments
model_agreement = analyze_label_audit.model_agreement


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


class AnalyzeLabelAuditCliTest(unittest.TestCase):
    def test_parser_requires_two_reviewer_files(self):
        parser = analyze_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--key", "k.json", "--reviewer", "a.csv"])

    def test_parser_accepts_exactly_two_reviewers(self):
        parser = analyze_parser()

        args = parser.parse_args([
            "--key", "k.json",
            "--reviewer", "owner.csv",
            "--reviewer", "assistant.csv",
            "--output-dir", "results/label-audit",
        ])

        self.assertEqual(len(args.reviewer), 2)

    def test_blank_judgment_rows_are_rejected(self):
        import io

        handle = io.StringIO("position,judgment\n000,FRESH\n001,\n")

        with self.assertRaises(ValueError):
            load_judgments(handle)

    def test_model_comparison_skips_undecidable_and_not_a_potato(self):
        import numpy as np
        import tempfile
        from pathlib import Path

        development_indices = np.array([10, 11, 12, 13], dtype=np.int64)
        entries = [
            {"position": 0, "source_index": 10, "group": "SUBJECT"},
            {"position": 1, "source_index": 11, "group": "SUBJECT"},
            {"position": 2, "source_index": 12, "group": "SUBJECT"},
            {"position": 3, "source_index": 13, "group": "CONTROL"},
        ]
        judgments = {0: "ROTTEN", 1: "UNDECIDABLE", 2: "NOT_A_POTATO", 3: "ROTTEN"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.npz"
            np.savez(path, predictions=np.array([12, 12, 12, 12], dtype=np.int64))
            result = model_agreement(path, development_indices, entries, judgments, 12)

        # Only position 0 is a FRESH/ROTTEN subject call.
        self.assertEqual(result["compared"], 1)
        self.assertAlmostEqual(result["agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
