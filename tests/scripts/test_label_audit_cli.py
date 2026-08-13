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
main = analyze_label_audit.main


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
    def test_main_refuses_reviewer_counts_other_than_two(self):
        # Drives main()'s own guard, not just argparse: both --key and
        # --output-dir are present and valid-shaped, so the only thing that
        # can raise SystemExit here is the "exactly two reviewers" check.
        with self.assertRaises(SystemExit):
            main([
                "--key", "k.json",
                "--reviewer", "a.csv",
                "--output-dir", "out",
            ])

        with self.assertRaises(SystemExit):
            main([
                "--key", "k.json",
                "--reviewer", "a.csv",
                "--reviewer", "b.csv",
                "--reviewer", "c.csv",
                "--output-dir", "out",
            ])

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

    def test_main_end_to_end_isolates_decision_from_model_predictions(self):
        import csv
        import json
        import numpy as np
        import tempfile
        from pathlib import Path

        entries = [
            {"position": 0, "source_index": 100, "group": "SUBJECT"},
            {"position": 1, "source_index": 101, "group": "SUBJECT"},
            {"position": 2, "source_index": 102, "group": "SUBJECT"},
            {"position": 3, "source_index": 103, "group": "SUBJECT"},
            {"position": 4, "source_index": 200, "group": "CONTROL"},
            {"position": 5, "source_index": 201, "group": "CONTROL"},
        ]
        key_payload = {
            "schema_version": 1,
            "review_set_count": len(entries),
            "entries": entries,
        }

        # Reviewer 1 clears the 0.15 threshold on the subject/control gap
        # (0.25 vs 0.0); reviewer 2 does not (0.0 vs 0.0). This fixes the
        # decision at SPLIT_OUTCOME regardless of what the model predicted.
        reviewer_1_rows = [
            (0, "ROTTEN"), (1, "FRESH"), (2, "FRESH"), (3, "FRESH"),
            (4, "ROTTEN"), (5, "ROTTEN"),
        ]
        reviewer_2_rows = [
            (0, "FRESH"), (1, "FRESH"), (2, "FRESH"), (3, "FRESH"),
            (4, "ROTTEN"), (5, "ROTTEN"),
        ]

        label_names = [
            "freshapples", "freshbanana", "freshcapsicum", "freshcucumber",
            "freshoranges", "freshpotato", "freshtomato", "rottenapples",
            "rottenbanana", "rottencapsicum", "rottencucumber", "rottenoranges",
            "rottenpotato", "rottentomato",
        ]
        development_indices = [100, 101, 102, 103, 200, 201]

        def write_csv(path, rows):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["position", "judgment"])
                for position, judgment in rows:
                    writer.writerow([f"{position:03d}", judgment])

        def run_with_predictions(tmp, predictions):
            key_path = Path(tmp) / "key.json"
            key_path.write_text(json.dumps(key_payload), encoding="utf-8")

            reviewer_1_path = Path(tmp) / "reviewer_1.csv"
            reviewer_2_path = Path(tmp) / "reviewer_2.csv"
            write_csv(reviewer_1_path, reviewer_1_rows)
            write_csv(reviewer_2_path, reviewer_2_rows)

            label_names_path = Path(tmp) / "label_names.json"
            label_names_path.write_text(json.dumps(label_names), encoding="utf-8")

            split_manifest_path = Path(tmp) / "split.json"
            split_manifest_path.write_text(
                json.dumps({"development_indices": development_indices}),
                encoding="utf-8",
            )

            predictions_path = Path(tmp) / "predictions.npz"
            np.savez(predictions_path, predictions=np.array(predictions, dtype=np.int64))

            output_dir = Path(tmp) / "output"

            exit_code = main([
                "--key", str(key_path),
                "--reviewer", str(reviewer_1_path),
                "--reviewer", str(reviewer_2_path),
                "--output-dir", str(output_dir),
                "--baseline-predictions", str(predictions_path),
                "--label-names", str(label_names_path),
                "--split-manifest", str(split_manifest_path),
            ])
            self.assertEqual(exit_code, 0)

            findings = json.loads(
                (output_dir / "label_audit_findings.json").read_text(encoding="utf-8")
            )
            disagreements = (output_dir / "label_audit_disagreements.csv").read_text(
                encoding="utf-8"
            )
            return findings, disagreements

        # Same reviewers, same key, same split -- only the stored model
        # predictions differ between the two runs.
        with tempfile.TemporaryDirectory() as tmp_a:
            findings_a, disagreements_a = run_with_predictions(
                tmp_a, [5, 5, 5, 5, 12, 12]
            )
        with tempfile.TemporaryDirectory() as tmp_b:
            findings_b, disagreements_b = run_with_predictions(
                tmp_b, [12, 12, 12, 12, 12, 12]
            )

        self.assertEqual(
            findings_a["decision"],
            {
                "outcome": "SPLIT_OUTCOME",
                "next_phase": (
                    "No phase selected automatically; owner decides after "
                    "reviewing disagreements"
                ),
                "clears_threshold": [True, False],
            },
        )
        # The property this task exists to protect: the decision is
        # mechanically independent of the model comparison.
        self.assertEqual(findings_a["decision"], findings_b["decision"])
        self.assertNotEqual(
            findings_a["baseline_model_comparison"],
            findings_b["baseline_model_comparison"],
        )
        self.assertEqual(disagreements_a, disagreements_b)
        self.assertIn("000,100,SUBJECT,ROTTEN,FRESH", disagreements_a.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
