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


LABEL_NAMES = [
    "freshapples", "freshbanana", "freshcapsicum", "freshcucumber",
    "freshoranges", "freshpotato", "freshtomato", "rottenapples",
    "rottenbanana", "rottencapsicum", "rottencucumber", "rottenoranges",
    "rottenpotato", "rottentomato",
]
FRESH_POTATO_INDEX = LABEL_NAMES.index("freshpotato")
ROTTEN_POTATO_INDEX = LABEL_NAMES.index("rottenpotato")


def _full_scale_entries(subject_source_offset=100, control_source_offset=10000):
    """347 SUBJECT + 150 CONTROL entries, matching the frozen protocol's counts.

    analyze_label_audit now enforces these exact counts against the sealed
    key, so any fixture exercising main() end-to-end needs full-scale data,
    not a handful of positions.
    """
    subject_count = build_label_audit_set.SUBJECT_COUNT
    control_count = build_label_audit_set.CONTROL_COUNT
    entries = [
        {"position": i, "source_index": subject_source_offset + i, "group": "SUBJECT"}
        for i in range(subject_count)
    ]
    entries += [
        {
            "position": subject_count + i,
            "source_index": control_source_offset + i,
            "group": "CONTROL",
        }
        for i in range(control_count)
    ]
    return entries


def _sealed_key_payload(entries):
    import hashlib

    import numpy as np

    return {
        "schema_version": 1,
        "review_set_count": len(entries),
        "presentation_indices_sha256": hashlib.sha256(
            np.asarray([e["source_index"] for e in entries], dtype="<i8").tobytes()
        ).hexdigest(),
        "entries": entries,
    }


def _write_judgment_csv(path, entries, judgment_for):
    """judgment_for(entry) -> category string, one row per entry."""
    import csv as csv_module

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["position", "judgment"])
        for entry in entries:
            writer.writerow([f"{entry['position']:03d}", judgment_for(entry)])


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

        # The key's directory keeps a reviewer working in review/ from
        # reaching it. This is a layout property, not proof of secrecy: the
        # seeds that determine group membership are published in the frozen
        # protocol, so anyone with repository access can recompute it.
        self.assertNotIn(str(outputs["review_dir"]), str(outputs["key_path"].parent))
        self.assertEqual(outputs["key_path"].name, "review_set_key.json")
        self.assertEqual(outputs["review_dir"].name, "review")

    def test_template_lives_in_the_reviewer_working_directory(self):
        # Reviewers copy judgment_template.csv to start their judgment file,
        # so it belongs alongside the images they are judging, not the key.
        outputs = partition_outputs("results/label-audit")

        self.assertEqual(outputs["template_path"].parent, outputs["review_dir"])

    def test_key_directory_is_disjoint_from_the_review_directory(self):
        # Neither directory may be an ancestor of the other: a reviewer
        # recursively copying or browsing review/ must never land on sealed/.
        outputs = partition_outputs("results/label-audit")
        review_parts = outputs["review_dir"].parts
        key_parts = outputs["key_path"].parent.parts

        self.assertNotEqual(review_parts, key_parts)
        self.assertNotEqual(key_parts[: len(review_parts)], review_parts)
        self.assertNotEqual(review_parts[: len(key_parts)], key_parts)

    def test_frozen_constants_match_the_protocol(self):
        # A seed or count changed here would silently audit a different set
        # of images than the frozen protocol describes, and nothing else in
        # the suite reads these module constants.
        self.assertEqual(build_label_audit_set.CONTROL_SAMPLE_SEED, 20260813)
        self.assertEqual(build_label_audit_set.PRESENTATION_ORDER_SEED, 20260813)
        self.assertEqual(build_label_audit_set.SUBJECT_COUNT, 347)
        self.assertEqual(build_label_audit_set.CONTROL_COUNT, 150)

    def test_counts_match_the_ones_analyze_label_audit_validates_against(self):
        # build_label_audit_set defines its own SUBJECT_COUNT/CONTROL_COUNT
        # literally; analyze_label_audit checks the sealed key against the
        # copies in src.datasets.label_audit. Two independently frozen copies
        # of the same numbers must never drift apart.
        from src.datasets.label_audit import CONTROL_COUNT, SUBJECT_COUNT

        self.assertEqual(build_label_audit_set.SUBJECT_COUNT, SUBJECT_COUNT)
        self.assertEqual(build_label_audit_set.CONTROL_COUNT, CONTROL_COUNT)


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
            result = model_agreement(path, development_indices, entries, judgments, 12, 5)

        # Only position 0 is a FRESH/ROTTEN subject call.
        self.assertEqual(result["compared"], 1)
        self.assertAlmostEqual(result["agreement"], 1.0)
        self.assertEqual(result["off_class"], 0)

    def test_model_comparison_reports_off_class_predictions_separately(self):
        # A prediction of some other produce entirely (here: class 3, neither
        # potato class) is not agreement or disagreement about freshness --
        # folding it into "agreement" as a mismatch would misreport roughly a
        # fifth of the real subject group, which is exactly what happened
        # when the diagnostic only checked predicted == rotten_label_index.
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
        judgments = {0: "ROTTEN", 1: "FRESH", 2: "FRESH", 3: "ROTTEN"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.npz"
            # rotten_label_index=12, fresh_label_index=5; position 1's
            # prediction (3) is neither -- an off-class prediction.
            np.savez(path, predictions=np.array([12, 3, 5, 12], dtype=np.int64))
            result = model_agreement(path, development_indices, entries, judgments, 12, 5)

        self.assertEqual(result["compared"], 2)
        self.assertEqual(result["off_class"], 1)
        self.assertAlmostEqual(result["agreement"], 1.0)

    def test_duplicate_reviewer_path_is_rejected(self):
        # The runbook's own command is two nearly identical --reviewer lines;
        # passing the same file twice must not fabricate a two-rater result.
        with self.assertRaises(SystemExit):
            main([
                "--key", "k.json",
                "--reviewer", "a.csv",
                "--reviewer", "a.csv",
                "--output-dir", "out",
            ])

    def test_duplicate_reviewer_path_is_rejected_even_when_spelled_differently(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "a.csv"
            csv_path.write_text("position,judgment\n000,FRESH\n", encoding="utf-8")
            (Path(tmp) / "sub").mkdir()
            alias = Path(tmp) / "sub" / ".." / "a.csv"

            # `Path(tmp) / "." / "a.csv"` normalizes away at construction time
            # and is byte-identical to csv_path, which would make this test an
            # exact duplicate of test_duplicate_reviewer_path_is_rejected and
            # never exercise the resolve()-based aliasing guard. Assert the
            # two spellings genuinely differ as strings before resolve().
            self.assertNotEqual(str(csv_path), str(alias))
            self.assertEqual(csv_path.resolve(), alias.resolve())

            with self.assertRaises(SystemExit):
                main([
                    "--key", "k.json",
                    "--reviewer", str(csv_path),
                    "--reviewer", str(alias),
                    "--output-dir", str(Path(tmp) / "out"),
                ])

    def test_rerunning_refuses_to_overwrite_existing_findings(self):
        # Re-running without a fresh --output-dir must not silently replace a
        # recorded outcome. The guard fires before the (missing) key is even
        # read, so a bad --key does not mask what's actually being tested.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            (output_dir / "label_audit_findings.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SystemExit):
                main([
                    "--key", str(Path(tmp) / "does_not_exist.json"),
                    "--reviewer", str(Path(tmp) / "a.csv"),
                    "--reviewer", str(Path(tmp) / "b.csv"),
                    "--output-dir", str(output_dir),
                ])

            self.assertEqual(
                (output_dir / "label_audit_findings.json").read_text(encoding="utf-8"),
                "{}",
            )

    def test_group_sizes_must_match_the_frozen_protocol(self):
        import json
        import tempfile
        from pathlib import Path

        entries = [
            {"position": 0, "source_index": 10, "group": "SUBJECT"},
            {"position": 1, "source_index": 20, "group": "CONTROL"},
        ]
        key_payload = _sealed_key_payload(entries)

        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "k.json"
            key_path.write_text(json.dumps(key_payload), encoding="utf-8")

            with self.assertRaises(SystemExit):
                main([
                    "--key", str(key_path),
                    "--reviewer", str(Path(tmp) / "a.csv"),
                    "--reviewer", str(Path(tmp) / "b.csv"),
                    "--output-dir", str(Path(tmp) / "out"),
                ])

    def test_review_set_count_field_must_match_entry_count(self):
        import json
        import tempfile
        from pathlib import Path

        entries = _full_scale_entries()
        key_payload = _sealed_key_payload(entries)
        key_payload["review_set_count"] = len(entries) - 1

        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "k.json"
            key_path.write_text(json.dumps(key_payload), encoding="utf-8")

            with self.assertRaises(SystemExit):
                main([
                    "--key", str(key_path),
                    "--reviewer", str(Path(tmp) / "a.csv"),
                    "--reviewer", str(Path(tmp) / "b.csv"),
                    "--output-dir", str(Path(tmp) / "out"),
                ])

    def test_tampered_presentation_hash_is_rejected(self):
        # entries carry source_index in position order, so the hash the build
        # script recorded is exactly recomputable. A key whose entries don't
        # match its own recorded hash must not be trusted silently.
        import json
        import tempfile
        from pathlib import Path

        entries = _full_scale_entries()
        key_payload = _sealed_key_payload(entries)
        key_payload["presentation_indices_sha256"] = "0" * 64

        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "k.json"
            key_path.write_text(json.dumps(key_payload), encoding="utf-8")

            with self.assertRaises(SystemExit):
                main([
                    "--key", str(key_path),
                    "--reviewer", str(Path(tmp) / "a.csv"),
                    "--reviewer", str(Path(tmp) / "b.csv"),
                    "--output-dir", str(Path(tmp) / "out"),
                ])

    def test_duplicate_position_in_a_reviewer_file_is_rejected(self):
        import io

        handle = io.StringIO("position,judgment\n000,FRESH\n000,ROTTEN\n")

        with self.assertRaises(ValueError):
            load_judgments(handle)

    def test_main_end_to_end_isolates_decision_from_model_predictions(self):
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        entries = _full_scale_entries()
        key_payload = _sealed_key_payload(entries)

        # Reviewer 1 clears the 0.15 threshold on the subject/control gap
        # (100/347 subject error vs 0/150 control error); reviewer 2 does not
        # (0 vs 0). This fixes the decision at SPLIT_OUTCOME regardless of
        # what the model predicted.
        def reviewer_1_call(entry):
            if entry["group"] == "SUBJECT":
                return "ROTTEN" if entry["position"] < 100 else "FRESH"
            return "ROTTEN"

        def reviewer_2_call(entry):
            return "FRESH" if entry["group"] == "SUBJECT" else "ROTTEN"

        development_indices = [e["source_index"] for e in entries]

        def run_with_predictions(tmp, predictions):
            key_path = Path(tmp) / "key.json"
            key_path.write_text(json.dumps(key_payload), encoding="utf-8")

            reviewer_1_path = Path(tmp) / "reviewer_1.csv"
            reviewer_2_path = Path(tmp) / "reviewer_2.csv"
            _write_judgment_csv(reviewer_1_path, entries, reviewer_1_call)
            _write_judgment_csv(reviewer_2_path, entries, reviewer_2_call)

            label_names_path = Path(tmp) / "label_names.json"
            label_names_path.write_text(json.dumps(LABEL_NAMES), encoding="utf-8")

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
            predictions_a = (
                [FRESH_POTATO_INDEX] * build_label_audit_set.SUBJECT_COUNT
                + [ROTTEN_POTATO_INDEX] * build_label_audit_set.CONTROL_COUNT
            )
            findings_a, disagreements_a = run_with_predictions(tmp_a, predictions_a)
        with tempfile.TemporaryDirectory() as tmp_b:
            predictions_b = [ROTTEN_POTATO_INDEX] * len(entries)
            findings_b, disagreements_b = run_with_predictions(tmp_b, predictions_b)

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

        # Findings surface per-group category counts, not just pooled rates
        # (the protocol's Outputs section commits to this breakdown).
        for score in findings_a["reviewer_scores"]:
            self.assertIn("subject_counts", score)
            self.assertIn("control_counts", score)
            self.assertEqual(
                sum(score["subject_counts"].values()), build_label_audit_set.SUBJECT_COUNT
            )
            self.assertEqual(
                sum(score["control_counts"].values()), build_label_audit_set.CONTROL_COUNT
            )

    def test_degenerate_kappa_serializes_as_null_not_nan(self):
        # cohen_kappa_score returns NaN for a degenerate agreement matrix
        # (here: both reviewers use exactly one category throughout).
        # json.dumps would emit a bare NaN token, which strict JSON parsers
        # reject, so the findings file must contain "null" instead.
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        entries = _full_scale_entries()
        key_payload = _sealed_key_payload(entries)

        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "key.json"
            key_path.write_text(json.dumps(key_payload), encoding="utf-8")

            reviewer_1_path = Path(tmp) / "reviewer_1.csv"
            reviewer_2_path = Path(tmp) / "reviewer_2.csv"
            _write_judgment_csv(reviewer_1_path, entries, lambda entry: "FRESH")
            _write_judgment_csv(reviewer_2_path, entries, lambda entry: "FRESH")

            label_names_path = Path(tmp) / "label_names.json"
            label_names_path.write_text(json.dumps(LABEL_NAMES), encoding="utf-8")

            split_manifest_path = Path(tmp) / "split.json"
            split_manifest_path.write_text(
                json.dumps({"development_indices": [e["source_index"] for e in entries]}),
                encoding="utf-8",
            )

            predictions_path = Path(tmp) / "predictions.npz"
            np.savez(
                predictions_path,
                predictions=np.array([FRESH_POTATO_INDEX] * len(entries), dtype=np.int64),
            )

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

            raw_text = (output_dir / "label_audit_findings.json").read_text(encoding="utf-8")

        self.assertNotIn("NaN", raw_text)
        findings = json.loads(raw_text)
        self.assertIsNone(findings["cohen_kappa"])


if __name__ == "__main__":
    unittest.main()
