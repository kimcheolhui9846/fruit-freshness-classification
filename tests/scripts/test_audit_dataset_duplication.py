"""Duplicate detection and contamination counting for the dataset audit."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_dataset_duplication.py"

# tests/scripts shadows scripts on the import path, so the CLI is loaded by
# file location rather than by module name.
_spec = importlib.util.spec_from_file_location("audit_duplication_cli", SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def _write(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class HashingTest(unittest.TestCase):
    def test_identical_bytes_hash_together_regardless_of_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write(root, "Train/freshpotato/a.png", b"same")
            b = _write(root, "Test/freshpotato/differently_named.png", b"same")

            self.assertEqual(audit.hash_file(a), audit.hash_file(b))

    def test_differing_bytes_hash_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = _write(root, "Train/freshpotato/a.png", b"one")
            b = _write(root, "Train/freshpotato/b.png", b"two")

            self.assertNotEqual(audit.hash_file(a), audit.hash_file(b))


class CollectionTest(unittest.TestCase):
    def test_copies_across_top_level_directories_group_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Train/freshpotato/a.png", b"same")
            _write(root, "Test/freshpotato/a.png", b"same")
            _write(root, "Train/freshpotato/b.png", b"other")
            groups = audit.collect_image_hashes(root)

        # The Train/Test layout is exactly how the source dataset stores its
        # copies, and it is the reason a row-wise split scatters them.
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(v) for v in groups.values()), [1, 2])

    def test_class_filter_excludes_other_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Train/freshpotato/a.png", b"kept")
            _write(root, "Train/freshokra/b.png", b"excluded")
            groups = audit.collect_image_hashes(root, classes={"freshpotato"})

        self.assertEqual(sum(len(v) for v in groups.values()), 1)

    def test_non_image_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Train/freshpotato/a.png", b"image")
            _write(root, "Train/freshpotato/notes.txt", b"image")
            groups = audit.collect_image_hashes(root)

        # Identical bytes in a text file must not be counted as a duplicate
        # image.
        self.assertEqual(sum(len(v) for v in groups.values()), 1)


class SummaryTest(unittest.TestCase):
    def test_counts_unique_images_and_extra_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Train/freshpotato/a.png", b"same")
            _write(root, "Test/freshpotato/a.png", b"same")
            _write(root, "Train/freshpotato/c.png", b"same")
            _write(root, "Train/freshpotato/b.png", b"other")
            summary = audit.duplicate_summary(audit.collect_image_hashes(root))

        self.assertEqual(summary["files"], 4)
        self.assertEqual(summary["unique_images"], 2)
        self.assertEqual(summary["duplicate_groups"], 1)
        self.assertEqual(summary["extra_copies"], 2)
        self.assertEqual(summary["extra_copies_per_class"], {"freshpotato": 2})

    def test_cross_class_duplicates_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Train/freshpotato/a.png", b"same")
            _write(root, "Train/rottenpotato/a.png", b"same")
            summary = audit.duplicate_summary(audit.collect_image_hashes(root))

        # Identical pixels under two labels is a worse problem than redundancy
        # and must not be folded into the same number.
        self.assertEqual(summary["cross_class_groups"], 1)

    def test_a_collection_without_duplicates_reports_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "Train/freshpotato/a.png", b"one")
            _write(root, "Train/freshpotato/b.png", b"two")
            summary = audit.duplicate_summary(audit.collect_image_hashes(root))

        self.assertEqual(summary["duplicate_groups"], 0)
        self.assertEqual(summary["extra_copies"], 0)
        self.assertEqual(summary["cross_class_groups"], 0)


class ContaminationTest(unittest.TestCase):
    def test_counts_evaluation_rows_that_repeat_a_training_row(self):
        result = audit.split_contamination(["a", "b", "c"], ["c", "d", "c"])

        # Two evaluation rows repeat one training image: the row count and the
        # distinct-image count are different questions and both are reported.
        self.assertEqual(result["contaminated_rows"], 2)
        self.assertEqual(result["distinct_images_on_both_sides"], 1)
        self.assertAlmostEqual(result["contaminated_fraction"], 2 / 3)

    def test_disjoint_splits_report_no_contamination(self):
        result = audit.split_contamination(["a", "b"], ["c", "d"])

        self.assertEqual(result["contaminated_rows"], 0)
        self.assertEqual(result["contaminated_fraction"], 0.0)

    def test_empty_evaluation_does_not_divide_by_zero(self):
        result = audit.split_contamination(["a"], [])

        self.assertEqual(result["contaminated_fraction"], 0.0)


class CliTest(unittest.TestCase):
    def test_main_writes_a_record_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _write(root, "Train/freshpotato/a.png", b"same")
            _write(root, "Test/freshpotato/a.png", b"same")
            out = Path(tmp) / "audit.json"
            code = audit.main(["--root", str(root), "--output", str(out)])
            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["extra_copies"], 1)

    def test_main_refuses_to_overwrite_an_existing_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _write(root, "Train/freshpotato/a.png", b"x")
            out = Path(tmp) / "audit.json"
            out.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit.main(["--root", str(root), "--output", str(out)])


if __name__ == "__main__":
    unittest.main()
