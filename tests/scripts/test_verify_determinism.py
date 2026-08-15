"""Digest and comparison behaviour of the determinism verification CLI."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_determinism.py"

# tests/scripts shadows scripts on the import path, so the CLI is loaded by
# file location rather than by module name.
_spec = importlib.util.spec_from_file_location("verify_determinism_cli", SCRIPT)
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)


def _state(scale: float = 1.0) -> dict:
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "model_state_dict": {
            "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]) * scale,
            "layer.bias": torch.tensor([0.5, 0.25]) * scale,
        },
        "ema_state_dict": {"layer.weight": torch.tensor([[1.0, 2.0]]) * scale},
        "completed_fold_histories": [{"val_acc": [0.9 * scale, 0.95 * scale]}],
    }


def _write_run(directory: Path, scale: float = 1.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(_state(scale), directory / "training_state.pt")
    return directory


class DigestTest(unittest.TestCase):
    def test_digest_is_stable_across_insertion_order(self):
        first = {"b": torch.tensor([1.0]), "a": torch.tensor([2.0])}
        second = {"a": torch.tensor([2.0]), "b": torch.tensor([1.0])}

        # Two identical runs may build dicts in a different order; that must
        # not read as a determinism failure.
        self.assertEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )

    def test_digest_detects_a_one_ulp_change(self):
        base = torch.tensor([1.0, 2.0])
        nudged = base.clone()
        # The smallest representable float32 step. A literal like 2.0000001
        # would round back to 2.0 and the test would assert nothing.
        nudged[1] = torch.nextafter(nudged[1], torch.tensor(3.0))
        first = {"a": base}
        second = {"a": nudged}
        self.assertNotEqual(base[1].item(), nudged[1].item())

        self.assertNotEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )

    def test_digest_separates_shape_from_contents(self):
        first = {"a": torch.tensor([[1.0, 2.0]])}
        second = {"a": torch.tensor([[1.0], [2.0]])}

        # Same bytes, different shape. Hashing only the buffer would call
        # these equal.
        self.assertNotEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )

    def test_digest_distinguishes_keys_from_values(self):
        first = {"ab": torch.tensor([1.0]), "c": torch.tensor([2.0])}
        second = {"a": torch.tensor([1.0]), "bc": torch.tensor([2.0])}

        self.assertNotEqual(
            verify.digest_tensor_mapping(first),
            verify.digest_tensor_mapping(second),
        )


class CompareRunsTest(unittest.TestCase):
    def test_identical_runs_are_bit_exact(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = _write_run(Path(root) / "b")
            result = verify.compare_runs(first, second)

        self.assertTrue(result["bit_exact"])
        self.assertIsNone(result["first_mismatch"])

    def test_differing_runs_report_the_first_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a", scale=1.0)
            second = _write_run(Path(root) / "b", scale=1.5)
            result = verify.compare_runs(first, second)

        self.assertFalse(result["bit_exact"])
        self.assertEqual(result["first_mismatch"], "model_state_dict")

    def test_history_difference_alone_breaks_bit_exactness(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = Path(root) / "b"
            second.mkdir()
            state = _state()
            state["completed_fold_histories"] = [{"val_acc": [0.9, 0.94]}]
            torch.save(state, second / "training_state.pt")
            result = verify.compare_runs(first, second)

        # Identical weights with different recorded metrics would still mean
        # the runs diverged.
        self.assertFalse(result["bit_exact"])
        self.assertEqual(result["first_mismatch"], "completed_fold_histories")

    def test_missing_state_file_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = Path(root) / "b"
            second.mkdir()
            with self.assertRaises(FileNotFoundError):
                verify.compare_runs(first, second)


class CliTest(unittest.TestCase):
    def test_main_writes_a_record_and_returns_zero_when_bit_exact(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = _write_run(Path(root) / "b")
            report = Path(root) / "record.json"
            code = verify.main(
                ["--first", str(first), "--second", str(second), "--output", str(report)]
            )
            payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertTrue(payload["bit_exact"])

    def test_main_returns_nonzero_when_not_bit_exact(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a", scale=1.0)
            second = _write_run(Path(root) / "b", scale=2.0)
            report = Path(root) / "record.json"
            code = verify.main(
                ["--first", str(first), "--second", str(second), "--output", str(report)]
            )
            payload = json.loads(report.read_text(encoding="utf-8"))

        # A verification tool that exits 0 on failure is a tool nobody checks.
        self.assertNotEqual(code, 0)
        self.assertFalse(payload["bit_exact"])

    def test_main_refuses_to_overwrite_an_existing_record(self):
        with tempfile.TemporaryDirectory() as root:
            first = _write_run(Path(root) / "a")
            second = _write_run(Path(root) / "b")
            report = Path(root) / "record.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                verify.main(
                    [
                        "--first",
                        str(first),
                        "--second",
                        str(second),
                        "--output",
                        str(report),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
