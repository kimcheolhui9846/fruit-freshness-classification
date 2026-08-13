"""Contract for the frozen Phase 9.6 decision rule."""

import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "phase96_apply_loss001_decision",
    Path(__file__).resolve().parents[2] / "scripts" / "apply_loss001_decision.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _metrics(macro_f1, top1):
    return {"metrics": {"macro_f1": macro_f1, "top1_accuracy": top1}}


class ApplyDecisionTest(unittest.TestCase):
    BASE = _metrics(0.9012, 0.9566)

    def test_clearing_both_thresholds_advances(self):
        result = module.apply_decision(self.BASE, _metrics(0.9112, 0.9466))

        self.assertEqual(result["outcome"], "ADVANCE")
        self.assertTrue(result["clears_macro_f1"])
        self.assertTrue(result["clears_top1_guardrail"])

    def test_macro_f1_just_below_the_threshold_does_not_advance(self):
        # One ten-thousandth under. The rule is a threshold, not a mood.
        result = module.apply_decision(self.BASE, _metrics(0.9111, 0.9600))

        self.assertEqual(result["outcome"], "NOT_ADVANCED")
        self.assertFalse(result["clears_macro_f1"])

    def test_the_top1_guardrail_can_veto_a_macro_f1_gain(self):
        result = module.apply_decision(self.BASE, _metrics(0.9300, 0.9400))

        self.assertEqual(result["outcome"], "NOT_ADVANCED")
        self.assertTrue(result["clears_macro_f1"])
        self.assertFalse(result["clears_top1_guardrail"])

    def test_not_advancing_names_h2_as_the_next_phase(self):
        result = module.apply_decision(self.BASE, _metrics(0.9012, 0.9566))

        self.assertIn("H2", result["next_phase"])

    def test_parser_requires_both_metric_files(self):
        with self.assertRaises(SystemExit):
            module.build_parser().parse_args(["--baseline-metrics", "a.json"])


if __name__ == "__main__":
    unittest.main()
