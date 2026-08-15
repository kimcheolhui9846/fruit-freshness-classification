"""The frozen measurement floor and the rules derived from it."""

import statistics as st
import unittest

from src.utils.measurement_floor import (
    ADVANCED,
    BELOW_RESOLUTION,
    MACRO_F1_MEAN,
    MACRO_F1_STDEV,
    MDE_FRESHPOTATO_F1,
    MDE_MACRO_F1,
    MDE_TOP1,
    REGRESSED,
    REPLICATE_MACRO_F1,
    REPLICATE_TOP1,
    VALIDITY_ENVELOPE,
    classify_effect,
    within_validity_envelope,
)


class FrozenConstantTest(unittest.TestCase):
    def test_constants_match_the_frozen_protocol(self):
        # Pinned directly. A document can be edited without the suite
        # noticing unless the numbers are also asserted in code.
        self.assertEqual(MDE_MACRO_F1, 0.012177)
        self.assertEqual(MDE_TOP1, 0.001969)
        self.assertEqual(MDE_FRESHPOTATO_F1, 0.147833)
        self.assertEqual(VALIDITY_ENVELOPE, (0.892845, 0.917199))

    def test_constants_are_derived_from_the_recorded_replicates(self):
        # These are measured, not chosen. If someone edits a constant, the
        # arithmetic that produced it must still hold.
        self.assertEqual(len(REPLICATE_MACRO_F1), 3)
        self.assertAlmostEqual(st.mean(REPLICATE_MACRO_F1), MACRO_F1_MEAN, places=6)
        # Every constant was computed from the full-precision metrics and
        # then recorded at six decimals. Two consequences, both real:
        #
        #   - recomputing the stdev from the six-decimal replicates gives
        #     0.00608844 where the full-precision value is 0.00608853, so
        #     that step is checked to five places;
        #   - doubling a six-decimal constant is not the same as the
        #     six-decimal rounding of the doubled value. 2 * 0.006089 is
        #     0.012178, while 2 * 0.00608853 rounds to 0.012177. Asserting
        #     equality between the two stored constants would be asserting
        #     something rounding makes false.
        #
        # The chain that matters runs from the recorded replicates to the
        # MDE, and that one does survive to six places.
        self.assertAlmostEqual(st.stdev(REPLICATE_MACRO_F1), MACRO_F1_STDEV, places=5)
        self.assertAlmostEqual(2 * st.stdev(REPLICATE_MACRO_F1), MDE_MACRO_F1, places=6)
        self.assertAlmostEqual(2 * MACRO_F1_STDEV, MDE_MACRO_F1, places=5)
        self.assertAlmostEqual(2 * st.stdev(REPLICATE_TOP1), MDE_TOP1, places=6)

    def test_validity_envelope_is_the_mean_plus_and_minus_the_mde(self):
        low, high = VALIDITY_ENVELOPE
        self.assertAlmostEqual(low, MACRO_F1_MEAN - MDE_MACRO_F1, places=6)
        self.assertAlmostEqual(high, MACRO_F1_MEAN + MDE_MACRO_F1, places=6)


class ClassifyEffectTest(unittest.TestCase):
    def test_improvement_at_or_above_the_mde_advances(self):
        result = classify_effect(0.0 + MDE_MACRO_F1, 0.0, mde=MDE_MACRO_F1)

        # Constructed so the difference is exactly the MDE, with no float
        # slack that would let the boundary pass under either comparison.
        self.assertEqual(result["difference"], MDE_MACRO_F1)
        self.assertEqual(result["verdict"], ADVANCED)

    def test_improvement_just_below_the_mde_is_below_resolution(self):
        result = classify_effect(0.9102, 0.9012, mde=MDE_MACRO_F1)

        # The recorded loss-001 comparison: +0.0090 against a floor of
        # 0.012177.
        self.assertEqual(result["verdict"], BELOW_RESOLUTION)

    def test_decline_at_or_beyond_the_mde_is_a_regression(self):
        result = classify_effect(0.0 - MDE_MACRO_F1, 0.0, mde=MDE_MACRO_F1)

        # A measured decline is not an unmeasurable result.
        self.assertEqual(result["verdict"], REGRESSED)

    def test_small_decline_is_below_resolution_not_a_regression(self):
        result = classify_effect(0.9000, 0.9012, mde=MDE_MACRO_F1)

        self.assertEqual(result["verdict"], BELOW_RESOLUTION)

    def test_no_difference_is_below_resolution(self):
        result = classify_effect(0.9012, 0.9012, mde=MDE_MACRO_F1)

        self.assertEqual(result["verdict"], BELOW_RESOLUTION)

    def test_every_verdict_is_reachable_and_they_are_distinct(self):
        verdicts = {
            classify_effect(1.0, 0.0, mde=MDE_MACRO_F1)["verdict"],
            classify_effect(0.0, 0.0, mde=MDE_MACRO_F1)["verdict"],
            classify_effect(0.0, 1.0, mde=MDE_MACRO_F1)["verdict"],
        }
        # A partition with an unreachable branch is not a partition.
        self.assertEqual(verdicts, {ADVANCED, BELOW_RESOLUTION, REGRESSED})

    def test_result_reports_its_own_inputs(self):
        result = classify_effect(0.92, 0.90, mde=MDE_MACRO_F1)

        # A verdict without its inputs cannot be audited later.
        self.assertEqual(
            set(result), {"verdict", "difference", "mde", "candidate", "baseline"}
        )
        self.assertEqual(result["candidate"], 0.92)
        self.assertEqual(result["baseline"], 0.90)
        self.assertEqual(result["mde"], MDE_MACRO_F1)

    def test_non_positive_mde_is_rejected(self):
        for bad in (0.0, -0.01):
            with self.subTest(mde=bad):
                # A zero floor would advance every non-negative difference.
                with self.assertRaises(ValueError):
                    classify_effect(0.92, 0.90, mde=bad)


class ValidityEnvelopeTest(unittest.TestCase):
    def test_every_recorded_replicate_falls_inside(self):
        for value in REPLICATE_MACRO_F1:
            with self.subTest(value=value):
                # The envelope is built from these; if one fell outside, the
                # envelope would be describing something else.
                self.assertTrue(within_validity_envelope(value))

    def test_both_boundaries_are_inclusive(self):
        low, high = VALIDITY_ENVELOPE
        self.assertTrue(within_validity_envelope(low))
        self.assertTrue(within_validity_envelope(high))

    def test_values_outside_are_rejected(self):
        low, high = VALIDITY_ENVELOPE
        self.assertFalse(within_validity_envelope(low - 0.001))
        self.assertFalse(within_validity_envelope(high + 0.001))


if __name__ == "__main__":
    unittest.main()
