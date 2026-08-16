"""Offline contract for the frozen Phase 9.9 negative methodological result."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-stability-measurement-limit.md"


class StabilityLimitContractTest(unittest.TestCase):
    def test_protocol_is_frozen_and_consumed_no_gpu(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "ROLE:\nNEGATIVE_METHODOLOGICAL_RESULT",
            "TRAINING_RUN_COUNT:\n0",
            "GPU_HOURS:\n0",
            "APPROVED_TRAINING_RUN_COUNT:\n0",
            "APPROVED_INTERVENTION_TEST:\nNO",
        ):
            self.assertIn(token, document)

    def test_execution_status_is_exactly_one_known_state(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        states = [
            state
            for state in ("NOT_YET_RUN", "IN_PROGRESS", "COMPLETED", "STOPPED")
            if f"EXECUTION_STATUS:\n{state}" in document
        ]
        self.assertEqual(
            len(states), 1, f"expected exactly one execution status, got {states}"
        )

    def test_the_refuting_calibration_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The table is the phase's entire output. Without it this document is
        # an opinion.
        for token in (
            "NULL_PAIRS_TESTED:\n6",
            "NULL_PAIRS_SIGNIFICANT_AT_0.05:\n4",
            "OUTCOME:\nPER_IMAGE_TESTING_DOES_NOT_ESCAPE_THE_MEASUREMENT_FLOOR",
        ):
            self.assertIn(token, document)

    def test_the_calibrations_own_limits_are_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Six non-independent pairs is a weak estimate of a rate, and the
        # document must not present 0.67 as precise.
        self.assertIn("the pairs are not independent of one another", document)
        self.assertIn('should be read as "far above nominal"', document)

    def test_the_effective_sample_size_argument_is_stated(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn(
            "The effective sample size is the number of runs, not the number "
            "of images",
            document,
        )

    def test_the_repeated_pattern_is_named(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Recorded so a fourth attempt at a finer-grained metric is not made
        # by default.
        self.assertIn("The same wall, three times", document)

    def test_the_descriptive_analysis_cannot_be_promoted(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "DESCRIPTIVE_STATUS:\nEXPLORATORY_DESCRIPTIVE",
            "MAY_ADVANCE_A_CANDIDATE:\nNO",
            "MAY_SUPPORT_A_CLAIM:\nNO",
        ):
            self.assertIn(token, document)

    def test_the_untested_hypothesis_is_marked_untested(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A recorded hypothesis that loses its "untested" marking becomes a
        # finding nobody verified.
        self.assertIn("Recorded hypothesis, untested", document)
        self.assertIn("It is not evidence that removing `ColorJitter` would help", document)

    def test_earlier_verdicts_are_carried_unchanged(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOSS001_VERDICT:\nNOT_ADVANCED, unchanged and not re-scored",
            "H1_STATUS:\nCLOSED_BELOW_RESOLUTION, unchanged",
            "DETERMINISTIC_BASELINE:\n0.901891, unchanged",
        ):
            self.assertIn(token, document)

    def test_the_open_question_is_not_claimed_closed(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The phase closes a method, not the science.
        self.assertIn("Not closed:", document)
        self.assertIn(
            "it says this project cannot tell whether a given change improved it",
            document,
        )

    def test_phase_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
        ):
            self.assertIn(token, document)

        for forbidden in (
            "APPROVED_TRAINING_RUN_COUNT:\n1",
            "APPROVED_INTERVENTION_TEST:\nYES",
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
