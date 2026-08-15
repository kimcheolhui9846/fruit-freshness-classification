"""Offline contract for the frozen Phase 9.8 measurement floor protocol."""

from pathlib import Path
import unittest

from src.utils.measurement_floor import (
    MDE_FRESHPOTATO_F1,
    MDE_MACRO_F1,
    MDE_TOP1,
    VALIDITY_ENVELOPE,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-measurement-floor-protocol.md"


class MeasurementFloorContractTest(unittest.TestCase):
    def test_protocol_is_frozen(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "SEED:\n20260815",
            "DETERMINISM_LEVEL:\nA_STRICT",
            "TRAINING_RUN_COUNT:\n1",
            "APPROVED_TRAINING_RUN_COUNT:\n1",
            "APPROVED_MDE_FRAMEWORK:\nYES",
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

    def test_code_constants_match_the_document(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A number pinned in prose and a number used in code can drift apart
        # silently. Assert they are the same number.
        self.assertIn(f"MDE_MACRO_F1:\n{MDE_MACRO_F1}", document)
        self.assertIn(f"MDE_TOP1:\n{MDE_TOP1}", document)
        self.assertIn(f"MDE_FRESHPOTATO_F1:\n{MDE_FRESHPOTATO_F1}", document)
        low, high = VALIDITY_ENVELOPE
        self.assertIn(f"VALIDITY_ENVELOPE:\n{low} to {high}", document)

    def test_h1_closure_and_its_basis_are_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "H1_STATUS:\nCLOSED_BELOW_RESOLUTION",
            "H1_CLOSURE_BASIS:\n71 to 212 GPU hours required to resolve the observed effect",
            "LOSS001_VERDICT:\nNOT_ADVANCED, unchanged and not re-scored",
        ):
            self.assertIn(token, document)

        # "Exhausted" is the claim the evidence does not support and the one
        # most likely to creep back in.
        self.assertNotIn("H1_STATUS:\nEXHAUSTED", document)

    def test_variance_decomposition_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The finding that reorganized the phase.
        self.assertIn("90.56%", document)
        self.assertIn("the class the research was trying to improve", document)

    def test_seeding_does_not_lower_the_floor_is_stated(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn(
            "Seeding makes a run reproducible. It does not make the outcome "
            "less variable across seeds.",
            document,
        )

    def test_run_duration_basis_is_measured_not_assumed(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The cost table drives the H1 closure, so its per-run figure must
        # name the run it came from.
        self.assertIn("530.98 minutes", document)
        self.assertIn("8.85", document)

    def test_diagnostic_cannot_be_promoted_to_a_result(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "DIAGNOSTIC_STATUS:\nEXPLORATORY_DESCRIPTIVE",
            "DIAGNOSTIC_MAY_ADVANCE_A_CANDIDATE:\nNO",
            "DIAGNOSTIC_MAY_SUPPORT_A_CLAIM:\nNO",
        ):
            self.assertIn(token, document)

    def test_phase_9_9_is_registered_but_not_authorized(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("PHASE_9_9_STATUS:\nREGISTERED_NOT_DESIGNED", document)
        self.assertIn("PHASE_9_9_AUTHORIZED:\nNO", document)
        self.assertNotIn("PHASE_9_9_AUTHORIZED:\nYES", document)

    def test_phase_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_LOSS001_RERUN:\nNO",
        ):
            self.assertIn(token, document)

        for forbidden in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "APPROVED_WEIGHT_PUBLICATION:\nYES",
            "APPROVED_LOSS001_RERUN:\nYES",
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
