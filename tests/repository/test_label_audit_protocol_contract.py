"""Offline contract for the frozen, not-yet-executed Phase 9.5 label quality audit."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-label-audit-protocol.md"
REGISTRY = ROOT / "docs" / "experiment-registry.md"
GOVERNANCE = ROOT / "docs" / "governance-decisions.md"
PLAN = ROOT / "docs" / "post-holdout-research-plan.md"


class LabelAuditProtocolContractTest(unittest.TestCase):
    def test_protocol_is_frozen_before_any_review(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "AUDIT_PROTOCOL_STATUS:\nFROZEN",
            "AUDIT_EXECUTION_STATUS:\nCOMPLETED",
            "AUDIT_OUTCOME:\nDEFECT_NOT_CONFIRMED",
            "EXPERIMENT_ID:\ndeep3-postholdout-research-01-label-audit",
            "ROLE:\nDEVELOPMENT_LABEL_QUALITY_AUDIT",
        ):
            self.assertIn(token, document)

    def test_review_set_and_control_are_fixed_in_advance(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The control group is what makes the subject error rate interpretable,
        # and a fixed seed is what stops it being reselected after the fact.
        for token in (
            "SUBJECT_COUNT:\n347",
            "CONTROL_COUNT:\n150",
            "REVIEW_SET_COUNT:\n497",
            "CONTROL_SAMPLE_SEED:\n20260813",
            "PRESENTATION_ORDER_SEED:\n20260813",
        ):
            self.assertIn(token, document)

    def test_every_judgment_category_is_defined(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for category in ("FRESH", "ROTTEN", "NOT_A_POTATO", "UNDECIDABLE"):
            self.assertIn(f"**`{category}`**", document)

    def test_decision_rule_is_stated_with_a_numeric_threshold(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A rule without a pre-committed threshold is decorative.
        self.assertIn("15 percentage points", document)
        self.assertIn("## Frozen decision rule", document)

    def test_execution_did_not_rewrite_the_frozen_method(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Recording an outcome must not become an opportunity to restate the
        # method that produced it. Every value the rule depends on stands.
        for token in (
            "count(ROTTEN or NOT_A_POTATO) / 347",
            "count(FRESH or NOT_A_POTATO) / 150",
            "MATERIAL_DIFFERENCE_THRESHOLD:\n15 percentage points",
            "CONTROL_SAMPLE_SEED:\n20260813",
            "PRESENTATION_ORDER_SEED:\n20260813",
        ):
            self.assertIn(token, document)
        self.assertNotIn("AUDIT_EXECUTION_STATUS:\nNOT_YET_RUN", document)

    def test_execution_record_carries_its_deviations(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # The two-reviewer standard was not met. A record that reported the
        # outcome without that caveat would overstate what the audit shows.
        self.assertIn("Recorded deviations", document)
        self.assertIn("two-reviewer standard written into this protocol was\nnot met", document)

    def test_audit_never_touches_the_locked_test_or_relabels_anything(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_INSPECTION:\nNO",
            "APPROVED_LOCKED_TEST_INSPECTION:\nNO",
            "APPROVED_RELABELING:\nNO",
            "APPROVED_IMAGE_PUBLICATION:\nNO",
            "LOCKED_TEST_LABEL_AUDIT:\nDEFERRED_TO_FINAL_EVALUATION",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
        ):
            self.assertIn(token, document)

        for forbidden in (
            "LOCKED_TEST_INSPECTION:\nYES",
            "APPROVED_RELABELING:\nYES",
            "APPROVED_IMAGE_PUBLICATION:\nYES",
            "MODEL_TRAINING:\nYES",
            "MODEL_INFERENCE:\nYES",
        ):
            self.assertNotIn(forbidden, document)

    def test_reordering_is_recorded_across_the_governance_documents(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in (REGISTRY, GOVERNANCE, PLAN)
        )

        # Deviating from the pre-registered hypothesis order must leave a trace.
        for token in (
            "PHASE_9_5:\nLABEL_AUDIT_PROTOCOL_FROZEN",
            "PHASE_9_6:\nH1_LOSS_AND_CLASS_IMBALANCE",
            "postholdout-label-audit-protocol.md",
        ):
            self.assertIn(token, combined)
        self.assertNotIn("PHASE_9_5:\nNOT STARTED", combined)
        # The audit selected H1; it did not authorize running it, and it never
        # authorized touching a label.
        for forbidden in (
            "APPROVED_RELABELING:\nYES",
            "LOCKED_TEST_INSPECTION:\nYES",
            "PHASE_9_6_EXECUTION:\nAUTHORIZED",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn("LABELS_MODIFIED:\n0", combined)


if __name__ == "__main__":
    unittest.main()
