from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]

class PostHoldoutResearchPlanContractTests(unittest.TestCase):
    def test_post_holdout_plan_preserves_closed_holdout_boundary(self):
        plan = (ROOT / "docs" / "post-holdout-research-plan.md").read_text(encoding="utf-8")
        registry = (ROOT / "docs" / "experiment-registry.md").read_text(encoding="utf-8")
        combined = plan + registry
        for token in ("POST_HOLDOUT", "deep3-canonical-reference-01", "deep3-postholdout-research-01", "CLOSED_REFERENCE", "HISTORICAL_EVALUATION_ONLY", "FINAL_CLAIM_REQUIRES_NEW_UNTOUCHED_EVALUATION", "CANONICAL_HOLDOUT_CHECKPOINT_SELECTION:\nPROHIBITED", "ONE_PRIMARY_CHANGE_PER_INITIAL_EXPERIMENT", "NO_RESULT_CHERRY_PICKING", "PHASE_9_2:\nPROTOCOL_FROZEN", "APPROVED_PHASE_9_DATA_PROTOCOL:\nDEV_PLUS_LOCKED_TEST", "APPROVED_PHASE_9_SPLIT_SEED:\n20260810", "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL", "PHASE_9_3_TRAINING_AUTHORIZATION:\nNOT GRANTED"):
            self.assertIn(token, combined)
        self.assertIn("prohibited for tuning", plan)
        self.assertIn("CANONICAL_CONFIG_MUTATION:\nNO", plan)
        self.assertIn("Future experiments must use a new experiment identity", plan)
        self.assertIn("No Phase 9.1 artifact, checkpoint, split, training, evaluation, download, or publication is authorized.", plan)
        self.assertIn("APPROVED_PHASE_9_DEVELOPMENT_METRIC:\nMACRO_F1", plan)

if __name__ == '__main__':
    unittest.main()