"""Offline contract for the approved but not-yet-executed Phase 9.3 baseline."""

from pathlib import Path
import unittest

from src.utils.config import validate_postholdout_baseline_config


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = ROOT / "configs" / "deep3_canonical.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"
BASELINE_DOCUMENT = ROOT / "docs" / "post-holdout-baseline.md"
REGISTRY = ROOT / "docs" / "experiment-registry.md"
PLAN = ROOT / "docs" / "post-holdout-research-plan.md"
GOVERNANCE = ROOT / "docs" / "governance-decisions.md"


class PostHoldoutBaselineContractTest(unittest.TestCase):
    def test_baseline_identity_and_recipe_equivalence_are_frozen(self):
        validation = validate_postholdout_baseline_config(CANONICAL_CONFIG, BASELINE_CONFIG)

        self.assertTrue(validation["recipe_equivalent"])
        self.assertEqual(
            validation["allowed_differences"]["post_holdout"]["experiment_id"],
            "deep3-postholdout-research-01-baseline",
        )

    def test_documents_authorize_only_development_cv_before_training(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (BASELINE_DOCUMENT, REGISTRY, PLAN, GOVERNANCE)
        )
        for token in (
            "OWNER_PHASE_9_3_APPROVAL:\nAPPROVED",
            "EXPERIMENT_ID:\ndeep3-postholdout-research-01-baseline",
            "ROLE:\nPOST_HOLDOUT_DEVELOPMENT_BASELINE",
            "BASELINE_RECIPE_EQUIVALENCE:\nPASS",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "CANONICAL_HOLDOUT_MODEL_ACCESS:\nNO",
            "BASELINE_ARTIFACT_PUBLICATION:\nLOCAL_ONLY",
            "BASELINE_EXECUTION_STATUS:\nNOT_YET_RUN",
            "PHASE_9_4:\nNOT STARTED",
        ):
            self.assertIn(token, combined)
        self.assertNotIn("COMPLETED_DEVELOPMENT_BASELINE", combined)


if __name__ == "__main__":
    unittest.main()