"""Offline contract for the executed Phase 9.3 baseline and its closed boundaries."""

import hashlib
import json
from pathlib import Path
import unittest

from src.utils.config import validate_postholdout_baseline_config


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = ROOT / "configs" / "deep3_canonical.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"
BASELINE_DOCUMENT = ROOT / "docs" / "post-holdout-baseline.md"
CV_MANIFEST = ROOT / "configs" / "splits" / "deep3-postholdout-research-01-baseline-cv.json"
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

    def test_documents_record_a_development_only_baseline_execution(self):
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
            "EXPERIMENT_STATUS:\nCOMPLETED_DEVELOPMENT_BASELINE",
            "BASELINE_EXECUTION_STATUS:\nCOMPLETED",
            "PHASE_9_4:\nBASELINE_EXECUTED",
            "PHASE_9_5:\nNOT STARTED",
        ):
            self.assertIn(token, combined)
        self.assertNotIn("PHASE_9_4:\nNOT STARTED", combined)
        # The execution status advanced exactly once and never reverts.
        self.assertNotIn("BASELINE_EXECUTION_STATUS:\nNOT_YET_RUN", combined)

    def test_completed_baseline_never_widens_into_evaluation_or_publication(self):
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (BASELINE_DOCUMENT, REGISTRY, PLAN, GOVERNANCE)
        )
        # Training approval is spent. It must not become locked-test access,
        # canonical-holdout reuse, or any publication of binary artifacts.
        for forbidden in (
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
            "CANONICAL_HOLDOUT_MODEL_ACCESS:\nYES",
            "LOCKED_TEST_EVALUATION:\nYES",
            "CANONICAL_HOLDOUT_REEVALUATION:\nYES",
            "BASELINE_ARTIFACT_PUBLICATION:\nPUBLISHED",
            "BINARY_PUBLICATION:\nYES",
            "RELEASE_OR_TAG_CREATION:\nYES",
        ):
            self.assertNotIn(forbidden, combined)
        for token in (
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "CANONICAL_HOLDOUT_MODEL_FORWARD_PASSES:\n0",
        ):
            self.assertIn(token, combined)


    def test_materialized_cv_identity_matches_the_frozen_baseline_contract(self):
        manifest_bytes = CV_MANIFEST.read_bytes()
        manifest = json.loads(manifest_bytes)
        document = BASELINE_DOCUMENT.read_text(encoding="utf-8")

        self.assertEqual(
            hashlib.sha256(manifest_bytes.replace(b"\r\n", b"\n")).hexdigest(),
            "494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f",
        )
        self.assertEqual(manifest["experiment_id"], "deep3-postholdout-research-01-baseline")
        self.assertEqual(manifest["development_count"], 17188)
        self.assertEqual(
            [(fold["train_count"], fold["validation_count"]) for fold in manifest["folds"]],
            [(11458, 5730), (11459, 5729), (11459, 5729)],
        )
        self.assertIn("CV_IDENTITY_STATUS:\nMATERIALIZED", document)
        self.assertIn("CV_MANIFEST_LF_NORMALIZED_SHA256:\n494bbc47a75aa35ab436d48899d531febc079301c15cdcf659df18e0fac2352f", document)

if __name__ == "__main__":
    unittest.main()