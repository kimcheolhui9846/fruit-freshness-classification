"""Offline contract for the frozen, not-yet-executed Phase 9.6 H1 loss experiment."""

import hashlib
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-loss001-protocol.md"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"
EXPERIMENT_CONFIG = ROOT / "configs" / "deep3_postholdout_loss001.toml"

ALLOWED_DIFFERENCES = {
    "loss.class_balanced_beta",
    "post_holdout.experiment_id",
    "post_holdout.parent_experiment_id",
    "post_holdout.artifact_namespace",
}


def _lf_normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class Loss001ProtocolContractTest(unittest.TestCase):
    def test_exactly_one_experimental_factor_differs_from_the_baseline(self):
        baseline = _flatten(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))
        experiment = _flatten(tomllib.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8")))

        differing = {
            key
            for key in set(baseline) | set(experiment)
            if baseline.get(key) != experiment.get(key)
        }

        # "One factor at a time" is the research plan's central rule. A rule that
        # relies on nobody mistyping a TOML key is not a rule.
        self.assertEqual(differing, ALLOWED_DIFFERENCES)
        self.assertEqual(baseline["loss.class_balanced_beta"], 0.999)
        self.assertEqual(experiment["loss.class_balanced_beta"], 0.9999)

    def test_comparison_basis_is_the_baseline_folds(self):
        experiment = _flatten(tomllib.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8")))

        # Different folds would make the Macro F1 comparison meaningless.
        self.assertEqual(
            experiment["post_holdout.cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )
        self.assertEqual(
            experiment["post_holdout.split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(experiment["cross_validation.random_state"], 42)
        self.assertEqual(experiment["cross_validation.n_splits"], 3)

    def test_protocol_is_frozen_and_unexecuted(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "EXPERIMENT_ID:\ndeep3-postholdout-research-01-loss-001",
            "APPROVED_EXECUTION:\nGRANTED",
            "APPROVED_EXECUTION_DATE:\n2026-08-14",
            "APPROVED_CANDIDATE_COUNT:\n1",
        ):
            self.assertIn(token, document)

    def test_execution_status_is_exactly_one_known_state(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Lets the status advance to COMPLETED without loosening the contract,
        # while a typo or a second status line still fails.
        states = [
            state
            for state in ("NOT_YET_RUN", "IN_PROGRESS", "COMPLETED", "STOPPED")
            if f"EXECUTION_STATUS:\n{state}" in document
        ]
        self.assertEqual(len(states), 1, f"expected exactly one execution status, got {states}")

    def test_granting_execution_did_not_widen_any_other_approval(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Training approval is not evaluation or publication approval.
        for token in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_RELEASE_CREATION:\nNO",
            "APPROVED_CANDIDATE_COUNT:\n1",
        ):
            self.assertIn(token, document)

    def test_documented_config_hash_matches_the_frozen_file(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn(_lf_normalized_sha256(EXPERIMENT_CONFIG), document)
        self.assertIn(_lf_normalized_sha256(BASELINE_CONFIG), document)

    def test_acceptance_threshold_and_failure_branch_are_numeric_and_fixed(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A criterion without a number is decided after the result is seen.
        for token in (
            "ADVANCE_THRESHOLD:\nMacro F1 >= 0.9112",
            "TOP1_GUARDRAIL:\nTop-1 >= 0.9466",
            "BASELINE_DEVELOPMENT_OOF_MACRO_F1:\n0.9012",
            "BASELINE_DEVELOPMENT_OOF_TOP1:\n0.9566",
        ):
            self.assertIn(token, document)
        # The failure branch is what stops an unregistered search through H1.
        self.assertIn("Phase 9.7 is H2 augmentation", document)
        self.assertIn("noise floor is unmeasured", document)

    def test_experiment_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "CANONICAL_HOLDOUT_MODEL_ACCESS:\nNO",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "ARTIFACT_PUBLICATION:\nLOCAL_ONLY",
        ):
            self.assertIn(token, document)

        # Execution is now granted; the guard narrows to the approvals that must
        # never widen rather than being dropped.
        for forbidden in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "APPROVED_WEIGHT_PUBLICATION:\nYES",
            "APPROVED_RELEASE_CREATION:\nYES",
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
