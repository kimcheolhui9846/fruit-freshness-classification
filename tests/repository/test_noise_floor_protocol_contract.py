"""Offline contract for the frozen, not-yet-executed noise floor measurement."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-noise-floor-protocol.md"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"
REPLICATE_CONFIGS = (
    ROOT / "configs" / "deep3_postholdout_baseline_rep002.toml",
    ROOT / "configs" / "deep3_postholdout_baseline_rep003.toml",
)

IDENTITY_ONLY = {"post_holdout.experiment_id", "post_holdout.artifact_namespace"}


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class NoiseFloorProtocolContractTest(unittest.TestCase):
    def test_replicates_change_identity_and_nothing_else(self):
        baseline = _flatten(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))

        for config_path in REPLICATE_CONFIGS:
            replicate = _flatten(tomllib.loads(config_path.read_text(encoding="utf-8")))
            differing = {
                key
                for key in set(baseline) | set(replicate)
                if baseline.get(key) != replicate.get(key)
            }
            # A replicate that changed any recipe value would measure something
            # other than run-to-run variation of the same pipeline.
            self.assertEqual(differing, IDENTITY_ONLY, f"{config_path.name}")

    def test_replicates_reuse_the_baseline_folds(self):
        for config_path in REPLICATE_CONFIGS:
            replicate = _flatten(tomllib.loads(config_path.read_text(encoding="utf-8")))
            # Different folds would measure split variation, a different quantity.
            self.assertEqual(
                replicate["post_holdout.cv_manifest_path"],
                "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
            )
            self.assertEqual(
                replicate["post_holdout.parent_experiment_id"],
                "deep3-postholdout-research-01",
            )

    def test_replicate_identities_are_distinct(self):
        identities = {
            _flatten(tomllib.loads(path.read_text(encoding="utf-8")))[
                "post_holdout.artifact_namespace"
            ]
            for path in REPLICATE_CONFIGS
        }
        # Shared namespaces would overwrite one another's artifacts.
        self.assertEqual(len(identities), len(REPLICATE_CONFIGS))

    def test_protocol_is_frozen_and_unexecuted(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "REPLICATE_COUNT:\n2",
            "APPROVED_INTERPRETATION_RULE:\nTWO_SIGMA",
            "APPROVED_EXECUTION:\nGRANTED",
            "APPROVED_REPLICATE_COUNT:\n2",
        ):
            self.assertIn(token, document)

    def test_execution_status_is_exactly_one_known_state(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        states = [
            state
            for state in ("NOT_YET_RUN", "IN_PROGRESS", "COMPLETED", "STOPPED")
            if f"EXECUTION_STATUS:\n{state}" in document
        ]
        self.assertEqual(len(states), 1, f"expected exactly one execution status, got {states}")

    def test_interpretation_rule_is_numeric_and_directional(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Both branches must be named before the numbers exist.
        for token in (
            "COMPARISON:\nd = 0.0090 against 2s",
            "INCONCLUSIVE:\nd <= 2s",
            "H1_EXHAUSTED_STANDS:\nd > 2s",
        ):
            self.assertIn(token, document)

    def test_protocol_refuses_to_rescore_an_earlier_result(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A measurement taken after a verdict may inform the next step; it may
        # never re-score the verdict.
        self.assertIn("does not reverse the loss-001 verdict", document)
        self.assertIn("APPROVED_SEEDING_CHANGE:\nDEFERRED_UNTIL_AFTER_MEASUREMENT", document)
        self.assertIn("Three samples estimate `s` poorly", document)

    def test_measurement_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("LOCKED_TEST_MODEL_ACCESS:\nNO", document)
        self.assertIn("POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0", document)
        for forbidden in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "APPROVED_SEEDING_CHANGE:\nGRANTED",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
