"""Offline contract for the frozen Phase 9.7 determinism protocol."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / "postholdout-determinism-protocol.md"
CHECK_CONFIG = ROOT / "configs" / "deep3_postholdout_determinism_check.toml"
BASELINE_CONFIG = ROOT / "configs" / "deep3_postholdout_baseline.toml"

FROZEN_CONFIGS = (
    "deep3_postholdout_baseline.toml",
    "deep3_postholdout_loss001.toml",
    "deep3_postholdout_baseline_rep002.toml",
    "deep3_postholdout_baseline_rep003.toml",
)


def _flatten(mapping: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


class DeterminismProtocolContractTest(unittest.TestCase):
    def test_protocol_is_frozen(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "PROTOCOL_STATUS:\nFROZEN",
            "SEED:\n20260815",
            "VERIFICATION_RUN_COUNT:\n2",
            "APPROVED_SEED:\n20260815",
            "APPROVED_DETERMINISM_LADDER:\nYES",
            "APPROVED_VERIFICATION_RUN_COUNT:\n2",
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

    def test_every_ladder_branch_is_named(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # A ladder with an unhandled branch returns discretion to whoever
        # reads the result first.
        for token in (
            "LEVEL_ORDER:\nA_STRICT then B_CUDNN",
            "A_ADOPTED:",
            "A_DEGRADED:",
            "B_ADOPTED:",
            "B_DEGRADED:",
            "A_FAILED_OTHER:",
            "BLOCKED:",
        ):
            self.assertIn(token, document)

    def test_seed_conditional_limitation_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        # Recorded before any result exists, so Phase 9.8 cannot replace one
        # overclaim with another.
        self.assertIn("It does not remove seed-to-seed variation", document)
        self.assertIn("common random numbers", document)

    def test_dataloader_generator_rejection_is_recorded(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        self.assertIn("The DataLoader generator is deliberately not added", document)

    def test_check_config_changes_only_registered_fields(self):
        baseline = _flatten(tomllib.loads(BASELINE_CONFIG.read_text(encoding="utf-8")))
        check = _flatten(tomllib.loads(CHECK_CONFIG.read_text(encoding="utf-8")))

        differing = {
            key
            for key in set(baseline) | set(check)
            if baseline.get(key) != check.get(key)
        }
        self.assertEqual(
            differing,
            {
                "runtime.cudnn_benchmark",
                "runtime.seed",
                "runtime.determinism_level",
                "training.epochs",
                "fine_tuning.epochs",
                "post_holdout.experiment_id",
                "post_holdout.parent_experiment_id",
                "post_holdout.artifact_namespace",
            },
        )

    def test_check_config_never_uses_the_canonical_route(self):
        check = _flatten(tomllib.loads(CHECK_CONFIG.read_text(encoding="utf-8")))

        # The canonical route trains on the 4,298 locked-test examples.
        self.assertEqual(
            check["post_holdout.split_manifest_path"],
            "configs/splits/deep3-postholdout-research-01.json",
        )
        self.assertEqual(
            check["post_holdout.cv_manifest_path"],
            "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
        )

    def test_frozen_configs_gained_no_determinism_keys(self):
        for name in FROZEN_CONFIGS:
            with self.subTest(name=name):
                config = tomllib.loads(
                    (ROOT / "configs" / name).read_text(encoding="utf-8")
                )
                # Adding required keys would have changed hashes recorded in
                # two frozen protocol documents.
                self.assertNotIn("seed", config["runtime"])
                self.assertNotIn("determinism_level", config["runtime"])

    def test_phase_never_widens_into_evaluation_or_publication(self):
        document = PROTOCOL.read_text(encoding="utf-8")

        for token in (
            "LOCKED_TEST_MODEL_ACCESS:\nNO",
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "POST_HOLDOUT_LOCKED_TEST_MODEL_FORWARD_PASSES:\n0",
            "APPROVED_LOCKED_TEST_EVALUATION:\nNO",
            "APPROVED_WEIGHT_PUBLICATION:\nNO",
            "APPROVED_LOSS001_RERUN:\nDEFERRED_TO_PHASE_9_8",
        ):
            self.assertIn(token, document)

        for forbidden in (
            "APPROVED_LOCKED_TEST_EVALUATION:\nYES",
            "APPROVED_WEIGHT_PUBLICATION:\nYES",
            "LOCKED_TEST_MODEL_ACCESS:\nYES",
        ):
            self.assertNotIn(forbidden, document)


if __name__ == "__main__":
    unittest.main()
