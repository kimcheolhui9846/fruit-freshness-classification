"""Offline contract for the frozen Phase 9 post-holdout split."""

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPOSITORY_ROOT / "configs/splits/deep3-postholdout-research-01.json"


def sha256_int64(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


class PostHoldoutSplitFreezeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        if not MANIFEST_PATH.is_file():
            raise AssertionError(f"Missing frozen split manifest: {MANIFEST_PATH}")
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_preserves_the_approved_data_protocol(self):
        manifest = self.manifest

        self.assertEqual(manifest["experiment_id"], "deep3-postholdout-research-01")
        self.assertEqual(manifest["parent_experiment_id"], "deep3-canonical-reference-01")
        self.assertEqual(manifest["dataset_name"], "Densu341/Fresh-rotten-fruit")
        self.assertEqual(
            manifest["dataset_revision"],
            "2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
        )
        self.assertEqual(manifest["source_pool_identity"], "HISTORICAL_CANONICAL_TRAIN_ONLY")
        self.assertEqual(manifest["source_pool_size"], 21486)
        self.assertEqual(manifest["canonical_holdout_size"], 5372)
        self.assertEqual(manifest["canonical_holdout_overlap"], 0)
        self.assertEqual(manifest["protocol"], "DEV_PLUS_LOCKED_TEST")
        self.assertEqual(manifest["locked_test_fraction"], 0.2)
        self.assertEqual(manifest["split_seed"], 20260810)
        self.assertTrue(manifest["stratified"])
        self.assertEqual(len(manifest["class_names"]), 14)

    def test_manifest_indices_are_disjoint_exhaustive_and_hashed(self):
        manifest = self.manifest
        development = np.asarray(manifest["development_indices"], dtype=np.int64)
        locked_test = np.asarray(manifest["locked_test_indices"], dtype=np.int64)
        source_pool_size = manifest["source_pool_size"]

        self.assertEqual(development.size, manifest["development_count"])
        self.assertEqual(locked_test.size, manifest["locked_test_count"])
        self.assertEqual(np.unique(development).size, development.size)
        self.assertEqual(np.unique(locked_test).size, locked_test.size)
        self.assertEqual(np.intersect1d(development, locked_test).size, 0)
        self.assertTrue(np.all((development >= 0) & (development < source_pool_size)))
        self.assertTrue(np.all((locked_test >= 0) & (locked_test < source_pool_size)))
        np.testing.assert_array_equal(
            np.sort(np.concatenate((development, locked_test))),
            np.arange(source_pool_size),
        )
        self.assertEqual(
            sha256_int64(development),
            manifest["development_indices_sha256"],
        )
        self.assertEqual(
            sha256_int64(locked_test),
            manifest["locked_test_indices_sha256"],
        )

    def test_frozen_boundary_forbids_training_and_locked_test_evaluation(self):
        manifest = self.manifest

        self.assertEqual(manifest["locked_test_status"], "FROZEN_UNOBSERVED_BY_MODEL")
        self.assertEqual(manifest["model_training"], "NO")
        self.assertEqual(manifest["model_evaluation"], "NO")
        self.assertEqual(manifest["locked_test_model_evaluation"], "NO")
        self.assertEqual(manifest["canonical_holdout_usage"], "HISTORICAL_EVIDENCE_ONLY")

    def test_protocol_documents_record_the_frozen_unevaluated_boundary(self):
        split_document = REPOSITORY_ROOT / "docs/post-holdout-split-freeze.md"
        self.assertTrue(split_document.is_file())
        text = split_document.read_text(encoding="utf-8")

        for required_text in (
            "POST_HOLDOUT_LOCKED_TEST_STATUS:\nFROZEN_UNOBSERVED_BY_MODEL",
            "Model predictions: NO",
            "Model metrics: NO",
            "Canonical holdout overlap: `0`",
            "SPLIT_SEED:\n20260810",
            "PROTOCOL:\nDEV_PLUS_LOCKED_TEST",
        ):
            with self.subTest(required_text=required_text):
                self.assertIn(required_text, text)

    def test_registry_and_research_plan_do_not_authorize_phase_9_3(self):
        registry = (REPOSITORY_ROOT / "docs/experiment-registry.md").read_text(
            encoding="utf-8"
        )
        plan = (REPOSITORY_ROOT / "docs/post-holdout-research-plan.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("`PROTOCOL_FROZEN`", registry)
        self.assertIn("baseline: `NOT RUN`", registry)
        self.assertIn("model experiments: `NOT RUN`", registry)
        self.assertIn("locked test model evaluation: `NOT RUN`", registry)
        self.assertIn("APPROVED_PHASE_9_DATA_PROTOCOL:\nDEV_PLUS_LOCKED_TEST", plan)
        self.assertIn("APPROVED_PHASE_9_SPLIT_SEED:\n20260810", plan)
        self.assertIn("APPROVED_PHASE_9_DEVELOPMENT_METRIC:\nMACRO_F1", plan)
        self.assertIn("PHASE_9_3_TRAINING_AUTHORIZATION:\nNOT GRANTED", plan)


if __name__ == "__main__":
    unittest.main()