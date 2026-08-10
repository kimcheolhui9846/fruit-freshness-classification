"""Offline tests for the Phase 9.3 frozen development baseline helpers."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.datasets.postholdout import (
    build_postholdout_cv_manifest,
    load_frozen_postholdout_manifest,
    load_postholdout_cv_manifest,
    sha256_json_identity_file,
)


ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "configs" / "splits" / "deep3-postholdout-research-01.json"


class PostHoldoutBaselineTest(unittest.TestCase):
    def test_frozen_manifest_loader_validates_phase_92_identity(self):
        manifest = load_frozen_postholdout_manifest(SPLIT_PATH)

        self.assertEqual(manifest["protocol"], "DEV_PLUS_LOCKED_TEST")
        self.assertEqual(manifest["source_pool_size"], 21486)
        self.assertEqual(manifest["development_count"], 17188)
        self.assertEqual(manifest["locked_test_count"], 4298)
        self.assertEqual(manifest["canonical_holdout_overlap"], 0)
        self.assertEqual(
            manifest["development_indices_sha256"],
            "329086d616fbf72e79bb65f00966259d6788cd8ff85daf4aff444688e06dfc19",
        )

    def test_cv_manifest_is_deterministic_disjoint_and_exhaustive(self):
        labels = np.repeat(np.arange(3, dtype=np.int64), 6)
        first = build_postholdout_cv_manifest(
            labels,
            experiment_id="example-baseline",
            parent_experiment_id="example-parent",
            development_manifest_sha256="a" * 64,
            n_splits=3,
            shuffle=True,
            random_state=42,
        )
        second = build_postholdout_cv_manifest(
            labels,
            experiment_id="example-baseline",
            parent_experiment_id="example-parent",
            development_manifest_sha256="a" * 64,
            n_splits=3,
            shuffle=True,
            random_state=42,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["development_count"], len(labels))
        self.assertEqual(first["cross_validation"]["random_state"], 42)
        validation_indices = [
            np.asarray(fold["validation_indices"], dtype=np.int64)
            for fold in first["folds"]
        ]
        self.assertEqual(len(validation_indices), 3)
        self.assertEqual(sum(len(indices) for indices in validation_indices), len(labels))
        self.assertEqual(
            sorted(np.concatenate(validation_indices).tolist()),
            list(range(len(labels))),
        )
        for left, right in zip(validation_indices, validation_indices[1:]):
            self.assertEqual(len(np.intersect1d(left, right)), 0)

    def test_cv_manifest_loader_rejects_changed_fold_indices(self):
        labels = np.repeat(np.arange(3, dtype=np.int64), 6)
        manifest = build_postholdout_cv_manifest(
            labels,
            experiment_id="example-baseline",
            parent_experiment_id="example-parent",
            development_manifest_sha256="b" * 64,
            n_splits=3,
            shuffle=True,
            random_state=42,
        )
        original_validation = manifest["folds"][0]["validation_indices"][0]
        original_train = manifest["folds"][0]["train_indices"][0]
        manifest["folds"][0]["validation_indices"][0] = original_train
        manifest["folds"][0]["train_indices"][0] = original_validation
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cv.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_postholdout_cv_manifest(
                    path,
                    development_manifest_sha256="b" * 64,
                    development_count=len(labels),
                )


    def test_json_identity_hash_normalizes_crlf_to_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            path.write_bytes(b"{\r\n  \"schema_version\": 1\r\n}\r\n")

            self.assertEqual(
                sha256_json_identity_file(path),
                hashlib.sha256(b"{\n  \"schema_version\": 1\n}\n").hexdigest(),
            )

if __name__ == "__main__":
    unittest.main()