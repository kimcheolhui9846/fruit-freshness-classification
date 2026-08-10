"""Offline orchestration tests for the Phase 9.3 development baseline route."""

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("phase93_train", TRAIN_PATH)
train = importlib.util.module_from_spec(TRAIN_SPEC)
sys.modules[TRAIN_SPEC.name] = train
TRAIN_SPEC.loader.exec_module(train)


class FinalDataset(dict):
    def __init__(self):
        super().__init__(train="historical-train", test="historical-holdout")
        self.accesses = []

    def __getitem__(self, key):
        self.accesses.append(key)
        return super().__getitem__(key)


class PostHoldoutTrainingRouteTest(unittest.TestCase):
    def test_canonical_config_preserves_existing_train_fold_route(self):
        final_dataset = FinalDataset()
        dependencies = SimpleNamespace(
            load_fruit_freshness_dataset=lambda: final_dataset,
            iter_stratified_folds=lambda dataset, n_splits, shuffle, random_state: iter(
                [("canonical-train", "canonical-validation")]
            ),
        )
        config = {
            "cross_validation": {"n_splits": 3, "shuffle": True, "random_state": 42},
        }

        development, folds, protocol = train.prepare_training_dataset_and_folds(
            config,
            dependencies,
        )

        self.assertEqual(development, "historical-train")
        self.assertEqual(folds, [("canonical-train", "canonical-validation")])
        self.assertIsNone(protocol)
        self.assertEqual(final_dataset.accesses, ["train"])

    def test_baseline_config_selects_only_frozen_development_and_tracked_cv_folds(self):
        final_dataset = FinalDataset()
        events = []
        frozen_manifest = {
            "development_count": 17188,
            "locked_test_count": 4298,
            "source_pool_size": 21486,
        }
        cv_manifest = {"folds": ["tracked-fold"]}
        dependencies = SimpleNamespace(
            load_fruit_freshness_dataset=lambda: final_dataset,
            load_frozen_postholdout_manifest=lambda path: events.append(("split", path)) or frozen_manifest,
            select_frozen_development_pool=lambda historical_train, historical_holdout, manifest: events.append(
                ("select", historical_train, historical_holdout, manifest)
            ) or "development-only",
            load_postholdout_cv_manifest=lambda path, **kwargs: events.append(("cv", path, kwargs)) or cv_manifest,
            cv_folds_from_manifest=lambda manifest: events.append(("folds", manifest)) or [
                ("tracked-train", "tracked-validation")
            ],
        )
        config = {
            "cross_validation": {"n_splits": 3, "shuffle": True, "random_state": 42},
            "post_holdout": {
                "experiment_id": "deep3-postholdout-research-01-baseline",
                "parent_experiment_id": "deep3-postholdout-research-01",
                "artifact_namespace": "deep3-postholdout-research-01-baseline",
                "split_manifest_path": "configs/splits/deep3-postholdout-research-01.json",
                "cv_manifest_path": "configs/splits/deep3-postholdout-research-01-baseline-cv.json",
            },
        }

        with patch.object(train, "_sha256_file", return_value="c" * 64):
            development, folds, protocol = train.prepare_training_dataset_and_folds(
                config,
                dependencies,
            )

        self.assertEqual(development, "development-only")
        self.assertEqual(folds, [("tracked-train", "tracked-validation")])
        self.assertEqual(protocol["development_count"], 17188)
        self.assertEqual(final_dataset.accesses, ["train", "test"])
        self.assertEqual(events[0][0], "split")
        self.assertEqual(events[1][:3], ("select", "historical-train", "historical-holdout"))
        self.assertEqual(events[2][0], "cv")
        self.assertEqual(events[2][2]["development_count"], 17188)
        self.assertEqual(events[3], ("folds", cv_manifest))


if __name__ == "__main__":
    unittest.main()