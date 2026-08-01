import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from zipfile import ZipFile


class LoaderCompatibilityTest(unittest.TestCase):
    def _load_module(self, cache_directory):
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.ClassLabel = object
        fake_datasets.DatasetDict = dict
        fake_datasets.config = types.SimpleNamespace(
            HF_DATASETS_CACHE=str(cache_directory),
        )
        fake_datasets.load_dataset = lambda *_args, **_kwargs: None
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.hf_hub_download = lambda *_args, **_kwargs: "unused-archive.zip"
        with patch.dict(
            sys.modules,
            {"datasets": fake_datasets, "huggingface_hub": fake_hub},
        ):
            sys.modules.pop("src.datasets.fruit_freshness", None)
            return importlib.import_module("src.datasets.fruit_freshness")

    def test_pinned_archive_is_extracted_to_a_managed_content_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            archive_path = temporary_path / "freshness_fruit.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("dataset/Train/freshapples/example.png", b"image")

            module = self._load_module(temporary_path / "datasets-cache")
            with patch.object(
                module,
                "hf_hub_download",
                return_value=str(archive_path),
            ) as mocked_download:
                data_directory = module._resolve_imagefolder_data_dir()

            self.assertEqual(
                data_directory,
                temporary_path
                / "datasets-cache"
                / "fruit_freshness"
                / module.DATASET_REVISION
                / "dataset",
            )
            self.assertTrue(
                (data_directory / "Train" / "freshapples" / "example.png").is_file()
            )
            mocked_download.assert_called_once_with(
                repo_id="Densu341/Fresh-rotten-fruit",
                repo_type="dataset",
                filename="freshness_fruit.zip",
                revision="2077850adc575aa1e8d6029e6cd6cefe9e403a1c",
            )

    def test_existing_complete_content_root_is_reused_without_hub_access(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            module = self._load_module(temporary_path / "datasets-cache")
            expected_directory = (
                temporary_path
                / "datasets-cache"
                / "fruit_freshness"
                / module.DATASET_REVISION
                / "dataset"
            )
            expected_directory.mkdir(parents=True)

            with patch.object(module, "hf_hub_download") as mocked_download:
                actual_directory = module._resolve_imagefolder_data_dir()

            self.assertEqual(actual_directory, expected_directory)
            mocked_download.assert_not_called()

    def test_unsafe_archive_member_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            archive_path = temporary_path / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", b"unsafe")

            module = self._load_module(temporary_path / "datasets-cache")
            with patch.object(
                module,
                "hf_hub_download",
                return_value=str(archive_path),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Unsafe dataset archive member",
                ):
                    module._resolve_imagefolder_data_dir()

            self.assertFalse((temporary_path / "outside.txt").exists())


if __name__ == "__main__":
    unittest.main()