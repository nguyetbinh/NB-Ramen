from types import SimpleNamespace
import unittest
from unittest import mock

from src.models import get_pretrained_model as loader_module


class PretrainedModelLoaderTests(unittest.TestCase):
    def test_verified_checkpoint_path_bypasses_symbolic_package_metadata(self):
        args = SimpleNamespace(model="clip_vitbase16", device="cpu")
        verified = "/verified/cache/ViT-B-16.pt"
        with mock.patch.object(
            loader_module.clip, "_MODELS", {"ViT-B/16": "https://evil.invalid/model.pt"},
            create=True,
        ), mock.patch.object(loader_module.clip, "load", return_value=("model", "preprocess")) as load:
            result = loader_module.get_pretrained_model(
                args, verified_checkpoint_path=verified
            )
        self.assertEqual(("model", "preprocess"), result)
        load.assert_called_once_with(verified, device="cpu")

    def test_off_mode_retains_symbolic_model_loading(self):
        args = SimpleNamespace(model="clip_vitbase16", device="cpu")
        with mock.patch.object(loader_module.clip, "load", return_value=("model", "preprocess")) as load:
            loader_module.get_pretrained_model(args)
        load.assert_called_once_with("ViT-B/16", device="cpu")


if __name__ == "__main__":
    unittest.main()
