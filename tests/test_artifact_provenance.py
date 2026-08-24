import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.runtime import artifact_provenance as provenance


class ArtifactProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "dataset"
        self.root.mkdir()
        (self.root / "labels.npy").write_bytes(b"labels")
        (self.root / "fog.npy").write_bytes(b"fog")

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _acquisition():
        return dict(provenance.CIFAR100C_OFFICIAL_ACQUISITION)

    def test_cifar_manifest_is_deterministic_and_fast_verifies(self):
        first = provenance.generate_cifar100c_provenance(self.root, acquisition=self._acquisition())
        sidecar = provenance.default_sidecar_path(self.root, "cifar100c")
        initial = sidecar.read_bytes()
        second = provenance.generate_cifar100c_provenance(self.root, acquisition=self._acquisition())

        self.assertEqual(first, second)
        self.assertEqual(initial, sidecar.read_bytes())
        report = provenance.verify_cifar100c_provenance(self.root)
        self.assertEqual(2, report["file_count"])
        self.assertFalse(report["verified_exact"])

    def test_archive_record_is_verified_before_it_can_be_recorded(self):
        archive = self.root / "CIFAR-100-C.tar"
        archive.write_bytes(b"archive")
        digest = hashlib.sha256(b"archive").hexdigest()
        record = provenance.archive_acquisition_record(
            archive, url="https://example.invalid/CIFAR-100-C.tar", algorithm="sha256", expected_checksum=digest
        )
        self.assertEqual(hashlib.sha256(b"archive").hexdigest(), record["actual_checksum"])
        with self.assertRaisesRegex(provenance.ProvenanceError, "mismatch"):
            provenance.archive_acquisition_record(
                archive, url="https://example.invalid/CIFAR-100-C.tar", algorithm="sha256", expected_checksum="0" * 64
            )

    def test_official_style_md5_archive_record_is_supported(self):
        archive = self.root / "CIFAR-100-C.tar"
        archive.write_bytes(b"zenodo archive")
        digest = hashlib.md5(b"zenodo archive").hexdigest()
        record = provenance.archive_acquisition_record(
            archive, url="https://zenodo.org/records/2535967/files/CIFAR-100-C.tar",
            algorithm="md5", expected_checksum=digest,
        )
        self.assertEqual("md5", record["algorithm"])
        self.assertEqual(digest, record["actual_checksum"])

    def test_cifar_inventory_rejects_nonofficial_matching_acquisition(self):
        acquisition = self._acquisition()
        acquisition.update({
            "url": "https://evil.invalid/CIFAR-100-C.tar",
            "expected_checksum": "0" * 32,
            "actual_checksum": "0" * 32,
        })
        with self.assertRaisesRegex(provenance.ProvenanceError, "pinned official Zenodo"):
            provenance.generate_cifar100c_provenance(self.root, acquisition=acquisition)

    def test_official_cifar_archive_checker_pins_size_and_publisher_fields(self):
        archive = self.root / "CIFAR-100-C.tar"
        archive.write_bytes(b"fixture")
        generic = {
            "url": provenance.CIFAR100C_OFFICIAL_ACQUISITION["url"],
            "algorithm": "md5",
            "expected_checksum": provenance.CIFAR100C_OFFICIAL_ACQUISITION["expected_checksum"],
            "actual_checksum": provenance.CIFAR100C_OFFICIAL_ACQUISITION["actual_checksum"],
            "size_bytes": provenance.CIFAR100C_OFFICIAL_ACQUISITION["size_bytes"],
        }
        with mock.patch.object(provenance, "archive_acquisition_record", return_value=generic):
            record = provenance.verify_official_cifar100c_archive(archive)
        self.assertEqual(provenance.CIFAR100C_OFFICIAL_ACQUISITION, record)
        generic["size_bytes"] -= 1
        with mock.patch.object(provenance, "archive_acquisition_record", return_value=generic):
            with self.assertRaisesRegex(provenance.ProvenanceError, "size mismatch"):
                provenance.verify_official_cifar100c_archive(archive)

    def test_exact_verification_detects_same_size_tampering(self):
        provenance.generate_domainnet_provenance(self.root)
        (self.root / "fog.npy").write_bytes(b"bad")

        self.assertEqual(2, provenance.verify_domainnet_provenance(self.root)["file_count"])
        with self.assertRaisesRegex(provenance.ProvenanceError, "SHA-256 mismatch"):
            provenance.verify_domainnet_provenance(self.root, exact=True)

    def test_nested_provenance_named_directory_is_dataset_content(self):
        nested = self.root / "class-a" / provenance.SIDECAR_DIRECTORY
        nested.mkdir(parents=True)
        nested_file = nested / "image.bin"
        nested_file.write_bytes(b"nested loader-visible content")
        provenance.generate_domainnet_provenance(self.root)
        report = provenance.verify_domainnet_provenance(self.root, exact=True)
        self.assertEqual(3, report["file_count"])
        nested_file.write_bytes(b"tampered loader-visible bytes")
        with self.assertRaisesRegex(provenance.ProvenanceError, "SHA-256 mismatch"):
            provenance.verify_domainnet_provenance(self.root, exact=True)

    def test_verification_rejects_stale_inventory_and_path_traversal(self):
        provenance.generate_domainnet_provenance(self.root)
        (self.root / "new-image.jpg").write_bytes(b"new")
        with self.assertRaisesRegex(provenance.ProvenanceError, "inventory is stale"):
            provenance.verify_domainnet_provenance(self.root)

        (self.root / "new-image.jpg").unlink()
        sidecar = provenance.default_sidecar_path(self.root, "domainnet")
        payload = json.loads(sidecar.read_text())
        payload["content"]["files"][0]["path"] = "../outside"
        sidecar.write_text(json.dumps(payload))
        with self.assertRaisesRegex(provenance.ProvenanceError, "non-canonical"):
            provenance.verify_domainnet_provenance(self.root)

    def test_symlinked_artifact_or_sidecar_is_rejected(self):
        target = self.root / "target.bin"
        target.write_bytes(b"target")
        linked = self.root / "linked.bin"
        try:
            linked.symlink_to(target)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(provenance.ProvenanceError, "symlinked"):
            provenance.generate_domainnet_provenance(self.root)

        linked.unlink()
        provenance.generate_domainnet_provenance(self.root)
        sidecar = provenance.default_sidecar_path(self.root, "domainnet")
        saved = sidecar.with_suffix(".saved")
        sidecar.rename(saved)
        sidecar.symlink_to(saved)
        with self.assertRaisesRegex(provenance.ProvenanceError, "symlinked"):
            provenance.verify_domainnet_provenance(self.root)

    def test_symlinked_canonical_sidecar_directory_is_rejected(self):
        provenance.generate_domainnet_provenance(self.root)
        canonical = self.root / provenance.SIDECAR_DIRECTORY
        external = self.root.parent / "external-provenance"
        try:
            canonical.rename(external)
            canonical.symlink_to(external, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("directory symlinks unavailable")
        with self.assertRaisesRegex(provenance.ProvenanceError, "symlinked artifact path"):
            provenance.verify_domainnet_provenance(self.root, exact=True)
        with self.assertRaisesRegex(provenance.ProvenanceError, "symlinked artifact path"):
            provenance.generate_domainnet_provenance(self.root)

    def test_injected_clip_metadata_is_explicitly_untrusted(self):
        expected = hashlib.sha256(b"checkpoint").hexdigest()
        url = "https://evil.invalid/models/{}/ViT-B-16.pt".format(expected)
        report = provenance.resolve_clip_model(
            "clip_vitbase16", model_urls={"ViT-B/16": url}
        )
        self.assertEqual(expected, report["expected_sha256"])
        self.assertEqual("untrusted_injected_test_metadata", report["trust"])
        self.assertIsNone(report["publisher"])

    def test_checkpoint_verification_uses_only_pinned_hash(self):
        checkpoint = self.root / "ViT-B-16.pt"
        checkpoint.write_bytes(b"test fixture")
        pinned = provenance.resolve_clip_model("clip_vitbase16")["expected_sha256"]
        with mock.patch.object(
            provenance, "sha256_regular_file",
            return_value={"algorithm": "sha256", "sha256": pinned, "size_bytes": 12},
        ):
            report = provenance.verify_clip_checkpoint("clip_vitbase16", checkpoint)
        self.assertEqual(pinned, report["actual_sha256"])
        with mock.patch.object(
            provenance, "sha256_regular_file",
            return_value={"algorithm": "sha256", "sha256": "0" * 64, "size_bytes": 12},
        ):
            with self.assertRaisesRegex(provenance.ProvenanceError, "SHA-256 mismatch"):
                provenance.verify_clip_checkpoint("clip_vitbase16", checkpoint)

    def test_default_clip_resolution_ignores_malicious_package_metadata(self):
        malicious = self.root / "clip.py"
        malicious.write_text("_MODELS = {'ViT-B/16': 'https://evil.invalid/" + "0" * 64 + "/evil.pt'}\n")
        with mock.patch("importlib.util.find_spec", side_effect=AssertionError("package metadata consulted")):
            report = provenance.resolve_clip_model("clip_vitbase16")
        self.assertEqual("pinned_official", report["trust"])
        self.assertEqual("openaipublic.azureedge.net", __import__("urllib.parse").parse.urlparse(report["url"]).hostname)
        self.assertEqual("5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f", report["expected_sha256"])
        with self.assertRaisesRegex(provenance.ProvenanceError, "not a trusted model source"):
            provenance.resolve_clip_model("clip_vitbase16", clip_source=malicious)

    def test_fast_verification_rejects_path_replaced_after_open(self):
        provenance.generate_domainnet_provenance(self.root)
        victim = self.root / "fog.npy"
        original_regular_fd = provenance._regular_fd
        replacement = self.root.parent / "replacement.npy"
        replacement.write_bytes(b"fog")
        replaced = False

        def swap_after_open(path):
            nonlocal replaced
            result = original_regular_fd(path)
            if path == victim and not replaced:
                victim.unlink()
                victim.symlink_to(replacement)
                replaced = True
            return result

        try:
            with mock.patch.object(provenance, "_regular_fd", side_effect=swap_after_open):
                with self.assertRaisesRegex(provenance.ProvenanceError, "changed while inspecting"):
                    provenance.verify_domainnet_provenance(self.root)
        finally:
            if victim.is_symlink():
                victim.unlink()
                victim.write_bytes(b"fog")

    def test_file_hash_refuses_symlink_and_replacement(self):
        artifact = self.root / "model.pt"
        artifact.write_bytes(b"model")
        link = self.root / "model-link.pt"
        try:
            link.symlink_to(artifact)
        except (NotImplementedError, OSError):
            self.skipTest("symlinks unavailable")
        with self.assertRaises(provenance.ProvenanceError):
            provenance.sha256_regular_file(link)

    def test_regular_fd_closes_descriptor_when_fstat_fails(self):
        artifact = self.root / "model.pt"
        artifact.write_bytes(b"model")
        descriptor = os.open(artifact, os.O_RDONLY)
        with mock.patch.object(provenance.os, "open", return_value=descriptor), \
                mock.patch.object(provenance.os, "fstat", side_effect=OSError("fixture")), \
                mock.patch.object(provenance.os, "close", wraps=os.close) as close:
            with self.assertRaisesRegex(provenance.ProvenanceError, "cannot safely inspect"):
                provenance._regular_fd(artifact)
        close.assert_called_once_with(descriptor)


if __name__ == "__main__":
    unittest.main()
