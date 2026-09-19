import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.runtime import artifact_provenance as provenance
from src.runtime.cifar100c_huggingface import (
    CIFAR100C_HF_ACQUISITION, CIFAR100C_NPY_MD5, verify_huggingface_cifar100c_files,
)


class HuggingFaceAcquisitionTests(unittest.TestCase):
    def test_pinned_acquisitions_are_distinct_and_accepted(self):
        self.assertNotEqual(CIFAR100C_HF_ACQUISITION, provenance.CIFAR100C_OFFICIAL_ACQUISITION)
        provenance._validate_cifar_acquisition(CIFAR100C_HF_ACQUISITION)
        provenance._validate_cifar_acquisition(provenance.CIFAR100C_OFFICIAL_ACQUISITION)
        self.assertNotIn("actual_checksum", CIFAR100C_HF_ACQUISITION)

    def test_changed_revision_url_or_checksum_is_rejected(self):
        for field in ("revision", "url", "files"):
            with self.subTest(field=field):
                bad = copy.deepcopy(CIFAR100C_HF_ACQUISITION)
                bad[field] = {} if field == "files" else "changed"
                with self.assertRaises(provenance.ProvenanceError):
                    provenance._validate_cifar_acquisition(bad)

    def test_generation_checks_original_bytes_before_writing_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in CIFAR100C_NPY_MD5:
                (root / name).write_bytes(b"not the official data")
            with self.assertRaisesRegex(provenance.ProvenanceError, "checksum mismatch"):
                provenance.generate_cifar100c_provenance(root, acquisition=CIFAR100C_HF_ACQUISITION)
            self.assertFalse(provenance.default_sidecar_path(root, "cifar100c").exists())

    def test_missing_or_extra_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(provenance.ProvenanceError, "twenty"):
                verify_huggingface_cifar100c_files(root)
            for name in CIFAR100C_NPY_MD5:
                (root / name).write_bytes(b"")
            (root / "extra.npy").write_bytes(b"")
            with self.assertRaisesRegex(provenance.ProvenanceError, "twenty"):
                verify_huggingface_cifar100c_files(root)

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            target = Path(tmp) / "target"
            target.write_bytes(b"data")
            (root / "labels.npy").symlink_to(target)
            with self.assertRaises(provenance.ProvenanceError):
                verify_huggingface_cifar100c_files(root)

    def test_strict_matrix_validation_accepts_pinned_hf_and_rejects_altered_receipt(self):
        from src.runtime.experiment_matrix import build_experiment_matrix, validate_completed_run, IncompleteRunError
        from tests.test_experiment_matrix import _write_valid_evidence

        with tempfile.TemporaryDirectory() as tmp:
            run = build_experiment_matrix(datasets=("CIFAR100C",), methods=("NoAdapt",),
                                          streams=("block",), seeds=(0,), device="cpu",
                                          evidence_dir=Path(tmp))[0]
            _write_valid_evidence(run)
            path = run.run_dir / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["artifacts"]["dataset"].update(
                acquisition=copy.deepcopy(CIFAR100C_HF_ACQUISITION), file_count=20,
            )
            path.write_text(json.dumps(manifest))
            validate_completed_run(run)
            for field, value in (("file_count", 19), ("acquisition", {})):
                with self.subTest(field=field):
                    bad = copy.deepcopy(manifest)
                    bad["artifacts"]["dataset"][field] = value
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(IncompleteRunError):
                        validate_completed_run(run)


if __name__ == "__main__":
    unittest.main()
