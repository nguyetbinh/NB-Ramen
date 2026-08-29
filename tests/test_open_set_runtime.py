import unittest

from src.runtime.experiment_matrix import build_command, build_experiment_matrix
from src.streams.builders import build_stream, verify_stream_fingerprint
from src.datasets.open_set import OpenSetCIFAR100C


class _Domain:
    def __init__(self):
        self.labels = [0] * 80 + [80] * 20

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, -1 if self.labels[index] == 80 else self.labels[index]

    def evaluator_metadata(self, index):
        original = self.labels[index]
        return {"original_label": original, "is_ood": original == 80,
                "known_label_or_minus_one": -1 if original == 80 else original}


class _Domains:
    datasets = (_Domain(), _Domain())
    environments = ("fog", "snow")


class OpenSetRuntimeTests(unittest.TestCase):
    def test_exact_preregistered_ratio_is_deterministic_and_fingerprinted(self):
        first = build_stream(_Domains(), "block", 7, ood_ratio=.3,
                             open_set_split_version="open-set-cifar100-split-v1")
        second = build_stream(_Domains(), "block", 7, ood_ratio=.3,
                              open_set_split_version="open-set-cifar100-split-v1")
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(verify_stream_fingerprint(first.to_dict()))
        self.assertEqual(36, sum(first.evaluator_metadata(i)["is_ood"] for i in range(len(first))))
        self.assertEqual("open-set-cifar100-split-v1", first.metadata["parameters"]["open_set_split_version"])

    def test_open_set_matrix_binds_role_and_contract_in_run_and_command(self):
        analysis = build_experiment_matrix(datasets=("CIFAR100C",), streams=("block",),
                                           methods=("NoAdapt",), seeds=(0,), device="cpu",
                                           open_set=True, ood_ratio=.3, analysis_role="analysis")[0]
        final = build_experiment_matrix(datasets=("CIFAR100C",), streams=("block",),
                                        methods=("NoAdapt",), seeds=(0,), device="cpu",
                                        open_set=True, ood_ratio=.3, analysis_role="final")[0]
        self.assertNotEqual(analysis.run_id, final.run_id)
        self.assertLessEqual(len(analysis.run_id), 128)
        self.assertEqual(analysis.run_id, build_experiment_matrix(
            datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
            device="cpu", open_set=True, ood_ratio=.3, analysis_role="analysis",
        )[0].run_id)
        command = build_command(analysis)
        self.assertIn("--open-set", command)
        self.assertEqual("0.3", command[command.index("--ood-ratio") + 1])
        self.assertEqual("analysis", command[command.index("--analysis-role") + 1])
        closed_command = build_command(build_experiment_matrix(
            datasets=("CIFAR100C",), streams=("block",), methods=("NoAdapt",), seeds=(0,),
            device="cpu", analysis_role="final",
        )[0])
        self.assertEqual("final", closed_command[closed_command.index("--analysis-role") + 1])

    def test_correlated_open_set_uses_filtered_source_ids_not_positions(self):
        stream = build_stream(
            _Domains(), "class_domain_correlated", 7, ood_ratio=.3,
            open_set_split_version="open-set-cifar100-split-v1",
        )
        expected_source_ids = set(range(42)) | set(range(80, 98))
        self.assertTrue(all(sample_idx in expected_source_ids for _, sample_idx in stream.references))
        self.assertEqual(36, sum(
            stream.evaluator_metadata(index)["is_ood"] for index in range(len(stream))
        ))
        self.assertTrue(verify_stream_fingerprint(stream.to_dict()))

    def test_invalid_open_set_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ood_ratio"):
            build_stream(_Domains(), "iid_mixed", 0, ood_ratio=.2)

    def test_model_labels_and_vocabulary_do_not_expose_unknown_identity(self):
        class SourceDomain:
            Y = [0, 80]
            def __len__(self): return 2
            def __getitem__(self, index): return index, self.Y[index]
        class Source:
            classes = [f"class-{index}" for index in range(100)]
            environments = ("fog",)
            datasets = (SourceDomain(),)
        dataset = OpenSetCIFAR100C(Source())
        self.assertIsInstance(dataset.environments, list)
        self.assertEqual(80, len(dataset.classes))
        self.assertEqual((80,), dataset.unknown_class_ids[:1])
        self.assertEqual((0, 0), dataset.datasets[0][0])
        self.assertEqual((1, -1), dataset.datasets[0][1])
        self.assertTrue(dataset.datasets[0].evaluator_metadata(1)["is_ood"])
