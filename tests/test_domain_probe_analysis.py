import unittest

from src.evaluation.domain_probe_analysis import (bind_feature_artifact, deterministic_linear_probe,
                                                   representation_probe_report, validate_feature_artifact_metadata)


def metadata():
    return {"stream_fingerprint": "stream", "source_fingerprint": "source", "model_fingerprint": "model",
            "feature_dim": 2, "feature_dtype": "float32", "split_role": "per_row"}


def row(index, feature, domain, cls, role):
    return {"timestep": index, "sample_idx": index, "ground_truth_domain": domain,
            "ground_truth_class": cls, "feature": feature, "split_role": role}


class DomainProbeTests(unittest.TestCase):
    def test_separable_probe_and_train_only_normalization_are_deterministic(self):
        features = [[-2, 0], [-1, 0], [1, 0], [2, 0], [-3, 10], [3, -10]]
        labels = [0, 0, 1, 1, 0, 1]
        roles = ["train", "train", "train", "train", "test", "test"]
        first = deterministic_linear_probe(features, labels, roles, seed=3)
        second = deterministic_linear_probe(features, labels, roles, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(1.0, first["splits"]["test"]["accuracy"])
        self.assertEqual(0.0, first["train_normalization"]["mean"][1])

    def test_nonseparable_and_one_class_are_explicit(self):
        report = deterministic_linear_probe([[0], [0], [0], [0]], [0, 1, 0, 1], ["train", "train", "test", "test"])
        self.assertEqual("computed", report["status"])
        self.assertLessEqual(report["splits"]["test"]["accuracy"], .5)
        insufficient = deterministic_linear_probe([[0], [1]], [0, 0], ["train", "train"])
        self.assertEqual("insufficient", insufficient["status"])

    def test_binding_and_class_conditioned_report(self):
        rows = [row(0, [-2, 0], "a", 0, "train"), row(1, [-1, 0], "a", 0, "train"),
                row(2, [1, 0], "b", 0, "train"), row(3, [2, 0], "b", 0, "test"),
                row(4, [-2, 1], "a", 1, "train"), row(5, [2, 1], "b", 1, "test")]
        self.assertEqual("computed", bind_feature_artifact(rows, metadata())["status"])
        report = representation_probe_report(rows, metadata())
        self.assertIn("0", report["class_conditioned_feature_to_domain"])
        with self.assertRaises(ValueError):
            validate_feature_artifact_metadata({"stream_fingerprint": "x"})
        malformed = list(rows); malformed[0] = dict(malformed[0], feature=[1])
        with self.assertRaises(ValueError): bind_feature_artifact(malformed, metadata())


if __name__ == "__main__": unittest.main()
