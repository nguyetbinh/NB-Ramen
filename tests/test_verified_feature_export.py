import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation.failure_analysis_artifacts import ReplaySidecarWriter, sha256_file
from src.evaluation.verified_feature_export import export_verified_replay_features, run_verified_replay_probes
from src.evaluation.evidence import FAILURE_ANALYSIS_REQUIRED_FIELDS
from src.streams import stream_fingerprint


class VerifiedFeatureExportTests(unittest.TestCase):
    def _run(self, directory, *, analysis_role="analysis", completed=True, query_count=6,
             labels=None, domains=None):
        root = Path(directory) / "run"
        root.mkdir()
        manifest = {
            "schema_version": 1, "run_id": "probe-run",
            "args": {"failure_analysis_profile": "replay_v1", "tta_algo": "CausalRamen", "analysis_role": analysis_role},
            "artifacts": {"model": {"actual_sha256": "a" * 64}},
            "git": {"source": {"fingerprint": "b" * 64}},
        }
        if analysis_role is None:
            manifest["args"].pop("analysis_role")
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        stream = {"metadata": {"fixture": True}, "references": [[0, 100 + index] for index in range(query_count)]}
        stream["fingerprint"] = stream_fingerprint(stream)
        (root / "stream.json").write_text(json.dumps(stream), encoding="utf-8")
        trace = []
        for index in range(query_count):
            failure_analysis = {field: 0 for field in FAILURE_ANALYSIS_REQUIRED_FIELDS}
            failure_analysis.update({"query_item_id": index, "producer_query_timestep": index,
                                     "evaluator_sample_identity": {"sample_idx": 100 + index,
                                                                   "ground_truth_domain": (index % 2 if domains is None else domains[index])},
                                     "batch_position": index, "schedule": "causal",
                                     "conflict_metric": "fraction_low_consensus_coordinates_v1", "conflict": 0.0,
                                     "segment_index": 0})
            trace.append({"schema_version": 2, "run_id": "probe-run", "timestep": index,
                          "sample_idx": 100 + index, "ground_truth_domain": (index % 2 if domains is None else domains[index]),
                          "ground_truth_class": (index % 3 if labels is None else labels[index]), "prediction": 0, "correct": False,
                          "failure_analysis": failure_analysis})
        (root / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
        writer = ReplaySidecarWriter(root / "failure-analysis", run_id="probe-run",
            manifest_sha256=sha256_file(root / "manifest.json"), stream_fingerprint=stream["fingerprint"],
            source_fingerprint="b" * 64, config={"counterfactual_thresholds": [0.5, 0.75, 1.0]},
            max_samples=query_count, max_bytes=100_000)
        for index, trace_row in enumerate(trace):
            feature = torch.tensor([float(index), float(index % 2), 1.0], dtype=torch.float32)
            identity = {"segment_index": 0, "producer_query_timestep": trace_row["timestep"],
                        "evaluator_sample_identity": {"sample_idx": trace_row["sample_idx"],
                                                      "ground_truth_domain": trace_row["ground_truth_domain"]}}
            self.assertTrue(writer.write(items=[{"item_id": index, "feature": feature, **identity}], query={
                "segment_index": 0, "item_id": index, "legal_candidates": [index],
                **identity,
                "ground_truth_class": trace_row["ground_truth_class"],
            }))
        writer.close(completed=completed)
        return root

    def test_export_decodes_exact_query_features_and_binds_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            exported = export_verified_replay_features(run, seed=11)
            self.assertEqual("computed", exported["status"])
            self.assertEqual(6, len(exported["rows"]))
            self.assertEqual([0.0, 0.0, 1.0], exported["rows"][0]["feature"])
            self.assertEqual("float32", exported["metadata"]["feature_dtype"])
            self.assertEqual(3, exported["metadata"]["feature_dim"])
            self.assertEqual("analysis", exported["metadata"]["analysis_role"])
            self.assertEqual("b" * 64, exported["metadata"]["source_fingerprint"])
            repeated = export_verified_replay_features(run, seed=11)
            self.assertEqual([row["split_role"] for row in exported["rows"]],
                             [row["split_role"] for row in repeated["rows"]])
            self.assertEqual(6, len({(row["timestep"], row["sample_idx"], row["ground_truth_domain"],
                                      row["ground_truth_class"]) for row in exported["rows"]}))
            report = run_verified_replay_probes(run, seed=11)
            self.assertEqual("computed", report["status"])

    def test_split_assignment_ignores_evaluator_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory, labels=[9, 8, 7, 6, 5, 4], domains=[4, 4, 4, 4, 4, 4])
            exported = export_verified_replay_features(run, seed=11)
            trace_path = run / "trace.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            queries_path = run / "failure-analysis" / "queries.jsonl"
            queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
            items_path = run / "failure-analysis" / "items.jsonl"
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            for index, (trace_row, query, item) in enumerate(zip(trace, queries, items)):
                trace_row["ground_truth_domain"] = index % 2
                trace_row["ground_truth_class"] = index % 3
                trace_row["failure_analysis"]["evaluator_sample_identity"]["ground_truth_domain"] = index % 2
                query["ground_truth_class"] = index % 3
                query["evaluator_sample_identity"]["ground_truth_domain"] = index % 2
                item["evaluator_sample_identity"]["ground_truth_domain"] = index % 2
            trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
            self._replace_sidecar_rows(run, "queries", queries)
            self._replace_sidecar_rows(run, "items", items)
            relabeled = export_verified_replay_features(run, seed=11)
            self.assertEqual([row["split_role"] for row in exported["rows"]],
                             [row["split_role"] for row in relabeled["rows"]])

    def test_query_must_bind_to_trace_item_and_feature_item_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            queries_path = run / "failure-analysis" / "queries.jsonl"
            queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
            queries[0]["item_id"] = 1
            self._replace_sidecar_rows(run, "queries", queries)
            with self.assertRaisesRegex(ValueError, "query item.*evaluator trace"):
                export_verified_replay_features(run)
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            items_path = run / "failure-analysis" / "items.jsonl"
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            items[0]["producer_query_timestep"] = 99
            self._replace_sidecar_rows(run, "items", items)
            with self.assertRaisesRegex(ValueError, "feature-bearing item.*sidecar query"):
                export_verified_replay_features(run)
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            items_path = run / "failure-analysis" / "items.jsonl"
            items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines()]
            items[0].pop("feature")
            self._replace_sidecar_rows(run, "items", items)
            with self.assertRaisesRegex(ValueError, "no feature-bearing item"):
                export_verified_replay_features(run)
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            queries_path = run / "failure-analysis" / "queries.jsonl"
            queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
            queries[1] = dict(queries[0])
            self._replace_sidecar_rows(run, "queries", queries)
            with self.assertRaisesRegex(ValueError, "duplicate evaluator identities"):
                export_verified_replay_features(run)
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            trace_path = run / "trace.jsonl"
            trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            trace[0]["failure_analysis"]["producer_query_timestep"] = 99
            trace_path.write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "query identity.*evaluator trace"):
                export_verified_replay_features(run)

    @staticmethod
    def _replace_sidecar_rows(run, name, rows):
        path = run / "failure-analysis" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        metadata_path = run / "failure-analysis" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["files"][f"{name}.jsonl"] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_final_or_missing_canonical_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "final-evaluation"):
                export_verified_replay_features(self._run(directory, analysis_role="final"))
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory, analysis_role=None)
            with self.assertRaisesRegex(ValueError, "args.analysis_role"):
                export_verified_replay_features(run)

    def test_tampered_or_partial_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory)
            feature_path = run / "failure-analysis" / "features.bin"
            feature_path.write_bytes(feature_path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                export_verified_replay_features(run)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not closed"):
                export_verified_replay_features(self._run(directory, completed=False))

    def test_partial_query_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._run(directory, query_count=6)
            queries = run / "failure-analysis" / "queries.jsonl"
            lines = queries.read_text(encoding="utf-8").splitlines()
            queries.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            metadata_path = run / "failure-analysis" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["files"]["queries.jsonl"] = {"bytes": queries.stat().st_size, "sha256": sha256_file(queries)}
            metadata["row_counts"]["queries"] = len(lines) - 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full evaluator-trace coverage"):
                export_verified_replay_features(run)


if __name__ == "__main__":
    unittest.main()
