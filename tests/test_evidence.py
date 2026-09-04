import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

from src import main as main_module
from src.evaluation.evidence import (
    SUMMARY_SCHEMA_VERSION, TRACE_SCHEMA_VERSION, JsonlTraceWriter, atomic_write_json, build_run_manifest,
    compare_trace_negative_adaptation, verify_reference_trace_stream_fingerprint,
    write_run_manifest,
)


class EvidenceTests(unittest.TestCase):
    def test_trace_writer_requires_complete_retrieval_profile_extension(self):
        base = {
            "timestep": 0, "sample_idx": 0, "ground_truth_domain": 0,
            "ground_truth_class": 0, "prediction": 0, "correct": True,
            "predicted_entropy": 0.0, "inferred_context": 0, "memory_size": 1,
            "num_active_contexts": 1, "memory_bytes": 32, "latency_ms": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "trace.jsonl", "profile") as writer:
                with self.assertRaisesRegex(ValueError, "retrieval profile trace fields"):
                    writer.write({**base, "retrieval_profile": "causal_sync_v1"})
                writer.write({
                    **base, "retrieval_profile": "causal_sync_v1", "retrieval_elapsed_ms": .1,
                    "retrieval_candidate_count": 1, "retrieval_eligible_candidate_count": 1,
                    "retrieval_returned_support_count": 1, "retrieval_active_class_count": 1,
                })

    def test_trace_writer_requires_atomic_valid_support_composition_extension(self):
        base = {
            "timestep": 0, "sample_idx": 0, "ground_truth_domain": 0,
            "ground_truth_class": 0, "prediction": 0, "correct": True,
            "predicted_entropy": 0.0, "inferred_context": 0, "memory_size": 1,
            "num_active_contexts": 1, "memory_bytes": 32, "latency_ms": 1.0,
        }
        composition = {
            "returned_support_count": 3, "active_class_count": 2,
            "class_coverage": .5, "same_domain_ratio": .4,
            "cross_domain_ratio": .6, "effective_sample_size": 2.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "trace.jsonl", "support") as writer:
                with self.assertRaisesRegex(ValueError, "support composition trace fields"):
                    writer.write({**base, "returned_support_count": 3})
                with self.assertRaisesRegex(ValueError, "class_coverage"):
                    writer.write({**base, **composition, "class_coverage": 1.1})
                with self.assertRaisesRegex(ValueError, "disagree with support count"):
                    writer.write({**base, **composition, "cross_domain_ratio": .5})
                writer.write({**base, **composition})

    def test_trace_writer_requires_atomic_valid_soft_routing_extension(self):
        base = {
            "timestep": 0, "sample_idx": 0, "ground_truth_domain": 0,
            "ground_truth_class": 0, "prediction": 0, "correct": True,
            "predicted_entropy": 0.0, "inferred_context": 0, "memory_size": 1,
            "num_active_contexts": 1, "memory_bytes": 32, "latency_ms": 1.0,
        }
        soft = {
            "context_strength": .25, "selection_change_ratio": .5,
            "mean_context_bonus": .1, "mean_rank_displacement": 1.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "soft.jsonl", "soft") as writer:
                with self.assertRaisesRegex(ValueError, "soft routing trace fields"):
                    writer.write({**base, "context_strength": .25})
                with self.assertRaisesRegex(ValueError, "selection_change_ratio"):
                    writer.write({**base, **soft, "selection_change_ratio": 2.0})
                writer.write({**base, **soft})

    @staticmethod
    def _git(repository, *arguments):
        subprocess.run(
            ["git", *arguments], cwd=repository, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _reference_evidence(directory):
        run_dir = Path(directory) / "baseline"
        run_dir.mkdir()
        data_root = str(Path(directory).resolve())
        config_path = str((run_dir / "NoAdapt.yaml").resolve())
        model_path = str((run_dir / "ViT-B-16.pt").resolve())
        dataset_root = str((Path(data_root) / "domainbed" / "domain_net").resolve())
        dataset_sidecar = str((run_dir / "dataset-sidecar.json").resolve())
        artifacts = {
            "status": "verified", "mode": "exact",
            "model": {
                "status": "verified", "model": "clip_vitbase16", "official_name": "ViT-B/16",
                "url": "https://example.invalid/model", "expected_sha256": "a" * 64,
                "filename": "ViT-B-16.pt", "publisher": "OpenAI", "trust": "pinned_official",
                "path": model_path, "actual_sha256": "a" * 64, "size_bytes": 1,
            },
            "dataset": {
                "status": "verified", "schema_version": 1, "dataset": "domainnet",
                "root": dataset_root, "sidecar": dataset_sidecar, "verified_exact": True,
                "content_algorithm": "sha256", "root_digest": "b" * 64,
                "sidecar_sha256": "c" * 64, "file_count": 1, "acquisition": {},
            },
        }
        reference_config = {"optimizer": "signsgd", "lr": 1e-9}
        trace_path = run_dir / "trace.jsonl"
        trace_rows = [
            {
                "schema_version": TRACE_SCHEMA_VERSION, "run_id": "baseline", "timestep": index,
                "sample_idx": sample, "ground_truth_domain": domain,
                "ground_truth_class": index, "prediction": index, "correct": True,
                "predicted_entropy": 0.0, "inferred_context": None,
                "memory_size": 0, "num_active_contexts": None, "memory_bytes": None,
                "latency_ms": 1.0,
            }
            for index, (domain, sample) in enumerate(((0, 0), (1, 3)))
        ]
        trace_path.write_text(
            "".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8"
        )
        stream = {
            "metadata": {"format_version": 1, "mode": "block", "num_samples": 2, "block_size": 64},
            "references": [[0, 0], [1, 3]],
        }
        encoded = json.dumps(stream, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        stream["metadata"]["fingerprint"] = fingerprint
        stream["fingerprint"] = fingerprint
        (run_dir / "stream.json").write_text(json.dumps(stream), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps({
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run_id": "baseline",
            "num_samples": 2,
            "micro_accuracy": 1.0,
            "macro_domain_accuracy": 1.0,
            "worst_domain_accuracy": 1.0,
            "domain_accuracies": {"domain-0": 1.0, "domain-1": 1.0},
            "domain_sample_counts": {"domain-0": 1, "domain-1": 1},
            "sliding_window": {
                "window_size": 1,
                "stride": 1,
                "values": [
                    {"start_timestep": 0, "end_timestep": 0, "accuracy": 1.0},
                    {"start_timestep": 1, "end_timestep": 1, "accuracy": 1.0},
                ],
            },
            "stream_fingerprint": fingerprint,
        }), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "run_id": "baseline",
            "args": {
                "dataset": "DomainNet", "model": "clip_vitbase16", "device": "cpu",
                "data_root": data_root, "tta_mode": "mixed", "batch_size": 2,
                "metric_window_size": 1, "metric_window_stride": 1,
                "stream_block_size": 64,
                "artifact_provenance": "exact", "tta_algo": "NoAdapt",
                "config_path": config_path,
            },
            "config": reference_config,
            "device": "cpu",
            "dataset": {"name": "DomainNet", "environments": ["domain-0", "domain-1"]},
            "artifacts": artifacts,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        identity = {
            "dataset": "DomainNet", "model": "clip_vitbase16", "device": "cpu",
            "data_root": data_root, "tta_mode": "mixed", "batch_size": 2,
            "metric_window_size": 1, "metric_window_stride": 1,
            "stream_block_size": 64,
            "artifact_provenance": "exact", "artifacts": artifacts,
            "reference_config": reference_config, "reference_config_path": config_path,
        }
        return trace_path, fingerprint, trace_rows, stream, identity

    @staticmethod
    def _verify_reference(trace_path, fingerprint, identity):
        return verify_reference_trace_stream_fingerprint(
            trace_path, fingerprint, expected_identity=identity
        )

    def test_trace_writer_injects_version_and_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with JsonlTraceWriter(path, "run-1") as writer:
                writer.write({
                    "timestep": 0, "sample_idx": 4, "ground_truth_domain": 1,
                    "ground_truth_class": 2, "prediction": 2, "correct": True,
                    "predicted_entropy": 0.1, "inferred_context": 3,
                    "memory_size": 8, "num_active_contexts": 2, "memory_bytes": None,
                    "latency_ms": 1.5,
                })
            row = json.loads(path.read_text().strip())
            self.assertEqual(TRACE_SCHEMA_VERSION, row["schema_version"])
            self.assertEqual("run-1", row["run_id"])

    def test_trace_writer_rejects_missing_required_field(self):
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "trace.jsonl", "run") as writer:
                with self.assertRaisesRegex(ValueError, "latency_ms"):
                    writer.write({"timestep": 0, "sample_idx": 0})

    def test_trace_writer_rejects_invalid_retained_memory_bytes(self):
        record = {
            "timestep": 0, "sample_idx": 0, "ground_truth_domain": 0,
            "ground_truth_class": 0, "prediction": 0, "correct": True,
            "predicted_entropy": 0.0, "inferred_context": None,
            "memory_size": 0, "num_active_contexts": None, "memory_bytes": -1,
            "latency_ms": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "trace.jsonl", "run") as writer:
                with self.assertRaisesRegex(ValueError, "memory_bytes"):
                    writer.write(record)

    def test_trace_writer_accepts_complete_admission_evidence_and_rejects_partial(self):
        record = {
            "timestep": 0, "sample_idx": 0, "ground_truth_domain": 0,
            "ground_truth_class": 0, "prediction": 0, "correct": True,
            "predicted_entropy": 0.0, "inferred_context": None,
            "memory_size": 0, "num_active_contexts": None, "memory_bytes": None,
            "latency_ms": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            with JsonlTraceWriter(Path(directory) / "trace.jsonl", "run") as writer:
                writer.write({**record, "admission_prediction": 0,
                              "admission_normalized_entropy": .5, "admitted_to_memory": True})
            with JsonlTraceWriter(Path(directory) / "partial.jsonl", "run") as writer:
                with self.assertRaisesRegex(ValueError, "all present"):
                    writer.write({**record, "admitted_to_memory": True})

    def test_trace_writer_refuses_to_mix_with_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("existing evidence\n")
            with self.assertRaises(FileExistsError):
                JsonlTraceWriter(path, "run")

    def test_manifest_is_canonical_and_has_requested_metadata(self):
        manifest = build_run_manifest(
            run_id="r", args={"seed": 0}, config={"beta": 5}, device="cpu",
            dataset={"name": "DomainNet", "environments": ["a"]},
            stream={"mode": "block", "seed": 0}, repository=None,
            package_names=["a-package-that-does-not-exist"],
            hardware={"requested_device": "cpu", "torch": {"version": "test"}},
        )
        self.assertEqual("not-installed", manifest["packages"]["a-package-that-does-not-exist"])
        self.assertEqual(False, manifest["git"]["available"])
        self.assertEqual("block", manifest["stream"]["mode"])
        self.assertEqual("cpu", manifest["hardware"]["requested_device"])
        self.assertEqual(manifest, json.loads(json.dumps(manifest)))
        self.assertEqual("off", manifest["artifacts"]["mode"])
        self.assertEqual("unavailable", manifest["artifacts"]["model"]["status"])

    def test_manifest_preserves_explicit_artifact_evidence(self):
        artifacts = {
            "status": "verified", "mode": "fast",
            "model": {"status": "verified", "model": "clip_vitbase16"},
            "dataset": {"status": "verified", "dataset": "cifar100c", "verified_exact": False},
        }
        manifest = build_run_manifest(run_id="r", args={}, repository=None, package_names=[], artifacts=artifacts)
        self.assertEqual(artifacts, manifest["artifacts"])

    def test_artifact_provenance_uses_fake_verifier_reports(self):
        args = SimpleNamespace(
            artifact_provenance="exact", dataset="CIFAR100C", model="clip_vitbase16",
            data_root="/temporary/data",
        )
        model_report = {"model": "clip_vitbase16", "actual_sha256": "a" * 64}
        dataset_report = {"dataset": "cifar100c", "verified_exact": True, "root_digest": "b" * 64}
        with mock.patch.object(main_module, "verify_cached_clip_checkpoint", return_value=model_report) as checkpoint, \
                mock.patch.object(main_module, "verify_cifar100c_provenance", return_value=dataset_report) as dataset:
            artifacts = main_module._artifact_provenance(args)
        self.assertEqual("verified", artifacts["status"])
        self.assertEqual("exact", artifacts["mode"])
        self.assertEqual(model_report, {key: value for key, value in artifacts["model"].items() if key != "status"})
        self.assertEqual(dataset_report, {key: value for key, value in artifacts["dataset"].items() if key != "status"})
        checkpoint.assert_called_once_with("clip_vitbase16", Path.home() / ".cache" / "clip")
        dataset.assert_called_once_with(Path("/temporary/data") / "corruption" / "CIFAR-100-C", exact=True)

    def test_artifact_provenance_off_is_explicitly_unavailable(self):
        args = SimpleNamespace(artifact_provenance="off")
        artifacts = main_module._artifact_provenance(args)
        self.assertEqual("unavailable", artifacts["status"])
        self.assertEqual("off", artifacts["mode"])
        self.assertEqual("unavailable", artifacts["model"]["status"])

    def test_artifact_revalidation_rejects_loader_time_replacement(self):
        args = SimpleNamespace(artifact_provenance="fast")
        before = {"status": "verified", "mode": "fast", "model": {"actual_sha256": "a" * 64}}
        after = {"status": "verified", "mode": "fast", "model": {"actual_sha256": "b" * 64}}
        with mock.patch.object(main_module, "_artifact_provenance", return_value=after):
            with self.assertRaisesRegex(main_module.ProvenanceError, "changed while model or dataset was loading"):
                main_module._revalidate_artifact_provenance(args, before)

    def test_main_passes_verified_checkpoint_path_exactly_to_loader(self):
        class StopAfterLoader(RuntimeError):
            pass

        checkpoint = "/verified/cache/ViT-B-16.pt"
        artifacts = {
            "status": "verified", "mode": "fast",
            "model": {"status": "verified", "path": checkpoint},
            "dataset": {"status": "verified"},
        }
        args = SimpleNamespace(
            run_id="verified-run", evidence_dir="/tmp/evidence", config={},
            dataset="CIFAR100C",
        )
        with mock.patch.object(main_module, "_artifact_provenance", return_value=artifacts), \
                mock.patch.object(main_module, "get_pretrained_model", return_value=(object(), object())) as loader, \
                mock.patch.object(main_module, "get_dataset_class", side_effect=StopAfterLoader):
            with self.assertRaises(StopAfterLoader):
                main_module.main(args)
        loader.assert_called_once_with(args, verified_checkpoint_path=checkpoint)

    def test_manifest_source_fingerprint_captures_experiment_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init")
            self._git(repository, "config", "user.email", "tests@example.invalid")
            self._git(repository, "config", "user.name", "Evidence Tests")
            (repository / "src").mkdir()
            (repository / "cfg").mkdir()
            (repository / "shell").mkdir()
            (repository / "src" / "experiment.py").write_text("seed = 1\n")
            (repository / "cfg" / "experiment.yaml").write_text("method: baseline\n")
            (repository / "shell" / "run.sh").write_text("python -m experiment\n")
            (repository / "environment.yml").write_text("name: test\n")
            (repository / "environment-cuda.yml").write_text("name: test-cuda\n")
            (repository / ".gitignore").write_text("src/ignored.py\n")
            self._git(repository, "add", ".")
            self._git(repository, "commit", "-m", "initial experiment")

            def manifest():
                return build_run_manifest(
                    run_id="r", args={}, repository=repository, package_names=[]
                )

            initial = manifest()["git"]["source"]
            self.assertEqual("sha256", initial["algorithm"])
            self.assertIn("src/experiment.py", initial["files"])
            self.assertIn("cfg/experiment.yaml", initial["files"])
            self.assertIn("shell/run.sh", initial["files"])
            self.assertIn("environment.yml", initial["files"])
            self.assertIn("environment-cuda.yml", initial["files"])
            self.assertEqual(initial, manifest()["git"]["source"])

            (repository / "src" / "experiment.py").write_text("seed = 2\n")
            edited = manifest()["git"]["source"]
            self.assertNotEqual(initial["fingerprint"], edited["fingerprint"])

            (repository / "src" / "experiment.py").write_text("seed = 1\n")
            (repository / "src" / "local_override.py").write_text("enabled = True\n")
            untracked = manifest()["git"]["source"]
            self.assertIn("src/local_override.py", untracked["files"])
            self.assertNotEqual(initial["fingerprint"], untracked["fingerprint"])

            (repository / "src" / "local_override.py").unlink()
            (repository / "src" / "ignored.py").write_text("ignored = True\n")
            (repository / "src" / "__pycache__").mkdir()
            (repository / "src" / "__pycache__" / "experiment.pyc").write_bytes(b"cache")
            (repository / "notes.txt").write_text("outside the experiment allowlist\n")
            unaffected = manifest()["git"]["source"]
            self.assertEqual(initial, unaffected)
            self.assertNotIn("src/ignored.py", unaffected["files"])
            self.assertNotIn("src/__pycache__/experiment.pyc", unaffected["files"])
            self.assertNotIn("notes.txt", unaffected["files"])

    def test_manifest_source_fingerprint_rejects_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as directory, \
                tempfile.TemporaryDirectory() as external_directory:
            repository = Path(directory)
            external = Path(external_directory)
            self._git(repository, "init")
            self._git(repository, "config", "user.email", "tests@example.invalid")
            self._git(repository, "config", "user.name", "Evidence Tests")
            included_parent = repository / "src" / "package"
            included_parent.mkdir(parents=True)
            included_file = included_parent / "experiment.py"
            included_file.write_text("inside = True\n")
            self._git(repository, "add", ".")
            self._git(repository, "commit", "-m", "initial experiment")

            initial = build_run_manifest(
                run_id="r", args={}, repository=repository, package_names=[]
            )["git"]["source"]
            self.assertIn("src/package/experiment.py", initial["files"])

            included_file.unlink()
            included_parent.rmdir()
            (external / "experiment.py").write_text("external = 1\n")
            included_parent.symlink_to(external, target_is_directory=True)
            redirected = build_run_manifest(
                run_id="r", args={}, repository=repository, package_names=[]
            )["git"]["source"]
            self.assertNotIn("src/package/experiment.py", redirected["files"])

            (external / "experiment.py").write_text("external = 2\n")
            after_external_edit = build_run_manifest(
                run_id="r", args={}, repository=repository, package_names=[]
            )["git"]["source"]
            self.assertEqual(redirected, after_external_edit)

    def test_manifest_non_utf8_git_path_is_stable_and_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            self._git(repository, "init")
            self._git(repository, "config", "user.email", "tests@example.invalid")
            self._git(repository, "config", "user.name", "Evidence Tests")
            (repository / "src").mkdir()
            (repository / "src" / "placeholder.py").write_text("value = 1\n")
            self._git(repository, "add", ".")
            self._git(repository, "commit", "-m", "add source")

            raw_path = b"src/nonutf-\xff.py"
            original_lstat = Path.lstat
            original_open = Path.open
            original_run = subprocess.run

            def is_non_utf8_source(path):
                return (
                    path.parent.name == "src"
                    and os.fsencode(path.name) == b"nonutf-\xff.py"
                )

            def git_listing(arguments, **kwargs):
                if "ls-files" in arguments:
                    output = raw_path + b"\0" if "--cached" in arguments else b""
                    return subprocess.CompletedProcess(arguments, 0, stdout=output)
                return original_run(arguments, **kwargs)

            def source_lstat(path):
                if is_non_utf8_source(path):
                    return os.stat_result((stat.S_IFREG,) + (0,) * 9)
                return original_lstat(path)

            def source_open(path, *arguments, **kwargs):
                if is_non_utf8_source(path):
                    return io.BytesIO(b"value = 1\n")
                return original_open(path, *arguments, **kwargs)

            output = repository / "manifest.json"
            with mock.patch("src.evaluation.evidence.subprocess.run", git_listing), \
                    mock.patch.object(Path, "lstat", source_lstat), \
                    mock.patch.object(Path, "open", source_open):
                manifest = write_run_manifest(
                    output, run_id="r", args={}, repository=repository, package_names=[]
                )
                source = manifest["git"]["source"]
                self.assertEqual(
                    "percent-encoded-posix-bytes", source["path_encoding"]
                )
                self.assertIn("src/nonutf-%FF.py", source["files"])
                self.assertEqual(manifest, json.loads(output.read_text()))

                repeated = build_run_manifest(
                    run_id="r", args={}, repository=repository, package_names=[]
                )["git"]["source"]
                self.assertEqual(source, repeated)

    def test_atomic_write_replaces_existing_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            atomic_write_json(path, {"value": 1})
            atomic_write_json(path, {"value": 2})
            self.assertEqual({"value": 2}, json.loads(path.read_text()))

    def test_trace_comparison_requires_identical_stream_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            adapted_path = Path(directory) / "adapted.jsonl"
            reference_path = Path(directory) / "reference.jsonl"
            identities = [
                {"timestep": index, "sample_idx": index, "ground_truth_domain": 0,
                 "ground_truth_class": 1}
                for index in range(4)
            ]
            adapted = [{**identity, "correct": value} for identity, value in zip(
                identities, [True, False, False, False]
            )]
            reference = [{**identity, "correct": value} for identity, value in zip(
                identities, [True, True, False, True]
            )]
            adapted_path.write_text("".join(json.dumps(row) + "\n" for row in adapted))
            reference_path.write_text("".join(json.dumps(row) + "\n" for row in reference))
            result = compare_trace_negative_adaptation(
                adapted_path, reference_path, window_size=2, stride=2
            )
            self.assertEqual(1.0, result["value"])
            reference[0]["sample_idx"] = 99
            reference_path.write_text("".join(json.dumps(row) + "\n" for row in reference))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                compare_trace_negative_adaptation(adapted_path, reference_path, window_size=2)

    def test_reference_trace_requires_matching_verified_sibling_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, _, stream, identity = self._reference_evidence(directory)
            run_dir = trace_path.parent

            self._verify_reference(trace_path, fingerprint, identity)
            with self.assertRaisesRegex(ValueError, "does not match"):
                self._verify_reference(trace_path, "0" * 64, identity)

            stream["references"][0][1] = 99
            (run_dir / "stream.json").write_text(json.dumps(stream), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not verify"):
                self._verify_reference(trace_path, fingerprint, identity)

            (run_dir / "stream.json").unlink()
            with self.assertRaisesRegex(ValueError, "lacks sibling stream evidence"):
                self._verify_reference(trace_path, fingerprint, identity)

    def test_reference_trace_requires_noadapt_and_current_run_identity(self):
        mutations = {
            "adapted method": lambda manifest: manifest["args"].__setitem__("tta_algo", "Ramen"),
            "different model": lambda manifest: manifest["args"].__setitem__("model", "clip_vitbase32"),
            "different dataset": lambda manifest: manifest["args"].__setitem__("dataset", "CIFAR100C"),
            "different device": lambda manifest: manifest.__setitem__("device", "cuda"),
            "different data root": lambda manifest: manifest["args"].__setitem__("data_root", "/tmp/foreign-data"),
            "different tta mode": lambda manifest: manifest["args"].__setitem__("tta_mode", "single"),
            "different batch size": lambda manifest: manifest["args"].__setitem__("batch_size", 99),
            "different window": lambda manifest: manifest["args"].__setitem__("metric_window_size", 99),
            "different block size": lambda manifest: manifest["args"].__setitem__("stream_block_size", 32),
            "different config": lambda manifest: manifest.__setitem__("config", {"lr": 0.5}),
            "different provenance mode": lambda manifest: (
                manifest["args"].__setitem__("artifact_provenance", "fast"),
                manifest["artifacts"].__setitem__("mode", "fast"),
                manifest["artifacts"]["dataset"].__setitem__("verified_exact", False),
            ),
            "different checkpoint": lambda manifest: manifest["artifacts"]["model"].update({
                "expected_sha256": "d" * 64, "actual_sha256": "d" * 64,
            }),
            "different checkpoint path": lambda manifest: manifest["artifacts"]["model"].__setitem__(
                "path", str(Path("/tmp/foreign-model.pt").resolve())
            ),
            "different model trust": lambda manifest: manifest["artifacts"]["model"].__setitem__(
                "trust", "untrusted"
            ),
            "different dataset content": lambda manifest: manifest["artifacts"]["dataset"].__setitem__(
                "root_digest", "d" * 64
            ),
            "missing legacy field": lambda manifest: manifest["args"].pop("model"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
                manifest_path = trace_path.with_name("manifest.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "reference"):
                    self._verify_reference(trace_path, fingerprint, identity)

        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
            with self.assertRaisesRegex(ValueError, "expected reference identity is required"):
                verify_reference_trace_stream_fingerprint(trace_path, fingerprint)

    def test_reference_trace_rejects_alternate_and_symlinked_files(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
            alternate_path = trace_path.with_name("foreign.jsonl")
            alternate_path.write_bytes(trace_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "canonical trace.jsonl"):
                self._verify_reference(alternate_path, fingerprint, identity)

            foreign_path = Path(directory) / "foreign-trace.jsonl"
            foreign_path.write_bytes(trace_path.read_bytes())
            trace_path.unlink()
            trace_path.symlink_to(foreign_path)
            with self.assertRaisesRegex(ValueError, "regular file, not a symlink"):
                self._verify_reference(trace_path, fingerprint, identity)

    def test_reference_trace_rejects_missing_or_symlinked_sibling_evidence(self):
        for sibling in ("stream.json", "summary.json", "manifest.json"):
            with self.subTest(sibling=sibling), tempfile.TemporaryDirectory() as directory:
                trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
                sibling_path = trace_path.with_name(sibling)
                external_path = Path(directory) / f"external-{sibling}"
                sibling_path.replace(external_path)
                sibling_path.symlink_to(external_path)
                with self.assertRaisesRegex(ValueError, "regular file, not a symlink"):
                    self._verify_reference(trace_path, fingerprint, identity)

        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
            trace_path.with_name("manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "lacks sibling manifest evidence"):
                self._verify_reference(trace_path, fingerprint, identity)

    def test_reference_trace_rejects_sibling_replacement_during_open(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
            summary_path = trace_path.with_name("summary.json")
            replacement_path = Path(directory) / "replacement-summary.json"
            replacement_path.write_bytes(summary_path.read_bytes())
            original_open = os.open

            def swapped_open(path, flags, *args, **kwargs):
                if Path(path) == summary_path:
                    return original_open(replacement_path, flags, *args, **kwargs)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch("src.evaluation.evidence.os.open", swapped_open), self.assertRaisesRegex(
                ValueError, "summary evidence changed while it was being opened"
            ):
                self._verify_reference(trace_path, fingerprint, identity)

    def test_reference_trace_rejects_same_inode_symlink_substitution_during_open(self):
        for sibling in ("trace.jsonl", "summary.json"):
            with self.subTest(sibling=sibling), tempfile.TemporaryDirectory() as directory:
                trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
                target_path = trace_path.with_name(sibling)
                hardlink_path = Path(directory) / f"hardlink-{sibling}"
                os.link(target_path, hardlink_path)
                original_open = os.open
                swapped = False

                def symlink_swap(path, flags, *args, **kwargs):
                    nonlocal swapped
                    if Path(path) == target_path and not swapped:
                        swapped = True
                        target_path.unlink()
                        target_path.symlink_to(hardlink_path)
                    return original_open(path, flags, *args, **kwargs)

                with mock.patch("src.evaluation.evidence.os.open", symlink_swap), \
                        self.assertRaisesRegex(ValueError, "reference (trace|sibling summary)"):
                    self._verify_reference(trace_path, fingerprint, identity)
                self.assertTrue(swapped)

    def test_reference_trace_rejects_foreign_malformed_and_incomplete_rows(self):
        mutations = {
            "foreign run": (
                lambda rows: rows[0].__setitem__("run_id", "foreign"), "foreign run_id"
            ),
            "foreign schema": (
                lambda rows: rows[0].__setitem__("schema_version", 999), "schema_version"
            ),
            "wrong stream identity": (
                lambda rows: rows[0].__setitem__("sample_idx", 99), "sample_idx"
            ),
            "missing identity": (
                lambda rows: rows[0].pop("ground_truth_class"), "missing"
            ),
            "short trace": (lambda rows: rows.pop(), "row count"),
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                trace_path, fingerprint, rows, _, identity = self._reference_evidence(directory)
                mutate(rows)
                trace_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, error):
                    self._verify_reference(trace_path, fingerprint, identity)

        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
            trace_path.write_text("not json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                self._verify_reference(trace_path, fingerprint, identity)

    def test_reference_trace_requires_complete_current_schema_and_consistent_results(self):
        mutations = {
            "missing prediction": (
                lambda rows, _summary: rows[0].pop("prediction"), "missing: prediction"
            ),
            "flipped correct": (
                lambda rows, _summary: rows[0].__setitem__("correct", False), "correct disagrees"
            ),
            "summary disagreement": (
                lambda _rows, summary: summary.__setitem__("micro_accuracy", 0.5), "micro_accuracy disagrees"
            ),
        }
        for name, (mutate, error) in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                trace_path, fingerprint, rows, _, identity = self._reference_evidence(directory)
                summary_path = trace_path.with_name("summary.json")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                mutate(rows, summary)
                trace_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    self._verify_reference(trace_path, fingerprint, identity)

    def test_reference_trace_rejects_tampered_domain_summary_evidence(self):
        mutations = {
            "count": lambda summary: summary["domain_sample_counts"].__setitem__("domain-0", 2),
            "accuracy": lambda summary: summary["domain_accuracies"].__setitem__("domain-0", 0.5),
            "macro": lambda summary: summary.__setitem__("macro_domain_accuracy", 0.5),
            "worst": lambda summary: summary.__setitem__("worst_domain_accuracy", 0.5),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                trace_path, fingerprint, _, _, identity = self._reference_evidence(directory)
                summary_path = trace_path.with_name("summary.json")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                mutate(summary)
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    self._verify_reference(trace_path, fingerprint, identity)

    def test_verified_reference_digest_rejects_a_post_validation_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_path, fingerprint, rows, _, identity = self._reference_evidence(directory)
            reference_sha256 = self._verify_reference(trace_path, fingerprint, identity)
            adapted_path = Path(directory) / "adapted.jsonl"
            adapted_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            rows[0]["correct"] = False
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "changed after provenance validation"):
                compare_trace_negative_adaptation(
                    adapted_path,
                    trace_path,
                    window_size=1,
                    _expected_reference_sha256=reference_sha256,
                )


if __name__ == "__main__":
    unittest.main()
