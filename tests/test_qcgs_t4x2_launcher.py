"""CPU-only operational tests; never launch inference or inspect real CUDA."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "notebooks/kaggle/launch-qcgs-stage-a-t4x2.py"
SPEC = importlib.util.spec_from_file_location("qcgs_t4x2_launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def registry():
    return {"sha256": "f" * 64, "source_revision": launcher.REVISION,
            "provenance": {"action_chunk_size": 1},
            "cells": {cell: {"selected": {"stage-a": [
                {"timestep": i, "sample_idx": i} for i in range(16)]}}
                      for cell, _ in launcher.CELLS}}


class LauncherTests(unittest.TestCase):
    def test_environment_isolates_gpu_and_overrides_threads(self):
        with patch.dict(launcher.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "8"}):
            env = launcher.child_env(Path("/frozen"), "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["PYTHONPATH"], "/frozen/src")
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(env["PYTHONHASHSEED"], "0")
        self.assertTrue(all(env[key] == "1" for key in launcher.THREAD_VARS))

    def test_source_rejects_wrong_revision_and_untracked_files(self):
        for outputs in (("wrong",), (launcher.REVISION, "?? launcher.py")):
            with patch.object(launcher.subprocess, "check_output", side_effect=outputs):
                with self.assertRaises(ValueError):
                    launcher.verify_source(Path("/frozen"))

    def test_operational_paths_cannot_dirty_source(self):
        with self.assertRaises(ValueError):
            launcher.outside(Path("/frozen"), [Path("/frozen/log")])
        launcher.outside(Path("/frozen"), [Path("/external/log")])

    def test_runtime_rejects_busy_physical_t4_without_inference(self):
        with patch.object(launcher.subprocess, "check_output", side_effect=[
                '{"device":"Tesla T4"}', "0, GPU-id, Tesla T4, 15360, 900, 20, 570"]):
            with self.assertRaises(ValueError):
                launcher.runtime_probe(Path("/frozen"), "0")

    def test_prepare_maps_cells_and_uses_separate_unique_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(repo=Path("/frozen"), output_root=root / "out",
                                   data_root=Path("/data"), smoke_registry=root / "smoke.json",
                                   artifact_receipt=root / "artifacts.json")
            launcher.write(args.smoke_registry, {"smoke": True})
            launcher.write(args.artifact_receipt, {"status": "verified"})
            def make_job(stage, cell, config, **kwargs):
                self.assertEqual(stage, "stage-a")
                self.assertEqual(kwargs["action_chunk_size"], 1)
                self.assertEqual(kwargs["registry"], registry())
                return {"cell": cell, "run_id": kwargs["run_id"], "command": ["python", "/frozen/src/main.py"]}
            with patch.object(launcher, "verify_source", return_value={"revision": launcher.REVISION}), \
                    patch.object(launcher, "verify_inputs", return_value=(registry(), {})), \
                    patch.object(launcher, "runtime_probe", return_value={"device": "T4"}), \
                    patch("runtime.qcgs_exact_objective_audit.make_job", side_effect=make_job):
                _, plan = launcher.prepare(args)
                _, second = launcher.prepare(args)
            jobs = plan["jobs"]
            self.assertEqual([(j["cell"], j["physical_gpu"]) for j in jobs], [("0.5", "0"), ("0", "1")])
            self.assertEqual(len({j["run_id"] for j in jobs + second["jobs"]}), 4)
            self.assertNotEqual(jobs[0]["evidence_root"], jobs[1]["evidence_root"])
            self.assertEqual(jobs[0]["command"][-2:], ["--num_threads", "1"])
            self.assertIsNone(plan["scientific_decision"])

    def test_audit_rejects_partial_and_truncated_durable_records(self):
        rows = [{"query": {"timestep": i, "sample_idx": i}} for i in range(16)]
        provenance = {"stage": "stage-a", "ood_cell": "0", "source_revision": launcher.REVISION,
                      "registry_sha256": registry()["sha256"], "scorer_complete": True,
                      "evaluator_complete": True, "trace_complete": True, "interrupted_jsonl_artifacts": []}
        job = {"evidence_root": "/unused", "run_id": "test", "cell": "0"}
        receipt = {"errors": [], "truncated_tail": False, "sha256": "a" * 64}
        with patch("evaluation.qcgs_exact_objective_audit.audit_raw") as audit, \
                patch("evaluation.qcgs_exact_objective_audit.audit_run_provenance", return_value=provenance), \
                patch("evaluation.qcgs_exact_objective_audit.read_jsonl_detailed") as reader:
            reader.side_effect = [(rows, receipt), (rows, receipt)]
            self.assertEqual(launcher.audit_cell(job, registry())["provenance"], provenance)
            audit.assert_called_once_with(rows, rows, registry())
            reader.side_effect = [(rows[:-1], receipt), (rows[:-1], receipt)]
            with self.assertRaises(ValueError):
                launcher.audit_cell(job, registry())
            reader.side_effect = [(rows, {**receipt, "truncated_tail": True}), (rows, receipt)]
            with self.assertRaises(ValueError):
                launcher.audit_cell(job, registry())

    def test_worker_preserves_nonzero_exit_and_writes_failed_receipt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config"
            config.write_text("locked")
            snapshot = root / "registry.json"
            launcher.write(snapshot, registry())
            job = {"run_id": "failed-run", "cell": "0", "physical_gpu": "1",
                   "config": str(config), "config_sha256": launcher.sha(config),
                   "log": str(root / "inference.log"), "receipt": str(root / "completion.json"),
                   "command": ["no-real-inference"]}
            plan = root / "launch.json"
            launcher.write(plan, {"repo": "/frozen", "jobs": [job], "registry": registry(),
                                 "registry_snapshot": str(snapshot), "launcher_sha256": launcher.sha(PATH)})
            with patch.object(launcher, "verify_source"), \
                    patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run, \
                    patch.object(launcher, "audit_cell", side_effect=ValueError("partial records")):
                self.assertEqual(launcher.worker(plan, 0), 1)
            result = launcher.read(job["receipt"])
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["scientific_decision"])
            self.assertIn("partial records", result["error"])
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            # The original failure receipt/log survive an accidental repeat.
            original = Path(job["receipt"]).read_bytes()
            with patch.object(launcher, "verify_source"), patch.object(launcher.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    launcher.worker(plan, 0)
                run.assert_not_called()
            self.assertEqual(Path(job["receipt"]).read_bytes(), original)

    def test_duplicate_live_launcher_is_rejected_before_preparation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "frozen"
            repo.mkdir()
            digest = launcher.hashlib.sha256(str(repo).encode()).hexdigest()[:16]
            lock_path = root / (".qcgs-stage-a-" + digest + ".lock")
            with lock_path.open("a+") as lock, patch.object(launcher, "prepare") as prepare:
                launcher.fcntl.flock(lock, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "live"):
                    launcher.main(["--repo", str(repo), "--prepare-only"])
                prepare.assert_not_called()

    def test_prepare_only_does_not_start_workers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "frozen"
            repo.mkdir()
            with patch.object(launcher, "prepare", return_value=(root / "launch.json", {"jobs": []})), \
                    patch.object(launcher.subprocess, "Popen") as start, patch("builtins.print"):
                self.assertEqual(launcher.main(["--repo", str(repo), "--prepare-only"]), 0)
                start.assert_not_called()

    def test_artifact_receipt_accepts_main_schema_and_rejects_fingerprint_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = {"sha256": "s" * 64, "cells": {cell: {"selected": [{"sample_idx": 100}]}
                                                   for cell, _ in launcher.CELLS}}
            value = registry()
            value.update(dataset_fingerprint="d" * 64, exclusions={"smoke_base_image_indices": [100]})
            value["provenance"]["smoke_registry_sha256"] = smoke["sha256"]
            dataset = {"root_digest": "d" * 64, "sidecar_sha256": "a" * 64, "file_count": 20,
                       "acquisition": {"source": "verified"}, "verified_exact": False}
            model = {"actual_sha256": "c" * 64, "path": "/cache/checkpoint"}
            args = SimpleNamespace(registry=root / "registry.json", smoke_registry=root / "smoke.json",
                                   artifact_receipt=root / "artifacts.json", data_root=root / "data")
            launcher.write(args.registry, value)
            launcher.write(args.smoke_registry, smoke)
            receipt = {"status": "verified", "dataset": {**dataset, "status": "verified", "verified_exact": True},
                       "model": {**model, "status": "verified"}}
            launcher.write(args.artifact_receipt, receipt)
            with patch("runtime.query_gradient_registry.validate_exact_objective_registry"), \
                    patch("runtime.qcgs_exact_objective_audit.validate_smoke_registry"), \
                    patch("runtime.artifact_provenance.verify_dataset_provenance", return_value=dataset), \
                    patch("runtime.artifact_provenance.verify_cached_clip_checkpoint", return_value=model):
                self.assertEqual(launcher.verify_inputs(args)[0], value)
                receipt["dataset"]["root_digest"] = "bad"
                launcher.write(args.artifact_receipt, receipt)
                with self.assertRaisesRegex(ValueError, "verification mismatch"):
                    launcher.verify_inputs(args)


if __name__ == "__main__":
    unittest.main()
