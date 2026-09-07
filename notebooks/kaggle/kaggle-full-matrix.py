# %% [markdown]
# # NB-Ramen — Full canonical CIFAR-100-C trên Kaggle
#
# Notebook này **thực thi 252 full runs**: 7 phương pháp × 4 tỷ lệ OOD × 3 stream × 3 seed.
# Pin source `26a7cd7c847b6630841dbae58067e5bb124f2f9d` đã qua CUDA smoke.
# Giữ batch=100, block=64, 400 source samples/domain, **không giới hạn prefix**.
# Đây là full matrix CIFAR-100-C đã định nghĩa, không phải toàn bộ mọi benchmark
# hay toàn bộ 50.000 ảnh trong từng corruption array. DomainNet/v2-v3 robustness là nghiên cứu riêng.
#
# Bật **Internet**, chọn **GPU** (T4 đã được kiểm chứng), rồi chạy các cell theo thứ tự.
# Dùng một GPU. Giữ cùng loại GPU qua các phiên để so sánh chi phí nhất quán.
#
# Mặc định dừng giữa các run sau ngân sách **8 giờ chạy matrix**; đây là ngân sách
# do notebook chọn, không phải cam kết về giới hạn Kaggle. Một run đang chạy được
# hoàn tất trước khi dừng, nên có thể vượt ngân sách. Đặt `SESSION_HOURS=0` để không
# dừng theo thời gian; `MAX_RUNS=0` nghĩa là thực hiện mọi run còn thiếu trong ngân sách.
# Giới hạn phiên chỉ chia công việc giữa các session, không cắt ngắn stream.
#
# ZIP được cập nhật sau mỗi run hoàn tất. Tải ZIP trước khi kết thúc phiên; ZIP
# trên máy Kaggle chưa được tải/lưu ra ngoài không bảo đảm tồn tại sau khi session mất.
# Lần sau gắn ZIP đó qua **Add Input**, đặt đường dẫn tại `RESUME_ARCHIVE`, chạy từ đầu.
# Checkpoint khôi phục cùng đường dẫn `/kaggle/working`; không sửa đường dẫn trong JSON.
# Không dùng ZIP smoke làm checkpoint full.

# %%
import os, sys, json, subprocess, shutil, uuid, importlib.util
from datetime import datetime, timezone
from pathlib import Path

REVISION = "26a7cd7c847b6630841dbae58067e5bb124f2f9d"
WORK = Path("/kaggle/working")
REPO = WORK / "NB-Ramen"
PYTHON = Path("/tmp/nb-ramen-venv/bin/python")
DATA = Path("/tmp/nb-ramen-data")
EVIDENCE = WORK / f"nb-ramen-full-{REVISION[:7]}"
# Chỉ sửa ba tùy chọn này và ARCHIVE_INPUT nếu cần.
RESUME_ARCHIVE = ""  # Ví dụ /kaggle/input/my-full-checkpoint/nb-ramen-full-26a7cd7.zip
MAX_RUNS = 0        # 0 = mọi run còn thiếu; ví dụ 7 = tối đa 7 run mới trong phiên
SESSION_HOURS = 8.0 # 0 = không dừng theo thời gian; kiểm tra ngân sách giữa các run
SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
RUNTIME = EVIDENCE / "runtime" / SESSION_ID
# Tùy chọn: ví dụ "/kaggle/input/my-official-cifar100c/CIFAR-100-C.tar"
ARCHIVE_INPUT = ""
WORK.mkdir(parents=True, exist_ok=True)
checkpoint_script = WORK / "nb-ramen-full-checkpoint.py"
checkpoint_script.write_text('"""Atomic ZIP checkpoints and non-overwriting restore for Kaggle campaigns."""\n\nimport argparse\nimport hashlib\nimport json\nfrom pathlib import Path, PurePosixPath\nimport shutil\nimport stat\nimport tempfile\nimport zipfile\n\n\ndef checkpoint(evidence):\n    evidence = Path(evidence).resolve()\n    if not evidence.is_dir():\n        raise FileNotFoundError(evidence)\n    archive = evidence.with_suffix(".zip")\n    partial = archive.with_suffix(".zip.part")\n    # Keep the previous complete ZIP until the replacement is closed successfully.\n    try:\n        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as output:\n            for path in sorted(evidence.rglob("*")):\n                if path.is_symlink():\n                    raise ValueError(f"Evidence must not contain symlinks: {path}")\n                if path.is_file():\n                    output.write(path, path.relative_to(evidence.parent).as_posix())\n        partial.replace(archive)\n    except BaseException:\n        partial.unlink(missing_ok=True)\n        raise\n    return archive\n\n\ndef restore(archive, evidence):\n    archive, evidence = Path(archive), Path(evidence)\n    with archive.open("rb") as handle:\n        digest = hashlib.file_digest(handle, "sha256").hexdigest()\n    marker_name = "restored-archive.json"\n    if evidence.exists():\n        marker = evidence / marker_name\n        if marker.is_file() and json.loads(marker.read_text()).get("sha256") == digest:\n            return  # Setup rerun: retain all progress since the same restore.\n        raise FileExistsError(\n            "Evidence already exists. Clear RESUME_ARCHIVE to use it; restore never merges or overwrites runs."\n        )\n    evidence.parent.mkdir(parents=True, exist_ok=True)\n    with zipfile.ZipFile(archive) as source:\n        seen = set()\n        for entry in source.infolist():\n            parts = PurePosixPath(entry.filename).parts\n            mode = entry.external_attr >> 16\n            normalized = PurePosixPath(entry.filename).as_posix()\n            if (not parts or parts[0] != evidence.name or ".." in parts\n                    or "\\\\" in entry.filename or PurePosixPath(entry.filename).is_absolute()\n                    or stat.S_ISLNK(mode) or normalized in seen):\n                raise ValueError(f"Invalid checkpoint member: {entry.filename}")\n            seen.add(normalized)\n        if not seen:\n            raise ValueError("Checkpoint is empty")\n        required = sum(entry.file_size for entry in source.infolist())\n        if required >= shutil.disk_usage(evidence.parent).free:\n            raise OSError("Insufficient disk space to restore this checkpoint")\n        with tempfile.TemporaryDirectory(prefix="restore-full-", dir=evidence.parent) as tmp:\n            # All paths/types were checked before any extraction. CRC errors abort staging.\n            source.extractall(tmp)\n            staged = Path(tmp) / evidence.name\n            if not staged.is_dir():\n                raise ValueError("Checkpoint root must be a directory")\n            (staged / marker_name).write_text(json.dumps({"sha256": digest}, indent=2) + "\\n")\n            staged.rename(evidence)\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("action", choices=("save", "restore"))\n    parser.add_argument("evidence", type=Path)\n    parser.add_argument("--archive", type=Path)\n    args = parser.parse_args()\n    if args.action == "restore":\n        if args.archive is None:\n            parser.error("restore requires --archive")\n        restore(args.archive, args.evidence)\n        print("Checkpoint restored; each run still requires strict validation.")\n    else:\n        print(checkpoint(args.evidence))\n')
if RESUME_ARCHIVE.strip():
    subprocess.run([sys.executable, str(checkpoint_script), "restore", str(EVIDENCE),
                    "--archive", RESUME_ARCHIVE], check=True)
RUNTIME.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(REPO / "src"),
           RAMEN_REPOSITORY=str(REPO), RAMEN_REVISION=REVISION,
           RAMEN_DATA_ROOT=str(DATA), RAMEN_EVIDENCE_ROOT=str(EVIDENCE),
           RAMEN_RUNTIME_ROOT=str(RUNTIME),
           RAMEN_ARCHIVE_INPUT=ARCHIVE_INPUT, CUDA_VISIBLE_DEVICES="0",
           PYTHONUNBUFFERED="1", UV_CACHE_DIR="/tmp/nb-ramen-uv-cache")

process_script = RUNTIME / "full-run-process.py"
process_script.write_text('"""Stream subprocess logs and stop the whole experiment group on interruption."""\n\nfrom contextlib import nullcontext\nimport os\nfrom pathlib import Path\nimport signal\nimport subprocess\n\n\ndef stop_group(process):\n    try:\n        os.killpg(process.pid, signal.SIGTERM)\n    except ProcessLookupError:\n        pass\n    try:\n        process.wait(timeout=15)\n    except subprocess.TimeoutExpired:\n        pass\n    # The parent can exit before a descendant; clear surviving group members too.\n    try:\n        os.killpg(process.pid, signal.SIGKILL)\n    except ProcessLookupError:\n        pass\n    process.wait()\n\n\ndef run_logged(argv, *, env=None, log=None, cwd=None):\n    argv = [str(value) for value in argv]\n    with (Path(log).open("w") if log is not None else nullcontext(None)) as handle:\n        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,\n                                   stderr=subprocess.STDOUT, text=True, bufsize=1,\n                                   start_new_session=True)\n        try:\n            for line in process.stdout:\n                if handle is not None:\n                    handle.write(line)\n                    handle.flush()\n                print(line, end="", flush=True)\n            code = process.wait()\n        except BaseException:\n            stop_group(process)\n            raise\n        finally:\n            process.stdout.close()\n    if code:\n        # Also stop descendants of a failed driver before its caller saves a ZIP.\n        stop_group(process)\n        raise subprocess.CalledProcessError(code, argv)\n    return subprocess.CompletedProcess(argv, code)\n')
spec = importlib.util.spec_from_file_location("full_run_process", process_script)
process_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_module)

def run(argv, *, log=None, cwd=None):
    return process_module.run_logged(argv, env=env, log=log, cwd=cwd)

run(["nvidia-smi"], log=RUNTIME / "nvidia-smi.txt")
if not REPO.exists():
    run(["git", "clone", "--branch", "open-world-gradient-memory", "--single-branch",
         "https://github.com/nguyetbinh/NB-Ramen.git", REPO])
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
if dirty:
    raise RuntimeError("Checkout có thay đổi chưa commit; giữ lại các thay đổi trước khi cập nhật.")
if actual != REVISION:
    run(["git", "fetch", "origin", "open-world-gradient-memory"], cwd=REPO)
    run(["git", "checkout", "--detach", REVISION], cwd=REPO)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
assert actual == REVISION
(RUNTIME / "git-head.txt").write_text(actual + "\n")
(RUNTIME / "git-status.txt").write_text(dirty)

run([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
if not PYTHON.exists():
    run([sys.executable, "-m", "uv", "venv", "--python", "3.11", PYTHON.parent.parent])
run([sys.executable, "-m", "uv", "pip", "install", "--python", PYTHON, "pip==24.2"])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "torch==2.4.1", "torchvision==0.19.1",
     "--index-url", "https://download.pytorch.org/whl/cu121"])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "numpy==1.26.4", "pillow==10.4.0",
     "pyyaml==6.0.2", "tqdm==4.66.5",
     "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"])
run([PYTHON, "-m", "pip", "check"], log=RUNTIME / "pip-check.txt")
run([PYTHON, "-m", "pip", "freeze"], log=RUNTIME / "pip-freeze.txt")

# %% [markdown]
# ## Runtime và tests
# Mỗi session đều kiểm tra GPU, interpreter và 354 test của source đã pin trước khi chạy full.

# %%
run([PYTHON, '-c', "\nimport json, os, platform, sys\nfrom pathlib import Path\nimport torch, torchvision\nfrom importlib.metadata import distribution, version\nfrom evaluation.evidence import TRACE_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION\nassert sys.version_info[:2] == (3, 11)\nassert torch.__version__.split('+')[0] == '2.4.1'\nassert torchvision.__version__.split('+')[0] == '0.19.1'\nassert torch.version.cuda == '12.1' and torch.cuda.is_available(), 'CUDA 12.1 runtime unavailable'\nassert (TRACE_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION) == (3, 4)\nimport subprocess\nfrom runtime.experiment_matrix import build_experiment_matrix, build_command\nprobe_run = build_experiment_matrix(datasets=('CIFAR100C',), streams=('block',),\n                                   methods=('NoAdapt',), seeds=(0,), device='cuda')[0]\nchild_python = build_command(probe_run)[0]\nassert child_python == sys.executable, 'Generated command changed the virtualenv interpreter'\nsubprocess.run([child_python, '-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'], check=True)\nclip_source = json.loads(distribution('clip').read_text('direct_url.json'))\nassert clip_source['vcs_info']['commit_id'] == 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'\nfor package, expected in [('numpy','1.26.4'),('pillow','10.4.0'),('pyyaml','6.0.2'),('tqdm','4.66.5')]:\n    assert version(package) == expected, package\nidentity = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,\n                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), platform=platform.platform(),\n                total_vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,\n                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), clip_source=clip_source)\n(Path(os.environ['RAMEN_RUNTIME_ROOT'])/'device.json').write_text(json.dumps(identity,indent=2))\nprint(json.dumps(identity, indent=2))\n"], cwd=REPO, log=RUNTIME / 'runtime-check.log')

focused = ["tests.test_by_sample_normalization", "tests.test_ramen_cuda_half", "tests.test_entropy_gated_ramen", "tests.test_consensus_ramen",
           "tests.test_oracle_id_gradient_ramen", "tests.test_oracle_consensus_ramen",
           "tests.test_open_set", "tests.test_open_set_metrics", "tests.test_open_set_consensus_analysis",
           "tests.test_ordered_stream_evidence", "tests.test_experiment_matrix"]
run([PYTHON, "-m", "unittest", *focused], cwd=REPO, log=RUNTIME / "focused-tests.log")
run([PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
    cwd=REPO, log=RUNTIME / "full-tests.log")
(RUNTIME / "test-status.json").write_text(json.dumps({"focused_exit": 0, "full_exit": 0}))

# %% [markdown]
# ## Dữ liệu và checkpoint CLIP
# Tải và xác minh cùng artifact chính thức như smoke. Dữ liệu/venv cần tạo lại nếu session trước đã bị xóa.

# %%
prepare_script = RUNTIME / 'prepare-data.py'
prepare_script.write_text('"""Prepare the checksum-verified official CIFAR-100-C archive and CLIP model."""\nimport json\nimport os\nfrom pathlib import Path, PurePosixPath\nimport shutil\nimport subprocess\nimport tarfile\n\nfrom runtime.artifact_provenance import (\n    CIFAR100C_OFFICIAL_ACQUISITION, verify_official_cifar100c_archive,\n    generate_cifar100c_provenance, verify_cifar100c_provenance,\n    resolve_clip_model, verify_cached_clip_checkpoint,\n)\nfrom evaluation.evidence import atomic_write_json\n\n\ndef download(url, path):\n    path.parent.mkdir(parents=True, exist_ok=True)\n    if not path.exists():\n        partial = path.with_suffix(path.suffix + ".part")\n        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--retry-delay", "5",\n                        "--output", str(partial), url], check=True)\n        partial.replace(path)\n\n\ndef prepare():\n    data = Path(os.environ["RAMEN_DATA_ROOT"])\n    runtime = Path(os.environ["RAMEN_RUNTIME_ROOT"])\n    runtime.mkdir(parents=True, exist_ok=True)\n    archive_input = os.environ.get("RAMEN_ARCHIVE_INPUT", "").strip()\n    archive = Path(archive_input) if archive_input else data.parent / "CIFAR-100-C.tar"\n    if archive_input and not archive.is_file():\n        raise FileNotFoundError(f"Attached archive not found: {archive}")\n    if not archive_input:\n        download(CIFAR100C_OFFICIAL_ACQUISITION["url"], archive)\n    print("Verifying the official archive MD5 and size...", flush=True)\n    acquisition = verify_official_cifar100c_archive(archive)\n    atomic_write_json(runtime / "archive-acquisition.json", acquisition)\n    dataset = data / "corruption/CIFAR-100-C"\n    if not dataset.exists():\n        staging = data / "extract-staging"\n        staging.mkdir(parents=True, exist_ok=False)\n        # Permit only the expected dataset tree and regular files/directories.\n        with tarfile.open(archive) as source:\n            members = source.getmembers()\n            for member in members:\n                parts = PurePosixPath(member.name).parts\n                if (not parts or parts[0] != "CIFAR-100-C" or ".." in parts\n                        or not (member.isfile() or member.isdir())):\n                    raise RuntimeError(f"Unexpected archive entry: {member.name}")\n            source.extractall(staging, members=members, filter="data")\n        staged = staging / "CIFAR-100-C"\n        # Inventory is created only for bytes extracted from the verified archive.\n        generate_cifar100c_provenance(staged, acquisition=acquisition)\n        dataset.parent.mkdir(parents=True, exist_ok=True)\n        staged.rename(dataset)\n        staging.rmdir()\n    # On resume, never bless an arbitrary existing tree by rebuilding its sidecar.\n    dataset_provenance = verify_cifar100c_provenance(dataset, exact=True)\n    resolved = resolve_clip_model("clip_vitbase16")\n    cache = Path.home() / ".cache/clip"\n    download(resolved["url"], cache / resolved["filename"])\n    model_provenance = verify_cached_clip_checkpoint("clip_vitbase16", cache)\n    atomic_write_json(runtime / "artifact-provenance.json", {\n        "dataset": dataset_provenance, "model": model_provenance,\n    })\n    print("Official data and model verified.", flush=True)\n\n\nif __name__ == "__main__":\n    prepare()\n')

run([PYTHON, prepare_script], cwd=REPO, log=RUNTIME / "prepare-data.log")
run([PYTHON, "-m", "runtime.preflight", "--data-root", DATA, "--dataset", "CIFAR100C",
     "--deep", "--json"], cwd=REPO, log=RUNTIME / "cifar100c-deep-preflight.json")

# %% [markdown]
# ## Thực thi full matrix và resume
#
# Run NoAdapt trước các phương pháp cùng cell. Mỗi run hoàn tất phải qua strict
# validation và khớp revision/config/fingerprint baseline, rồi mới ghi checkpoint.
# Các run đủ summary nhưng không hợp lệ gây lỗi, không bị bỏ qua.
# Run bị ngắt và chưa có summary được chuyển vào `interrupted/` để giữ chẩn đoán,
# sau đó chạy lại nguyên run. Không nối tiếp từ giữa một stream.
#
# Logs từng run ở `runtime/<session>/`; status toàn matrix ở `status.json`.
# Khi đạt **252/252**, tạo `analysis/open-set-consensus.json`, `per-cell-metrics.csv`
# và README. Chỉ `status="complete"` cùng `full_matrix_complete=true` mới là full evidence.
# Trạng thái `paused_session_budget` cần tiếp tục ở phiên sau.

# %%
full_script = RUNTIME / 'run-full-matrix.py'
full_script.write_text('"""Run the frozen canonical 252-run matrix with per-run checkpointing."""\n\nimport argparse\nimport csv\nimport importlib.util\nimport json\nimport os\nfrom pathlib import Path\nimport subprocess\nimport sys\nimport time\n\nfrom evaluation.evidence import atomic_write_json, _source_tree_fingerprint\nfrom evaluation.open_set_consensus_analysis import analyse_open_set_completed_runs\nfrom runtime.experiment_matrix import build_canonical_open_set_evidence_matrix, build_command, validate_completed_run\n\n\nREVISION = "26a7cd7c847b6630841dbae58067e5bb124f2f9d"\n\n\ndef load_checkpoint_helper():\n    path = Path(__file__).with_name("full-run-checkpoint.py")\n    spec = importlib.util.spec_from_file_location("full_run_checkpoint", path)\n    module = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(module)\n    return module.checkpoint\n\n\ndef verify_run(run, source_identity, baselines):\n    evidence = validate_completed_run(run)\n    git = evidence["manifest"]["git"]\n    if git.get("commit") != REVISION or git.get("dirty") is not False or git.get("source") != source_identity:\n        raise RuntimeError(f"Run source identity differs from the frozen revision: {run.run_id}")\n    fingerprint = evidence["summary"]["stream_fingerprint"]\n    if run.reference_trace is not None:\n        if baselines.get(run.reference_trace) != fingerprint:\n            raise RuntimeError(f"Paired NoAdapt is absent or has a different stream: {run.run_id}")\n    else:\n        baselines[run.run_dir / "trace.jsonl"] = fingerprint\n    # Keep summaries/manifests for descriptive analysis, not all stream exports in RAM.\n    return {"manifest": evidence["manifest"], "summary": evidence["summary"]}\n\n\ndef run_command(command, repo, log):\n    with log.open("w") as handle:\n        process = subprocess.Popen(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT)\n        started = time.monotonic()\n        try:\n            while True:\n                try:\n                    code = process.wait(timeout=60)\n                    break\n                except subprocess.TimeoutExpired:\n                    print(f"Still running ({time.monotonic() - started:.0f}s); log: {log.name}", flush=True)\n        except BaseException:\n            process.terminate()\n            try:\n                process.wait(timeout=15)\n            except subprocess.TimeoutExpired:\n                process.kill()\n                process.wait()\n            raise\n    if code:\n        with log.open() as handle:\n            from collections import deque\n            print("".join(deque(handle, maxlen=35)), flush=True)\n        raise subprocess.CalledProcessError(code, command)\n\n\ndef write_analysis(root, completed):\n    report = analyse_open_set_completed_runs(completed)\n    if report["classification"] != "canonical_cuda_expected" or not report["coverage"]["complete"]:\n        raise RuntimeError("All 252 runs must satisfy the canonical analysis contract")\n    destination = root / "analysis"\n    destination.mkdir(exist_ok=True)\n    atomic_write_json(destination / "open-set-consensus.json", report)\n    fields = ["ood_ratio", "stream_mode", "seed", "method", "id_accuracy", "auroc", "fpr95", "h_score"]\n    with (destination / "per-cell-metrics.csv").open("w", newline="") as handle:\n        writer = csv.DictWriter(handle, fieldnames=fields)\n        writer.writeheader()\n        for cell in report["comparisons"]:\n            for method, metrics in cell["methods"].items():\n                writer.writerow({**{key: cell[key] for key in fields[:3]}, "method": method,\n                                 **{key: metrics.get(key) for key in fields[4:]}})\n    (destination / "README.md").write_text(\n        "# Complete canonical CIFAR-100-C evidence\\n\\n"\n        f"Revision: `{REVISION}`. All 252 full-stream runs passed strict validation.\\n\\n"\n        "Read open-set-consensus.json for paired metrics, oracle diagnostics, stability and costs; "\n        "per-cell-metrics.csv retains all three seeds and each stream/OOD ratio separately. "\n        "Undefined OOD detection metrics at OOD=0 remain empty/null.\\n\\n"\n        "This is descriptive evidence, not a certification that Consensus improves performance. "\n        "It covers the primary CIFAR-100-C matrix; DomainNet and split-robustness studies are separate.\\n"\n    )\n\n\ndef main():\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--plan-only", action="store_true")\n    parser.add_argument("--max-runs", type=int, default=0, help="New complete runs this invocation; 0 means all remaining.")\n    parser.add_argument("--session-hours", type=float, default=8.0, help="Stop between runs after this budget; 0 disables it.")\n    args = parser.parse_args()\n    if args.max_runs < 0 or not 0 <= args.session_hours < float("inf"):\n        parser.error("budgets must be finite and nonnegative")\n    repo = Path(os.environ["RAMEN_REPOSITORY"]).resolve()\n    data = Path(os.environ["RAMEN_DATA_ROOT"]).resolve()\n    root = Path(os.environ["RAMEN_EVIDENCE_ROOT"]).resolve()\n    runtime = Path(os.environ["RAMEN_RUNTIME_ROOT"]).resolve()\n    runtime.mkdir(parents=True, exist_ok=True)\n    runs = build_canonical_open_set_evidence_matrix(\n        data_root=data, evidence_dir=root / "canonical", config_dir=repo / "cfg",\n        device="cuda", artifact_provenance="fast",\n    )\n    atomic_write_json(runtime / "canonical-plan.json", {\n        "execute": not args.plan_only, "runs": [run.to_dict() for run in runs],\n        "commands": [build_command(run, python_executable=sys.executable) for run in runs],\n    })\n    print(f"Canonical plan: {len(runs)} full-stream runs, B=100, 400 source samples/domain.", flush=True)\n    if args.plan_only:\n        return\n    import torch\n    if not torch.cuda.is_available():\n        raise RuntimeError("A real CUDA device is required")\n    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()\n    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)\n    if head != REVISION or os.environ.get("RAMEN_REVISION") != REVISION or dirty:\n        raise RuntimeError("Use the clean frozen experiment revision")\n    tests = json.loads((runtime / "test-status.json").read_text())\n    preflight = json.loads((runtime / "cifar100c-deep-preflight.json").read_text())\n    if tests != {"focused_exit": 0, "full_exit": 0} or not preflight.get("valid"):\n        raise RuntimeError("Pass this session\'s tests and deep preflight first")\n    source_identity = _source_tree_fingerprint(repo)\n    if source_identity is None:\n        raise RuntimeError("Cannot fingerprint experiment source")\n    device = json.loads((runtime / "device.json").read_text())\n    identity = {"revision": REVISION, "data_root": str(data), "runs": 252,\n                "runtime": {key: device[key] for key in ("gpu", "torch", "torchvision", "cuda")}}\n    campaign_path = root / "campaign.json"\n    if campaign_path.exists() and json.loads(campaign_path.read_text()) != identity:\n        raise RuntimeError("Campaign identity changed; use the same revision, data path, GPU type and runtime")\n    atomic_write_json(campaign_path, identity)\n    checkpoint = load_checkpoint_helper()\n    completed, baselines = [], {}\n    state = {"status": "running", "revision": REVISION, "planned": 252, "completed": [],\n             "new_runs_this_session": 0, "session": runtime.name, "full_matrix_complete": False}\n    started = time.monotonic()\n    failed = False\n    try:\n        for index, run in enumerate(runs, 1):\n            new_run = False\n            state["current_run"] = run.run_id\n            if run.run_dir.exists() and not (run.run_dir / "summary.json").exists():\n                # Preserve interrupted work; restart this whole run, never append a partial trace.\n                interrupted = root / "interrupted" / runtime.name\n                interrupted.mkdir(parents=True, exist_ok=True)\n                preserved = interrupted / f"{run.run_id}-{time.time_ns()}"\n                run.run_dir.rename(preserved)\n                print(f"Preserved incomplete run under {preserved}", flush=True)\n            if run.run_dir.exists():\n                checked = verify_run(run, source_identity, baselines)\n                print(f"[{index}/252] validated resume: {run.method} {run.stream_mode} seed={run.seed} OOD={run.ood_ratio}", flush=True)\n            else:\n                if ((args.max_runs and state["new_runs_this_session"] >= args.max_runs)\n                        or (args.session_hours and time.monotonic() - started >= args.session_hours * 3600)):\n                    state["status"] = "paused_session_budget"\n                    break\n                if run.reference_trace is not None and run.reference_trace not in baselines:\n                    raise RuntimeError("No validated paired NoAdapt before adapted run")\n                print(f"[{index}/252] RUN {run.method} {run.stream_mode} seed={run.seed} OOD={run.ood_ratio}", flush=True)\n                atomic_write_json(root / "status.json", state)\n                run_command(build_command(run, python_executable=sys.executable), repo, runtime / f"{run.run_id}.log")\n                checked = verify_run(run, source_identity, baselines)\n                state["new_runs_this_session"] += 1\n                new_run = True\n            completed.append((run, checked))\n            state["completed"].append(run.run_id)\n            state["current_run"] = None\n            atomic_write_json(root / "status.json", state)\n            if new_run:\n                # Per-run atomic snapshots retain the last complete ZIP if the session is killed.\n                print(f"Checkpoint: {checkpoint(root)}", flush=True)\n        if len(completed) == 252:\n            write_analysis(root, completed)\n            state.update(status="complete", full_matrix_complete=True)\n    except BaseException as exc:\n        failed = True\n        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")\n        raise\n    finally:\n        atomic_write_json(root / "status.json", state)\n        try:\n            print(f"Evidence archive: {checkpoint(root)}", flush=True)\n        except Exception as exc:\n            if not failed:\n                raise\n            print(f"Checkpoint failed too: {exc}. The previous complete ZIP is retained.", flush=True)\n    print(json.dumps({key: value for key, value in state.items() if key != "completed"}, indent=2), flush=True)\n    print(f"Validated {len(completed)}/252. Download the ZIP before ending the session.", flush=True)\n\n\nif __name__ == "__main__":\n    main()\n')
checkpoint_script = RUNTIME / 'full-run-checkpoint.py'
checkpoint_script.write_text('"""Atomic ZIP checkpoints and non-overwriting restore for Kaggle campaigns."""\n\nimport argparse\nimport hashlib\nimport json\nfrom pathlib import Path, PurePosixPath\nimport shutil\nimport stat\nimport tempfile\nimport zipfile\n\n\ndef checkpoint(evidence):\n    evidence = Path(evidence).resolve()\n    if not evidence.is_dir():\n        raise FileNotFoundError(evidence)\n    archive = evidence.with_suffix(".zip")\n    partial = archive.with_suffix(".zip.part")\n    # Keep the previous complete ZIP until the replacement is closed successfully.\n    try:\n        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as output:\n            for path in sorted(evidence.rglob("*")):\n                if path.is_symlink():\n                    raise ValueError(f"Evidence must not contain symlinks: {path}")\n                if path.is_file():\n                    output.write(path, path.relative_to(evidence.parent).as_posix())\n        partial.replace(archive)\n    except BaseException:\n        partial.unlink(missing_ok=True)\n        raise\n    return archive\n\n\ndef restore(archive, evidence):\n    archive, evidence = Path(archive), Path(evidence)\n    with archive.open("rb") as handle:\n        digest = hashlib.file_digest(handle, "sha256").hexdigest()\n    marker_name = "restored-archive.json"\n    if evidence.exists():\n        marker = evidence / marker_name\n        if marker.is_file() and json.loads(marker.read_text()).get("sha256") == digest:\n            return  # Setup rerun: retain all progress since the same restore.\n        raise FileExistsError(\n            "Evidence already exists. Clear RESUME_ARCHIVE to use it; restore never merges or overwrites runs."\n        )\n    evidence.parent.mkdir(parents=True, exist_ok=True)\n    with zipfile.ZipFile(archive) as source:\n        seen = set()\n        for entry in source.infolist():\n            parts = PurePosixPath(entry.filename).parts\n            mode = entry.external_attr >> 16\n            normalized = PurePosixPath(entry.filename).as_posix()\n            if (not parts or parts[0] != evidence.name or ".." in parts\n                    or "\\\\" in entry.filename or PurePosixPath(entry.filename).is_absolute()\n                    or stat.S_ISLNK(mode) or normalized in seen):\n                raise ValueError(f"Invalid checkpoint member: {entry.filename}")\n            seen.add(normalized)\n        if not seen:\n            raise ValueError("Checkpoint is empty")\n        required = sum(entry.file_size for entry in source.infolist())\n        if required >= shutil.disk_usage(evidence.parent).free:\n            raise OSError("Insufficient disk space to restore this checkpoint")\n        with tempfile.TemporaryDirectory(prefix="restore-full-", dir=evidence.parent) as tmp:\n            # All paths/types were checked before any extraction. CRC errors abort staging.\n            source.extractall(tmp)\n            staged = Path(tmp) / evidence.name\n            if not staged.is_dir():\n                raise ValueError("Checkpoint root must be a directory")\n            (staged / marker_name).write_text(json.dumps({"sha256": digest}, indent=2) + "\\n")\n            staged.rename(evidence)\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("action", choices=("save", "restore"))\n    parser.add_argument("evidence", type=Path)\n    parser.add_argument("--archive", type=Path)\n    args = parser.parse_args()\n    if args.action == "restore":\n        if args.archive is None:\n            parser.error("restore requires --archive")\n        restore(args.archive, args.evidence)\n        print("Checkpoint restored; each run still requires strict validation.")\n    else:\n        print(checkpoint(args.evidence))\n')

try:
    run([PYTHON, full_script, "--max-runs", str(MAX_RUNS), "--session-hours", str(SESSION_HOURS)],
        cwd=REPO, log=RUNTIME / "full-execution.log")
finally:
    run([PYTHON, checkpoint_script, "save", EVIDENCE])

# %% [markdown]
# ## Tải evidence / tiếp tục phiên sau
#
# Tải `nb-ramen-full-26a7cd7.zip`. Để resume phiên mới, upload ZIP đã tải thành Kaggle
# Input và điền `RESUME_ARCHIVE` ở cell đầu. Nếu đang tiếp tục ngay trong phiên
# hiện tại, để trống `RESUME_ARCHIVE`: notebook dùng evidence đang có.
# File Input cần là ZIP full đúng tên root, không phải ZIP smoke hay chỉ vài summary.
# Nếu Input đã được giải nén tự động, nén lại cây `nb-ramen-full-26a7cd7/` thành ZIP
# với đúng một root đó trước khi dùng. Không merge các checkpoint bằng tay.
#
# Ngân sách/hết phiên có thể buộc chạy lại run đang dở, nhưng các run đã checkpoint
# và được validator chấp nhận sẽ được giữ. Sau 252/252, gửi ZIP để đánh giá kết quả.

# %%
status_path = EVIDENCE / "status.json"
if status_path.exists():
    status = json.loads(status_path.read_text())
    print("Status:", status["status"], "| Validated:", len(status["completed"]), "/", status["planned"])
    if status.get("error"):
        print(status["error"])
archive = EVIDENCE.with_suffix(".zip")
if not archive.exists():
    run([sys.executable, checkpoint_script, "save", EVIDENCE])
from IPython.display import display, FileLink
display(FileLink(str(archive)))
