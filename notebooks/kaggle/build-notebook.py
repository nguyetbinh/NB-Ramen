import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
cells = []


def add(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add("markdown", """
# NB-Ramen — Pre-full CUDA smoke trên Kaggle

Trong Settings bật **Internet** và chọn **GPU** rồi chạy các cell theo thứ tự.
Notebook tải đúng commit `26a7cd7c847b6630841dbae58067e5bb124f2f9d`, tạo Python 3.11
riêng với PyTorch 2.4.1/cu121 và CLIP commit cố định. Không cần restart kernel.

Chạy **14 smoke**, mỗi run 256 mẫu, gồm bảy phương pháp chính và các control.
Đây là evidence **noncanonical**. Full matrix 252 runs chỉ được dry-plan, không chạy.
Batch size 100 giữ nguyên. Code dùng một GPU; hai GPU không tự cộng VRAM.
Nếu thiếu VRAM, notebook ghi lỗi và dừng, không tự giảm batch hoặc precision.

Có thể điền đường dẫn archive chính thức `CIFAR-100-C.tar` trong Kaggle Input
ở cell cấu hình; để trống để tải từ Zenodo. Archive phải khớp checksum chính thức.
Data và môi trường nằm trong `/tmp`; evidence ở `/kaggle/working/nb-ramen-evidence-<revision>`.
Nếu phiên dừng giữa một run, thư mục dở dang được giữ lại và từ chối resume.
Đổi tên thư mục run lỗi để lưu chẩn đoán rồi chạy lại cell smoke; không sửa JSON.
Sau khi khởi động phiên Kaggle mới cần chạy lại setup/data. Resume trong cùng
phiên sẽ skip những run đã được strict validator chấp nhận.

Nguồn thiết lập: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks),
[PyTorch 2.4.1](https://docs.pytorch.org/get-started/previous-versions/),
[uv environments](https://docs.astral.sh/uv/pip/environments/).
""")
add("code", r'''
import os, sys, json, subprocess, shutil
from pathlib import Path

REVISION = "26a7cd7c847b6630841dbae58067e5bb124f2f9d"
WORK = Path("/kaggle/working")
REPO = WORK / "NB-Ramen"
PYTHON = Path("/tmp/nb-ramen-venv/bin/python")
DATA = Path("/tmp/nb-ramen-data")
EVIDENCE = WORK / f"nb-ramen-evidence-{REVISION[:7]}"
RUNTIME = EVIDENCE / "runtime"
# Tùy chọn: ví dụ "/kaggle/input/my-official-cifar100c/CIFAR-100-C.tar"
ARCHIVE_INPUT = ""
WORK.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(REPO / "src"),
           RAMEN_REPOSITORY=str(REPO), RAMEN_REVISION=REVISION,
           RAMEN_DATA_ROOT=str(DATA), RAMEN_EVIDENCE_ROOT=str(EVIDENCE),
           RAMEN_ARCHIVE_INPUT=ARCHIVE_INPUT, CUDA_VISIBLE_DEVICES="0",
           PYTHONUNBUFFERED="1", UV_CACHE_DIR="/tmp/nb-ramen-uv-cache")

def run(argv, *, log=None, cwd=None):
    argv = [str(x) for x in argv]
    if log is None:
        return subprocess.run(argv, cwd=cwd, env=env, check=True)
    with Path(log).open("w") as handle:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, argv)

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
''')
add("markdown", """## Kiểm tra runtime và tests
Cell phải hoàn tất thành công trước khi tải data/chạy smoke.
""")
probe = '''
import json, os, platform, sys
from pathlib import Path
import torch, torchvision
from importlib.metadata import distribution, version
from evaluation.evidence import TRACE_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.split('+')[0] == '2.4.1'
assert torchvision.__version__.split('+')[0] == '0.19.1'
assert torch.version.cuda == '12.1' and torch.cuda.is_available(), 'CUDA 12.1 runtime unavailable'
assert (TRACE_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION) == (3, 4)
import subprocess
from runtime.experiment_matrix import build_experiment_matrix, build_command
probe_run = build_experiment_matrix(datasets=('CIFAR100C',), streams=('block',),
                                   methods=('NoAdapt',), seeds=(0,), device='cuda')[0]
child_python = build_command(probe_run)[0]
assert child_python == sys.executable, 'Generated command changed the virtualenv interpreter'
subprocess.run([child_python, '-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'], check=True)
clip_source = json.loads(distribution('clip').read_text('direct_url.json'))
assert clip_source['vcs_info']['commit_id'] == 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
for package, expected in [('numpy','1.26.4'),('pillow','10.4.0'),('pyyaml','6.0.2'),('tqdm','4.66.5')]:
    assert version(package) == expected, package
identity = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
                total_vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), clip_source=clip_source)
(Path(os.environ['RAMEN_EVIDENCE_ROOT'])/'runtime/device.json').write_text(json.dumps(identity,indent=2))
print(json.dumps(identity, indent=2))
'''
add("code", "run([PYTHON, '-c', " + repr(textwrap.dedent(probe)) + "], cwd=REPO, log=RUNTIME / 'runtime-check.log')\n" + '''
focused = ["tests.test_by_sample_normalization", "tests.test_ramen_cuda_half", "tests.test_entropy_gated_ramen", "tests.test_consensus_ramen",
           "tests.test_oracle_id_gradient_ramen", "tests.test_oracle_consensus_ramen",
           "tests.test_open_set", "tests.test_open_set_metrics", "tests.test_open_set_consensus_analysis",
           "tests.test_ordered_stream_evidence", "tests.test_experiment_matrix"]
run([PYTHON, "-m", "unittest", *focused], cwd=REPO, log=RUNTIME / "focused-tests.log")
run([PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
    cwd=REPO, log=RUNTIME / "full-tests.log")
(RUNTIME / "test-status.json").write_text(json.dumps({"focused_exit": 0, "full_exit": 0}))
''')
add("markdown", """## Data chính thức và model
Tải archive khoảng 2.92 GB nếu chưa gắn Kaggle Input. Xác minh MD5 archive,
giải nén, tạo inventory SHA-256, xác minh checkpoint, rồi chạy deep preflight.
Không dùng CIFAR-100 sạch hoặc bộ corruption tự tạo thay cho CIFAR-100-C.
""")
prepare = (ROOT / "prepare-data.py").read_text()
add("code", "prepare_script = RUNTIME / 'prepare-data.py'\nprepare_script.write_text(" + repr(prepare) + ")\n" + '''
run([PYTHON, prepare_script], cwd=REPO, log=RUNTIME / "prepare-data.log")
run([PYTHON, "-m", "runtime.preflight", "--data-root", DATA, "--dataset", "CIFAR100C",
     "--deep", "--json"], cwd=REPO, log=RUNTIME / "cifar100c-deep-preflight.json")
''')
add("markdown", """## Chạy 14 smoke và strict validation
Giữ batch=100, block=64, source budget=400/domain, prefix=256, CUDA, fast provenance.
Chạy tuần tự trên GPU 0; mỗi method chạy trong process riêng để giải phóng VRAM.
NoAdapt B=1 được tạo riêng cho Ramen B=1. V2 có split byte lock.
Mỗi run có log riêng trong `runtime/prefull-*.log`.

Notebook chỉ đánh dấu artifact smoke hợp lệ sau strict validation. Kết quả
causal sensitivity vẫn cần đọc/đánh giá; không tự công nhận toàn bộ pre-full gate.
""")
smoke = (ROOT / "run-smokes.py").read_text()
add("code", "smoke_script = RUNTIME / 'run-smokes.py'\nsmoke_script.write_text(" + repr(smoke) + ")\n" + '''
# Log và script được giữ ngoài checkout để Git identity của thí nghiệm sạch.
try:
    run([PYTHON, smoke_script], cwd=REPO, log=RUNTIME / "smoke-execution.log")
finally:
    archive_path = shutil.make_archive(str(EVIDENCE), "zip",
                                       root_dir=WORK, base_dir=EVIDENCE.name)
    print("Evidence archive:", archive_path)
''')
add("markdown", """## Xem kết quả và tải evidence
Cell này có thể chạy riêng sau lỗi để đóng gói cả log chẩn đoán. Tải ZIP về
trước khi kết thúc phiên; dùng **Save Version** để lưu notebook/output.
Nếu GPU hết VRAM, giữ log và chuyển sang runner đủ VRAM cho batch=100.
Không giảm batch trong cùng run ID rồi gọi kết quả đó là cùng smoke.

File cần đọc: `runtime/smoke-status.json`, `runtime/causal-sensitivity.json`,
`runtime/device.json` và các `summary.json`. Gửi ZIP để kiểm tra trước full run.
""")
add("code", '''
for name in ("smoke-status.json", "causal-sensitivity.json"):
    p = RUNTIME / name
    if p.exists():
        print(name, p.read_text())
archive_path = shutil.make_archive(str(EVIDENCE), "zip",
                                   root_dir=WORK, base_dir=EVIDENCE.name)
from IPython.display import display, FileLink
display(FileLink(archive_path))
''')
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}, "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells):
    cell["id"] = f"ramen-{i:02d}"
(ROOT / "kaggle-pre-full-smoke.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
export = []
for cell in cells:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'code':
        export.append('# %%\n' + source)
    else:
        export.append('# %% [markdown]\n' + '\n'.join(('# ' + line if line else '#') for line in source.splitlines()) + '\n')
(ROOT / "kaggle-pre-full-smoke.py").write_text('\n'.join(export))
print('Wrote notebook and Python export.')
