"""Embed the current committed source (including ancestry) in one Kaggle notebook."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import textwrap

ROOT=Path(__file__).resolve().parents[2]


def build(revision, bundle):
    checksum=hashlib.sha256(bundle).hexdigest()
    def cell(kind,source):
        result={'cell_type':kind,'metadata':{},'source':textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind=='code': result.update(execution_count=None,outputs=[])
        return result
    cells=[cell('markdown',f'''# QCGS label-free diagnostic — Stage A → Stage B

Bật **Internet** và **GPU T4**, rồi chạy từ trên xuống hoặc **Save Version → Save & Run All**.
Notebook này chỉ chạy diagnostic: smoke, query registry, 16 query ID mới/cell ở Stage A và 128/cell ở Stage B (OOD 0 và 0.5).
Source cố định `{revision}`, có sẵn trong notebook; không cần push nhánh lên GitHub.

**Chưa phải evidence thực nghiệm:** mọi output hiện để trống. Runner sẽ kiểm tra CPU, CUDA, dữ liệu/model, rồi mới thu kết quả.
`/kaggle/working/qcgs-label-free-evidence.zip` được thay thế atomically sau mỗi cell và khoảng 30 giây trong lúc chạy.
Nếu bị ngắt, tải ZIP, thêm ZIP làm Kaggle Input ở phiên mới và điền `RESUME_ARCHIVE` bên dưới. Chạy lại nguyên cell chưa hoàn tất; không ghép phần kết quả dở dang.
Giữ nguyên source và các tham số khi resume. Môi trường/GPU khác sẽ bị từ chối nếu không khớp provenance đã khóa.

Stage A có exact oracle. Stage B chỉ có best-verified subset. Không sửa score/ngưỡng theo kết quả.
''')]
    setup='''import base64, hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

RESUME_ARCHIVE = ""  # Ví dụ: /kaggle/input/my-qcgs-checkpoint/qcgs-label-free-evidence.zip
REPO = Path("/tmp/NB-Ramen-QCGS")
PYTHON = Path("/tmp/nb-ramen-qcgs-venv/bin/python")
DATA = Path("/tmp/nb-ramen-qcgs-data")
EVIDENCE = Path("/kaggle/working/qcgs-label-free-evidence")
RUNTIME = EVIDENCE / "runtime"
REVISION = __REVISION__
BUNDLE_SHA256 = __CHECKSUM__
SOURCE_BUNDLE = __PAYLOAD__

def run(command, *, env=None, log=None):
    command = [str(x) for x in command]
    if log is None:
        subprocess.run(command, check=True, cwd=REPO if REPO.exists() else None, env=env)
    else:
        with Path(log).open("w") as handle:
            subprocess.run(command, check=True, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)

bundle = Path("/tmp/qcgs-source.bundle")
raw = base64.b64decode(SOURCE_BUNDLE, validate=True)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256
bundle.write_bytes(raw)
if not REPO.exists():
    run(["git", "clone", "--no-checkout", bundle, REPO])
    run(["git", "checkout", "--detach", REVISION])
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == REVISION
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
run(["git", "merge-base", "--is-ancestor", "3a80623b074f16b8ef87d8d5507427277ea2a54f", REVISION])

if RESUME_ARCHIVE:
    spec = importlib.util.spec_from_file_location("checkpoint", REPO / "notebooks/kaggle/full-run-checkpoint.py")
    support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)
    support.restore(RESUME_ARCHIVE, EVIDENCE)
RUNTIME.mkdir(parents=True, exist_ok=True)
(RUNTIME / "source-bundle.json").write_text(json.dumps({"revision": REVISION, "bundle_sha256": BUNDLE_SHA256}, indent=2))
print("Pinned source ready:", REVISION)
'''
    setup=setup.replace('__REVISION__',repr(revision)).replace('__CHECKSUM__',repr(checksum)).replace('__PAYLOAD__',repr(base64.b64encode(bundle).decode()))
    cells.append(cell('code',setup))
    cells.append(cell('code','''
# Môi trường riêng: không thay Torch của kernel Kaggle.
run([sys.executable, "-m", "pip", "install", "--quiet", "uv"])
if not PYTHON.exists():
    run([sys.executable, "-m", "uv", "venv", "--python", "3.11", PYTHON.parent.parent])
run([sys.executable, "-m", "uv", "pip", "install", "--python", PYTHON, "pip==24.2"])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "torch==2.4.1", "torchvision==0.19.1",
     "--index-url", "https://download.pytorch.org/whl/cu121"])
run([PYTHON, "-m", "pip", "install", "--no-cache-dir", "numpy==1.26.4", "pillow==10.4.0", "pyyaml==6.0.2",
     "tqdm==4.66.5", "pyarrow==18.1.0", "huggingface-hub==0.26.2", "pytest==8.2.2",
     "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"])
run([PYTHON, "-m", "pip", "check"], log=RUNTIME / "pip-check.txt")
run([PYTHON, "-m", "pip", "freeze"], log=RUNTIME / "pip-freeze.txt")
env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
           CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
           RAMEN_DATA_ROOT=str(DATA), RAMEN_RUNTIME_ROOT=str(RUNTIME))
run([PYTHON, "-c", "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'Enable Kaggle GPU first'; print(torch.cuda.get_device_name(0))"], env=env)
'''))
    cells.append(cell('code','''
# CPU tests trước khi tải dữ liệu / thu bất kỳ utility nào.
# GPU runner cũng xác minh receipt/tests trước CUDA smoke.
run([PYTHON, REPO / "scripts/run-oracle-support-utility.py", "--diagnostic", "qcgs",
     "--preflight-only", "--evidence-dir", RUNTIME / "cpu-preflight"], env=env)
'''))
    cells.append(cell('code','''
# Dùng pipeline Hugging Face đã có; kiểm tra checksum gốc của toàn bộ 20 NPY và CLIP.
shutil.copyfile(REPO / "notebooks/kaggle/prepare-data.py", RUNTIME / "download-support.py")
run([PYTHON, REPO / "notebooks/kaggle/prepare-huggingface-data.py"], env=env)
'''))
    cells.append(cell('code','''
# Tự chạy đúng thứ tự: CPU checks → smoke → exclusions/registry → A → bins → B → audit/report.
# Không đổi timeout sau khi campaign đã khóa. Mặc định 3600 giây mỗi cell.
command = [PYTHON, REPO / "scripts/run-oracle-support-utility.py", "--diagnostic", "qcgs",
           "--execute", "--data-root", DATA, "--evidence-dir", EVIDENCE]
# runtime chỉ chứa setup; --resume dùng lại campaign nếu đã có preflight lock.
if (EVIDENCE / "locks").exists():
    command.append("--resume")
run(command, env=env)
'''))
    cells.append(cell('code','''
# Audit độc lập từ raw records; không fit lại score/ngưỡng.
run([PYTHON, REPO / "scripts/run-oracle-support-utility.py", "--diagnostic", "qcgs",
     "--audit", "--evidence-dir", EVIDENCE], env=env, log=RUNTIME / "audit.log")
run([PYTHON, REPO / "notebooks/kaggle/full-run-checkpoint.py", "save", EVIDENCE], env=env)
from IPython.display import FileLink, Markdown, display
display(Markdown((EVIDENCE / "report.md").read_text()))
display(FileLink(str(EVIDENCE.with_suffix(".zip"))))
print("Nếu Save & Run All: tải qcgs-label-free-evidence.zip trong tab Output.")
'''))
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'qcgs':{'source_revision':revision,'bundle_sha256':checksum,'experimental_outputs':False}},'cells':cells}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('kaggle-qcgs-label-free.ipynb'))
    args=parser.parse_args()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts','cfg','tests','pytest.ini',
                    'docs/research/query-conditioned-gradient-selection.md','notebooks/kaggle/build-qcgs-notebook.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory() as tmp:
        bundle=Path(tmp)/'source.bundle'
        subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=ROOT,check=True)
        notebook=build(revision,bundle.read_bytes())
    for i,c in enumerate(notebook['cells']):
        c['id']=f'qcgs-{i}'
        if c['cell_type']=='code': compile(''.join(c['source']),f'cell-{i}','exec')
    args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print(args.output,revision,args.output.stat().st_size)


if __name__=='__main__': main()
