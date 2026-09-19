"""Build size-checked Kaggle notebooks with an immutable Git source revision."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

ROOT=Path(__file__).resolve().parents[2]
MAX_NOTEBOOK_BYTES=1_000_000
SOURCE_REPOSITORY='https://github.com/nguyetbinh/NB-Ramen'


def build(revision, bundle=None, *, rescue=False, repository_url=None, source_tree=None):
    if (bundle is None) == (repository_url is None):
        raise ValueError('provide exactly one source transport: bundle or repository URL')
    if repository_url is not None and not source_tree:
        raise ValueError('Git source requires the expected tree hash')
    source_info={'source_revision':revision}
    if bundle is not None:
        checksum=hashlib.sha256(bundle).hexdigest()
        source_info['bundle_sha256']=checksum
        constants='BUNDLE_SHA256 = '+repr(checksum)+'\nSOURCE_BUNDLE = '+repr(base64.b64encode(bundle).decode())
        restore='''bundle = Path("/tmp/qcgs-source.bundle")
raw = base64.b64decode(SOURCE_BUNDLE, validate=True)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256
bundle.write_bytes(raw)
if not REPO.exists():
    run(["git", "clone", "--no-checkout", bundle, REPO])
    run(["git", "checkout", "--detach", REVISION])'''
        verify=''
        receipt_name='source-bundle.json'
        receipt='{"revision": REVISION, "bundle_sha256": BUNDLE_SHA256}'
    else:
        source_info.update(source_transport='git',source_repository=repository_url,source_tree=source_tree)
        constants='SOURCE_REPOSITORY = '+repr(repository_url)+'\nSOURCE_TREE = '+repr(source_tree)
        restore='''if not REPO.exists():
    # A failed download leaves no partial checkout at REPO; rerunning is safe.
    with tempfile.TemporaryDirectory(prefix="qcgs-source-", dir=REPO.parent) as tmp:
        staged = Path(tmp) / "source"
        run(["git", "clone", "--no-checkout", SOURCE_REPOSITORY, staged])
        run(["git", "-C", staged, "checkout", "--detach", REVISION])
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=staged, text=True).strip() == REVISION
        assert subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=staged, text=True).strip() == SOURCE_TREE
        staged.rename(REPO)'''
        verify='assert subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=REPO, text=True).strip() == SOURCE_TREE'
        receipt_name='source-git.json'
        receipt='{"revision": REVISION, "repository": SOURCE_REPOSITORY, "tree": SOURCE_TREE}'
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
    setup='''import base64, hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

RESUME_ARCHIVE = ""  # Ví dụ: /kaggle/input/my-qcgs-checkpoint/qcgs-label-free-evidence.zip
REPO = Path("/tmp/NB-Ramen-QCGS")
PYTHON = Path("/tmp/nb-ramen-qcgs-venv/bin/python")
DATA = Path("/tmp/nb-ramen-qcgs-data")
EVIDENCE = Path("/kaggle/working/qcgs-label-free-evidence")
RUNTIME = EVIDENCE / "runtime"
REVISION = __REVISION__
__TRANSPORT_CONSTANTS__

def run(command, *, env=None, log=None):
    command = [str(x) for x in command]
    if log is None:
        subprocess.run(command, check=True, cwd=REPO if REPO.exists() else None, env=env)
    else:
        with Path(log).open("w") as handle:
            subprocess.run(command, check=True, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)

__RESTORE_SOURCE__
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == REVISION
__VERIFY_SOURCE__
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
run(["git", "merge-base", "--is-ancestor", "3a80623b074f16b8ef87d8d5507427277ea2a54f", REVISION])

if RESUME_ARCHIVE:
    spec = importlib.util.spec_from_file_location("checkpoint", REPO / "notebooks/kaggle/full-run-checkpoint.py")
    support = importlib.util.module_from_spec(spec); spec.loader.exec_module(support)
    support.restore(RESUME_ARCHIVE, EVIDENCE)
RUNTIME.mkdir(parents=True, exist_ok=True)
(RUNTIME / "__RECEIPT_NAME__").write_text(json.dumps(__SOURCE_RECEIPT__, indent=2))
print("Pinned source ready:", REVISION)
'''
    setup=setup.replace('__REVISION__',repr(revision)).replace('__TRANSPORT_CONSTANTS__',constants).replace('__RESTORE_SOURCE__',restore)
    setup=setup.replace('__VERIFY_SOURCE__',verify).replace('__RECEIPT_NAME__',receipt_name).replace('__SOURCE_RECEIPT__',receipt)
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
    if rescue:
        for c in cells:
            text = ''.join(c['source'])
            for before, after in (
                ('QCGS label-free diagnostic — Stage A → Stage B', 'QCGS multi-view rescue — Stage A gate → Stage B'),
                ('qcgs-label-free-evidence', 'qcgs-multiview-evidence'),
                ('NB-Ramen-QCGS', 'NB-Ramen-QCGS-MV'),
                ('nb-ramen-qcgs-', 'nb-ramen-qcgs-mv-'),
                ('qcgs-source.bundle', 'qcgs-multiview-source.bundle'),
                ('"qcgs"', '"qcgs-multiview"'),
                ('3a80623b074f16b8ef87d8d5507427277ea2a54f', 'cb3c92cded0e6ad4601b5e1f876835a9cf3992c4'),
                ('A → bins → B', 'A → committed GO_CONFIRM → B'),
                ('Stage A có exact oracle.', 'Stage A có exact oracle. Stage B chỉ chạy khi cả hai cell đạt GO_CONFIRM; nếu STOP thì notebook xuất báo cáo và ZIP ngay.'),
            ):
                text = text.replace(before, after)
            c['source'] = text.splitlines(keepends=True)
    if repository_url is not None:
        cells[0]['source']=''.join(cells[0]['source']).replace(
            'có sẵn trong notebook; không cần push nhánh lên GitHub.',
            'được tải từ GitHub và xác minh commit/tree hash trước khi chạy.').splitlines(keepends=True)
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},'qcgs':{**source_info,'experimental_outputs':False, **({'diagnostic':'qcgs-multiview'} if rescue else {})}},'cells':cells}


def write_notebook(notebook, output):
    for i,c in enumerate(notebook['cells']):
        c['id']=f'qcgs-{i}'
        if c['cell_type']=='code': compile(''.join(c['source']),f'cell-{i}','exec')
    encoded=(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n').encode('utf-8')
    if len(encoded) >= MAX_NOTEBOOK_BYTES:
        raise ValueError(f'Notebook is {len(encoded)} bytes; Kaggle requires less than {MAX_NOTEBOOK_BYTES}. Use Git transport instead of embedding a bundle.')
    Path(output).write_bytes(encoded)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--multiview',action='store_true')
    parser.add_argument('--revision',help='Keep an already validated source commit; defaults to HEAD')
    parser.add_argument('--repository',default=SOURCE_REPOSITORY)
    args=parser.parse_args()
    if args.output is None:
        args.output=Path(__file__).with_name('kaggle-qcgs-multiview.ipynb' if args.multiview else 'kaggle-qcgs-label-free.ipynb')
    revision=subprocess.check_output(['git','rev-parse','--verify',(args.revision or 'HEAD')+'^{commit}'],cwd=ROOT,text=True).strip()
    if args.revision is None:
        subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts','cfg','tests','pytest.ini',
                    'docs/research/query-conditioned-gradient-selection.md','docs/research/qcgs-multiview-rescue-protocol.md','notebooks/kaggle/build-qcgs-notebook.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    tree=subprocess.check_output(['git','rev-parse',revision+'^{tree}'],cwd=ROOT,text=True).strip()
    notebook=build(revision,rescue=args.multiview,repository_url=args.repository,source_tree=tree)
    write_notebook(notebook,args.output)
    print(args.output,revision,args.output.stat().st_size)


if __name__=='__main__': main()
