"""Build a separate reproducible source patch and Kaggle HF matrix notebook."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
BASE = "26a7cd7c847b6630841dbae58067e5bb124f2f9d"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one source anchor: {old[:80]!r}")
    return source.replace(old, new)


def source_patch():
    changes = {}
    for filename in ("artifact_provenance.py", "experiment_matrix.py"):
        path = f"src/runtime/{filename}"
        source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=REPO, text=True)
        original = source
        if filename == "artifact_provenance.py":
            source = replace_once(source, "SCHEMA_VERSION = 1", '''try:
    from .cifar100c_huggingface import CIFAR100C_HF_ACQUISITION, verify_huggingface_cifar100c_files
except ImportError:
    from cifar100c_huggingface import CIFAR100C_HF_ACQUISITION, verify_huggingface_cifar100c_files


SCHEMA_VERSION = 1''')
            source = replace_once(source,
                "if dict(acquisition) != CIFAR100C_OFFICIAL_ACQUISITION:",
                "if dict(acquisition) not in (CIFAR100C_OFFICIAL_ACQUISITION, CIFAR100C_HF_ACQUISITION):")
            source = replace_once(source,
                'CIFAR-100-C acquisition does not match the pinned official Zenodo artifact',
                'CIFAR-100-C acquisition does not match the pinned official Zenodo artifact or pinned Hugging Face mirror')
            source = replace_once(source,
                '''        acquisition_record["expected_checksum"] = acquisition_record["expected_checksum"].lower()
        acquisition_record["actual_checksum"] = acquisition_record["actual_checksum"].lower()''',
                '''        if acquisition_record == CIFAR100C_HF_ACQUISITION:
            verify_huggingface_cifar100c_files(root)
        else:
            acquisition_record["expected_checksum"] = acquisition_record["expected_checksum"].lower()
            acquisition_record["actual_checksum"] = acquisition_record["actual_checksum"].lower()''')
            source = replace_once(source,
                '''    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,''',
                '''    if exact and dataset == "cifar100c" and payload.get("acquisition") == CIFAR100C_HF_ACQUISITION:
        verify_huggingface_cifar100c_files(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,''')
        else:
            if source.count("        CIFAR100C_OFFICIAL_ACQUISITION,") != 2:
                raise RuntimeError("Matrix provenance import anchors changed")
            source = source.replace("        CIFAR100C_OFFICIAL_ACQUISITION,",
                                    "        CIFAR100C_OFFICIAL_ACQUISITION, CIFAR100C_HF_ACQUISITION,")
            source = replace_once(source,
                '    expected_acquisition = CIFAR100C_OFFICIAL_ACQUISITION if run.dataset == "CIFAR100C" else {}',
                '''    expected_acquisition = CIFAR100C_OFFICIAL_ACQUISITION if run.dataset == "CIFAR100C" else {}
    if run.dataset == "CIFAR100C" and dataset.get("acquisition") == CIFAR100C_HF_ACQUISITION:
        expected_acquisition = CIFAR100C_HF_ACQUISITION
        _require_equal(dataset.get("file_count"), 20, "manifest.artifacts.dataset.file_count", run)''')
        compile(source, path, "exec")
        changes[path] = (original, source)
    for filename, destination in (
        ("cifar100c_huggingface.py", "src/runtime/cifar100c_huggingface.py"),
        ("test_cifar100c_huggingface.py", "tests/test_cifar100c_huggingface.py"),
    ):
        changes[destination] = ("", (ROOT / "huggingface-support" / filename).read_text())
    patch = "".join("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}" if before else "/dev/null", tofile=f"b/{path}",
    )) for path, (before, after) in changes.items())
    (ROOT / "huggingface-source.patch").write_text(patch)
    return patch


def build(revision):
    patch = source_patch()
    notebook = json.loads((ROOT / "kaggle-full-matrix.ipynb").read_text())
    cells = notebook["cells"]
    cells[0]["source"] = [f'''# NB-Ramen — 252 runs với dữ liệu Hugging Face

Bật Internet, chọn GPU T4, rồi Save Version → Save & Run All để chạy nền.
Giữ 252 runs, batch=100, block=64, 400 source samples/domain và một GPU.
`MAX_RUNS=0`, `SESSION_HOURS=0` cho phép chạy mọi run còn thiếu đến giới hạn Kaggle.

Notebook tải ảnh từ Hugging Face, chuyển về NumPy và đối chiếu cả 20 file
với checksum của CIFAR-100-C gốc. Nguồn Hugging Face được ghi rõ trong evidence;
không khai rằng đã tải hay đo checksum archive Zenodo.

Source nền: `{BASE}`. Patch acquisition được nhúng trong notebook và tạo commit
local xác định `{revision}`. Mỗi phiên đều chạy tests trên source đã patch.
Các thuật toán, cấu hình, CLIP và ma trận không đổi. Dùng checkpoint riêng của
notebook này để resume. Chưa có kết quả CUDA đầy đủ cho bản acquisition này.
''']
    setup = "".join(cells[1]["source"])
    setup = replace_once(setup, 'REPO = WORK / "NB-Ramen"', 'REPO = WORK / "NB-Ramen-HF"')
    setup = replace_once(setup, 'DATA = Path("/tmp/nb-ramen-data")', 'DATA = Path("/tmp/nb-ramen-hf-data")')
    setup = replace_once(setup, 'EVIDENCE = WORK / f"nb-ramen-full-{REVISION[:7]}"',
                         f'EVIDENCE = WORK / "nb-ramen-full-hf-{revision[:7]}"')
    setup = replace_once(setup, 'SESSION_HOURS = 8.0', 'SESSION_HOURS = 0.0')
    setup = setup.replace('nb-ramen-full-26a7cd7.zip', f'nb-ramen-full-hf-{revision[:7]}.zip')
    bootstrap = (ROOT / "bootstrap-huggingface-source.py").read_text()
    injection = ('assert actual == REVISION\n'
        + 'source_patch = RUNTIME / "huggingface-source.patch"\nsource_patch.write_text(' + repr(patch) + ')\n'
        + 'bootstrap_script = RUNTIME / "bootstrap-huggingface-source.py"\nbootstrap_script.write_text(' + repr(bootstrap) + ')\n'
        + 'run([sys.executable, bootstrap_script, REPO, source_patch], cwd=REPO, log=RUNTIME / "source-patch.log")\n'
        + 'actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()\n'
        + f'assert actual == {revision!r}, "Hugging Face source commit is not reproducible"\n'
        + 'REVISION = actual\nenv["RAMEN_REVISION"] = REVISION\n')
    setup = replace_once(setup, 'assert actual == REVISION\n', injection)
    setup = replace_once(setup, '"pyyaml==6.0.2", "tqdm==4.66.5",',
                         '"pyyaml==6.0.2", "tqdm==4.66.5", "pyarrow==18.1.0", "huggingface-hub==0.26.2",')
    setup = replace_once(setup, '\nenv = dict(os.environ,',
                         '\nenv = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DOWNLOAD_TIMEOUT="60",')
    setup = setup.replace('# Tùy chọn: ví dụ "/kaggle/input/my-official-cifar100c/CIFAR-100-C.tar"',
                          '# ARCHIVE_INPUT không dùng trong notebook Hugging Face này.')
    cells[1]["source"] = setup.splitlines(keepends=True)
    cells[2]["source"] = ['## Runtime và tests\nKiểm tra CUDA và chạy toàn bộ tests của source đã thêm hỗ trợ Hugging Face.\n']
    cells[4]["source"] = ['## Dữ liệu từ Hugging Face\nTải 95 Parquet ở revision cố định. Mỗi file NumPy sau chuyển đổi phải khớp checksum gốc; mọi lỗi dừng notebook.\n']
    prepare = (ROOT / "prepare-huggingface-data.py").read_text()
    download = (ROOT / "prepare-data.py").read_text()
    cell6 = ('support_script = RUNTIME / "download-support.py"\nsupport_script.write_text(' + repr(download) + ')\n'
             + 'prepare_script = RUNTIME / "prepare-huggingface-data.py"\nprepare_script.write_text(' + repr(prepare) + ')\n'
             + '''run([PYTHON, prepare_script], cwd=REPO, log=RUNTIME / "prepare-data.log")
run([PYTHON, "-m", "runtime.preflight", "--data-root", DATA,
     "--dataset", "CIFAR100C", "--deep", "--json"],
    cwd=REPO, log=RUNTIME / "cifar100c-deep-preflight.json")
''')
    cells[5]["source"] = cell6.splitlines(keepends=True)
    cells[7]["source"] = "".join(cells[7]["source"]).replace(BASE, revision).splitlines(keepends=True)
    cells[8]["source"] = "".join(cells[8]["source"]).replace('nb-ramen-full-26a7cd7', f'nb-ramen-full-hf-{revision[:7]}').splitlines(keepends=True)
    for index, cell in enumerate(cells):
        cell["id"] = f"ramen-hf-{index:02d}"
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
            compile("".join(cell["source"]), f"cell-{index+1}", "exec")
    path = ROOT / "kaggle-huggingface-full-matrix.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    export = []
    for cell in cells:
        source = "".join(cell["source"])
        export.append("# %%\n" + source if cell["cell_type"] == "code" else
                      "# %% [markdown]\n" + "\n".join(("# " + line).rstrip() for line in source.splitlines()) + "\n")
    (ROOT / "kaggle-huggingface-full-matrix.py").write_text("\n".join(export))
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.revision:
        build(args.revision)
    else:
        patch = source_patch()
        print("Patch SHA256:", hashlib.sha256(patch.encode()).hexdigest())
