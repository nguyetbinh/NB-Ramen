"""Build the full matrix notebook from the validated smoke setup cells."""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
smoke = json.loads((ROOT / "kaggle-pre-full-smoke.ipynb").read_text())
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": (source.strip() + "\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add("markdown", """
# NB-Ramen — Full canonical CIFAR-100-C trên Kaggle

Notebook này **thực thi 252 full runs**: 7 phương pháp × 4 tỷ lệ OOD × 3 stream × 3 seed.
Pin source `26a7cd7c847b6630841dbae58067e5bb124f2f9d` đã qua CUDA smoke.
Giữ batch=100, block=64, 400 source samples/domain, **không giới hạn prefix**.
Đây là full matrix CIFAR-100-C đã định nghĩa, không phải toàn bộ mọi benchmark
hay toàn bộ 50.000 ảnh trong từng corruption array. DomainNet/v2-v3 robustness là nghiên cứu riêng.

Bật **Internet**, chọn **GPU** (T4 đã được kiểm chứng), rồi chạy các cell theo thứ tự.
Dùng một GPU. Giữ cùng loại GPU qua các phiên để so sánh chi phí nhất quán.

Mặc định dừng giữa các run sau ngân sách **8 giờ chạy matrix**; đây là ngân sách
do notebook chọn, không phải cam kết về giới hạn Kaggle. Một run đang chạy được
hoàn tất trước khi dừng, nên có thể vượt ngân sách. Đặt `SESSION_HOURS=0` để không
dừng theo thời gian; `MAX_RUNS=0` nghĩa là thực hiện mọi run còn thiếu trong ngân sách.
Giới hạn phiên chỉ chia công việc giữa các session, không cắt ngắn stream.

ZIP được cập nhật sau mỗi run hoàn tất. Tải ZIP trước khi kết thúc phiên; ZIP
trên máy Kaggle chưa được tải/lưu ra ngoài không bảo đảm tồn tại sau khi session mất.
Lần sau gắn ZIP đó qua **Add Input**, đặt đường dẫn tại `RESUME_ARCHIVE`, chạy từ đầu.
Checkpoint khôi phục cùng đường dẫn `/kaggle/working`; không sửa đường dẫn trong JSON.
Không dùng ZIP smoke làm checkpoint full.
""")

setup = "".join(next(cell["source"] for cell in smoke["cells"] if cell["cell_type"] == "code"))
setup = setup.replace('import os, sys, json, subprocess, shutil', 'import os, sys, json, subprocess, shutil, uuid, importlib.util\nfrom datetime import datetime, timezone')
setup = setup.replace('EVIDENCE = WORK / f"nb-ramen-evidence-{REVISION[:7]}"', 'EVIDENCE = WORK / f"nb-ramen-full-{REVISION[:7]}"')
setup = setup.replace('RUNTIME = EVIDENCE / "runtime"', '''# Chỉ sửa ba tùy chọn này và ARCHIVE_INPUT nếu cần.
RESUME_ARCHIVE = ""  # Ví dụ /kaggle/input/my-full-checkpoint/nb-ramen-full-26a7cd7.zip
MAX_RUNS = 0        # 0 = mọi run còn thiếu; ví dụ 7 = tối đa 7 run mới trong phiên
SESSION_HOURS = 8.0 # 0 = không dừng theo thời gian; kiểm tra ngân sách giữa các run
SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
RUNTIME = EVIDENCE / "runtime" / SESSION_ID''')
checkpoint_source = (ROOT / "full-run-checkpoint.py").read_text()
restore_code = (
    'checkpoint_script = WORK / "nb-ramen-full-checkpoint.py"\n'
    'checkpoint_script.write_text(' + repr(checkpoint_source) + ')\n'
    'if RESUME_ARCHIVE.strip():\n'
    '    subprocess.run([sys.executable, str(checkpoint_script), "restore", str(EVIDENCE),\n'
    '                    "--archive", RESUME_ARCHIVE], check=True)\n'
)
setup = setup.replace('WORK.mkdir(parents=True, exist_ok=True)\n', 'WORK.mkdir(parents=True, exist_ok=True)\n' + restore_code)
setup = setup.replace('RAMEN_DATA_ROOT=str(DATA), RAMEN_EVIDENCE_ROOT=str(EVIDENCE),',
                      'RAMEN_DATA_ROOT=str(DATA), RAMEN_EVIDENCE_ROOT=str(EVIDENCE),\n           RAMEN_RUNTIME_ROOT=str(RUNTIME),')
process_source = (ROOT / "full-run-process.py").read_text()
process_code = (
    'process_script = RUNTIME / "full-run-process.py"\n'
    'process_script.write_text(' + repr(process_source) + ')\n'
    'spec = importlib.util.spec_from_file_location("full_run_process", process_script)\n'
    'process_module = importlib.util.module_from_spec(spec)\n'
    'spec.loader.exec_module(process_module)\n\n'
    'def run(argv, *, log=None, cwd=None):\n'
    '    return process_module.run_logged(argv, env=env, log=log, cwd=cwd)\n\n'
)
start, end = setup.index("def run("), setup.index('run(["nvidia-smi"]')
setup = setup[:start] + process_code + setup[end:]
add("code", setup)

add("markdown", "## Runtime và tests\nMỗi session đều kiểm tra GPU, interpreter và 354 test của source đã pin trước khi chạy full.")
code_cells = ["".join(cell["source"]) for cell in smoke["cells"] if cell["cell_type"] == "code"]
runtime_code = code_cells[1].replace(
    "(Path(os.environ['RAMEN_EVIDENCE_ROOT'])/'runtime/device.json')",
    "(Path(os.environ['RAMEN_RUNTIME_ROOT'])/'device.json')",
)
assert runtime_code != code_cells[1], "Smoke runtime probe location changed; update full-session adaptation"
add("code", runtime_code)

add("markdown", "## Dữ liệu và checkpoint CLIP\nTải và xác minh cùng artifact chính thức như smoke. Dữ liệu/venv cần tạo lại nếu session trước đã bị xóa.")
prepare_source = (ROOT / "prepare-data.py").read_text()
# Preserve the tested acquisition logic; put its per-session record alongside this session's tests.
prepare_source = prepare_source.replace(
    'runtime = Path(os.environ["RAMEN_EVIDENCE_ROOT"]) / "runtime"',
    'runtime = Path(os.environ["RAMEN_RUNTIME_ROOT"])',
)
add("code", "prepare_script = RUNTIME / 'prepare-data.py'\nprepare_script.write_text(" + repr(prepare_source) + ")\n" + '''
run([PYTHON, prepare_script], cwd=REPO, log=RUNTIME / "prepare-data.log")
run([PYTHON, "-m", "runtime.preflight", "--data-root", DATA, "--dataset", "CIFAR100C",
     "--deep", "--json"], cwd=REPO, log=RUNTIME / "cifar100c-deep-preflight.json")
''')

add("markdown", """
## Thực thi full matrix và resume

Run NoAdapt trước các phương pháp cùng cell. Mỗi run hoàn tất phải qua strict
validation và khớp revision/config/fingerprint baseline, rồi mới ghi checkpoint.
Các run đủ summary nhưng không hợp lệ gây lỗi, không bị bỏ qua.
Run bị ngắt và chưa có summary được chuyển vào `interrupted/` để giữ chẩn đoán,
sau đó chạy lại nguyên run. Không nối tiếp từ giữa một stream.

Logs từng run ở `runtime/<session>/`; status toàn matrix ở `status.json`.
Khi đạt **252/252**, tạo `analysis/open-set-consensus.json`, `per-cell-metrics.csv`
và README. Chỉ `status="complete"` cùng `full_matrix_complete=true` mới là full evidence.
Trạng thái `paused_session_budget` cần tiếp tục ở phiên sau.
""")
runner = (ROOT / "run-full-matrix.py").read_text()
add("code", "full_script = RUNTIME / 'run-full-matrix.py'\nfull_script.write_text(" + repr(runner) + ")\n"
    + "checkpoint_script = RUNTIME / 'full-run-checkpoint.py'\ncheckpoint_script.write_text(" + repr(checkpoint_source) + ")\n" + '''
try:
    run([PYTHON, full_script, "--max-runs", str(MAX_RUNS), "--session-hours", str(SESSION_HOURS)],
        cwd=REPO, log=RUNTIME / "full-execution.log")
finally:
    run([PYTHON, checkpoint_script, "save", EVIDENCE])
''')

add("markdown", """
## Tải evidence / tiếp tục phiên sau

Tải `nb-ramen-full-26a7cd7.zip`. Để resume phiên mới, upload ZIP đã tải thành Kaggle
Input và điền `RESUME_ARCHIVE` ở cell đầu. Nếu đang tiếp tục ngay trong phiên
hiện tại, để trống `RESUME_ARCHIVE`: notebook dùng evidence đang có.
File Input cần là ZIP full đúng tên root, không phải ZIP smoke hay chỉ vài summary.
Nếu Input đã được giải nén tự động, nén lại cây `nb-ramen-full-26a7cd7/` thành ZIP
với đúng một root đó trước khi dùng. Không merge các checkpoint bằng tay.

Ngân sách/hết phiên có thể buộc chạy lại run đang dở, nhưng các run đã checkpoint
và được validator chấp nhận sẽ được giữ. Sau 252/252, gửi ZIP để đánh giá kết quả.
""")
add("code", '''
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
''')

notebook = {"cells": cells, "metadata": copy.deepcopy(smoke["metadata"]), "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"ramen-full-{index:02d}"
(ROOT / "kaggle-full-matrix.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
export = []
for cell in cells:
    source = "".join(cell["source"])
    if cell["cell_type"] == "code":
        export.append("# %%\n" + source)
    else:
        export.append("# %% [markdown]\n" + "\n".join(("# " + line if line else "#") for line in source.splitlines()) + "\n")
(ROOT / "kaggle-full-matrix.py").write_text("\n".join(export))
print("Wrote full matrix notebook and Python export.")
