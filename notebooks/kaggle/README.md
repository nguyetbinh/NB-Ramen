# Kaggle workflows

- **Full matrix 252 runs:** [kaggle-full-matrix.ipynb](kaggle-full-matrix.ipynb),
  [Python export](kaggle-full-matrix.py), [hướng dẫn chạy/resume](full-run-guide.md).
- **Pre-full smoke 14 runs:** dùng notebook mô tả bên dưới.

## Pre-full CUDA smoke

Import [kaggle-pre-full-smoke.ipynb](kaggle-pre-full-smoke.ipynb) vào Kaggle,
bật **Internet** và chọn **GPU** (đã kiểm chứng trên Tesla T4), rồi chạy các cell
từ trên xuống. Có [Python export](kaggle-pre-full-smoke.py) tương ứng.

Notebook pin source tại `26a7cd7c847b6630841dbae58067e5bb124f2f9d`, kể cả khi
nhánh repo có commit mới hơn. Đây là revision đã chạy 354 test và 14 smoke
thành công trên CUDA. Commit chứa notebook/báo cáo có thể mới hơn commit thí
nghiệm; không thay pin theo HEAD nếu chưa kiểm chứng source mới.

- Tạo Python 3.11 virtualenv, PyTorch 2.4.1/cu121 và các dependency cố định.
- Dùng một GPU; chạy 14 smoke, mỗi run 256 mẫu, gồm các control B=1, causal,
  split v2 và OOD=0. **252 full runs chỉ được lập kế hoạch, không được chạy.**
- Tải CIFAR-100-C chính thức hoặc nhận đường dẫn archive ở `ARCHIVE_INPUT`;
  kiểm tra checksum, provenance và deep preflight trước khi chạy.
- Lưu logs và ZIP tại `/kaggle/working/nb-ramen-evidence-26a7cd7.zip`.
  Tải ZIP trước khi kết thúc phiên. Data/venv trong `/tmp` có thể cần tạo lại
  khi đổi session; chạy lại setup từ đầu.
- Resume chỉ bỏ qua các run đã qua strict validation. Giữ lại thư mục run
  lỗi để chẩn đoán bằng cách đổi tên nó trước khi chạy lại; không sửa evidence
  JSON để vượt validation.
- `Could not resolve host: github.com`: kiểm tra Internet trong Settings,
  rồi chạy lại cell setup. Notebook không tự chứng nhận causal equivalence;
  cần đọc `runtime/causal-sensitivity.json`.

Kết quả và giới hạn đã được ghi trong
[báo cáo CUDA sau sửa reset](../../plans/20260825-open-world-gradient-memory-evidence/reports/kaggle-cuda-reset-validation-20260907.md).
Raw evidence ZIP không được commit vào Git; SHA-256 và kết quả kiểm tra được
lưu trong báo cáo và validation JSON đi kèm.

## Cập nhật notebook

`build-notebook.py` chứa các cell thiết lập và nhúng nguyên nội dung của
`prepare-data.py` và `run-smokes.py`. Sau khi sửa nguồn, tái sinh notebook và
Python export từ repo root:

```sh
python notebooks/kaggle/build-notebook.py
```

Không chạy Python export trên máy local để kiểm tra cú pháp: nó thực thi
workflow Kaggle, có cài dependency và chạy thí nghiệm. Các script helper cần
PYTHONPATH trỏ đến `src` của revision thí nghiệm; notebook tự thiết lập.
