# Full canonical CIFAR-100-C trên Kaggle

Import [kaggle-full-matrix.ipynb](kaggle-full-matrix.ipynb) vào Kaggle và chạy
các cell từ trên xuống. Có [Python export](kaggle-full-matrix.py) để đọc/copy.
Notebook dùng source `26a7cd7c847b6630841dbae58067e5bb124f2f9d`, đã qua 354
test và 14 smoke trên Tesla T4; xem [báo cáo CUDA](../../plans/20260825-open-world-gradient-memory-evidence/reports/kaggle-cuda-reset-validation-20260907.md).

## Phạm vi evidence

Thực thi đủ **252 runs**: NoAdapt và sáu phương pháp adaptation/oracle,
OOD ratios 0/.1/.3/.5, streams iid_mixed/block/recurring, seeds 0/1/2.
Giữ B=100, block=64, source budget=400/domain, CUDA và fast provenance;
không truyền `--max-eval-samples`. Full ở đây là toàn bộ stream canonical đã
định nghĩa từ source budget này, không phải toàn bộ mọi ảnh trong archive.
DomainNet, split robustness v2/v3 và các probe utility là nghiên cứu riêng.

Chỉ khi đủ **252/252** và analyzer chấp nhận coverage canonical, notebook mới
ghi `status="complete"`, `full_matrix_complete=true` cùng báo cáo phân tích.
Số run hoàn tất là tiến độ, không phải chứng nhận phương pháp thắng baseline.

## Phiên đầu

1. Bật Internet, chọn GPU. T4 khớp môi trường CUDA đã smoke; notebook dùng GPU 0.
2. Để `RESUME_ARCHIVE=""`. Có thể gắn archive CIFAR-100-C chính thức qua
   `ARCHIVE_INPUT` để tránh tải lại; checksum vẫn phải khớp.
3. Chạy setup, runtime/tests và data/preflight, rồi chạy cell full matrix.
4. Tải `/kaggle/working/nb-ramen-full-26a7cd7.zip` trước khi kết thúc phiên.

Mặc định:

```python
RESUME_ARCHIVE = ""
MAX_RUNS = 0
SESSION_HOURS = 8.0
```

`MAX_RUNS=0` cho phép mọi run còn thiếu. `SESSION_HOURS=8.0` là ngân sách
matrix do notebook chọn, không phải giới hạn Kaggle được cam kết. Kiểm tra
ngân sách giữa các run: run đang chạy được hoàn tất nên tổng thời gian có thể
vượt mốc đó. Đặt `SESSION_HOURS=0` để tắt giới hạn này, hoặc `MAX_RUNS=7` để
chạy tối đa 7 run mới mỗi lần. Những lựa chọn này không thay nội dung từng run.
Thời gian setup/data/tests không tính vào ngân sách matrix.

## Tiếp tục ở phiên khác

Upload ZIP full đã tải thành Kaggle Input, rồi điền đường dẫn ZIP thực tế:

```python
RESUME_ARCHIVE = "/kaggle/input/my-full-checkpoint/nb-ramen-full-26a7cd7.zip"
```

Đường dẫn trên là ví dụ; chọn đúng file trong Input. ZIP phải chứa một root
`nb-ramen-full-26a7cd7/`. Nếu dịch vụ giải nén ZIP thành Input, nén lại cây đó
thành ZIP giữ đúng root trước khi restore. Không dùng ZIP smoke.

Chạy từ cell đầu để tạo lại venv/data nếu cần. Restore giữ nguyên đường dẫn
run trong evidence và không merge/ghi đè cây evidence đang có. Nếu tiếp tục
trong cùng session, để `RESUME_ARCHIVE=""` để dùng cây hiện tại. Chạy lại setup
với cùng ZIP đã restore cũng giữ nguyên tiến độ mới phát sinh.

Mỗi run hiện có đều phải qua strict validation; manifest/source phải khớp
commit sạch đã pin và adapted run phải khớp NoAdapt cùng cell. Giữ cùng loại
GPU, phiên bản torch/torchvision/CUDA và đường dẫn data qua các phiên.
Các thay đổi đang phát triển ở nhánh repo không được nhập vào thí nghiệm.

## Checkpoint và lỗi

Sau mỗi run mới hoàn tất, tạo ZIP tạm rồi thay ZIP cũ sau khi ghi thành công.
Nếu phiên bị ngắt lúc nén, ZIP hoàn chỉnh trước đó vẫn được giữ. Download/lưu
ZIP ra ngoài phiên mới bảo vệ được kết quả khi môi trường Kaggle bị xóa.
Notebook không tự upload evidence ra dịch vụ ngoài.

Nút Interrupt dừng nhóm subprocess thí nghiệm trước khi đóng gói evidence.
Nếu còn run thiếu summary, lần chạy sau chuyển nguyên thư mục đó vào
`interrupted/` và thực thi lại run từ đầu; không nối trace dở dang. Run có
summary nhưng không qua validator gây lỗi và cần chẩn đoán, không tự đánh dấu
hoàn tất. Log lỗi và các lần thử bị ngắt vẫn được giữ trong checkpoint.

`status="paused_session_budget"` nghĩa là cần tiếp tục; `status="failed"`
nghĩa là đọc lỗi/log trước khi tiếp tục. Không chỉnh JSON để ép resume.

## Output cuối

- `campaign.json`: revision, đường dẫn data và môi trường GPU cố định.
- `status.json`: tiến độ và cờ full completion.
- `canonical/<run-id>/`: manifest, summary, stream, trace và kết quả từng run.
- `runtime/<session-id>/`: setup/data/provenance/tests, plan, driver và logs.
- `analysis/open-set-consensus.json`: descriptive analysis chính thức của repo,
  bao gồm paired outcomes, oracle diagnostics, stability và costs.
- `analysis/per-cell-metrics.csv`: accuracy ID, AUROC, FPR95, H-score riêng
  từng seed/stream/OOD ratio; chỉ số OOD không xác định tại OOD=0 để trống.

Gửi ZIP đầy đủ để đánh giá hiệu quả sau khi hoàn tất. Giữ riêng kết quả của
từng stream và seed; không retune theo các cell đang chạy.

## Bảo trì và kiểm tra local

`build-full-notebook.py` tái sử dụng setup/test cells từ smoke notebook và nhúng
`run-full-matrix.py`, `full-run-checkpoint.py`, `full-run-process.py` cùng
logic acquisition của `prepare-data.py` (record chuyển sang runtime từng phiên).
Sau khi sửa nguồn, tái sinh artifact theo thứ tự:

```sh
python notebooks/kaggle/build-notebook.py
python notebooks/kaggle/build-full-notebook.py
python -m unittest discover -s notebooks/kaggle -p 'test_full_run_support.py'
```

Các test helper dùng filesystem/subprocess thật để kiểm tra round-trip ZIP,
không overwrite, CRC, path/symlink rejection, giữ ZIP cũ khi checkpoint lỗi
và dừng cả tiến trình con khi bị interrupt. Kiểm tra local không thực thi
252 GPU runs; notebook mới là entry point để thu evidence đó trên Kaggle.
