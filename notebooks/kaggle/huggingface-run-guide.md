# Full matrix dùng CIFAR-100-C từ Hugging Face

Import [kaggle-huggingface-full-matrix.ipynb](kaggle-huggingface-full-matrix.ipynb)
vào Kaggle, bật Internet và chọn GPU T4. Chọn **Save Version → Save & Run All**
để thực thi các cell từ đầu trong phiên nền. `SESSION_HOURS=0`, `MAX_RUNS=0`
cho phép mọi run còn thiếu; giới hạn tài nguyên/thời gian của Kaggle vẫn áp dụng.

Notebook này là một workflow riêng. Cần import cả notebook: cell chuẩn bị dữ
liệu Hugging Face không thể ghép độc lập với setup/driver của notebook Zenodo.
Không thay đổi phương pháp, CLIP, batch=100, block=64, source budget=400/domain,
7 phương pháp × 4 OOD ratios × 3 stream × 3 seed = 252 runs.

## Nguồn dữ liệu và xác minh

- Repository: [WNJXYK/TTA-CIFAR-100-C](https://huggingface.co/datasets/WNJXYK/TTA-CIFAR-100-C).
- Revision dữ liệu: `a12f0bcc1da33fa26d8c76ce8c1fb32e6f913bea`.
- Tải 95 file Parquet, khoảng 2,13 GB; chuyển thành 19 corruption arrays và
  `labels.npy`, giữ nguyên thứ tự và byte ảnh. Dữ liệu NumPy khoảng 2,92 GB.
- Mỗi `.npy` phải khớp MD5 trong [bảng CIFAR100C của TorchUncertainty](https://torch-uncertainty.github.io/_modules/torch_uncertainty/datasets/classification/cifar/cifar_c.html).
  Kiểm tra cả header, nhãn và toàn bộ mức severity, không chỉ kích thước file.
- Dataset inventory tiếp tục dùng SHA-256. CLIP vẫn phải khớp SHA-256 chính thức.
- Evidence ghi rõ Hugging Face và revision dữ liệu, không ghi một checksum
  archive Zenodo chưa được đo. File thiếu, sai nội dung/thứ tự hoặc metadata
  provenance bị sửa đều gây lỗi; không giảm kiểm tra để chạy tiếp.

## Phiên bản source và checkpoint

Setup clone riêng vào `/kaggle/working/NB-Ramen-HF` từ base
`26a7cd7c847b6630841dbae58067e5bb124f2f9d`, áp dụng patch acquisition đã nhúng
và tạo commit local xác định `dc3cbf02215cff59046f048e6580dd8f8d40af89`.
Commit chỉ thay hỗ trợ provenance và tests; không push lên GitHub. Mỗi session
chạy tests trên source sau patch; matrix driver kiểm tra checkout sạch và đúng
revision này. Patch và log tạo source nằm trong evidence của session.

Dữ liệu dùng `/tmp/nb-ramen-hf-data`. Evidence và ZIP riêng:
`/kaggle/working/nb-ramen-full-hf-dc3cbf0.zip`.
Chỉ dùng ZIP của notebook Hugging Face này cho `RESUME_ARCHIVE`; không trộn
campaign/checkpoint của nguồn Zenodo. ZIP vẫn được cập nhật sau mỗi run đã qua
strict validation. `status="complete"` và `full_matrix_complete=true` chỉ xuất
hiện sau khi đủ 252/252 và analyzer chấp nhận coverage.

## Bảo trì

`huggingface-support/` chứa phần source/test bổ sung. Builder lấy hai module
gốc bằng `git show <base>:<path>` rồi tạo patch; không sửa source đang phát triển
trong checkout chính. Sau khi thay patch, tạo lại commit trong checkout thử
nghiệm sạch từ base bằng `bootstrap-huggingface-source.py`, lấy revision thực,
rồi tái sinh notebook bằng:

```sh
python notebooks/kaggle/build-huggingface-notebook.py
python notebooks/kaggle/build-huggingface-notebook.py --revision <verified-local-commit>
```

Không thực thi Python export trên máy local: export sẽ cài môi trường Kaggle
và chạy workflow. Chỉ compile để kiểm tra cú pháp.

## Đã kiểm tra ngày 2026-09-14

Source sau patch chạy 360 tests: 355 đạt, 5 bỏ qua do yêu cầu phần cứng.
Đã tải và đối chiếu SHA-256 của cả 5 file brightness thật từ Hugging Face;
chuyển đủ 50.000 ảnh và nhãn bằng đúng phiên bản thư viện trong notebook.
Cả `brightness.npy` và `labels.npy` khớp MD5 gốc. Kiểm tra từ chối nhãn sai
thứ tự và Parquet thiếu hàng đã đạt. Cú pháp các cell/script nhúng hợp lệ;
clone thử thứ hai tái tạo đúng commit source, checkout sạch.

Chưa chạy chuyển đổi đủ 19 corruption hoặc ma trận 252 runs CUDA trên Kaggle.
Notebook sẽ xác minh toàn bộ 20 file NumPy trước khi bắt đầu ma trận.
