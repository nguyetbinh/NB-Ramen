# QCGS multi-view rescue — trạng thái triển khai

Ngày 2026-09-19, nhánh `qcgs-label-free-diagnostic`.

**INCONCLUSIVE — chưa có evidence CUDA cho rescue.** Máy local có PyTorch 2.4.1 nhưng không có CUDA. Chưa chạy smoke trên CLIP, chưa tạo registry khoa học từ dataset thật, chưa chạy Stage A/Stage B. Unit tests và notebook không phải kết quả thí nghiệm. STOP của entropy-sign một view giữ nguyên.

Đã implement [protocol](../../../docs/research/qcgs-multiview-rescue-protocol.md): 8 view cố định từ ảnh corrupted, teacher tại θ₀, primary gradient tại θ_R, entropy controls, hai random controls chung draw, hai ensemble controls và supervised reference. Baseline/cache vẫn là Ramen; mọi candidate reset θ₀ rồi đúng một update. Schema 2 và namespace mới loại 507 ảnh gốc cũ, sau đó thêm smoke của campaign mới. Registry/config/source/view/provenance được khóa trước Stage A.

Runner audit raw records, xác minh Stage A exhaustive, tính U/D/E và Spearman từng query. Chỉ GO_CONFIRM đã commit trong ledger mới cho chạy Stage B; Stage B báo best-verified subset, không gọi là exact oracle. Mean/drop-two gates, transitions, uncertainty theo block, teacher trên query Ramen sai và chi phí từng nhánh được tính lại từ raw records.

Validation CPU: kiểm thử autograd thật với B=100/k=5/M=10; output/cache parity; label isolation; crop/flip và batch slots; target detach; finite-difference tại θ_R; no-swap/ties/shared RNG; reset khi lỗi; raw-record mutations; registry/exclusion/gate; ZIP atomic và notebook/source bundle. Review độc lập đã kiểm tra các ranh giới khoa học; các thiếu sót về phân tích phụ, chi phí evaluator và finite difference đã được bổ sung.

Kết quả kiểm thử và source pin cuối cùng được ghi trong receipt sau bước đóng gói. CUDA smoke và cả hai scored stages vẫn phải chạy trên Kaggle.

Chạy [notebook rescue](../../../notebooks/kaggle/kaggle-qcgs-multiview.ipynb) với Internet/GPU từ trên xuống hoặc Save & Run All. Notebook nhúng source, dùng pipeline Hugging Face và lưu `qcgs-multiview-evidence.zip`. Nếu Stage A STOP, notebook xuất báo cáo/ZIP và không chạy Stage B. Nếu ngắt, thêm ZIP làm Kaggle Input, đặt `RESUME_ARCHIVE` rồi chạy lại cùng notebook.

Sau khi tải ZIP về, cần audit lại artifact trước khi kết luận nghiên cứu. Không mở full matrix, full reranking hoặc adaptive support size trong task này.
