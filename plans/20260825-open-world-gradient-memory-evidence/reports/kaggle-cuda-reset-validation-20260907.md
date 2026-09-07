# Đánh giá CUDA smoke sau sửa reset — 2026-09-07

## Kết luận

**Đạt smoke kỹ thuật trên commit `26a7cd7c847b6630841dbae58067e5bb124f2f9d`.**
Lỗi reset B=1 đã được kiểm chứng bằng test CUDA và trace mới. Có thể chuyển
sang chuẩn bị thực thi matrix chính với đúng commit/config đã kiểm chứng.
Đây là đánh giá đủ điều kiện vận hành thử nghiệm chính theo giao thức
batch-atomic B=100; không phải chứng nhận hiệu quả phương pháp hoặc khẳng định
hai cách triển khai causal tương đương số học tuyệt đối.

Không tự động khởi chạy matrix. Chưa có kết quả của 252 full runs.
Đánh giá này chỉ áp dụng cho commit trên, không áp dụng cho các thay đổi đang
được phát triển trong workspace. Nếu dùng source mới thì cần kiểm chứng lại.

## Nguồn và kiểm tra

- ZIP: `nb-ramen-evidence-26a7cd7.zip`, 490,577 bytes, 122 entries.
- SHA-256: `7b513aa83862ca2ef468e567d80f32e73d7d7b9e9a4d7094a06f42d335d35f1d`.
- Git archived: đúng commit, worktree sạch.
- Tesla T4, CUDA 12.1, torch 2.4.1+cu121, Python 3.11.16.
- Full: **354 tests OK, không skip**. Focused: **148 tests OK**.
- **14/14** run qua independent strict validation bằng source snapshot đúng
  commit, gồm summary v4, trace v3, config/split locks, baseline pairing.
- Các file trong source inventory của cả 14 manifest khớp SHA-256 với snapshot.
- Deep preflight báo valid; official archive MD5 khớp; dataset verified_exact
  và model SHA-256 khớp theo provenance lưu trong ZIP.
- OOD=0 null control passed; v2 pair passed.
- Kế hoạch **252** full runs khớp semantic identities/config locks/reference
  pairing với kế hoạch tái sinh từ source snapshot; execution_requested=false.

Kiểm tra offline dùng validation mirror với hai prefix đường dẫn Kaggle được
chuyển sang snapshot/evidence local. Raw evidence và ZIP được giữ nguyên;
không thực thi script trong ZIP và không hash lại dataset/model trên Kaggle.
Khi kiểm tra canonical plan, run-ID chứa hash data-root: Linux `/tmp` khác
macOS `/private/tmp`, nên so sánh trực tiếp ban đầu lệch. Đã kiểm tra bằng cách
ánh xạ rõ hai hash tương ứng; không thay cấu hình hay run-ID trong raw evidence.
Kết quả kiểm tra được ghi trong [validation JSON](kaggle-cuda-reset-validation-20260907.json).
ZIP gốc được nhận và lưu riêng; không đính kèm trong Git.

## Reset và causal sensitivity

| Chỉ số | Trước sửa | Sau sửa |
| --- | ---: | ---: |
| Ramen B=1 ID accuracy | 3.23% (6/186) | 56.45% (105/186) |
| Ramen B=1 / NoAdapt B=1: khác pre-prediction | 250/256 | 0/256 |
| Ramen B=1 / CausalRamen B=100: khác post-prediction | 248/256 | 2/256 |

CausalRamen B=100 đạt 55.91% (104/186). Hai post-prediction khác nhau nằm ở
`timestep=56` (ID, B=1 đúng còn Causal B=100 sai) và `timestep=123` (OOD).
Một pre-prediction khác nằm ở timestep=101 (OOD), nhưng post-prediction khớp.
Max absolute post OOD-score difference là **0.390625**.

Đây là sự nhất quán thực nghiệm cao trên prefix này, không phải trùng khớp
bit-for-bit. Batching và các đường tính toán FP16 khác nhau là giải thích
khả dĩ; trace không có full logits, margins và toàn bộ support decisions để
quy nguyên nhân chính xác. Không đặt ngưỡng pass mới sau khi nhìn kết quả.
Nếu cần tuyên bố hai cách triển khai tương đương chính xác, phải có probe
riêng; điều đó chưa được chứng minh bởi ZIP.

Ramen B=100 / B=1 khác **26/256 post-predictions (10.16%)**, accuracy lệch
2.69 điểm phần trăm. Điều này củng cố việc phải công bố rõ primary protocol
là batch-atomic: thay batch size làm thay đổi phép thử. Không đổi batch size
của matrix chính hoặc chọn phương pháp dựa vào thứ hạng smoke.

So với ZIP cũ, tất cả 13 run ngoài Ramen B=1 giữ nguyên toàn bộ pre/post class
predictions. Một số score/entropy/diagnostics đổi nhẹ; không gọi là toàn bộ
artifact bất biến bit-for-bit.

## Kết quả chính: v1, B=100

Tất cả dùng cùng 256 mẫu: 186 ID và 70 OOD. Requested OOD ratio=.3;
realized ratio trong prefix=.2734375. Prefix chỉ chứa 4 corruption domains,
không đại diện đủ toàn bộ benchmark. AUROC càng cao càng tốt; FPR95 càng
thấp càng tốt.

| Phương pháp | ID accuracy | OOD AUROC | FPR95 |
| --- | ---: | ---: | ---: |
| NoAdapt | 51.08% | 0.6903 | 74.73% |
| Ramen | 53.76% | 0.7186 | 83.87% |
| EntropyGatedRamen | 53.23% | 0.7110 | 81.72% |
| OracleDropOODRamen | 55.91% | 0.8166 | 50.54% |
| OracleIDGradientRamen | 54.84% | 0.8177 | 51.08% |
| ConsensusRamen | 54.84% | 0.7478 | 63.44% |
| OracleConsensusRamen | 54.84% | 0.7905 | 60.22% |

Ramen đúng thêm 5 mẫu ID so với NoAdapt: sửa được 11 mẫu sai nhưng làm sai
6 mẫu vốn đúng. EntropyGatedRamen sửa 13, làm sai 9 (net +4).
ConsensusRamen sửa 14, làm sai 7 (net +7), tức chỉ hơn Ramen 2 mẫu.
Đây không phải bằng chứng đủ mạnh để xếp hạng phương pháp.

FPR95 trong code là tỷ lệ ID bị đánh dấu OOD tại ngưỡng phát hiện ít nhất
95% OOD. Ramen tăng AUROC nhưng FPR95 tăng từ 74.73% lên 83.87%: chất lượng
xếp hạng tổng thể và chất lượng tại một ngưỡng vận hành có thể khác nhau.
Consensus giảm FPR95 xuống 63.44%, song mức này vẫn cao. Không chỉ dùng
accuracy hoặc H-score để tuyên bố open-set detection tốt; H-score của repo
kết hợp ID classification accuracy và OOD recall, không trực tiếp phạt FPR.

Entropy gate nhận 96/256 mẫu, gồm 77 ID và 19 OOD. Tỷ lệ OOD trong nhóm
được nhận là 19.79%, thấp hơn 27.34% trong toàn prefix; pseudo-label accuracy
trên ID được nhận là 74.03%. Gate có dấu hiệu chọn nhóm sạch hơn, nhưng OOD
vẫn lọt vào và accuracy cuối chưa hơn Ramen.

Các oracle có quyền dùng nhãn ID/OOD thật theo thiết kế kiểm soát thí nghiệm;
không phải phương pháp triển khai thông thường hoặc upper bound được chứng
minh. Khoảng cách OOD metrics là tín hiệu đáng nghiên cứu trên full matrix,
không phải lý do retune trên smoke.

## Code tái lập

Dùng [notebook Kaggle](../../../notebooks/kaggle/kaggle-pre-full-smoke.ipynb)
hoặc [Python export](../../../notebooks/kaggle/kaggle-pre-full-smoke.py).
[Hướng dẫn](../../../notebooks/kaggle/README.md) mô tả thiết lập, cách tái sinh
notebook và phạm vi 14 smoke. Notebook giữ pin source ở đúng commit được đánh giá.

## Bước tiếp theo

Giữ nguyên commit/config, lưu ZIP và báo cáo, chuẩn bị full matrix B=100 trên
checkout sạch đúng revision. Giữ các sensitivity controls tách khỏi 252 runs.
Chạy theo paired baseline, strict resume và evidence riêng cho canonical;
đánh giá across seeds/streams/OOD ratios sau khi có kết quả full. Smoke chưa
đo chi phí toàn bộ matrix hoặc bảo đảm một phiên Kaggle đủ thời gian chạy hết.
