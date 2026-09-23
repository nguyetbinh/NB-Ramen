# QCGS multi-view rescue — kết quả diagnostic

Audit ngày **2026-09-23**, nhánh `qcgs-label-free-diagnostic`.

**STOP tại Stage A theo protocol đã khóa.** Evidence hợp lệ qua các kiểm tra offline và đủ 16 query ID/cell. Primary có CE gain trung bình dương ở cả hai mức OOD, nhưng mean per-query Spearman giữa score và true utility âm ở OOD=0.5. Stage B **không chạy**, đúng điều kiện dừng; đây không phải thiếu evidence do lỗi vận hành. Kết luận chỉ áp dụng cho biến thể MV-target-θ_R và thiết lập này, không chứng minh mọi QCGS label-free đều bất khả thi.

## Evidence và kiểm chứng

- ZIP nhận lại: `qcgs-multiview-evidence.zip`, 17.794.842 bytes; SHA-256 `033877f5642d8a8c4ae09704561b0d87ef126c6ef540da12011282825f86983d`.
- [Raw queries](../../../evidence/qcgs-multiview-kaggle-033877f5642d/qcgs-multiview-evidence/queries.jsonl), [analysis đầy đủ](../../../evidence/qcgs-multiview-kaggle-033877f5642d/qcgs-multiview-evidence/analysis.json), [preflight/provenance](../../../evidence/qcgs-multiview-kaggle-033877f5642d/qcgs-multiview-evidence/locks/preflight.json), [registry](../../../evidence/qcgs-multiview-kaggle-033877f5642d/qcgs-multiview-evidence/locks/registry.json), [gate đã commit](../../../evidence/qcgs-multiview-kaggle-033877f5642d/qcgs-multiview-evidence/locks/stage-a-gate.json). Raw evidence lưu local trong `evidence/`, không commit; khôi phục từ ZIP có SHA trên để dùng các liên kết này ở máy khác.
- [Receipt audit đã commit](scientific-validation.json) lưu hashes, 8 completed-run receipts, provenance và metrics tính độc lập. Source thực nghiệm `b96488cd410f6d28e6de52a308338ddb4879e779`; tree `8a182373f67a4c9fa9b92163c07c639440b41b5e`. Protocol đúng phiên bản trong commit này; không thay gate sau kết quả.
- Đã kiểm CRC/path ZIP, Git ledger sạch và `fsck`, config/source/model/data/view/parameter-order locks, exclusions/RNG, registry replay, legal action coverage, selected-action arithmetic và mọi completed artifact hash. Rebuild registry từ scan không chấm utility; tiền tố 600 chưa đủ quota, 1.200 đủ. Hai cell có cap đã khóa 800/1.100.
- Chạy auditor tin cậy vào thư mục tạm, tính lại analysis/gate từ raw; kết quả **khớp chính xác** aggregate trong ZIP. Tính riêng U/D/E, drop-two, exact oracle và tied-rank Spearman từ CE/score với Python `statistics`/`math`, khớp trong `1e-12`. Canonical analysis SHA-256: `c7a017f38e09513f48ecb85628de292c91b29bbe255dbf8dda95c9557ada8dde`. Toàn bộ 209 file evidence giữ nguyên bytes.

Kaggle ghi nhận Python 3.11.16, PyTorch 2.4.1+cu121, CUDA 12.1, Tesla T4; máy liệt kê 2 GPU, không suy ra chạy song song trên cả hai. CPU preflight trong ZIP: **33 passed, 1 skipped** focused; **416 passed, 6 skipped** full (MPS không có trên Linux và CUDA bị ẩn trong CPU tests). Sau đó **CUDA smoke 2 query/cell**, 4 scan và 2 Stage A run hoàn thành. [Receipt CPU/đóng gói trước đây](validation.json) là lịch sử local, không thay cho evidence Kaggle này; notebook vẫn dùng source pin trên và nhỏ hơn 1 MB.

## Kết quả Stage A và quyết định theo thứ tự gate

U = CE_Ramen − CE_primary; D = CE_random-matched − CE_primary; E = CE_Ramen-MV − CE_primary. Giá trị dương là có lợi; đơn vị CE, không phải điểm phần trăm accuracy.

| Chỉ số | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Query / primary swaps / rho xác định | 16 / 16 / 16 | 16 / 16 / 16 |
| Mean exact-oracle U | +0.326281 | +0.407242 |
| Mean primary U | +0.171246 | +0.123769 |
| Mean D | +0.172130 | +0.120197 |
| Mean E | +0.255355 | +0.021415 |
| Mean per-query Spearman primary | +0.014331 | **−0.125254** |
| Accuracy Ramen → primary | 8/16 → 9/16 | 8/16 → 10/16 |
| Wrong→correct / correct→wrong | 1 / 0 | 2 / 0 |

1. **INVALID:** không phát hiện vi phạm qua audit raw, receipts và các guards/tests của source đã khóa. Giới hạn xác minh offline được nêu bên dưới.
2. **INCONCLUSIVE:** không áp dụng; đủ quota Stage A và exhaustive verification: **16.500 / 17.965 legal swaps**, cộng no-swap. Mỗi query có positive exact-oracle headroom trong không gian này.
3. **GO_CONFIRM:** OOD=0 qua mọi điều kiện. OOD=0.5 qua oracle/U/D/E/quota/replacements nhưng **không qua mean rho>0**. Gate yêu cầu cả hai cell cùng qua.
4. **STOP:** dừng diagnostic rescue; không mở Stage B, full matrix, reranking hay adaptive support. Không đổi control thành primary hoặc chỉnh score/threshold từ kết quả này. Không có quyết định REVISE trong protocol rescue.

Không có số liệu Stage B để báo best-verified subset hay áp gate GO_PILOT. Exact oracle ở bảng trên chỉ tối ưu same-pseudo-class one-swap của Stage A, không bao gồm ensemble prediction controls.

## Diễn giải và giới hạn

Kết quả này **không phải mọi chỉ số đều âm**: CE/accuracy trung bình có cải thiện. Tuy nhiên, khả năng xếp hạng swap theo utility chưa ổn định và lợi ích nhạy với vài ảnh:

| Kiểm tra độ nhạy: mean sau bỏ hai giá trị lớn nhất của từng metric | OOD=0 | OOD=0.5 |
|---|---:|---:|
| U | −0.015157 | −0.184415 |
| D | −0.016155 | −0.185216 |
| E | +0.131841 | −0.153928 |

Hai ảnh đóng góp U lớn nhất ở **cả hai cell** là `sample_idx=6345` và `8165`, nằm trong tập ảnh trùng giữa OOD. Drop-two là phân tích độ nhạy đã định trước, **không phải lý do áp gate STOP ở Stage A**. Các khoảng bootstrap block 95% của U/D/E đều chứa 0; đây là mô tả có điều kiện trên một stream, không phải kiểm định với các lần chạy độc lập.

Supervised reference tại cùng anchor đạt mean rho **0.9280 / 0.9098**, trong khi primary đạt **0.0143 / −0.1253**. Teacher Frozen-MV không sửa đúng prediction nào trong 8 query Ramen sai ở mỗi cell, dù tăng xác suất nhãn thật ở 4/8. Phân nhóm evaluator cho thấy mean primary U trên query Ramen sai là **+0.5113 / +0.6173**, nhưng trên query Ramen đúng là **−0.1688 / −0.3697**. Những quan sát này phù hợp với việc soft target chưa cung cấp hướng sửa đáng tin cho mọi query; chúng chưa xác lập nguyên nhân duy nhất. Correctness dùng nhãn thật nên không thể dùng trực tiếp thành selector label-free.

32 quan sát Stage A tương ứng **27 ảnh gốc khác nhau**, với 5 ảnh trùng giữa hai OOD cell (`1219, 1719, 4407, 6345, 8165`); không trùng trong mỗi cell, giữa A/B đã đăng ký, hoặc với exclusions. Mỗi cell chỉ có seed 0, một stream và 7 scored blocks. Không suy generalization từ mẫu nhỏ hoặc coi các candidate swaps/views là replicate độc lập.

Audit offline xác minh arithmetic, selection, registry và provenance đã lưu; không chạy lại CUDA hay khôi phục logits/gradient từ model. Dataset/checkpoint không nằm trong ZIP, nên checksum của chúng được đối chiếu từ receipts của source đã pin, không phải rehash lại bytes local. STOP là quyết định dừng đầu tư theo gate bảo thủ đã đăng ký, không phải chứng minh hiệu ứng thật bằng 0. **Giữ STOP cũ của single-view; đóng cả lần cứu multi-view này trong thiết lập hiện tại.**
