# Pilot A — hoàn thành 16 query đủ điều kiện

Ngày: 2026-09-07. **Mục tiêu diagnostic đã hoàn thành:** 16 ID query, duyệt đủ
110 swap/query, tổng 1.760 swap. Đây là exhaustive one-swap oracle, không phải
chọn một tập con bằng first-order score.

## Kết quả

| Chỉ số | Kết quả |
|---|---:|
| Query đủ điều kiện | 16/16 |
| Swap hợp lệ đã kiểm tra exact | 1.760/1.760 |
| Headroom rate, Delta CE > 0 | 16/16 = 100% |
| CE trung bình Ramen | 1,758541 |
| CE trung bình oracle | 1,544206 |
| Mean loss gain | 0,214335 |
| Median loss gain | 0,173319 |
| p25 / p75 loss gain | 0,000418 / 0,297908 |
| Mean margin gain | +0,261719 |
| Accuracy Ramen → oracle | 8/16 → 8/16 |
| Sai → đúng / đúng → sai | 0 / 0 |
| Net correction | 0 |
| Spearman(first-order, exact), 1.760 swap | 0,441306 |

Mọi query đều có ít nhất một swap làm giảm CE, nhưng độ lớn không đồng đều:
loss gain từ 0,000043 đến 1,031368. Không có query đổi nhãn dự đoán đúng/sai.
Headroom rate dùng đúng điều kiện strict Delta CE > 0 của plan; không có ngưỡng
hiệu ứng tối thiểu được thêm sau khi xem kết quả.

## So sánh lựa chọn support

| Lựa chọn | Mean CE gain so với Ramen |
|---|---:|
| Ramen / feature similarity, không swap | 0 |
| Random same-class swap | +0,014635 |
| Gradient-cosine preference | +0,076242 |
| First-order best predicted swap | +0,160166 |
| First-order worst predicted swap | −0,099877 |
| Exact best one-swap oracle, có no-swap fallback | +0,214335 |

Random cải thiện CE ở 8/16 query; first-order best cải thiện ở 13/16.
First-order chọn đúng mức gain cực đại ở 2/16 query, nhưng thu được khoảng
74,7% mean oracle headroom. Đây là so sánh diagnostic có sử dụng nhãn, không
phải kết quả của selector có thể triển khai.

Spearman trên toàn bộ swap: feature-distance preference 0,0467; raw cosine gain
0,0797; first-order SignSGD-direction score 0,4413. Với first-order, Spearman
trung bình theo query là 0,3372 và median là 0,3815. Tín hiệu first-order có ích
nhưng chưa đủ để thay việc kiểm tra exact trong mọi trường hợp.

## OOD và domain

Best swaps gồm ID→ID: 7; ID→OOD: 3; OOD→ID: 3; OOD→OOD: 3.
Tỷ lệ loại OOD để lấy ID và tỷ lệ lấy OOD thay ID đều là 18,75%.
Same-domain→cross-domain: 8/16; cross-domain→same-domain: 2/16.
Chưa có bằng chứng từ cell này rằng lợi ích chủ yếu đến từ loại bỏ OOD.

Cả 16 query thuộc corruption `defocus_blur`, với timestep:
100, 102, 103, 104, 105, 106, 108, 109, 110, 111, 117, 119, 122, 124, 125, 126.
Đây là 16 ID query đủ điều kiện đầu tiên theo stream đã khóa, không chọn lại
query dựa trên kết quả. Sau 100 mẫu, cache lớn nhất chỉ có 8 support, chưa đủ
m=10. Batch thứ hai tạo điều kiện đủ; cache bao gồm toàn bộ batch như Ramen.

## Cách chạy và kiểm chứng

Official CIFAR-100-C; CLIP ViT-B/16; MPS/PyTorch2.4.1; split v1; OOD ratio=.5;
block64; seed0; batch100; k5/m10; capacity750; beta5; SignSGD lr=.01.
Trong 200 mẫu đã ghi trace có 88 OOD (44.0%); .5 là tỷ lệ cấu hình của source pool.
Model và dữ liệu qua fast provenance của repo. Config và source hashes được
lưu trong launch plan và đã kiểm tra khớp sau chạy.

Mỗi query có 110 hướng cập nhật khác nhau cần kiểm tra. ViT dùng tham số
LayerNorm riêng từng mẫu, nên mỗi forward kiểm tra một swap/query trên nguyên
batch100, giữ nguyên vị trí các mẫu. Có 110 vòng, mỗi vòng 16 query. Output
baseline được kiểm tra bằng tensor equality trên CLIP thật; hai query đầu của
vòng gộp cũng được đối chiếu với chạy riêng trên nguyên batch. Các check đều qua.
Xem [review và kiểm thử](grouped-verification-review.md).

[Audit tái tính](pilot-a-audit.json) xác nhận:

- Đúng 16 query ID đủ điều kiện đầu tiên.
- Đủ toàn bộ 1.760 tuple class/outgoing/incoming theo cache tại đúng batch.
- Metadata OOD/domain/pseudo-class của từng support khớp trace.
- Support chỉ đến từ các timestep nhìn thấy theo batch-atomic Ramen.
- Loss gain, lựa chọn cực đại và toàn bộ summary khớp các JSON row thô.
- Source hashes không đổi so với lúc launch.

Mục tiêu 16 query đã hoàn tất và được lưu sau batch thứ hai, cùng 200 trace rows.
Tiến trình evaluator tổng quát có upper budget600 mẫu đã được chủ động dừng sau
khi mục tiêu này hoàn tất, trong batch bổ sung tiếp theo. Vì vậy **không có full
600-sample benchmark summary**; không được gọi run này là benchmark600 hoàn tất.
Việc dừng không ảnh hưởng 16 query đã kiểm tra hoặc 200 trace rows dùng để audit.
Tất cả tiến trình thí nghiệm đã dừng. Pilot B chưa chạy trong yêu cầu này.

Dữ liệu gốc:

- [Query JSONL](../../../evidence/oracle-support-pilot-a-complete-20260907/oracle-support-a-mps-seed0/oracle-support-queries.jsonl)
- [Summary](../../../evidence/oracle-support-pilot-a-complete-20260907/oracle-support-a-mps-seed0/oracle-support-summary.json)
- [Trace](../../../evidence/oracle-support-pilot-a-complete-20260907/oracle-support-a-mps-seed0/trace.jsonl)
- [Launch plan](../../../evidence/oracle-support-pilot-a-complete-20260907/launch-plan.json)
- [Runtime log](../../../evidence/oracle-support-pilot-a-complete-20260907/runtime.log)
- [Script audit](pilot-a-audit.py)

## Ý nghĩa

Pilot A chứng minh **có oracle support-selection headroom trong local candidate
pool của Ramen trên 16 query này**. Mean/median loss gain dương, lớn hơn random
control, nhưng accuracy chưa tăng. Kết quả đủ để tiếp tục đo Pilot B; chưa đủ để
khẳng định hiệu quả rộng hơn, vai trò nhân quả của OOD contamination, hoặc đã có
một phương pháp gradient compatibility không cần nhãn. Cell OOD=0 và các query/
corruption khác vẫn cần được kiểm tra.
