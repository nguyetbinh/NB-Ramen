# Pilot B hoàn thành — 2026-09-07

Cả hai cấu hình OOD=0 và OOD=0.5 đã hoàn thành **128 query ID đủ điều kiện mỗi cấu hình**, tổng 256 query. Audit độc lập PASS cho cả hai. Kết quả ủng hộ tiếp tục nghiên cứu cách dự đoán utility của support không dùng nhãn; chưa chứng minh một selector triển khai được hoặc tác động nhân quả của OOD.

## Kết quả chính

Gain = CE của Ramen − CE sau swap; dương là tốt. “Oracle” dưới đây chọn phương án tốt nhất trong các swap đã kiểm chứng và no-swap, là **cận dưới screened**, không phải exhaustive oracle.

| Chỉ số | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Query đủ điều kiện | 128 | 128 |
| Query có headroom dương | 114/128 (89.06%) | 120/128 (93.75%) |
| CE Ramen trung bình | 2.041963 | 2.300949 |
| CE screened oracle trung bình | 1.995286 | 1.969162 |
| Mean loss gain | 0.046677 | 0.331788 |
| Median loss gain | 0.008720 | 0.053992 |
| P25 / P75 gain | 0.000061 / 0.049484 | 0.000766 / 0.161366 |
| Số dự đoán đúng, Ramen → oracle | 70 → 72 | 62 → 64 |
| Accuracy, Ramen → oracle | 54.69% → 56.25% | 48.44% → 50.00% |
| Sai → đúng / đúng → sai | 2 / 0 | 2 / 0 |
| Mean margin gain | 0.084961 | 0.415039 |
| Swap hợp lệ đã xếp hạng | 4,600 | 30,845 |
| Swap đã kiểm chứng loss | 493 | 509 |

## Các cách chọn support

| Mean loss gain | OOD=0 | OOD=0.5 |
|---|---:|---:|
| Ramen feature top-k / no-swap | 0 | 0 |
| Best first-order | 0.034269 | 0.320014 |
| Gradient cosine | 0.015678 | 0.192436 |
| Random swap | −0.009044 | −0.002948 |
| Worst first-order | −0.034424 | −0.116632 |

Best first-order cải thiện loss ở 86/128 và 109/128 query; random ở 58/128 và 65/128. Best first-order có accuracy 71/128 và 64/128; cosine 72/128 và 62/128. Vì vậy chất lượng xếp hạng theo CE không đồng nhất với xếp hạng theo accuracy.

Spearman giữa điểm dự đoán và utility thực đo trên các swap đã kiểm chứng là 0.400647 và 0.675261. **Các hệ số này chịu thiên lệch screening** (best/worst/random/cosine được chọn trước), không phải correlation của toàn bộ swap. Feature top-k ở đây là baseline không swap, không phải một phép đo đầy đủ khả năng dự đoán utility của feature similarity.

## Ý nghĩa và quyết định nghiên cứu

1. **Có headroom ngay cả khi không có OOD.** Trên tập query OOD=0, thay danh tính support trong cùng pseudo-class vẫn cải thiện CE cho 114/128 query. Vấn đề chọn support không chỉ xuất hiện do contamination.
2. **Có tín hiệu đáng nghiên cứu từ first-order score.** Mean gain tốt hơn random và gradient cosine ở cả hai cấu hình; worst score cho mean gain âm. Tuy nhiên score dùng gradient từ nhãn query thật, chỉ là công cụ chẩn đoán.
3. **Loss cải thiện nhiều hơn accuracy.** Mỗi cấu hình chỉ sửa thêm hai câu đúng, tương đương +1.56 điểm phần trăm accuracy. Không nên mô tả đây là bước nhảy lớn về độ chính xác.
4. **OOD=0.5 có headroom đo được lớn hơn, nhưng chưa chứng minh nguyên nhân do OOD.** Hai tập query, domain, thời điểm đủ điều kiện, lịch sử cache và số swap hợp lệ khác nhau. Không có so sánh ghép cặp giữ nguyên mọi yếu tố ngoài OOD.
5. **Mean bị chi phối bởi một số query.** Năm query có gain lớn nhất đóng góp 36.77% tổng gain ở OOD=0 và 60.98% ở OOD=0.5; mười query đóng góp 54.39% và 76.96%. Gain lớn nhất lần lượt 0.9912 và 7.3706. Median dương vẫn cho thấy hiệu ứng không chỉ nằm ở các trường hợp cực trị.
6. **Không có quy luật đơn giản “bỏ OOD là tốt”.** Trong 120 swap được oracle chọn ở OOD=0.5: OOD→ID 23, ID→OOD 41, ID→ID 47, OOD→OOD 9. Đây là số đếm mô tả, chưa chuẩn hóa theo số cơ hội swap; nó không chứng minh OOD nói chung hữu ích hoặc có hại.

Theo tiêu chí định tính của plan, **GO cho bước nghiên cứu khả năng dự đoán utility không dùng nhãn**: headroom phổ biến, mean/median dương và net corrections dương. Không tự đặt ngưỡng ý nghĩa thống kê; đây là pilot một seed, 128 query/cấu hình. Pattern headroom tăng khi có OOD được quan sát mô tả, nhưng phần diễn giải nhân quả về contamination chưa được kiểm chứng. Không tự động mở rộng sang triển khai selector trong lần chạy này.

Claim phù hợp: *Trong candidate pool cục bộ, cân bằng theo pseudo-class của Ramen, danh tính các cached support gradient ảnh hưởng đến adaptation của query và để lại headroom lựa chọn support đo được.* Chưa claim đã giải quyết gradient compatibility.

## Đối chiếu Pilot A

16 query trùng với A khớp **chính xác** baseline và toàn bộ bản ghi swap B đã kiểm chứng. Trên chính 16 query đó, tổng gain screened/exhaustive = **75.2003%**. B không tìm được toàn bộ headroom exhaustive; không dùng tỷ lệ này để suy rộng sang 112 query còn lại hoặc OOD=0.

## Phạm vi dữ liệu và thực thi

CIFAR100C, CLIP ViT-B/16, split v1 gồm 80 known classes, seed=0, batch=100, k=5, m=10, cache=750, beta=5, SignSGD lr=.01, block=64, source budget=400/domain. OOD=0 vẫn dùng classifier/split 80 known classes. Nhãn chỉ vào evaluator một lần; OOD label −1 không được chấm supervised loss và known label không vào cache. Cache nhận toàn bộ batch trước retrieval. Mỗi swap cùng pseudo-class, giữ nguyên số support, tính lại trọng số thực.

- OOD=0: timestep query 0–127; fog 64, defocus_blur 64; trace 200 dòng, OOD thực tế 0%.
- OOD=0.5: timestep query 100–346; defocus_blur 17, motion_blur 36, zoom_blur 32, shot_noise 32, brightness 11; trace 400 dòng, OOD thực tế 48.5%.
- Mỗi query rank mọi swap hợp lệ, kiểm chứng best/worst/random/cosine (tối đa bốn lựa chọn khác nhau), kèm no-swap fallback.
- Các trace thông thường giữ nguyên output Ramen. Kết quả oracle nằm trong sidecar diagnostic, không phải phương pháp mới trong canonical matrix.
- Bộ đánh giá tổng quát bị dừng có chủ đích sau khi đủ query và ghi đủ các batch trace bao phủ chúng. Cả hai **không phải benchmark đủ 600 mẫu**. `KeyboardInterrupt` trong log tương ứng dừng sau mục tiêu; controller ghi `TARGET_COMPLETE`.
- Lần đầu dừng để vectorize phép cộng gradient trước khi có query hoàn chỉnh. Lần vectorized tiếp theo mất tiến trình khi máy reset; chưa xác định nguyên nhân. Các lần này được lưu riêng, không trộn với kết quả hoàn thành.

## Kiểm chứng và bằng chứng

367 tests đã chạy: 362 đạt, 5 bỏ qua. Review read-only không thấy lỗi semantic trong vectorization. Mỗi batch kiểm tra aggregate baseline khớp Ramen, toàn bộ candidate của query đầu khớp arithmetic serial, temporary baseline khớp logits gốc, và hai trial gộp khớp trial riêng trên full batch. Audit kiểm tra thứ tự query đủ điều kiện, support provenance, tập swap hợp lệ, no-swap fallback, CE gain và tái tính summary từ raw rows. Source/config/provenance được khóa và lưu khi chạy. Pilot A lưu lại source khớp hash gốc để đối chiếu.

- [Kết quả và audit đầy đủ](../../../evidence/oracle-support-pilot-b-resumed-20260907/audit-and-analysis.json)
- [Phân tích bổ sung](../../../evidence/oracle-support-pilot-b-resumed-20260907/additional-analysis.json)
- [Launch plan và hashes](../../../evidence/oracle-support-pilot-b-resumed-20260907/launch-plan.json)
- [Raw query OOD=0](../../../evidence/oracle-support-pilot-b-resumed-20260907/oracle-support-b-null-mps-seed0/oracle-support-queries.jsonl)
- [Raw query OOD=0.5](../../../evidence/oracle-support-pilot-b-resumed-20260907/oracle-support-b-open-mps-seed0/oracle-support-queries.jsonl)
- [Audit script](pilot-b-audit.py), [controller](run-pilot-b.py), [review vectorization](vectorized-ranking-review.md)
- [Pilot A](pilot-a-completed.md)
