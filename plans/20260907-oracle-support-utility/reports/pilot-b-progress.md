Superseded by [completed Pilot B report](pilot-b-completed.md). Historical intermediate progress follows.

# Pilot B — tiến độ ngày 2026-09-07

OOD=0 hoàn thành 128 query, audit PASS. OOD=0.5 đang chạy, chưa có kết luận so sánh cuối.

## Cấu hình và nguồn

CIFAR100C, CLIP ViT-B/16, split 80 known classes v1, seed 0, batch 100, k=5, m=10, cache 750, beta=5, SignSGD lr=.01. Mỗi query xếp hạng tất cả swap hợp lệ và kiểm chứng best/worst/random/cosine (tối đa bốn swap khác nhau), kèm lựa chọn không đổi support. Kết quả screened là cận dưới của exhaustive one-swap oracle; phép chọn vẫn dùng nhãn query chỉ trong evaluator.

## OOD=0

128 query đầu tiên, timestep 0–127: 64 fog, 64 defocus_blur. Đã xếp hạng 4.600 swap, kiểm chứng 493 swap. CE trung bình 2.041963→1.995286; mean gain 0.046677, median 0.008720. Headroom dương 114/128; đúng 70→72 (hai sai→đúng, không có đúng→sai). First-order mean gain 0.034269; cosine 0.015678; random −0.009044; worst first-order −0.034424. Spearman pooled 0.400647, có thiên lệch do chọn trước swap để kiểm chứng.

Ý nghĩa: trên tập query này, headroom vẫn có khi không có OOD. Loss giảm phổ biến nhưng cải thiện accuracy nhỏ. Không suy ra OOD là nguyên nhân trước khi kiểm tra cấu hình còn lại; hai tập query/cache history không ghép cặp.

## Runtime và trạng thái

Lần đầu bị dừng trước query hoàn chỉnh để tối ưu phép cộng gradient. Lần vectorized kế tiếp mất tiến trình khi máy reset; nhật ký hệ thống ghi watchdog/reset, chưa xác định nguyên nhân. Lần khôi phục hiện tại lưu tại `evidence/oracle-support-pilot-b-resumed-20260907`. OOD=0 lưu đủ 200 trace rows và dừng có chủ đích sau mục tiêu diagnostic, không phải benchmark đủ 600 mẫu. Nguồn và config được hash/snapshot khi khởi chạy. Tests: 367 chạy, 362 đạt, 5 bỏ qua.
