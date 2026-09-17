---
phase: 5
title: "Full evaluation and thesis"
status: pending
priority: P2
dependencies: [4]
---

# Phase 5: Full evaluation and thesis

## Overview

Khóa phương pháp sau gate cơ chế, chạy benchmark đầy đủ trên GPU và viết đóng góp theo đúng evidence. Tham chiếu [phase 4](./phase-04-mechanism-and-robustness.md) và [evidence program hiện có](../20260825-open-world-gradient-memory-evidence/plan.md).

## Requirements

- Campaign QCGS có manifest/version/source riêng; không sửa danh tính matrix Consensus 252 runs đã khóa.
- Dùng full streams, official data/model provenance và baseline cùng protocol. Không dùng query-limited pilot như benchmark hoàn chỉnh.
- Không tune score, M/k, threshold, augmentation hoặc budget theo labels của final test.
- Đánh giá ID accuracy, pre/post OOD metrics, per-stream/domain outcomes và latency/memory; không chỉ loss.

## Files

- Create: `scripts/run-query-gradient-matrix.py`, `cfg/research/query-gradient-evaluation/`, `tests/test_query_gradient_matrix.py`, `reports/final-evaluation.md` trong plan này.
- Modify sau khi có dữ liệu: `docs/research/query-conditioned-gradient-selection.md`, `docs/research/gradient-compatibility-research-direction.md`, `docs/research/open-world-gradient-memory-thesis-report.md`.
- Read/reuse khi tương thích: `src/runtime/experiment_matrix.py`, `src/runtime/open_set_split_robustness_matrix.py`, `src/runtime/open_set_domainnet_matrix.py`, `notebooks/kaggle/full-run-guide.md`, `notebooks/kaggle/huggingface-run-guide.md`.

## Implementation Steps

1. Khóa method/config và danh sách baselines. Lập manifest QCGS riêng, ước lượng GPU-hours từ profiling, lưu toàn bộ planned run IDs trước khi chạy.
2. Đề xuất CIFAR100C open-set full streams với OOD={0,0.1,0.3,0.5}, streams={iid_mixed,block,recurring}, seeds={0,1,2}; dùng phần giữ lại chưa tune để đánh giá xác nhận. Nếu seed0 đã dùng phát triển, công khai điều đó và dành additional held-out setting/seed cho xác nhận.
3. Kiểm tra split robustness và DomainNet sau primary benchmark; thiết kế known/unknown split, đủ class coverage và data provenance trước thực thi. Không coi notebook đã chuẩn bị là data/GPU đã sẵn sàng.
4. Chỉ tái dùng baseline artifacts khi source/config/model/data/stream/semantics và schema khớp; không ghép tùy ý kết quả campaign cũ.
5. Báo cáo mean và variation across seeds, paired effects, worst-domain/stream degradation, detection change và compute. Ghi failed/incomplete cells và không tổng hợp như đã hoàn thành.
6. Viết câu chuyện: oracle cho thấy có headroom → score không nhãn dự đoán utility → v0 chọn support thật → ablation chứng minh phần giá trị → robustness và hạn chế.
7. Nếu method thất bại, viết negative result có kiểm soát, giữ oracle finding riêng. Không tự đổi tên hoặc scope để biến loss gain thành accuracy gain.

## Todo

- [ ] Planner/schema tests, current-source GPU smoke và full-run artifact validation đạt.
- [ ] Báo cáo nguồn, bảng kết quả, chi phí, related-work distinction và giới hạn khớp artifacts.

## Success Criteria

Mọi cell bắt buộc trong campaign mới có evidence hợp lệ và phân tích đúng phạm vi. Kết luận effectiveness/novelty chỉ ở mức dữ liệu và related-work review hỗ trợ; có thể kết thúc với kết quả âm được xác minh.

## Risks and rollback

GPU/data chưa sẵn sàng và small pilot gain có thể không tổng quát. Không giảm scope âm thầm hoặc dùng MPS pilot thay kết quả CUDA. Khi cần thay protocol, version campaign mới, giữ campaign cũ và lý do thay đổi.
