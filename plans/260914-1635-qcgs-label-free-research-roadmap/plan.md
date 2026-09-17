---
title: "QCGS label-free research roadmap"
description: "Lộ trình kiểm chứng gradient query không dùng nhãn, xây QCGS tối thiểu, rồi đánh giá cơ chế và khả năng tổng quát."
status: pending
priority: P2
effort: "Phụ thuộc kết quả pilot và thời gian GPU; chưa ước lượng lịch hoàn thành"
branch: open-world-gradient-memory
tags: [research, experimental, qcgs, test-time-adaptation]
blockedBy: []
blocks: []
created: 2026-09-14
---

# QCGS label-free research roadmap

## Overview

Đề xuất ngày 2026-09-14. Đây là kế hoạch nghiên cứu; chưa triển khai hoặc chạy QCGS.

**Câu hỏi chính:** gradient của query không dùng nhãn có giúp chọn cached support tốt hơn feature retrieval của Ramen không?

Pilot oracle đã cho thấy có dư địa cải thiện khi thay support, nhưng dùng nhãn thật để tính gradient query. Bước tiếp theo là thay tín hiệu oracle bằng tín hiệu không nhãn và đo utility thật sau cập nhật.

Lộ trình: chốt protocol → diagnostic không nhãn → QCGS tối thiểu → kiểm tra cơ chế → full evaluation.

Đề xuất v0 chỉ thay tối đa một support trong cùng pseudo-class. Đây là phiên bản giới hạn để nối trực tiếp với pilot A/B; rerank toàn bộ top-M thành top-k như slide là phần mở rộng cần kiểm chứng riêng.

## Goals

| # | Goal | Priority |
|---|------|----------|
| 1 | Kiểm chứng khả năng chọn support bằng gradient query không nhãn | P1 |
| 2 | Tách lợi ích của query-conditioning khỏi retrieval, tự cập nhật và compute tăng thêm | P1 |
| 3 | Chỉ mở rộng benchmark sau khi có tín hiệu lặp lại và chi phí chấp nhận được | P2 |

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Phase 1: Start](./phase-01-start.md) | Pending |

Các phase tiếp theo được tạo bằng `ak plan add-phase` (đều pending):

2. [Diagnostic không nhãn](./phase-02-label-free-utility-diagnostic.md)
3. [QCGS tối thiểu](./phase-03-minimal-qcgs-method.md)
4. [Cơ chế và độ ổn định](./phase-04-mechanism-and-robustness.md)
5. [Đánh giá đầy đủ và luận văn](./phase-05-full-evaluation-and-thesis.md)

## Dependencies and existing evidence

- Đầu vào đã có: [oracle plan và kết quả](../20260907-oracle-support-utility/plan.md), [Pilot B](../20260907-oracle-support-utility/reports/pilot-b-completed.md), [runtime contract](../../docs/research/oracle-support-utility-probe.md).
- [Gap B/C](../../docs/research/gradient-compatibility-research-direction.md#gap-b--directly-test-whether-feature-similarity-predicts-gradient-compatibility) phân biệt compatibility từng support với consensus giữa các lớp.
- [Chiến dịch Consensus hiện có](../20260825-open-world-gradient-memory-evidence/plan.md) độc lập với pilot QCGS; không sửa matrix 252 runs hoặc thay mục tiêu đã khóa. Baseline cũ chỉ tái sử dụng khi source/config/stream tương thích được xác minh.
- [Hạ tầng Kaggle](../20260914-kaggle-huggingface-data/plan.md) có thể tái sử dụng sau khi kiểm tra source pin; notebook hiện tại không mặc nhiên chứa QCGS.
- Không có phụ thuộc chặn qua lại giữa các plan hiện có. Các phase QCGS chạy tuần tự và chia sẻ hạ tầng có kiểm soát.

## Success Criteria

- [ ] Khóa protocol, bộ phát triển và bộ đánh giá chưa dùng để chọn phương pháp.
- [ ] Selector quyết định trước khi evaluator đọc nhãn; dữ liệu oracle không vào cache hoặc lựa chọn.
- [ ] Diagnostic đo utility thật, có random/no-swap và mẫu kiểm chứng không thiên lệch theo oracle.
- [ ] QCGS tối thiểu vượt kiểm tra parity, reset, class balance và label isolation.
- [ ] Quyết định GO/REVISE/STOP dựa trên loss, accuracy, nhóm query và chi phí; không dựa chỉ vào entropy giảm.
- [ ] Full evaluation và claim cuối chỉ thực hiện sau khi cấu hình được khóa và gate cơ chế đạt.

## First deliverable

Một báo cáo so sánh **entropy-query score, gradient cosine, random và Ramen** trên cùng query/candidate pool. Gradient supervised chỉ làm tham chiếu evaluator. Đừng bắt đầu bằng full matrix hoặc thêm mạng học score.

<!-- slug: qcgs-label-free-research-roadmap -->
